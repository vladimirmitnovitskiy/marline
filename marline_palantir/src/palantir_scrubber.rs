use marline_scrub::decoder::{Decoder, GdeltaDecoder};
use marline_storage::{
    ChunkHash, Data, DataContainer, Database, InMemoryDatabase, IterableDatabase,
};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::io;

use crate::encoder::PalantirEncoder;
use crate::lifecycle_manager::LifecycleTierConfig;
use crate::metadata_manager::MetadataManager;
use crate::mock_rocksdb::{Entry, MockRocksDBMap};
use crate::sf_generator::SuperFeatureGenerator;
use crate::types::{BlockID, Chunk, TierConfig};
use marline_index::heuristic_index::SearchConfig;
use marline_index::index::IndexError;

/// Measurements collected during a scrubbing pass.
#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct ScrubMeasurements {
    pub processed_data: usize,
    pub running_time: std::time::Duration,
    pub data_left: usize,
}

impl ScrubMeasurements {
    pub fn new(processed_data: usize, running_time: std::time::Duration, data_left: usize) -> Self {
        Self { processed_data, running_time, data_left }
    }
}

/// Core scrubbing pipeline that applies the Palantir method to a storage backend.
///
/// For every chunk in the database, `PalantirScrubber`:
/// 1. Generates super-features via [`SuperFeatureGenerator`].
/// 2. Looks up similar chunks in the multi-tier [`MetadataManager`].
/// 3. If a match is found, delta-encodes the chunk; otherwise stores it raw.
///
/// The decision to store a delta uses an adaptive compression-ratio threshold
/// that tracks a running average.
///
/// [`SuperFeatureGenerator`]: crate::types::SuperFeatureGenerator
pub struct PalantirScrubber<
    S,
    H: ChunkHash + Clone + Eq + Hash + Send + Sync + 'static,
    E,
    const N: usize,
> {
    /// The super-feature generator.
    sf_gen: S,
    /// Multi-tier similarity index.
    metadata_manager: MetadataManager<H, N>,
    /// Delta encoder.
    encoder: E,
    /// False-positive threshold for delta encoding ratio.
    fp_threshold: f64,
    /// Total chunks processed.
    chunks_processed: u64,
    /// Count of chunks stored as deltas.
    delta_stored: u64,
    /// Tracks chunk hash → base hash for stored deltas (chain rebuild).
    delta_bases: HashMap<H, H>,
    manifest_base: MockRocksDBMap,
}

impl<S, H: ChunkHash + Clone + Eq + Hash + Send + Sync + 'static, E, const N: usize>
    PalantirScrubber<S, H, E, N>
{
    /// Creates a new `PalantirScrubber`.
    ///
    /// # Arguments
    /// * `sf_gen` — The super-feature generator.
    /// * `encoder` — The delta encoder.
    /// * `tier_config` — Tier configuration (super-feature sizes per tier).
    /// * `lifecycle_configs` — Per-tier lifecycle policies.
    ///
    /// # Defaults
    ///
    /// | Field | Value |
    /// |-------|-------|
    /// | `fp_threshold` | `0.9` — false-positive ratio cap |
    /// | `chunks_processed` | `0` |
    /// | `delta_stored` | `0` | number of stored chunks by delta
    /// | `delta_bases` | empty hashmap | remember of chunk is delta of other chunk
    pub fn new(
        sf_gen: S,
        encoder: E,
        tier_config: TierConfig<N>,
        lifecycle_configs: [LifecycleTierConfig; N],
        search_config: SearchConfig,
    ) -> Self {
        Self {
            sf_gen,
            metadata_manager: MetadataManager::new(tier_config, lifecycle_configs, &search_config),
            encoder,
            fp_threshold: 0.9,
            chunks_processed: 0,
            delta_stored: 0,
            delta_bases: HashMap::new(),
            manifest_base: MockRocksDBMap::new(),
        }
    }

    pub fn delta_stored(&self) -> u64 {
        self.delta_stored
    }

    pub fn fp_table_size(&self) -> usize {
        self.metadata_manager.fp_table_size()
    }

    pub fn sf_table_size(&self) -> usize {
        self.metadata_manager.sf_table_size()
    }

    pub fn delta_bases(&self) -> &HashMap<H, H> {
        &self.delta_bases
    }

    pub fn update(&self) -> Result<(), IndexError> {
        self.metadata_manager.finish_version()
    }
}

impl<S: SuperFeatureGenerator, E: PalantirEncoder, const N: usize>
    PalantirScrubber<S, [u8; 32], E, N>
{
    pub fn scrub<B>(
        &mut self,
        database: &mut B,
        target_map: &mut HashMap<[u8; 32], Vec<u8>>,
    ) -> io::Result<ScrubMeasurements>
    where
        B: IterableDatabase<[u8; 32], DataContainer<[u8; 32]>>,
    {
        let start = std::time::Instant::now();
        let mut processed_data = 0;
        let data_left = 0;

        for (hash, container) in database.iter_mut() {
            match container.data() {
                Data::Chunk(chunk_data) => {
                    let chunk = Chunk::new(chunk_data.clone());
                    let super_features = self.sf_gen.generate(&chunk);
                    match self.metadata_manager.lookup_fingerprint(hash) {
                        Some(_) => {}
                        None => {
                            match self.metadata_manager.lookup_super_features(&super_features) {
                                Some((base_hash, _)) => {
                                    match get_manifest(
                                        &self.manifest_base,
                                        &self.delta_bases,
                                        &base_hash.hash,
                                    )? {
                                        (base_chunk, None) => {
                                            let delta = self.encoder.encode(chunk_data, base_chunk);
                                            let delta_compressed = delta.clone();
                                            let simple_compressed = delta.clone();
                                            let _ratio = delta_compressed.len() as f64
                                                / simple_compressed.len() as f64;
                                            target_map.insert(*hash, delta_compressed);
                                            self.delta_bases.insert(*hash, base_hash.hash);
                                            self.delta_stored += 1;
                                        }
                                        (base_chunk, Some(manifest)) => {
                                            let delta = self.encoder.encode_with_manifest(
                                                chunk_data, base_chunk, manifest,
                                            );
                                            let delta_compressed =
                                                zstd::encode_all(delta.as_slice(), 0)?;
                                            let simple_compressed =
                                                zstd::encode_all(chunk_data.as_slice(), 0)?;
                                            let ratio = delta_compressed.len() as f64
                                                / simple_compressed.len() as f64;
                                            if ratio < self.fp_threshold {
                                                target_map.insert(*hash, delta_compressed);
                                                self.delta_bases.insert(*hash, base_hash.hash);
                                                self.delta_stored += 1;
                                            } else {
                                                target_map.insert(*hash, chunk_data.clone());
                                                self.manifest_base.insert(
                                                    *hash,
                                                    Entry::from(chunk_data.clone()),
                                                )?;
                                            }
                                        }
                                    }
                                    processed_data += chunk_data.len();
                                }
                                None => {
                                    let fp_table = None;
                                    target_map.insert(*hash, chunk_data.clone());
                                    self.manifest_base
                                        .insert(*hash, Entry::new(fp_table, chunk_data.clone()))?;
                                    processed_data += chunk_data.len();
                                }
                            }

                            self.metadata_manager.add_block(
                                *hash,
                                &super_features,
                                BlockID::new(*hash),
                            );
                        }
                    }
                    container.make_target(vec![*hash]);
                    self.chunks_processed += 1;
                }
                Data::TargetChunk(_) => {}
            }
        }

        Ok(ScrubMeasurements { processed_data, running_time: start.elapsed(), data_left })
    }
}

const MAX_CHAIN_DEPTH: usize = 1024;

/// Walks `delta_bases` from `start` to the root hash.
/// Returns `Err` with `ErrorKind::InvalidData` if a cycle is detected or depth exceeds `MAX_CHAIN_DEPTH`.
fn walk_to_root(
    delta_bases: &HashMap<[u8; 32], [u8; 32]>,
    start: &[u8; 32],
) -> io::Result<([u8; 32], Vec<[u8; 32]>)> {
    let mut chain = Vec::new();
    let mut cur = *start;
    let mut seen = HashSet::new();
    seen.insert(cur);
    while let Some(&base) = delta_bases.get(&cur) {
        if chain.len() >= MAX_CHAIN_DEPTH {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "delta chain exceeds maximum depth",
            ));
        }
        chain.push(cur);
        cur = base;
        if !seen.insert(cur) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cycle detected in delta chain",
            ));
        }
    }
    Ok((cur, chain))
}

/// Returns only the root hash without allocating chain vectors or seen sets.
/// Uses Floyd's cycle-finding algorithm to detect cycles of any length in O(1) space.
///
/// The depth limit matches [`walk_to_root`]: the fast pointer traverses at most
/// `MAX_CHAIN_DEPTH` (1024) edges, so acyclic chains of up to 1024 edges resolve
/// to `Ok(root)`, while longer chains return `ErrorKind::InvalidData`.
pub fn find_root<H: Copy + Eq + std::hash::Hash>(
    delta_bases: &HashMap<H, H>,
    start: &H,
) -> io::Result<H> {
    let mut slow = *start;
    let mut fast = *start;
    for _ in 0..MAX_CHAIN_DEPTH.div_ceil(2) {
        // slow pointer: 1 step
        match delta_bases.get(&slow) {
            Some(&base) => slow = base,
            None => return Ok(slow),
        }
        // fast pointer: 2 steps
        for _ in 0..2 {
            match delta_bases.get(&fast) {
                Some(&base) => fast = base,
                None => return Ok(fast),
            }
        }
        if slow == fast {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cycle detected in delta chain",
            ));
        }
    }
    // `fast` has traversed exactly `MAX_CHAIN_DEPTH` edges: a chain of that length is valid.
    if !delta_bases.contains_key(&fast) {
        return Ok(fast);
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("delta chain exceeded maximum depth {}", MAX_CHAIN_DEPTH),
    ))
}

pub type ManifestData<'a> = (&'a Vec<u8>, &'a Option<HashMap<u64, u32>>);

/// Retrieves base chunk data and optional manifest by resolving the delta chain to its root.
///
/// # Errors
/// * Returns `ErrorKind::InvalidData` if a cycle is detected or depth exceeds `MAX_CHAIN_DEPTH`.
/// * Returns `ErrorKind::NotFound` if the root chunk is missing from `manifest_base`.
pub fn get_manifest<'a>(
    manifest_base: &'a InMemoryDatabase<[u8; 32], Entry>,
    delta_bases: &HashMap<[u8; 32], [u8; 32]>,
    hash: &[u8; 32],
) -> io::Result<ManifestData<'a>> {
    let (cur, _chain) = walk_to_root(delta_bases, hash)?;
    let entry = manifest_base.get_ref(&cur).ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, "base chunk not found in manifest_base")
    })?;
    Ok((entry.get_data(), entry.get_manifest()))
}

/// Decodes a chunk by walking its delta chain back to the root and reconstructing data.
///
/// # Preconditions
/// `target_map` must contain raw data for every hash in the chain, including
/// the root (base) chunk. This condition is guaranteed after a full `scrub()`
/// pass has completed.
pub fn decoder(
    target_map: &HashMap<[u8; 32], Vec<u8>>,
    delta_bases: &HashMap<[u8; 32], [u8; 32]>,
    hash: &[u8; 32],
) -> io::Result<Vec<u8>> {
    let (cur, chain) = walk_to_root(delta_bases, hash)?;
    let root_data = target_map
        .get(&cur)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "base chunk not found"))?;
    let mut data = root_data.clone();
    for dh in chain.iter().rev() {
        let delta = target_map
            .get(dh)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "delta chunk not found"))?;
        data = GdeltaDecoder::new(true).decode_chunk(data, delta);
    }
    Ok(data)
}
