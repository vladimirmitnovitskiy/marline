use crate::encoder::PalantirEncoder;
use crate::error::PalantirError;
use crate::lifecycle_manager::LifecycleTierConfig;
use crate::metadata_manager::MetadataManager;
use crate::palantir_scrubber::find_root;
use crate::sf_generator::SuperFeatureGenerator;
use crate::types::{BlockID, TierConfig};
use marline_cdc::{collect_raw_features, ChunkerBuilder};
use marline_index::heuristic_index::SearchConfig;
use marline_storage::{ChunkHash, MetricsDatabase};
use std::collections::HashMap;

fn zstd_error(code: zstd::zstd_safe::ErrorCode) -> PalantirError {
    PalantirError::ChunkError(zstd::zstd_safe::get_error_name(code).to_owned())
}

pub trait OnlineSBC<S, H: ChunkHash, E, const N: usize> {
    fn write(&mut self, data: &[u8]) -> Result<(), PalantirError>;
    fn write_stream<R: std::io::Read>(&mut self, reader: R) -> Result<(), PalantirError>;
    fn bytes_stored(&self) -> usize;
}

pub struct PalantirOnline<S, H: ChunkHash, E: PalantirEncoder, const N: usize> {
    sf_gen: S,
    /// Multi-tier similarity index.
    metadata_manager: MetadataManager<H, N>,
    /// Delta encoder.
    encoder: E,
    /// False-positive threshold for delta encoding ratio.
    fp_threshold: f64,
    /// Compute super-features with a second scan over each chunk (baseline
    /// two-pass mode) instead of reusing raw features collected during chunking.
    naive: bool,
    /// Total chunks processed through the pipeline (including duplicates and small chunks).
    chunks_processed: u64,
    /// Sum of lengths of all processed chunks, including duplicates and small chunks;
    /// the denominator of the data reduction ratio.
    logical_bytes: u64,
    /// Set after a delta-index invariant violation; all further writes are rejected.
    poisoned: bool,
    /// Database for chunk storage and metrics.
    db: MetricsDatabase<H>,
    /// Reusable hasher to avoid re-allocating state per chunk.
    hasher: blake3::Hasher,
    /// Reusable zstd compression context (avoids creating a context per chunk).
    cctx: zstd::zstd_safe::CCtx<'static>,
    /// Reusable compression buffer for evaluating simple compression size.
    /// `clear` + `reserve` up to the worst-case compressed bound before each use;
    /// capacity stabilizes after the first large chunk.
    compress_buf: Vec<u8>,
    /// Reusable compression buffer for evaluating delta compression size.
    /// `clear` + `reserve` up to the worst-case compressed bound before each use;
    /// capacity stabilizes after the first large chunk.
    delta_compress_buf: Vec<u8>,
    /// Whether manifests (precalculated word-offset tables) are generated for new base chunks.
    manifests_enabled: bool,
    /// Store of manifests for base chunks: chunk hash -> word offset map.
    ///
    /// Each table occupies roughly 10–16× the size of its base chunk. There is no
    /// eviction policy: memory is reclaimed only via [`Self::remove_manifest`] /
    /// [`Self::clear_manifests`].
    manifests: HashMap<H, HashMap<u64, u32>>,
}

impl<S: SuperFeatureGenerator, E: PalantirEncoder, const N: usize>
    PalantirOnline<S, [u8; 32], E, N>
{
    fn process_chunk(&mut self, chunk_data: &[u8], sf_raw: [u32; 12]) -> Result<(), PalantirError> {
        self.chunks_processed += 1;
        self.logical_bytes += chunk_data.len() as u64;

        self.hasher.reset();
        self.hasher.update(chunk_data);
        let hash: [u8; 32] = *self.hasher.finalize().as_bytes();

        // Защита от дубликатов (как base, так и deltas)
        if self.db.contains(&hash) {
            return Ok(());
        }

        let min_size = ChunkerBuilder::default().min_size();
        // Защита от ложных совпадений: если чанк <= min_size, feature-окно пусто, сохраняем как base
        if chunk_data.len() <= min_size {
            self.db.insert(hash, chunk_data);
            return Ok(());
        }

        let super_features = if self.naive {
            let sf = collect_raw_features(chunk_data, min_size);
            debug_assert_eq!(sf, sf_raw, "two-pass features must equal in-pass");
            self.sf_gen.generate_from_raw(&sf)
        } else {
            self.sf_gen.generate_from_raw(&sf_raw)
        };

        match self.metadata_manager.lookup_super_features(&super_features) {
            Some((base_hash, _)) => {
                // Находим корень цепочки: алгоритм Флойда, O(1) по памяти (по времени O(глубина))
                let root_hash = match find_root(self.db.delta_bases(), &base_hash.hash) {
                    Ok(root) => root,
                    Err(e) => {
                        self.poisoned = true;
                        return Err(PalantirError::MetadataError(e.to_string()));
                    }
                };

                // Политика B: рассинхрон индекса фатален в debug и release, без fallback.
                let Some(base_data) = self.db.get_ref(&root_hash) else {
                    self.poisoned = true;
                    return Err(PalantirError::MetadataError(format!(
                        "index desync: root {} is not a stored base chunk",
                        blake3::Hash::from_bytes(root_hash).to_hex()
                    )));
                };

                let delta = if let Some(manifest) = self.manifests.get(&root_hash) {
                    self.encoder.encode_with_manifest(chunk_data, base_data, manifest)
                } else {
                    self.encoder.encode(chunk_data, base_data)
                };

                // Сжимаем дельту в переиспользуемый буфер
                self.delta_compress_buf.clear();
                self.delta_compress_buf.reserve(zstd::zstd_safe::compress_bound(delta.len()));
                let delta_comp_size = self
                    .cctx
                    .compress(&mut self.delta_compress_buf, &delta, 0)
                    .map_err(zstd_error)?;

                // Вычисляем размер простого сжатия чанка в переиспользуемый буфер
                self.compress_buf.clear();
                self.compress_buf.reserve(zstd::zstd_safe::compress_bound(chunk_data.len()));
                let simple_size = self
                    .cctx
                    .compress(&mut self.compress_buf, chunk_data, 0)
                    .map_err(zstd_error)?;

                if simple_size > 0
                    && (delta_comp_size as f64) < (simple_size as f64) * self.fp_threshold
                {
                    // Кодируем против корня намеренно: цепочки остаются плоскими,
                    // restore — один шаг декодирования (O(1)). Известный компромисс:
                    // на эволюционирующих данных дельты к корню растут со временем.
                    // Не заменять на `base_hash.hash`.
                    if !self.db.insert_delta(hash, root_hash, &self.delta_compress_buf) {
                        self.poisoned = true;
                        return Err(PalantirError::MetadataError(
                            "index desync: insert_delta rejected a verified root".into(),
                        ));
                    }
                } else {
                    self.save_base_chunk(hash, chunk_data);
                }
            }
            None => {
                self.save_base_chunk(hash, chunk_data);
            }
        }

        self.metadata_manager.add_block(hash, &super_features, BlockID::new(hash));
        Ok(())
    }

    fn ensure_not_poisoned(&self) -> Result<(), PalantirError> {
        if self.poisoned {
            return Err(PalantirError::MetadataError(
                "instance poisoned by earlier index corruption".into(),
            ));
        }
        Ok(())
    }

    fn save_base_chunk(&mut self, hash: [u8; 32], chunk_data: &[u8]) {
        self.db.insert(hash, chunk_data);
        if self.manifests_enabled {
            if let Some(manifest) = E::generate_fp_table(chunk_data) {
                self.manifests.insert(hash, manifest);
            }
        }
    }
}

impl<S: SuperFeatureGenerator, E: PalantirEncoder, const N: usize> OnlineSBC<S, [u8; 32], E, N>
    for PalantirOnline<S, [u8; 32], E, N>
{
    /// Chunks `data` in memory and processes every chunk through the pipeline.
    ///
    /// # Errors
    /// * [`PalantirError::MetadataError`] — delta-index corruption (cycle, excessive chain
    ///   depth, a root missing from storage, or a rejected delta insert). The instance is
    ///   poisoned: this and all further writes are rejected (see [`Self::is_poisoned`]).
    /// * Any other error (chunking, I/O, compression) aborts only this call; the instance
    ///   stays usable and chunks processed before the error remain stored.
    fn write(&mut self, data: &[u8]) -> Result<(), PalantirError> {
        self.ensure_not_poisoned()?;
        let chunker = ChunkerBuilder::default().chunker(data);
        for (chunk, sf_raw) in chunker {
            let chunk_data = chunk
                .offset
                .checked_add(chunk.length)
                .and_then(|end| data.get(chunk.offset..end))
                .ok_or_else(|| PalantirError::ChunkError("chunk out of bounds".into()))?;
            self.process_chunk(chunk_data, sf_raw)?;
        }
        Ok(())
    }

    /// Chunks the `reader` stream and processes every chunk through the pipeline.
    ///
    /// # Errors
    /// * [`PalantirError::MetadataError`] — delta-index corruption (cycle, excessive chain
    ///   depth, a root missing from storage, or a rejected delta insert). The instance is
    ///   poisoned: this and all further writes are rejected (see [`Self::is_poisoned`]).
    /// * Any other error (chunking, I/O, compression) aborts only this call; the instance
    ///   stays usable and chunks processed before the error remain stored.
    fn write_stream<R: std::io::Read>(&mut self, reader: R) -> Result<(), PalantirError> {
        self.ensure_not_poisoned()?;
        let chunker = ChunkerBuilder::default().stream_chunker(reader);
        for result in chunker {
            let (chunk, sf_raw) = result.map_err(|e| PalantirError::ChunkError(e.to_string()))?;
            self.process_chunk(&chunk.data, sf_raw)?;
        }
        Ok(())
    }

    fn bytes_stored(&self) -> usize {
        self.db.total_saved_bytes()
    }
}

impl<S, H: ChunkHash, E: PalantirEncoder, const N: usize> PalantirOnline<S, H, E, N> {
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
            naive: false,
            chunks_processed: 0,
            logical_bytes: 0,
            poisoned: false,
            db: MetricsDatabase::new(),
            hasher: blake3::Hasher::new(),
            cctx: zstd::zstd_safe::CCtx::create(),
            compress_buf: Vec::new(),
            delta_compress_buf: Vec::new(),
            manifests_enabled: false,
            manifests: HashMap::new(),
        }
    }

    /// Builder method to configure automatic manifest generation for base chunks.
    ///
    /// Memory: each manifest occupies roughly 10–16× the size of its base chunk.
    /// Manifests are never evicted automatically; reclaim memory via
    /// [`Self::remove_manifest`] / [`Self::clear_manifests`].
    pub fn with_manifests(mut self, enabled: bool) -> Self {
        self.manifests_enabled = enabled;
        self
    }

    /// Dynamically switches automatic manifest generation on or off.
    pub fn set_manifests_enabled(&mut self, enabled: bool) {
        self.manifests_enabled = enabled;
    }

    /// Returns whether automatic manifest generation is enabled.
    pub fn manifests_enabled(&self) -> bool {
        self.manifests_enabled
    }

    /// Generates and stores a manifest for an existing base chunk in storage.
    /// Returns true if the chunk exists in `db` and the manifest was generated.
    pub fn generate_manifest_for(&mut self, hash: &H) -> bool {
        if let Some(base_data) = self.db.get_ref(hash) {
            if let Some(manifest) = E::generate_fp_table(base_data) {
                self.manifests.insert(hash.clone(), manifest);
                return true;
            }
        }
        false
    }

    /// Removes a precomputed manifest for a base chunk to reclaim memory.
    /// The base chunk remains in storage; future deltas against it will use on-the-fly encoding.
    pub fn remove_manifest(&mut self, hash: &H) -> Option<HashMap<u64, u32>> {
        self.manifests.remove(hash)
    }

    /// Clears all stored manifests to reclaim memory.
    pub fn clear_manifests(&mut self) {
        self.manifests.clear();
    }

    /// Returns whether a manifest is stored for the given chunk hash.
    pub fn has_manifest(&self, hash: &H) -> bool {
        self.manifests.contains_key(hash)
    }

    /// Returns a reference to the manifest for the given chunk hash, if present.
    pub fn manifest(&self, hash: &H) -> Option<&HashMap<u64, u32>> {
        self.manifests.get(hash)
    }

    /// Returns the number of manifests currently cached in memory.
    pub fn manifests_count(&self) -> usize {
        self.manifests.len()
    }

    pub fn set_naive(&mut self, naive: bool) {
        self.naive = naive;
    }

    /// Returns total raw uncompressed bytes stored for base chunks.
    pub fn base_bytes(&self) -> usize {
        self.db.base_bytes()
    }

    /// Returns total compressed bytes accounted for delta chunks.
    pub fn delta_bytes(&self) -> usize {
        self.db.delta_bytes()
    }

    pub fn chunks_processed(&self) -> u64 {
        self.chunks_processed
    }

    /// Sum of lengths of all processed chunks, including duplicates and small chunks.
    pub fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    /// Whether the instance was poisoned by delta-index corruption (all writes are rejected).
    /// Metric getters remain available and reflect the state at the moment of failure
    /// (`logical_bytes` / `chunks_processed` include the failing chunk).
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn unique_chunks_count(&self) -> usize {
        self.db.unique_chunks_count()
    }

    pub fn delta_chunks_count(&self) -> usize {
        self.db.delta_chunks_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::GdeltaEncoder;
    use crate::lifecycle_manager::LifecycleManager;
    use crate::sf_generator::PalantirHasher;
    use std::io::Cursor;

    fn prng_data(seed: u64, len: usize) -> Vec<u8> {
        let mut buf = Vec::with_capacity(len);
        let mut s = seed;
        for _ in 0..len {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            buf.push((s >> 33) as u8);
        }
        buf
    }

    #[test]
    fn test_palantir_online_exact_duplicate() {
        let data = prng_data(42, 2 * 1024 * 1024);
        let sf_gen = PalantirHasher::new(7, vec![4, 3, 2]);
        let tier_config = TierConfig::new([4, 3, 2]);
        let lifecycle_configs = LifecycleManager::<3>::default_configs();
        let mut online = PalantirOnline::new(
            sf_gen,
            GdeltaEncoder,
            tier_config,
            lifecycle_configs,
            SearchConfig::default(),
        );

        online.write(&data).unwrap();
        let stored_first = online.bytes_stored();
        assert!(stored_first > 0);
        assert_eq!(online.delta_chunks_count(), 0);

        online.write(&data).unwrap();
        assert_eq!(online.bytes_stored(), stored_first);
        assert_eq!(online.delta_chunks_count(), 0);
    }

    #[test]
    fn test_palantir_online_similarity_delta() {
        let base = prng_data(42, 2 * 1024 * 1024);
        let mut similar = base.clone();
        let mut_len = (base.len() * 2) / 100;
        for b in &mut similar[1024 * 512..1024 * 512 + mut_len] {
            *b = b.wrapping_add(42);
        }

        let sf_gen = PalantirHasher::new(7, vec![4, 3, 2]);
        let tier_config = TierConfig::new([4, 3, 2]);
        let lifecycle_configs = LifecycleManager::<3>::default_configs();
        let mut online = PalantirOnline::new(
            sf_gen,
            GdeltaEncoder,
            tier_config,
            lifecycle_configs,
            SearchConfig::default(),
        );

        online.write(&base).unwrap();
        let base_stored = online.bytes_stored();
        assert!(base_stored > 0);
        assert_eq!(online.delta_chunks_count(), 0);

        online.write(&similar).unwrap();
        assert!(online.delta_chunks_count() > 0);
        assert!(online.bytes_stored() < base_stored * 2);
        assert!(online.base_bytes() > 0);
        assert!(online.delta_bytes() > 0);
        assert_eq!(online.bytes_stored(), online.base_bytes() + online.delta_bytes());

        // Re-writing similar data (containing deltas) must be completely idempotent:
        let stored_second = online.bytes_stored();
        let deltas_second = online.delta_chunks_count();
        online.write(&similar).unwrap();
        assert_eq!(
            online.bytes_stored(),
            stored_second,
            "delta re-write must not increase bytes_stored"
        );
        assert_eq!(
            online.delta_chunks_count(),
            deltas_second,
            "delta re-write must not increase delta_stored"
        );
    }

    #[test]
    fn test_palantir_online_streaming_matches_slice() {
        let data = prng_data(42, 2 * 1024 * 1024);
        let sf_gen = PalantirHasher::new(7, vec![4, 3, 2]);

        let mut online_slice = PalantirOnline::new(
            sf_gen.clone(),
            GdeltaEncoder,
            TierConfig::new([4, 3, 2]),
            LifecycleManager::<3>::default_configs(),
            SearchConfig::default(),
        );
        let mut online_stream = PalantirOnline::new(
            sf_gen,
            GdeltaEncoder,
            TierConfig::new([4, 3, 2]),
            LifecycleManager::<3>::default_configs(),
            SearchConfig::default(),
        );

        online_slice.write(&data).unwrap();
        online_stream.write_stream(Cursor::new(&data)).unwrap();

        assert_eq!(online_slice.bytes_stored(), online_stream.bytes_stored());
        assert_eq!(online_slice.chunks_processed(), online_stream.chunks_processed());
        assert!(online_slice.chunks_processed() > 0);
    }

    fn new_online() -> PalantirOnline<PalantirHasher, [u8; 32], GdeltaEncoder, 3> {
        PalantirOnline::new(
            PalantirHasher::new(7, vec![4, 3, 2]),
            GdeltaEncoder,
            TierConfig::new([4, 3, 2]),
            LifecycleManager::<3>::default_configs(),
            SearchConfig::default(),
        )
    }

    #[test]
    fn logical_bytes_counts_every_processed_byte() {
        let first = prng_data(1, 3 * 1024 * 1024);
        let second = prng_data(2, 1024 * 1024 + 12_345);
        let mut online = new_online();

        online.write(&first).unwrap();
        online.write(&second).unwrap();
        let total = (first.len() + second.len()) as u64;
        assert_eq!(online.logical_bytes(), total);

        let stored = online.bytes_stored();
        online.write(&first).unwrap();
        online.write(&second).unwrap();
        assert_eq!(online.logical_bytes(), 2 * total, "duplicates count as logical bytes");
        assert_eq!(online.bytes_stored(), stored, "duplicates must not be stored again");
    }

    #[test]
    fn logical_bytes_streaming_matches_slice() {
        let data = prng_data(3, 5 * 1024 * 1024 + 777);
        let mut online_slice = new_online();
        let mut online_stream = new_online();

        online_slice.write(&data).unwrap();
        online_stream.write_stream(Cursor::new(&data)).unwrap();

        assert_eq!(online_slice.logical_bytes(), data.len() as u64);
        assert_eq!(online_stream.logical_bytes(), data.len() as u64);
    }

    #[test]
    fn index_desync_is_fatal_and_poisons_instance() {
        let min_size = ChunkerBuilder::default().min_size();
        let chunk = prng_data(4, min_size + 64 * 1024);
        let sf_raw = collect_raw_features(&chunk, min_size);
        let mut online = new_online();
        online.write(&prng_data(6, 3 * 1024 * 1024)).unwrap();
        let super_features = online.sf_gen.generate_from_raw(&sf_raw);

        // Similarity index points at a hash that was never stored in `db`.
        let ghost_hash = *blake3::hash(b"ghost base").as_bytes();
        assert!(!online.db.contains(&ghost_hash));
        online.metadata_manager.add_block(ghost_hash, &super_features, BlockID::new(ghost_hash));

        let chunk_hash = *blake3::hash(&chunk).as_bytes();
        let snapshot = |db: &MetricsDatabase<[u8; 32]>| {
            (
                db.base_bytes(),
                db.delta_bytes(),
                db.unique_chunks_count(),
                db.delta_chunks_count(),
                db.delta_bases().len(),
            )
        };
        let before = snapshot(&online.db);

        assert!(!online.is_poisoned());
        let err = online.process_chunk(&chunk, sf_raw).unwrap_err();
        assert!(online.is_poisoned());
        assert!(matches!(err, PalantirError::MetadataError(ref m) if m.contains("index desync")));
        assert!(!online.db.contains(&chunk_hash), "failed chunk must not be stored");
        assert_eq!(snapshot(&online.db), before, "db must not change on index desync");

        // Poisoned: every further write is rejected without touching the store.
        let err = online.write(&prng_data(8, 2 * 1024 * 1024)).unwrap_err();
        assert!(matches!(err, PalantirError::MetadataError(ref m) if m.contains("poisoned")));
        let err = online.write_stream(Cursor::new(prng_data(9, 1024))).unwrap_err();
        assert!(matches!(err, PalantirError::MetadataError(_)));
        assert_eq!(snapshot(&online.db), before);
    }

    struct FailingReader;

    impl std::io::Read for FailingReader {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("simulated I/O failure"))
        }
    }

    #[test]
    fn io_error_does_not_poison_instance() {
        let mut online = new_online();
        let err = online.write_stream(FailingReader).unwrap_err();
        assert!(!online.is_poisoned());
        assert!(matches!(err, PalantirError::ChunkError(_)));

        let data = prng_data(10, 3 * 1024 * 1024);
        online.write(&data).unwrap();
        assert!(!online.is_poisoned());
        assert!(online.bytes_stored() > 0);
        assert_eq!(online.logical_bytes(), data.len() as u64);
    }

    #[test]
    fn test_palantir_online_manifest_configuration_and_lifecycle() {
        let base = prng_data(42, 2 * 1024 * 1024);
        let mut similar = base.clone();
        let mut_len = (base.len() * 2) / 100;
        for b in &mut similar[1024 * 512..1024 * 512 + mut_len] {
            *b = b.wrapping_add(42);
        }

        let sf_gen = PalantirHasher::new(7, vec![4, 3, 2]);
        let mut online = PalantirOnline::new(
            sf_gen,
            GdeltaEncoder,
            TierConfig::new([4, 3, 2]),
            LifecycleManager::<3>::default_configs(),
            SearchConfig::default(),
        )
        .with_manifests(true);

        assert!(online.manifests_enabled());
        assert_eq!(online.manifests_count(), 0);

        // Write base: manifests should be automatically created for large base chunks
        online.write(&base).unwrap();
        assert!(online.manifests_count() > 0, "Manifests must be generated when enabled");

        // Collect hashes of stored manifests
        let manifest_hashes: Vec<[u8; 32]> = online.manifests.keys().cloned().collect();
        let target_hash = manifest_hashes[0];
        assert!(online.has_manifest(&target_hash));
        assert!(online.manifest(&target_hash).is_some());

        // Test removing a manifest ("убирать")
        let removed = online.remove_manifest(&target_hash);
        assert!(removed.is_some(), "Manifest must be removed");
        assert!(!online.has_manifest(&target_hash));

        // Test generating manifest back on demand ("добавлять")
        assert!(online.generate_manifest_for(&target_hash));
        assert!(online.has_manifest(&target_hash));

        // Write similar: delta encoding works with manifest
        online.write(&similar).unwrap();
        assert!(online.delta_chunks_count() > 0);

        // Test disabling manifests dynamically
        online.set_manifests_enabled(false);
        assert!(!online.manifests_enabled());

        // Clear all manifests
        online.clear_manifests();
        assert_eq!(online.manifests_count(), 0);

        // Even with manifests cleared, delta encoding falls back to encode() smoothly
        let mut similar2 = base.clone();
        for b in &mut similar2[1024 * 512..1024 * 512 + mut_len] {
            *b = b.wrapping_add(43);
        }
        online.write(&similar2).unwrap();
        assert!(
            online.delta_chunks_count() > 1,
            "Delta encoding must work without manifests via fallback"
        );
    }
}
