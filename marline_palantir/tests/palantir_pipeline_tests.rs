use std::collections::HashMap;

use marline_index::heuristic_index::SearchConfig;
use marline_palantir::encoder::GdeltaEncoder;
use marline_palantir::lifecycle_manager::LifecycleManager;
use marline_palantir::metadata_manager::MetadataManager;
use marline_palantir::mock_rocksdb::Entry;
use marline_palantir::palantir_scrubber::{get_manifest, PalantirScrubber};
use marline_palantir::sf_generator::{PalantirHasher, SuperFeatureGenerator};
use marline_palantir::types::{BlockID, Chunk, TierConfig};
use marline_storage::{Data, DataContainer, Database, InMemoryDatabase};
use sha2::{Digest, Sha256};

fn hash_data(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

#[test]
fn identical_chunks() {
    let sf_gen = PalantirHasher::new(7, vec![4, 3, 2]);
    let tier_config = TierConfig::new([4, 3, 2]);
    let lifecycle_configs = LifecycleManager::<3>::default_configs();
    let mut mm = MetadataManager::<[u8; 32], 3>::new(
        tier_config,
        lifecycle_configs,
        &SearchConfig::default(),
    );

    let base: Vec<u8> = b"This is a base chunk for the Palantir similarity pipeline. \
                           It has enough bytes for the gear-hash rolling hash to produce \
                           stable super-features across multiple tiers."
        .to_vec();
    let hash = hash_data(&base);

    let chunk = Chunk::new(base);
    let sfs = sf_gen.generate(&chunk);
    mm.add_block(hash, &sfs, BlockID::new(hash));

    let found = mm.lookup_fingerprint(&hash);
    assert!(found.is_some(), "Identical chunk not found via fingerprint lookup");
}

#[test]
fn similar_chunks() {
    let sf_gen = PalantirHasher::new(7, vec![4, 3, 2]);
    let encoder = GdeltaEncoder;
    let tier_config = TierConfig::new([4, 3, 2]);
    let lifecycle_configs = LifecycleManager::<3>::default_configs();
    let mut scrubber = PalantirScrubber::new(
        sf_gen,
        encoder,
        tier_config,
        lifecycle_configs,
        SearchConfig::default(),
    );

    let base: Vec<u8> = (0..8192).map(|i| (i % 256) as u8).collect();
    let mut similar = base.clone();
    for item in similar[4096..4916].iter_mut() {
        *item = item.wrapping_add(7);
    }

    let hash_base = hash_data(&base);
    let hash_similar = hash_data(&similar);

    let mut database = InMemoryDatabase::new();
    database.insert(hash_base, DataContainer::from(base.clone())).unwrap();
    database.insert(hash_similar, DataContainer::from(similar.clone())).unwrap();

    let mut target_map = HashMap::new();
    let result = scrubber.scrub(&mut database, &mut target_map).unwrap();

    assert_eq!(result.processed_data, base.len() + similar.len());
    assert_eq!(target_map.len(), 2);

    let delta_count = [(&hash_base, &base), (&hash_similar, &similar)]
        .into_iter()
        .filter(|(h, data)| target_map.get(*h).unwrap().len() < data.len())
        .count();
    assert!(
        delta_count >= 1,
        "At least one chunk should be delta-encoded, got {} deltas",
        delta_count
    );

    for (hash, data) in [(&hash_base, &base), (&hash_similar, &similar)] {
        let v = target_map.get(hash).unwrap();
        if v.len() < data.len() {
            let delta_zstd = zstd::encode_all(v.as_slice(), 0).unwrap();
            let raw_zstd = zstd::encode_all(data.as_slice(), 0).unwrap();
            let ratio = delta_zstd.len() as f64 / raw_zstd.len() as f64;
            assert!(ratio < 0.9, "Filter should preserve delta, ratio={:.2}", ratio);
            break;
        }
    }

    for hash in [&hash_base, &hash_similar] {
        let container = database.get_ref(hash).expect("chunk should exist in db");
        match container.data() {
            Data::TargetChunk(keys) => assert_eq!(keys.as_slice(), &[*hash]),
            _ => panic!("Expected TargetChunk after scrub"),
        }
    }
}

#[test]
fn test_delta_chain_cycle_detection() {
    let mut delta_bases = HashMap::new();
    let hash_a = [1u8; 32];
    let hash_b = [2u8; 32];
    // Create cycle A -> B -> A
    delta_bases.insert(hash_a, hash_b);
    delta_bases.insert(hash_b, hash_a);

    let target_map = HashMap::new();
    let res = marline_palantir::palantir_scrubber::decoder(&target_map, &delta_bases, &hash_a);
    assert!(res.is_err());
    assert_eq!(res.unwrap_err().kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn test_delta_chain_missing_base_not_found() {
    let mut delta_bases = HashMap::new();
    let hash_a = [1u8; 32];
    let hash_b = [2u8; 32];
    delta_bases.insert(hash_a, hash_b);

    let manifest_base = marline_storage::InMemoryDatabase::new();
    let res =
        marline_palantir::palantir_scrubber::get_manifest(&manifest_base, &delta_bases, &hash_a);
    assert!(res.is_err());
    assert_eq!(res.unwrap_err().kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn test_delta_chain_depth_limit() {
    let mut delta_bases = HashMap::new();
    let mut cur = [0u8; 32];
    for i in 1..=1025 {
        let mut next = [0u8; 32];
        next[0..4].copy_from_slice(&(i as u32).to_le_bytes());
        delta_bases.insert(cur, next);
        cur = next;
    }

    let manifest_base = marline_storage::InMemoryDatabase::new();
    let res =
        marline_palantir::palantir_scrubber::get_manifest(&manifest_base, &delta_bases, &[0u8; 32]);
    assert!(res.is_err());
    assert_eq!(res.unwrap_err().kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn test_find_root_multi_node_cycle_detection() {
    let mut delta_bases = HashMap::new();
    let hash_a = [1u8; 32];
    let hash_b = [2u8; 32];
    // Create 2-node cycle A -> B -> A
    delta_bases.insert(hash_a, hash_b);
    delta_bases.insert(hash_b, hash_a);

    let res = marline_palantir::palantir_scrubber::find_root(&delta_bases, &hash_a);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("cycle detected"));
}

#[test]
fn test_find_root_self_loop_cycle_detection() {
    let mut delta_bases = HashMap::new();
    let hash_a = [1u8; 32];
    // Create self-loop A -> A
    delta_bases.insert(hash_a, hash_a);

    let res = marline_palantir::palantir_scrubber::find_root(&delta_bases, &hash_a);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("cycle detected"));
}

#[test]
fn test_find_root_three_node_cycle_detection() {
    let mut delta_bases = HashMap::new();
    let hash_a = [1u8; 32];
    let hash_b = [2u8; 32];
    let hash_c = [3u8; 32];
    // Create 3-node cycle A -> B -> C -> A
    delta_bases.insert(hash_a, hash_b);
    delta_bases.insert(hash_b, hash_c);
    delta_bases.insert(hash_c, hash_a);

    let res = marline_palantir::palantir_scrubber::find_root(&delta_bases, &hash_a);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("cycle detected"));
}

/// Must stay in sync with `MAX_CHAIN_DEPTH` in `palantir_scrubber`.
const MAX_CHAIN_DEPTH: usize = 1024;

fn node_hash(i: u64) -> [u8; 32] {
    hash_data(&i.to_le_bytes())
}

/// Builds a linear chain `node(0) -> node(1) -> ... -> node(edges)`; returns `(map, root)`.
fn linear_chain(edges: usize) -> (HashMap<[u8; 32], [u8; 32]>, [u8; 32]) {
    let mut delta_bases = HashMap::with_capacity(edges);
    for i in 0..edges as u64 {
        delta_bases.insert(node_hash(i), node_hash(i + 1));
    }
    (delta_bases, node_hash(edges as u64))
}

#[test]
fn find_root_returns_start_when_absent_from_map() {
    let (delta_bases, _) = linear_chain(3);
    let start = node_hash(1_000_000);
    let res = marline_palantir::palantir_scrubber::find_root(&delta_bases, &start).unwrap();
    assert_eq!(res, start);
}

#[test]
fn find_root_resolves_short_chain() {
    let (delta_bases, root) = linear_chain(2);
    let res = marline_palantir::palantir_scrubber::find_root(&delta_bases, &node_hash(0)).unwrap();
    assert_eq!(res, root);
}

#[test]
fn find_root_accepts_chain_of_max_depth() {
    let (delta_bases, root) = linear_chain(MAX_CHAIN_DEPTH);
    let res = marline_palantir::palantir_scrubber::find_root(&delta_bases, &node_hash(0)).unwrap();
    assert_eq!(res, root);
}

#[test]
fn find_root_rejects_chain_exceeding_max_depth() {
    let (delta_bases, _) = linear_chain(MAX_CHAIN_DEPTH + 1);
    let err =
        marline_palantir::palantir_scrubber::find_root(&delta_bases, &node_hash(0)).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn walk_to_root_depth_limit_matches_find_root() {
    let (delta_bases, root) = linear_chain(MAX_CHAIN_DEPTH);
    let mut manifest_base = InMemoryDatabase::new();
    let root_data = hash_data(b"root payload").to_vec();
    manifest_base.insert(root, Entry::from(root_data.clone())).unwrap();

    let (data, manifest) = get_manifest(&manifest_base, &delta_bases, &node_hash(0)).unwrap();
    assert_eq!(data, &root_data);
    assert!(manifest.is_none());

    let (delta_bases, root) = linear_chain(MAX_CHAIN_DEPTH + 1);
    manifest_base.insert(root, Entry::from(root_data)).unwrap();
    let err = get_manifest(&manifest_base, &delta_bases, &node_hash(0)).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}
