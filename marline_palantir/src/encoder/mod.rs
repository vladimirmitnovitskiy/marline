//! Delta encoding abstractions for the Palantir pipeline.
//!
//! The [`PalantirEncoder`] trait defines the interface for encoding a new
//! chunk as a delta relative to a similar base chunk.  [`GdeltaEncoder`]
//! provides a concrete implementation using the gdelta algorithm.

use crate::GEAR;
use std::collections::HashMap;

/// Produces a delta encoding of `new_chunk` relative to `base_chunk`.
///
/// If two chunks are similar enough, the delta will be significantly smaller
/// than the raw chunk, saving storage space in a deduplication system.
pub trait PalantirEncoder {
    /// Encodes `new_chunk` as a delta with respect to `base_chunk`.
    ///
    /// # Arguments
    /// * `new_chunk` — The chunk to be delta-encoded.
    /// * `base_chunk` — The similar reference chunk.
    ///
    /// # Returns
    /// A byte vector containing the delta representation.
    fn encode(&self, new_chunk: &[u8], base_chunk: &[u8]) -> Vec<u8>;

    fn encode_with_manifest(
        &self,
        new_chunk: &[u8],
        base_chunk: &[u8],
        manifest: &HashMap<u64, u32>,
    ) -> Vec<u8>;
    fn generate_fp_table(chunk: &[u8]) -> Option<HashMap<u64, u32>>;
}

/// A [`PalantirEncoder`] that delegates to the gdelta diff algorithm.
///
/// Gdelta computes byte-level differences between two chunks and produces
/// a compact edit script.
pub struct GdeltaEncoder;

impl PalantirEncoder for GdeltaEncoder {
    fn encode(&self, new_chunk: &[u8], base_chunk: &[u8]) -> Vec<u8> {
        marline_scrub::encoder::gdelta_diff(new_chunk, base_chunk)
    }
    fn encode_with_manifest(
        &self,
        new_chunk: &[u8],
        base_chunk: &[u8],
        manifest: &HashMap<u64, u32>,
    ) -> Vec<u8> {
        marline_scrub::encoder::gdelta_diff_new(new_chunk, base_chunk, manifest)
    }

    /// Precomputes a word hash offset index (manifest) for a base chunk.
    ///
    /// # Design Invariants
    /// * `word_size = 16`: Matches the word window size expected by [`marline_scrub::encoder::gdelta_diff_new`].
    /// * `mask_bts = len.ilog2()` (exact `⌊log2 len⌋`; `len >= word_size` is guaranteed by the guard above):
    ///   consistent with [`marline_scrub::encoder::gdelta_diff_new`] for arbitrary, unequal base/new lengths.
    ///   The top `mask_bts` bits of the rolling hash `fp` form the lookup key (`word_hash`),
    ///   while full byte offsets `i` are preserved in the hash map values.
    fn generate_fp_table(base_chunk: &[u8]) -> Option<HashMap<u64, u32>> {
        let word_size: usize = 16;
        if base_chunk.len() < word_size {
            return None;
        }

        let move_bts: usize = 64 / word_size;
        let mask_bts: usize = base_chunk.len().ilog2() as usize;
        let mut word_hash_offsets: HashMap<u64, u32> = HashMap::with_capacity(base_chunk.len());
        let mut fp: u64 = 0;

        for i in 0..(word_size - 1) {
            fp = (fp << move_bts).wrapping_add(GEAR[base_chunk[i] as usize]);
        }

        for i in word_size..(base_chunk.len() - word_size + 1) {
            fp = (fp << move_bts).wrapping_add(GEAR[base_chunk[i + word_size - 1] as usize]);
            let word_hash: u64 = fp >> (64 - mask_bts);
            word_hash_offsets.insert(word_hash, i as u32);
        }
        Some(word_hash_offsets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use marline_scrub::decoder::{Decoder, GdeltaDecoder};

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
    fn manifest_encoding_matches_plain_encoding() {
        let enc = GdeltaEncoder;
        let patch = prng_data(99, 20_000);
        for (base_len, new_len) in
            [(1 << 20, 1 << 20), (1_000_003, 1_000_003), (700_001, 1_300_007), (1_900_013, 600_011)]
        {
            let base = prng_data(7, base_len);
            let mut new = prng_data(7, new_len);
            new[300_000..320_000].copy_from_slice(&patch);

            let manifest = GdeltaEncoder::generate_fp_table(&base).unwrap();
            let with_manifest = enc.encode_with_manifest(&new, &base, &manifest);
            let plain = enc.encode(&new, &base);

            assert_eq!(
                with_manifest, plain,
                "manifest encoding diverges for base_len={base_len}, new_len={new_len}"
            );
            let restored = GdeltaDecoder::default().decode_chunk(base, &with_manifest);
            assert!(
                restored == new,
                "round-trip failed for base_len={base_len}, new_len={new_len}"
            );
        }
    }

    #[test]
    fn fp_table_rejects_inputs_shorter_than_word() {
        let data = prng_data(13, 16);
        assert!(GdeltaEncoder::generate_fp_table(&data[..15]).is_none());
        assert!(GdeltaEncoder::generate_fp_table(&data).is_some());
    }
}
