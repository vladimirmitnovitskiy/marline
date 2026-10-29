pub mod fastcdc;

pub use fastcdc::{
    collect_raw_features, cut_gear, Chunk, ChunkData, FastCDC, Normalization,
    StreamCDC, GEAR, MASKS,
};

const KB: usize = 1024;

#[derive(Debug, Clone, Copy)]
pub struct CdcConfig {
    pub min_size: usize,
    pub avg_size: usize,
    pub max_size: usize,
}

impl Default for CdcConfig {
    fn default() -> Self {
        Self {
            min_size: 512 * KB,
            avg_size: KB * KB,
            max_size: 2 * KB * KB,
        }
    }
}

impl CdcConfig {
    pub fn new(min_size: usize, avg_size: usize, max_size:usize) -> Self{
        Self {
            min_size,
            avg_size,
            max_size,
        }
    }

    pub fn chunk_slice<'a>(&self, data: &'a [u8]) -> FastCDC<'a> {
        FastCDC::new(data, self.min_size, self.avg_size, self.max_size)
    }

    pub fn chunk_stream<R: std::io::Read>(&self, reader: R) -> StreamCDC<R> {
        StreamCDC::new(reader, self.min_size, self.avg_size, self.max_size)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // Детерминированный генератор без коротких циклов (в отличие от `% 256`),
    // чтобы скользящий хэш GEAR работал в реалистичных условиях.
    fn generate_test_data(size: usize) -> Vec<u8> {
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        (0..size)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 0xFF) as u8
            })
            .collect()
    }

    #[test]
    fn test_cdc_config_default() {
        let config = CdcConfig::default();
        assert_eq!(config.min_size, 512 * KB);
        assert_eq!(config.avg_size, 1024 * KB);
        assert_eq!(config.max_size, 2048 * KB);
    }

    #[test]
    fn test_chunk_slice_empty() {
        let config = CdcConfig::new(256, 1024, 4096);
        let data: &[u8] = &[];
        let chunks: Vec<_> = config.chunk_slice(data).collect();

        assert!(chunks.is_empty(), "Empty input should not yield any chunks");
    }

    #[test]
    fn test_chunk_stream_empty() {
        let config = CdcConfig::new(256, 1024, 4096);
        let reader = Cursor::new(Vec::<u8>::new());
        let chunks: Vec<_> = config
            .chunk_stream(reader)
            .collect::<Result<Vec<_>, _>>()
            .expect("Reading an empty stream should succeed");

        assert!(chunks.is_empty(), "Empty stream should produce no chunks");
    }

    #[test]
    fn test_chunk_slice_data_smaller_than_min_size() {
        let config = CdcConfig::new(256, 1024, 4096);
        let data = vec![0xAB; 128]; // Меньше чем min_size (256)
        let chunks: Vec<_> = config.chunk_slice(&data).collect();

        assert_eq!(chunks.len(), 1, "Exactly one trailing chunk should be produced");

        let (chunk, _features) = &chunks[0];
        assert_eq!(chunk.offset, 0);
        assert_eq!(chunk.length, 128);
    }

    #[test]
    fn test_chunk_slice_invariants() {
        let min_size = 256;
        let avg_size = 1024;
        let max_size = 4096;
        let config = CdcConfig::new(min_size, avg_size, max_size);

        let data = generate_test_data(64 * KB);
        let chunks: Vec<_> = config.chunk_slice(&data).collect();

        assert!(!chunks.is_empty());

        let mut total_bytes = 0;
        let num_chunks = chunks.len();

        for (i, (chunk, _features)) in chunks.iter().enumerate() {
            assert_eq!(chunk.offset, total_bytes, "Chunk offset sequence is broken at chunk {i}");
            total_bytes += chunk.length;

            if i < num_chunks - 1 {
                assert!(
                    chunk.length >= min_size,
                    "Chunk {i} length {} is smaller than min_size ({})",
                    chunk.length,
                    min_size
                );
            }

            assert!(
                chunk.length <= max_size,
                "Chunk {i} length {} exceeds max_size ({})",
                chunk.length,
                max_size
            );
        }

        assert_eq!(total_bytes, data.len());
    }

    #[test]
    fn test_uniform_data_hits_max_size() {
        let min_size = 256;
        let avg_size = 1024;
        let max_size = 4096;
        let config = CdcConfig::new(min_size, avg_size, max_size);

        // Однородные данные проверяют принудительную отсечку по max_size
        let data = vec![0u8; 20 * KB];
        let chunks: Vec<_> = config.chunk_slice(&data).collect();

        for (chunk, _) in &chunks {
            assert!(chunk.length <= max_size);
        }
    }

    #[test]
    fn test_chunk_stream_matches_chunk_slice() {
        let config = CdcConfig::new(256, 1024, 4096);
        let data = generate_test_data(32 * KB);

        let slice_chunks: Vec<_> = config.chunk_slice(&data).collect();

        let reader = Cursor::new(&data);
        let stream_chunks: Vec<_> = config
            .chunk_stream(reader)
            .collect::<Result<Vec<_>, _>>()
            .expect("Reading from Cursor should not fail");

        assert_eq!(
            slice_chunks.len(),
            stream_chunks.len(),
            "Chunk count must match between slice and stream"
        );

        for ((slice_chunk, slice_feat), (stream_chunk, stream_feat)) in
            slice_chunks.iter().zip(stream_chunks.iter())
        {
            let offset = slice_chunk.offset;
            let length = slice_chunk.length;

            assert_eq!(offset, stream_chunk.offset as usize, "Offset mismatch");
            assert_eq!(length, stream_chunk.length as usize, "Length mismatch");
            assert_eq!(slice_chunk.hash, stream_chunk.hash, "Hash mismatch");
            assert_eq!(slice_feat, stream_feat, "Extracted features mismatch");
            assert_eq!(
                &data[offset..offset + length],
                stream_chunk.data.as_slice(),
                "Stream chunk payload does not match original slice bytes"
            );
        }
    }
}