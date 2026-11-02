use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Formatter;
use std::io;

pub trait ChunkHash: Clone + Eq + std::hash::Hash + Send + Sync + 'static {}
impl<T: Clone + Eq + std::hash::Hash + Send + Sync + 'static> ChunkHash for T {}

pub trait Database<K, V> {
    /// Inserts a key-value pair into the database.
    ///
    /// # Idempotency and Overwrite Behavior
    /// In content-addressable storage (CAS), inserting with an identical key is idempotent
    /// when the underlying payload is identical. If the key already exists, the entry
    /// is overwritten without error.
    fn insert(&mut self, key: K, value: V) -> io::Result<()>;
    fn get(&self, key: &K) -> io::Result<V>;
    fn contains(&self, key: &K) -> bool;
    fn remove(&mut self, key: &K) -> io::Result<Option<V>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait IterableDatabase<K, V>: Database<K, V> {
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a K, &'a V)> + 'a>;
    fn iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = (&'a K, &'a mut V)> + 'a>;
    fn clear(&mut self) -> io::Result<()>;
}

#[derive(Clone)]
pub enum Data<K> {
    Chunk(Vec<u8>),
    TargetChunk(Vec<K>),
}

impl<K> std::fmt::Debug for Data<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Data::Chunk(chunk) => write!(f, "Chunk with len {}", chunk.len()),
            Data::TargetChunk(keys) => write!(f, "TargetChunk with {} keys", keys.len()),
        }
    }
}

impl<K> Default for Data<K> {
    fn default() -> Self {
        Self::Chunk(Vec::new())
    }
}

#[derive(Clone, Debug, Default)]
pub struct DataContainer<K>(Data<K>);

impl<K> DataContainer<K> {
    pub fn new(data: Data<K>) -> Self {
        Self(data)
    }

    pub fn make_target(&mut self, keys: Vec<K>) {
        self.0 = Data::TargetChunk(keys);
    }

    pub fn data(&self) -> &Data<K> {
        &self.0
    }

    pub fn data_mut(&mut self) -> &mut Data<K> {
        &mut self.0
    }

    pub fn as_chunk(&self) -> Option<&[u8]> {
        match &self.0 {
            Data::Chunk(chunk) => Some(chunk.as_slice()),
            Data::TargetChunk(_) => None,
        }
    }

    pub fn as_chunk_mut(&mut self) -> Option<&mut Vec<u8>> {
        match &mut self.0 {
            Data::Chunk(chunk) => Some(chunk),
            Data::TargetChunk(_) => None,
        }
    }

    pub fn as_target(&self) -> Option<&[K]> {
        match &self.0 {
            Data::Chunk(_) => None,
            Data::TargetChunk(keys) => Some(keys.as_slice()),
        }
    }
}

impl<K> From<Vec<u8>> for DataContainer<K> {
    fn from(value: Vec<u8>) -> Self {
        Self(Data::Chunk(value))
    }
}

#[derive(Debug, Clone, Default)]
pub struct InMemoryDatabase<K: std::hash::Hash + Eq, V> {
    inner: HashMap<K, V>,
}

impl<K: std::hash::Hash + Eq, V> InMemoryDatabase<K, V> {
    pub fn new() -> Self {
        Self { inner: HashMap::new() }
    }

    pub fn get_ref(&self, key: &K) -> Option<&V> {
        self.inner.get(key)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.inner.get_mut(key)
    }
}

impl<K: ChunkHash, V: Clone> Database<K, V> for InMemoryDatabase<K, V> {
    fn insert(&mut self, key: K, value: V) -> io::Result<()> {
        self.inner.insert(key, value);
        Ok(())
    }

    fn get(&self, key: &K) -> io::Result<V> {
        self.inner
            .get(key)
            .cloned()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Key not found"))
    }

    fn contains(&self, key: &K) -> bool {
        self.inner.contains_key(key)
    }

    fn remove(&mut self, key: &K) -> io::Result<Option<V>> {
        Ok(self.inner.remove(key))
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K: ChunkHash, V: Clone> IterableDatabase<K, V> for InMemoryDatabase<K, V> {
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a K, &'a V)> + 'a> {
        Box::new(self.inner.iter())
    }

    fn iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = (&'a K, &'a mut V)> + 'a> {
        Box::new(self.inner.iter_mut())
    }

    fn clear(&mut self) -> io::Result<()> {
        self.inner.clear();
        Ok(())
    }
}

/// Metrics-oriented chunk store.
///
/// Keeps raw payloads of unique base chunks and a single delta index
/// (delta hash → root base hash). Delta payloads are accounted for in metrics
/// but not kept in RAM.
#[derive(Debug, Clone)]
pub struct MetricsDatabase<H: ChunkHash = [u8; 32]> {
    base_bytes: usize,
    delta_bytes: usize,
    unique_chunks_count: usize,
    delta_chunks_count: usize,
    /// Stores unique base chunks for subsequent delta encoding references.
    base_chunks: HashMap<H, Vec<u8>>,
    /// Delta index: delta chunk hash → hash of the base chunk it was encoded against.
    /// Also guarantees metric idempotency for repeated delta inserts.
    delta_bases: HashMap<H, H>,
}

impl<H: ChunkHash> Default for MetricsDatabase<H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: ChunkHash> MetricsDatabase<H> {
    pub fn new() -> Self {
        Self {
            base_bytes: 0,
            delta_bytes: 0,
            unique_chunks_count: 0,
            delta_chunks_count: 0,
            base_chunks: HashMap::new(),
            delta_bases: HashMap::new(),
        }
    }

    /// Raw uncompressed bytes stored for unique base chunks.
    pub fn base_bytes(&self) -> usize {
        self.base_bytes
    }

    /// Compressed bytes accounted for delta chunks.
    pub fn delta_bytes(&self) -> usize {
        self.delta_bytes
    }

    /// Total bytes that would be stored:
    /// raw bytes for unique base chunks + compressed bytes for delta chunks.
    pub fn total_saved_bytes(&self) -> usize {
        self.base_bytes + self.delta_bytes
    }

    pub fn unique_chunks_count(&self) -> usize {
        self.unique_chunks_count
    }

    pub fn delta_chunks_count(&self) -> usize {
        self.delta_chunks_count
    }

    /// Preserving the unique base chunk (storing the body + monitoring its size).
    ///
    /// Idempotent: inserting an already existing hash (as base or delta) does not increase metrics.
    pub fn insert(&mut self, hash: H, data: &[u8]) {
        if self.delta_bases.contains_key(&hash) {
            return;
        }
        if let Entry::Vacant(entry) = self.base_chunks.entry(hash) {
            self.base_bytes += data.len();
            self.unique_chunks_count += 1;
            entry.insert(data.to_vec());
        }
    }

    /// Records a delta chunk encoded against `base` for metrics accounting.
    ///
    /// Returns `false` and records nothing if `base` is not a stored base chunk
    /// or `hash` is already known (as base or delta).
    ///
    /// # Note
    /// This store is metrics-oriented and does NOT persist delta payload bytes in RAM.
    /// Since `base` must be a stored base chunk and base/delta hashes never overlap,
    /// self-loops and delta-to-delta links are rejected, keeping the index flat.
    #[must_use]
    pub fn insert_delta(&mut self, hash: H, base: H, delta: &[u8]) -> bool {
        if self.base_chunks.contains_key(&hash) || !self.base_chunks.contains_key(&base) {
            return false;
        }
        match self.delta_bases.entry(hash) {
            Entry::Vacant(entry) => {
                entry.insert(base);
                self.delta_bytes += delta.len();
                self.delta_chunks_count += 1;
                true
            }
            Entry::Occupied(_) => false,
        }
    }

    /// Read-only view of the delta index (delta hash → base hash).
    pub fn delta_bases(&self) -> &HashMap<H, H> {
        &self.delta_bases
    }

    /// Obtaining a reference to a base chunk without unnecessary cloning.
    pub fn get_ref(&self, hash: &H) -> Option<&[u8]> {
        self.base_chunks.get(hash).map(|v| v.as_slice())
    }

    pub fn contains(&self, hash: &H) -> bool {
        self.base_chunks.contains_key(hash) || self.delta_bases.contains_key(hash)
    }

    /// Number of stored base chunks (delta chunks are not counted).
    pub fn len(&self) -> usize {
        self.base_chunks.len()
    }

    /// Whether no base chunks are stored (delta chunks are not counted).
    pub fn is_empty(&self) -> bool {
        self.base_chunks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use marline_utils::KB;

    fn make_hash(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    // =========================================================================
    // Tests for InMemoryDatabase
    // =========================================================================

    #[test]
    fn test_in_memory_db_crud_and_direct_access() {
        let mut db: InMemoryDatabase<[u8; 32], Vec<u8>> = InMemoryDatabase::new();

        assert!(db.is_empty(), "Newly created database should be empty");
        assert_eq!(db.len(), 0);

        let key1 = make_hash(1);
        let val1 = vec![10, 20, 30];

        // Insert first entry
        assert!(db.insert(key1, val1.clone()).is_ok());
        assert!(!db.is_empty());
        assert_eq!(db.len(), 1);
        assert!(db.contains(&key1));

        // Verify get(), get_ref() and get_mut()
        assert_eq!(db.get(&key1).unwrap(), val1);
        assert_eq!(db.get_ref(&key1), Some(&val1));

        // Modify in-place via get_mut()
        if let Some(val_mut) = db.get_mut(&key1) {
            val_mut.push(40);
        }
        assert_eq!(db.get_ref(&key1), Some(&vec![10, 20, 30, 40]));

        // Missing key checks
        let missing_key = make_hash(99);
        assert!(!db.contains(&missing_key));
        assert!(matches!(
            db.get(&missing_key),
            Err(e) if e.kind() == io::ErrorKind::NotFound
        ));
        assert_eq!(db.get_ref(&missing_key), None);
        assert_eq!(db.get_mut(&missing_key), None);

        // Remove entry
        let removed = db.remove(&key1).unwrap();
        assert_eq!(removed, Some(vec![10, 20, 30, 40]));
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
        assert!(!db.contains(&key1));

        // Remove non-existent key returns Ok(None)
        assert_eq!(db.remove(&key1).unwrap(), None);
    }

    #[test]
    fn test_in_memory_db_iteration_and_mutation() {
        let mut db: InMemoryDatabase<[u8; 32], u32> = InMemoryDatabase::default();

        let entries = [(make_hash(1), 100), (make_hash(2), 200), (make_hash(3), 300)];

        for (k, v) in entries {
            db.insert(k, v).unwrap();
        }

        assert_eq!(db.len(), 3);

        // Immutable iteration
        let collected: HashMap<[u8; 32], u32> = db.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(collected.len(), 3);
        for (k, v) in entries {
            assert_eq!(collected.get(&k), Some(&v));
        }

        // Mutable iteration: add 10 to every value
        for (_k, v) in db.iter_mut() {
            *v += 10;
        }

        for (k, v) in entries {
            assert_eq!(db.get(&k).unwrap(), v + 10);
        }

        // Clear database
        assert!(db.clear().is_ok());
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
        assert_eq!(db.iter().count(), 0);
    }

    // =========================================================================
    // Tests for MetricsDatabase
    // =========================================================================

    #[test]
    fn test_metrics_db_insert_idempotency_and_state() {
        let mut db: MetricsDatabase<[u8; 32]> = MetricsDatabase::new();

        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
        assert_eq!(db.total_saved_bytes(), 0);
        assert_eq!(db.unique_chunks_count(), 0);
        assert_eq!(db.delta_chunks_count(), 0);

        let hash1 = make_hash(1);
        let payload1 = vec![0xAB; 2 * KB];

        // First insertion
        db.insert(hash1, &payload1);
        assert_eq!(db.len(), 1);
        assert!(!db.is_empty());
        assert_eq!(db.unique_chunks_count(), 1);
        assert_eq!(db.total_saved_bytes(), 2 * KB);
        assert_eq!(db.get_ref(&hash1), Some(payload1.as_slice()));

        // Repeated insertion with the same hash (same payload) - MUST BE IDEMPOTENT
        db.insert(hash1, &payload1);
        assert_eq!(db.len(), 1, "len() must not change on duplicate insert");
        assert_eq!(
            db.unique_chunks_count(),
            1,
            "unique_chunks_count must not increase on duplicate insert"
        );
        assert_eq!(
            db.total_saved_bytes(),
            2 * KB,
            "total_saved_bytes must not increase on duplicate insert"
        );

        // Repeated insertion with the same hash but different payload - still idempotent
        let different_payload = vec![0xCD; 4 * KB];
        db.insert(hash1, &different_payload);
        assert_eq!(db.unique_chunks_count(), 1);
        assert_eq!(db.total_saved_bytes(), 2 * KB);
        assert_eq!(db.get_ref(&hash1), Some(payload1.as_slice()));

        // Insert distinct second chunk
        let hash2 = make_hash(2);
        let payload2 = vec![0xEF; KB];
        db.insert(hash2, &payload2);
        assert_eq!(db.len(), 2);
        assert_eq!(db.unique_chunks_count(), 2);
        assert_eq!(db.total_saved_bytes(), 3 * KB);

        // Insert delta chunk
        let delta_hash = make_hash(3);
        let delta_payload = vec![0x11; 512];
        assert!(db.insert_delta(delta_hash, hash1, &delta_payload));
        assert_eq!(db.delta_chunks_count(), 1);
        assert_eq!(db.delta_bases().get(&delta_hash), Some(&hash1));
        assert_eq!(db.base_bytes(), 3 * KB);
        assert_eq!(db.delta_bytes(), 512);
        assert_eq!(db.total_saved_bytes(), 3 * KB + 512);
        assert_eq!(db.len(), 2, "Delta chunks are not stored in base_chunks map");
        assert!(db.contains(&delta_hash), "contains checks both base chunks and seen deltas");
        assert_eq!(db.get_ref(&delta_hash), None);

        // Repeated insert_delta with the same hash - MUST BE IDEMPOTENT
        assert!(!db.insert_delta(delta_hash, hash1, &delta_payload));
        assert_eq!(
            db.delta_chunks_count(),
            1,
            "delta_chunks_count must not increase on duplicate delta insert"
        );
        assert_eq!(
            db.total_saved_bytes(),
            3 * KB + 512,
            "total_saved_bytes must not increase on duplicate delta insert"
        );

        // Repeated insert_delta with same hash but different payload - still idempotent
        let diff_delta = vec![0x22; 1024];
        assert!(!db.insert_delta(delta_hash, hash2, &diff_delta));
        assert_eq!(
            db.delta_bases().get(&delta_hash),
            Some(&hash1),
            "repeated insert_delta with a different base must not overwrite the original base"
        );
        assert_eq!(
            db.delta_chunks_count(),
            1,
            "delta_chunks_count must not increase on duplicate delta insert with different payload"
        );
        assert_eq!(
            db.total_saved_bytes(),
            3 * KB + 512,
            "total_saved_bytes must not increase on duplicate delta insert with different payload"
        );

        // Cross-idempotency: inserting already-seen delta as a base chunk MUST BE A NO-OP
        db.insert(delta_hash, &vec![0x33; 2048]);
        assert_eq!(db.len(), 2, "delta hash must not be inserted into base_chunks");
        assert_eq!(db.unique_chunks_count(), 2, "unique_chunks_count must not increase");
        assert_eq!(db.total_saved_bytes(), 3 * KB + 512, "total_saved_bytes must not increase");

        // Cross-idempotency: inserting already-existing base as a delta MUST BE A NO-OP
        assert!(!db.insert_delta(hash1, hash2, &vec![0x44; 256]));
        assert!(
            !db.delta_bases().contains_key(&hash1),
            "existing base must not enter the delta index"
        );
        assert_eq!(
            db.delta_chunks_count(),
            1,
            "delta_chunks_count must not increase for existing base chunk"
        );
        assert_eq!(
            db.total_saved_bytes(),
            3 * KB + 512,
            "total_saved_bytes must not increase for existing base chunk"
        );
    }

    #[test]
    fn test_metrics_db_insert_delta_rejects_invalid_base() {
        let mut db: MetricsDatabase<[u8; 32]> = MetricsDatabase::new();
        let base = make_hash(1);
        let delta = make_hash(2);
        let dangling = make_hash(3);
        let other = make_hash(4);
        let payload = [0x5Au8; 300];

        db.insert(base, &[0xA5; KB]);
        let snapshot = |db: &MetricsDatabase<[u8; 32]>| {
            (db.delta_bytes(), db.delta_chunks_count(), db.delta_bases().len())
        };

        // Dangling base: not stored at all.
        let before = snapshot(&db);
        assert!(!db.insert_delta(other, dangling, &payload), "dangling base must be rejected");
        assert_eq!(snapshot(&db), before, "rejected insert must not change metrics");
        assert!(!db.contains(&other));

        // Base that is itself a delta: delta-to-delta links are rejected.
        assert!(db.insert_delta(delta, base, &payload));
        let before = snapshot(&db);
        assert!(!db.insert_delta(other, delta, &payload), "delta as base must be rejected");
        assert_eq!(snapshot(&db), before);
        assert!(!db.contains(&other));

        // Self-loop: hash == base, both for unknown and stored hashes.
        assert!(!db.insert_delta(other, other, &payload), "self-loop must be rejected");
        assert!(!db.insert_delta(base, base, &payload), "self-loop on base must be rejected");
        assert!(!db.insert_delta(delta, delta, &payload), "self-loop on delta must be rejected");
        assert_eq!(snapshot(&db), before);
        assert_eq!(db.delta_bases().get(&delta), Some(&base));
    }

    // =========================================================================
    // Tests for DataContainer
    // =========================================================================

    #[test]
    fn test_data_container_safe_accessors() {
        let chunk_data = vec![1, 2, 3, 4, 5];
        let mut container: DataContainer<[u8; 32]> = DataContainer::from(chunk_data.clone());

        // Test chunk accessors
        assert_eq!(container.as_chunk(), Some(chunk_data.as_slice()));
        assert_eq!(container.as_target(), None);

        // Test data() accessor
        assert!(matches!(container.data(), Data::Chunk(c) if c == &chunk_data));

        // Mutate chunk in-place via as_chunk_mut()
        if let Some(chunk_mut) = container.as_chunk_mut() {
            chunk_mut.push(6);
        }
        assert_eq!(container.as_chunk(), Some(&[1, 2, 3, 4, 5, 6][..]));

        // Mutate chunk in-place via data_mut()
        if let Data::Chunk(c) = container.data_mut() {
            c.push(7);
        }
        assert_eq!(container.as_chunk(), Some(&[1, 2, 3, 4, 5, 6, 7][..]));

        // Convert into TargetChunk
        let keys = vec![make_hash(10), make_hash(20)];
        container.make_target(keys.clone());

        // Verify target accessors
        assert_eq!(container.as_chunk(), None);
        assert_eq!(container.as_chunk_mut(), None);
        assert_eq!(container.as_target(), Some(keys.as_slice()));
        assert!(matches!(container.data(), Data::TargetChunk(k) if k == &keys));
    }
}
