//! Storage abstractions for Marline: key-value database traits, chunk containers
//! and an in-memory database implementation.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Formatter;
use std::io;

/// Key type usable as a chunk hash in storage and indexes.
pub trait ChunkHash: Clone + Eq + std::hash::Hash + Send + Sync + 'static {}
impl<T: Clone + Eq + std::hash::Hash + Send + Sync + 'static> ChunkHash for T {}

/// Key-value storage for chunks and their metadata.
pub trait Database<K, V> {
    /// Inserts a key-value pair into the database.
    ///
    /// # Idempotency and Overwrite Behavior
    /// In content-addressable storage (CAS), inserting with an identical key is idempotent
    /// when the underlying payload is identical. If the key already exists, the entry
    /// is overwritten without error.
    fn insert(&mut self, key: K, value: V) -> io::Result<()>;
    /// Returns a copy of the value stored under `key`.
    ///
    /// # Errors
    ///
    /// Returns an error of kind [`io::ErrorKind::NotFound`] if `key` is absent.
    fn get(&self, key: &K) -> io::Result<V>;
    /// Whether a value is stored under `key`.
    fn contains(&self, key: &K) -> bool;
    /// Removes `key` and returns its value, or `None` if the key was absent.
    fn remove(&mut self, key: &K) -> io::Result<Option<V>>;
    /// Number of stored entries.
    fn len(&self) -> usize;
    /// Whether the database holds no entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// [`Database`] that can enumerate and clear its entries.
pub trait IterableDatabase<K, V>: Database<K, V> {
    /// Iterates over all entries in unspecified order.
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a K, &'a V)> + 'a>;
    /// Iterates over all entries with mutable access to values, in unspecified order.
    fn iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = (&'a K, &'a mut V)> + 'a>;
    /// Removes all entries.
    fn clear(&mut self) -> io::Result<()>;
}

/// Content of a stored chunk: either raw bytes or references to the keys that replaced it.
#[derive(Clone)]
pub enum Data<K> {
    /// Raw chunk bytes.
    Chunk(Vec<u8>),
    /// Keys of the entries that now hold this chunk's content (e.g. after delta encoding).
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

/// Owner of a chunk's [`Data`], stored as the value in a chunk database.
#[derive(Clone, Debug, Default)]
pub struct DataContainer<K>(Data<K>);

impl<K> DataContainer<K> {
    /// Wraps `data`.
    pub fn new(data: Data<K>) -> Self {
        Self(data)
    }

    /// Replaces the content with references to `keys`.
    pub fn make_target(&mut self, keys: Vec<K>) {
        self.0 = Data::TargetChunk(keys);
    }

    /// Returns the stored content.
    pub fn data(&self) -> &Data<K> {
        &self.0
    }

    /// Returns the stored content for in-place modification.
    pub fn data_mut(&mut self) -> &mut Data<K> {
        &mut self.0
    }

    /// Returns the raw bytes, or `None` if the content is a target reference.
    pub fn as_chunk(&self) -> Option<&[u8]> {
        match &self.0 {
            Data::Chunk(chunk) => Some(chunk.as_slice()),
            Data::TargetChunk(_) => None,
        }
    }

    /// Returns the raw bytes for modification, or `None` if the content is a target reference.
    pub fn as_chunk_mut(&mut self) -> Option<&mut Vec<u8>> {
        match &mut self.0 {
            Data::Chunk(chunk) => Some(chunk),
            Data::TargetChunk(_) => None,
        }
    }

    /// Returns the target keys, or `None` if the content is raw bytes.
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

/// [`IterableDatabase`] backed by a `HashMap`.
#[derive(Debug, Clone, Default)]
pub struct InMemoryDatabase<K: std::hash::Hash + Eq, V> {
    inner: HashMap<K, V>,
}

impl<K: std::hash::Hash + Eq, V> InMemoryDatabase<K, V> {
    /// Creates an empty database.
    pub fn new() -> Self {
        Self { inner: HashMap::new() }
    }

    /// Returns a reference to the value under `key` without cloning.
    pub fn get_ref(&self, key: &K) -> Option<&V> {
        self.inner.get(key)
    }

    /// Returns a mutable reference to the value under `key`.
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
    /// Creates an empty store with zeroed metrics.
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

    /// Number of unique base chunks stored.
    pub fn unique_chunks_count(&self) -> usize {
        self.unique_chunks_count
    }

    /// Number of unique deltas stored.
    pub fn delta_chunks_count(&self) -> usize {
        self.delta_chunks_count
    }

    /// Stores a unique base chunk and accounts for its size.
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

    /// Returns the bytes of a stored base chunk without cloning.
    pub fn get_ref(&self, hash: &H) -> Option<&[u8]> {
        self.base_chunks.get(hash).map(|v| v.as_slice())
    }

    /// Whether a base chunk with `hash` is stored (deltas are not counted).
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
        assert!(db.get(&key1).unwrap() == val1, "get() must return the inserted value");
        assert!(db.get_ref(&key1) == Some(&val1), "get_ref() must return the inserted value");

        // Modify in-place via get_mut()
        if let Some(val_mut) = db.get_mut(&key1) {
            val_mut.push(40);
        }
        assert!(
            db.get_ref(&key1) == Some(&vec![10, 20, 30, 40]),
            "get_mut() changes must be visible"
        );

        // Missing key checks
        let missing_key = make_hash(99);
        assert!(!db.contains(&missing_key));
        assert!(matches!(
            db.get(&missing_key),
            Err(e) if e.kind() == io::ErrorKind::NotFound
        ));
        assert!(db.get_ref(&missing_key).is_none());
        assert!(db.get_mut(&missing_key).is_none());

        // Remove entry
        let removed = db.remove(&key1).unwrap();
        assert!(removed == Some(vec![10, 20, 30, 40]), "remove() must return the stored value");
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
        assert!(!db.contains(&key1));

        // Remove non-existent key returns Ok(None)
        assert!(db.remove(&key1).unwrap().is_none());
    }

    #[test]
    fn test_in_memory_db_insert_overwrites_existing_key() {
        let mut db: InMemoryDatabase<[u8; 32], Vec<u8>> = InMemoryDatabase::new();
        let key = make_hash(1);

        db.insert(key, vec![1, 2, 3]).unwrap();
        db.insert(key, vec![4, 5]).unwrap();

        assert_eq!(db.len(), 1, "Re-inserting a key must not add an entry");
        assert!(db.get(&key).unwrap() == vec![4, 5], "Re-inserting a key must overwrite the value");
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
        assert!(
            db.get_ref(&hash1) == Some(payload1.as_slice()),
            "Stored payload must match the inserted one"
        );

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
        assert!(
            db.get_ref(&hash1) == Some(payload1.as_slice()),
            "Duplicate insert must not replace the stored payload"
        );

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
        assert!(db.get_ref(&delta_hash).is_none());

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
        assert!(
            container.as_chunk() == Some(chunk_data.as_slice()),
            "as_chunk() must return the payload"
        );
        assert!(container.as_target().is_none());

        // Test data() accessor
        assert!(matches!(container.data(), Data::Chunk(c) if c == &chunk_data));

        // Mutate chunk in-place via as_chunk_mut()
        if let Some(chunk_mut) = container.as_chunk_mut() {
            chunk_mut.push(6);
        }
        assert!(
            container.as_chunk() == Some(&[1, 2, 3, 4, 5, 6][..]),
            "as_chunk_mut() changes must be visible"
        );

        // Mutate chunk in-place via data_mut()
        if let Data::Chunk(c) = container.data_mut() {
            c.push(7);
        }
        assert!(
            container.as_chunk() == Some(&[1, 2, 3, 4, 5, 6, 7][..]),
            "data_mut() changes must be visible"
        );

        // Convert into TargetChunk
        let keys = vec![make_hash(10), make_hash(20)];
        container.make_target(keys.clone());

        // Verify target accessors
        assert!(container.as_chunk().is_none());
        assert!(container.as_chunk_mut().is_none());
        assert!(container.as_target() == Some(keys.as_slice()), "as_target() must return the keys");
        assert!(matches!(container.data(), Data::TargetChunk(k) if k == &keys));
    }
}
