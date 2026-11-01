use marline_storage::InMemoryDatabase;
use std::collections::HashMap;

/// Storage entry holding raw chunk payload and optional manifest metadata.
#[derive(Clone, Debug, Default)]
pub struct Entry {
    manifest: Option<HashMap<u64, u32>>,
    data: Vec<u8>,
}

impl Entry {
    pub fn new(manifest: Option<HashMap<u64, u32>>, data: Vec<u8>) -> Self {
        Self { manifest, data }
    }
    pub fn get_manifest(&self) -> &Option<HashMap<u64, u32>> {
        &self.manifest
    }
    pub fn get_manifest_mut(&mut self) -> &mut Option<HashMap<u64, u32>> {
        &mut self.manifest
    }
    pub fn get_data(&self) -> &Vec<u8> {
        &self.data
    }
}

impl From<Vec<u8>> for Entry {
    fn from(data: Vec<u8>) -> Self {
        Self::new(None, data)
    }
}

/// Type alias for the in-memory database backing chunk storage with manifests.
pub type MockRocksDBMap = InMemoryDatabase<[u8; 32], Entry>;
