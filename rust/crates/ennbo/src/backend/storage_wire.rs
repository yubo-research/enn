//! Wire names for [`EnnStorage`](super::EnnStorage).

use super::EnnStorage;

impl EnnStorage {
    /// Wire name shared with Python `ENNStorage`.
    pub fn as_wire(self) -> &'static str {
        match self {
            Self::Disk => "DISK",
            Self::InMemory => "MEMORY",
        }
    }

    /// Parse a wire name. Returns `None` when `name` is not a known storage kind.
    pub fn from_wire(name: &str) -> Option<Self> {
        match name {
            "DISK" => Some(Self::Disk),
            "MEMORY" => Some(Self::InMemory),
            _ => None,
        }
    }
}
