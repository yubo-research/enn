
use super::EnnStorage;

impl EnnStorage {
    pub fn as_wire(self) -> &'static str {
        match self {
            Self::Disk => "DISK",
            Self::InMemory => "MEMORY",
        }
    }

    pub fn from_wire(name: &str) -> Option<Self> {
        match name {
            "DISK" => Some(Self::Disk),
            "MEMORY" => Some(Self::InMemory),
            _ => None,
        }
    }
}
