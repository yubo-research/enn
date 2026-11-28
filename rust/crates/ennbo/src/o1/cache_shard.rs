use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use super::cache::{CacheKey, Cached};
use super::lsh::mix64;

#[derive(Default)]
struct MixHasher(u64);

impl Hasher for MixHasher {
    fn finish(&self) -> u64 {
        mix64(self.0)
    }

    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(*b);
        }
    }

    fn write_u8(&mut self, v: u8) {
        self.write_u64(u64::from(v));
    }

    fn write_u32(&mut self, v: u32) {
        self.write_u64(u64::from(v));
    }

    fn write_u64(&mut self, v: u64) {
        self.0 = mix64(self.0 ^ v);
    }

    fn write_isize(&mut self, v: isize) {
        self.write_u64(v as u64);
    }
}

const NONE: u32 = u32::MAX;

struct Slot {
    key: CacheKey,
    value: Option<Cached>,
    bytes: usize,
    prev: u32,
    next: u32,
}

pub(crate) struct Shard {
    map: HashMap<CacheKey, u32, BuildHasherDefault<MixHasher>>,
    slots: Vec<Slot>,
    free: Vec<u32>,
    head: u32,
    tail: u32,
    pub(crate) bytes: [usize; 2],
    pub(crate) entries: [usize; 2],
}

impl Default for Shard {
    fn default() -> Self {
        Self {
            map: HashMap::default(),
            slots: Vec::new(),
            free: Vec::new(),
            head: NONE,
            tail: NONE,
            bytes: [0; 2],
            entries: [0; 2],
        }
    }
}

impl Shard {
    fn unlink(&mut self, i: u32) {
        let (prev, next) = (self.slots[i as usize].prev, self.slots[i as usize].next);
        match prev {
            NONE => self.head = next,
            p => self.slots[p as usize].next = next,
        }
        match next {
            NONE => self.tail = prev,
            n => self.slots[n as usize].prev = prev,
        }
    }

    fn push_front(&mut self, i: u32) {
        self.slots[i as usize].prev = NONE;
        self.slots[i as usize].next = self.head;
        if self.head != NONE {
            self.slots[self.head as usize].prev = i;
        }
        self.head = i;
        if self.tail == NONE {
            self.tail = i;
        }
    }

    pub(crate) fn touch(&mut self, key: &CacheKey) -> Option<Cached> {
        let i = *self.map.get(key)?;
        if self.head != i {
            self.unlink(i);
            self.push_front(i);
        }
        self.slots[i as usize].value.clone()
    }

    fn drop_slot(&mut self, i: u32) -> CacheKey {
        self.unlink(i);
        let slot = &mut self.slots[i as usize];
        slot.value = None;
        let key = slot.key;
        self.bytes[key.class()] -= slot.bytes;
        self.entries[key.class()] -= 1;
        self.free.push(i);
        self.map.remove(&key);
        key
    }

    pub(crate) fn remove(&mut self, key: &CacheKey) -> bool {
        match self.map.get(key) {
            Some(&i) => {
                self.drop_slot(i);
                true
            }
            None => false,
        }
    }

    pub(crate) fn put(
        &mut self,
        key: CacheKey,
        value: Cached,
        budget: usize,
        evictions: &[AtomicU64; 2],
    ) {
        self.remove(&key);
        let bytes = value.bytes();
        if bytes > budget {
            return;
        }
        while self.bytes[0] + self.bytes[1] + bytes > budget && self.tail != NONE {
            let old = self.drop_slot(self.tail);
            evictions[old.class()].fetch_add(1, Ordering::Relaxed);
        }
        let slot = Slot {
            key,
            value: Some(value),
            bytes,
            prev: NONE,
            next: NONE,
        };
        let i = match self.free.pop() {
            Some(i) => {
                self.slots[i as usize] = slot;
                i
            }
            None => {
                self.slots.push(slot);
                (self.slots.len() - 1) as u32
            }
        };
        self.push_front(i);
        self.map.insert(key, i);
        self.bytes[key.class()] += bytes;
        self.entries[key.class()] += 1;
    }
}
