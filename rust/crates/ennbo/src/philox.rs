//! NumPy `Generator(Philox(seed)).random()`.

use crate::numpy_seed::seed_sequence_u64;

const PHILOX_M0: u64 = 0xD2E7_470E_E14C_6C93;
const PHILOX_M1: u64 = 0xCA5A_8263_9512_1157;
const BUMP0: u64 = 0x9E37_79B9_7F4A_7C15;
const BUMP1: u64 = 0xBB67_AE85_84CA_A73B;

fn mulhilo(a: u64, b: u64) -> (u64, u64) {
    let product = u128::from(a).wrapping_mul(u128::from(b));
    (product as u64, (product >> 64) as u64)
}

fn round(ctr: [u64; 4], key: [u64; 2]) -> [u64; 4] {
    let (lo0, hi0) = mulhilo(PHILOX_M0, ctr[0]);
    let (lo1, hi1) = mulhilo(PHILOX_M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

fn philox10(mut ctr: [u64; 4], mut key: [u64; 2]) -> [u64; 4] {
    for _ in 0..9 {
        ctr = round(ctr, key);
        key[0] = key[0].wrapping_add(BUMP0);
        key[1] = key[1].wrapping_add(BUMP1);
    }
    round(ctr, key)
}

pub struct NumpyPhilox {
    key: [u64; 2],
    ctr: [u64; 4],
    buffer: [u64; 4],
    buffer_pos: usize,
}

impl NumpyPhilox {
    pub fn from_seed(seed: u64) -> Self {
        let words = seed_sequence_u64(seed, 2);
        Self {
            key: [words[0], words[1]],
            ctr: [0; 4],
            buffer: [0; 4],
            buffer_pos: 4,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        if self.buffer_pos < 4 {
            let out = self.buffer[self.buffer_pos];
            self.buffer_pos += 1;
            return out;
        }
        self.ctr[0] = self.ctr[0].wrapping_add(1);
        if self.ctr[0] == 0 {
            self.ctr[1] = self.ctr[1].wrapping_add(1);
            if self.ctr[1] == 0 {
                self.ctr[2] = self.ctr[2].wrapping_add(1);
                if self.ctr[2] == 0 {
                    self.ctr[3] = self.ctr[3].wrapping_add(1);
                }
            }
        }
        self.buffer = philox10(self.ctr, self.key);
        self.buffer_pos = 1;
        self.buffer[0]
    }

    /// NumPy `Generator.random()` in `[0, 1)`.
    pub fn random(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }
}
