//! NumPy `SeedSequence` (pool size 4) for integer entropy.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

fn hashmix(mut value: u32, hash_const: &mut u32) -> u32 {
    value ^= *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let mut result = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^= result >> XSHIFT;
    result
}

fn entropy_words(entropy: u64) -> Vec<u32> {
    if entropy == 0 {
        return vec![0];
    }
    let mut words = Vec::new();
    let mut n = entropy;
    while n > 0 {
        words.push(n as u32);
        n >>= 32;
    }
    words
}

fn mix_entropy(pool: &mut [u32], entropy: &[u32]) {
    let mut hash_const = INIT_A;
    for (i, slot) in pool.iter_mut().enumerate() {
        let word = if i < entropy.len() { entropy[i] } else { 0 };
        *slot = hashmix(word, &mut hash_const);
    }
    let n = pool.len();
    for i_src in 0..n {
        for i_dst in 0..n {
            if i_src != i_dst {
                let mixed = hashmix(pool[i_src], &mut hash_const);
                pool[i_dst] = mix(pool[i_dst], mixed);
            }
        }
    }
    for &word in entropy.iter().skip(n) {
        for slot in pool.iter_mut() {
            let mixed = hashmix(word, &mut hash_const);
            *slot = mix(*slot, mixed);
        }
    }
}

/// `SeedSequence(entropy).generate_state(n_u64, dtype=uint64)`.
pub fn seed_sequence_u64(entropy: u64, n_u64: usize) -> Vec<u64> {
    let mut pool = [0u32; 4];
    let words = entropy_words(entropy);
    mix_entropy(&mut pool, &words);
    let n_u32 = n_u64 * 2;
    let mut state = vec![0u32; n_u32];
    let mut hash_const = INIT_B;
    for (i, slot) in state.iter_mut().enumerate() {
        let mut data = pool[i % pool.len()] ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        data = data.wrapping_mul(hash_const);
        *slot = data ^ (data >> XSHIFT);
    }
    state
        .chunks(2)
        .map(|c| u64::from(c[0]) | (u64::from(c[1]) << 32))
        .collect()
}
