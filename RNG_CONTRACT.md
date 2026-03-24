# RNG Contract for enn_hash Functions

## Overview

The `enn_hash` module provides deterministic hash-based random number generation
critical for reproducibility in ENN surrogate models.

## Two Implementations

### 1. `normal_hash_batch_multi_seed` (Reference)

**Algorithm:**
- Uses `numpy.random.Philox` counter-based RNG
- Applies `scipy.special.ndtri` (inverse normal CDF)
- Slower but statistically well-behaved

**Seed Combination:**
```python
combined_seed = (function_seed * 1_000_003 + data_index) * 1_000_003 + metric_index
```

**Output Range:**
- Clipped to `[1e-10, 1 - 1e-10]` before inverse CDF

### 2. `normal_hash_batch_multi_seed_fast` (Production)

**Algorithm:**
- Uses **SplitMix64** for fast bit-mixing
- Applies **Box-Muller transform** for normal distribution
- Significantly faster, maintains deterministic output

## SplitMix64 Algorithm Contract

The Rust implementation must match these exact operations:

```rust
fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9E3779B97F4A7C15);  // Golden ratio
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
```

**Constants (must match exactly):**
| Constant | Value | Purpose |
|----------|-------|---------|
| Golden ratio | `0x9E3779B97F4A7C15` | Increment for sequence |
| Multiplier 1 | `0xBF58476D1CE4E5B9` | First mixing step |
| Multiplier 2 | `0x94D049BB133111EB` | Second mixing step |
| XOR offset | `0xD2B74407B1CE6E93` | Second stream differentiation |

## Box-Muller Transform Contract

```rust
fn box_muller(u1: f64, u2: f64) -> f64 {
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}
```

**Conversion from u64 to f64:**
- Take high 53 bits: `u1 = (r1 >> 11) as f64 / 2^53`
- Range: `[0, 1)` after division
- Clip to `[1e-12, 1 - 1e-12]` to avoid log(0)

## Seed Structure Contract

**Input shapes:**
- `function_seeds`: `(num_seeds,)` array of int64
- `data_indices`: `(...,)` arbitrary shape of int
- `num_metrics`: int, must be > 0

**Output shape:** `(num_seeds, *data_indices.shape, num_metrics)`

**Seed combination for fast version:**
```python
base = (function_seed * 1_000_003 + unique_data_index) * 1_000_003
combined1 = base + metric_index
combined2 = combined1 ^ 0xD2B74407B1CE6E93  # Second stream
r1 = splitmix64(combined1)
r2 = splitmix64(combined2)
```

## Parity Requirements

1. **Bitwise equality** for same inputs on same platform
2. **Deterministic**: Same seeds always produce same outputs
3. **Seed sensitivity**: Different seeds produce different outputs
4. **Statistical properties**: Output should pass basic normality tests

## Testing Strategy

1. Fixed seed fixtures with expected outputs
2. Cross-implementation parity (Philox vs SplitMix64)
3. Statistical distribution tests (mean ≈ 0, std ≈ 1)
4. Edge case testing (empty inputs, single elements)
