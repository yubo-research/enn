import math

_SPLITMIX64_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX64_MUL1 = 0xBF58476D1CE4E5B9
_SPLITMIX64_MUL2 = 0x94D049BB133111EB
_MASK_64 = (1 << 64) - 1


def get_sparse_jl_contribution(
    input_idx: int,
    value: float,
    num_dim_embed: int,
    s: int,
    embed_seed: int,
) -> list[tuple[int, float]]:
    inv_sqrt_s = 1.0 / math.sqrt(float(s))
    acc = (
        _SPLITMIX64_INCREMENT ^ (embed_seed & _MASK_64) ^ ((input_idx & _MASK_64) << 1)
    )
    chosen_rows: list[int] = []
    contributions: list[tuple[int, float]] = []

    while len(chosen_rows) < s:
        acc = (acc + _SPLITMIX64_INCREMENT) & _MASK_64
        z = acc
        z ^= z >> 30
        z = (z * _SPLITMIX64_MUL1) & _MASK_64
        z ^= z >> 27
        z = (z * _SPLITMIX64_MUL2) & _MASK_64
        z ^= z >> 31
        row = z % num_dim_embed

        if row in chosen_rows:
            continue
        chosen_rows.append(row)

        acc = (acc + _SPLITMIX64_INCREMENT) & _MASK_64
        z2 = acc
        z2 ^= z2 >> 30
        z2 = (z2 * _SPLITMIX64_MUL1) & _MASK_64
        z2 ^= z2 >> 27
        z2 = (z2 * _SPLITMIX64_MUL2) & _MASK_64
        z2 ^= z2 >> 31
        sign = 1.0 if (z2 & 1) == 1 else -1.0

        contributions.append((row, sign * value * inv_sqrt_s))

    return contributions
