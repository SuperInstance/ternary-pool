# ternary-pool

**Ternary pooling operations for {-1, 0, +1} matrices — the missing piece in ternary neural network layers.**

[![crate](https://img.shields.io/badge/crates.io-ternary--pool-orange)](https://crates.io)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Overview

`ternary-pool` provides pooling (downsampling) operations for ternary-valued matrices where every element is in {-1, 0, +1}. Pooling reduces spatial dimensions while preserving important features — a critical operation in convolutional neural networks and signal processing.

The crate implements **eight pooling strategies**, each with distinct semantics:

| Strategy | Description | Selects |
|----------|-------------|---------|
| **Max Pool** | Maximum value in window | Strongest positive signal |
| **Min Pool** | Minimum value in window | Strongest negative signal |
| **Majority Pool** | Most common trit (with tie-breaking) | Consensus value |
| **Average Pool** | Mean rounded to nearest trit | Balanced summary |
| **Global Pool** | Reduces entire matrix to single trit | Full-matrix summary |
| **Adaptive Pool** | Pools to arbitrary target dimensions | Flexible downsampling |
| **Stochastic Pool** | Weighted random sampling | Probabilistic, regularizable |

## Why Ternary Pooling?

In **Quantized Neural Networks (QNNs)**, both weights and activations are constrained to {-1, 0, +1}. After ternary convolution (see: `ternary-conv`), you need pooling that operates entirely in the ternary domain — no dequantization, no floating-point fallback.

Benefits of ternary-aware pooling:

- **No precision loss** from quantize→pool→dequantize round-trips
- **Hardware-friendly** — comparisons and lookups instead of floating-point max/avg
- **Majority voting** — a natural operation for ternary data with no float analog
- **Stochastic pooling** — provides regularization in ternary networks without dropout
- **End-to-end ternary** — entire inference pipeline stays in {-1, 0, +1}

## Quick Start

```rust
use ternary_pool::*;

// Create a ternary matrix
let input = TernaryMatrix::from_vec(4, 4, vec![
     1, -1,  0,  1,
     0,  0,  1, -1,
    -1,  1,  0,  0,
     1,  0, -1,  1,
]);

// Max pooling with 2×2 kernel, stride 2
let maxed = max_pool(&input, 2, 2);
// → [[1, 1], [1, 1]]

// Min pooling
let mined = min_pool(&input, 2, 2);
// → [[-1, -1], [-1, -1]]

// Majority-vote pooling
let majority = majority_pool(&input, 2, 2);

// Average pooling (rounded to nearest trit)
let averaged = avg_pool(&input, 2, 2);

// Global pooling — reduces entire matrix to one trit
let g_avg = global_avg_pool(&input);   // i8
let g_max = global_max_pool(&input);   // 1
let g_min = global_min_pool(&input);   // -1
let g_maj = global_majority_pool(&input);

// Adaptive pooling to target size
let adapted = adaptive_avg_pool(&input, 2, 2);

// Stochastic pooling (deterministic with seed)
let stoch = stochastic_pool(&input, 2, 2, 42);
```

## Strategy Details

### Max Pool

For each window, selects the largest trit value: **1 > 0 > -1**. This is the ternary equivalent of ReLU-based max pooling. It captures the strongest positive activation in each region.

**Use when:** You want to detect the presence of positive features, similar to standard max pooling in CNNs.

### Min Pool

Selects the smallest trit value: **-1 < 0 < 1**. Useful for detecting strong negative signals — in ternary networks, -1 often represents "inhibition" or "negative evidence."

**Use when:** You need to track negative feature presence, or as a complement to max pooling.

### Majority Pool

Counts occurrences of each trit value in the window and selects the most common. Ties are broken by preference order: **0 > 1 > -1** (neutral first). This is a uniquely ternary operation with no direct floating-point analog — it's essentially a **vote** among the elements.

**Use when:** You want noise-resistant pooling that reflects the dominant signal. Especially effective in ternary networks where individual activations are noisy.

### Average Pool

Computes the integer mean of window values and rounds to the nearest trit:
- Mean < 0 → **-1**
- Mean = 0 → **0**
- Mean > 0 → **+1**

This is equivalent to computing the sign of the sum.

**Use when:** You want a balanced representation of all values in the window. Common in global average pooling for classification heads.

### Global Pool

Reduces the entire matrix to a single trit value. Available in four variants:
- `global_avg_pool` — average of all elements
- `global_max_pool` — maximum element
- `global_min_pool` — minimum element
- `global_majority_pool` — most common element

**Use when:** You need to collapse spatial dimensions entirely, e.g., for a classification logit.

### Adaptive Pool

Pools to an arbitrary target size `(target_rows, target_cols)`, regardless of input dimensions. Divides the input into approximately equal regions and applies average or max pooling to each.

**Use when:** You need a specific output size (e.g., before a fully-connected layer) and don't want to manually compute kernel/stride parameters.

### Stochastic Pool

Samples from the window proportional to shifted values: weight(v) = v + 1, giving weights {0, 1, 2} for {-1, 0, +1}. This means:
- **-1** has weight 0 — never selected
- **0** has weight 1 — sometimes selected
- **+1** has weight 2 — most likely selected

Uses a deterministic PRNG seeded by the provided value.

**Use when:** You want built-in regularization (similar to stochastic pooling in float networks by Zeiler & Fergus, 2013). Also useful during training to prevent co-adaptation of features.

## Comparison with Float Pooling

| Aspect | Float Pooling | Ternary Pooling |
|--------|---------------|-----------------|
| Data type | f32/f64 | i8 ({-1,0,+1}) |
| Max pool | `f32::max` | Simple comparison |
| Average pool | True mean | Sign of sum |
| Majority vote | Not applicable | Natural operation |
| Stochastic | Continuous distribution | 3-valued distribution |
| Memory | 32 bits/element | 2 bits/element |
| Compute | Floating-point ALU | Integer comparison |

## Accumulation Strategy

For average and adaptive average pooling, the accumulation uses integer sums and rounds the result:

```rust
fn round_to_trit(v: i32) -> i8 {
    match v {
        ..=-1 => -1,
        0 => 0,
        1.. => 1,
    }
}
```

This is equivalent to computing `sign(sum)` — a natural choice for ternary arithmetic.

## Testing

```bash
cargo test
```

The comprehensive test suite (20 tests) covers:
- Max/min pool correctness on known matrices
- All-negative and all-positive edge cases
- Majority vote with known counts and tie-breaking
- Average rounding to trits (positive, negative, zero sums)
- Global pool variants
- Adaptive pool with divisible and non-divisible sizes
- Stochastic pool determinism with same seed
- Output dimension correctness for various strides
- All outputs verified to be valid trits

## Related Crates

- [`ternary-matmul`](https://github.com/SuperInstance/ternary-matmul) — Ternary matrix multiplication
- [`ternary-conv`](https://github.com/SuperInstance/ternary-conv) — Ternary convolution operations

Together, these three crates provide a complete foundation for building ternary neural network inference pipelines entirely in Rust.

## License

MIT
