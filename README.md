# ternary-pool

**Downsampling that speaks ternary — eight ways to collapse {-1, 0, +1} matrices.**

[![crate](https://img.shields.io/badge/crates.io-ternary--pool-orange)](https://crates.io)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Why This Exists

After convolution, you need to shrink. Pooling reduces spatial dimensions while keeping the important signals — it's what makes deep networks computationally tractable. But standard pooling assumes floating-point values. Max pooling picks the largest number. Average pooling computes a mean. These operations have clear semantics in ℝ.

In ternary space {-1, 0, +1}, pooling takes on new meaning. "Max" becomes a three-way comparison. "Average" becomes a sign vote. And a new operation appears — **majority pooling** — that has no float analog. It's a vote among elements, and it's naturally noise-resistant in a way float pooling can never be.

## The Key Insight

Ternary pooling has a superpower that float pooling lacks: **the zero trit is a meaningful signal, not just "small."** In a float network, values near zero are weak activations. In a ternary network, zero means "neutral" — it's a first-class opinion. This makes majority voting (count {-1, 0, +1} and pick the winner) a principled, robust pooling operation. It's the Condorcet jury theorem applied to neural network features.

## Quick Start

```toml
[dependencies]
ternary-pool = "0.1"
```

```rust
use ternary_pool::*;

let input = TernaryMatrix::from_vec(4, 4, vec![
     1, -1,  0,  1,
     0,  0,  1, -1,
    -1,  1,  0,  0,
     1,  0, -1,  1,
]);

// Standard pooling
let maxed   = max_pool(&input, 2, 2);     // strongest positive signal
let mined   = min_pool(&input, 2, 2);     // strongest negative signal
let maj     = majority_pool(&input, 2, 2); // the vote winner
let avg     = avg_pool(&input, 2, 2);      // sign of the sum

// Global pooling — collapse entire matrix to one trit
let g_avg = global_avg_pool(&input);       // balanced summary
let g_max = global_max_pool(&input);       // any +1 present?
let g_min = global_min_pool(&input);       // any -1 present?
let g_maj = global_majority_pool(&input);  // overall consensus

// Adaptive — pool to arbitrary target dimensions
let adapted = adaptive_avg_pool(&input, 2, 2);

// Stochastic — regularization-friendly, seeded for reproducibility
let stoch = stochastic_pool(&input, 2, 2, 42);
```

## Architecture

```
                  ┌──────────────────┐
                  │  TernaryMatrix   │
                  └────────┬─────────┘
                           │
         ┌─────────────────┼──────────────────┐
         │                 │                  │
   ┌─────▼─────┐   ┌──────▼──────┐   ┌──────▼──────┐
   │  Max Pool  │   │  Min Pool   │   │  Majority   │
   │  (strongest│   │  (strongest │   │  Pool       │
   │   positive)│   │   negative) │   │  (vote)     │
   └───────────┘   └─────────────┘   └─────────────┘
         │                 │                  │
   ┌─────▼─────┐   ┌──────▼──────┐   ┌──────▼──────┐
   │  Avg Pool  │   │  Global     │   │  Stochastic │
   │  (sign of  │   │  (entire    │   │  (sampled,  │
   │   sum)     │   │   matrix)   │   │   seeded)   │
   └───────────┘   └─────────────┘   └─────────────┘
                           │
                    ┌──────▼──────┐
                    │  Adaptive   │
                    │  (arbitrary │
                    │   target)   │
                    └─────────────┘
```

## Strategy Guide

### Max Pool: Detect Presence

Selects the largest trit: 1 > 0 > -1. Every non-overlapping window contributes one value. If *any* position in the window is +1, that's what you get.

**Use when:** You care whether a positive feature exists anywhere in the region. Object detection, presence triggers.

### Min Pool: Detect Absence

The mirror of max pool. Selects the smallest trit: -1 < 0 < 1.

**Use when:** You need to know if negative evidence (inhibition) exists. Complementary to max pool in dual-channel architectures.

### Majority Pool: The Condorcet Voter

Counts occurrences of each trit in the window. The winner takes the position. Tie-breaking: 0 > 1 > -1 (neutral before positive, positive before negative).

This is **uniquely ternary** — there's no float analog because float values don't form a discrete consensus space. It's naturally robust to noise: a single stray activation can't override a clear majority.

**Use when:** Noise resistance matters. Individual ternary activations are noisy; majority voting smooths them out.

### Average Pool: The Sign Vote

Computes the integer sum and maps via sign: negative → -1, zero → 0, positive → +1. Equivalent to a weighted vote where each trit votes with its sign.

**Use when:** You want balanced representation. Classification heads (global average pooling → logit).

### Global Pool: Full Collapse

Reduces the entire matrix to one trit. Four variants: avg, max, min, majority.

**Use when:** You need a single summary statistic. Final layer before a ternary classifier.

### Adaptive Pool: Flexible Sizing

Pools to any target (rows, cols), regardless of input dimensions. Divides the input into approximately equal regions.

**Use when:** You need a specific output size and don't want to manually compute kernel/stride. Before fully-connected layers.

### Stochastic Pool: Built-in Regularization

Samples from the window proportional to `v + 1`, giving weights {0, 1, 2} for {-1, 0, +1}. The value -1 is *never* selected. Zero is sometimes selected. +1 is most likely.

**Use when:** You want regularization during training (prevents co-adaptation). Deterministic with seed — reproducible experiments.

## API Reference

### Standard Pooling

```rust
fn max_pool(input: &TernaryMatrix, kernel: usize, stride: usize) -> TernaryMatrix;
fn min_pool(input: &TernaryMatrix, kernel: usize, stride: usize) -> TernaryMatrix;
fn majority_pool(input: &TernaryMatrix, kernel: usize, stride: usize) -> TernaryMatrix;
fn avg_pool(input: &TernaryMatrix, kernel: usize, stride: usize) -> TernaryMatrix;
fn stochastic_pool(input: &TernaryMatrix, kernel: usize, stride: usize, seed: u64) -> TernaryMatrix;
```

### Global Pooling

```rust
fn global_avg_pool(input: &TernaryMatrix) -> i8;
fn global_max_pool(input: &TernaryMatrix) -> i8;
fn global_min_pool(input: &TernaryMatrix) -> i8;
fn global_majority_pool(input: &TernaryMatrix) -> i8;
```

### Adaptive Pooling

```rust
fn adaptive_avg_pool(input: &TernaryMatrix, target_rows: usize, target_cols: usize) -> TernaryMatrix;
fn adaptive_max_pool(input: &TernaryMatrix, target_rows: usize, target_cols: usize) -> TernaryMatrix;
```

### Core Type

```rust
struct TernaryMatrix { /* rows, cols, data: Vec<i8> */ }

impl TernaryMatrix {
    fn zeros(rows: usize, cols: usize) -> Self;
    fn from_vec(rows: usize, cols: usize, data: Vec<i8>) -> Self;
    fn random(rows: usize, cols: usize, seed: u64) -> Self;
    fn get(&self, r: usize, c: usize) -> i8;
    fn set(&mut self, r: usize, c: usize, v: i8);
}
```

## Real-World Example: Ternary Majority Voting for Sensor Fusion

Three acoustic sensors on a wildlife monitoring station each classify a sound as "predator" (+1), "unknown" (0), or "prey" (-1). They're noisy individually — wind, interference, distance. But by pooling their ternary outputs with majority voting:

```rust
let sensor_a = TernaryMatrix::from_vec(1, 1, vec![1]);   // predator
let sensor_b = TernaryMatrix::from_vec(1, 1, vec![-1]);  // prey (wrong)
let sensor_c = TernaryMatrix::from_vec(1, 1, vec![1]);   // predator

let fused = TernaryMatrix::from_vec(1, 3, vec![1, -1, 1]);
let consensus = majority_pool(&fused, 3, 1);
// → 1 (predator wins 2-1)

// Equivalently: global majority vote
let all_sensors = TernaryMatrix::from_vec(3, 1, vec![1, -1, 1]);
let vote = global_majority_pool(&all_sensors); // → 1
```

A single sensor error can't override the majority. That's the power of ternary voting — it's Condorcet's theorem in action, and it's why ternary networks are surprisingly robust to quantization noise.

## Performance Characteristics

- **Max/Min Pool**: O(H × W × k²) — one comparison per element per window. Trivially parallelizable.
- **Majority Pool**: O(H × W × k²) — counts via HashMap, slightly more overhead than max/min.
- **Average Pool**: O(H × W × k²) — integer addition + sign check. No division needed (sign of sum).
- **Stochastic Pool**: O(H × W × k²) — adds PRNG sampling per element.
- **Global Pool**: O(H × W) — single pass over all elements.
- **Adaptive Pool**: O(H × W) — visits each element exactly once.

Memory: Output is always smaller than input (that's the point). A 4×4 input with 2×2 kernel produces a 2×2 output — 4× reduction.

## Ecosystem Connections

Pooling sits between convolution and classification in the ternary network pipeline:

- [`ternary-conv`](https://github.com/SuperInstance/ternary-conv) — produces the feature maps this crate downsamples
- [`ternary-matmul`](https://github.com/SuperInstance/ternary-matmul) — fully-connected layers after global pooling
- [`ternary-norm`](https://github.com/SuperInstance/ternary-norm) — normalize before or after pooling
- [`ternary-activation`](https://github.com/SuperInstance/ternary-activation) — apply non-linearity after pooling

## Open Questions

- **Learnable pooling**: Can the tie-breaking order (0 > 1 > -1) be learned? Different tasks might prefer different biases.
- **Fractional strides**: Currently stride must evenly divide the input minus kernel. Fractional strides could enable smoother downsampling.
- **Mixed-precision pooling**: Pool ternary activations with float statistics for gradient computation (straight-through estimator).

## Testing

```bash
cargo test
```

20 tests covering: max/min on known matrices, all-negative/all-positive edge cases, majority vote counts and tie-breaking, average rounding (positive/negative/zero sums), all global variants, adaptive with divisible and non-divisible sizes, stochastic determinism with same seed, stride effects on output dimensions, and all outputs validated as ternary.

## License

MIT
