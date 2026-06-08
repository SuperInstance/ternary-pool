# ternary-pool

Downsampling operations for matrices where every element is {−1, 0, +1}.

## The Problem

After convolution in a ternary neural network, you have feature maps — matrices of trits — that are too large. You need to reduce spatial dimensions while preserving the signal. Standard pooling (max, average) works for floats, but ternary space has different semantics: zero isn't "small," it's "neutral." When you average {-1, +1, +1, -1} and get 0, you've lost the information that there was strong signal in both directions. The mean is uninformative.

The problem: you need pooling operations that understand ternary structure — that zero is a meaningful vote, that {-1, 0, +1} form a group, and that "majority" is a more natural aggregation than "mean" when values are discrete opinions.

## The Insight

Ternary pooling has a superpower: **majority voting**. In float space, "major vote" is meaningless — values are continuous. In ternary space, every element casts one of three votes: negative, neutral, or positive. Aggregating by majority is the Condorcet jury theorem applied to neural network features: if each element is more likely right than wrong, the majority is almost certainly right, and the error rate drops exponentially with window size.

This is why majority pooling is the star of this crate. It's noise-resistant in a way float pooling can never be: a single stray activation can't override a clear majority. For ternary networks where quantization noise is the primary failure mode, this is the right default.

The second insight: average pooling in ternary space is just `sign(sum)`. The average of {-1, 0, +1} values is {-1, 0, +1} after rounding. No floating point needed — you sum integers and check the sign. This is O(1) per element with no division.

## How It Works

Eight pooling operations, four categories:

### Windowed Pooling (2D, kernel × kernel, stride)

Each operation slides a window across the matrix and produces one output trit per window:

| Operation | Semantics | Output rule |
|-----------|-----------|-------------|
| `max_pool` | Strongest positive signal | `max(window)` where 1 > 0 > -1 |
| `min_pool` | Strongest negative signal | `min(window)` where -1 < 0 < 1 |
| `majority_pool` | Consensus vote | Count {-1, 0, +1}, pick the most common. Tie-break: 0 > 1 > -1 |
| `avg_pool` | Sign of the sum | `sign(Σ window)` — positive→1, zero→0, negative→-1 |
| `stochastic_pool` | Weighted random sample | Sample proportional to `v+1` (weights: -1→0, 0→1, +1→2). Seeded for reproducibility |

### Global Pooling (entire matrix → one trit)

`global_avg_pool`, `global_max_pool`, `global_min_pool`, `global_majority_pool` — same semantics, applied to the whole matrix.

### Adaptive Pooling (arbitrary target size)

`adaptive_avg_pool`, `adaptive_max_pool` — divide the input into `(target_rows × target_cols)` regions and pool each region. Handles non-divisible sizes by partitioning into approximately equal regions.

### The rounding rule

Average pooling uses `round_to_trit`: sum < 0 → -1, sum = 0 → 0, sum > 0 → 1. This is sign-of-sum, not arithmetic mean. For a 2×2 window of {1, 1, -1, -1}, the sum is 0 → output 0. The information that both extremes were present is lost. This is by design: the output must be ternary, and 0 is the honest answer when signal cancels.

## Code Example

```rust
use ternary_pool::*;

let input = TernaryMatrix::from_vec(4, 4, vec![
     1, -1,  0,  1,
     0,  0,  1, -1,
    -1,  1,  0,  0,
     1,  0, -1,  1,
]);

// Windowed pooling (2×2 kernel, stride 2)
let maxed = max_pool(&input, 2, 2);      // strongest positive per window
let mined = min_pool(&input, 2, 2);      // strongest negative per window
let maj   = majority_pool(&input, 2, 2); // vote winner per window
let avg   = avg_pool(&input, 2, 2);      // sign of sum per window

assert_eq!(maxed.get(0, 0), 1);  // window {1,-1,0,0}: max = 1
assert_eq!(mined.get(0, 0), -1); // window {1,-1,0,0}: min = -1

// Global pooling — entire matrix to one trit
let g_max = global_max_pool(&input);       // 1 (at least one +1 exists)
let g_min = global_min_pool(&input);       // -1 (at least one -1 exists)
let g_avg = global_avg_pool(&input);       // sign of total sum
let g_maj = global_majority_pool(&input);  // most common trit overall

// Adaptive — pool to specific output dimensions
let adapted = adaptive_avg_pool(&input, 2, 2);  // 2×2 output

// Stochastic — reproducible randomness, regularization-friendly
let s1 = stochastic_pool(&input, 2, 2, 42);
let s2 = stochastic_pool(&input, 2, 2, 42);
assert_eq!(s1, s2);  // same seed → same result

// Construction helpers
let zeros = TernaryMatrix::zeros(6, 6);
let random = TernaryMatrix::random(6, 6, 12345);
```

## Module Map

```
ternary_pool
├── TernaryMatrix
│   ├── zeros(rows, cols)
│   ├── from_vec(rows, cols, data: Vec<i8>)  — asserts all values in {-1,0,1}
│   ├── random(rows, cols, seed)              — seeded PRNG, uniform over {-1,0,1}
│   ├── get(r, c) → i8
│   ├── set(r, c, v)
│   ├── rows() / cols()
│   └── window(r, c, h, w) → Vec<i8>         — extract sub-matrix (private)
│
├── Windowed Pooling
│   ├── max_pool(input, kernel, stride) → TernaryMatrix
│   ├── min_pool(input, kernel, stride) → TernaryMatrix
│   ├── majority_pool(input, kernel, stride) → TernaryMatrix
│   ├── avg_pool(input, kernel, stride) → TernaryMatrix
│   └── stochastic_pool(input, kernel, stride, seed) → TernaryMatrix
│
├── Global Pooling
│   ├── global_avg_pool(input) → i8
│   ├── global_max_pool(input) → i8
│   ├── global_min_pool(input) → i8
│   └── global_majority_pool(input) → i8
│
├── Adaptive Pooling
│   ├── adaptive_avg_pool(input, target_rows, target_cols) → TernaryMatrix
│   └── adaptive_max_pool(input, target_rows, target_cols) → TernaryMatrix
│
└── Internal
    └── round_to_trit(v: i32) → i8         — negative→-1, zero→0, positive→1
```

## Design Decisions

**Majority tie-breaking: 0 > 1 > -1.** When two or three trits tie in count, neutral wins over positive, positive wins over negative. The rationale: in the absence of clear signal, "no opinion" (0) is a safer default than committing to a direction. This bias matters — you can argue for 1 > 0 > -1 (optimistic) or -1 > 0 > 1 (pessimistic) depending on the application. The choice is arbitrary but consistent.

**Stochastic pooling uses `v+1` weighting: -1→0, 0→1, +1→2.** This means -1 is *never* sampled (weight 0). The -1 trit is treated as pure noise — it contributes to the probability distribution but can't be the output. This is a design choice, not a mathematical necessity. It makes stochastic pooling a one-sided operation: it samples from {0, +1} only.

**No padding.** Windows that don't fit are simply not computed. The output size is `(rows - kernel) / stride + 1`. If you need padding, pre-pad the `TernaryMatrix` with zeros before calling the pool function.

**Separate global functions, not methods on TernaryMatrix.** Pooling operations are free functions that take `&TernaryMatrix`. This keeps `TernaryMatrix` as a plain data container and lets the pool functions compose freely.

**Linear congruential generator (LCG) for stochastic pooling.** The PRNG is `state = state * 6364136223846793005 + 1442695040888963407`. This is a fast, deterministic, no-allocation generator. It's not cryptographically secure — it doesn't need to be. The seed makes experiments reproducible.

**HashMap for majority counts.** With only 3 possible keys, a HashMap is overkill — three counters would be faster. The HashMap approach is cleaner code and the performance difference is negligible for typical window sizes. If this becomes a bottleneck, replace with three `usize` counters.

## Status

| Aspect | State |
|--------|-------|
| Max / Min pool | Stable, tested |
| Majority pool | Stable, tested |
| Average pool | Stable, tested |
| Global pool (4 variants) | Stable, tested |
| Adaptive pool | Stable, tested |
| Stochastic pool | Stable, tested |
| Padding modes | Not supported |
| Fractional strides | Not supported |
| Learnable pooling | Not supported |
| Backpropagation | Not supported |
| MSRV | Edition 2024 |
| Tests | 20 |

**Known limitations:** No padding means input dimensions must accommodate the kernel. Stochastic pooling never outputs -1 due to zero weighting. The crate operates on CPU with no SIMD or parallel computation. The `TernaryMatrix` type is row-major with no stride support — submatrices require copying.

## Related Crates

- **[ternary-conv](https://github.com/SuperInstance/ternary-conv)** — Produces the feature maps this crate downsamples
- **[ternary-regression](https://github.com/SuperInstance/ternary-regression)** — Predict continuous targets from pooled features
- **[ternary-logistic](https://github.com/SuperInstance/ternary-logistic)** — Classify after global pooling
- **[ternary-norm](https://github.com/SuperInstance/ternary-norm)** — Normalize before or after pooling

## License

MIT
