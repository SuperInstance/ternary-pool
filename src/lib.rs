//! # ternary-pool
//!
//! Ternary pooling operations for matrices with elements in {-1, 0, +1}.
//!
//! Provides max, min, majority-vote, average (rounded to trit), global,
//! adaptive, and stochastic pooling. All operations use Z₃-aware logic
//! with explicit match arms.

use std::collections::HashMap;

/// A ternary matrix storing elements as i8 in {-1, 0, +1}.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TernaryMatrix {
    rows: usize,
    cols: usize,
    data: Vec<i8>,
}

impl TernaryMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0; rows * cols] }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<i8>) -> Self {
        assert_eq!(data.len(), rows * cols);
        for &v in &data {
            assert!(v >= -1 && v <= 1, "element {} not in {{-1, 0, 1}}", v);
        }
        Self { rows, cols, data }
    }

    pub fn random(rows: usize, cols: usize, seed: u64) -> Self {
        let mut s = seed;
        let data: Vec<i8> = (0..rows * cols).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            match (s >> 62) % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            }
        }).collect();
        Self { rows, cols, data }
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn get(&self, r: usize, c: usize) -> i8 { self.data[r * self.cols + c] }
    pub fn set(&mut self, r: usize, c: usize, v: i8) {
        assert!(v >= -1 && v <= 1);
        self.data[r * self.cols + c] = v;
    }

    /// Extract a window starting at (r, c) of size (h, w).
    fn window(&self, r: usize, c: usize, h: usize, w: usize) -> Vec<i8> {
        let mut out = Vec::with_capacity(h * w);
        for i in 0..h {
            for j in 0..w {
                let ri = r + i;
                let cj = c + j;
                if ri < self.rows && cj < self.cols {
                    out.push(self.get(ri, cj));
                }
            }
        }
        out
    }
}

/// Round integer average to nearest trit.
fn round_to_trit(v: i32) -> i8 {
    match v {
        ..=-1 => -1,
        0 => 0,
        1.. => 1,
    }
}

// ---------------------------------------------------------------------------
// Max Pool
// ---------------------------------------------------------------------------

/// 2D max pooling with given kernel size and stride.
///
/// For each window, selects the maximum trit value: 1 > 0 > -1.
pub fn max_pool(input: &TernaryMatrix, kernel: usize, stride: usize) -> TernaryMatrix {
    let out_rows = (input.rows() - kernel) / stride + 1;
    let out_cols = (input.cols() - kernel) / stride + 1;
    let mut out = TernaryMatrix::zeros(out_rows, out_cols);
    for i in 0..out_rows {
        for j in 0..out_cols {
            let r = i * stride;
            let c = j * stride;
            let mut max_val = -1;
            for ki in 0..kernel {
                for kj in 0..kernel {
                    let v = input.get(r + ki, c + kj);
                    if v > max_val { max_val = v; }
                }
            }
            out.set(i, j, max_val);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Min Pool
// ---------------------------------------------------------------------------

/// 2D min pooling with given kernel size and stride.
///
/// For each window, selects the minimum trit value: -1 < 0 < 1.
pub fn min_pool(input: &TernaryMatrix, kernel: usize, stride: usize) -> TernaryMatrix {
    let out_rows = (input.rows() - kernel) / stride + 1;
    let out_cols = (input.cols() - kernel) / stride + 1;
    let mut out = TernaryMatrix::zeros(out_rows, out_cols);
    for i in 0..out_rows {
        for j in 0..out_cols {
            let r = i * stride;
            let c = j * stride;
            let mut min_val = 1;
            for ki in 0..kernel {
                for kj in 0..kernel {
                    let v = input.get(r + ki, c + kj);
                    if v < min_val { min_val = v; }
                }
            }
            out.set(i, j, min_val);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Majority Pool
// ---------------------------------------------------------------------------

/// 2D majority-vote pooling.
///
/// For each window, selects the most common trit value. Ties broken by
/// preference order: 0 > 1 > -1 (neutral first).
pub fn majority_pool(input: &TernaryMatrix, kernel: usize, stride: usize) -> TernaryMatrix {
    let out_rows = (input.rows() - kernel) / stride + 1;
    let out_cols = (input.cols() - kernel) / stride + 1;
    let mut out = TernaryMatrix::zeros(out_rows, out_cols);
    for i in 0..out_rows {
        for j in 0..out_cols {
            let r = i * stride;
            let c = j * stride;
            let mut counts: HashMap<i8, usize> = HashMap::new();
            for ki in 0..kernel {
                for kj in 0..kernel {
                    *counts.entry(input.get(r + ki, c + kj)).or_insert(0) += 1;
                }
            }
            // Find max count, tie-break: 0 > 1 > -1
            let winner = [0, 1, -1].iter()
                .max_by_key(|&&v| (counts.get(&v).copied().unwrap_or(0), match v {
                    0 => 2, 1 => 1, -1 => 0, _ => 0
                }))
                .copied()
                .unwrap_or(0);
            out.set(i, j, winner);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Average Pool
// ---------------------------------------------------------------------------

/// 2D average pooling rounded to nearest trit.
///
/// Computes integer mean of window values, then rounds: negative → -1, zero → 0, positive → 1.
pub fn avg_pool(input: &TernaryMatrix, kernel: usize, stride: usize) -> TernaryMatrix {
    let out_rows = (input.rows() - kernel) / stride + 1;
    let out_cols = (input.cols() - kernel) / stride + 1;
    let mut out = TernaryMatrix::zeros(out_rows, out_cols);
    for i in 0..out_rows {
        for j in 0..out_cols {
            let r = i * stride;
            let c = j * stride;
            let mut sum: i32 = 0;
            for ki in 0..kernel {
                for kj in 0..kernel {
                    sum += input.get(r + ki, c + kj) as i32;
                }
            }
            out.set(i, j, round_to_trit(sum));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Global Pool
// ---------------------------------------------------------------------------

/// Global average pooling: reduces entire matrix to a single trit.
pub fn global_avg_pool(input: &TernaryMatrix) -> i8 {
    let sum: i32 = input.data.iter().map(|&v| v as i32).sum();
    round_to_trit(sum)
}

/// Global max pooling: maximum trit in the entire matrix.
pub fn global_max_pool(input: &TernaryMatrix) -> i8 {
    input.data.iter().copied().max().unwrap_or(0)
}

/// Global min pooling: minimum trit in the entire matrix.
pub fn global_min_pool(input: &TernaryMatrix) -> i8 {
    input.data.iter().copied().min().unwrap_or(0)
}

/// Global majority pooling: most common trit in the entire matrix.
pub fn global_majority_pool(input: &TernaryMatrix) -> i8 {
    let mut counts: HashMap<i8, usize> = HashMap::new();
    for &v in &input.data {
        *counts.entry(v).or_insert(0) += 1;
    }
    [0, 1, -1].iter()
        .max_by_key(|&&v| (counts.get(&v).copied().unwrap_or(0), match v {
            0 => 2, 1 => 1, -1 => 0, _ => 0
        }))
        .copied()
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Adaptive Pool
// ---------------------------------------------------------------------------

/// Adaptive average pooling to target output size.
///
/// Computes output by dividing the input into `(target_rows, target_cols)` regions
/// and averaging each region, rounded to nearest trit.
pub fn adaptive_avg_pool(input: &TernaryMatrix, target_rows: usize, target_cols: usize) -> TernaryMatrix {
    let mut out = TernaryMatrix::zeros(target_rows, target_cols);
    for i in 0..target_rows {
        for j in 0..target_cols {
            let r_start = i * input.rows() / target_rows;
            let r_end = (i + 1) * input.rows() / target_rows;
            let c_start = j * input.cols() / target_cols;
            let c_end = (j + 1) * input.cols() / target_cols;
            let mut sum: i32 = 0;
            let mut count: i32 = 0;
            for r in r_start..r_end {
                for c in c_start..c_end {
                    sum += input.get(r, c) as i32;
                    count += 1;
                }
            }
            out.set(i, j, if count > 0 { round_to_trit(sum) } else { 0 });
        }
    }
    out
}

/// Adaptive max pooling to target output size.
pub fn adaptive_max_pool(input: &TernaryMatrix, target_rows: usize, target_cols: usize) -> TernaryMatrix {
    let mut out = TernaryMatrix::zeros(target_rows, target_cols);
    for i in 0..target_rows {
        for j in 0..target_cols {
            let r_start = i * input.rows() / target_rows;
            let r_end = (i + 1) * input.rows() / target_rows;
            let c_start = j * input.cols() / target_cols;
            let c_end = (j + 1) * input.cols() / target_cols;
            let mut max_val = -1;
            for r in r_start..r_end {
                for c in c_start..c_end {
                    let v = input.get(r, c);
                    if v > max_val { max_val = v; }
                }
            }
            out.set(i, j, max_val);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Stochastic Pool
// ---------------------------------------------------------------------------

/// Stochastic pooling: randomly samples from the window proportional to value.
///
/// Values are shifted to be non-negative (v+1 → {0, 1, 2}) for probability weighting.
/// Uses the provided seed for deterministic behavior.
pub fn stochastic_pool(input: &TernaryMatrix, kernel: usize, stride: usize, seed: u64) -> TernaryMatrix {
    let out_rows = (input.rows() - kernel) / stride + 1;
    let out_cols = (input.cols() - kernel) / stride + 1;
    let mut out = TernaryMatrix::zeros(out_rows, out_cols);
    let mut rng_state = seed;

    for i in 0..out_rows {
        for j in 0..out_cols {
            let r = i * stride;
            let c = j * stride;
            // Collect values with weights (shift to non-negative: v+1)
            let mut weights = Vec::new();
            let mut total_weight = 0u32;
            for ki in 0..kernel {
                for kj in 0..kernel {
                    let v = input.get(r + ki, c + kj);
                    let w = (v + 1) as u32; // -1→0, 0→1, 1→2
                    weights.push((v, w));
                    total_weight += w;
                }
            }
            if total_weight == 0 {
                out.set(i, j, 0);
            } else {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let rand_val = (rng_state >> 33) as u32 % total_weight;
                let mut cumulative = 0u32;
                let mut chosen = 0i8;
                for &(v, w) in &weights {
                    cumulative += w;
                    if rand_val < cumulative {
                        chosen = v;
                        break;
                    }
                }
                out.set(i, j, chosen);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool_selects_correctly() {
        let m = TernaryMatrix::from_vec(4, 4, vec![
            1, -1, 0,  1,
            0,  0, 1, -1,
           -1,  1, 0,  0,
            1,  0, -1, 1,
        ]);
        let pooled = max_pool(&m, 2, 2);
        assert_eq!(pooled.rows(), 2);
        assert_eq!(pooled.cols(), 2);
        // Window (0,0): max of {1,-1,0,0} = 1
        assert_eq!(pooled.get(0, 0), 1);
        // Window (0,1): max of {0,1,1,0} = 1
        assert_eq!(pooled.get(0, 1), 1);
        // Window (1,0): max of {-1,1,1,0} = 1
        assert_eq!(pooled.get(1, 0), 1);
        // Window (1,1): max of {0,0,-1,1} = 1
        assert_eq!(pooled.get(1, 1), 1);
    }

    #[test]
    fn test_max_pool_all_negative() {
        let m = TernaryMatrix::from_vec(2, 2, vec![-1, -1, -1, -1]);
        let pooled = max_pool(&m, 2, 1);
        assert_eq!(pooled.get(0, 0), -1);
    }

    #[test]
    fn test_min_pool_selects_correctly() {
        let m = TernaryMatrix::from_vec(4, 4, vec![
            1, -1, 0,  1,
            0,  0, 1, -1,
           -1,  1, 0,  0,
            1,  0, -1, 1,
        ]);
        let pooled = min_pool(&m, 2, 2);
        // Window (0,0): min of {1,-1,0,0} = -1
        assert_eq!(pooled.get(0, 0), -1);
        // Window (1,1): min of {0,0,-1,1} = -1
        assert_eq!(pooled.get(1, 1), -1);
    }

    #[test]
    fn test_min_pool_all_positive() {
        let m = TernaryMatrix::from_vec(2, 2, vec![1, 1, 1, 1]);
        let pooled = min_pool(&m, 2, 1);
        assert_eq!(pooled.get(0, 0), 1);
    }

    #[test]
    fn test_majority_pool_basic() {
        let m = TernaryMatrix::from_vec(3, 3, vec![
            1, 1, 1,
            1, 0, -1,
            -1, -1, 0,
        ]);
        let pooled = majority_pool(&m, 3, 1);
        // 3x(1), 2x(0), 3x(-1) → tie between 1 and -1 → 1 wins over -1
        assert_eq!(pooled.get(0, 0), 1);
    }

    #[test]
    fn test_majority_pool_zeros_dominate() {
        let m = TernaryMatrix::from_vec(3, 3, vec![
            0, 0, 1,
            0, 0, -1,
            1, -1, 0,
        ]);
        let pooled = majority_pool(&m, 3, 1);
        // 5x(0), 2x(1), 2x(-1) → 0 wins
        assert_eq!(pooled.get(0, 0), 0);
    }

    #[test]
    fn test_avg_pool_rounds_to_trit() {
        let m = TernaryMatrix::from_vec(2, 2, vec![1, 1, 1, 1]);
        let pooled = avg_pool(&m, 2, 1);
        // Sum = 4, round_to_trit(4) = 1
        assert_eq!(pooled.get(0, 0), 1);

        let m2 = TernaryMatrix::from_vec(2, 2, vec![-1, -1, -1, -1]);
        let pooled2 = avg_pool(&m2, 2, 1);
        // Sum = -4, round_to_trit(-4) = -1
        assert_eq!(pooled2.get(0, 0), -1);

        let m3 = TernaryMatrix::from_vec(2, 2, vec![1, -1, -1, 1]);
        let pooled3 = avg_pool(&m3, 2, 1);
        // Sum = 0, round_to_trit(0) = 0
        assert_eq!(pooled3.get(0, 0), 0);
    }

    #[test]
    fn test_avg_pool_mixed() {
        let m = TernaryMatrix::from_vec(4, 4, vec![
            1, 1, -1, -1,
            1, 1, -1, -1,
            0, 0,  0,  0,
            0, 0,  0,  0,
        ]);
        let pooled = avg_pool(&m, 2, 2);
        // Window (0,0): sum=4 → 1
        assert_eq!(pooled.get(0, 0), 1);
        // Window (0,1): sum=-4 → -1
        assert_eq!(pooled.get(0, 1), -1);
        // Window (1,0): sum=0 → 0
        assert_eq!(pooled.get(1, 0), 0);
    }

    #[test]
    fn test_global_avg_pool() {
        let m = TernaryMatrix::from_vec(2, 2, vec![1, 1, 1, 1]);
        assert_eq!(global_avg_pool(&m), 1);

        let m2 = TernaryMatrix::from_vec(2, 2, vec![1, -1, 0, 0]);
        assert_eq!(global_avg_pool(&m2), 0); // sum=0
    }

    #[test]
    fn test_global_max_pool() {
        let m = TernaryMatrix::from_vec(3, 3, vec![
            -1, 0, -1,
             0, 1,  0,
            -1, 0, -1,
        ]);
        assert_eq!(global_max_pool(&m), 1);
    }

    #[test]
    fn test_global_min_pool() {
        let m = TernaryMatrix::from_vec(3, 3, vec![
            -1, 0, -1,
             0, 1,  0,
            -1, 0, -1,
        ]);
        assert_eq!(global_min_pool(&m), -1);
    }

    #[test]
    fn test_global_majority_pool() {
        let m = TernaryMatrix::from_vec(2, 2, vec![0, 0, 1, -1]);
        assert_eq!(global_majority_pool(&m), 0);
    }

    #[test]
    fn test_adaptive_avg_pool_target_size() {
        let m = TernaryMatrix::from_vec(4, 4, vec![
            1, 1, -1, -1,
            1, 1, -1, -1,
            0, 0,  0,  0,
            0, 0,  0,  0,
        ]);
        let pooled = adaptive_avg_pool(&m, 2, 2);
        assert_eq!(pooled.rows(), 2);
        assert_eq!(pooled.cols(), 2);
        // Region (0,0): sum=4 → 1
        assert_eq!(pooled.get(0, 0), 1);
        // Region (1,1): sum=0 → 0
        assert_eq!(pooled.get(1, 1), 0);
    }

    #[test]
    fn test_adaptive_max_pool() {
        let m = TernaryMatrix::from_vec(4, 4, vec![
            1, -1, 0,  1,
            0,  0, 1, -1,
           -1,  1, 0,  0,
            1,  0, -1, 1,
        ]);
        let pooled = adaptive_max_pool(&m, 1, 1);
        assert_eq!(pooled.rows(), 1);
        assert_eq!(pooled.cols(), 1);
        assert_eq!(pooled.get(0, 0), 1); // max of entire matrix
    }

    #[test]
    fn test_adaptive_pool_non_divisible() {
        let m = TernaryMatrix::from_vec(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ]);
        let pooled = adaptive_avg_pool(&m, 2, 2);
        assert_eq!(pooled.rows(), 2);
        assert_eq!(pooled.cols(), 2);
        // All ones → all 1
        assert_eq!(pooled.get(0, 0), 1);
        assert_eq!(pooled.get(1, 1), 1);
    }

    #[test]
    fn test_stochastic_pool_deterministic() {
        let m = TernaryMatrix::from_vec(4, 4, vec![
            1, 1, -1, -1,
            1, 1, -1, -1,
            0, 0,  0,  0,
            0, 0,  0,  0,
        ]);
        let p1 = stochastic_pool(&m, 2, 2, 42);
        let p2 = stochastic_pool(&m, 2, 2, 42);
        assert_eq!(p1, p2, "same seed should produce same result");
        // All values should be valid trits
        for i in 0..p1.rows() {
            for j in 0..p1.cols() {
                let v = p1.get(i, j);
                assert!(v >= -1 && v <= 1);
            }
        }
    }

    #[test]
    fn test_stochastic_pool_all_same_weight() {
        // All zeros → weight 1 each → uniform sampling
        let m = TernaryMatrix::from_vec(2, 2, vec![0, 0, 0, 0]);
        let pooled = stochastic_pool(&m, 2, 1, 123);
        assert_eq!(pooled.get(0, 0), 0); // all zeros → must pick 0
    }

    #[test]
    fn test_stride_affects_output_size() {
        let m = TernaryMatrix::zeros(6, 6);
        let p1 = max_pool(&m, 2, 1);
        let p2 = max_pool(&m, 2, 2);
        assert_eq!(p1.rows(), 5);
        assert_eq!(p2.rows(), 3);
    }

    #[test]
    fn test_pool_output_is_ternary() {
        let m = TernaryMatrix::random(8, 8, 42);
        for &v in &max_pool(&m, 2, 2).data {
            assert!(v >= -1 && v <= 1);
        }
        for &v in &min_pool(&m, 2, 2).data {
            assert!(v >= -1 && v <= 1);
        }
        for &v in &majority_pool(&m, 2, 2).data {
            assert!(v >= -1 && v <= 1);
        }
        for &v in &avg_pool(&m, 2, 2).data {
            assert!(v >= -1 && v <= 1);
        }
    }

    #[test]
    fn test_global_pool_on_zeros() {
        let m = TernaryMatrix::zeros(3, 4);
        assert_eq!(global_avg_pool(&m), 0);
        assert_eq!(global_max_pool(&m), 0);
        assert_eq!(global_min_pool(&m), 0);
        assert_eq!(global_majority_pool(&m), 0);
    }
}
