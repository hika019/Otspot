//! Ruiz equilibration scaling for LP preprocessing
//!
//! Iteratively scales rows and columns of the constraint matrix so that
//! the maximum absolute entry in each row/column is close to 1.

use crate::sparse::CscMatrix;
use crate::tolerances::UNDERFLOW_GUARD;
use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum Ruiz equilibration sweeps for LP presolve.
/// QP uses a separate RUIZ_SWEEPS (53) in `qp/ipm_core/scaling.rs`.
const LP_RUIZ_MAX_SWEEPS: usize = 20;

/// Convergence tolerance for LP Ruiz scaling: stop early when max scale change < this value.
const LP_RUIZ_CONV_TOL: f64 = 1e-4;

/// Scaled LP data: `(matrix, b, c, row_scale, col_scale)`.
pub type LpScalingResult = (CscMatrix, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

static LP_SCALE_PROFILE_US: AtomicU64 = AtomicU64::new(0);
static LP_SCALE_PROFILE_CALLS: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LpScaleProfileSnapshot {
    pub calls: u64,
    pub scale_us: u64,
}

pub fn lp_scale_profile_enabled() -> bool {
    std::env::var("OTSPOT_LP_SOLVE_PROFILE")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

pub fn reset_lp_scale_profile() {
    LP_SCALE_PROFILE_CALLS.store(0, Ordering::Relaxed);
    LP_SCALE_PROFILE_US.store(0, Ordering::Relaxed);
}

pub fn lp_scale_profile_snapshot() -> LpScaleProfileSnapshot {
    LpScaleProfileSnapshot {
        calls: LP_SCALE_PROFILE_CALLS.load(Ordering::Relaxed),
        scale_us: LP_SCALE_PROFILE_US.load(Ordering::Relaxed),
    }
}

pub fn lp_scale_profile_delta(
    before: LpScaleProfileSnapshot,
    after: LpScaleProfileSnapshot,
) -> LpScaleProfileSnapshot {
    LpScaleProfileSnapshot {
        calls: after.calls.saturating_sub(before.calls),
        scale_us: after.scale_us.saturating_sub(before.scale_us),
    }
}

/// LP equilibration via Ruiz scaling.
///
/// Static `scale()` only; not intended for instantiation.
pub struct LpEquilibration;

impl LpEquilibration {
    /// Apply Ruiz equilibration to a matrix, RHS vector, and cost vector.
    ///
    /// Returns `(scaled_matrix, scaled_b, scaled_c, row_scale, col_scale)`.
    ///
    /// The scaled problem satisfies:
    ///   Ã = diag(row_scale) * A * diag(col_scale)
    ///   b̃ = diag(row_scale) * b
    ///   c̃ = diag(col_scale) * c
    ///
    /// Iterates up to 20 rounds, stopping early when max scale change < 1e-4.
    pub fn scale(
        matrix: &CscMatrix,
        b: &[f64],
        c: &[f64],
    ) -> LpScalingResult {
        Self::scale_with_deadline(matrix, b, c, None)
            .expect("LpEquilibration::scale without deadline must not time out")
    }

    /// Deadline-aware Ruiz scaling. Returns `None` when the deadline expires
    /// while scaling is in progress.
    pub fn scale_with_deadline(
        matrix: &CscMatrix,
        b: &[f64],
        c: &[f64],
        deadline: Option<std::time::Instant>,
    ) -> Option<LpScalingResult> {
        let profile = lp_scale_profile_enabled();
        let t_profile = profile.then(std::time::Instant::now);
        let result = Self::scale_with_deadline_inner(matrix, b, c, deadline);
        if let Some(t0) = t_profile {
            LP_SCALE_PROFILE_CALLS.fetch_add(1, Ordering::Relaxed);
            LP_SCALE_PROFILE_US.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        result
    }

    fn scale_with_deadline_inner(
        matrix: &CscMatrix,
        b: &[f64],
        c: &[f64],
        deadline: Option<std::time::Instant>,
    ) -> Option<LpScalingResult> {
        #[inline]
        fn expired(deadline: Option<std::time::Instant>) -> bool {
            deadline.is_some_and(|d| std::time::Instant::now() >= d)
        }

        let m = matrix.nrows;
        let n = matrix.ncols;

        let mut cumul_row = vec![1.0f64; m];
        let mut cumul_col = vec![1.0f64; n];

        let mut a = matrix.clone();
        let mut cur_b = b.to_vec();
        let mut cur_c = c.to_vec();

        for _ in 0..LP_RUIZ_MAX_SWEEPS {
            if expired(deadline) {
                return None;
            }
            // Compute row maximums (iterate over all non-zeros)
            let mut row_max = vec![0.0f64; m];
            for k in 0..a.row_ind.len() {
                if expired(deadline) {
                    return None;
                }
                let row = a.row_ind[k];
                let v = a.values[k].abs();
                if v > row_max[row] {
                    row_max[row] = v;
                }
            }

            // Compute column maximums
            let mut col_max = vec![0.0f64; n];
            for (j, col_max_j) in col_max.iter_mut().enumerate().take(n) {
                if expired(deadline) {
                    return None;
                }
                let start = a.col_ptr[j];
                let end = a.col_ptr[j + 1];
                for k in start..end {
                    if expired(deadline) {
                        return None;
                    }
                    let v = a.values[k].abs();
                    if v > *col_max_j {
                        *col_max_j = v;
                    }
                }
            }

            // Compute scale factors: 1/sqrt(max), or 1.0 for empty rows/cols
            let row_factor: Vec<f64> = row_max
                .iter()
                .map(|&mx| {
                    if mx > UNDERFLOW_GUARD {
                        1.0 / mx.sqrt()
                    } else {
                        1.0
                    }
                })
                .collect();
            let col_factor: Vec<f64> = col_max
                .iter()
                .map(|&mx| {
                    if mx > UNDERFLOW_GUARD {
                        1.0 / mx.sqrt()
                    } else {
                        1.0
                    }
                })
                .collect();

            // Check convergence: max deviation of factors from 1.0
            let max_change = row_factor
                .iter()
                .chain(col_factor.iter())
                .map(|&f| (f - 1.0).abs())
                .fold(0.0f64, f64::max);

            // Apply scaling to matrix entries: a[i,j] *= row_factor[i] * col_factor[j]
            for (j, &cf) in col_factor.iter().enumerate().take(n) {
                if expired(deadline) {
                    return None;
                }
                let start = a.col_ptr[j];
                let end = a.col_ptr[j + 1];
                for k in start..end {
                    if expired(deadline) {
                        return None;
                    }
                    let row = a.row_ind[k];
                    a.values[k] *= row_factor[row] * cf;
                }
            }

            // Apply scaling to b and c
            for i in 0..m {
                if expired(deadline) {
                    return None;
                }
                cur_b[i] *= row_factor[i];
            }
            for j in 0..n {
                if expired(deadline) {
                    return None;
                }
                cur_c[j] *= col_factor[j];
            }

            // Accumulate cumulative scales
            for i in 0..m {
                if expired(deadline) {
                    return None;
                }
                cumul_row[i] *= row_factor[i];
            }
            for j in 0..n {
                if expired(deadline) {
                    return None;
                }
                cumul_col[j] *= col_factor[j];
            }

            if max_change < LP_RUIZ_CONV_TOL {
                break;
            }
        }

        Some((a, cur_b, cur_c, cumul_row, cumul_col))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dense_csc(data: &[Vec<f64>], nrows: usize, ncols: usize) -> CscMatrix {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (i, row) in data.iter().enumerate().take(nrows) {
            for (j, &val) in row.iter().enumerate().take(ncols) {
                if val.abs() > 1e-15 {
                    rows.push(i);
                    cols.push(j);
                    vals.push(val);
                }
            }
        }
        CscMatrix::from_triplets(&rows, &cols, &vals, nrows, ncols).unwrap()
    }

    #[test]
    fn test_ruiz_scale_identity() {
        // Identity matrix — already balanced, scales should stay near 1
        let a = make_dense_csc(&[vec![1.0, 0.0], vec![0.0, 1.0]], 2, 2);
        let b = vec![1.0, 2.0];
        let c = vec![1.0, 1.0];
        let (_, scaled_b, scaled_c, row_scale, col_scale) = LpEquilibration::scale(&a, &b, &c);

        // Scales should be close to 1
        for &s in row_scale.iter().chain(col_scale.iter()) {
            assert!((s - 1.0).abs() < 0.01, "Scale {} far from 1", s);
        }
        // b and c should be unchanged (already balanced)
        for i in 0..2 {
            assert!((scaled_b[i] - b[i] * row_scale[i]).abs() < 1e-10);
            assert!((scaled_c[i] - c[i] * col_scale[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ruiz_scale_unbalanced() {
        // Matrix with very unbalanced entries
        // [[1000, 1], [1, 0.001]]
        let a = make_dense_csc(&[vec![1000.0, 1.0], vec![1.0, 0.001]], 2, 2);
        let b = vec![1.0, 1.0];
        let c = vec![1.0, 1.0];
        let (scaled_a, _, _, _, _) = LpEquilibration::scale(&a, &b, &c);

        // After scaling, all entries should be close to 1 in magnitude
        for k in 0..scaled_a.values.len() {
            assert!(
                scaled_a.values[k].abs() < 2.0,
                "Entry {} too large after scaling",
                scaled_a.values[k]
            );
        }
    }
}
