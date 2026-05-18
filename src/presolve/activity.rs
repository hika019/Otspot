//! Row activity range over variable bounds, shared by LP/QP presolve.

/// Compute `[row_lb, row_ub]` of `sum_j a_ij * x_j` over `x_j ∈ bounds[j]`.
///
/// `exclude_col = Some(k)` skips column `k` (used by QP implied-bound logic).
/// The boolean flags indicate whether each bound is finite; when `false`, the
/// corresponding `f64` is unspecified and must not be relied on.
pub(crate) fn activity_range(
    entries: &[(usize, f64)],
    bounds: &[(f64, f64)],
    exclude_col: Option<usize>,
) -> (f64, f64, bool, bool) {
    let mut row_lb = 0.0f64;
    let mut row_ub = 0.0f64;
    let mut lb_finite = true;
    let mut ub_finite = true;

    for &(j, a_ij) in entries {
        if Some(j) == exclude_col {
            continue;
        }
        let (lb_j, ub_j) = bounds[j];
        if a_ij > 0.0 {
            if lb_j == f64::NEG_INFINITY {
                lb_finite = false;
            } else if lb_finite {
                row_lb += a_ij * lb_j;
            }
            if ub_j == f64::INFINITY {
                ub_finite = false;
            } else if ub_finite {
                row_ub += a_ij * ub_j;
            }
        } else if a_ij < 0.0 {
            if ub_j == f64::INFINITY {
                lb_finite = false;
            } else if lb_finite {
                row_lb += a_ij * ub_j;
            }
            if lb_j == f64::NEG_INFINITY {
                ub_finite = false;
            } else if ub_finite {
                row_ub += a_ij * lb_j;
            }
        }
    }
    (row_lb, row_ub, lb_finite, ub_finite)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEG_INF: f64 = f64::NEG_INFINITY;
    const POS_INF: f64 = f64::INFINITY;

    #[test]
    fn finite_bounds_positive_and_negative_coeffs() {
        // row: 2 x0 - 3 x1, x0 ∈ [1, 4], x1 ∈ [-2, 5]
        // lb = 2*1 + (-3)*5 = -13, ub = 2*4 + (-3)*(-2) = 14
        let entries = vec![(0, 2.0), (1, -3.0)];
        let bounds = vec![(1.0, 4.0), (-2.0, 5.0)];
        let (lb, ub, lf, uf) = activity_range(&entries, &bounds, None);
        assert!(lf && uf);
        assert!((lb - (-13.0)).abs() < 1e-12);
        assert!((ub - 14.0).abs() < 1e-12);
    }

    #[test]
    fn infinite_lower_bound_marks_lb_infinite_for_positive_coeff() {
        let entries = vec![(0, 1.5)];
        let bounds = vec![(NEG_INF, 10.0)];
        let (_, ub, lf, uf) = activity_range(&entries, &bounds, None);
        assert!(!lf);
        assert!(uf);
        assert!((ub - 15.0).abs() < 1e-12);
    }

    #[test]
    fn infinite_upper_bound_with_negative_coeff_flips_lb_flag() {
        // a < 0 + ub = +inf -> lb_finite=false; lb_j finite -> ub finite
        let entries = vec![(0, -2.0)];
        let bounds = vec![(1.0, POS_INF)];
        let (_, ub, lf, uf) = activity_range(&entries, &bounds, None);
        assert!(!lf);
        assert!(uf);
        assert!((ub - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn zero_coefficient_is_skipped() {
        let entries = vec![(0, 0.0), (1, 1.0)];
        let bounds = vec![(NEG_INF, POS_INF), (2.0, 3.0)];
        let (lb, ub, lf, uf) = activity_range(&entries, &bounds, None);
        // x0 has infinite bounds but coeff=0, so flags remain true.
        assert!(lf && uf);
        assert!((lb - 2.0).abs() < 1e-12);
        assert!((ub - 3.0).abs() < 1e-12);
    }

    #[test]
    fn exclude_col_skips_target_entry() {
        // Same row as test 1 but excluding column 1; result must match a row
        // with only (0, 2.0) over x0 ∈ [1, 4].
        let entries = vec![(0, 2.0), (1, -3.0)];
        let bounds = vec![(1.0, 4.0), (-2.0, 5.0)];
        let (lb, ub, lf, uf) = activity_range(&entries, &bounds, Some(1));
        assert!(lf && uf);
        assert!((lb - 2.0).abs() < 1e-12);
        assert!((ub - 8.0).abs() < 1e-12);
    }

    #[test]
    fn empty_row_yields_zero_finite() {
        let (lb, ub, lf, uf) = activity_range(&[], &[(0.0, 1.0)], None);
        assert!(lf && uf);
        assert_eq!(lb, 0.0);
        assert_eq!(ub, 0.0);
    }

    #[test]
    fn both_directions_infinite() {
        // x0 free, coeff positive -> both flags should be false.
        let entries = vec![(0, 1.0)];
        let bounds = vec![(NEG_INF, POS_INF)];
        let (_, _, lf, uf) = activity_range(&entries, &bounds, None);
        assert!(!lf && !uf);
    }
}
