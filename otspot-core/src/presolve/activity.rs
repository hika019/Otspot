//! Row activity range and implied-bound propagation, shared by LP/QP/MIP presolve.

use crate::problem::ConstraintType;
use crate::tolerances::ZERO_TOL;

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

/// Propagate implied variable bounds for one constraint row.
///
/// For each `(j, a_ij)` in `entries`, derives the tightest implied lb/ub consistent
/// with the constraint `sum_j a_ij x_j {ct} b`. When `int_mask[j]` is `true`, the
/// implied bound is additionally rounded (`floor` for ub, `ceil` for lb) — this is
/// the MIP-presolve rounding that is strictly stronger than the continuous implied bound.
///
/// Returns `None` when infeasibility is detected (the rounded implied bound crosses
/// the existing opposite bound). Returns the tightened `(j, new_lb, new_ub)` triples
/// for variables whose bounds changed; unchanged variables are omitted.
pub(crate) fn propagate_row_bounds(
    entries: &[(usize, f64)],
    bounds: &[(f64, f64)],
    ct: ConstraintType,
    b: f64,
    int_mask: Option<&[bool]>,
) -> Option<Vec<(usize, f64, f64)>> {
    if entries.is_empty() {
        return Some(vec![]);
    }

    // Per-entry activity contributions (row_lb/ub = sum; inf counts guard division).
    let mut row_lb = 0.0f64;
    let mut row_ub = 0.0f64;
    let mut inf_lb_count = 0usize;
    let mut inf_ub_count = 0usize;
    let mut e_lb_contrib = Vec::with_capacity(entries.len());
    let mut e_ub_contrib = Vec::with_capacity(entries.len());
    let mut e_lb_inf = Vec::with_capacity(entries.len());
    let mut e_ub_inf = Vec::with_capacity(entries.len());

    for &(j, a_ij) in entries {
        let (lb, ub) = bounds[j];
        if a_ij > 0.0 {
            if lb == f64::NEG_INFINITY {
                inf_lb_count += 1;
                e_lb_inf.push(true);
                e_lb_contrib.push(0.0);
            } else {
                let c = a_ij * lb;
                e_lb_inf.push(false);
                e_lb_contrib.push(c);
                row_lb += c;
            }
            if ub == f64::INFINITY {
                inf_ub_count += 1;
                e_ub_inf.push(true);
                e_ub_contrib.push(0.0);
            } else {
                let c = a_ij * ub;
                e_ub_inf.push(false);
                e_ub_contrib.push(c);
                row_ub += c;
            }
        } else if a_ij < 0.0 {
            if ub == f64::INFINITY {
                inf_lb_count += 1;
                e_lb_inf.push(true);
                e_lb_contrib.push(0.0);
            } else {
                let c = a_ij * ub;
                e_lb_inf.push(false);
                e_lb_contrib.push(c);
                row_lb += c;
            }
            if lb == f64::NEG_INFINITY {
                inf_ub_count += 1;
                e_ub_inf.push(true);
                e_ub_contrib.push(0.0);
            } else {
                let c = a_ij * lb;
                e_ub_inf.push(false);
                e_ub_contrib.push(c);
                row_ub += c;
            }
        } else {
            e_lb_inf.push(false);
            e_ub_inf.push(false);
            e_lb_contrib.push(0.0);
            e_ub_contrib.push(0.0);
        }
    }

    let mut updates = Vec::new();

    for (k, &(j, a_ij)) in entries.iter().enumerate() {
        if a_ij.abs() < ZERO_TOL {
            continue;
        }
        let (old_lb, old_ub) = bounds[j];
        let is_int = int_mask.and_then(|m| m.get(j)).copied().unwrap_or(false);

        let rest_inf_lb = if e_lb_inf[k] { inf_lb_count - 1 } else { inf_lb_count };
        let rest_inf_ub = if e_ub_inf[k] { inf_ub_count - 1 } else { inf_ub_count };
        let rest_lb = row_lb - e_lb_contrib[k];
        let rest_ub = row_ub - e_ub_contrib[k];
        let rest_lb_fin = rest_inf_lb == 0;
        let rest_ub_fin = rest_inf_ub == 0;

        let mut new_lb = old_lb;
        let mut new_ub = old_ub;

        match ct {
            ConstraintType::Le => {
                if a_ij > 0.0 && rest_lb_fin {
                    let mut implied_ub = (b - rest_lb) / a_ij;
                    if is_int { implied_ub = implied_ub.floor(); }
                    if implied_ub < old_lb - ZERO_TOL { return None; }
                    if implied_ub < new_ub - ZERO_TOL { new_ub = implied_ub; }
                } else if a_ij < 0.0 && rest_lb_fin {
                    let mut implied_lb = (b - rest_lb) / a_ij;
                    if is_int { implied_lb = implied_lb.ceil(); }
                    if implied_lb > old_ub + ZERO_TOL { return None; }
                    if implied_lb > new_lb + ZERO_TOL { new_lb = implied_lb; }
                }
            }
            ConstraintType::Ge => {
                if a_ij > 0.0 && rest_ub_fin {
                    let mut implied_lb = (b - rest_ub) / a_ij;
                    if is_int { implied_lb = implied_lb.ceil(); }
                    if implied_lb > old_ub + ZERO_TOL { return None; }
                    if implied_lb > new_lb + ZERO_TOL { new_lb = implied_lb; }
                } else if a_ij < 0.0 && rest_ub_fin {
                    let mut implied_ub = (b - rest_ub) / a_ij;
                    if is_int { implied_ub = implied_ub.floor(); }
                    if implied_ub < old_lb - ZERO_TOL { return None; }
                    if implied_ub < new_ub - ZERO_TOL { new_ub = implied_ub; }
                }
            }
            ConstraintType::Eq => {
                if a_ij > 0.0 {
                    if rest_lb_fin {
                        let mut implied_ub = (b - rest_lb) / a_ij;
                        if is_int { implied_ub = implied_ub.floor(); }
                        if implied_ub < old_lb - ZERO_TOL { return None; }
                        if implied_ub < new_ub - ZERO_TOL { new_ub = implied_ub; }
                    }
                    if rest_ub_fin {
                        let mut implied_lb = (b - rest_ub) / a_ij;
                        if is_int { implied_lb = implied_lb.ceil(); }
                        if implied_lb > old_ub + ZERO_TOL { return None; }
                        if implied_lb > new_lb + ZERO_TOL { new_lb = implied_lb; }
                    }
                } else {
                    // a_ij < 0
                    if rest_lb_fin {
                        let mut implied_lb = (b - rest_lb) / a_ij;
                        if is_int { implied_lb = implied_lb.ceil(); }
                        if implied_lb > old_ub + ZERO_TOL { return None; }
                        if implied_lb > new_lb + ZERO_TOL { new_lb = implied_lb; }
                    }
                    if rest_ub_fin {
                        let mut implied_ub = (b - rest_ub) / a_ij;
                        if is_int { implied_ub = implied_ub.floor(); }
                        if implied_ub < old_lb - ZERO_TOL { return None; }
                        if implied_ub < new_ub - ZERO_TOL { new_ub = implied_ub; }
                    }
                }
            }
        }

        if (new_lb - old_lb).abs() > ZERO_TOL || (new_ub - old_ub).abs() > ZERO_TOL {
            updates.push((j, new_lb, new_ub));
        }
    }

    Some(updates)
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
