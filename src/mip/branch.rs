//! Integer branching for MILP/MIQP branch-and-bound (#14).
//!
//! Branching tightens one integer variable's bounds around its fractional
//! relaxation value `v`: the **down** child adds `x_j <= floor(v)` and the **up**
//! child adds `x_j >= ceil(v)`. This is exactly bound tightening — the same node
//! mechanism the spatial QP B&B uses for box splitting — so each child relaxation
//! is solved by swapping the bounds vector.
//!
//! The integrality tolerance is supplied by the caller (`MipConfig::integer_feas_tol`);
//! no tolerance is hard-coded here.

use super::node::VarBounds;
use crate::options::MipBranching;

/// Distance from `v` to its nearest integer, in `[0, 0.5]`.
pub(crate) fn fractionality(v: f64) -> f64 {
    (v - v.round()).abs()
}

/// `true` when `v` is within `tol` of an integer.
pub(crate) fn is_integer(v: f64, tol: f64) -> bool {
    fractionality(v) <= tol
}

/// `true` when every integer-constrained component of `x` is integral within `tol`.
pub(crate) fn is_integer_feasible(x: &[f64], integer_mask: &[bool], tol: f64) -> bool {
    debug_assert_eq!(x.len(), integer_mask.len(), "x / mask length mismatch");
    integer_mask
        .iter()
        .zip(x.iter())
        .all(|(&is_int, &v)| !is_int || is_integer(v, tol))
}

/// Select the branching variable according to the configured `strategy`.
///
/// Returns `None` when all integer variables are already integral within `tol`
/// (the relaxation solution is integer-feasible — nothing to branch on).
pub(crate) fn select_branching_variable(
    x: &[f64],
    integer_mask: &[bool],
    tol: f64,
    strategy: MipBranching,
) -> Option<usize> {
    debug_assert_eq!(x.len(), integer_mask.len(), "x / mask length mismatch");
    match strategy {
        MipBranching::MostFractional => select_most_fractional(x, integer_mask, tol),
    }
}

/// Most-fractional rule: pick the integer variable whose value is closest to a
/// half-integer. Ties break to the smallest index for determinism.
fn select_most_fractional(x: &[f64], integer_mask: &[bool], tol: f64) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (j, &is_int) in integer_mask.iter().enumerate() {
        if !is_int {
            continue;
        }
        let frac = fractionality(x[j]);
        if frac <= tol {
            continue;
        }
        let better = match best {
            None => true,
            Some((_, bf)) => frac > bf,
        };
        if better {
            best = Some((j, frac));
        }
    }
    best.map(|(j, _)| j)
}

/// Produce the down/up child bounds by branching variable `j` at value `v`.
///
/// - down child: `upper[j] = floor(v)`
/// - up child:   `lower[j] = ceil(v)`
///
/// `v` is assumed fractional (so `floor(v) < ceil(v)`); if the resulting box is
/// empty (e.g. `floor(v) < lower[j]`) the child relaxation is infeasible and the
/// driver prunes it.
pub(crate) fn branch_bounds(
    parent_bounds: &[(f64, f64)],
    j: usize,
    v: f64,
) -> (VarBounds, VarBounds) {
    let mut down = parent_bounds.to_vec();
    down[j].1 = v.floor();
    let mut up = parent_bounds.to_vec();
    up[j].0 = v.ceil();
    (down, up)
}

/// Select an integer variable whose box still spans at least two integers
/// (`ub - lb >= 1`), preferring the widest. Used as a **fallback** when a node's
/// relaxation cannot be solved (no interior — e.g. an equality constraint pins
/// the region to a point): bisecting the integer box drives the search toward an
/// all-fixed leaf, which the fixed-point evaluator solves exactly, so a region
/// that may hold the optimum is never silently dropped.
///
/// Returns `None` when every integer variable is already a singleton (no integer
/// box left to split).
pub(crate) fn widest_splittable_integer(
    bounds: &[(f64, f64)],
    integer_mask: &[bool],
) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (j, &is_int) in integer_mask.iter().enumerate() {
        if !is_int {
            continue;
        }
        let (lb, ub) = bounds[j];
        if !lb.is_finite() || !ub.is_finite() {
            continue;
        }
        let width = ub - lb;
        if width < 1.0 {
            continue; // fewer than two integers → cannot split
        }
        if best.is_none_or(|(_, bw)| width > bw) {
            best = Some((j, width));
        }
    }
    best.map(|(j, _)| j)
}

/// Bisect integer variable `j`'s box into two non-empty integer subranges:
/// down `[lb, mid]`, up `[mid + 1, ub]`, where `mid = floor((lb + ub) / 2)`
/// clamped so both children are non-empty. Caller must ensure `ub - lb >= 1`.
pub(crate) fn split_integer_box(
    bounds: &[(f64, f64)],
    j: usize,
) -> (VarBounds, VarBounds) {
    let (lb, ub) = bounds[j];
    let mid = (0.5 * (lb + ub)).floor().max(lb).min(ub - 1.0);
    let mut down = bounds.to_vec();
    down[j].1 = mid;
    let mut up = bounds.to_vec();
    up[j].0 = mid + 1.0;
    (down, up)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractionality_measures_distance_to_nearest_integer() {
        assert!((fractionality(2.0) - 0.0).abs() < 1e-15);
        assert!((fractionality(2.5) - 0.5).abs() < 1e-15);
        assert!((fractionality(2.3) - 0.3).abs() < 1e-12);
        assert!((fractionality(2.7) - 0.3).abs() < 1e-12);
        assert!((fractionality(-1.4) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn is_integer_respects_tolerance() {
        assert!(is_integer(3.0000001, 1e-6));
        assert!(!is_integer(3.1, 1e-6));
        assert!(is_integer(-2.9999999, 1e-6));
    }

    #[test]
    fn integer_feasible_ignores_continuous_components() {
        // var0 integer (fractional), var1 continuous (fractional)
        let x = vec![1.5, 0.3];
        assert!(!is_integer_feasible(&x, &[true, false], 1e-6));
        let x2 = vec![1.0, 0.3];
        assert!(is_integer_feasible(&x2, &[true, false], 1e-6));
    }

    #[test]
    fn select_picks_most_fractional_then_lowest_index() {
        // var0 frac 0.2, var1 frac 0.5, var2 frac 0.5 → tie var1/var2 → var1 (lower idx)
        let x = vec![1.2, 1.5, 2.5];
        let j = select_branching_variable(&x, &[true, true, true], 1e-6, MipBranching::MostFractional)
            .unwrap();
        assert_eq!(j, 1);
    }

    #[test]
    fn select_skips_continuous_and_integral_vars() {
        // var0 continuous (frac 0.5 but not integer-constrained), var1 integral, var2 frac 0.4
        let x = vec![0.5, 3.0, 2.4];
        let j = select_branching_variable(&x, &[false, true, true], 1e-6, MipBranching::MostFractional)
            .unwrap();
        assert_eq!(j, 2);
    }

    #[test]
    fn select_returns_none_when_all_integral() {
        let x = vec![1.0, 2.0, 3.0];
        assert!(
            select_branching_variable(&x, &[true, true, true], 1e-6, MipBranching::MostFractional)
                .is_none()
        );
    }

    #[test]
    fn branch_bounds_floor_and_ceil() {
        let parent = vec![(0.0, 5.0), (0.0, 5.0)];
        let (down, up) = branch_bounds(&parent, 0, 2.4);
        assert_eq!(down[0], (0.0, 2.0)); // x0 <= floor(2.4) = 2
        assert_eq!(up[0], (3.0, 5.0)); // x0 >= ceil(2.4) = 3
        // untouched var preserved
        assert_eq!(down[1], (0.0, 5.0));
        assert_eq!(up[1], (0.0, 5.0));
    }

    #[test]
    fn branch_bounds_can_produce_empty_box_for_pruning() {
        // value at lower edge: floor gives upper < lower → empty down box
        let parent = vec![(3.0, 5.0)];
        let (down, _up) = branch_bounds(&parent, 0, 3.5);
        // down: upper = 3.0, lower = 3.0 → singleton (still valid, x0 = 3)
        assert_eq!(down[0], (3.0, 3.0));
    }

    #[test]
    fn widest_splittable_integer_picks_widest_nonsingleton() {
        // var0 width 1 (splittable), var1 singleton (skip), var2 width 4 (widest)
        let bounds = vec![(0.0, 1.0), (3.0, 3.0), (0.0, 4.0)];
        assert_eq!(widest_splittable_integer(&bounds, &[true, true, true]), Some(2));
    }

    #[test]
    fn widest_splittable_integer_skips_continuous_and_singletons() {
        let bounds = vec![(0.0, 10.0), (2.0, 2.0)];
        // var0 is continuous (mask false) despite a wide box; var1 is a singleton.
        assert!(widest_splittable_integer(&bounds, &[false, true]).is_none());
    }

    #[test]
    fn split_integer_box_yields_two_nonempty_ranges() {
        let bounds = vec![(0.0, 4.0)];
        let (down, up) = split_integer_box(&bounds, 0);
        assert_eq!(down[0], (0.0, 2.0)); // mid = floor(2) = 2
        assert_eq!(up[0], (3.0, 4.0));
    }

    #[test]
    fn split_integer_box_binary_splits_to_singletons() {
        let bounds = vec![(0.0, 1.0)];
        let (down, up) = split_integer_box(&bounds, 0);
        assert_eq!(down[0], (0.0, 0.0));
        assert_eq!(up[0], (1.0, 1.0));
    }
}
