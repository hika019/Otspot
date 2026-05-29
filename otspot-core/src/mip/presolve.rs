//! MIP-specific presolve: integer bound tightening via coefficient propagation.
//!
//! LP bound tightening infers implied variable bounds from linear constraints.
//! For integer variables the implied bound is additionally rounded (floor/ceil),
//! which is strictly stronger: a real-valued implied ub of 3.7 gives only `x ≤ 3`
//! for an integer variable rather than `x ≤ 3.7`.

use crate::presolve::activity::propagate_row_bounds;
use crate::problem::LpProblem;
use crate::tolerances::ZERO_TOL;

/// Tighten variable bounds by one pass of coefficient propagation.
///
/// For each constraint row and each variable in that row the implied bound is
/// derived from the activity of the remaining variables. Integer variables are
/// additionally rounded (floor for ub, ceil for lb).
///
/// Returns `None` when infeasibility is detected (implied lb > existing ub or
/// implied ub < existing lb after rounding). Returns the tightened bounds
/// otherwise.
pub(crate) fn tighten_integer_bounds(
    lp: &LpProblem,
    integer_mask: &[bool],
) -> Option<Vec<(f64, f64)>> {
    let n = lp.num_vars;
    let m = lp.num_constraints;

    // Build CSR row index from the CSC matrix (one pass, filtering near-zero entries).
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for j in 0..n {
        for k in lp.a.col_ptr[j]..lp.a.col_ptr[j + 1] {
            let row = lp.a.row_ind[k];
            let val = lp.a.values[k];
            if val.abs() >= ZERO_TOL {
                rows[row].push((j, val));
            }
        }
    }

    let mut bounds = lp.bounds.clone();

    for i in 0..m {
        let updates = propagate_row_bounds(
            &rows[i],
            &bounds,
            lp.constraint_types[i],
            lp.b[i],
            Some(integer_mask),
        )?;
        for (j, new_lb, new_ub) in updates {
            bounds[j] = (new_lb, new_ub);
        }
    }

    Some(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{ConstraintType, LpProblem};
    use crate::sparse::CscMatrix;

    fn single_var_lp(a_val: f64, b_val: f64, ct: ConstraintType, domain: (f64, f64)) -> LpProblem {
        let a = CscMatrix::from_triplets(&[0], &[0], &[a_val], 1, 1).unwrap();
        LpProblem::new_general(vec![1.0], a, vec![b_val], vec![ct], vec![domain], None).unwrap()
    }

    /// Floor-rounding fires for integer ub from Le constraint.
    ///
    /// Sentinel: removing `floor()` leaves ub = 3.7 instead of 3.0 → assertion fails.
    #[test]
    fn integer_ub_is_floored_from_le_constraint() {
        let lp = single_var_lp(1.0, 3.7, ConstraintType::Le, (0.0, 10.0));
        let bounds = tighten_integer_bounds(&lp, &[true]).expect("feasible");
        assert!(
            (bounds[0].1 - 3.0).abs() < 1e-9,
            "integer ub from x ≤ 3.7 must be floor(3.7)=3, got {}",
            bounds[0].1
        );
    }

    /// Ceil-rounding fires for integer lb from Ge constraint.
    ///
    /// Sentinel: removing `ceil()` leaves lb = 2.3 instead of 3.0 → assertion fails.
    #[test]
    fn integer_lb_is_ceiled_from_ge_constraint() {
        let lp = single_var_lp(1.0, 2.3, ConstraintType::Ge, (0.0, 10.0));
        let bounds = tighten_integer_bounds(&lp, &[true]).expect("feasible");
        assert!(
            (bounds[0].0 - 3.0).abs() < 1e-9,
            "integer lb from x ≥ 2.3 must be ceil(2.3)=3, got {}",
            bounds[0].0
        );
    }

    /// Infeasibility detected when integer rounding produces an empty domain.
    ///
    /// x ≤ 3.7 ∧ x ≥ 3.5 is LP-feasible (domain [3.5, 3.7]).
    /// Integer rounding: floor(3.7)=3 (ub), ceil(3.5)=4 (lb) → lb=4 > ub=3 → None.
    ///
    /// Sentinel: without floor/ceil the bounds stay [3.5, 3.7] and None is not
    /// returned. The LP-feasibility of rhs=[3.7, 3.5] ensures integer rounding is
    /// required to detect infeasibility (rhs=[3.7, 4.0] would fail without rounding).
    #[test]
    fn infeasibility_detected_by_integer_rounding() {
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![3.7, 3.5],
            vec![ConstraintType::Le, ConstraintType::Ge],
            vec![(0.0, 10.0)],
            None,
        )
        .unwrap();
        let result = tighten_integer_bounds(&lp, &[true]);
        assert!(result.is_none(), "x ≤ 3.7 ∧ x ≥ 3.5 integer → empty domain → None");
    }

    /// Continuous variables are not rounded.
    #[test]
    fn continuous_var_not_rounded() {
        let lp = single_var_lp(1.0, 3.7, ConstraintType::Le, (0.0, 10.0));
        let bounds = tighten_integer_bounds(&lp, &[false]).expect("feasible");
        assert!(
            (bounds[0].1 - 3.7).abs() < 1e-9,
            "continuous var ub must stay 3.7, got {}",
            bounds[0].1
        );
    }

    /// No constraints → bounds unchanged.
    #[test]
    fn no_constraints_unchanged() {
        let a = CscMatrix::new(0, 2);
        let lp =
            LpProblem::new_general(vec![1.0, 1.0], a, vec![], vec![], vec![(0.0, 5.0); 2], None)
                .unwrap();
        let bounds = tighten_integer_bounds(&lp, &[true, true]).expect("feasible");
        assert_eq!(bounds, vec![(0.0, 5.0), (0.0, 5.0)]);
    }

    /// Two-variable instance: coefficient propagation tightens both vars.
    ///
    /// x + y ≤ 5, x,y ∈ [0,10] integer.
    /// At k=0 (j=x): rest_lb=0 (y's min contrib), implied_ub = 5 → floor(5)=5 → x ≤ 5.
    /// At k=1 (j=y): rest_lb=0 (x's min after tightening is still 0),
    ///               implied_ub = 5 → floor(5)=5 → y ≤ 5.
    #[test]
    fn two_var_le_tightens_both_ubs() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 10.0), (0.0, 10.0)],
            None,
        )
        .unwrap();
        let bounds = tighten_integer_bounds(&lp, &[true, true]).expect("feasible");
        assert!((bounds[0].1 - 5.0).abs() < 1e-9, "x ub must be 5, got {}", bounds[0].1);
        assert!((bounds[1].1 - 5.0).abs() < 1e-9, "y ub must be 5, got {}", bounds[1].1);
    }

    /// Eq constraint tightens both lb and ub.
    ///
    /// x = 4 (Eq, one-var), integer, domain [0,10] → both bounds → 4.
    #[test]
    fn eq_constraint_fixes_integer_var() {
        let lp = single_var_lp(1.0, 4.0, ConstraintType::Eq, (0.0, 10.0));
        let bounds = tighten_integer_bounds(&lp, &[true]).expect("feasible");
        assert!((bounds[0].0 - 4.0).abs() < 1e-9, "lb={}", bounds[0].0);
        assert!((bounds[0].1 - 4.0).abs() < 1e-9, "ub={}", bounds[0].1);
    }

    /// Negative coefficient: x ≥ 2.3 expressed as -x ≤ -2.3. Integer lb → ceil(2.3)=3.
    #[test]
    fn negative_coeff_le_tightens_lb() {
        let lp = single_var_lp(-1.0, -2.3, ConstraintType::Le, (0.0, 10.0));
        let bounds = tighten_integer_bounds(&lp, &[true]).expect("feasible");
        // a_ij = -1 < 0, rest_lb_fin = true (no other vars)
        // implied_lb = (-2.3 - 0) / (-1) = 2.3, ceil(2.3) = 3
        assert!(
            (bounds[0].0 - 3.0).abs() < 1e-9,
            "integer lb from -x ≤ -2.3 must be ceil(2.3)=3, got {}",
            bounds[0].0
        );
    }

    /// Finite lower bound propagates implied ub even when one variable has an
    /// infinite upper bound.
    ///
    /// x + y ≤ 5, y ∈ [0, ∞), x ∈ [0, 10] integer.
    /// For x (Le, a=1): rest_lb uses y's lb=0 (finite) → implied_ub=5 → floor(5)=5.
    /// For y (Le, a=1): rest_lb uses x's lb=0 (finite) → implied_ub=5 (continuous, no floor).
    /// y's infinite ub does not block the Le-direction propagation (Le uses rest_lb).
    #[test]
    fn finite_lower_bound_propagates_with_infinite_upper_bound() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 10.0), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let bounds = tighten_integer_bounds(&lp, &[true, false]).expect("feasible");
        // x: rest_lb=y*1*0=0 (finite), implied_ub=5→floor(5)=5
        assert!((bounds[0].1 - 5.0).abs() < 1e-9, "x ub: {}", bounds[0].1);
        // y: rest_lb=x*1*0=0 (finite), implied_ub=5 (continuous, no floor)
        assert!((bounds[1].1 - 5.0).abs() < 1e-9, "y ub: {}", bounds[1].1);
    }

    /// Infinite upper bound of a variable blocks Ge-direction propagation.
    ///
    /// x + y ≥ 3, y ∈ [0, ∞), x ∈ [0, 10] integer.
    /// For x (Ge, a=1): rest_ub uses y's contribution: a_y=1, ub_y=∞ → rest_ub infinite
    ///                  → rest_ub_fin=false → propagation skipped → x lb stays 0.
    ///
    /// Sentinel: removing the `rest_ub_fin` guard propagates with an invalid rest_ub
    /// and would tighten x lb incorrectly.
    #[test]
    fn infinite_upper_skips_ge_propagation() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![3.0],
            vec![ConstraintType::Ge],
            vec![(0.0, 10.0), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let bounds = tighten_integer_bounds(&lp, &[true, false]).expect("feasible");
        // x: rest_ub infinite (y_ub=∞) → skip → x lb stays 0
        assert_eq!(bounds[0].0, 0.0, "x lb must stay 0 (Ge propagation skipped, y_ub=∞)");
    }
}
