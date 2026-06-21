//! MIP-specific presolve: integer bound tightening via coefficient propagation.
//!
//! LP bound tightening infers implied variable bounds from linear constraints.
//! For integer variables the implied bound is additionally rounded (floor/ceil),
//! which is strictly stronger: a real-valued implied ub of 3.7 gives only `x ≤ 3`
//! for an integer variable rather than `x ≤ 3.7`.

use crate::presolve::activity::propagate_row_bounds;
use crate::problem::ConstraintType;
#[cfg(test)]
use crate::problem::LpProblem;
use crate::sparse::CscMatrix;
use crate::tolerances::ZERO_TOL;

/// Maximum number of bound-propagation rounds in the iterative tightening loop.
const MAX_PRESOLVE_ROUNDS: usize = 10;

/// Maximum number of binary variables probed per presolve invocation.
///
/// Probing runs one propagation pass for each fixing (0 and 1) of each candidate.
/// For large problems this can dominate wall time; capping at this limit keeps
/// presolve overhead bounded regardless of problem size.
const MAX_PROBE_CANDIDATES: usize = 40;

/// Summary returned by [`tighten_bounds_with_probing`].
#[derive(Debug, Clone, Default)]
pub struct PresolveSummary {
    /// Number of bound-propagation rounds completed.
    pub rounds: usize,
    /// Variables whose bounds were tightened by propagation across all rounds.
    pub tightened_by_propagation: usize,
    /// Variables fixed or tightened by probing.
    pub tightened_by_probing: usize,
}

/// Build CSR row lists from a CSC matrix.
fn build_rows(n: usize, a: &CscMatrix, m: usize) -> Vec<Vec<(usize, f64)>> {
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let row = a.row_ind[k];
            let val = a.values[k];
            if val.abs() >= ZERO_TOL {
                rows[row].push((j, val));
            }
        }
    }
    rows
}

/// One pass of bound propagation over all rows.
///
/// Updates `bounds` in-place. Returns `None` on infeasibility,
/// `Some(count)` where count is the number of variables tightened.
fn propagation_pass(
    rows: &[Vec<(usize, f64)>],
    constraint_types: &[ConstraintType],
    b: &[f64],
    bounds: &mut [(f64, f64)],
    integer_mask: &[bool],
) -> Option<usize> {
    let mut tightened = 0usize;
    for (i, row) in rows.iter().enumerate() {
        let updates =
            propagate_row_bounds(row, bounds, constraint_types[i], b[i], Some(integer_mask))?;
        for (j, new_lb, new_ub) in updates {
            if (new_lb - bounds[j].0).abs() > ZERO_TOL || (new_ub - bounds[j].1).abs() > ZERO_TOL
            {
                tightened += 1;
            }
            bounds[j] = (new_lb, new_ub);
        }
    }
    Some(tightened)
}

/// Core bound-tightening logic operating on raw linear-constraint data.
///
/// Iterates bound propagation until convergence or [`MAX_PRESOLVE_ROUNDS`].
///
/// Shared by MILP (`tighten_integer_bounds`) and MIQP.
pub(crate) fn tighten_bounds_linear(
    n: usize,
    a: &CscMatrix,
    b: &[f64],
    constraint_types: &[ConstraintType],
    bounds: &[(f64, f64)],
    integer_mask: &[bool],
) -> Option<Vec<(f64, f64)>> {
    let m = b.len();
    let rows = build_rows(n, a, m);

    let mut new_bounds = bounds.to_vec();
    for _ in 0..MAX_PRESOLVE_ROUNDS {
        let tightened =
            propagation_pass(&rows, constraint_types, b, &mut new_bounds, integer_mask)?;
        if tightened == 0 {
            break;
        }
    }
    Some(new_bounds)
}

/// Maximum propagation passes in [`tighten_bounds_at_node`].
pub(crate) const MAX_PROPAGATION_PASSES: usize = 3;

/// Multi-pass bound tightening at a B&B node.
///
/// Runs up to [`MAX_PROPAGATION_PASSES`] passes of [`tighten_bounds_linear`],
/// stopping early when no bound changed. Returns `Ok(tightened)` on success or
/// `Err(())` when a contradiction is detected (lb > ub after rounding).
pub(crate) fn tighten_bounds_at_node(
    n: usize,
    a: &CscMatrix,
    b: &[f64],
    constraint_types: &[ConstraintType],
    bounds: &[(f64, f64)],
    integer_mask: &[bool],
) -> Result<Vec<(f64, f64)>, ()> {
    let mut current = bounds.to_vec();
    for _ in 0..MAX_PROPAGATION_PASSES {
        match tighten_bounds_linear(n, a, b, constraint_types, &current, integer_mask) {
            None => return Err(()),
            Some(updated) => {
                if updated == current {
                    break;
                }
                current = updated;
            }
        }
    }
    Ok(current)
}

/// Tighten variable bounds by iterative coefficient propagation (MILP entry).
///
/// Returns `None` when infeasibility is detected. Returns the tightened bounds
/// otherwise.
#[cfg(test)]
fn tighten_integer_bounds(
    lp: &LpProblem,
    integer_mask: &[bool],
) -> Option<Vec<(f64, f64)>> {
    tighten_bounds_linear(
        lp.num_vars,
        &lp.a,
        &lp.b,
        &lp.constraint_types,
        &lp.bounds,
        integer_mask,
    )
}

/// Indices of binary variables (integer, lb=0, ub=1) among `integer_vars`.
fn binary_var_indices(integer_vars: &[usize], bounds: &[(f64, f64)]) -> Vec<usize> {
    integer_vars
        .iter()
        .copied()
        .filter(|&j| {
            j < bounds.len()
                && (bounds[j].0 - 0.0).abs() < ZERO_TOL
                && (bounds[j].1 - 1.0).abs() < ZERO_TOL
        })
        .collect()
}

/// One probing pass over `candidates` (binary variable indices).
///
/// For each candidate `j`:
/// - Fix `x_j = 0`, run propagation, record implied bounds.
/// - Fix `x_j = 1`, run propagation, record implied bounds.
/// - Bounds tightened under **both** fixings are applied unconditionally.
/// - If one fixing is infeasible, `x_j` is fixed to the opposite value.
///
/// Returns `None` on global infeasibility; otherwise returns indices of
/// variables whose bounds changed.
fn probing_pass(
    rows: &[Vec<(usize, f64)>],
    constraint_types: &[ConstraintType],
    b: &[f64],
    bounds: &mut [(f64, f64)],
    integer_mask: &[bool],
    candidates: &[usize],
) -> Option<Vec<usize>> {
    let mut changed = Vec::new();

    for &j in candidates {
        let (orig_lb, orig_ub) = bounds[j];
        if (orig_ub - orig_lb).abs() < ZERO_TOL {
            continue; // already fixed
        }

        let bounds0: Option<Vec<(f64, f64)>> = {
            let mut b0 = bounds.to_vec();
            b0[j] = (0.0, 0.0);
            propagation_pass(rows, constraint_types, b, &mut b0, integer_mask).map(|_| b0)
        };

        let bounds1: Option<Vec<(f64, f64)>> = {
            let mut b1 = bounds.to_vec();
            b1[j] = (1.0, 1.0);
            propagation_pass(rows, constraint_types, b, &mut b1, integer_mask).map(|_| b1)
        };

        match (bounds0, bounds1) {
            (None, None) => return None,
            (None, Some(_)) => {
                // x_j = 0 infeasible → fix x_j = 1.
                if (bounds[j].0 - 1.0).abs() > ZERO_TOL || (bounds[j].1 - 1.0).abs() > ZERO_TOL {
                    bounds[j] = (1.0, 1.0);
                    changed.push(j);
                }
                propagation_pass(rows, constraint_types, b, bounds, integer_mask)?;
            }
            (Some(_), None) => {
                // x_j = 1 infeasible → fix x_j = 0.
                if (bounds[j].0 - 0.0).abs() > ZERO_TOL || (bounds[j].1 - 0.0).abs() > ZERO_TOL {
                    bounds[j] = (0.0, 0.0);
                    changed.push(j);
                }
                propagation_pass(rows, constraint_types, b, bounds, integer_mask)?;
            }
            (Some(b0), Some(b1)) => {
                // Global bound tightening from probing: in any feasible solution x_j is
                // binary, so either x_j=0 (giving b0 bounds) or x_j=1 (giving b1 bounds).
                // Valid global bounds are the envelope of both cases:
                //   new_lb = min(b0.lb, b1.lb)  — weakest implied lb covers both
                //   new_ub = max(b0.ub, b1.ub)  — weakest implied ub covers both
                // These can still tighten when BOTH implied bounds are tighter than current.
                // Skip variable j itself — its bounds in b0 and b1 are forced, not implied.
                let n = bounds.len();
                for k in 0..n {
                    if k == j {
                        continue;
                    }
                    let new_lb = b0[k].0.min(b1[k].0).max(bounds[k].0);
                    let new_ub = b0[k].1.max(b1[k].1).min(bounds[k].1);
                    if new_lb > new_ub + ZERO_TOL {
                        return None;
                    }
                    if (new_lb - bounds[k].0).abs() > ZERO_TOL
                        || (new_ub - bounds[k].1).abs() > ZERO_TOL
                    {
                        bounds[k] = (new_lb, new_ub);
                        changed.push(k);
                    }
                }
            }
        }
    }

    Some(changed)
}

/// Tighten bounds via iterative propagation and probing of binary variables.
///
/// 1. Multi-pass propagation to convergence.
/// 2. Probing over binary variables; first round probes all, subsequent rounds
///    probe only variables that changed in the previous round.
/// 3. After each probing pass, re-run propagation to convergence.
/// 4. Terminate when probing produces no changes or [`MAX_PRESOLVE_ROUNDS`] exceeded.
///
/// Returns `None` on infeasibility; otherwise writes the tightened bounds into
/// `bounds` and returns a [`PresolveSummary`].
pub fn tighten_bounds_with_probing(
    a: &CscMatrix,
    b: &[f64],
    constraint_types: &[ConstraintType],
    bounds: &mut [(f64, f64)],
    integer_vars: &[usize],
) -> Option<PresolveSummary> {
    let n = bounds.len();
    let m = b.len();
    let rows = build_rows(n, a, m);

    let mut integer_mask = vec![false; n];
    for &j in integer_vars {
        if j < n {
            integer_mask[j] = true;
        }
    }

    let mut summary = PresolveSummary::default();
    let mut bounds_vec = bounds.to_vec();

    // Initial propagation to convergence (only count rounds that tightened something).
    for _ in 0..MAX_PRESOLVE_ROUNDS {
        let t = propagation_pass(&rows, constraint_types, b, &mut bounds_vec, &integer_mask)?;
        if t == 0 {
            break;
        }
        summary.tightened_by_propagation += t;
        summary.rounds += 1;
    }

    // Probing loop.
    let mut candidates = binary_var_indices(integer_vars, &bounds_vec);
    candidates.truncate(MAX_PROBE_CANDIDATES);
    for _ in 0..MAX_PRESOLVE_ROUNDS {
        if candidates.is_empty() {
            break;
        }
        let changed = probing_pass(
            &rows,
            constraint_types,
            b,
            &mut bounds_vec,
            &integer_mask,
            &candidates,
        )?;
        summary.tightened_by_probing += changed.len();

        if changed.is_empty() {
            break;
        }

        // Re-propagate after probing fixes.
        for _ in 0..MAX_PRESOLVE_ROUNDS {
            let t =
                propagation_pass(&rows, constraint_types, b, &mut bounds_vec, &integer_mask)?;
            if t == 0 {
                break;
            }
            summary.tightened_by_propagation += t;
            summary.rounds += 1;
        }

        // Next round: only re-probe binary vars whose bounds were affected.
        candidates = changed
            .into_iter()
            .filter(|&k| {
                integer_mask[k]
                    && (bounds_vec[k].0 - 0.0).abs() < ZERO_TOL
                    && (bounds_vec[k].1 - 1.0).abs() < ZERO_TOL
            })
            .collect();
    }

    bounds.copy_from_slice(&bounds_vec);
    Some(summary)
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
    /// x ≤ 3.7 ∧ x ≥ 3.5 is LP-feasible but integer rounding makes lb > ub.
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
        assert!(
            result.is_none(),
            "x ≤ 3.7 ∧ x ≥ 3.5 integer → empty domain → None"
        );
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
        assert!(
            (bounds[0].1 - 5.0).abs() < 1e-9,
            "x ub must be 5, got {}",
            bounds[0].1
        );
        assert!(
            (bounds[1].1 - 5.0).abs() < 1e-9,
            "y ub must be 5, got {}",
            bounds[1].1
        );
    }

    /// Eq constraint tightens both lb and ub.
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
        assert!(
            (bounds[0].0 - 3.0).abs() < 1e-9,
            "integer lb from -x ≤ -2.3 must be ceil(2.3)=3, got {}",
            bounds[0].0
        );
    }

    /// Tolerance-aware rounding retains integer values within float-arithmetic drift.
    #[test]
    fn tolerance_aware_floor_retains_boundary_integer() {
        let lp = single_var_lp(0.1, 0.3, ConstraintType::Le, (0.0, 5.0));
        let bounds =
            tighten_integer_bounds(&lp, &[true]).expect("feasible: x=3 satisfies 0.1*3=0.3");
        assert!(
            (bounds[0].1 - 3.0).abs() < 1e-9,
            "integer ub from 0.1*x<=0.3 must be 3 (not 2 from raw floor), got {}",
            bounds[0].1
        );
    }

    /// Finite lower bound propagates implied ub even when one variable has an
    /// infinite upper bound.
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
        assert!((bounds[0].1 - 5.0).abs() < 1e-9, "x ub: {}", bounds[0].1);
        assert!((bounds[1].1 - 5.0).abs() < 1e-9, "y ub: {}", bounds[1].1);
    }

    /// Eq row with non-integer rhs: lb and ub cross after rounding → infeasible.
    #[test]
    fn eq_non_integer_rhs_integer_var_crossed_bounds_is_infeasible() {
        let lp = single_var_lp(1.0, 3.5, ConstraintType::Eq, (0.0, 10.0));
        let result = tighten_integer_bounds(&lp, &[true]);
        assert!(
            result.is_none(),
            "x=3.5 integer → floor(3.5)=3 < ceil(3.5)=4 → infeasible"
        );
    }

    /// Infinite upper bound of a variable blocks Ge-direction propagation.
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
        assert_eq!(
            bounds[0].0, 0.0,
            "x lb must stay 0 (Ge propagation skipped, y_ub=∞)"
        );
    }

    // ── tighten_bounds_at_node tests ──────────────────────────────────────────

    /// Infeasibility at a B&B node returns Err(()).
    ///
    /// x+y<=2.9, x+y>=2.1 with node bounds x∈[1,1], y∈[1,2] integer.
    /// Pass 1, row 0 (Le): y_ub → floor(2.9-1)=1. row 1 (Ge): x_lb → ceil(2.1-1)=2 > x_ub=1.
    ///
    /// Sentinel: replacing `Err(())` return with Ok(current) silently accepts the
    /// contradiction and skips LP pruning → assertion fails.
    #[test]
    fn at_node_infeasible_returns_err() {
        let a =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2)
                .unwrap();
        let b = vec![2.9, 2.1];
        let ct = vec![ConstraintType::Le, ConstraintType::Ge];
        let mask = vec![true, true];
        let result = tighten_bounds_at_node(2, &a, &b, &ct, &[(1.0, 1.0), (1.0, 2.0)], &mask);
        assert!(
            result.is_err(),
            "x∈[1,1], y∈[1,2]: row 0 tightens y_ub to 1, row 1 forces x_lb=2>ub=1 → Err"
        );
    }

    /// Feasible node: bounds tighten and Ok is returned.
    ///
    /// x+y≤3.5 with x∈[2,5], y∈[0,5] integer.
    /// Pass 1: x_ub→floor(3.5-0)=3, y_ub→floor(3.5-2)=1.
    ///
    /// Sentinel: no-op (return Ok(bounds.to_vec()) directly) gives y_ub=5 ≠ 1 → fails.
    #[test]
    fn at_node_feasible_tightens_bounds() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![3.5];
        let ct = vec![ConstraintType::Le];
        let mask = vec![true, true];
        let bounds = tighten_bounds_at_node(2, &a, &b, &ct, &[(2.0, 5.0), (0.0, 5.0)], &mask)
            .expect("feasible");
        assert!(
            (bounds[0].1 - 3.0).abs() < 1e-9,
            "x_ub must be 3 (floor(3.5-0)=3), got {}",
            bounds[0].1
        );
        assert!(
            (bounds[1].1 - 1.0).abs() < 1e-9,
            "y_ub must be 1 (floor(3.5-2)=1), got {}",
            bounds[1].1
        );
    }

    // -----------------------------------------------------------------------
    // Multi-pass tests
    // -----------------------------------------------------------------------

    /// Multi-pass tightens a var that depends on an update from a later row.
    ///
    /// x0, x1 ∈ {0,1} integer.
    /// Row 0 (processed first):  x0 - x1 ≤ 0.3
    /// Row 1 (processed second): x1       ≤ 0
    ///
    /// Pass 1: row0 uses x1_ub=1 → implied_ub(x0) = 0.3+1=1.3 → 1 (no change).
    ///         row1 tightens x1_ub → 0.
    /// Pass 2: row0 uses x1_ub=0 → implied_ub(x0) = 0.3+0=0.3 → floor(0.3)=0. Tightened.
    #[test]
    fn multi_pass_tightens_chain_that_single_pass_misses() {
        let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[1.0, -1.0, 1.0], 2, 2)
            .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![0.3, 0.0],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, 1.0), (0.0, 1.0)],
            None,
        )
        .unwrap();
        let bounds = tighten_integer_bounds(&lp, &[true, true]).expect("feasible");
        assert!(
            (bounds[0].1 - 0.0).abs() < 1e-9,
            "multi-pass must tighten x0 ub to 0, got {}",
            bounds[0].1
        );
        assert!(
            (bounds[1].1 - 0.0).abs() < 1e-9,
            "x1 ub must be 0, got {}",
            bounds[1].1
        );
    }

    /// No-change case: returns Ok with unchanged bounds and terminates early.
    ///
    /// x≤5 with x∈[0,3] integer — no propagation possible.
    ///
    /// Sentinel: infinite loop (no early termination) would time out.
    #[test]
    fn at_node_no_change_returns_same_bounds() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let b = vec![5.0];
        let ct = vec![ConstraintType::Le];
        let mask = vec![true];
        let bounds =
            tighten_bounds_at_node(1, &a, &b, &ct, &[(0.0, 3.0)], &mask).expect("feasible");
        assert_eq!(bounds, vec![(0.0, 3.0)], "no tightening possible → unchanged");
    }

    /// Multi-pass: second pass tightens beyond what one pass achieves.
    ///
    /// Constraints: x+y≤3.5 (row 0), z+y≥2.5 (row 1); node x∈[2,5], y∈[0,5], z∈[0,5] integer.
    /// Pass 1, row 0: y_ub→floor(3.5-2)=1.
    /// Pass 1, row 1: y_ub=1 (updated), z_lb→ceil(2.5-1)=2.
    /// Pass 2: no further change → fixed-point.
    #[test]
    fn at_node_multi_pass_tightens_cascade() {
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 2, 1],
            &[1.0, 1.0, 1.0, 1.0],
            2,
            3,
        )
        .unwrap();
        let b = vec![3.5, 2.5];
        let ct = vec![ConstraintType::Le, ConstraintType::Ge];
        let mask = vec![true, true, true];
        let start = vec![(2.0, 5.0), (0.0, 5.0), (0.0, 5.0)];
        let bounds = tighten_bounds_at_node(3, &a, &b, &ct, &start, &mask).expect("feasible");
        assert!(
            (bounds[1].1 - 1.0).abs() < 1e-9,
            "y_ub must be 1 (Le row tightens with x_lb=2), got {}",
            bounds[1].1
        );
        assert!(
            (bounds[2].0 - 2.0).abs() < 1e-9,
            "z_lb must be 2 (Ge row with updated y_ub=1), got {}",
            bounds[2].0
        );
    }

    /// Multi-pass terminates without hanging.
    #[test]
    fn multi_pass_terminates() {
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            2,
            2,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![5.0, 0.0],
            vec![ConstraintType::Le, ConstraintType::Ge],
            vec![(0.0, 10.0), (0.0, 10.0)],
            None,
        )
        .unwrap();
        let bounds = tighten_integer_bounds(&lp, &[true, true]).expect("feasible");
        assert!(bounds[0].1 <= 5.0 + 1e-9);
        assert!(bounds[1].1 <= 5.0 + 1e-9);
    }

    // -----------------------------------------------------------------------
    // Probing tests
    // -----------------------------------------------------------------------

    /// Probing detects infeasibility when both fixings are infeasible.
    ///
    /// x0, x1 ∈ {0,1}. Constraint: x0 + x1 ≥ 2.5.
    /// Max sum = 2 < 2.5; both fixings lead to infeasible propagation.
    #[test]
    fn probing_detects_infeasibility_both_fixings() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let mut bounds = vec![(0.0_f64, 1.0_f64), (0.0, 1.0)];
        let result = tighten_bounds_with_probing(
            &a,
            &[2.5],
            &[ConstraintType::Ge],
            &mut bounds,
            &[0, 1],
        );
        assert!(result.is_none(), "both fixings infeasible → None");
    }

    /// Fixing one binary value is infeasible → the variable is fixed to the other.
    ///
    /// x0, x1 ∈ {0,1}. Constraint: x0 + x1 ≥ 1.5.
    /// For x0: fixing x0=0 requires x1 ≥ 1.5 → ceil=2 > x1_ub=1 → infeasible.
    /// Hence x0=1.
    #[test]
    fn probing_fixes_var_when_zero_fixing_is_infeasible() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let mut bounds = vec![(0.0_f64, 1.0_f64), (0.0, 1.0)];
        tighten_bounds_with_probing(
            &a,
            &[1.5],
            &[ConstraintType::Ge],
            &mut bounds,
            &[0, 1],
        )
        .expect("feasible");
        assert!(
            (bounds[0].0 - 1.0).abs() < 1e-9 && (bounds[0].1 - 1.0).abs() < 1e-9,
            "x0 must be fixed to 1, got [{},{}]",
            bounds[0].0,
            bounds[0].1
        );
    }

    /// x0 ∈ {0,1}. Constraint: x0 ≤ 0.5 → propagation forces x0=0.
    #[test]
    fn probing_fixes_var_when_one_fixing_is_infeasible() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let mut bounds = vec![(0.0_f64, 1.0_f64)];
        tighten_bounds_with_probing(
            &a,
            &[0.5],
            &[ConstraintType::Le],
            &mut bounds,
            &[0],
        )
        .expect("feasible");
        assert!(
            (bounds[0].0 - 0.0).abs() < 1e-9 && (bounds[0].1 - 0.0).abs() < 1e-9,
            "x0 must be fixed to 0, got [{},{}]",
            bounds[0].0,
            bounds[0].1
        );
    }

    /// Probing tightens a second variable via the envelope of implied bounds.
    ///
    /// x0 ∈ {0,1}, x1 ∈ [0,10] integer.
    /// Constraint: x0 + x1 ≤ 2.5.
    /// x0=0 implies x1 ≤ 2.5 → floor(2.5)=2.
    /// x0=1 implies x1 ≤ 1.5 → floor(1.5)=1.
    /// Envelope: new_ub = max(2, 1) = 2.
    #[test]
    fn probing_tightens_integer_var_via_envelope() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let mut bounds = vec![(0.0_f64, 1.0_f64), (0.0, 10.0)];
        let summary = tighten_bounds_with_probing(
            &a,
            &[2.5],
            &[ConstraintType::Le],
            &mut bounds,
            &[0, 1],
        )
        .expect("feasible");
        assert!(
            bounds[1].1 <= 2.0 + 1e-9,
            "probing must tighten x1 ub to 2 via envelope, got {}",
            bounds[1].1
        );
        assert!(summary.tightened_by_probing > 0 || summary.tightened_by_propagation > 0);
    }

    /// PresolveSummary is returned on a trivial no-constraint instance.
    #[test]
    fn probing_returns_summary_on_trivial_instance() {
        let a = CscMatrix::new(0, 1);
        let mut bounds = vec![(0.0_f64, 1.0_f64)];
        let summary =
            tighten_bounds_with_probing(&a, &[], &[], &mut bounds, &[0]).expect("feasible");
        assert_eq!(summary.rounds, 0, "no constraints → 0 rounds");
    }

    /// No integer variables: probing has no candidates; propagation still runs.
    #[test]
    fn probing_no_integer_vars_propagation_still_runs() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let mut bounds = vec![(0.0_f64, 5.0_f64)];
        tighten_bounds_with_probing(&a, &[3.0], &[ConstraintType::Le], &mut bounds, &[])
            .expect("feasible");
        assert!(
            (bounds[0].1 - 3.0).abs() < 1e-9,
            "propagation must tighten ub to 3.0 even with no integer vars, got {}",
            bounds[0].1
        );
    }

    /// Non-binary integer vars are not probed.
    #[test]
    fn probing_skips_non_binary_integer_vars() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let mut bounds = vec![(0.0_f64, 2.0_f64)];
        let summary =
            tighten_bounds_with_probing(&a, &[5.0], &[ConstraintType::Le], &mut bounds, &[0])
                .expect("feasible");
        assert_eq!(
            summary.tightened_by_probing, 0,
            "no probing for non-binary var (ub=2)"
        );
    }
}
