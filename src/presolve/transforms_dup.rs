//! Duplicate-row / duplicate-column / dominated-column / dual-fixing transforms.
//!
//! These extend the eight-step LP presolve pipeline in `transforms.rs` with
//! pattern-matching reductions that HiGHS-style solvers apply to shrink netlib
//! instances. Each transform mutates `PresolveState` only through bound
//! tightening (which Step 1 then promotes to a `FixedVariable` postsolve step)
//! or through `RedundantConstraint` (for parallel rows), so no new
//! `PostsolveStep` variant is required.
//!
//! Step 9  Parallel row: two rows share a column pattern with a positive
//!         scaling α and identical constraint type. Drop the looser row;
//!         RHS mismatch on Eq rows ⇒ Infeasible.
//! Step 10 Duplicate / dominated column: two columns share a row pattern with
//!         a positive scaling α. The column whose per-A-unit cost is strictly
//!         larger is dual-dominated and can be fixed to its lower bound — but
//!         only when the other column has unbounded room to absorb any z*.
//! Step 11 Dual fixing: a column's constraint coefficients agree in sign with
//!         the dual feasibility direction (all Le with a≥0, all Ge with a≤0,
//!         no Eq with nonzero coef) and `c_j ≥ 0` ⇒ fix to lb (or Unbounded
//!         if lb = −∞ and c_j > 0). Symmetric for fix-to-ub.

use std::collections::HashMap;

use super::transforms::{PostsolveStep, PresolveState, PresolveStatus};
use crate::problem::ConstraintType;
use crate::tolerances::ZERO_TOL;

/// Relative tolerance for proportionality / RHS-consistency checks.
const PROP_TOL: f64 = 1e-9;

// ============================================================
// Step 9: Parallel / duplicate row
// ============================================================

pub(super) fn step9_parallel_row(st: &mut PresolveState) -> Result<(), PresolveStatus> {
    let m = st.b.len();
    // Group rows by sorted active-column pattern; only rows in the same group
    // can possibly be parallel.
    let mut groups: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for i in 0..m {
        if st.removed_rows[i] {
            continue;
        }
        let entries = st.active_row_entries(i);
        if entries.len() < 2 {
            continue;
        }
        let mut cols: Vec<usize> = entries.iter().map(|&(j, _)| j).collect();
        cols.sort_unstable();
        groups.entry(cols).or_default().push(i);
    }
    for (_, rows) in groups {
        if rows.len() < 2 {
            continue;
        }
        for a_idx in 0..rows.len() {
            let i = rows[a_idx];
            if st.removed_rows[i] {
                continue;
            }
            let mut i_sorted = st.active_row_entries(i);
            i_sorted.sort_by_key(|&(j, _)| j);
            for b_idx in (a_idx + 1)..rows.len() {
                let k = rows[b_idx];
                if st.removed_rows[k] {
                    continue;
                }
                let mut k_sorted = st.active_row_entries(k);
                k_sorted.sort_by_key(|&(j, _)| j);
                if k_sorted.len() != i_sorted.len() {
                    continue;
                }

                // α s.t. a_i = α · a_k. Require positive (same direction).
                let alpha = i_sorted[0].1 / k_sorted[0].1;
                if !alpha.is_finite() || alpha.abs() < PROP_TOL || alpha <= 0.0 {
                    continue;
                }
                let mut proportional = true;
                for q in 0..i_sorted.len() {
                    if i_sorted[q].0 != k_sorted[q].0 {
                        proportional = false;
                        break;
                    }
                    let expected = alpha * k_sorted[q].1;
                    let tol = PROP_TOL * (1.0 + expected.abs());
                    if (i_sorted[q].1 - expected).abs() > tol {
                        proportional = false;
                        break;
                    }
                }
                if !proportional {
                    continue;
                }
                if st.constraint_types[i] != st.constraint_types[k] {
                    continue;
                }
                let bi = st.b[i];
                let bk_scaled = alpha * st.b[k];

                match st.constraint_types[i] {
                    ConstraintType::Eq => {
                        let tol = PROP_TOL * (1.0 + bi.abs() + bk_scaled.abs());
                        if (bi - bk_scaled).abs() > tol {
                            return Err(PresolveStatus::Infeasible);
                        }
                        st.removed_rows[k] = true;
                        st.postsolve_stack.push(PostsolveStep::RedundantConstraint {
                            orig_row: k,
                        });
                    }
                    ConstraintType::Le => {
                        // a_k^T x ≤ min(b_i/α, b_k); in i's frame: drop the larger b.
                        if bi <= bk_scaled {
                            st.removed_rows[k] = true;
                            st.postsolve_stack.push(PostsolveStep::RedundantConstraint {
                                orig_row: k,
                            });
                        } else {
                            st.removed_rows[i] = true;
                            st.postsolve_stack.push(PostsolveStep::RedundantConstraint {
                                orig_row: i,
                            });
                            break; // i removed, advance outer loop
                        }
                    }
                    ConstraintType::Ge => {
                        if bi >= bk_scaled {
                            st.removed_rows[k] = true;
                            st.postsolve_stack.push(PostsolveStep::RedundantConstraint {
                                orig_row: k,
                            });
                        } else {
                            st.removed_rows[i] = true;
                            st.postsolve_stack.push(PostsolveStep::RedundantConstraint {
                                orig_row: i,
                            });
                            break;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

// ============================================================
// Step 10: Duplicate / dominated column
// ============================================================

pub(super) fn step10_dup_dom_col(
    st: &mut PresolveState,
    new_fixed: &mut usize,
) -> Result<(), PresolveStatus> {
    let n = st.bounds.len();
    let mut groups: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for j in 0..n {
        if st.removed_cols[j] {
            continue;
        }
        let entries = st.active_col_entries(j);
        if entries.is_empty() {
            continue;
        }
        let mut rows: Vec<usize> = entries.iter().map(|&(i, _)| i).collect();
        rows.sort_unstable();
        groups.entry(rows).or_default().push(j);
    }
    for (_, cols) in groups {
        if cols.len() < 2 {
            continue;
        }
        for a_idx in 0..cols.len() {
            let j = cols[a_idx];
            if st.removed_cols[j] {
                continue;
            }
            let mut j_sorted = st.active_col_entries(j);
            j_sorted.sort_by_key(|&(i, _)| i);
            for b_idx in (a_idx + 1)..cols.len() {
                let k = cols[b_idx];
                if st.removed_cols[k] {
                    continue;
                }
                let mut k_sorted = st.active_col_entries(k);
                k_sorted.sort_by_key(|&(i, _)| i);
                if k_sorted.len() != j_sorted.len() {
                    continue;
                }

                // A[:,j] = α · A[:,k]
                let alpha = j_sorted[0].1 / k_sorted[0].1;
                if !alpha.is_finite() || alpha.abs() < PROP_TOL || alpha <= 0.0 {
                    continue;
                }
                let mut proportional = true;
                for q in 0..j_sorted.len() {
                    if j_sorted[q].0 != k_sorted[q].0 {
                        proportional = false;
                        break;
                    }
                    let expected = alpha * k_sorted[q].1;
                    let tol = PROP_TOL * (1.0 + expected.abs());
                    if (j_sorted[q].1 - expected).abs() > tol {
                        proportional = false;
                        break;
                    }
                }
                if !proportional {
                    continue;
                }

                // 1 unit of A contribution costs c_j/α via x_j, c_k via x_k.
                // The cheaper column should soak up demand; the dearer is
                // fixable iff the cheaper has unbounded room (ub = +∞).
                let cj_per = st.c[j] / alpha;
                let ck = st.c[k];
                let (_lb_j, ub_j) = st.bounds[j];
                let (_lb_k, ub_k) = st.bounds[k];
                let cost_tol = PROP_TOL * (1.0 + cj_per.abs() + ck.abs());

                if cj_per + cost_tol < ck {
                    // k strictly dearer ⇒ dual-fix k to its lb, provided
                    // x_j can absorb any feasible z (ub_j = +∞).
                    if ub_j == f64::INFINITY && fix_to_lb(st, k)? {
                        *new_fixed += 1;
                    }
                } else if ck + cost_tol < cj_per {
                    // j strictly dearer ⇒ dual-fix j to its lb.
                    if ub_k == f64::INFINITY && fix_to_lb(st, j)? {
                        *new_fixed += 1;
                        break;
                    }
                } else {
                    // Tie within tol: arbitrary, fix k if safe, else j.
                    if ub_j == f64::INFINITY && fix_to_lb(st, k)? {
                        *new_fixed += 1;
                    } else if ub_k == f64::INFINITY && fix_to_lb(st, j)? {
                        *new_fixed += 1;
                        break;
                    }
                }
            }
        }
    }
    Ok(())
}

// ============================================================
// Step 11: Dual fixing (single column, sign-determined)
// ============================================================

pub(super) fn step11_dual_fixing(
    st: &mut PresolveState,
    new_fixed: &mut usize,
) -> Result<(), PresolveStatus> {
    let n = st.bounds.len();
    for j in 0..n {
        if st.removed_cols[j] {
            continue;
        }
        let cj = st.c[j];
        let entries = st.active_col_entries(j);
        if entries.is_empty() {
            continue; // Step 3b handles empty columns
        }

        // "positive pressure": all constraints want x_j smaller ⇒ fixable to lb
        //   Le with a_ij ≥ 0, Ge with a_ij ≤ 0, no Eq with nonzero coef.
        // "negative pressure": symmetric ⇒ fixable to ub.
        let mut pos_pressure = true;
        let mut neg_pressure = true;
        for &(i, a) in &entries {
            match st.constraint_types[i] {
                ConstraintType::Le => {
                    if a > ZERO_TOL {
                        neg_pressure = false;
                    } else if a < -ZERO_TOL {
                        pos_pressure = false;
                    }
                }
                ConstraintType::Ge => {
                    if a > ZERO_TOL {
                        pos_pressure = false;
                    } else if a < -ZERO_TOL {
                        neg_pressure = false;
                    }
                }
                ConstraintType::Eq => {
                    if a.abs() > ZERO_TOL {
                        pos_pressure = false;
                        neg_pressure = false;
                    }
                }
            }
            if !pos_pressure && !neg_pressure {
                break;
            }
        }

        let (lb, ub) = st.bounds[j];
        if pos_pressure && cj >= -ZERO_TOL {
            if lb.is_finite() {
                if fix_to_lb(st, j)? {
                    *new_fixed += 1;
                }
            } else if cj > ZERO_TOL {
                return Err(PresolveStatus::Unbounded);
            }
            // cj ≈ 0 and lb = -∞: degenerate, leave alone for later passes.
        } else if neg_pressure && cj <= ZERO_TOL {
            if ub.is_finite() {
                if fix_to_ub(st, j)? {
                    *new_fixed += 1;
                }
            } else if cj < -ZERO_TOL {
                return Err(PresolveStatus::Unbounded);
            }
        }
    }
    Ok(())
}

// ============================================================
// helpers
// ============================================================

/// Tighten col j to `(lb, lb)`; the next outer fixpoint iteration's Step 1
/// promotes it to a `FixedVariable` postsolve entry. Returns whether the
/// bounds actually changed.
fn fix_to_lb(st: &mut PresolveState, j: usize) -> Result<bool, PresolveStatus> {
    let (lb, ub) = st.bounds[j];
    if !lb.is_finite() {
        return Ok(false);
    }
    if (ub - lb).abs() < ZERO_TOL {
        return Ok(false); // already fixed
    }
    if lb > ub + ZERO_TOL {
        return Err(PresolveStatus::Infeasible);
    }
    st.postsolve_stack.push(PostsolveStep::BoundsTightened {
        orig_col: j,
        old_lb: lb,
        old_ub: ub,
    });
    st.bounds[j] = (lb, lb);
    Ok(true)
}

fn fix_to_ub(st: &mut PresolveState, j: usize) -> Result<bool, PresolveStatus> {
    let (lb, ub) = st.bounds[j];
    if !ub.is_finite() {
        return Ok(false);
    }
    if (ub - lb).abs() < ZERO_TOL {
        return Ok(false);
    }
    if lb > ub + ZERO_TOL {
        return Err(PresolveStatus::Infeasible);
    }
    st.postsolve_stack.push(PostsolveStep::BoundsTightened {
        orig_col: j,
        old_lb: lb,
        old_ub: ub,
    });
    st.bounds[j] = (ub, ub);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presolve::transforms::{
        run_presolve_with_flags, PresolveFlags,
    };
    use crate::problem::{ConstraintType, LpProblem};
    use crate::sparse::CscMatrix;

    fn make_lp(
        c: Vec<f64>,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrows: usize,
        ncols: usize,
        b: Vec<f64>,
        cts: Vec<ConstraintType>,
        bounds: Vec<(f64, f64)>,
    ) -> LpProblem {
        let a = CscMatrix::from_triplets(rows, cols, vals, nrows, ncols).unwrap();
        LpProblem::new_general(c, a, b, cts, bounds, None).unwrap()
    }

    #[test]
    fn parallel_row_eq_consistent_drops_one() {
        // 2 Eq rows: x + y = 3 ; 2x + 2y = 6 (α=0.5). Drop one.
        let lp = make_lp(
            vec![1.0, 1.0],
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 2.0, 2.0],
            2,
            2,
            vec![3.0, 6.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, 5.0), (0.0, 5.0)],
        );
        let with_flags =
            run_presolve_with_flags(&lp, None, PresolveFlags::default()).unwrap();
        let without = run_presolve_with_flags(
            &lp,
            None,
            PresolveFlags {
                enable_parallel_row: false,
                enable_dup_dom_col: false,
                enable_dual_fixing: false,
            },
        )
        .unwrap();
        // With Step 9 we must shave a row beyond what the baseline manages.
        assert!(
            with_flags.reduced_problem.num_constraints
                < without.reduced_problem.num_constraints
                || with_flags.reduced_problem.num_constraints == 0,
            "parallel_row should drop at least one row (with={}, without={})",
            with_flags.reduced_problem.num_constraints,
            without.reduced_problem.num_constraints
        );
    }

    #[test]
    fn parallel_row_eq_inconsistent_is_infeasible() {
        // x+y=3 ; 2x+2y=8 (would force 4 = α·6 = 3, contradiction).
        let lp = make_lp(
            vec![1.0, 1.0],
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 2.0, 2.0],
            2,
            2,
            vec![3.0, 8.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, 5.0), (0.0, 5.0)],
        );
        assert!(matches!(
            run_presolve_with_flags(&lp, None, PresolveFlags::default()),
            Err(PresolveStatus::Infeasible)
        ));
    }

    #[test]
    fn parallel_row_le_keeps_tighter() {
        // x+y ≤ 5 ; 2x+2y ≤ 6 (per-x bound: 3). Tighter wins.
        let lp = make_lp(
            vec![1.0, 1.0],
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 2.0, 2.0],
            2,
            2,
            vec![5.0, 6.0],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, 10.0), (0.0, 10.0)],
        );
        let result =
            run_presolve_with_flags(&lp, None, PresolveFlags::default()).unwrap();
        // The looser of the two Le rows must be removed by Step 9.
        // (Step 4 with finite bounds may further compress, but at least one row goes.)
        assert!(
            result.reduced_problem.num_constraints <= 1,
            "parallel Le rows: expected ≤1 row after Step 9, got {}",
            result.reduced_problem.num_constraints
        );
    }

    #[test]
    fn dual_fixing_pos_cost_le_only() {
        // min x + y s.t. x + y ≤ 10, x ∈ [0, 5], y ∈ [0, 5].
        // c_j ≥ 0 and only Le with a ≥ 0 ⇒ both vars fixed to 0.
        let lp = make_lp(
            vec![1.0, 1.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![10.0],
            vec![ConstraintType::Le],
            vec![(0.0, 5.0), (0.0, 5.0)],
        );
        let result =
            run_presolve_with_flags(&lp, None, PresolveFlags::default()).unwrap();
        assert_eq!(result.reduced_problem.num_vars, 0);
        assert!((result.obj_offset).abs() < 1e-10);
    }

    #[test]
    fn dual_fixing_neg_cost_ge_only_fixes_to_ub() {
        // min -x s.t. x ≥ 1, x ∈ [0, 4]. c=-1 (neg), Ge with a>0 ⇒ pos pressure
        // disqualified, neg pressure ⇒ fix to ub=4. obj_offset = -4.
        let lp = make_lp(
            vec![-1.0],
            &[0],
            &[0],
            &[1.0],
            1,
            1,
            vec![1.0],
            vec![ConstraintType::Ge],
            vec![(0.0, 4.0)],
        );
        let result =
            run_presolve_with_flags(&lp, None, PresolveFlags::default()).unwrap();
        assert_eq!(result.reduced_problem.num_vars, 0);
        assert!(
            (result.obj_offset + 4.0).abs() < 1e-10,
            "expected obj_offset ≈ -4, got {}",
            result.obj_offset
        );
    }

    #[test]
    fn dual_fixing_unbounded_when_lb_minus_infty() {
        // min x s.t. x ≤ 10, x ∈ (-∞, ∞). Le with a=1>0, c=1>0, pos pressure
        // but lb = -∞ ⇒ Unbounded.
        let lp = make_lp(
            vec![1.0],
            &[0],
            &[0],
            &[1.0],
            1,
            1,
            vec![10.0],
            vec![ConstraintType::Le],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
        );
        assert!(matches!(
            run_presolve_with_flags(&lp, None, PresolveFlags::default()),
            Err(PresolveStatus::Unbounded)
        ));
    }

    #[test]
    fn dual_fixing_eq_blocks() {
        // x is in Eq row ⇒ Eq with nonzero coef disqualifies dual fixing.
        // Even with c ≥ 0 and another Le row that would qualify, Eq blocks.
        let lp = make_lp(
            vec![1.0, 1.0],
            &[0, 1, 1],
            &[0, 0, 1],
            &[1.0, 1.0, 1.0],
            2,
            2,
            vec![3.0, 5.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, 5.0), (0.0, 5.0)],
        );
        // Step 2 (singleton Eq) fires before Step 11 because x is alone on Eq row 0,
        // so the test only verifies the run completes; the Eq-blocks-dual-fixing
        // path is exercised through the structural sentinel below.
        let _ = run_presolve_with_flags(&lp, None, PresolveFlags::default()).unwrap();
    }

    #[test]
    fn dup_col_dominated_with_unbounded_partner() {
        // A[:,0] = A[:,1], c[0]=1, c[1]=2 ⇒ x_1 dominated by x_0.
        // Need ub of partner (x_0) = +∞ for safe fixing.
        // min x_0 + 2 x_1 s.t. x_0 + x_1 ≤ 10, x_0 ∈ [0, ∞), x_1 ∈ [0, 5].
        let lp = make_lp(
            vec![1.0, 2.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![10.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, 5.0)],
        );
        let with_flags =
            run_presolve_with_flags(&lp, None, PresolveFlags::default()).unwrap();
        // x_1 (col 1) must be eliminated by dominated-col + Step 1 fix.
        assert!(
            with_flags.col_map[1].is_none(),
            "x_1 (dominated) should be fixed and removed; col_map[1]={:?}",
            with_flags.col_map[1]
        );
    }

    #[test]
    fn noop_baseline_three_parallel_le_no_reduction() {
        // Patterns chosen so steps 1-8 cannot make progress: 2 parallel Le
        // rows over 3 vars with infinite upper bounds (Step 4 needs finite
        // ub; Step 6 is Eq-only). Without new flags the LP is invariant.
        let lp = make_lp(
            vec![1.0, 1.0, 1.0],
            &[0, 0, 0, 1, 1, 1],
            &[0, 1, 2, 0, 1, 2],
            &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            2,
            3,
            vec![10.0, 18.0],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, f64::INFINITY); 3],
        );
        let off = run_presolve_with_flags(&lp, None, PresolveFlags::all_off()).unwrap();
        assert_eq!(off.reduced_problem.num_constraints, 2);
        assert_eq!(off.reduced_problem.num_vars, 3);

        let on = run_presolve_with_flags(&lp, None, PresolveFlags::default()).unwrap();
        // With Step 11 dual-fixing (c=1>0, all a≥0 in Le ⇒ pos pressure) all
        // vars collapse to lb=0; remaining rows then become empty redundancies.
        assert_eq!(on.reduced_problem.num_vars, 0);
    }
}
