//! Feasibility pump heuristic for MILP (Fischetti, Glover & Lodi, 2005).
//!
//! Generates an initial integer-feasible incumbent before branch-and-bound.
//! Starting from the LP relaxation solution, the algorithm alternates between
//! rounding integer variables and solving an LP whose objective drives the
//! continuous solution toward the rounded target. Convergence (x_lp ≈ x_int)
//! yields a feasible integer point.

use crate::mip::branch::{fractionality, is_integer_feasible};
use crate::lp::solve_lp_with;
use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolveStatus, SolverResult};

/// Maximum number of FP projection iterations before giving up.
const MAX_FP_ITER: usize = 30;

/// Consecutive iterations with an unchanged rounded solution trigger perturbation.
const STALL_THRESHOLD: usize = 5;

/// Number of most-fractional integer variables to flip on perturbation.
const PERTURB_FLIP_COUNT: usize = 3;

/// Run the feasibility pump heuristic on a MILP LP relaxation.
///
/// Returns an integer-feasible `SolverResult` (objective evaluated under the
/// original `lp.c`) if the pump converges within [`MAX_FP_ITER`] iterations,
/// or `None` on failure. A `None` return is benign — the caller proceeds with
/// pure branch-and-bound.
pub(crate) fn run_feasibility_pump(
    lp: &LpProblem,
    integer_vars: &[usize],
    integer_feas_tol: f64,
    opts: &SolverOptions,
) -> Option<SolverResult> {
    if integer_vars.is_empty() {
        return None;
    }

    let n = lp.num_vars;
    let mask = build_mask(n, integer_vars);

    let root = solve_lp_with(lp, opts);
    if !matches!(root.status, SolveStatus::Optimal) || root.solution.is_empty() {
        return None;
    }

    if is_integer_feasible(&root.solution, &mask, integer_feas_tol) {
        return Some(make_result(&lp.c, root.solution));
    }

    let mut x_lp = root.solution;
    let mut x_int = round_integer_vars(&x_lp, &mask);
    let mut prev_x_int: Option<Vec<f64>> = None;
    let mut stall_count = 0usize;

    for _ in 0..MAX_FP_ITER {
        let fp_cost = signed_fp_cost(&x_lp, &x_int, &mask, n);
        let mut fp_lp = lp.clone();
        fp_lp.c = fp_cost;

        let fp_res = solve_lp_with(&fp_lp, opts);
        if !matches!(fp_res.status, SolveStatus::Optimal) || fp_res.solution.is_empty() {
            break;
        }

        x_lp = fp_res.solution;

        if is_integer_feasible(&x_lp, &mask, integer_feas_tol) {
            return Some(make_result(&lp.c, x_lp));
        }

        let new_x_int = round_integer_vars(&x_lp, &mask);

        let stalled = prev_x_int.as_ref().is_some_and(|p| integers_same(p, &new_x_int, &mask));
        stall_count = if stalled { stall_count + 1 } else { 0 };

        x_int = if stall_count >= STALL_THRESHOLD {
            stall_count = 0;
            perturb(&new_x_int, &x_lp, &mask, PERTURB_FLIP_COUNT)
        } else {
            new_x_int.clone()
        };
        prev_x_int = Some(new_x_int);
    }

    None
}

/// Build a boolean integer mask.
fn build_mask(n: usize, integer_vars: &[usize]) -> Vec<bool> {
    let mut mask = vec![false; n];
    for &j in integer_vars {
        if j < n {
            mask[j] = true;
        }
    }
    mask
}

/// Round integer-constrained components to the nearest integer; leave others unchanged.
fn round_integer_vars(x: &[f64], mask: &[bool]) -> Vec<f64> {
    x.iter()
        .zip(mask.iter())
        .map(|(&xi, &is_int)| if is_int { xi.round() } else { xi })
        .collect()
}

/// Build the signed FP objective coefficient vector.
///
/// For integer variable `j`:
/// - `+1` if `x_lp[j] > x_int[j]`: minimising pushes `x_j` down toward `x_int[j]`
/// - `-1` if `x_lp[j] < x_int[j]`: minimising pushes `x_j` up toward `x_int[j]`
/// - `0` if already equal
///
/// Continuous variables receive `0`.
fn signed_fp_cost(x_lp: &[f64], x_int: &[f64], mask: &[bool], n: usize) -> Vec<f64> {
    let mut cost = vec![0.0; n];
    for j in 0..n {
        if !mask[j] {
            continue;
        }
        let diff = x_lp[j] - x_int[j];
        cost[j] = if diff > 0.0 { 1.0 } else if diff < 0.0 { -1.0 } else { 0.0 };
    }
    cost
}

/// True when the integer components of `a` and `b` are the same rounded value.
fn integers_same(a: &[f64], b: &[f64], mask: &[bool]) -> bool {
    a.iter().zip(b.iter()).zip(mask.iter()).all(|((&ai, &bi), &is_int)| {
        !is_int || (ai - bi).abs() < 0.5
    })
}

/// Perturb `x_int` by flipping the rounding direction of the `flip_count`
/// most-fractional integer variables. Returns the perturbed rounded point.
fn perturb(x_int: &[f64], x_lp: &[f64], mask: &[bool], flip_count: usize) -> Vec<f64> {
    let mut frac_vars: Vec<(usize, f64)> = mask
        .iter()
        .enumerate()
        .filter(|(_, &is_int)| is_int)
        .map(|(j, _)| (j, fractionality(x_lp[j])))
        .collect();
    frac_vars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = x_int.to_vec();
    for &(j, _) in frac_vars.iter().take(flip_count) {
        let floor_val = x_lp[j].floor();
        let ceil_val = x_lp[j].ceil();
        result[j] = if (result[j] - floor_val).abs() < 0.5 { ceil_val } else { floor_val };
    }
    result
}

/// Build a `SolverResult` from a feasible integer solution and the original objective.
fn make_result(c: &[f64], x: Vec<f64>) -> SolverResult {
    let obj: f64 = c.iter().zip(x.iter()).map(|(ci, xi)| ci * xi).sum();
    SolverResult { status: SolveStatus::Optimal, objective: obj, solution: x, ..SolverResult::default() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SolverOptions;
    use crate::problem::{ConstraintType, LpProblem};
    use crate::sparse::CscMatrix;

    fn opts() -> SolverOptions {
        SolverOptions { timeout_secs: Some(10.0), ..Default::default() }
    }

    fn single_constraint_lp(c: Vec<f64>, a_vals: &[f64], b: f64, bounds: Vec<(f64, f64)>) -> LpProblem {
        let n = c.len();
        let rows: Vec<usize> = vec![0; n];
        let cols: Vec<usize> = (0..n).collect();
        let a = CscMatrix::from_triplets(&rows, &cols, a_vals, 1, n).unwrap();
        LpProblem::new_general(c, a, vec![b], vec![ConstraintType::Le], bounds, None).unwrap()
    }

    /// FP on a pure-LP problem (no integer vars) returns None.
    #[test]
    fn fp_skips_empty_integer_vars() {
        let lp = single_constraint_lp(vec![1.0, 1.0], &[1.0, 1.0], 5.0, vec![(0.0, 3.0); 2]);
        assert!(run_feasibility_pump(&lp, &[], 1e-6, &opts()).is_none());
    }

    /// FP returns the LP solution directly when it is already integer feasible.
    #[test]
    fn fp_returns_integral_lp_root_immediately() {
        // min x s.t. x <= 3, x in [0,5] integer. LP root: x=0 (integer).
        let lp = single_constraint_lp(vec![1.0], &[1.0], 3.0, vec![(0.0, 5.0)]);
        let r = run_feasibility_pump(&lp, &[0], 1e-6, &opts()).expect("integer root → Some");
        assert!((r.solution[0] - 0.0).abs() < 1e-6, "sol={}", r.solution[0]);
    }

    /// FP converges on a 4-variable binary knapsack in one iteration.
    ///
    /// LP relaxation: x=(0,1,0,0.5), fractional at x3.
    /// Round: x_int=(0,1,0,1) (infeasible but target).
    /// FP LP (maximise x3): x=(1,0,0,1), integer feasible.
    ///
    /// Sentinel: removing the FP loop causes `run_feasibility_pump` to return `None`
    /// → assertion fails.
    #[test]
    fn fp_converges_binary_knapsack_one_iter() {
        // min -(3x0+5x1+2x2+4x3) s.t. 3x0+5x1+2x2+4x3 <= 7, x in {0,1}^4
        let lp = single_constraint_lp(
            vec![-3.0, -5.0, -2.0, -4.0],
            &[3.0, 5.0, 2.0, 4.0],
            7.0,
            vec![(0.0, 1.0); 4],
        );
        let r = run_feasibility_pump(&lp, &[0, 1, 2, 3], 1e-6, &opts())
            .expect("FP must converge");
        // Solution must be integer feasible.
        let frac: f64 = r.solution.iter().map(|&v| (v - v.round()).abs()).sum();
        assert!(frac < 1e-6, "solution not integer: {:?}", r.solution);
        // Objective is computed under original c.
        let obj_recheck: f64 = [-3.0f64, -5.0, -2.0, -4.0].iter().zip(r.solution.iter()).map(|(c, x)| c * x).sum();
        assert!((r.objective - obj_recheck).abs() < 1e-6);
    }

    /// Perturbation is applied after STALL_THRESHOLD consecutive identical roundings.
    ///
    /// Uses a 1-variable problem where the LP always returns the same fractional
    /// value (0.5). After STALL_THRESHOLD iterations, the perturb() call flips
    /// the rounding, potentially breaking the cycle.
    ///
    /// This test verifies the perturbation path is reachable (code coverage)
    /// and doesn't panic. No-op proof: removing the stall counter keeps stall_count=0
    /// and perturbation never fires; the stall loop still produces None (not wrong)
    /// but this test serves as a coverage guard rather than a correctness sentinel.
    #[test]
    fn fp_stall_perturbation_does_not_panic() {
        // x in [0.4, 0.6] integer — LP always returns 0.5, rounds to 0 or 1, never feasible.
        // FP should exhaust MAX_FP_ITER and return None.
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, -1.0], 2, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![0.6, -0.4],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, 1.0)],
            None,
        ).unwrap();
        // This problem has no integer feasible solution (0.4 < x < 0.6 forces non-integer).
        let result = run_feasibility_pump(&lp, &[0], 1e-6, &opts());
        // FP should return None (no integer feasible solution found).
        assert!(result.is_none(), "expected None for integer-infeasible problem, got Some");
    }
}
