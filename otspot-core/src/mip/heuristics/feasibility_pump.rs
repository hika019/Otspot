//! Feasibility pump heuristic for MILP (Fischetti, Glover & Lodi, 2005).
//!
//! Generates an initial integer-feasible incumbent before branch-and-bound.
//! Starting from the LP relaxation solution, the algorithm alternates between
//! rounding integer variables and solving an LP whose objective drives the
//! continuous solution toward the rounded target. Convergence (x_lp ≈ x_int)
//! yields a feasible integer point.

use crate::lp::solve_lp_with;
use crate::mip::branch::{fractionality, is_integer_feasible};
use crate::mip::integer_mask;
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::sparse::CscMatrix;
use crate::tolerances::PIVOT_TOL;

/// Maximum number of FP projection iterations before giving up.
const MAX_FP_ITER: usize = 30;

/// Consecutive iterations with an unchanged rounded solution trigger perturbation.
const STALL_THRESHOLD: usize = 5;

/// Number of most-fractional integer variables to flip on perturbation.
const PERTURB_FLIP_COUNT: usize = 3;

/// Threshold for deciding which rounding direction a variable is closest to.
/// Values within this distance of their floor are considered "rounded down".
const HALF_INTEGER_THRESHOLD: f64 = 0.5;

/// Run the feasibility pump heuristic on a MILP LP relaxation.
///
/// Returns an integer-feasible `SolverResult` (objective evaluated under the
/// original `lp.c`) if the pump converges within [`MAX_FP_ITER`] iterations,
/// or `None` on failure. A `None` return is benign — the caller proceeds with
/// pure branch-and-bound.
///
/// Before accepting a rounded candidate as an incumbent the solution is validated
/// against the original variable bounds and linear constraints. A rounded value
/// can violate a bound when the LP solution sits within `integer_feas_tol` of an
/// integer that lies outside the feasible region (e.g. UB = 0.9999999, rounded
/// to 1.0). Seeding B&B with such a point would yield an out-of-bounds optimal.
///
/// Sentinel: removing the `integer_vars.is_empty()` guard causes a vacuously
/// integer-feasible LP root to be returned for pure-LP problems → callers
/// observe a spurious incumbent (`fp_incumbent_found = true`) → FAILS.
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
    let mask = integer_mask(n, integer_vars);

    let root = solve_lp_with(lp, opts);
    if !matches!(root.status, SolveStatus::Optimal) || root.solution.is_empty() {
        return None;
    }

    if is_integer_feasible(&root.solution, &mask, integer_feas_tol) {
        let x_rounded = round_integer_vars(&root.solution, &mask);
        if validate_against_bounds(&x_rounded, &lp.bounds)
            && validate_against_constraints(&x_rounded, &lp.a, &lp.b, &lp.constraint_types)
        {
            return Some(make_result(&lp.c, x_rounded));
        }
        return None;
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
            let x_rounded = round_integer_vars(&x_lp, &mask);
            if validate_against_bounds(&x_rounded, &lp.bounds)
                && validate_against_constraints(&x_rounded, &lp.a, &lp.b, &lp.constraint_types)
            {
                return Some(make_result(&lp.c, x_rounded));
            }
            break;
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
        !is_int || (ai - bi).abs() < HALF_INTEGER_THRESHOLD
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
        result[j] = if (result[j] - floor_val).abs() < HALF_INTEGER_THRESHOLD {
            ceil_val
        } else {
            floor_val
        };
    }
    result
}

/// Returns `true` if every component of `x` satisfies its variable bound.
///
/// Checks `lb ≤ x_j ≤ ub` exactly. No tolerance is applied because `x` is a
/// rounded integer solution — a bound violation here reflects a genuine
/// infeasibility, not floating-point noise.
fn validate_against_bounds(x: &[f64], bounds: &[(f64, f64)]) -> bool {
    x.iter().zip(bounds.iter()).all(|(&xi, &(lb, ub))| lb <= xi && xi <= ub)
}

/// Returns `true` if `x` satisfies all linear constraints within [`PIVOT_TOL`].
///
/// Computes `A x` and checks each row against `b` per [`ConstraintType`].
/// Returns `false` on dimension mismatch.
fn validate_against_constraints(
    x: &[f64],
    a: &CscMatrix,
    b: &[f64],
    constraint_types: &[ConstraintType],
) -> bool {
    let ax = match a.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return false,
    };
    ax.iter().zip(b.iter()).zip(constraint_types.iter()).all(|((&ax_i, &b_i), &ct)| match ct {
        ConstraintType::Le => ax_i <= b_i + PIVOT_TOL,
        ConstraintType::Ge => ax_i >= b_i - PIVOT_TOL,
        ConstraintType::Eq => (ax_i - b_i).abs() <= PIVOT_TOL,
    })
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
    ///
    /// Sentinel: removing `if integer_vars.is_empty() { return None; }` causes a
    /// vacuously integer-feasible LP root (empty mask ⇒ all vars pass the check)
    /// to be returned as `Some(_)` → FAILS.
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

    /// FP converges on a 4-variable binary knapsack within one iteration.
    ///
    /// Sentinel: removing the FP iteration loop causes `run_feasibility_pump` to
    /// return `None` → assertion fails.
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
        let frac: f64 = r.solution.iter().map(|&v| (v - v.round()).abs()).sum();
        assert!(frac < 1e-6, "solution not integer: {:?}", r.solution);
        let obj_recheck: f64 =
            [-3.0f64, -5.0, -2.0, -4.0].iter().zip(r.solution.iter()).map(|(c, x)| c * x).sum();
        assert!((r.objective - obj_recheck).abs() < 1e-6);
    }

    /// Perturbation path is reachable and doesn't panic (coverage guard).
    #[test]
    fn fp_stall_perturbation_does_not_panic() {
        // x in [0.4, 0.6] integer — LP always returns ~0.5, rounds but never converges.
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, -1.0], 2, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![0.6, -0.4],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, 1.0)],
            None,
        ).unwrap();
        let result = run_feasibility_pump(&lp, &[0], 1e-6, &opts());
        assert!(result.is_none(), "expected None for integer-infeasible problem, got Some");
    }

    /// FP incumbent objective is computed on the rounded solution, not the raw LP point.
    ///
    /// Setup: bounds `[near_one, 1.5]` where `near_one = 1.0 - 1e-7`. The LP root
    /// sits at the lower bound `near_one` (fractionality 1e-7 ≤ 1e-6 tol), so
    /// `is_integer_feasible` fires immediately. Rounded value is 1.0, which is inside
    /// `[near_one, 1.5]`.
    ///
    /// Sentinel: removing `round_integer_vars` leaves solution = near_one ≈ 0.9999999
    /// and obj = 0.9999999 ≠ 1.0 → FAIL.
    #[test]
    fn fp_incumbent_uses_rounded_objective_valid_bounds() {
        let near_one = 1.0 - 1e-7;
        // min x s.t. x ≤ 1.5, x ∈ [near_one, 1.5] integer.
        // LP optimal: x = near_one (lb active). Rounded to 1.0, within bounds.
        let lp = single_constraint_lp(vec![1.0], &[1.0], 1.5, vec![(near_one, 1.5)]);
        let r = run_feasibility_pump(&lp, &[0], 1e-6, &opts())
            .expect("LP root integer-feasible within tol, rounded in bounds → Some");
        assert!((r.solution[0] - 1.0).abs() < 1e-9, "solution should be rounded to 1.0, got {}", r.solution[0]);
        assert!((r.objective - 1.0).abs() < 1e-9, "objective should use rounded value 1.0, got {}", r.objective);
    }

    /// FP rejects a rounded incumbent that violates the original variable bounds.
    ///
    /// Setup: `x ∈ [0, 0.9999999]` (near_ub < 1), objective `min -x` (LP pushes x to UB).
    /// LP root = near_ub ≈ 0.9999999, within `integer_feas_tol = 1e-6` of 1.
    /// Rounded x = 1.0 exceeds UB = 0.9999999 → bounds validation rejects it.
    ///
    /// Sentinel: removing `validate_against_bounds` lets rounded x = 1.0 become the
    /// incumbent, causing `run_feasibility_pump` to return `Some` with solution[0] = 1.0
    /// even though 1.0 violates the original bound → FAIL.
    #[test]
    fn fp_rejects_rounded_incumbent_violating_bounds() {
        let near_ub = 1.0 - 1e-7;
        // min -x  s.t. x ≤ near_ub,  x ∈ [0, near_ub]  (integer)
        // LP root: x = near_ub (maximises x within LP).
        // is_integer_feasible: |near_ub − 1| = 1e-7 ≤ 1e-6 → true.
        // round(near_ub) = 1.0,  but 1.0 > near_ub = UB → validation rejects.
        let lp = single_constraint_lp(vec![-1.0], &[1.0], near_ub, vec![(0.0, near_ub)]);
        let result = run_feasibility_pump(&lp, &[0], 1e-6, &opts());
        assert!(
            result.is_none(),
            "FP must reject rounded incumbent that violates UB={}; \
             no-op (remove validate_against_bounds) returns Some(solution[0]=1.0) → FAIL",
            near_ub
        );
    }

    /// `perturb` flips the rounded value of the most-fractional variable.
    ///
    /// Sentinel: a no-op `perturb` (returning `x_int` unchanged) leaves the flipped
    /// variable at its original rounded value → assertion fails.
    #[test]
    fn perturb_flips_most_fractional_variable() {
        // x_int = [1, 0], x_lp = [0.4, 0.7]
        // fractionality: j=0 → 0.4; j=1 → 0.3  (j=0 most fractional)
        // j=0: result[0]=1, floor(0.4)=0; |1-0|=1 ≥ 0.5 → flip to floor → 0
        let x_int = vec![1.0, 0.0];
        let x_lp = vec![0.4, 0.7];
        let mask = vec![true, true];
        let result = perturb(&x_int, &x_lp, &mask, 1);
        assert!(
            (result[0] - 0.0).abs() < 1e-9,
            "most-fractional var should flip 1→0; got {}",
            result[0],
        );
        assert!(
            (result[1] - 0.0).abs() < 1e-9,
            "unflipped var should remain 0; got {}",
            result[1],
        );
    }
}
