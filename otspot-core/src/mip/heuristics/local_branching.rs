//! Local-branching primal heuristic for MILP (Fischetti & Lodi, 2003).
//!
//! Given an incumbent `x_inc`, the binary neighborhood
//! `{ x : Δ(x, x_inc) <= k }` — where `Δ` is the Hamming distance over the
//! binary variables — is searched by adding the single linear *local-branching*
//! constraint to the model and solving the resulting sub-MIP. The incumbent
//! itself satisfies the constraint (Hamming distance 0), so the sub-MIP is
//! always feasible; any better point it finds strictly improves the incumbent.
//!
//! The local-branching cut for binaries split by their incumbent value into
//! `S0 = {j : x_inc[j] = 0}` and `S1 = {j : x_inc[j] = 1}` is
//!
//! ```text
//!   Σ_{j∈S0} x_j + Σ_{j∈S1} (1 − x_j) ≤ k
//! ⇔ Σ_{j∈S0} x_j − Σ_{j∈S1} x_j      ≤ k − |S1|.
//! ```

use crate::mip::{MilpProblem, MipConfig};
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, LpProblem, SolverResult};
use crate::sparse::CscMatrix;
use crate::tolerances::INT_ROUND_TOL;
use std::time::Instant;

/// Run local branching every this many B&B nodes.
pub(crate) const LOCAL_BRANCHING_INTERVAL: usize = 200;

/// Neighborhood radius `k`: maximum Hamming distance (number of flipped
/// binaries) explored around the incumbent. The classic Fischetti–Lodi default.
pub(crate) const LOCAL_BRANCHING_K: usize = 20;

/// Node limit for the local-branching sub-MIP.
const LOCAL_BRANCHING_NODE_LIMIT: usize = 2_000;

/// Fraction of remaining wall-clock budget given to the sub-MIP.
const LOCAL_BRANCHING_TIME_FRACTION: f64 = 0.10;

/// Absolute upper bound on sub-MIP wall time (seconds).
const LOCAL_BRANCHING_MAX_TIME_SECS: f64 = 10.0;

/// Minimum remaining budget below which local branching is skipped.
const LOCAL_BRANCHING_MIN_REMAINING_SECS: f64 = 1.0;

/// Run the local-branching heuristic around incumbent `x_inc`.
///
/// Uses the [`LOCAL_BRANCHING_K`] neighborhood radius and the module budget
/// constants. Returns a feasible sub-MIP `SolverResult` (which may or may not
/// improve the incumbent; the caller's incumbent test decides) or `None` when
/// there are no binary variables, the deadline has passed, or the sub-MIP
/// produced no usable point.
pub(crate) fn run_local_branching(
    problem: &MilpProblem,
    x_inc: &[f64],
    cfg: &MipConfig,
    deadline: &Option<Instant>,
    parent_opts: &SolverOptions,
) -> Option<SolverResult> {
    local_branching_with_k(
        problem,
        x_inc,
        LOCAL_BRANCHING_K,
        cfg,
        deadline,
        parent_opts,
    )
}

/// Core implementation parameterized by the neighborhood radius `k`, so tests
/// can drive a controlled radius while the public entry point uses the named
/// default constant.
fn local_branching_with_k(
    problem: &MilpProblem,
    x_inc: &[f64],
    k: usize,
    cfg: &MipConfig,
    deadline: &Option<Instant>,
    parent_opts: &SolverOptions,
) -> Option<SolverResult> {
    let remaining_secs = remaining_budget(deadline);
    if remaining_secs < LOCAL_BRANCHING_MIN_REMAINING_SECS {
        return None;
    }

    // Binary variables = integer variables with a [0,1] box.
    let binaries: Vec<usize> = problem
        .integer_vars
        .iter()
        .copied()
        .filter(|&j| {
            j < x_inc.len() && {
                let (lb, ub) = problem.lp.bounds[j];
                (lb - 0.0).abs() <= INT_ROUND_TOL && (ub - 1.0).abs() <= INT_ROUND_TOL
            }
        })
        .collect();
    if binaries.is_empty() {
        return None;
    }

    let sub_lp = augment_with_local_branching_cut(&problem.lp, &binaries, x_inc, k)?;
    let sub_problem = MilpProblem::new(sub_lp, problem.integer_vars.clone()).ok()?;

    let sub_timeout =
        (remaining_secs * LOCAL_BRANCHING_TIME_FRACTION).min(LOCAL_BRANCHING_MAX_TIME_SECS);

    let mut sub_cfg = cfg.clone();
    sub_cfg.max_nodes = LOCAL_BRANCHING_NODE_LIMIT;
    sub_cfg.rins_enabled = false;
    sub_cfg.rens_enabled = false;
    sub_cfg.local_branching_enabled = false;

    let mut sub_opts = parent_opts.clone();
    sub_opts.timeout_secs = Some(sub_timeout);
    sub_opts.deadline = None;
    sub_opts.warm_start = None;
    sub_opts.warm_start_qp = None;
    sub_opts.warm_start_lp = None;
    sub_opts.known_optimal_obj = None;
    sub_opts.presolve = true;
    sub_opts.use_lp_crash_basis = true;
    sub_opts.recover_warm_start_basis = false;
    sub_opts.threads = 1;

    let result = super::solve_sub_milp(&sub_problem, &sub_opts, &sub_cfg);
    super::usable_sub_mip_result_for_original(problem, result, cfg.integer_feas_tol)
}

/// Append the local-branching Hamming-distance row to a copy of `lp`.
///
/// Returns `None` if the constraint matrix cannot be rebuilt (only on a
/// dimension error, which cannot occur for a well-formed `lp`).
fn augment_with_local_branching_cut(
    lp: &LpProblem,
    binaries: &[usize],
    x_inc: &[f64],
    k: usize,
) -> Option<LpProblem> {
    let new_row = lp.num_constraints;

    // Existing nonzeros (CSC → triplets).
    let col_ptr = lp.a.col_ptr();
    let row_ind = lp.a.row_ind();
    let values = lp.a.values();
    let mut rows = Vec::with_capacity(values.len() + binaries.len());
    let mut cols = Vec::with_capacity(values.len() + binaries.len());
    let mut vals = Vec::with_capacity(values.len() + binaries.len());
    for col in 0..lp.a.ncols() {
        for idx in col_ptr[col]..col_ptr[col + 1] {
            rows.push(row_ind[idx]);
            cols.push(col);
            vals.push(values[idx]);
        }
    }

    // Local-branching row: +1 for S0 (inc=0), −1 for S1 (inc=1); rhs = k − |S1|.
    let mut s1_count = 0usize;
    for &j in binaries {
        if x_inc[j].round() >= 0.5 {
            // inc = 1 → coefficient −1
            rows.push(new_row);
            cols.push(j);
            vals.push(-1.0);
            s1_count += 1;
        } else {
            // inc = 0 → coefficient +1
            rows.push(new_row);
            cols.push(j);
            vals.push(1.0);
        }
    }

    let a =
        CscMatrix::from_triplets(&rows, &cols, &vals, lp.num_constraints + 1, lp.num_vars).ok()?;

    let mut b = lp.b.clone();
    b.push(k as f64 - s1_count as f64);
    let mut constraint_types = lp.constraint_types.clone();
    constraint_types.push(ConstraintType::Le);

    let mut new_lp = LpProblem::new_general(
        lp.c.clone(),
        a,
        b,
        constraint_types,
        lp.bounds.clone(),
        lp.name.clone(),
    )
    .ok()?;
    new_lp.obj_offset = lp.obj_offset;
    Some(new_lp)
}

fn remaining_budget(deadline: &Option<Instant>) -> f64 {
    match deadline {
        None => f64::INFINITY,
        Some(d) => {
            let now = Instant::now();
            if now >= *d {
                0.0
            } else {
                (*d - now).as_secs_f64()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ConstraintType;
    use crate::sparse::CscMatrix;

    /// min c·x  s.t.  Σx <= b,  x ∈ {0,1}^n.
    fn binary_knapsack(c: Vec<f64>, b: f64) -> MilpProblem {
        let n = c.len();
        let rows = vec![0usize; n];
        let cols: Vec<usize> = (0..n).collect();
        let a = CscMatrix::from_triplets(&rows, &cols, &vec![1.0; n], 1, n).unwrap();
        let lp = LpProblem::new_general(
            c,
            a,
            vec![b],
            vec![ConstraintType::Le],
            vec![(0.0, 1.0); n],
            None,
        )
        .unwrap();
        let int_vars: Vec<usize> = (0..n).collect();
        MilpProblem::new(lp, int_vars).unwrap()
    }

    /// SENTINEL: local branching STRICTLY improves a planted suboptimal incumbent.
    ///
    /// Problem: min −(x0+x1+x2) s.t. x0+x1+x2 <= 2, x ∈ {0,1}^3. Optimum −2.
    /// Planted incumbent (1,0,0), obj −1 (suboptimal). With k=2 the neighborhood
    /// reaches (1,1,0) at Hamming distance 1 → obj −2.
    ///
    /// A no-op heuristic (always `None`) fails `.expect`; a heuristic that merely
    /// echoes the incumbent fails the strict-improvement assertion.
    #[test]
    fn local_branching_strictly_improves_planted_incumbent() {
        let problem = binary_knapsack(vec![-1.0, -1.0, -1.0], 2.0);
        let cfg = MipConfig::default();
        let x_inc = vec![1.0, 0.0, 0.0];
        let inc_obj = -1.0;

        let res = run_local_branching(&problem, &x_inc, &cfg, &None, &SolverOptions::default())
            .expect("local branching must return a feasible neighborhood solution");
        assert!(
            res.objective < inc_obj - 1e-6,
            "local branching must strictly improve incumbent {inc_obj}; got {}",
            res.objective
        );
        assert!(
            (res.objective - (-2.0)).abs() < 1e-6,
            "neighborhood optimum is -2; got {}",
            res.objective
        );
    }

    #[test]
    fn local_branching_run_path_accepts_feasible_timeout_incumbent() {
        let problem = binary_knapsack(vec![-1.0, -1.0, -1.0], 2.0);
        let cfg = MipConfig::default();
        let x_inc = vec![1.0, 0.0, 0.0];
        super::super::set_next_sub_mip_result(SolverResult {
            status: crate::problem::SolveStatus::Timeout,
            objective: -1.0e100,
            solution: vec![1.0, 1.0, 0.0],
            ..SolverResult::default()
        });

        let result = run_local_branching(&problem, &x_inc, &cfg, &None, &SolverOptions::default())
            .expect("local branching must keep feasible timeout incumbent from sub-MIP");

        assert_eq!(result.solution, vec![1.0, 1.0, 0.0]);
        assert_eq!(result.objective, -2.0);
    }

    /// SENTINEL: the local-branching constraint is load-bearing — radius `k`
    /// genuinely restricts the search, it is not silently dropped.
    ///
    /// Problem: min −(x0+x1+x2) s.t. x0+x1+x2 <= 3 (non-binding), x ∈ {0,1}^3.
    /// Global optimum is (1,1,1) = −3. Incumbent (0,0,0). With k=1 only a single
    /// flip is allowed → best reachable is −1, NOT −3. If the cut were dropped the
    /// sub-MIP would return −3, failing this assertion.
    #[test]
    fn local_branching_radius_restricts_neighborhood() {
        let problem = binary_knapsack(vec![-1.0, -1.0, -1.0], 3.0);
        let cfg = MipConfig::default();
        let x_inc = vec![0.0, 0.0, 0.0];

        let res =
            local_branching_with_k(&problem, &x_inc, 1, &cfg, &None, &SolverOptions::default())
                .expect("k=1 neighborhood is feasible (contains the incumbent)");
        assert!(
            (res.objective - (-1.0)).abs() < 1e-6,
            "k=1 caps improvement at one flip (-1), not the global -3; got {}. \
             A dropped local-branching cut would return -3.",
            res.objective
        );
    }

    /// k=0 pins the search to the incumbent itself (no flip allowed).
    #[test]
    fn local_branching_k_zero_returns_incumbent() {
        let problem = binary_knapsack(vec![-1.0, -1.0, -1.0], 3.0);
        let cfg = MipConfig::default();
        let x_inc = vec![1.0, 0.0, 0.0];

        let res =
            local_branching_with_k(&problem, &x_inc, 0, &cfg, &None, &SolverOptions::default())
                .expect("k=0 neighborhood still contains the incumbent");
        assert!(
            (res.objective - (-1.0)).abs() < 1e-6,
            "k=0 must keep the incumbent objective -1; got {}",
            res.objective
        );
    }

    /// Returns `None` when there are no binary variables (general-integer model).
    #[test]
    fn local_branching_skips_without_binaries() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![-1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 5.0)], // not [0,1] → not binary
            None,
        )
        .unwrap();
        let problem = MilpProblem::new(lp, vec![0]).unwrap();
        let cfg = MipConfig::default();
        assert!(
            run_local_branching(&problem, &[3.0], &cfg, &None, &SolverOptions::default()).is_none(),
            "local branching requires binary variables"
        );
    }

    #[test]
    fn local_branching_run_path_passes_recursive_sub_mip_config() {
        let problem = binary_knapsack(vec![-1.0, -1.0, -1.0], 2.0);
        let cfg = MipConfig {
            max_nodes: 99_999,
            rins_enabled: true,
            rens_enabled: true,
            local_branching_enabled: true,
            ..MipConfig::default()
        };
        let x_inc = vec![1.0, 0.0, 0.0];

        super::super::clear_recorded_sub_mip_configs();
        let result = run_local_branching(&problem, &x_inc, &cfg, &None, &SolverOptions::default());
        let configs = super::super::take_recorded_sub_mip_configs();

        assert!(
            result.is_some(),
            "test premise: local branching must call the recursive sub-MIP"
        );
        assert_eq!(
            configs.len(),
            1,
            "local branching run path must solve exactly one sub-MIP"
        );
        let sub_cfg = &configs[0];
        assert_eq!(sub_cfg.max_nodes, LOCAL_BRANCHING_NODE_LIMIT);
        assert!(!sub_cfg.rins_enabled, "recursive RINS must be disabled");
        assert!(!sub_cfg.rens_enabled, "recursive RENS must be disabled");
        assert!(
            !sub_cfg.local_branching_enabled,
            "recursive local branching must be disabled"
        );
    }

    /// Skips when the deadline is already past (no work after expiry).
    #[test]
    fn local_branching_skips_on_expired_deadline() {
        let problem = binary_knapsack(vec![-1.0, -1.0, -1.0], 2.0);
        let cfg = MipConfig::default();
        let x_inc = vec![1.0, 0.0, 0.0];
        let past = Instant::now() - std::time::Duration::from_secs(1);
        assert!(
            run_local_branching(
                &problem,
                &x_inc,
                &cfg,
                &Some(past),
                &SolverOptions::default()
            )
            .is_none(),
            "local branching must not run after the deadline"
        );
    }

    /// The augmented cut row encodes the Hamming distance exactly: rhs = k − |S1|
    /// and coefficients ±1 split by incumbent value.
    #[test]
    fn cut_row_encodes_hamming_distance() {
        let problem = binary_knapsack(vec![-1.0, -1.0, -1.0], 3.0);
        let x_inc = vec![1.0, 0.0, 1.0]; // S1 = {0,2}, |S1| = 2
        let k = 2;
        let lp = augment_with_local_branching_cut(&problem.lp, &[0, 1, 2], &x_inc, k).unwrap();
        assert_eq!(lp.num_constraints, problem.lp.num_constraints + 1);
        let new_row = problem.lp.num_constraints;
        assert!(
            (lp.b[new_row] - (k as f64 - 2.0)).abs() < 1e-12,
            "rhs must be k - |S1| = 0; got {}",
            lp.b[new_row]
        );
        // Hamming distance of incumbent to itself is 0 ≤ rhs ⇒ feasible.
        // Verify coefficient signs via a dense row read.
        let ax = lp.a.mat_vec_mul(&x_inc).unwrap();
        // For x = x_inc: Σ_{S0} 0 − Σ_{S1} 1 = −|S1| = −2 ≤ rhs(0). distance term.
        assert!(
            (ax[new_row] - (-2.0)).abs() < 1e-12,
            "incumbent row activity must be -|S1| = -2; got {}",
            ax[new_row]
        );
    }
}
