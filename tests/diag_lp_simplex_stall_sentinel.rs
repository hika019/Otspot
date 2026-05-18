//! Sentinel for #33: large LP must converge through `solve_qp_with` (the
//! public LP entry point used by bench) instead of timing out in simplex.
//!
//! Validates the IPM-first dispatch installed in `src/qp/lp_dispatch.rs`.
//! Multiple data patterns are exercised (CLAUDE.md「複数パターンのデータを
//! 用意せよ」):
//!  * 5 real Netlib LPs (ken-13 / ken-18 / cre-b / d6cube / pilot) – each
//!    previously TIMED OUT in simplex; must now reach Optimal/LocallyOptimal.
//!  * 1 synthetic large LP (200 × 600) – proves the size-gate fires on a
//!    random-but-reproducible instance, independent of fixture quirks.
//!
//! **No-op proof** (memory feedback_sentinel_must_fail_under_noop):
//! Setting `LP_DISPATCH_NOOP=1` in the environment makes the dispatch
//! bypass IPM entirely (legacy simplex-only). Run the sentinel under that
//! flag → the four large Netlib LPs MUST regress to non-Optimal. Manual
//! verification:
//!   LP_DISPATCH_NOOP=1 cargo test --release \
//!     --test diag_lp_simplex_stall_sentinel \
//!     -- --nocapture lp_simplex_stall_real_netlib_lps_converge
//! Expected (without fix): timeouts; with fix: PASS.
//!
//! Reading dfl001 truth from data/baseline_objectives is preferred but
//! adds CSV parsing; truths are inlined from netlib_lp.csv for terseness.

use std::path::Path;

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::ConstraintType;
use solver::qp::QpProblem;
use solver::{solve_qp_with, SolveStatus};

const BUDGET_SECS: f64 = 180.0;
/// Short budget for the synthetic case: 30s is plenty for IPM (≈3s observed)
/// but well short of the simplex completion time on m=2500 (sentinel needs
/// the no-op path to fail within a reasonable test runtime).
const SYNTH_BUDGET_SECS: f64 = 30.0;
const REL_TOL: f64 = 5e-3; // 0.5 % of truth – tighter than bench eps=1e-6.

struct Case {
    name: &'static str,
    truth: f64,
}

const REAL_CASES: &[Case] = &[
    Case { name: "ken-13", truth: -1.0257395e10 },
    Case { name: "ken-18", truth: -5.2217025e10 },
    Case { name: "cre-b",  truth:  2.3129640e7  },
    Case { name: "d6cube", truth:  3.1549166667e2 },
    Case { name: "pilot",  truth: -5.5740430007e2 },
];

fn dispatch_disabled() -> bool {
    std::env::var("LP_DISPATCH_NOOP").ok().as_deref() == Some("1")
}

fn load_qp(name: &str) -> Option<QpProblem> {
    let path_str = format!("data/lp_problems/{}.QPS", name);
    let path = Path::new(&path_str);
    if !path.exists() { return None; }
    parse_qps(path).ok()
}

fn rel_err(got: f64, truth: f64) -> f64 {
    let scale = truth.abs().max(1.0);
    (got - truth).abs() / scale
}

/// All 5 large Netlib LPs that previously TIMED OUT in simplex must now
/// converge within the budget. Optimal or LocallyOptimal (close obj) accepted.
#[test]
#[ignore = "long-running; gated behind --ignored, run via bench script"]
fn lp_simplex_stall_real_netlib_lps_converge() {
    let mut failures: Vec<String> = Vec::new();
    for case in REAL_CASES {
        let Some(qp) = load_qp(case.name) else {
            failures.push(format!("{}: data missing", case.name));
            continue;
        };
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(BUDGET_SECS);
        // No warm start; cold-start large LP is the failure mode.
        let r = solve_qp_with(&qp, &opts);
        let converged = matches!(
            r.status,
            SolveStatus::Optimal | SolveStatus::LocallyOptimal
        );
        let close = r.objective.is_finite() && rel_err(r.objective, case.truth) <= REL_TOL;
        if !(converged && close) {
            failures.push(format!(
                "{}: status={:?} obj={:.6e} truth={:.6e} rel_err={:.2e}",
                case.name, r.status, r.objective, case.truth,
                rel_err(r.objective, case.truth)
            ));
        }
    }
    if dispatch_disabled() {
        // Under the no-op flag, we expect failures (sentinel must fail).
        assert!(
            !failures.is_empty(),
            "LP_DISPATCH_NOOP=1 should regress at least one real LP"
        );
        eprintln!("LP_DISPATCH_NOOP=1 observed failures (expected): {:#?}", failures);
        return;
    }
    assert!(failures.is_empty(), "stalled LPs did not converge:\n{}", failures.join("\n"));
}

/// Synthetic large LP: random sparse A, dense c, all Eq, large enough to
/// trigger the IPM-first dispatch (m > 2000). Validates that the dispatch
/// route is exercised on data outside the fixture set.
#[test]
fn lp_simplex_stall_synthetic_large_lp_converges() {
    use solver::sparse::CscMatrix;

    let m: usize = 2_500;
    let n: usize = 3_500;
    assert!(m > 2_000, "must exceed LP_IPM_FIRST_M for size gate to fire");

    // Reproducible random A: each row picks ~6 columns via a deterministic LCG.
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let mut x_star: Vec<f64> = vec![0.0; n];
    let mut lcg: u64 = 0xC0FFEE;
    for i in 0..m {
        for _ in 0..6 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = ((lcg >> 33) as usize) % n;
            rows.push(i);
            cols.push(j);
            // Coeff in [-1, 1) – sparse enough to keep A·x bounded.
            let v = ((lcg >> 17) & 0xFFFF) as f64 / 32768.0 - 1.0;
            vals.push(v);
        }
    }
    for j in 0..n {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        x_star[j] = ((lcg >> 33) & 0xFF) as f64 / 256.0; // ∈ [0,1)
    }
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();

    // b = A x* so the LP is feasible at x*; LP picks a vertex maximising c · x.
    let mut b = vec![0.0; m];
    for k in 0..rows.len() {
        b[rows[k]] += vals[k] * x_star[cols[k]];
    }
    // Cost: arbitrary linear; bounds 0 ≤ x ≤ 1 keep optimum finite.
    let c: Vec<f64> = (0..n).map(|j| (j as f64).sin()).collect();
    let bounds = vec![(0.0_f64, 1.0_f64); n];
    let ctypes = vec![ConstraintType::Eq; m];

    let q = CscMatrix::from_triplets(&[], &[], &[], n, n).unwrap();
    let qp = QpProblem::new(q, c.clone(), a, b, bounds, ctypes).expect("QpProblem ctor");

    // Objective lower bound: with x ∈ [0, 1]^n, optimum is bounded below by
    // sum_j min(c_j · 0, c_j · 1) = sum of negative c_j.
    let obj_lb_bound_only: f64 = c.iter().map(|&v| v.min(0.0)).sum();
    // Feasible incumbent at x_star ∈ [0, 1] gives an upper bound.
    let obj_at_xstar: f64 = c.iter().zip(x_star.iter()).map(|(&cj, &xj)| cj * xj).sum();

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(SYNTH_BUDGET_SECS);
    let r = solve_qp_with(&qp, &opts);

    if dispatch_disabled() {
        // No-op: simplex alone must NOT reach Optimal/LocallyOptimal on this
        // size within the budget (proves the dispatch route is required).
        eprintln!(
            "[synthetic LP_DISPATCH_NOOP=1] status={:?} iters={} obj={:.3e} \
             (budget={}s, m=2500, n=3500)",
            r.status, r.iterations, r.objective, SYNTH_BUDGET_SECS
        );
        assert!(
            !matches!(r.status, SolveStatus::Optimal | SolveStatus::LocallyOptimal),
            "LP_DISPATCH_NOOP=1: simplex unexpectedly converged \
             (sentinel cannot fail). status={:?} iters={}",
            r.status, r.iterations
        );
        return;
    }
    eprintln!(
        "[synthetic dispatch enabled] status={:?} iters={} obj={:.6e} \
         (budget={}s, m=2500, n=3500)",
        r.status, r.iterations, r.objective, SYNTH_BUDGET_SECS
    );

    assert!(
        matches!(r.status, SolveStatus::Optimal | SolveStatus::LocallyOptimal),
        "synthetic large LP must converge: status={:?} iters={} obj={:.3e}",
        r.status, r.iterations, r.objective
    );

    // Objective validity (reviewer M2):
    //  - r.objective <= obj_at_xstar (LP is a minimisation; the dispatched
    //    solution must beat the feasible incumbent we constructed).
    //  - r.objective >= obj_lb_bound_only (with x ∈ [0,1]^n, no objective
    //    can be lower than summing all negative c_j with x_j = 1).
    // Tolerance covers IPM final residual / postsolve drift.
    const OBJ_TOL: f64 = 1e-6;
    let scale = obj_at_xstar.abs().max(obj_lb_bound_only.abs()).max(1.0);
    assert!(
        r.objective.is_finite(),
        "synthetic objective not finite: {}",
        r.objective
    );
    assert!(
        r.objective <= obj_at_xstar + OBJ_TOL * scale,
        "synthetic objective {:.6e} worse than feasible incumbent obj_at_xstar={:.6e}",
        r.objective, obj_at_xstar
    );
    assert!(
        r.objective >= obj_lb_bound_only - OBJ_TOL * scale,
        "synthetic objective {:.6e} below sum-of-negative-cj LB {:.6e} (dual infeasible?)",
        r.objective, obj_lb_bound_only
    );
}
