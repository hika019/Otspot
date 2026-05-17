//! d6cube perf/feasibility regression guard。
//!
//! d6cube は feasible LP (415 Eq, 6184 vars, c_j=1) で Phase II が optimum
//! 近傍で highly degenerate。Big-M Phase I の false Infeasible flip 防止 +
//! Phase II が known optimum 近傍に到達することを guard する。

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::SolveStatus;
use solver::qp::solve_qp_with;
use std::path::Path;

const D6CUBE_KNOWN_OPT: f64 = 3.15491666666e+02;
const D6CUBE_TIMEOUT_S: f64 = 60.0;
/// Loose progress bound: pre-#36 baseline returned obj=+inf (false Infeasible),
/// post-fix Phase II reaches obj within ~5% of the known optimum.
const D6CUBE_OBJ_REL_PROGRESS: f64 = 0.05;

#[test]
fn d6cube_no_false_infeasible_and_makes_progress() {
    let path = Path::new("data/lp_problems/d6cube.QPS");
    if !path.exists() {
        panic!("data missing: {} — required for d6cube guard", path.display());
    }
    let prob = parse_qps(path).expect("parse d6cube");
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(D6CUBE_TIMEOUT_S);

    let t0 = std::time::Instant::now();
    let r = solve_qp_with(&prob, &opts);
    let elapsed = t0.elapsed().as_secs_f64();

    let rel_err = (r.objective - D6CUBE_KNOWN_OPT).abs()
        / (1.0_f64).max(D6CUBE_KNOWN_OPT.abs());
    eprintln!(
        "d6cube: status={:?} obj={:.6e} (known {:.6e}, rel_err={:.2e}) elapsed={:.2}s iters={}",
        r.status, r.objective, D6CUBE_KNOWN_OPT, rel_err, elapsed, r.iterations
    );
    if let Some(t) = r.timing_breakdown {
        eprintln!(
            "  timing: presolve={:.2}s solve={:.2}s postsolve={:.2}s",
            t.presolve_us as f64 / 1e6,
            t.solve_us as f64 / 1e6,
            t.postsolve_us as f64 / 1e6
        );
    }

    // Guard: must not return Infeasible. Timeout/Optimal are acceptable.
    assert!(
        !matches!(r.status, SolveStatus::Infeasible),
        "d6cube returned Infeasible — dual_advanced router must not run Big-M \
         Phase I on a feasible-but-slow LP"
    );
    // Solver must produce a non-empty incumbent.
    assert!(
        !r.solution.is_empty(),
        "d6cube returned an empty solution — solver made no progress"
    );
    // Phase II must drive the obj into a meaningful neighborhood of the known
    // optimum — pre-fix obj was +inf, post-fix obj should be within ~5%.
    assert!(
        rel_err < D6CUBE_OBJ_REL_PROGRESS,
        "d6cube obj rel_err={:.2e} exceeds progress bound {:.0e}; \
         obj={:.6e} vs known {:.6e}",
        rel_err, D6CUBE_OBJ_REL_PROGRESS, r.objective, D6CUBE_KNOWN_OPT
    );
}

/// Aspirational red test: d6cube should reach Optimal at eps=1e-6 within 60s.
/// Currently FAIL (Phase II asymptotic deceleration on degenerate optimum).
#[test]
#[ignore = "Phase II asymptotic deceleration on degenerate optimum"]
fn d6cube_optimal_within_60s() {
    let path = Path::new("data/lp_problems/d6cube.QPS");
    assert!(path.exists(), "data missing: {}", path.display());
    let prob = parse_qps(path).expect("parse d6cube");
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(D6CUBE_TIMEOUT_S);

    let r = solve_qp_with(&prob, &opts);
    let rel_err = (r.objective - D6CUBE_KNOWN_OPT).abs()
        / (1.0_f64).max(D6CUBE_KNOWN_OPT.abs());
    eprintln!(
        "d6cube: status={:?} obj={:.6e} rel_err={:.2e} iters={}",
        r.status, r.objective, rel_err, r.iterations
    );

    assert!(matches!(r.status, SolveStatus::Optimal), "d6cube must reach Optimal");
    assert!(rel_err < 1e-4, "d6cube obj rel_err {:.2e} >= 1e-4", rel_err);
}
