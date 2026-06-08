//! Sentinel for #33: large LP must converge through `solve_qp_with` (the
//! public LP entry point used by bench) — now via simplex alone.
//!
//! LP は IPM を撤廃し simplex 一本化した (#19/#22)。本 sentinel は「以前 IPM が
//! 隠していた大規模 LP の収束を、simplex 単独で達成できるか」を検証する
//! Phase2 worklist そのもの。複数パターンのデータを用意 (CLAUDE.md):
//!  * 6 real Netlib LPs (ken-13 / ken-18 / cre-b / d6cube / pilot / greenbea) – each
//!    previously TIMED OUT in simplex; simplex 単独で Optimal/LocallyOptimal に
//!    到達し truth と一致することを要求する。
//!  * 1 synthetic large LP (1800 × 3600) – simplex 単独で既知最適を証明し、
//!    primal/objective 不変式を満たす sentinel。
//!
//! これらは simplex 単独では fail しうる。それが Phase2 の正しい信号なので、
//! fail を消すために assert 緩和/test 削除をしてはならない。heavy-ignore な
//! ので標準 suite は緑、heavy run で honest に赤が出る = 想定通り。
//!
//! Reading dfl001 truth from data/baseline_objectives is preferred but
//! adds CSV parsing; truths are inlined from netlib_lp.csv for terseness.

use std::path::Path;

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::ConstraintType;
use otspot::qp::QpProblem;
use otspot::{solve_qp_with, SolveStatus};

/// Quiet single-thread convergence (harris bench, 1000s 逐次, obj_err 0.000%):
/// cre-b 242s / ken-13 187s / d6cube 150s / pilot 90s. 180s sat below
/// cre-b/ken-13 even when quiet and failed under --test-threads 3 contention (cre-b 8.8%,
/// ken-13 17.6% —未収束 timing artifact). 360s covers the slowest quiet case (cre-b 242s)
/// with margin. These 4 run in a dedicated `--test-threads 1` heavy step for fair timing
/// (see test-heavy.yml). REL_TOL stays 5e-3. ken-18 is excluded (真の非収束, #23).
const BUDGET_SECS: f64 = 360.0;
/// Budget for the synthetic case follows the large-LP bench baseline.
/// The generated block LP has a known optimum and should solve far below it.
const SYNTH_BUDGET_SECS: f64 = 1000.0;
const ROUTE_SENTINEL_BUDGET_SECS: f64 = 1000.0;
const REL_TOL: f64 = 5e-3; // 0.5 % of truth – tighter than bench eps=1e-6.

struct Case {
    name: &'static str,
    truth: f64,
}

const REAL_CASES: &[Case] = &[
    Case {
        name: "ken-13",
        truth: -1.0257395e10,
    },
    Case {
        name: "ken-18",
        truth: -5.2217025e10,
    },
    Case {
        name: "dfl001",
        truth: 1.1266400000e7,
    },
    Case {
        name: "pds-20",
        truth: 2.3821659e10,
    },
    Case {
        name: "cre-b",
        truth: 2.3129640e7,
    },
    Case {
        name: "d6cube",
        truth: 3.1549166667e2,
    },
    Case {
        name: "pilot",
        truth: -5.5740430007e2,
    },
    Case {
        name: "greenbea",
        truth: -7.2555248130e7,
    },
];

fn load_qp(name: &str) -> Option<QpProblem> {
    for prefix in ["data/lp_problems", "data/data/lp_problems"] {
        let path_str = format!("{prefix}/{}.QPS", name);
        let path = Path::new(&path_str);
        if path.exists() {
            return parse_qps(path).ok();
        }
    }
    None
}

fn rel_err(got: f64, truth: f64) -> f64 {
    let scale = truth.abs().max(1.0);
    (got - truth).abs() / scale
}

/// simplex 単独で real Netlib LP が BUDGET_SECS 以内に収束し truth と一致すること。
fn assert_real_netlib_lp_converges(case: &Case) {
    let Some(qp) = load_qp(case.name) else {
        panic!("{}: data/lp_problems/{}.QPS missing", case.name, case.name);
    };
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(BUDGET_SECS);
    opts.known_optimal_obj = Some(case.truth);
    let r = solve_qp_with(&qp, &opts);
    let converged = matches!(r.status, SolveStatus::Optimal | SolveStatus::LocallyOptimal);
    let close = r.objective.is_finite() && rel_err(r.objective, case.truth) <= REL_TOL;
    assert!(
        converged && close,
        "{}: status={:?} obj={:.6e} truth={:.6e} rel_err={:.2e}",
        case.name,
        r.status,
        r.objective,
        case.truth,
        rel_err(r.objective, case.truth)
    );
}

macro_rules! real_netlib_case_test {
    ($test_name:ident, $case_name:literal) => {
        #[test]
        #[ignore = "heavy: real Netlib LP sentinel; --profile heavy で実行"]
        fn $test_name() {
            let case = REAL_CASES
                .iter()
                .find(|case| case.name == $case_name)
                .expect("case must exist");
            assert_real_netlib_lp_converges(case);
        }
    };
}

real_netlib_case_test!(lp_simplex_stall_ken13_converges, "ken-13");
real_netlib_case_test!(lp_simplex_stall_ken18_converges, "ken-18");
real_netlib_case_test!(lp_simplex_stall_cre_b_converges, "cre-b");
real_netlib_case_test!(lp_simplex_stall_d6cube_converges, "d6cube");
real_netlib_case_test!(lp_simplex_stall_pilot_converges, "pilot");
real_netlib_case_test!(lp_simplex_stall_greenbea_converges, "greenbea");

fn assert_real_netlib_uses_bounded_eq_ub_path(name: &str, truth: f64) {
    let Some(qp) = load_qp(name) else {
        panic!("{name}: data/lp_problems/{name}.QPS missing");
    };
    assert!(
        qp.constraint_types
            .iter()
            .all(|t| !matches!(t, ConstraintType::Ge)),
        "{name}: this route sentinel is scoped to non-Ge Eq/Le instances"
    );
    assert!(
        qp.bounds.iter().any(|&(_, ub)| ub.is_finite()),
        "{name}: finite UB precondition missing"
    );
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(ROUTE_SENTINEL_BUDGET_SECS);
    opts.known_optimal_obj = Some(truth);
    let r = solve_qp_with(&qp, &opts);
    assert!(
        r.stats.bounded_eq_ub_path,
        "{name}: bounded Eq+UB path not used; status={:?} iters={}",
        r.status, r.iterations
    );
}

macro_rules! real_netlib_bounded_eq_ub_route_test {
    ($test_name:ident, $case_name:literal) => {
        #[test]
        #[ignore = "heavy: route sentinel for large bounded Eq+UB Netlib LP"]
        fn $test_name() {
            let case = REAL_CASES
                .iter()
                .find(|case| case.name == $case_name)
                .expect("case must exist");
            assert_real_netlib_uses_bounded_eq_ub_path(case.name, case.truth);
        }
    };
}

real_netlib_bounded_eq_ub_route_test!(lp_route_dfl001_uses_bounded_eq_ub, "dfl001");
real_netlib_bounded_eq_ub_route_test!(lp_route_ken18_uses_bounded_eq_ub, "ken-18");
real_netlib_bounded_eq_ub_route_test!(lp_route_pds20_uses_bounded_eq_ub, "pds-20");

#[test]
#[ignore = "heavy: bounded Eq+UB pricing efficiency sentinel"]
fn lp_bounded_eq_ub_ken13_iteration_efficiency() {
    let case = REAL_CASES
        .iter()
        .find(|case| case.name == "ken-13")
        .expect("case must exist");
    let Some(qp) = load_qp(case.name) else {
        panic!("{}: data/lp_problems/{}.QPS missing", case.name, case.name);
    };
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(ROUTE_SENTINEL_BUDGET_SECS);
    opts.known_optimal_obj = Some(case.truth);
    let r = solve_qp_with(&qp, &opts);
    eprintln!(
        "ken-13 bounded Eq+UB efficiency: status={:?} obj={:.6e} iterations={} bounded_eq_ub={}",
        r.status, r.objective, r.iterations, r.stats.bounded_eq_ub_path
    );
    assert!(
        matches!(r.status, SolveStatus::Optimal | SolveStatus::LocallyOptimal)
            && r.objective.is_finite()
            && rel_err(r.objective, case.truth) <= REL_TOL
            && r.stats.bounded_eq_ub_path,
        "ken-13: expected certified Optimal on bounded Eq+UB path, got status={:?} obj={:.6e} rel_err={:.2e} bounded_eq_ub={}",
        r.status,
        r.objective,
        rel_err(r.objective, case.truth),
        r.stats.bounded_eq_ub_path
    );
    // Main-era expanded-UB path solved ken-13 in about 62,928 pivots; the
    // regressed compact Eq+UB path took about 145,525. 100k is deliberately
    // between those regimes, with wide headroom over main but below the
    // regression, so this catches pricing-efficiency loss without pinning an
    // exact machine-dependent count.
    assert!(
        r.iterations < 100_000,
        "ken-13 bounded Eq+UB iterations regressed: got {}, expected < 100000",
        r.iterations
    );
}

/// Synthetic large LP: deterministic block identity constraints with slack
/// columns (m=1800, n=3600). The optimum is known exactly: each row chooses the
/// cheaper of `x_i` and `s_i` in `x_i + s_i = 1`, `0 <= x,s <= 1`.
///
/// This is a real sentinel rather than an incumbent smoke test: removing solve
/// work, returning an empty/no-op result, or reporting an objective inconsistent
/// with `x` fails on status, primal feasibility, and known-optimum checks.
/// Run via `--profile heavy --run-ignored ignored-only`.
#[test]
#[ignore = "heavy: synthetic known-optimum sentinel; --profile heavy で実行"]
fn lp_simplex_stall_synthetic_large_lp_dispatches_to_valid_incumbent() {
    use otspot::sparse::CscMatrix;

    let m: usize = 1_800;
    let n: usize = 2 * m;

    let mut rows: Vec<usize> = Vec::with_capacity(n);
    let mut cols: Vec<usize> = Vec::with_capacity(n);
    let mut vals: Vec<f64> = Vec::with_capacity(n);
    let mut c: Vec<f64> = Vec::with_capacity(n);
    let mut expected_obj = 0.0_f64;
    for i in 0..m {
        let x_cost = -1.0 - ((i % 17) as f64) * 0.01;
        rows.push(i);
        cols.push(i);
        vals.push(1.0);
        rows.push(i);
        cols.push(m + i);
        vals.push(1.0);
        c.push(x_cost);
        expected_obj += x_cost;
    }
    for i in 0..m {
        c.push(0.5 + ((i % 11) as f64) * 0.02);
    }
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
    let b = vec![1.0; m];
    let bounds = vec![(0.0_f64, 1.0_f64); n];
    let ctypes = vec![ConstraintType::Eq; m];

    let q = CscMatrix::from_triplets(&[], &[], &[], n, n).unwrap();
    let qp = QpProblem::new(q, c.clone(), a, b, bounds, ctypes).expect("QpProblem ctor");

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(SYNTH_BUDGET_SECS);
    opts.known_optimal_obj = Some(expected_obj);
    let r = solve_qp_with(&qp, &opts);

    eprintln!(
        "[synthetic simplex] status={:?} iters={} obj={:.6e} \
         expected={:.6e} (budget={}s, m={}, n={})",
        r.status, r.iterations, r.objective, expected_obj, SYNTH_BUDGET_SECS, m, n
    );

    assert!(
        matches!(r.status, SolveStatus::Optimal | SolveStatus::LocallyOptimal),
        "synthetic large LP must prove optimality: status={:?} iters={} obj={:.3e}",
        r.status,
        r.iterations,
        r.objective
    );
    assert_eq!(
        r.solution.len(),
        n,
        "synthetic incumbent must be in the original variable space"
    );
    const OBJ_TOL: f64 = 1e-6;
    let scale = expected_obj.abs().max(1.0);
    let obj_from_x: f64 = c
        .iter()
        .zip(r.solution.iter())
        .map(|(&cj, &xj)| cj * xj)
        .sum();
    assert!(
        r.objective.is_finite(),
        "synthetic objective not finite: {}",
        r.objective
    );
    assert!(
        (r.objective - obj_from_x).abs() <= OBJ_TOL * scale,
        "synthetic reported objective {:.6e} differs from c^T x {:.6e}",
        r.objective,
        obj_from_x
    );
    assert!(
        (r.objective - expected_obj).abs() <= OBJ_TOL * scale,
        "synthetic objective {:.6e} differs from known optimum {:.6e}",
        r.objective,
        expected_obj
    );
    let ax = qp.a.mat_vec_mul(&r.solution).expect("A*x");
    let max_primal = ax
        .iter()
        .zip(qp.b.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_primal <= 1e-8,
        "synthetic primal equality residual too large: {:.3e}",
        max_primal
    );
}
