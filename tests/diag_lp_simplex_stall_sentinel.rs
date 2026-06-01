//! Sentinel for #33: large LP must converge through `solve_qp_with` (the
//! public LP entry point used by bench) — now via simplex alone.
//!
//! LP は IPM を撤廃し simplex 一本化した (#19/#22)。本 sentinel は「以前 IPM が
//! 隠していた大規模 LP の収束を、simplex 単独で達成できるか」を検証する
//! Phase2 worklist そのもの。複数パターンのデータを用意 (CLAUDE.md):
//!  * 6 real Netlib LPs (ken-13 / ken-18 / cre-b / d6cube / pilot / greenbea) – each
//!    previously TIMED OUT in simplex; simplex 単独で Optimal/LocallyOptimal に
//!    到達し truth と一致することを要求する。
//!  * 1 synthetic large LP (2500 × 3500) – simplex 単独で budget 内に有限
//!    incumbent を返す robustness 検証。
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

const BUDGET_SECS: f64 = 180.0;
/// Budget for the synthetic case. simplex 単独で有限 incumbent を返すことの
/// 確認用 (最適性証明ではない)。
const SYNTH_BUDGET_SECS: f64 = 30.0;
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
    let path_str = format!("data/lp_problems/{}.QPS", name);
    let path = Path::new(&path_str);
    if !path.exists() {
        return None;
    }
    parse_qps(path).ok()
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

/// Synthetic large LP: random sparse A, dense c, all Eq, large (m=2500, n=3500).
/// simplex 単独で budget 内に有限 incumbent を返す robustness 検証。
///
/// **Heavy tier**: simplex returns a primal-feasible incumbent within the 30s
/// budget under full nextest contention. This test is not an optimality proof:
/// there is no known optimum for this generated instance, so it checks that a
/// valid incumbent is returned rather than claiming `Optimal`.
/// Run via `--profile heavy --run-ignored ignored-only`.
#[test]
#[ignore = "heavy: CPU contention 30s budget hit (2026-05-30 #130 lead-verify retest で再現); permanent heavy-tier ignore"]
fn lp_simplex_stall_synthetic_large_lp_dispatches_to_valid_incumbent() {
    use otspot::sparse::CscMatrix;

    let m: usize = 2_500;
    let n: usize = 3_500;

    // Reproducible random A: each row picks ~6 columns via a deterministic LCG.
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let mut x_star: Vec<f64> = vec![0.0; n];
    let mut lcg: u64 = 0xC0FFEE;
    for i in 0..m {
        for _ in 0..6 {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = ((lcg >> 33) as usize) % n;
            rows.push(i);
            cols.push(j);
            // Coeff in [-1, 1) – sparse enough to keep A·x bounded.
            let v = ((lcg >> 17) & 0xFFFF) as f64 / 32768.0 - 1.0;
            vals.push(v);
        }
    }
    for xj in x_star.iter_mut() {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *xj = ((lcg >> 33) & 0xFF) as f64 / 256.0; // ∈ [0,1)
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

    eprintln!(
        "[synthetic simplex] status={:?} iters={} obj={:.6e} \
         (budget={}s, m=2500, n=3500)",
        r.status, r.iterations, r.objective, SYNTH_BUDGET_SECS
    );

    assert!(
        !matches!(
            r.status,
            SolveStatus::Timeout
                | SolveStatus::MaxIterations
                | SolveStatus::NumericalError
                | SolveStatus::Infeasible
                | SolveStatus::Unbounded
        ),
        "synthetic large LP must return a finite incumbent: status={:?} iters={} obj={:.3e}",
        r.status,
        r.iterations,
        r.objective
    );
    assert_eq!(
        r.solution.len(),
        n,
        "synthetic incumbent must be in the original variable space"
    );

    // Objective validity (reviewer M2):
    //  - r.objective <= obj_at_xstar (LP is a minimisation; the solution must
    //    beat the feasible incumbent we constructed).
    //  - r.objective >= obj_lb_bound_only (with x ∈ [0,1]^n, no objective can be
    //    lower than summing all negative c_j with x_j = 1).
    // Tolerance covers final residual / postsolve drift.
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
        r.objective,
        obj_at_xstar
    );
    assert!(
        r.objective >= obj_lb_bound_only - OBJ_TOL * scale,
        "synthetic objective {:.6e} below sum-of-negative-cj LB {:.6e} (dual infeasible?)",
        r.objective,
        obj_lb_bound_only
    );
}
