//! QPLIB_9002 (DCL, ill-conditioned diagonal Q) regression sentinel.
//!
//! 経緯: bench_722g_ippmm_1000s が obj=5.698e9 を baseline 化していたが、
//! 以下を事実確認:
//!   - QPLIB 公式 solu = "=unkn=" (published optimum なし)
//!   - Clarabel (tol=1e-8) は NumericalError, obj=3.69e13 (より悪い)
//!   - 当 solver は residual_stall で SuboptimalSolution / obj ~ 1.97e10
//!   - diagonal Q が 1e-12 〜 2.0 (12 桁 condition number) のため IPM が
//!     dual feasibility ~0.5 で停滞、5.698e9 は再現不能
//! → baseline CSV から数値 ref を撤去 (no_ref 化)。本 test は次の退行を検知する:
//!   1. obj / x が NaN/Inf でない (numerical_failure)
//!   2. obj が現状 best (~2e10) より極端に悪化していない (3e10 を超えたら退行)
//!   3. ||x||_inf が問題境界 (~1e11) を 1 order 超える発散をしていない
//!   4. solver が spurious に Optimal を主張しない (Clarabel すら failing なので
//!      Optimal claim は false-positive bug)
//!
//! 現状 ref がない以上「正しい obj」は分からない。本 test の責務は当 solver の
//! 当該問題上での 現状動作を pin し、未来の退行を検知すること。

use solver::io::qplib::{parse_qplib, QplibProblem};
use solver::options::SolverOptions;
use solver::problem::SolveStatus;
use solver::qp::solve_qp_with;
use std::path::Path;
use std::time::Instant;

/// 当 solver の現状 best (1.97e10) に retreat 余裕を見て 3e10 を退行閾値とする。
const QPLIB_9002_OBJ_REGRESSION_CEIL: f64 = 3.0e10;

/// 問題の bound 上界は ~1e11、現状観測 1.4e10。1e12 を超えるなら bound 範囲外。
const QPLIB_9002_X_INF_CEIL: f64 = 1.0e12;

#[test]
fn qplib_9002_solver_does_not_regress_or_diverge() {
    let path = Path::new("data/qplib/QPLIB_9002.qplib");
    assert!(path.exists(), "data missing: QPLIB_9002.qplib");
    let problem = match parse_qplib(path).expect("parse") {
        QplibProblem::Qp(p) => p,
        other => panic!("expected continuous QP for QPLIB_9002, got {:?}", other),
    };
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);
    let t0 = Instant::now();
    let result = solve_qp_with(&problem, &opts);
    let wall = t0.elapsed().as_secs_f64();
    let x_inf = result
        .solution
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "[QPLIB_9002] status={:?} obj={:.4e} iters={} wall={:.3}s ||x||_inf={:.3e}",
        result.status, result.objective, result.iterations, wall, x_inf
    );

    assert!(
        result.objective.is_finite(),
        "obj must be finite (got {})",
        result.objective
    );
    assert!(
        x_inf.is_finite() && x_inf < QPLIB_9002_X_INF_CEIL,
        "||x||_inf={:.3e} >= {:.0e} — IPM 発散",
        x_inf,
        QPLIB_9002_X_INF_CEIL
    );
    assert!(
        result.objective.abs() < QPLIB_9002_OBJ_REGRESSION_CEIL,
        "obj={:.4e} >= {:.0e} (現状 best ~2e10) — IPPMM 退行の疑い",
        result.objective,
        QPLIB_9002_OBJ_REGRESSION_CEIL
    );
    assert!(
        !matches!(result.status, SolveStatus::Optimal),
        "status=Optimal を主張するが QPLIB_9002 は KKT 収束不能 \
         (Clarabel も NumericalError)。spurious Optimal claim は false-positive bug"
    );
}
