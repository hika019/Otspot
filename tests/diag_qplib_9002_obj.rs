//! QPLIB_9002 (DCL, ill-conditioned diagonal Q) regression sentinel.
//!
//! 事実確認 (2026-08-05 再検証):
//!   - QPLIB 公式 solu = "=unkn=" (published optimum なし)
//!   - Clarabel (tol=1e-8) は NumericalError, obj=3.69e13
//!   - Q は対角のみで成分は `[9.288e-12, 2.0]` = **全て正 → 凸 QP**。
//!     したがって KKT 点は大域最適点であり、KKT 残差を独立再計算して user_eps
//!     以内なら `Optimal` の主張は正当である。
//!   - 条件数 ~2e11 のため IPM は KKT を user_eps まで詰め切れない。
//!     18763867 実測: Stalled / obj=1.880e10 / 独立 KKT max=2.889e-1。
//!     行列正則化の床を LP 経路に限定した後: Stalled / obj=5.698e9 /
//!     独立 KKT max=1.525e-6 (obj は 3.3 倍改善、status は正直に Stalled)。
//!
//! 本 test は次の退行を検知する:
//!   1. obj / x が NaN/Inf でない (numerical_failure)
//!   2. report obj が独立再計算 obj と一致する
//!   3. obj が停滞解 (1.88e10) 側へ戻っていない
//!   4. `||x||_inf` が問題境界 (~1e11) を 1 order 超える発散をしていない
//!   5. `Optimal` を主張するなら独立再計算した KKT 残差が user_eps 以内であること
//!      (収束していないのに Optimal を名乗る false-positive の検知)
//!
//! **5 は現状 dead branch である**: 現在の status は `Stalled` なので分岐に入らない。
//! かつ現在の独立 KKT max=1.525e-6 は閾値 user_eps=1e-6 のわずか 1.5 倍上にあり、
//! 軌跡が少し変わって `Optimal` を名乗った瞬間に閾値をまたぐ可能性がある。
//! この test が赤くなったら「false-positive を捕まえた」と「収束が 1.5 倍だけ進んで
//! status が Optimal に変わった」の 2 通りがあるので、`kkt_max` の実測値を見て
//! 切り分けること (前者なら 1e-6 を大きく超える)。

use otspot::io::qplib::{parse_qplib, QplibProblem};
use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::solve_qp_with;
use otspot_dev::bench_utils::compute_qp_kkt_max;
use std::path::Path;
use std::time::Instant;

/// 現状 obj=5.698e9 と旧 obj=1.880e10 の間。悪い停留点側への復帰を検知する。
const QPLIB_9002_OBJ_REGRESSION_CEIL: f64 = 1.0e10;

/// 問題の bound 上界は ~1e11、現状観測 7.9e9。1e12 を超えるなら bound 範囲外。
const QPLIB_9002_X_INF_CEIL: f64 = 1.0e12;

/// `SolverOptions::default()` の要求精度。
const QPLIB_9002_USER_EPS: f64 = 1e-6;

#[test]
fn qplib_9002_objective_and_optimal_claim_are_independently_checked() {
    let path = Path::new("data/qplib/QPLIB_9002.qplib");
    assert!(path.exists(), "data missing: QPLIB_9002.qplib");
    let problem = match parse_qplib(path).expect("parse") {
        QplibProblem::Qp(p) => p,
        other => panic!("expected continuous QP for QPLIB_9002, got {:?}", other),
    };
    // 凸性の前提を test 内で pin する (Q が対角かつ全成分 > 0 なら PSD)。
    let q_min = problem
        .q
        .values()
        .iter()
        .fold(f64::INFINITY, |a, &v| a.min(v));
    assert!(
        q_min > 0.0,
        "QPLIB_9002 の Q は対角正 (凸) である前提が崩れた: min diag={q_min:.3e}"
    );

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(120.0);
    let t0 = Instant::now();
    let result = solve_qp_with(&problem, &opts);
    let wall = t0.elapsed().as_secs_f64();
    let x_inf = result
        .solution
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);

    // 独立再計算: 目的値と KKT 残差 (solver の内部状態ではなく parse 済み問題から)。
    let qx = problem.q.mat_vec_mul(&result.solution).expect("Qx");
    let obj_recomputed = 0.5
        * qx.iter()
            .zip(result.solution.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
        + problem
            .c
            .iter()
            .zip(result.solution.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    let kkt_max = compute_qp_kkt_max(
        &problem,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    eprintln!(
        "[QPLIB_9002] status={:?} obj={:.4e} obj_recomputed={:.4e} kkt_max={:.3e} iters={} wall={:.3}s ||x||_inf={:.3e}",
        result.status, result.objective, obj_recomputed, kkt_max, result.iterations, wall, x_inf
    );

    assert!(
        result.objective.is_finite(),
        "obj must be finite (got {})",
        result.objective
    );
    assert!(
        (result.objective - obj_recomputed).abs()
            <= 1e-9 * result.objective.abs().max(obj_recomputed.abs()).max(1.0),
        "report obj={:.6e} != 独立再計算 obj={:.6e}",
        result.objective,
        obj_recomputed
    );
    assert!(
        x_inf.is_finite() && x_inf < QPLIB_9002_X_INF_CEIL,
        "||x||_inf={:.3e} >= {:.0e} — IPM 発散",
        x_inf,
        QPLIB_9002_X_INF_CEIL
    );
    assert!(
        result.objective.abs() < QPLIB_9002_OBJ_REGRESSION_CEIL,
        "obj={:.4e} >= {:.0e} (現状 5.698e9 / 旧 1.880e10) — 悪い停留点側への退行",
        result.objective,
        QPLIB_9002_OBJ_REGRESSION_CEIL
    );
    if matches!(result.status, SolveStatus::Optimal) {
        assert!(
            kkt_max <= QPLIB_9002_USER_EPS,
            "status=Optimal を主張するが独立再計算 KKT max={kkt_max:.3e} > \
             user_eps={QPLIB_9002_USER_EPS:.0e} — false-positive な Optimal claim"
        );
    }
}
