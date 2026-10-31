//! LISWET7 の吸引点 (obj ≈ 25.003 = −2500.257 + 目的定数 2525.26) を IP-PMM inner が
//! 収束と判定しないこと。
//!
//! LISWET7 (n=10002, m=10000, Q=I, 全変数 free) では全 lane が最初の十数反復で
//! 残差・μ が小さく、相対 duality gap だけが ≈8.5e-7 残る点に入る。inner の停止
//! 判定が gap を lane の eps と無関係な定数 (1e-3) と比べていた間は、eps≈6.3e-8 の
//! lane がこの点を収束と判定し、目的定数を除いて `Optimal obj=−2500.257` (真値 −2026.419 比
//! 23% 誤差) を返していた。真値 p* = 498.8408907371 (目的定数込み) は HiGHS 解を
//! `scripts/highs_baseline_truth.py` で弱双対検算した値 (92ce6cc2 Era の HiGHS 検算値。
//! 現行 baseline CSV の exact_rational_kkt 認証値とは絶対差 4.9e-8 — ベンチ tol 1e-6 内
//! だが同値ではない。docs/baseline-certificate-contract.md 参照)。
//!
//! Sentinel: inner の gap 許容を定数 1e-3 に戻すと、Optimal 25.003 が返って fail する。

use otspot::io::qps::parse_qps;
use otspot::options::{IpmOptions, SolverOptions, Tolerance};
use otspot::qp::solve_qp_with;
use otspot::SolveStatus;
use std::path::Path;

/// 目的定数込みの HiGHS Era 検算値。現行 `data/baseline_objectives` の exact_rational_kkt
/// 認証値 (4.988408907861e2) とは絶対差 4.9e-8 (tol 1e-6 内、同値ではない)。
const LISWET7_TRUTH_WITH_CONSTANT: f64 = 498.8408907371;
/// bench (`qps_benchmark`) が目的値照合に使う相対許容誤差。
const BENCH_OBJ_REL_TOL: f64 = 1e-2;
const USER_EPS: f64 = 1e-6;
const TIMEOUT_SECS: f64 = 120.0;

#[test]
fn liswet7_attractor_with_constant_is_not_reported_optimal() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/maros_meszaros/LISWET7.QPS");
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行",
        path.display()
    );
    let prob = parse_qps(&path).expect("parse LISWET7");
    let mut opts = SolverOptions::default();
    opts.tolerance = Some(Tolerance::Custom(USER_EPS));
    opts.ipm = IpmOptions {
        eps: USER_EPS,
        ..IpmOptions::default()
    };
    opts.timeout_secs = Some(TIMEOUT_SECS);
    let result = solve_qp_with(&prob, &opts);

    let rel_err =
        (result.objective - LISWET7_TRUTH_WITH_CONSTANT).abs() / LISWET7_TRUTH_WITH_CONSTANT.abs();
    eprintln!(
        "LISWET7 eps={USER_EPS:.0e}: status={:?} obj={:.10e} rel_err={rel_err:.3e} iters={}",
        result.status, result.objective, result.iterations
    );
    if result.status == SolveStatus::Optimal {
        assert!(
            rel_err <= BENCH_OBJ_REL_TOL,
            "LISWET7 returned Optimal obj={:.10e} but truth is {LISWET7_TRUTH_WITH_CONSTANT:.10e} \
             (rel_err={rel_err:.3e}): inner accepted the gap≈8.5e-7 attractor",
            result.objective
        );
    }
}
