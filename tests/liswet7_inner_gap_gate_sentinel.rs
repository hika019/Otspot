//! LISWET7 の吸引点 (obj ≈ −2500.257) を IP-PMM inner が収束と判定しないこと。
//!
//! LISWET7 (n=10002, m=10000, Q=I, 全変数 free) では全 lane が最初の十数反復で
//! 残差・μ が小さく、相対 duality gap だけが ≈8.5e-7 残る点に入る。inner の停止
//! 判定が gap を lane の eps と無関係な定数 (1e-3) と比べていた間は、eps≈6.3e-8 の
//! lane がこの点を収束と判定し、`Optimal obj=−2500.257` (真値比 23% 誤差) を返していた。
//! 真値 p* = −2026.419109214 は HiGHS の Optimal 値 (baseline CSV と同値)。
//!
//! Sentinel: inner の gap 許容を定数 1e-3 に戻すと、Optimal −2500.257 が返って fail する。

use otspot::io::qps::parse_qps;
use otspot::options::{IpmOptions, SolverOptions, Tolerance};
use otspot::qp::solve_qp_with;
use otspot::SolveStatus;
use std::path::Path;

/// HiGHS の Optimal 値 (`data/baseline_objectives` と同値)。
const LISWET7_TRUTH: f64 = -2026.419109214;
/// bench (`qps_benchmark`) が目的値照合に使う相対許容誤差。
const BENCH_OBJ_REL_TOL: f64 = 1e-2;
const USER_EPS: f64 = 1e-6;
const TIMEOUT_SECS: f64 = 120.0;

#[test]
fn liswet7_attractor_is_not_reported_optimal() {
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

    let rel_err = (result.objective - LISWET7_TRUTH).abs() / LISWET7_TRUTH.abs();
    eprintln!(
        "LISWET7 eps={USER_EPS:.0e}: status={:?} obj={:.10e} rel_err={rel_err:.3e} iters={}",
        result.status, result.objective, result.iterations
    );
    if result.status == SolveStatus::Optimal {
        assert!(
            rel_err <= BENCH_OBJ_REL_TOL,
            "LISWET7 returned Optimal obj={:.10e} but truth is {LISWET7_TRUTH:.10e} \
             (rel_err={rel_err:.3e}): inner accepted the gap≈8.5e-7 attractor",
            result.objective
        );
    }
}
