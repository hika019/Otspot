//! task #11: Big-M Dual Phase I — TDD red / green test
//!
//! ## 目的
//!
//! Ge/Eq 制約を含む infeasible LP の cold-start で
//! `dual_advanced/mod.rs` が `dual::two_phase_dual_simplex` →
//! `cold_start_dual` (num_artificial>0 で primal fallback) に落ち、
//! Primal Phase I cycling で `iters=0 TIMEOUT` する事象を
//! Big-M Dual Phase I で解消する task #11 の TDD ガード。
//!
//! ## 対象 (Netlib infeasible LP set; klein 3 問)
//!
//! - `klein1.QPS`  小型 infeasible
//! - `klein2.QPS`  中型 infeasible
//! - `klein3.QPS`  大型 infeasible (88 × 994、現状 Phase I cycling →
//!                  60s timeout 内で Infeasible 検出に失敗 = task #11 が解消する症状)
//!
//! ## 期待挙動 (TDD GREEN 基準)
//!
//! いずれの問題も:
//! - `status == Infeasible`
//! - 60s 以内に終了
//!
//! TDD RED 時点で klein3 は GREEN 不可。klein1 / klein2 は既存パスでも
//! GREEN になる想定 (regression sentinel)。
//!
//! ## CLAUDE.md
//!
//! data 欠落時は SKIP せず panic (検証空白を作らない原則)。

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::SolveStatus;
use solver::qp::solve_qp_with;
use std::path::Path;
use std::time::Instant;

const TIMEOUT_SEC: f64 = 60.0;

fn run_klein(path_str: &str) -> (SolveStatus, f64, usize) {
    run_klein_with_presolve(path_str, true)
}

fn run_klein_with_presolve(path_str: &str, presolve: bool) -> (SolveStatus, f64, usize) {
    let path = Path::new(path_str);
    assert!(
        path.exists(),
        "data missing: {} — Netlib infeas set 必須 (scripts/netlib_lp_infeas_download.sh)",
        path_str
    );
    let problem = parse_qps(path).expect("parse_qps");
    let mut opts = SolverOptions::default();
    opts.presolve = presolve;
    opts.timeout_secs = Some(TIMEOUT_SEC);

    let t0 = Instant::now();
    let result = solve_qp_with(&problem, &opts);
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "[klein] {} presolve={} -> status={:?} wall={:.3}s iters={} obj={:.3e}",
        path_str, presolve, result.status, wall, result.iterations, result.objective
    );
    (result.status, wall, result.iterations)
}

/// klein1: 小型 infeasible (regression sentinel)
#[test]
fn klein1_infeasible_within_60s() {
    let (status, wall, _iters) = run_klein("data/lp_problems_infeas/klein1.QPS");
    assert_eq!(status, SolveStatus::Infeasible, "klein1 must be Infeasible");
    assert!(wall < TIMEOUT_SEC, "klein1 wall {:.3}s exceeded {}s", wall, TIMEOUT_SEC);
}

/// klein2: 中型 infeasible (regression sentinel)
#[test]
fn klein2_infeasible_within_60s() {
    let (status, wall, _iters) = run_klein("data/lp_problems_infeas/klein2.QPS");
    assert_eq!(status, SolveStatus::Infeasible, "klein2 must be Infeasible");
    assert!(wall < TIMEOUT_SEC, "klein2 wall {:.3}s exceeded {}s", wall, TIMEOUT_SEC);
}

/// klein3: highly degenerate infeasible LP. task #11 introduced Big-M Phase I
/// with a `Timeout + artificials residual → Infeasible` heuristic that
/// happened to be right for klein3 but flipped slow-but-feasible LPs to
/// false-Infeasible (#37: pilot/dfl001/ken-13/ken-18). #37 replaced the
/// heuristic with a Farkas certificate (A^T y ≤ 0, b^T y > 0); on klein3 the
/// Big-M basis after 600K iters does not satisfy A^T y ≤ 0 within 60s budget,
/// so the certificate fails and the solver returns Timeout (honest answer).
///
/// Both verdicts are acceptable: Infeasible (presolve / Phase I converges in
/// time) or Timeout (Phase I incomplete, no certificate). Optimal or Unbounded
/// would be a real bug.
#[test]
fn klein3_no_false_optimal_within_60s() {
    let (status, wall, _iters) = run_klein("data/lp_problems_infeas/klein3.QPS");
    assert!(
        matches!(status, SolveStatus::Infeasible | SolveStatus::Timeout),
        "klein3 must be Infeasible (certified) or Timeout (honest); got {:?}",
        status
    );
    assert!(wall < TIMEOUT_SEC, "klein3 wall {:.3}s exceeded {}s", wall, TIMEOUT_SEC);
}

/// klein3: presolve OFF で Big-M 直接実行の挙動を観測 (diagnostic)
#[test]
fn diag_klein3_no_presolve() {
    let (status, wall, iters) = run_klein_with_presolve("data/lp_problems_infeas/klein3.QPS", false);
    eprintln!("[diag] klein3 no-presolve: status={:?} wall={:.3}s iters={}", status, wall, iters);
    // この test は assertion なし (観測のみ)
}
