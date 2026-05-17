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

/// klein3 専用 Bland anti-cycling 上限。
///
/// basis-hash 観測で「concurrent path の `dual_simplex_core_advanced` が
/// 280k iter で 4 distinct basis を周回」が事実。Bland's rule (smallest-idx
/// leaving + min-ratio/smallest-idx entering) は有限終了保証を持ち、cycle 長
/// m=88 オーダーなら数百〜数千 iter で抜けるのが物理的妥当。30s timeout に
/// 余裕を持って 20s 上限とした。
const KLEIN3_BLAND_BUDGET_SEC: f64 = 20.0;

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

/// task #6 (anti-cycling): bland_mode 起動時に lex 摂動 (`x_b += B^{-1} delta`、
/// `delta[i] = LEX_PERTURB_BASE * LEX_PERTURB_RATIO^i`) を注入することで
/// degeneracy が解消され、Bland's rule が klein3 を有限ステップで Infeasible
/// 判定することを確認する。
#[test]
fn klein3_infeasible_via_bland_anticycling() {
    let path = Path::new("data/lp_problems_infeas/klein3.QPS");
    assert!(path.exists(), "data missing: {}", path.display());
    let problem = parse_qps(path).expect("parse_qps");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(30.0);

    let t0 = Instant::now();
    let result = solver::qp::solve_qp_with(&problem, &opts);
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "[klein3-bland] status={:?} wall={:.3}s iters={}",
        result.status, wall, result.iterations
    );

    assert_eq!(
        result.status,
        SolveStatus::Infeasible,
        "Bland anti-cycling で klein3 は Infeasible 判定されるべき"
    );
    assert!(
        wall < KLEIN3_BLAND_BUDGET_SEC,
        "klein3 wall {:.3}s — anti-cycling 効いていれば {:.1}s 未満で終わるはず",
        wall,
        KLEIN3_BLAND_BUDGET_SEC
    );
}

/// klein3: presolve OFF で Big-M 直接実行の挙動を観測 (diagnostic)
#[test]
fn diag_klein3_no_presolve() {
    let (status, wall, iters) = run_klein_with_presolve("data/lp_problems_infeas/klein3.QPS", false);
    eprintln!("[diag] klein3 no-presolve: status={:?} wall={:.3}s iters={}", status, wall, iters);
    // この test は assertion なし (観測のみ)
}

/// SPEED #1 (task #37): LP cold-start (Ge/Eq) で `solve_dual_advanced` は
/// Primal (`two_phase_dual_simplex`) を deadline の半分で実行し、Timeout なら
/// Big-M Phase I へ fall back する。klein3 は Primal Phase I が cycling 確実な
/// degenerate infeasible LP で、Primal が **進歩なしで half-deadline を完全に
/// 食いつぶし**、Big-M が残り半分で間に合わないと Timeout する症状を示す。
///
/// ## 観測事実 (speed-profiler #36)
///
/// - timeout=30s: wall ≈ 17.6s (Primal 15s 浪費 + Big-M 2.6s 成功)
/// - timeout=60s: wall ≈ 32.3s (Primal 30s 浪費 + Big-M 2.3s 成功) ← 浪費が deadline に比例
///
/// 修正方針: Primal `revised_simplex_core` に **no-progress 早期 bail**
/// (K iter 連続で `c^T x_B` 改善なし → Timeout 返却) を追加し、Primal が
/// cycling を検出した時点で速やかに Big-M に時間を譲る。
///
/// ## 期待 (TDD GREEN)
///
/// timeout=60s 設定で wall < 25s (現状 RED ~32s)。
/// 修正後実測 ≈ 13s (Primal 半 deadline 内 bail + Big-M 完走)、25s は安全
/// マージン込み。bail 閾値を perold/dfl001 等 slow-but-progressing LP を
/// 巻き込まない値に調整した上限。
#[test]
fn klein3_primal_early_bail_speedup() {
    let path = Path::new("data/lp_problems_infeas/klein3.QPS");
    assert!(path.exists(), "data missing: {}", path.display());
    let problem = parse_qps(path).expect("parse_qps");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);

    let t0 = Instant::now();
    let result = solver::qp::solve_qp_with(&problem, &opts);
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "[klein3-speedup] status={:?} wall={:.3}s iters={}",
        result.status, wall, result.iterations
    );

    assert_eq!(
        result.status,
        SolveStatus::Infeasible,
        "klein3 は Big-M Phase I で Infeasible 判定されるべき"
    );
    const KLEIN3_PRIMAL_EARLY_BAIL_BUDGET_SEC: f64 = 25.0;
    assert!(
        wall < KLEIN3_PRIMAL_EARLY_BAIL_BUDGET_SEC,
        "klein3 wall {:.3}s — Primal early-bail が効いていれば {:.1}s 未満で終わるはず (現状 ~32s で FAIL)",
        wall,
        KLEIN3_PRIMAL_EARLY_BAIL_BUDGET_SEC
    );
}
