//! Big-M Dual Phase I — TDD red / green test
//!
//! ## 目的
//!
//! Ge/Eq 制約を含む infeasible LP の cold-start で
//! `dual_advanced/mod.rs` が `dual::two_phase_dual_simplex` →
//! `cold_start_dual` (num_artificial>0 で primal fallback) に落ち、
//! Primal Phase I cycling で `iters=0 TIMEOUT` する事象を
//! Big-M Dual Phase I で解消する TDD ガード。
//!
//! ## 対象 (Netlib infeasible LP set; klein 3 問)
//!
//! - `klein1.QPS`  小型 infeasible
//! - `klein2.QPS`  中型 infeasible
//! - `klein3.QPS`  大型 infeasible (88 × 994、Phase I cycling で
//!   60s timeout 内に Infeasible 検出に失敗していた症状)
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

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::solve_qp_with;
use std::path::Path;
use std::time::Instant;

const TIMEOUT_SEC: f64 = 60.0;

fn run_klein(path_str: &str) -> (SolveStatus, f64, usize) {
    let path = Path::new(path_str);
    assert!(
        path.exists(),
        "data missing: {} — Netlib infeas set 必須 (scripts/netlib_lp_infeas_download.sh)",
        path_str
    );
    let problem = parse_qps(path).expect("parse_qps");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(TIMEOUT_SEC);

    let t0 = Instant::now();
    let result = solve_qp_with(&problem, &opts);
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "[klein] {} -> status={:?} wall={:.3}s iters={} obj={:.3e}",
        path_str, result.status, wall, result.iterations, result.objective
    );
    (result.status, wall, result.iterations)
}

/// klein1: 小型 infeasible (regression sentinel)
#[test]
fn klein1_infeasible_within_60s() {
    let (status, wall, _iters) = run_klein("data/lp_problems_infeas/klein1.QPS");
    assert_eq!(status, SolveStatus::Infeasible, "klein1 must be Infeasible");
    assert!(
        wall < TIMEOUT_SEC,
        "klein1 wall {:.3}s exceeded {}s",
        wall,
        TIMEOUT_SEC
    );
}

/// klein2: 中型 infeasible (regression sentinel)
#[test]
fn klein2_infeasible_within_60s() {
    let (status, wall, _iters) = run_klein("data/lp_problems_infeas/klein2.QPS");
    assert_eq!(status, SolveStatus::Infeasible, "klein2 must be Infeasible");
    assert!(
        wall < TIMEOUT_SEC,
        "klein2 wall {:.3}s exceeded {}s",
        wall,
        TIMEOUT_SEC
    );
}

/// klein3: highly degenerate infeasible LP. Big-M Phase I does not extract a
/// basis-derived certificate before its anti-cycling cap; LP dispatch must
/// still certify the original nonnegative LP through a verified Farkas fallback.
#[test]
fn klein3_no_false_optimal_within_60s() {
    let (status, wall, _iters) = run_klein("data/lp_problems_infeas/klein3.QPS");
    assert_eq!(
        status,
        SolveStatus::Infeasible,
        "klein3 must be certified Infeasible"
    );
    assert!(
        wall < TIMEOUT_SEC,
        "klein3 wall {:.3}s exceeded {}s",
        wall,
        TIMEOUT_SEC
    );
}

/// Solver internal deadline for the anti-cycling sentinel. Kept strictly below
/// `KLEIN3_WALL_CEILING_SEC` so that when the solver runs to its deadline it
/// returns an honest Timeout *before* the wall assert — the gap
/// (`KLEIN3_WALL_CEILING_SEC - KLEIN3_SOLVER_DEADLINE_SEC`) is teardown/unwind
/// headroom under CI load. Equalizing the two (the previous design's 30s ==
/// 30s) makes the Timeout branch unreachable: a deadline hit pushes wall over
/// the ceiling and the assert fails, so honest Timeout was mislabeled runaway.
const KLEIN3_SOLVER_DEADLINE_SEC: f64 = 20.0;

/// Wall-clock runaway ceiling. Strictly above `KLEIN3_SOLVER_DEADLINE_SEC` so
/// both Infeasible (Farkas certified within the deadline) and honest Timeout
/// (deadline hit, then return) are reachable terminal states.
const KLEIN3_WALL_CEILING_SEC: f64 = 30.0;

/// Anti-cycling runaway sentinel: bland_mode 起動時に lex 摂動を注入することで
/// degeneracy が解消され、Bland's rule が klein3 を有限ステップで処理する。
/// runaway (finite-time 未終了) を wall ceiling で検出する。
///
/// 両終端が正当: Phase I が Optimal 到達 + Farkas 証明書通過なら Infeasible、
/// solver 内部期限に到達したら honest Timeout。旧仕様の strict
/// `status == Infeasible` は Big-M Phase I の「Optimal + artificials residual
/// → Infeasible」heuristic に依存した verdict (heuristic 自体が pilot 等で
/// false-Infeasible を生み撤去済)。
///
/// Replaces `klein3_infeasible_via_bland_anticycling` (deleted 2026-07-09),
/// whose solver deadline equaled its wall assert (both 30s) — a structural
/// defect that made the Timeout branch unreachable and flaked under load.
/// Measured 2026-07-09 with the 20s solver deadline: Infeasible, wall=14.6s
/// (iters=5889, deterministic) — well inside the 30s ceiling. The Timeout
/// branch was verified reachable by temporarily shrinking the deadline to 0.1s
/// (→ honest Timeout, wall well under the ceiling).
#[test]
fn klein3_anticycling_terminates_within_deadline() {
    let path = Path::new("data/lp_problems_infeas/klein3.QPS");
    assert!(path.exists(), "data missing: {}", path.display());
    let problem = parse_qps(path).expect("parse_qps");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(KLEIN3_SOLVER_DEADLINE_SEC);

    let t0 = Instant::now();
    let result = otspot::qp::solve_qp_with(&problem, &opts);
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "[klein3-bland] status={:?} wall={:.3}s iters={}",
        result.status, wall, result.iterations
    );

    assert!(
        matches!(
            result.status,
            SolveStatus::Infeasible | SolveStatus::Timeout
        ),
        "klein3 は Infeasible (Farkas certified) または Timeout (honest) で
         なければならない; got {:?}",
        result.status
    );
    assert!(
        wall < KLEIN3_WALL_CEILING_SEC,
        "klein3 wall {:.3}s — ceiling {}s を超えて runaway",
        wall,
        KLEIN3_WALL_CEILING_SEC
    );
}

/// Solver internal deadline for the no-presolve raw-path sentinel. Strictly
/// below `KLEIN3_NOPRESOLVE_CEILING_SEC` (same rationale as
/// `KLEIN3_SOLVER_DEADLINE_SEC`): a deadline-hit honest Timeout must return
/// before the wall assert. presolve=off の自然終了は Infeasible wall≈24.7s
/// (local) / 27.9s (CI) なので 45s は自然終了を観測でき、負荷で超過しても
/// 期限到達 Timeout として天井内に収まる。
const KLEIN3_NOPRESOLVE_DEADLINE_SEC: f64 = 45.0;

/// Wall-clock runaway ceiling for the no-presolve sentinel. Strictly above the
/// deadline so both Infeasible と honest Timeout が到達可能; 15s gap は teardown。
const KLEIN3_NOPRESOLVE_CEILING_SEC: f64 = 60.0;

/// klein3: presolve OFF で Big-M 直接実行 (presolve に頼らない raw path)。
/// presolve が infeasibility を捕まえないため Big-M Phase I が直接走る。
/// Infeasible (Farkas certified) と honest Timeout (solver 内部期限到達) の
/// 双方が正当な終端。Optimal/Unbounded は real bug。検証空白を作らないため
/// status を必ず assert する。
///
/// Replaces `diag_klein3_no_presolve` (deleted 2026-07-09), which routed
/// through `run_klein_with_presolve` (solver deadline == wall assert ==
/// TIMEOUT_SEC 60s) — the same structural defect as the old klein3 anticycling
/// test: a deadline-hit honest Timeout lands at wall ≈ 60s and trips the wall
/// assert, so the accepted-Timeout branch was unreachable. Here the solver
/// deadline (45s) is kept strictly below the wall ceiling (60s). Measured
/// 2026-07-09: Infeasible, wall=24.7s local / 27.9s CI (natural termination,
/// deadline not binding).
#[test]
fn klein3_no_presolve_terminates_within_deadline() {
    let path = Path::new("data/lp_problems_infeas/klein3.QPS");
    assert!(path.exists(), "data missing: {}", path.display());
    let problem = parse_qps(path).expect("parse_qps");
    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.timeout_secs = Some(KLEIN3_NOPRESOLVE_DEADLINE_SEC);

    let t0 = Instant::now();
    let result = otspot::qp::solve_qp_with(&problem, &opts);
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "[diag] klein3 no-presolve: status={:?} wall={:.3}s iters={}",
        result.status, wall, result.iterations
    );
    assert!(
        matches!(
            result.status,
            SolveStatus::Infeasible | SolveStatus::Timeout
        ),
        "klein3 no-presolve must be Infeasible (certified) or Timeout (honest); got {:?}",
        result.status
    );
    assert!(
        wall < KLEIN3_NOPRESOLVE_CEILING_SEC,
        "klein3 no-presolve wall {:.3}s — ceiling {}s を超えて runaway",
        wall,
        KLEIN3_NOPRESOLVE_CEILING_SEC
    );
}

/// Solver deadline for the Primal early-bail speedup sentinel. Generous so a
/// healthy-but-loaded run certifies Infeasible rather than being cut off.
const KLEIN3_SPEEDUP_DEADLINE_SEC: f64 = 60.0;

/// Wall ceiling for the speedup sentinel, strictly below the deadline. Healthy
/// klein3 (presolve on) certifies Infeasible at ~15s local / ~17.6s CI; 40s is
/// ~2.3× CI headroom while an early-bail regression burns the full deadline
/// (Big-M starved), landing far above 40s.
const KLEIN3_SPEEDUP_CEILING_SEC: f64 = 40.0;

/// LP cold-start (Ge/Eq) で `solve_dual_advanced` は
/// Primal (`two_phase_dual_simplex`) を deadline の半分で実行し、Timeout なら
/// Big-M Phase I へ fall back する。klein3 は Primal Phase I が cycling 確実な
/// degenerate infeasible LP で、Primal の早期 bail が効かないと Big-M に時間が
/// 残らない症状を持つ。本 test は Primal early-bail の effectiveness sentinel:
/// early-bail が効けば klein3 は deadline を大きく下回って Infeasible を証明する。
///
/// Replaces `klein3_primal_early_bail_speedup` (deleted 2026-07-09), which
/// accepted `Infeasible | Timeout` yet asserted `wall < 60s` where the solver
/// deadline was also 60s. Empirically (2026-07-09) klein3 presolve=on returns
/// Timeout only when the deadline is hit (wall ≈ deadline: 0.106s@0.1s,
/// 5.004s@5s), never below it — so the accepted-Timeout branch was unreachable
/// under the wall assert. A deadline-hit Timeout is precisely the regression
/// this sentinel must catch, so Timeout is not an accepted status here; the
/// healthy outcome is a fast Infeasible certificate. Solver deadline 60s, wall
/// ceiling 40s strictly below it (measured healthy Infeasible wall=14.97s
/// local / 17.6s CI).
#[test]
fn klein3_primal_early_bail_fast() {
    let path = Path::new("data/lp_problems_infeas/klein3.QPS");
    assert!(path.exists(), "data missing: {}", path.display());
    let problem = parse_qps(path).expect("parse_qps");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(KLEIN3_SPEEDUP_DEADLINE_SEC);

    let t0 = Instant::now();
    let result = otspot::qp::solve_qp_with(&problem, &opts);
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "[klein3-speedup] status={:?} wall={:.3}s iters={}",
        result.status, wall, result.iterations
    );

    assert_eq!(
        result.status,
        SolveStatus::Infeasible,
        "klein3 early-bail は Infeasible を証明すべき (Timeout = early-bail 退行で deadline 張り付き); got {:?}",
        result.status
    );
    assert!(
        wall < KLEIN3_SPEEDUP_CEILING_SEC,
        "klein3 wall {:.3}s — Primal early-bail 失敗で deadline に張り付いている (ceiling {}s)",
        wall,
        KLEIN3_SPEEDUP_CEILING_SEC
    );
}
