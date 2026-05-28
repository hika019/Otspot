//! user-specified timeout が halving されてしまうバグ regression sentinel。
//!
//! ## 観測事実 (bench_results/all_af7b1db/eps_1e-6 run.log L174/190/191)
//!
//! ```
//! neos       TIMEOUT  wall=750.432s  ([ipm] iters=19067)
//! rail2586   TIMEOUT  wall=752.728s  (        iters=42171)
//! rail4284   TIMEOUT  wall=753.805s  (        iters=31091)
//! ```
//!
//! bench は `--timeout 1000`。wall / budget ≈ 0.75 ということは budget が
//! 1 段ではなく 2 段で半分にされている。
//!
//! ## 真因
//!
//! cold-start Ge/Eq 経路 (`simplex/dual_advanced/mod.rs::solve_dual_advanced`):
//! 1. 外側: `clone_options_with_half_deadline` で Primal-first に budget/2 を割り当て。
//! 2. Primal が cycling 早期 bail 以外で Timeout (empty solution) → Big-M に fallback。
//! 3. 内側: `phase1::big_m_cold_start::clone_options_with_half_deadline` で
//!    残り (= budget/2) の半分を Phase I に割り当て。
//!    → 0.5 + 0.5×0.5 = 0.75 で終了。
//!
//! 既存の anti-cycling fix は Primal 早期 bail を追加したが
//! halving 自体は残した。slow-but-progress LP (rail*/neos) では Primal が早期
//! bail せず half budget を使い切るため、本症状が顕在化する。
//!
//! ## TDD 期待
//!
//! 同じ Big-M 経路を踏む `neos.QPS` (n=36786 m=479119) に budget=20s。
//! Timeout 返却時に `wall / budget ≥ TIMEOUT_HONOR_RATIO_MIN` を assert。
//! 修正前: 約 0.76 (RED), 修正後: 約 1.0 (GREEN)。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::solve_qp_with;
use std::path::Path;
use std::time::Instant;

/// `wall / budget` の許容下限。budget の 90% 未満を timeout 通知すれば bug 確定。
/// 10% slack は parse / postsolve / 終了判定の dispersion 吸収用。
const TIMEOUT_HONOR_RATIO_MIN: f64 = 0.90;

/// neos.QPS budget。20s で `solve_dual_advanced` の primal+big_m 経路を踏ませる
/// (#48 bench で 1000s → wall 750.432s の 0.75 比率が観測された問題)。
/// 3 分上限内 (CLAUDE.md) かつ parse (0.5s) + solve (20s) で実時間 < 25s。
const NEOS_BUDGET_SEC: f64 = 20.0;

#[test]
#[ignore = "requires data/lp_problems_hard/neos.QPS (heavy excluded from CI)"]
fn timeout_honored_on_neos_bigm_cold_start() {
    let path_str = "data/lp_problems_hard/neos.QPS";
    let path = Path::new(path_str);
    assert!(
        path.exists(),
        "data missing: {} — lp_problems_hard 必須",
        path_str
    );
    let problem = parse_qps(path).expect("parse_qps neos");

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(NEOS_BUDGET_SEC);

    let t0 = Instant::now();
    let result = solve_qp_with(&problem, &opts);
    let wall = t0.elapsed().as_secs_f64();
    let ratio = wall / NEOS_BUDGET_SEC;
    eprintln!(
        "[timeout-honor neos] status={:?} wall={:.3}s budget={}s ratio={:.3} iters={}",
        result.status, wall, NEOS_BUDGET_SEC, ratio, result.iterations
    );

    if result.status == SolveStatus::Timeout {
        assert!(
            ratio >= TIMEOUT_HONOR_RATIO_MIN,
            "user-specified timeout halving 残存: wall {:.3}s / budget {}s = {:.3} < {:.2}",
            wall, NEOS_BUDGET_SEC, ratio, TIMEOUT_HONOR_RATIO_MIN
        );
    }
}
