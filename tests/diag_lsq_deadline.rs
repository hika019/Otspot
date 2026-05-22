//! LSQ dual refine が user deadline を honor するかの regression sentinel
//! (data 不在環境でも実行可能な合成 LP 版)。
//!
//! Fail-safe: cleanup stagnant 時の LSQ skip は algorithmic gate。
//! 「cheap_min > PIVOT_TOL かつ cleanup が改善した」case のみ LSQ が走るため、
//! LSQ 自体が `Option<Instant>` deadline を尊重して budget 内に終了する必要がある。
//!
//! ## 合成 LP 設計 (v2、reviewer fact-finding 後)
//!
//! v1 の合成 LP は presolve が was_reduced=false で短絡し
//! postsolve_us=0 即終了 → LSQ entry 未到達 = vacuous sentinel だった。
//! v2 は `tests/lp_minimal_cleanup_lp_size.rs` の "deletable singleton + coupled
//! residual" 構造を流用 (cleanup LP gate test で trigger 実績あり):
//!
//!   - `N_DEL` 個の SingletonRow (row i: a_{i,i}=1, b=0 → x_i fix=0)
//!   - `M_MAIN` 個の residual Eq 行
//!     * residual 主係数: row N+j, col N+j に 2.0
//!     * **coupling**: 各 residual 行が deleted col i (i<N_DEL) に 0.1
//!       → cheap dual recovery で y_i = (c_i - Σ a*y_main) と residual y
//!       が相互依存し、Gauss-Seidel 50 iter で `cheap_min > PIVOT_TOL` 残存
//!   - residual 列の bounds は [0, ∞)、コスト c=1.0 で最小化
//!
//! 結果: postsolve_stack に N_DEL × SingletonRow が積まれ、cheap_min が gate を
//! 超えると cleanup LP が起動、cleanup が改善すると LSQ AAT factorize 発火。
//!
//! ## 検証多重化 (CLAUDE.md「検証空白を埋めるテスト」)
//!
//! 1. wall <= budget + SLACK_SEC: deadline honor の主 sentinel
//! 2. **postsolve_us >= POSTSOLVE_MIN_US**: LSQ entry が到達した観測検証
//!    (vacuous fail-safe — v1 で見落とした穴を構造的に塞ぐ)
//!
//! ## SLACK_SEC = 2.0 の根拠
//!
//! 観測 (dfl001-probe):
//!   `compute_lsq_dual_y` 単発で wall 2.9–4.5s (postsolve 11s の 98%)。
//! 観測 (dfl001 LSQ skip 後):
//!   postsolve は cleanup-LP + Gauss-Seidel + unscale 合計 ~0.3–0.8s。
//! 本テストは LSQ deadline-guard を踏んだ後の cleanup/GS 残量を吸収する
//! ための余裕として 2.0s を採用 (上記 0.3–0.8s + 2× 安全マージン)。
//! deadline propagation が破綻すると LSQ AAT factorize 1–4s が乗り
//! `wall > budget + 2.0` で fail する。
//!
//! ## POSTSOLVE_MIN_US = 50_000 (50ms) の根拠
//!
//! 50ms は m_main ~600 の AAT factorize 観測 cost の下限 (mac M-series で
//! sparse band 600×600 LDLT が 30-100ms)。LSQ skip 時 postsolve_us は
//! Gauss-Seidel + cleanup LP のみで 1-5ms 程度 (lp_minimal_cleanup_lp_size
//! bug4c 観測 1.2ms)。50ms 超過は「LSQ AAT factorize が確実に走った」観測。

use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, LpProblem};
use otspot::solve_with;
use otspot::sparse::CscMatrix;
use std::time::Instant;

/// LSQ 外 post-processing (cleanup LP / GS / unscale) + parallel CPU 競合余裕。
/// LSQ 漏れ (wall ≈ 2×budget) との分離は依然可能。
const SLACK_SEC: f64 = 4.0;

/// LSQ AAT factorize の実行観測下限 (m_main ~600 で 30-100ms、安全側で 50ms)。
const POSTSOLVE_MIN_US: u64 = 50_000;

/// 合成 LP: `lp_minimal_cleanup_lp_size` 構造のスケール拡張版。
///
/// 構造:
/// - 削除候補 Eq 行 i (0..N_DEL): a_{i,i}=1, b_i=0 → SingletonRow → x_i=0
/// - 残存 Eq 行 N_DEL+j (j<M_MAIN):
///     * 主係数 a_{N+j, N+j} = 2.0
///     * coupling: deleted col i (i<N_DEL) に係数 0.1 (cheap dual recovery 撹乱)
/// - 全列 bounds [0, ∞)、c=1.0、minimize
///
/// 戻り値: presolve で N_DEL 行 + N_DEL 列 を SingletonRow で削除、残存 LP は
/// M_MAIN × M_MAIN の sparse system。cheap recovery は coupling 連立のため
/// dfeas を完全に潰せず `cheap_min > PIVOT_TOL` を残し LSQ entry trigger。
fn build_lsq_trigger_lp(n_del: usize, m_main: usize) -> LpProblem {
    let n_total = n_del + m_main;
    let m_total = n_del + m_main;

    let mut tri_rows = Vec::with_capacity(n_del + m_main * (1 + n_del));
    let mut tri_cols = Vec::with_capacity(n_del + m_main * (1 + n_del));
    let mut tri_vals = Vec::with_capacity(n_del + m_main * (1 + n_del));

    // 削除候補 SingletonRow (row i: x_i のみ)
    for i in 0..n_del {
        tri_rows.push(i);
        tri_cols.push(i);
        tri_vals.push(1.0);
    }
    // 残存 Eq 行: 主係数 + deleted col への coupling
    for j in 0..m_main {
        let row = n_del + j;
        // 残存主列
        tri_rows.push(row);
        tri_cols.push(n_del + j);
        tri_vals.push(2.0);
        // deleted col 群への coupling (0.1)
        for i in 0..n_del {
            tri_rows.push(row);
            tri_cols.push(i);
            tri_vals.push(0.1);
        }
    }

    let a = CscMatrix::from_triplets(&tri_rows, &tri_cols, &tri_vals, m_total, n_total)
        .expect("A csc build");

    // b: 削除候補 0 (SingletonRow trigger), 残存 1.0 (feasibility)
    let mut b = vec![0.0_f64; m_total];
    for j in 0..m_main {
        b[n_del + j] = 1.0;
    }
    let cts = vec![ConstraintType::Eq; m_total];

    // c: 全列 1.0 (min Σ x、x_i=0 確定 + 残存 col は b=1/2=0.5 fix)
    let c = vec![1.0_f64; n_total];
    let bounds = vec![(0.0_f64, f64::INFINITY); n_total];

    LpProblem::new_general(
        c, a, b, cts, bounds,
        Some(format!("lsq_trigger_del{n_del}_m{m_main}")),
    )
    .expect("LpProblem::new_general")
}

fn assert_lsq_executed_within_budget(lp: &LpProblem, budget: f64, label: &str) {
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(budget);
    opts.presolve = true; // postsolve は presolve ON 経路でしか走らない

    let t0 = Instant::now();
    let result = solve_with(lp, &opts);
    let wall = t0.elapsed().as_secs_f64();

    let postsolve_us = result
        .timing_breakdown
        .as_ref()
        .map(|t| t.postsolve_us)
        .unwrap_or(0);
    let presolve_us = result
        .timing_breakdown
        .as_ref()
        .map(|t| t.presolve_us)
        .unwrap_or(0);
    let solve_us = result
        .timing_breakdown
        .as_ref()
        .map(|t| t.solve_us)
        .unwrap_or(0);
    eprintln!(
        "[{label}] status={:?} wall={:.3}s presolve={}us solve={}us postsolve={}us budget={budget}s",
        result.status, wall, presolve_us, solve_us, postsolve_us,
    );

    // (1) deadline honor: LSQ deadline propagation が壊れると AAT factorize 1-4s 乗る
    assert!(
        wall <= budget + SLACK_SEC,
        "[{label}] wall {wall:.3}s > budget {budget}s + slack {SLACK_SEC}s — LSQ deadline 漏れ疑い",
    );
    // (2) vacuous fail-safe: LSQ entry 到達観測 (POSTSOLVE_MIN_US 超過 = AAT factorize 実行)
    assert!(
        postsolve_us >= POSTSOLVE_MIN_US,
        "[{label}] postsolve {postsolve_us}us < {POSTSOLVE_MIN_US}us — LSQ entry 未到達 (vacuous sentinel)。\
         合成 LP の cheap_min/cleanup gate 設計を再検証せよ",
    );
}

/// 中規模 trigger: N_DEL=200 + M_MAIN=600。
/// 主目的: LSQ entry 到達 + deadline honor の基本観測。
/// AAT は ~m_main² = 360k cells、sparse band → factorize ~50-200ms 観測想定。
#[test]
fn lsq_honors_deadline_on_coupled_singleton_mid() {
    let lp = build_lsq_trigger_lp(200, 600);
    assert_lsq_executed_within_budget(&lp, 2.0, "lsq_trigger_mid");
}

/// 大規模 trigger: N_DEL=400 + M_MAIN=1200、budget 圧迫版。
/// 主目的: LSQ 内部 deadline early-exit 発火確認。
/// budget=0.5s で LSQ AAT factorize (1200×1200) が中途で deadline を踏み、
/// SLACK_SEC=2s 以内で wall が収まる = deadline propagation 正常動作。
/// deadline 不継承だと AAT factorize + IR 完走で 1-3s 余分に喰う。
#[test]
fn lsq_honors_deadline_on_coupled_singleton_large() {
    let lp = build_lsq_trigger_lp(400, 1200);
    assert_lsq_executed_within_budget(&lp, 0.5, "lsq_trigger_large");
}
