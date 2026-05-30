//! Task #17 mini-corpus — **bug class 4**: cleanup LP の巨大化と deadline 不継承
//! (Task #3 / ken-18 真因対処済)。
//!
//! ## 構造的特徴
//!
//! `presolve::postsolve::build_and_solve_cleanup_lp` は削除行 k 個と未削除列 n 個に
//! 対して (m_clean, total_vars) ≈ (n, k + 2*k_eq) サイズの 2 段目 LP を組む。
//!
//! 旧バグ (ken-18, 95k 削除行 × 322k 列):
//!   - `cleanup_lp` の `timeout_secs = 5.0` (magic number) を SolverOptions に渡し
//!     ていたが、`simplex::solve_with` は `timeout_secs → deadline` 変換を
//!     **Ruiz scaling / standard-form build の後** に行うため、cleanup LP 構築の
//!     setup 段階だけで分単位の時間を浪費し parent deadline を完全無視。
//!   - 結果: bench 外部 `gtimeout` で SIGKILL → "異常終了" (group_failed)。
//!
//! ## 本 fix (5652027)
//!
//! `cleanup_lp` の `opts.deadline = deadline` (parent から継承)。Ruiz/build 等の
//! 内部 phase も同じ clock を見るため、ken-18 の cleanup LP は 5s 内に確実に
//! 短絡 (`Instant::now() >= deadline` で early return)。
//!
//! ## このテストの設計
//!
//! 元バグの「分単位浪費」を mini で再現するのは不可能 (削除行数 N×M スケール
//! 必須)。代わりに:
//!   (i) N=20 deletable row × M=20 col の中規模 LP を build し、
//!   (ii) `timeout_secs = 2.0` (ゆとり) で解いて Optimal を確認、
//!   (iii) cleanup LP が走った形跡を `timing_breakdown.postsolve_us` から観測。
//!
//! cleanup LP の deadline 継承が壊れた場合、N=20, M=20 の cleanup LP は
//! presolve_us 数百 ms に膨らむ (Ruiz が無駄なスケーリング) — 退行検知の
//! upper bound として 2.0s budget を採用。

use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot::solve_with;
use otspot::sparse::CscMatrix;

/// 削除行 N 個 + 残存行 M 個 + 列 (N + M) 個の合成 LP。
///
/// 構造:
/// - 各削除候補 Eq 行 i (0..N): a_{i, i} = 1, b_i = 0 → SingletonRow で削除
/// - 各残存 Eq 行 j (N..N+M): a_{j, N+j-N} = 1, plus 全列に微小係数 → 残存
/// - dummy col i (i<N): 削除 row i の対応 — SingletonRow で fix=0
///
/// 結果: postsolve_stack に SingletonRow × N、cleanup LP build で N × total_cols
/// の dual feasibility 拘束が積まれる。
fn build_deletable_workload(n_deletable: usize, m_remain: usize) -> LpProblem {
    let n_total = n_deletable + m_remain;
    let m = n_deletable + m_remain;

    let mut tri_rows = Vec::new();
    let mut tri_cols = Vec::new();
    let mut tri_vals = Vec::new();

    // 削除候補 Eq 行 (SingletonRow trigger): row i, col i, value 1
    for i in 0..n_deletable {
        tri_rows.push(i);
        tri_cols.push(i);
        tri_vals.push(1.0);
    }
    // 残存 Eq 行: row N+j, full row over [N, N+M) + 削除候補列にも微小値 (cleanup LP 連立を膨らます)
    for j in 0..m_remain {
        // 残存列 N+j 自身に主係数
        tri_rows.push(n_deletable + j);
        tri_cols.push(n_deletable + j);
        tri_vals.push(2.0);
        // 削除候補列 (i < N) にも微小係数 → cleanup LP で rc_known 計算を非自明化
        for i in 0..n_deletable {
            tri_rows.push(n_deletable + j);
            tri_cols.push(i);
            tri_vals.push(0.1);
        }
    }

    let a = CscMatrix::from_triplets(&tri_rows, &tri_cols, &tri_vals, m, n_total).unwrap();

    // b: 削除候補は 0 (SingletonRow trigger value=0)、残存は 1.0
    let mut b = vec![0.0_f64; m];
    for j in 0..m_remain {
        b[n_deletable + j] = 1.0;
    }
    let cts = vec![ConstraintType::Eq; m];

    // c: 全列 1.0 (min Σ x)
    let c = vec![1.0_f64; n_total];
    // bounds: 全列 [0, INF)
    let bounds = vec![(0.0_f64, f64::INFINITY); n_total];

    LpProblem::new_general(
        c,
        a,
        b,
        cts,
        bounds,
        Some(format!("cleanup_workload_N{}_M{}", n_deletable, m_remain)),
    )
    .unwrap()
}

/// 共通 assert: presolve ON で解き、timing_breakdown.postsolve_us と全体時間を
/// 観測する。budget 超過は cleanup LP deadline 継承の退行とみなす。
fn assert_cleanup_under_budget(lp: &LpProblem, budget_secs: f64, label: &str) {
    let mut opts = SolverOptions::default();
    opts.presolve = true; // cleanup LP は presolve ON 経路でしか走らない
    opts.timeout_secs = Some(budget_secs);

    let r = solve_with(lp, &opts);

    let pp_us = r
        .timing_breakdown
        .map(|t| t.postsolve_us)
        .unwrap_or(u64::MAX);
    let solve_us = r.timing_breakdown.map(|t| t.solve_us).unwrap_or(0);
    let pre_us = r.timing_breakdown.map(|t| t.presolve_us).unwrap_or(0);
    eprintln!(
        "[{}] status={:?} obj={:.3e} timing: presolve={}us solve={}us postsolve={}us",
        label, r.status, r.objective, pre_us, solve_us, pp_us
    );

    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "[{}] status={:?}",
        label,
        r.status
    );

    // postsolve_us が budget の 80% を超えたら deadline 継承が破綻している可能性。
    // 80% threshold は固定 (mini test の polynomial 性質を考えると 1 秒 budget で
    // 800ms 以上 cleanup LP に費やすのは異常)。
    let pp_secs = pp_us as f64 / 1_000_000.0;
    assert!(
        pp_secs < budget_secs * 0.8,
        "[{}] postsolve {:.3}s exceeds 80% of {:.3}s budget — cleanup LP deadline 退行?",
        label,
        pp_secs,
        budget_secs
    );
}

/// N=10 削除行 × M=10 残存行 (中規模)。timeout 1s で十分余裕。
#[test]
fn bug4a_cleanup_lp_n10_m10() {
    let lp = build_deletable_workload(10, 10);
    assert_cleanup_under_budget(&lp, 1.0, "bug4a_n10_m10");
}

/// N=20 × M=20。
#[test]
fn bug4b_cleanup_lp_n20_m20() {
    let lp = build_deletable_workload(20, 20);
    assert_cleanup_under_budget(&lp, 2.0, "bug4b_n20_m20");
}

/// N=50 × M=10 — 削除行多め (ken-18 type)。
#[test]
fn bug4c_cleanup_lp_many_deletable_n50_m10() {
    let lp = build_deletable_workload(50, 10);
    assert_cleanup_under_budget(&lp, 2.0, "bug4c_n50_m10_ken18_proxy");
}
