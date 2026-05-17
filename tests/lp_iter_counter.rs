//! task #19: SolverResult.iterations が LP simplex 経路で populated されることを保証する。
//!
//! ## 真因 (bisecter task #16 で発見)
//! `pds-10` が `iters=0` のまま Optimal を返していた。LP simplex (primal.rs /
//! dual_advanced/core.rs) の main loop で `iteration` カウンタが SolverResult に
//! 伝播していなかった (Default::default() で 0 のまま)。
//!
//! ## 影響
//! 過去の bench/diag で `iters=0` を「solver が動いてない」と誤解させた:
//! - task #2 の fome12/ns1688926 解釈
//! - bisecter task #16 で d6cube/dfl001 が「setup hang」と誤推定
//!
//! ## 修正方針
//! `revised_simplex_core` / `dual_simplex_core_advanced` / `dual_simplex_core` に
//! `iter_count_out: &mut usize` を out-param で追加。各 main loop で
//! `*iter_count_out = iter_count_out.saturating_add(1)` を 1 行追加。
//! 上位の `two_phase_simplex` / `solve_dual_advanced` / `two_phase_dual_simplex`
//! で `total_iters` を累積し、最終的に `SolverResult.iterations` に格納。

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::SolveStatus;
use solver::qp::solve_qp_with;
use std::path::Path;

/// afiro (32×27、軽量) で簡単な解が iter > 0 を持つことを保証。
/// LP simplex は必ず少なくとも 1 iter は回る (basis update / 最適性確認)。
#[test]
fn afiro_iterations_positive() {
    let path = Path::new("data/lp_problems/afiro.QPS");
    if !path.exists() {
        eprintln!("[SKIP] afiro.QPS not found");
        return;
    }
    let prob = parse_qps(path).expect("parse afiro");
    let mut opts = SolverOptions::default();
    opts.presolve = false; // postsolve 経由を避け、純粋に simplex 経路のみ計測
    opts.timeout_secs = Some(10.0);
    let r = solve_qp_with(&prob, &opts);

    eprintln!(
        "afiro[presolve=off]: status={:?} obj={:.4e} iterations={}",
        r.status, r.objective, r.iterations
    );
    assert!(
        matches!(r.status, SolveStatus::Optimal),
        "afiro must be Optimal, got {:?}",
        r.status
    );
    assert!(
        r.iterations > 0,
        "afiro iterations={} must be > 0 (LP simplex counter bug regression防壁)",
        r.iterations
    );
}

/// presolve=true でも iterations が populate される (postsolve 経由でも保持)。
#[test]
fn afiro_iterations_positive_with_presolve() {
    let path = Path::new("data/lp_problems/afiro.QPS");
    if !path.exists() {
        eprintln!("[SKIP] afiro.QPS not found");
        return;
    }
    let prob = parse_qps(path).expect("parse afiro");
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(10.0);
    let r = solve_qp_with(&prob, &opts);

    eprintln!(
        "afiro[presolve=on]: status={:?} obj={:.4e} iterations={}",
        r.status, r.objective, r.iterations
    );
    assert!(
        matches!(r.status, SolveStatus::Optimal),
        "afiro must be Optimal, got {:?}",
        r.status
    );
    assert!(
        r.iterations > 0,
        "afiro[presolve=on] iterations={} must be > 0",
        r.iterations
    );
}

/// 中規模 LP (perold) でも iter > 0 (大規模 LP では iter は数百〜数千になる)。
#[test]
fn perold_iterations_positive() {
    let path = Path::new("data/lp_problems/perold.QPS");
    if !path.exists() {
        eprintln!("[SKIP] perold.QPS not found");
        return;
    }
    let prob = parse_qps(path).expect("parse perold");
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(60.0);
    let r = solve_qp_with(&prob, &opts);

    eprintln!(
        "perold: status={:?} iterations={}",
        r.status, r.iterations
    );
    assert!(
        matches!(r.status, SolveStatus::Optimal),
        "perold must be Optimal, got {:?}",
        r.status
    );
    assert!(
        r.iterations > 0,
        "perold iterations={} must be > 0",
        r.iterations
    );
}
