//! SolverResult.iterations が LP simplex 経路で populated されることを保証する
//! observability regression guard。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::solve_qp_with;
use std::path::Path;

/// afiro (32×27、軽量) で簡単な解が iter > 0 を持つことを保証。
/// LP simplex は必ず少なくとも 1 iter は回る (basis update / 最適性確認)。
#[test]
fn afiro_iterations_positive() {
    let path = Path::new("data/lp_problems/afiro.QPS");
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        path.display()
    );
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
        "afiro iterations={} must be > 0 (LP simplex counter regression guard)",
        r.iterations
    );
}

/// presolve=true でも iterations が populate される (postsolve 経由でも保持)。
#[test]
fn afiro_iterations_positive_with_presolve() {
    let path = Path::new("data/lp_problems/afiro.QPS");
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        path.display()
    );
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
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        path.display()
    );
    let prob = parse_qps(path).expect("parse perold");
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(60.0);
    let r = solve_qp_with(&prob, &opts);

    eprintln!("perold: status={:?} iterations={}", r.status, r.iterations);
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
