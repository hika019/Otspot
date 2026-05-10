//! Netlib LP additional coverage (Step 4 of LP coverage task).
//!
//! `tests/netlib_integration.rs` で既にカバーされている問題と重複しないよう、
//! 別の MPS feature / 規模を露出する問題を追加する。
//!
//! 入力: `data/lp_problems/<name>.QPS` (netlib_lp_download.sh で取得)
//! 経路: parse_qps + solve_qp_with — Q=0 の LP 問題は内部で simplex/IPM dispatch される。
//!
//! 実行: `cargo test --release --test netlib_lp_extra`

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::SolveStatus;
use solver::qp::solve_qp_with;
use std::path::Path;
use std::time::Instant;

const TIMEOUT_SEC: f64 = 60.0;
/// 相対誤差許容: 0.1% (タスク要件)
const REL_TOL: f64 = 1e-3;

fn solve_and_check(name: &str, expected_obj: f64, max_secs: u64) {
    let path_str = format!("data/lp_problems/{}.QPS", name);
    let path = Path::new(&path_str);
    if !path.exists() {
        eprintln!("[SKIP] {} not found at {}", name, path_str);
        return;
    }
    let problem = parse_qps(path).unwrap_or_else(|e| panic!("parse {} failed: {:?}", name, e));
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(TIMEOUT_SEC);
    let start = Instant::now();
    let result = solve_qp_with(&problem, &opts);
    let elapsed = start.elapsed();

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "{}: expected Optimal, got {:?} obj={} time={:?}",
        name,
        result.status,
        result.objective,
        elapsed
    );

    let denom = expected_obj.abs().max(1.0);
    let rel = (result.objective - expected_obj).abs() / denom;
    assert!(
        rel < REL_TOL,
        "{}: obj={:.6e} expected={:.6e} rel_err={:.2e} (>{:.0e}) time={:?}",
        name,
        result.objective,
        expected_obj,
        rel,
        REL_TOL,
        elapsed
    );

    assert!(
        elapsed.as_secs() < max_secs,
        "{}: solve time {:.2}s >= {}s budget",
        name,
        elapsed.as_secs_f64(),
        max_secs
    );

    println!(
        "{} solved: obj={:.6e} (expected {:.6e}, rel_err={:.2e}) time={:.2}s",
        name,
        result.objective,
        expected_obj,
        rel,
        elapsed.as_secs_f64()
    );
}

// === 中規模問題 (各 < 60s) ===
// 既存 netlib_integration.rs と重複しない問題のみ。

#[test]
fn test_lp_25fv47() {
    // 25fv47: 822 vars, 822 constraints. 中規模 LP の代表
    solve_and_check("25fv47", 5.5018458883e+03, 60);
}

#[test]
fn test_lp_agg() {
    // agg: 高条件数で知られる
    solve_and_check("agg", -3.5991767287e+07, 30);
}

#[test]
fn test_lp_agg2() {
    solve_and_check("agg2", -2.0239252356e+07, 30);
}

#[test]
fn test_lp_bandm() {
    // bandm: BOUNDS と E 制約混在
    solve_and_check("bandm", -1.5862801845e+02, 30);
}

#[test]
fn test_lp_beaconfd() {
    solve_and_check("beaconfd", 3.3592485807e+04, 30);
}

#[test]
fn test_lp_e226() {
    // e226: obj_offset 利用問題 (Netlib OBJ 定義に -7.113 オフセット)
    solve_and_check("e226", -2.5864929066e+01, 30);
}

#[test]
fn test_lp_etamacro() {
    solve_and_check("etamacro", -7.557152333e+02, 30);
}

#[test]
fn test_lp_finnis() {
    solve_and_check("finnis", 1.7279106560e+05, 30);
}

#[test]
fn test_lp_grow15() {
    // grow15 / grow22: BOUNDS で大量の負LO制約 (grow7 は netlib_integration.rs にあり)
    solve_and_check("grow15", -1.0687094129e+08, 30);
}

#[test]
fn test_lp_grow22() {
    solve_and_check("grow22", -1.6083433648e+08, 30);
}

#[test]
fn test_lp_scfxm1() {
    solve_and_check("scfxm1", 1.8416759028e+04, 30);
}

#[test]
fn test_lp_seba() {
    solve_and_check("seba", 1.5711600000e+04, 30);
}

#[test]
fn test_lp_shell() {
    solve_and_check("shell", 1.2088253460e+09, 30);
}

#[test]
fn test_lp_ship04l() {
    // ship04l: 2118 vars, 402 cons. 中規模、tight な BOUNDS UP 多数
    solve_and_check("ship04l", 1.7933245380e+06, 60);
}

#[test]
fn test_lp_ship04s() {
    solve_and_check("ship04s", 1.7987147004e+06, 30);
}

#[test]
fn test_lp_stair() {
    solve_and_check("stair", -2.5126695119e+02, 30);
}

#[test]
fn test_lp_standata() {
    solve_and_check("standata", 1.2576995e+03, 30);
}

#[test]
fn test_lp_stocfor2() {
    // stocfor2: 2031 vars, 2157 cons. 中規模 (stocfor1 は netlib_integration.rs にあり)
    solve_and_check("stocfor2", -3.9024408538e+04, 60);
}

#[test]
fn test_lp_tuff() {
    // tuff: 587 vars, 333 cons. 数値的に厄介
    solve_and_check("tuff", 2.9214776509e-01, 30);
}

// ==========================================================================
// === BUG regression test: forplan ===
// ==========================================================================

/// BUG: forplan が NumericalError obj=inf を返す。
/// boeing1 / capri と並ぶ既知バグ (lp_coverage_screen で検出)。
/// 推定原因: 大量 UP 境界 + 数値スケール広域 (1e0 ~ 1e7) で破綻。
/// 修正されたら #[ignore] を外す。
#[test]
#[ignore = "BUG: forplan NumericalError obj=inf. 大量UP+広域スケール"]
fn test_lp_forplan_bug() {
    solve_and_check("forplan", -6.6421873953e+02, 30);
}
