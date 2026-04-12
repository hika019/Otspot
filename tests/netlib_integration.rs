//! Integration tests: Parse Netlib MPS files and verify problem dimensions + solving
use solver::io::mps::parse_mps_file;
use solver::io::qps::parse_qps;
use solver::options::{SimplexMethod, SolverOptions};
use solver::problem::{ConstraintType, SolveStatus};
use solver::qp::solve_qp_with;
use solver::{solve, solve_with};
use std::path::Path;
use std::time::Instant;

#[test]
fn test_parse_afiro() {
    let path = Path::new("tests/netlib/afiro.mps");
    let problem = parse_mps_file(path).expect("Failed to parse afiro.mps");
    // afiro: 27 rows (excluding objective), 32 columns
    assert!(problem.num_constraints > 0, "afiro should have constraints");
    assert!(problem.num_vars > 0, "afiro should have variables");
    println!("afiro: {} constraints, {} vars", problem.num_constraints, problem.num_vars);
}

#[test]
fn test_parse_kb2() {
    let path = Path::new("tests/netlib/kb2.mps");
    let problem = parse_mps_file(path).expect("Failed to parse kb2.mps");
    assert!(problem.num_constraints > 0);
    assert!(problem.num_vars > 0);
    // kb2 uses BOUNDS heavily — verify bounds are not all default
    let has_non_default_bounds = problem.bounds.iter().any(|&(lo, hi)| lo != 0.0 || hi != f64::INFINITY);
    assert!(has_non_default_bounds, "kb2 should have non-default bounds");
    println!("kb2: {} constraints, {} vars", problem.num_constraints, problem.num_vars);
}

#[test]
fn test_parse_sc50a() {
    let path = Path::new("tests/netlib/sc50a.mps");
    let problem = parse_mps_file(path).expect("Failed to parse sc50a.mps");
    assert!(problem.num_constraints > 0);
    assert!(problem.num_vars > 0);
    println!("sc50a: {} constraints, {} vars", problem.num_constraints, problem.num_vars);
}

#[test]
fn test_parse_sc50b() {
    let path = Path::new("tests/netlib/sc50b.mps");
    let problem = parse_mps_file(path).expect("Failed to parse sc50b.mps");
    assert!(problem.num_constraints > 0);
    assert!(problem.num_vars > 0);
    println!("sc50b: {} constraints, {} vars", problem.num_constraints, problem.num_vars);
}

#[test]
fn test_parse_blend() {
    let path = Path::new("tests/netlib/blend.mps");
    let problem = parse_mps_file(path).expect("Failed to parse blend.mps");
    assert!(problem.num_constraints > 0);
    assert!(problem.num_vars > 0);
    // blend has equality constraints
    let has_eq = problem.constraint_types.contains(&ConstraintType::Eq);
    assert!(has_eq, "blend should have equality constraints");
    println!("blend: {} constraints, {} vars", problem.num_constraints, problem.num_vars);
}

// --- Solving tests ---

#[test]
fn test_solve_afiro() {
    let path = Path::new("tests/netlib/afiro.mps");
    let problem = parse_mps_file(path).expect("parse failed");
    let result = solve(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
    // afiro optimal: -4.6475314286E+02
    let expected = -464.7531428571429;
    assert!(
        (result.objective - expected).abs() < 0.01,
        "afiro: expected ~{}, got {}",
        expected,
        result.objective
    );
    println!("afiro solved: obj={}", result.objective);
}

#[test]
fn test_solve_kb2() {
    let path = Path::new("tests/netlib/kb2.mps");
    let problem = parse_mps_file(path).expect("parse failed");
    let result = solve(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
    // kb2 optimal: -1.7499001299E+03
    let expected = -1749.9001299;
    assert!(
        (result.objective - expected).abs() < 0.1,
        "kb2: expected ~{}, got {}",
        expected,
        result.objective
    );
    println!("kb2 solved: obj={}", result.objective);
}

#[test]
fn test_solve_sc50a() {
    let path = Path::new("tests/netlib/sc50a.mps");
    let problem = parse_mps_file(path).expect("parse failed");
    let result = solve(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
    // sc50a optimal: -6.4575077059E+01
    let expected = -64.575077059;
    assert!(
        (result.objective - expected).abs() < 0.01,
        "sc50a: expected ~{}, got {}",
        expected,
        result.objective
    );
    println!("sc50a solved: obj={}", result.objective);
}

#[test]
fn test_solve_sc50b() {
    let path = Path::new("tests/netlib/sc50b.mps");
    let problem = parse_mps_file(path).expect("parse failed");
    let result = solve(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
    // sc50b optimal: -7.0000000000E+01
    let expected = -70.0;
    assert!(
        (result.objective - expected).abs() < 0.01,
        "sc50b: expected ~{}, got {}",
        expected,
        result.objective
    );
    println!("sc50b solved: obj={}", result.objective);
}

#[test]
fn test_solve_blend() {
    let path = Path::new("tests/netlib/blend.mps");
    let problem = parse_mps_file(path).expect("parse failed");
    let result = solve(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
    // blend optimal: -3.0812149846E+01
    let expected = -30.812149846;
    assert!(
        (result.objective - expected).abs() < 0.01,
        "blend: expected ~{}, got {}",
        expected,
        result.objective
    );
    println!("blend solved: obj={}", result.objective);
}

// --- Medium-scale Netlib problems (cmd_068 Phase D2) ---

#[test]
fn test_netlib_adlittle() {
    let path = Path::new("tests/netlib/adlittle.mps");
    let problem = parse_mps_file(path).expect("Failed to parse adlittle.mps");

    let start = Instant::now();
    let result = solve(&problem);
    let elapsed = start.elapsed();

    assert_eq!(result.status, SolveStatus::Optimal, "adlittle should reach Optimal");

    // adlittle optimal: 2.2549496316E+05
    let expected = 225494.96316;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "adlittle: expected ~{}, got {}",
        expected,
        result.objective
    );

    assert!(
        elapsed.as_secs() < 10,
        "adlittle solve time should be < 10 sec, got {:?}",
        elapsed
    );

    println!("adlittle solved: obj={}, time={:?}", result.objective, elapsed);
}

#[test]
fn test_netlib_share2b() {
    let path = Path::new("tests/netlib/share2b.mps");
    let problem = parse_mps_file(path).expect("Failed to parse share2b.mps");

    let start = Instant::now();
    let result = solve(&problem);
    let elapsed = start.elapsed();

    assert_eq!(result.status, SolveStatus::Optimal, "share2b should reach Optimal");

    // share2b optimal: -4.1573224074E+02
    let expected = -415.73224074;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "share2b: expected ~{}, got {}",
        expected,
        result.objective
    );

    assert!(
        elapsed.as_secs() < 10,
        "share2b solve time should be < 10 sec, got {:?}",
        elapsed
    );

    println!("share2b solved: obj={}, time={:?}", result.objective, elapsed);
}

#[test]
fn test_netlib_stocfor1() {
    let path = Path::new("tests/netlib/stocfor1.mps");
    let problem = parse_mps_file(path).expect("Failed to parse stocfor1.mps");

    let start = Instant::now();
    let result = solve(&problem);
    let elapsed = start.elapsed();

    assert_eq!(result.status, SolveStatus::Optimal, "stocfor1 should reach Optimal");

    // stocfor1 optimal: -4.1131976219E+04
    let expected = -41131.976219;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "stocfor1: expected ~{}, got {}",
        expected,
        result.objective
    );

    assert!(
        elapsed.as_secs() < 10,
        "stocfor1 solve time should be < 10 sec, got {:?}",
        elapsed
    );

    println!("stocfor1 solved: obj={}, time={:?}", result.objective, elapsed);
}

// --- §4-2 Netlib拡充: brandy, scorpion, fit1d, share1b (cmd_089) ---

#[test]
fn test_parse_brandy() {
    let path = Path::new("tests/netlib/brandy.mps");
    let problem = parse_mps_file(path).expect("Failed to parse brandy.mps");
    assert!(problem.num_constraints > 0, "brandy should have constraints");
    assert!(problem.num_vars > 0, "brandy should have variables");
    println!("brandy: {} constraints, {} vars", problem.num_constraints, problem.num_vars);
}

#[test]
fn test_solve_brandy() {
    let path = Path::new("tests/netlib/brandy.mps");
    let problem = parse_mps_file(path).expect("parse failed");
    let start = Instant::now();
    let result = solve(&problem);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "brandy should reach Optimal");
    let expected = 1518.5098965;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "brandy: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(elapsed.as_secs() < 30, "brandy solve time < 30 sec, got {:?}", elapsed);
    println!("brandy solved: obj={}, time={:?}", result.objective, elapsed);
}

#[test]
fn test_parse_scorpion() {
    let path = Path::new("tests/netlib/scorpion.mps");
    let problem = parse_mps_file(path).expect("Failed to parse scorpion.mps");
    assert!(problem.num_constraints > 0, "scorpion should have constraints");
    assert!(problem.num_vars > 0, "scorpion should have variables");
    println!("scorpion: {} constraints, {} vars", problem.num_constraints, problem.num_vars);
}

#[test]
fn test_solve_scorpion() {
    let path = Path::new("tests/netlib/scorpion.mps");
    let problem = parse_mps_file(path).expect("parse failed");
    let start = Instant::now();
    let result = solve(&problem);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "scorpion should reach Optimal");
    let expected = 1878.1248227;
    // scorpionは条件数1.47×10^16の高退化問題（多くのソルバーで苦戦）。
    // 数値精度の限界により許容誤差を5.0に設定（相対誤差0.27%未満）。
    assert!(
        (result.objective - expected).abs() < 5.0,
        "scorpion: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(elapsed.as_secs() < 30, "scorpion solve time < 30 sec, got {:?}", elapsed);
    println!("scorpion solved: obj={}, time={:?}", result.objective, elapsed);
}

#[test]
fn test_parse_fit1d() {
    let path = Path::new("tests/netlib/fit1d.mps");
    let problem = parse_mps_file(path).expect("Failed to parse fit1d.mps");
    assert!(problem.num_constraints > 0, "fit1d should have constraints");
    assert!(problem.num_vars > 0, "fit1d should have variables");
    // fit1d uses BOUNDS (UP type) — verify non-default bounds exist
    let has_non_default_bounds = problem.bounds.iter().any(|&(lo, hi)| lo != 0.0 || hi != f64::INFINITY);
    assert!(has_non_default_bounds, "fit1d should have non-default bounds from BOUNDS section");
    println!("fit1d: {} constraints, {} vars", problem.num_constraints, problem.num_vars);
}

#[test]
fn test_solve_fit1d() {
    let path = Path::new("tests/netlib/fit1d.mps");
    let problem = parse_mps_file(path).expect("parse failed");

    // --- Presolve ON (デフォルト) ---
    let start_on = Instant::now();
    let result = solve(&problem);
    let elapsed_on = start_on.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "fit1d should reach Optimal");
    let expected = -9146.3780924;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "fit1d: expected ~{}, got {}",
        expected,
        result.objective
    );
    // fit1dは1026変数の大規模問題。Ruizスケーリング有効時はdebugモードで300秒程度。
    assert!(elapsed_on.as_secs() < 360, "fit1d solve time < 360 sec, got {:?}", elapsed_on);

    // --- Presolve OFF ---
    let mut opts_off = SolverOptions::default();
    opts_off.presolve = false;
    let start_off = Instant::now();
    let result_off = solve_with(&problem, &opts_off);
    let elapsed_off = start_off.elapsed();
    assert_eq!(result_off.status, SolveStatus::Optimal, "fit1d (no presolve) should reach Optimal");

    // タイミング比較出力
    eprintln!(
        "[fit1d presolve timing] WITH presolve: {:?}  WITHOUT presolve: {:?}  speedup: {:.2}x",
        elapsed_on,
        elapsed_off,
        elapsed_off.as_secs_f64() / elapsed_on.as_secs_f64().max(1e-6)
    );
    println!("fit1d solved: obj={}, time_with_presolve={:?}, time_without_presolve={:?}",
        result.objective, elapsed_on, elapsed_off);
}

#[test]
fn test_parse_share1b() {
    let path = Path::new("tests/netlib/share1b.mps");
    let problem = parse_mps_file(path).expect("Failed to parse share1b.mps");
    assert!(problem.num_constraints > 0, "share1b should have constraints");
    assert!(problem.num_vars > 0, "share1b should have variables");
    println!("share1b: {} constraints, {} vars", problem.num_constraints, problem.num_vars);
}

#[test]
fn test_solve_share1b() {
    let path = Path::new("tests/netlib/share1b.mps");
    let problem = parse_mps_file(path).expect("parse failed");
    let start = Instant::now();
    let result = solve(&problem);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "share1b should reach Optimal");
    let expected = -76589.318579;
    assert!(
        (result.objective - expected).abs() < 10.0,
        "share1b: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(elapsed.as_secs() < 30, "share1b solve time < 30 sec, got {:?}", elapsed);
    println!("share1b solved: obj={}, time={:?}", result.objective, elapsed);
}

#[test]
fn test_parse_boeing2() {
    let path = Path::new("tests/netlib/boeing2.mps");
    let problem = parse_mps_file(path).expect("Failed to parse boeing2.mps");
    // boeing2: 167 rows, 143 cols + RANGES追加行
    assert!(problem.num_constraints > 0, "boeing2 should have constraints");
    assert!(problem.num_vars > 0, "boeing2 should have variables");
    // RANGES使用問題: 通常より多くの制約行が生成される
    assert!(problem.num_constraints > 167, "boeing2 should have extra constraints from RANGES");
    println!("boeing2: {} constraints, {} vars", problem.num_constraints, problem.num_vars);
}

#[test]
fn test_solve_boeing2() {
    let path = Path::new("tests/netlib/boeing2.mps");
    let problem = parse_mps_file(path).expect("parse failed");
    let start = Instant::now();
    let result = solve(&problem);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "boeing2 should reach Optimal");
    let expected = -315.01872802;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "boeing2: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(elapsed.as_secs() < 30, "boeing2 solve time < 30 sec, got {:?}", elapsed);
    println!("boeing2 solved: obj={}, time={:?}", result.objective, elapsed);
}

// --- HIGH-2: Dual Simplex Netlib検証 (cmd_155) ---

#[test]
fn test_dual_simplex_netlib_1() {
    // afiro: 27 constraints, 32 vars。Dual Simplex強制で最適解を確認
    let path = Path::new("tests/netlib/afiro.mps");
    let problem = parse_mps_file(path).expect("Failed to parse afiro.mps");
    let mut opts = SolverOptions::default();
    opts.simplex_method = SimplexMethod::Dual;
    let start = Instant::now();
    let result = solve_with(&problem, &opts);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "afiro (Dual) should reach Optimal");
    let expected = -464.7531428571429;
    assert!(
        (result.objective - expected).abs() < 0.01,
        "afiro (Dual): expected ~{}, got {}",
        expected,
        result.objective
    );
    println!("afiro (Dual Simplex) solved: obj={}, time={:?}", result.objective, elapsed);
}

#[test]
fn test_dual_simplex_netlib_2() {
    // sc50a: 50 constraints, 48 vars。Dual Simplex強制で最適解を確認
    let path = Path::new("tests/netlib/sc50a.mps");
    let problem = parse_mps_file(path).expect("Failed to parse sc50a.mps");
    let mut opts = SolverOptions::default();
    opts.simplex_method = SimplexMethod::Dual;
    let start = Instant::now();
    let result = solve_with(&problem, &opts);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "sc50a (Dual) should reach Optimal");
    let expected = -64.575077059;
    assert!(
        (result.objective - expected).abs() < 0.01,
        "sc50a (Dual): expected ~{}, got {}",
        expected,
        result.objective
    );
    println!("sc50a (Dual Simplex) solved: obj={}, time={:?}", result.objective, elapsed);
}

#[test]
fn test_dual_simplex_netlib_3() {
    // adlittle: ~55 constraints, ~97 vars。Dual Simplex強制で最適解を確認
    let path = Path::new("tests/netlib/adlittle.mps");
    let problem = parse_mps_file(path).expect("Failed to parse adlittle.mps");
    let mut opts = SolverOptions::default();
    opts.simplex_method = SimplexMethod::Dual;
    let start = Instant::now();
    let result = solve_with(&problem, &opts);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "adlittle (Dual) should reach Optimal");
    let expected = 225494.96316;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "adlittle (Dual): expected ~{}, got {}",
        expected,
        result.objective
    );
    println!("adlittle (Dual Simplex) solved: obj={}, time={:?}", result.objective, elapsed);
}

// --- bore3d: QP Presolve Eq制約修正確認 (cmd_703) ---

#[test]
fn test_solve_bore3d_primal() {
    // T1: bore3d を Primal Simplex 経路（QP presolve → LP simplex）で解く
    // 正常修正後: Optimal, obj ≈ 1373.08
    let path = Path::new("tests/lp_problems/bore3d.QPS");
    let prob = parse_qps(path).expect("Failed to parse bore3d.QPS");
    let opts = SolverOptions::default();
    let start = Instant::now();
    let result = solve_qp_with(&prob, &opts);
    let elapsed = start.elapsed();
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "bore3d (Primal): expected Optimal, got {:?}",
        result.status
    );
    let expected = 1373.08;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "bore3d (Primal): expected obj ≈ {}, got {}",
        expected,
        result.objective
    );
    println!("bore3d (Primal Simplex) solved: obj={}, time={:?}", result.objective, elapsed);
}

#[test]
fn test_solve_bore3d_dual() {
    // T4: bore3d を Dual Simplex 経路で解く
    // 正常修正後: Optimal, obj ≈ 1373.08
    let path = Path::new("tests/lp_problems/bore3d.QPS");
    let prob = parse_qps(path).expect("Failed to parse bore3d.QPS");
    let mut opts = SolverOptions::default();
    opts.simplex_method = SimplexMethod::Dual;
    let start = Instant::now();
    let result = solve_qp_with(&prob, &opts);
    let elapsed = start.elapsed();
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "bore3d (Dual): expected Optimal, got {:?}",
        result.status
    );
    let expected = 1373.08;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "bore3d (Dual): expected obj ≈ {}, got {}",
        expected,
        result.objective
    );
    println!("bore3d (Dual Simplex) solved: obj={}, time={:?}", result.objective, elapsed);
}

/// BOYD1 regression test: 87GBメモリ爆発防止 [cmd_824]
///
/// BOYD1はb_max=3.75e12の巨大スケール問題（n=93261, m=18）。
/// cmd_800で追加されたmod.rs dfeasチェックが絶対閾値でRuizスケーリングの増幅を
/// 正しく扱えず、Optimalな解をSuboptimalSolutionと誤判定→iterative_refineが
/// n=93261のA^T*A構築で87GBメモリ爆発を起こしていた。
///
/// 修正: dfeas閾値を相対化（KKT項ノルムで正規化）。スケール非依存の判定。
///
/// このテストがタイムアウトまたはOOMで失敗した場合、dfeas閾値の退行を示す。
#[test]
fn test_boyd1_no_memory_explosion() {
    let path = Path::new("data/maros_meszaros/BOYD1.QPS");
    if !path.exists() {
        eprintln!("BOYD1.QPS not found, skipping");
        return;
    }
    let problem = parse_qps(path).expect("Failed to parse BOYD1.QPS");
    assert_eq!(problem.num_vars, 93261, "BOYD1: expected 93261 vars");
    assert_eq!(problem.num_constraints, 18, "BOYD1: expected 18 constraints");

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(30.0);
    let start = Instant::now();
    let result = solve_qp_with(&problem, &opts);
    let elapsed = start.elapsed();

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "BOYD1: should be Optimal, got {:?} (elapsed={:?}). If SuboptimalSolution, dfeas check regression likely.",
        result.status,
        elapsed
    );
    assert!(
        elapsed.as_secs() < 15,
        "BOYD1: should complete in <15s, took {:?}. Memory explosion may be occurring.",
        elapsed
    );
    println!(
        "BOYD1: status={:?}, obj={:.6e}, time={:.3}s",
        result.status, result.objective, elapsed.as_secs_f64()
    );
}

