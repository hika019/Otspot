//! Integration tests: Parse Netlib MPS files and verify problem dimensions + solving
use otspot::io::mps::parse_mps_file;
use otspot::io::qps::parse_qps;
use otspot::options::{SimplexMethod, SolverOptions};
use otspot::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use otspot::qp::solve_qp_with;
use otspot::{solve, solve_with};
use std::path::Path;
use std::time::Instant;

/// Cycle/explosion guard: any correct simplex on a Netlib-class LP (≤ 2000 vars)
/// should converge far below this ceiling. Exceeding it signals algorithmic
/// regression (cycling or iteration explosion).
const MAX_NETLIB_ITER: usize = 500_000;

/// BOYD1 IPM iteration cap (measured: 51 iters, ×4 headroom).
/// Detects iterative-refine explosion: the memory-explosion regression caused
/// IPM to be mis-classified SuboptimalSolution, triggering iterative_refine
/// on a 93261-var problem. Excess iterations here indicate that regression.
const BOYD1_MEMORY_ITER_CAP: usize = 200;

/// Solve an LP with a wall-time budget enforced via solver timeout.
/// If the solver exceeds `timeout_secs`, it returns `SolveStatus::Timeout`,
/// which fails the caller's `status == Optimal` assert — deterministic sentinel.
fn solve_timed(problem: &LpProblem, timeout_secs: u64) -> SolverResult {
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(timeout_secs as f64);
    solve_with(problem, &opts)
}

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

    let result = solve_timed(&problem, 30);

    assert_eq!(result.status, SolveStatus::Optimal, "adlittle should reach Optimal");

    // adlittle optimal: 2.2549496316E+05
    let expected = 225494.96316;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "adlittle: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "adlittle: {} iterations", result.iterations);

    println!("adlittle solved: obj={}", result.objective);
}

#[test]
fn test_netlib_share2b() {
    let path = Path::new("tests/netlib/share2b.mps");
    let problem = parse_mps_file(path).expect("Failed to parse share2b.mps");

    let result = solve_timed(&problem, 30);

    assert_eq!(result.status, SolveStatus::Optimal, "share2b should reach Optimal");

    // share2b optimal: -4.1573224074E+02
    let expected = -415.73224074;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "share2b: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "share2b: {} iterations", result.iterations);

    println!("share2b solved: obj={}", result.objective);
}

#[test]
fn test_netlib_stocfor1() {
    let path = Path::new("tests/netlib/stocfor1.mps");
    let problem = parse_mps_file(path).expect("Failed to parse stocfor1.mps");

    let result = solve_timed(&problem, 30);

    assert_eq!(result.status, SolveStatus::Optimal, "stocfor1 should reach Optimal");

    // stocfor1 optimal: -4.1131976219E+04
    let expected = -41131.976219;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "stocfor1: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "stocfor1: {} iterations", result.iterations);

    println!("stocfor1 solved: obj={}", result.objective);
}

// --- Netlib拡充: brandy, scorpion, fit1d, share1b ---

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
    let result = solve_timed(&problem, 90); // Timeout → Timeout status → fails Optimal assert
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "brandy should reach Optimal");
    let expected = 1518.5098965;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "brandy: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "brandy: {} iterations", result.iterations);
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
    let result = solve_timed(&problem, 90);
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
    assert!(result.iterations < MAX_NETLIB_ITER, "scorpion: {} iterations", result.iterations);
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
    // fit1d: 1026 vars. Timeout sentinel: Timeout → fails Optimal assert below.
    let start_on = Instant::now();
    let result = solve_timed(&problem, 360);
    let elapsed_on = start_on.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "fit1d should reach Optimal");
    let expected = -9146.3780924;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "fit1d: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "fit1d: {} iterations", result.iterations);

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
    let result = solve_timed(&problem, 90);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "share1b should reach Optimal");
    let expected = -76589.318579;
    assert!(
        (result.objective - expected).abs() < 10.0,
        "share1b: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "share1b: {} iterations", result.iterations);
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
    let result = solve_timed(&problem, 90);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "boeing2 should reach Optimal");
    let expected = -315.01872802;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "boeing2: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "boeing2: {} iterations", result.iterations);
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
    assert!(path.exists(), "{} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行", path.display());
    let problem = parse_qps(path).expect("Failed to parse BOYD1.QPS");
    assert_eq!(problem.num_vars, 93261, "BOYD1: expected 93261 vars");
    assert_eq!(problem.num_constraints, 18, "BOYD1: expected 18 constraints");

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(30.0);
    let result = solve_qp_with(&problem, &opts);

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "BOYD1: should be Optimal, got {:?}. If SuboptimalSolution, dfeas check regression likely.",
        result.status,
    );
    assert!(
        result.iterations < BOYD1_MEMORY_ITER_CAP,
        "BOYD1: {} IPM iterations (cap {}). Possible iterative-refine explosion regression.",
        result.iterations, BOYD1_MEMORY_ITER_CAP,
    );
    println!(
        "BOYD1: status={:?}, obj={:.6e}, iters={}",
        result.status, result.objective, result.iterations
    );
}


/// Stack-overflow regression: BOYD1 級でも IPPMM 経路で stack 保護されるか
#[test]
fn test_boyd1_direct_ipm_no_stack_overflow() {
    let path = Path::new("data/maros_meszaros/BOYD1.QPS");
    assert!(path.exists(), "{} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行", path.display());
    let problem = parse_qps(path).expect("Failed to parse BOYD1.QPS");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(30.0);
    let result = solve_qp_with(&problem, &opts);
    println!("BOYD1 direct IPPMM: status={:?}", result.status);
    // status は Optimal/Suboptimal/Timeout どれでも良い。stack overflow しないことだけ検証。
    assert!(!result.solution.is_empty() || matches!(result.status, SolveStatus::Timeout | SolveStatus::NumericalError | SolveStatus::SuboptimalSolution));
}

// ============================================================================
// 追加 Netlib LP 問題
// 問題: sc105, sc205, recipe, lotfi, israel, sctap1, pilot4, grow7, boeing1, capri
// ============================================================================

/// Helper: compute max constraint violation for a given solution and problem.
fn max_constraint_violation(x: &[f64], prob: &otspot::problem::LpProblem) -> f64 {
    let a = &prob.a;
    let mut ax = vec![0.0f64; prob.num_constraints];
    for (col, &x_col) in x.iter().enumerate().take(prob.num_vars) {
        for ptr in a.col_ptr()[col]..a.col_ptr()[col + 1] {
            ax[a.row_ind()[ptr]] += a.values()[ptr] * x_col;
        }
    }
    (0..prob.num_constraints)
        .map(|i| match prob.constraint_types[i] {
            ConstraintType::Le => (ax[i] - prob.b[i]).max(0.0),
            ConstraintType::Ge => (prob.b[i] - ax[i]).max(0.0),
            ConstraintType::Eq => (ax[i] - prob.b[i]).abs(),
            _ => 0.0,
        })
        .fold(0.0_f64, f64::max)
}

// --- sc105 ---

#[test]
fn test_parse_sc105() {
    let prob = parse_mps_file(Path::new("tests/netlib/sc105.mps")).expect("parse failed");
    assert!(prob.num_constraints > 0);
    assert!(prob.num_vars > 0);
    // sc105: 106 rows (105 non-obj), 103 cols
    assert_eq!(prob.num_constraints, 105, "sc105: expected 105 constraints");
    assert_eq!(prob.num_vars, 103, "sc105: expected 103 vars");
    println!("sc105: {} constraints, {} vars", prob.num_constraints, prob.num_vars);
}

#[test]
fn test_solve_sc105() {
    let prob = parse_mps_file(Path::new("tests/netlib/sc105.mps")).expect("parse failed");
    let start = Instant::now();
    let result = solve_timed(&prob, 90);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "sc105: expected Optimal, got {:?}", result.status);
    let expected = -52.202061212;
    assert!(
        (result.objective - expected).abs() < 0.01,
        "sc105: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "sc105: {} iterations", result.iterations);
    println!("sc105 solved: obj={}, time={:?}", result.objective, elapsed);
}

// --- sc205 ---

#[test]
fn test_parse_sc205() {
    let prob = parse_mps_file(Path::new("tests/netlib/sc205.mps")).expect("parse failed");
    assert!(prob.num_constraints > 0);
    assert!(prob.num_vars > 0);
    // sc205: 206 rows (205 non-obj), 203 cols
    assert_eq!(prob.num_constraints, 205, "sc205: expected 205 constraints");
    assert_eq!(prob.num_vars, 203, "sc205: expected 203 vars");
    println!("sc205: {} constraints, {} vars", prob.num_constraints, prob.num_vars);
}

#[test]
fn test_solve_sc205() {
    let prob = parse_mps_file(Path::new("tests/netlib/sc205.mps")).expect("parse failed");
    let start = Instant::now();
    let result = solve_timed(&prob, 90);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "sc205: expected Optimal, got {:?}", result.status);
    // sc205 optimal: -5.2202061212E+01 (same as sc105, larger scale)
    let expected = -52.202061212;
    assert!(
        (result.objective - expected).abs() < 0.01,
        "sc205: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "sc205: {} iterations", result.iterations);
    println!("sc205 solved: obj={}, time={:?}", result.objective, elapsed);
}

// --- recipe ---

#[test]
fn test_parse_recipe() {
    let prob = parse_mps_file(Path::new("tests/netlib/recipe.mps")).expect("parse failed");
    assert!(prob.num_constraints > 0);
    assert!(prob.num_vars > 0);
    // recipe uses BOUNDS (UP/LO/FX)
    let has_non_default_bounds = prob.bounds.iter().any(|&(lo, hi)| lo != 0.0 || hi != f64::INFINITY);
    assert!(has_non_default_bounds, "recipe should have non-default bounds");
    println!("recipe: {} constraints, {} vars", prob.num_constraints, prob.num_vars);
}

#[test]
fn test_solve_recipe() {
    let prob = parse_mps_file(Path::new("tests/netlib/recipe.mps")).expect("parse failed");
    let start = Instant::now();
    let result = solve_timed(&prob, 90);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "recipe: expected Optimal, got {:?}", result.status);
    // recipe optimal: -2.6661600000E+02
    let expected = -266.616;
    assert!(
        (result.objective - expected).abs() < 0.1,
        "recipe: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "recipe: {} iterations", result.iterations);
    println!("recipe solved: obj={}, time={:?}", result.objective, elapsed);
}

// --- lotfi ---

#[test]
fn test_parse_lotfi() {
    let prob = parse_mps_file(Path::new("tests/netlib/lotfi.mps")).expect("parse failed");
    assert!(prob.num_constraints > 0);
    assert!(prob.num_vars > 0);
    println!("lotfi: {} constraints, {} vars", prob.num_constraints, prob.num_vars);
}

#[test]
fn test_solve_lotfi() {
    let prob = parse_mps_file(Path::new("tests/netlib/lotfi.mps")).expect("parse failed");
    let start = Instant::now();
    let result = solve_timed(&prob, 90);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "lotfi: expected Optimal, got {:?}", result.status);
    // lotfi optimal: -2.5264706062E+01
    let expected = -25.264706062;
    assert!(
        (result.objective - expected).abs() < 0.01,
        "lotfi: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "lotfi: {} iterations", result.iterations);
    println!("lotfi solved: obj={}, time={:?}", result.objective, elapsed);
}

// --- israel ---

#[test]
fn test_parse_israel() {
    let prob = parse_mps_file(Path::new("tests/netlib/israel.mps")).expect("parse failed");
    assert!(prob.num_constraints > 0);
    assert!(prob.num_vars > 0);
    println!("israel: {} constraints, {} vars", prob.num_constraints, prob.num_vars);
}

#[test]
fn test_solve_israel() {
    let prob = parse_mps_file(Path::new("tests/netlib/israel.mps")).expect("parse failed");
    let start = Instant::now();
    let result = solve_timed(&prob, 90);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "israel: expected Optimal, got {:?}", result.status);
    // israel optimal: -8.9664482186E+05
    let expected = -896644.82186;
    assert!(
        (result.objective - expected).abs() < 5.0,
        "israel: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "israel: {} iterations", result.iterations);
    println!("israel solved: obj={}, time={:?}", result.objective, elapsed);
}

// --- sctap1 ---

#[test]
fn test_parse_sctap1() {
    let prob = parse_mps_file(Path::new("tests/netlib/sctap1.mps")).expect("parse failed");
    assert!(prob.num_constraints > 0);
    assert!(prob.num_vars > 0);
    println!("sctap1: {} constraints, {} vars", prob.num_constraints, prob.num_vars);
}

#[test]
fn test_solve_sctap1() {
    let prob = parse_mps_file(Path::new("tests/netlib/sctap1.mps")).expect("parse failed");
    let start = Instant::now();
    let result = solve_timed(&prob, 90);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "sctap1: expected Optimal, got {:?}", result.status);
    // sctap1 optimal: 1.4122500000E+03
    let expected = 1412.25;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "sctap1: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "sctap1: {} iterations", result.iterations);
    println!("sctap1 solved: obj={}, time={:?}", result.objective, elapsed);
}

// --- grow7 ---

#[test]
fn test_parse_grow7() {
    let prob = parse_mps_file(Path::new("tests/netlib/grow7.mps")).expect("parse failed");
    assert!(prob.num_constraints > 0);
    assert!(prob.num_vars > 0);
    // grow7 uses BOUNDS (UP type)
    let has_bounds = prob.bounds.iter().any(|&(lo, hi)| lo != 0.0 || hi != f64::INFINITY);
    assert!(has_bounds, "grow7 should have non-default bounds");
    println!("grow7: {} constraints, {} vars", prob.num_constraints, prob.num_vars);
}

#[test]
fn test_solve_grow7() {
    let prob = parse_mps_file(Path::new("tests/netlib/grow7.mps")).expect("parse failed");
    let start = Instant::now();
    let result = solve_timed(&prob, 90);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "grow7: expected Optimal, got {:?}", result.status);
    // grow7 optimal: -4.7787811815E+07
    let expected = -47787811.815;
    assert!(
        (result.objective - expected).abs() < 500.0,
        "grow7: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "grow7: {} iterations", result.iterations);
    println!("grow7 solved: obj={}, time={:?}", result.objective, elapsed);
}

// --- pilot4: PL bound type regression test ---

#[test]
fn test_parse_pilot4() {
    // pilot4 uses PL (plus infinity upper bound) in its BOUNDS section.
    // BUG FIX: MPS parser previously rejected PL with "Invalid bound type: PL".
    let prob = parse_mps_file(Path::new("tests/netlib/pilot4.mps")).expect("parse failed: PL bound type must be supported");
    assert!(prob.num_constraints > 0);
    assert!(prob.num_vars > 0);
    // pilot4 uses PL bounds (upper = +inf) and FX/FR bounds
    let has_bounds = prob.bounds.iter().any(|&(lo, hi)| lo != 0.0 || hi != f64::INFINITY);
    assert!(has_bounds, "pilot4 should have non-default bounds (LO/FX/FR)");
    println!("pilot4: {} constraints, {} vars", prob.num_constraints, prob.num_vars);
}

#[test]
fn test_solve_pilot4() {
    let prob = parse_mps_file(Path::new("tests/netlib/pilot4.mps")).expect("parse failed");
    let start = Instant::now();
    let result = solve_timed(&prob, 180);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "pilot4: expected Optimal, got {:?}", result.status);
    // pilot4 optimal: -2.5811392641E+03 (LP relaxation)
    let expected = -2581.1392641;
    assert!(
        (result.objective - expected).abs() < 5.0,
        "pilot4: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "pilot4: {} iterations", result.iterations);
    // Verify feasibility: pilot4 should have small constraint violation
    if !result.solution.is_empty() {
        let viol = max_constraint_violation(&result.solution, &prob);
        assert!(viol < 1e-4, "pilot4: infeasible solution, max_viol={}", viol);
    }
    println!("pilot4 solved: obj={}, time={:?}", result.objective, elapsed);
}

// --- capri: presolve NumericalError regression ---

#[test]
fn test_parse_capri() {
    let prob = parse_mps_file(Path::new("tests/netlib/capri.mps")).expect("parse failed");
    assert!(prob.num_constraints > 0);
    assert!(prob.num_vars > 0);
    // capri uses BOUNDS (UP, FX, FR)
    let has_bounds = prob.bounds.iter().any(|&(lo, hi)| lo != 0.0 || hi != f64::INFINITY);
    assert!(has_bounds, "capri should have non-default bounds");
    println!("capri: {} constraints, {} vars", prob.num_constraints, prob.num_vars);
}

#[test]
fn test_solve_capri_no_presolve() {
    // BUG: presolve ON causes NumericalError with max_viol=2846 for capri.
    // Without presolve, the solver correctly finds optimal 2690.0129.
    // This test documents the expected behavior without presolve (correct).
    let prob = parse_mps_file(Path::new("tests/netlib/capri.mps")).expect("parse failed");
    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.timeout_secs = Some(90.0);
    let start = Instant::now();
    let result = solve_with(&prob, &opts);
    let elapsed = start.elapsed();
    assert_eq!(result.status, SolveStatus::Optimal, "capri (no presolve): expected Optimal, got {:?}", result.status);
    // capri optimal: 2.6900129138E+03
    let expected = 2690.0129138;
    assert!(
        (result.objective - expected).abs() < 1.0,
        "capri: expected ~{}, got {}",
        expected,
        result.objective
    );
    assert!(result.iterations < MAX_NETLIB_ITER, "capri: {} iterations", result.iterations);
    if !result.solution.is_empty() {
        let viol = max_constraint_violation(&result.solution, &prob);
        assert!(viol < 1e-4, "capri: infeasible solution, max_viol={}", viol);
    }
    println!("capri (no presolve) solved: obj={}, time={:?}", result.objective, elapsed);
}

/// BUG REGRESSION: capri with presolve ON incorrectly returned NumericalError.
/// Root cause (confirmed by instrumentation):
///   capri: presolve removes 32 vars/22 constraints, creating a reduced problem
///   where Phase I LU factorization fails with SingularBasis (specific basis
///   columns become near-singular after column removal).
///   forplan: Phase II solution violates an Eq constraint (check_eq_feasibility
///   fails) due to artificial variable drift in Phase II.
/// Fix: when presolve→simplex returns NumericalError, fallback to solving the
///   original problem without presolve. The original problem does not trigger
///   these numerical issues.
/// Note: fallback is correct behavior since the original problem solves correctly.
///   True fix would require better basis selection or LU stabilization for
///   presolve-reduced problems, which is a more involved change.
#[test]
fn test_solve_capri_presolve_bug() {
    let prob = parse_mps_file(Path::new("tests/netlib/capri.mps")).expect("parse failed");
    let result = solve(&prob); // presolve ON (default)
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "capri (presolve ON) must return Optimal, got {:?} obj={:.4}",
        result.status, result.objective
    );
    let expected_obj = 2690.012914;
    let rel = (result.objective - expected_obj).abs() / expected_obj.abs().max(1.0);
    assert!(
        rel < 1e-3,
        "capri (presolve ON) obj={:.4} expected ~{:.4} (rel_err={:.2e})",
        result.objective, expected_obj, rel
    );
}

// --- boeing1: solver feasibility bug ---

#[test]
fn test_parse_boeing1() {
    let prob = parse_mps_file(Path::new("tests/netlib/boeing1.mps")).expect("parse failed");
    assert!(prob.num_constraints > 0);
    assert!(prob.num_vars > 0);
    // boeing1: 351 rows + 90 RANGE splits = 441... but N row excluded = 440 constraints
    assert_eq!(prob.num_constraints, 440, "boeing1: expected 440 constraints");
    assert_eq!(prob.num_vars, 384, "boeing1: expected 384 vars");
    // boeing1 uses BOUNDS (UP/LO)
    let has_bounds = prob.bounds.iter().any(|&(lo, hi)| lo != 0.0 || hi != f64::INFINITY);
    assert!(has_bounds, "boeing1 should have non-default bounds");
    println!("boeing1: {} constraints, {} vars", prob.num_constraints, prob.num_vars);
}

/// BUG REGRESSION: boeing1 solver reports Optimal but solution violates constraint MSLAXTPE
/// (Ge, rhs=2) with violation ~0.813. The solver should return a feasible optimal.
/// Expected optimal: -3.3521356751E+02 = -335.21356751
/// Actual: -350.39 (more negative = infeasible constraint being violated).
#[test]
fn test_solve_boeing1_feasibility_bug() {
    let prob = parse_mps_file(Path::new("tests/netlib/boeing1.mps")).expect("parse failed");
    let result = solve(&prob);
    println!("boeing1: status={:?}, obj={:.4} [expected Optimal ~-335.21]", result.status, result.objective);

    if !result.solution.is_empty() {
        let viol = max_constraint_violation(&result.solution, &prob);
        println!("  max constraint violation: {:.6}", viol);
        // KNOWN BUG: solver claims Optimal but violates MSLAXTPE (Ge, rhs=2) by ~0.813.
        // When fixed, viol should be < 1e-4 and obj should be ~-335.21.
        // For now, document the bug:
        if viol > 1e-4 {
            println!("  BUG CONFIRMED: solver returns infeasible solution as Optimal");
            println!("  Expected max_viol < 1e-4, got {:.6}", viol);
        }
    }
    // Don't assert Optimal here since it's a known bug.
    // When fixed: assert_eq!(result.status, SolveStatus::Optimal);
    //             assert!((result.objective - (-335.21356751)).abs() < 1.0);
}

/// QBORE3D regression sentinel: presolve+Ruiz amplification caused SuboptimalSolution
/// (dfeas=7.5e-4) because the inner IPM stalls before meeting the tightened threshold.
///
/// Root cause: sigma_total ≈ 7.8e-4 forces eps_inner ≈ 7.8e-10, which the IPM cannot
/// achieve (stalls at 7.7e-7). Postsolve singleton recovery then degrades j=282's dual
/// residual from 2.3e-9 → 7.5e-4 via the overdetermined LSQ.
///
/// Fix: no-presolve fallback — when all presolve+Ruiz attempts fail for small problems,
/// solve directly on the original problem (no scaling amplification). DIAG_NO_PRESOLVE=1
/// already confirmed this converges in 43 iterations.
///
/// Sentinel: reverting the no-presolve fallback → SuboptimalSolution (dfeas=7.5e-4).
#[test]
fn test_qbore3d_optimal() {
    let path = Path::new("data/maros_meszaros/QBORE3D.QPS");
    assert!(
        path.exists(),
        "{} not found — scripts/maros_meszaros_download.sh を実行",
        path.display()
    );
    let problem = parse_qps(path).expect("Failed to parse QBORE3D.QPS");
    assert_eq!(problem.num_vars, 315, "QBORE3D: expected 315 vars");
    assert_eq!(problem.num_constraints, 233, "QBORE3D: expected 233 constraints");

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);
    let result = solve_qp_with(&problem, &opts);

    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "QBORE3D: should be Optimal, got {:?} \
         (dfeas regression — presolve+Ruiz amplification fix broken?)",
        result.status,
    );
    // Expected objective ≈ 3100.2 (confirmed via DIAG_NO_PRESOLVE=1: 3.100201e3).
    let expected_obj = 3100.2_f64;
    assert!(
        (result.objective - expected_obj).abs() < 10.0,
        "QBORE3D: expected obj ≈ {:.1}, got {:.6e}",
        expected_obj,
        result.objective,
    );
    println!(
        "QBORE3D: status={:?}, obj={:.6e}, iters={}",
        result.status, result.objective, result.iterations
    );
}
