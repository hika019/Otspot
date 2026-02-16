//! Integration tests: Parse Netlib MPS files and verify problem dimensions + solving
use solver::io::mps::parse_mps_file;
use solver::problem::{ConstraintType, SolveStatus};
use solver::simplex::solve;
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
    let has_eq = problem.constraint_types.iter().any(|ct| *ct == ConstraintType::Eq);
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
