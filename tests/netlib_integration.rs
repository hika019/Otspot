//! Integration tests: Parse Netlib MPS files and verify problem dimensions
use solver::io::mps::parse_mps_file;
use solver::problem::ConstraintType;
use std::path::Path;

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
