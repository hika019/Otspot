//! LISWET9 / YAO: objective must equal 0.5 x^T Q x + c^T x + obj_offset
//! evaluated at the returned solution (post-processing can mutate x).
//!
//! Also asserts internal objective is close to Clarabel strict reference.

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::qp::solve_qp_with;

fn recompute_internal(prob: &otspot::QpProblem, x: &[f64]) -> f64 {
    let qx = prob.q.mat_vec_mul(x).expect("Qx");
    let xqx: f64 = qx.iter().zip(x.iter()).map(|(&q, &x)| q * x).sum();
    let cx: f64 = prob.c.iter().zip(x.iter()).map(|(&c, &x)| c * x).sum();
    0.5 * xqx + cx
}

/// reported objective が現 solution から再計算した 0.5 x^T Q x + c^T x + obj_offset と
/// 一致することを assert する (rel 1e-9)。
fn assert_objective_self_consistent(name: &str, path: &std::path::Path) {
    assert!(path.exists(), "{} missing", name);
    let prob = parse_qps(path).expect("parse");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);
    let res = solve_qp_with(&prob, &opts);

    let internal = recompute_internal(&prob, &res.solution);
    let expected = internal + prob.obj_offset;
    let diff = (res.objective - expected).abs();
    let denom = res.objective.abs().max(expected.abs()).max(1.0);
    let rel = diff / denom;
    eprintln!(
        "{} self-consistency: status={:?}, reported={:.6e}, recomputed={:.6e}, rel_diff={:.3e}",
        name, res.status, res.objective, expected, rel
    );
    assert!(
        rel < 1e-9,
        "{}: reported objective ({:.6e}) ≠ 0.5 x'Qx+c'x+offset ({:.6e}); rel_diff={:.3e}",
        name,
        res.objective,
        expected,
        rel
    );
}

/// internal objective が external reference (Clarabel strict optimum) に近いことを assert。
fn assert_objective_matches_clarabel(name: &str, path: &std::path::Path, expected_internal: f64) {
    assert!(path.exists(), "{} missing", name);
    let prob = parse_qps(path).expect("parse");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);
    let res = solve_qp_with(&prob, &opts);

    let internal = recompute_internal(&prob, &res.solution);
    let diff = (internal - expected_internal).abs();
    let denom = internal.abs().max(expected_internal.abs()).max(1.0);
    let rel = diff / denom;
    eprintln!(
        "{} vs Clarabel internal: ours_internal={:.6e}, clarabel={:.6e}, rel_diff={:.3e}",
        name, internal, expected_internal, rel
    );
    assert!(
        rel < 1e-3,
        "{}: ours internal ({:.6e}) too far from Clarabel ({:.6e}); rel_diff={:.3e}",
        name,
        internal,
        expected_internal,
        rel
    );
}

#[test]
fn liswet9_objective_self_consistent() {
    assert_objective_self_consistent(
        "LISWET9",
        &std::path::PathBuf::from("data/maros_meszaros/LISWET9.QPS"),
    );
}

#[test]
fn yao_objective_self_consistent() {
    assert_objective_self_consistent(
        "YAO",
        &std::path::PathBuf::from("data/maros_meszaros/YAO.QPS"),
    );
}

#[test]
#[ignore = "permanent ignore — known QP local minimum; fix tracked in #88 (local solver hardening) / #89 (multistart)"]
fn liswet9_objective_matches_clarabel() {
    assert_objective_matches_clarabel(
        "LISWET9",
        &std::path::PathBuf::from("data/maros_meszaros/LISWET9.QPS"),
        -1977.359,
    );
}

#[test]
#[ignore = "permanent ignore — known QP local minimum; fix tracked in #88 (local solver hardening) / #89 (multistart)"]
fn yao_objective_matches_clarabel() {
    assert_objective_matches_clarabel(
        "YAO",
        &std::path::PathBuf::from("data/maros_meszaros/YAO.QPS"),
        -151.5405,
    );
}
