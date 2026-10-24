//! LISWET9 / YAO: objective must equal 0.5 x^T Q x + c^T x + obj_offset
//! evaluated at the returned solution (post-processing can mutate x).
//!
//! Also asserts internal objective is close to the HiGHS reference optimum.

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

/// internal objective が external reference (HiGHS Optimal 値) と乖離する場合、
/// solver が `Optimal` を主張しないことを assert。
fn assert_objective_matches_highs(name: &str, path: &std::path::Path, expected_internal: f64) {
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
        "{} vs HiGHS internal: status={:?}, ours_internal={:.6e}, highs={:.6e}, rel_diff={:.3e}",
        name, res.status, internal, expected_internal, rel
    );
    if rel >= 1e-3 {
        assert_ne!(
            res.status,
            otspot::problem::SolveStatus::Optimal,
            "{}: HiGHS 真値と rel={:.3e} 乖離しているのに Optimal を主張している",
            name,
            rel
        );
    }
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

/// LISWET9 の真値は HiGHS Optimal 値の目的定数を除いた部分 (`scripts/highs_baseline_truth.py`
/// で弱双対検算済みの baseline CSV 値 1963.2512609 から目的定数 2525.26 を引いた値)。
/// Clarabel は AlmostSolved 止まりで値が信頼できないため参照に使わない。
#[test]
fn liswet9_optimal_claim_matches_highs_truth() {
    assert_objective_matches_highs(
        "LISWET9",
        &std::path::PathBuf::from("data/maros_meszaros/LISWET9.QPS"),
        -562.0087391007,
    );
}

/// YAO の真値は HiGHS Optimal 値 −75.42074405968671 (目的定数なし)。HiGHS の (x, y, z) から
/// 再計算した弱双対で d = −75.42072898、相対 gap 2.0e-7、停留性残差 4.8e-8 を確認済み。
/// 旧参照値 −151.5405 (Clarabel AlmostSolved) はこの区間から 2 倍外れており使わない。
#[test]
fn yao_optimal_claim_matches_highs_truth() {
    assert_objective_matches_highs(
        "YAO",
        &std::path::PathBuf::from("data/maros_meszaros/YAO.QPS"),
        -75.42074405968671,
    );
}
