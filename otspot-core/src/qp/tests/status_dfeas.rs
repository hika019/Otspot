use super::super::*;
use super::{assert_close, EPS};
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

/// 全変数固定退化ケース (presolve=false で本体検証)。
#[test]
fn test_qp_all_vars_fixed() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::new(0, 1);
    let b: Vec<f64> = vec![];
    let bounds = vec![(1.0_f64, 1.0_f64)];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![]).unwrap();

    let mut opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    opts.presolve = false;
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "got {:?}",
        result.status
    );
    assert_close(result.solution[0], 1.0, EPS, "x[0]");
}

/// 反復予算枯渇は MaxIterations として正直に報告される (Timeout / Suboptimal に
/// 丸めない)。postsolve が 1 iter の iterate を user_eps まで磨き切り証明できた
/// 場合のみ Optimal を許容する。非収束 iterate が SuboptimalSolution を名乗ったら
/// FAIL (attempt.rs finalize の品質ゲート sentinel)。
#[test]
fn test_max_iter_exhaustion_reports_honest_status() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ipm: crate::options::IpmOptions {
            max_iter: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert!(
        matches!(
            result.status,
            SolveStatus::Optimal | SolveStatus::MaxIterations
        ),
        "budget exhaustion must be honest MaxIterations (or proven Optimal), got {:?}",
        result.status
    );
}

/// 正常解で dfeas check が Optimal を維持。
#[test]
fn test_dfeas_optimal_preserved() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
}

/// スケール不変性 (1e6 倍) で Optimal 維持。
#[test]
fn test_dfeas_scale_invariant() {
    let scale = 1e6_f64;
    let q = CscMatrix::from_triplets(
        &[0, 1],
        &[0, 1],
        &[2.0 * scale * scale, 2.0 * scale * scale],
        2,
        2,
    )
    .unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-scale, -scale], 1, 2).unwrap();
    let b = vec![-scale];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "got {:?}",
        result.status
    );
    assert_close(result.solution[0], 0.5, 1e-4, "x[0]");
    assert_close(result.solution[1], 0.5, 1e-4, "x[1]");
}

/// 大 KKT スケール (2e12) でも相対閾値が正規化。
#[test]
fn test_dfeas_relative_threshold_large_kkt() {
    let n = 1usize;
    let q = CscMatrix::from_triplets(&[0], &[0], &[2e12], n, n).unwrap();
    let c = vec![-1e6];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "got {:?}",
        result.status
    );
    assert!(
        (result.solution[0] - 5e-7).abs() < 1e-9,
        "x*=5e-7, got {:.2e}",
        result.solution[0]
    );
}
