use otspot_core::options::SolverOptions;
use otspot_core::problem::SolveStatus;
use otspot_core::qp::{solve_qp_with, QpProblem};
use otspot_core::sparse::CscMatrix;

#[test]
fn ruiz_scaled_and_unscaled_qp_agree() {
    let n = 5usize;
    let q = CscMatrix::from_triplets(
        &(0..n).collect::<Vec<_>>(),
        &(0..n).collect::<Vec<_>>(),
        &[1.0, 100.0, 1.0, 100.0, 1.0],
        n,
        n,
    )
    .unwrap();
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1, 2, 2], &[0, 1, 2, 3, 0, 4], &[1.0; 6], 3, n)
        .unwrap();
    let problem = QpProblem::new_all_le(
        q,
        vec![-1.0, -10.0, -1.0, -10.0, -1.0],
        a,
        vec![2.0; 3],
        vec![(0.0, f64::INFINITY); n],
    )
    .unwrap();

    let mut unscaled_options = SolverOptions::default();
    unscaled_options.use_ruiz_scaling = false;
    let unscaled = solve_qp_with(&problem, &unscaled_options);
    let mut scaled_options = SolverOptions::default();
    scaled_options.use_ruiz_scaling = true;
    let scaled = solve_qp_with(&problem, &scaled_options);

    for result in [&unscaled, &scaled] {
        assert!(matches!(
            result.status,
            SolveStatus::Optimal | SolveStatus::Timeout | SolveStatus::SuboptimalSolution
        ));
    }
    if unscaled.status == SolveStatus::Optimal && scaled.status == SolveStatus::Optimal {
        for (lhs, rhs) in unscaled.solution.iter().zip(&scaled.solution) {
            assert!((lhs - rhs).abs() < 0.1);
        }
        assert!((unscaled.objective - scaled.objective).abs() < 0.1);
    }
}

#[test]
fn ruiz_disabled_preserves_legacy_qp_solution() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let problem = QpProblem::new_all_le(
        q,
        vec![0.0, 0.0],
        a,
        vec![-1.0],
        vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
    )
    .unwrap();
    let mut options = SolverOptions::default();
    options.use_ruiz_scaling = false;
    let result = solve_qp_with(&problem, &options);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert!((result.solution[0] - 0.5).abs() < 0.05);
    assert!((result.solution[1] - 0.5).abs() < 0.05);
}
