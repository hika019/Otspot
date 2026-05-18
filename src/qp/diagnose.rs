//! QP 問題の事前診断 API。

use super::problem::QpProblem;

const DIAG_TOL: f64 = 1e-10;
const BOUND_TOL: f64 = 1e-10;
/// IPM KKT 行列条件数の経験的許容上限。
const SCALE_WARN_THRESHOLD: f64 = 1e8;
const ZERO_B_TOL: f64 = 1e-12;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosticCode {
    QNegativeDiagonal,
    QNotSymmetric,
    VariableBoundsConflict,
    PoorScaling,
    ZeroRowInA,
    ProblemSize,
}

#[derive(Debug, Clone)]
pub struct DiagnosticWarning {
    pub code: DiagnosticCode,
    pub severity: Severity,
    pub message: String,
    pub variable_index: Option<usize>,
    pub constraint_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct ProblemInfo {
    pub n: usize,
    pub m: usize,
    pub nnz_q: usize,
    pub nnz_a: usize,
}

#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    pub warnings: Vec<DiagnosticWarning>,
    pub info: ProblemInfo,
    pub has_error: bool,
}

fn coefficient_ratio(values: &[f64]) -> Option<f64> {
    let mut max_v = 0.0_f64;
    let mut min_v = f64::INFINITY;
    for &v in values {
        let av = v.abs();
        if av > 1e-15 {
            if av > max_v {
                max_v = av;
            }
            if av < min_v {
                min_v = av;
            }
        }
    }
    if min_v == f64::INFINITY {
        None
    } else {
        Some(max_v / min_v)
    }
}

/// solve() 前の軽量チェック。コストは O(nnz_Q + nnz_A + n + m)。
pub fn diagnose(problem: &QpProblem) -> DiagnosticReport {
    let mut warnings: Vec<DiagnosticWarning> = Vec::new();

    for col in 0..problem.num_vars {
        let start = problem.q.col_ptr[col];
        let end = problem.q.col_ptr[col + 1];
        for k in start..end {
            if problem.q.row_ind[k] == col && problem.q.values[k] < -DIAG_TOL {
                warnings.push(DiagnosticWarning {
                    code: DiagnosticCode::QNegativeDiagonal,
                    severity: Severity::Error,
                    message: format!(
                        "Q[{},{}] = {:.6e} < 0: Q is not PSD",
                        col, col, problem.q.values[k]
                    ),
                    variable_index: Some(col),
                    constraint_index: None,
                });
            }
        }
    }

    let mut found_lower = false;
    'outer: for col in 0..problem.num_vars {
        let start = problem.q.col_ptr[col];
        let end = problem.q.col_ptr[col + 1];
        for k in start..end {
            if problem.q.row_ind[k] > col {
                found_lower = true;
                break 'outer;
            }
        }
    }
    if found_lower {
        warnings.push(DiagnosticWarning {
            code: DiagnosticCode::QNotSymmetric,
            severity: Severity::Warning,
            message: "Q has sub-diagonal entries: input may not be upper-triangular or symmetric"
                .to_string(),
            variable_index: None,
            constraint_index: None,
        });
    }

    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        if lb > ub + BOUND_TOL {
            warnings.push(DiagnosticWarning {
                code: DiagnosticCode::VariableBoundsConflict,
                severity: Severity::Error,
                message: format!(
                    "variable {}: lb ({:.6e}) > ub ({:.6e}): infeasible bounds",
                    j, lb, ub
                ),
                variable_index: Some(j),
                constraint_index: None,
            });
        }
    }

    if let Some(ratio) = coefficient_ratio(&problem.q.values) {
        if ratio > SCALE_WARN_THRESHOLD {
            warnings.push(DiagnosticWarning {
                code: DiagnosticCode::PoorScaling,
                severity: Severity::Warning,
                message: format!(
                    "Q coefficient ratio = {:.2e} > {:.2e}: poor scaling may cause numerical issues",
                    ratio, SCALE_WARN_THRESHOLD
                ),
                variable_index: None,
                constraint_index: None,
            });
        }
    }
    if let Some(ratio) = coefficient_ratio(&problem.a.values) {
        if ratio > SCALE_WARN_THRESHOLD {
            warnings.push(DiagnosticWarning {
                code: DiagnosticCode::PoorScaling,
                severity: Severity::Warning,
                message: format!(
                    "A coefficient ratio = {:.2e} > {:.2e}: poor scaling may cause numerical issues",
                    ratio, SCALE_WARN_THRESHOLD
                ),
                variable_index: None,
                constraint_index: None,
            });
        }
    }
    if let Some(ratio) = coefficient_ratio(&problem.c) {
        if ratio > SCALE_WARN_THRESHOLD {
            warnings.push(DiagnosticWarning {
                code: DiagnosticCode::PoorScaling,
                severity: Severity::Warning,
                message: format!(
                    "c coefficient ratio = {:.2e} > {:.2e}: poor scaling may cause numerical issues",
                    ratio, SCALE_WARN_THRESHOLD
                ),
                variable_index: None,
                constraint_index: None,
            });
        }
    }

    if problem.num_constraints > 0 {
        let mut row_has_nonzero = vec![false; problem.num_constraints];
        for &row in &problem.a.row_ind {
            row_has_nonzero[row] = true;
        }
        for (i, &present) in row_has_nonzero.iter().enumerate() {
            if !present {
                let severity = if problem.b[i] < -ZERO_B_TOL {
                    Severity::Error
                } else {
                    Severity::Warning
                };
                let msg = if severity == Severity::Error {
                    format!(
                        "constraint {}: zero row in A with b[{}] = {:.6e} < 0: infeasible (0 <= {})",
                        i, i, problem.b[i], problem.b[i]
                    )
                } else {
                    format!(
                        "constraint {}: zero row in A with b[{}] = {:.6e} >= 0: redundant constraint",
                        i, i, problem.b[i]
                    )
                };
                warnings.push(DiagnosticWarning {
                    code: DiagnosticCode::ZeroRowInA,
                    severity,
                    message: msg,
                    variable_index: None,
                    constraint_index: Some(i),
                });
            }
        }
    }

    let info = ProblemInfo {
        n: problem.num_vars,
        m: problem.num_constraints,
        nnz_q: problem.q.nnz(),
        nnz_a: problem.a.nnz(),
    };
    warnings.push(DiagnosticWarning {
        code: DiagnosticCode::ProblemSize,
        severity: Severity::Info,
        message: format!(
            "problem size: n={}, m={}, nnz_Q={}, nnz_A={}",
            info.n, info.m, info.nnz_q, info.nnz_a
        ),
        variable_index: None,
        constraint_index: None,
    });

    let has_error = warnings.iter().any(|w| w.severity == Severity::Error);

    DiagnosticReport { warnings, info, has_error }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;
    use crate::qp::problem::QpProblem;

    fn make_simple_problem() -> QpProblem {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        QpProblem::new_all_le(q, c, a, b, bounds).unwrap()
    }

    #[test]
    fn test_q_negative_diagonal_clean() {
        let prob = make_simple_problem();
        let report = diagnose(&prob);
        assert!(!report.has_error);
        let neg = report.warnings.iter().any(|w| w.code == DiagnosticCode::QNegativeDiagonal);
        assert!(!neg);
    }

    #[test]
    fn test_q_negative_diagonal_detected() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let report = diagnose(&prob);
        assert!(report.has_error);
        let w = report.warnings.iter().find(|w| w.code == DiagnosticCode::QNegativeDiagonal);
        assert!(w.is_some());
        assert_eq!(w.unwrap().variable_index, Some(0));
    }

    #[test]
    fn test_q_symmetric_clean() {
        let prob = make_simple_problem();
        let report = diagnose(&prob);
        let w = report.warnings.iter().any(|w| w.code == DiagnosticCode::QNotSymmetric);
        assert!(!w);
    }

    #[test]
    fn test_q_not_symmetric_detected() {
        let q = CscMatrix::from_triplets(
            &[0, 1, 0, 1],
            &[0, 0, 1, 1],
            &[2.0, 1.0, 1.0, 2.0],
            2, 2,
        ).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let report = diagnose(&prob);
        let w = report.warnings.iter().find(|w| w.code == DiagnosticCode::QNotSymmetric);
        assert!(w.is_some());
        assert_eq!(w.unwrap().severity, Severity::Warning);
    }

    #[test]
    fn test_bounds_conflict_clean() {
        let prob = make_simple_problem();
        let report = diagnose(&prob);
        let w = report.warnings.iter().any(|w| w.code == DiagnosticCode::VariableBoundsConflict);
        assert!(!w);
    }

    #[test]
    fn test_bounds_conflict_detected() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0, 1.0), (2.0, 1.0)];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let report = diagnose(&prob);
        assert!(report.has_error);
        let w = report.warnings.iter().find(|w| w.code == DiagnosticCode::VariableBoundsConflict);
        assert!(w.is_some());
        assert_eq!(w.unwrap().variable_index, Some(1));
    }

    #[test]
    fn test_poor_scaling_clean() {
        let prob = make_simple_problem();
        let report = diagnose(&prob);
        let w = report.warnings.iter().any(|w| w.code == DiagnosticCode::PoorScaling);
        assert!(!w);
    }

    #[test]
    fn test_poor_scaling_detected() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1e10, 1.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let report = diagnose(&prob);
        let w = report.warnings.iter().find(|w| w.code == DiagnosticCode::PoorScaling);
        assert!(w.is_some());
        assert_eq!(w.unwrap().severity, Severity::Warning);
    }

    #[test]
    fn test_zero_row_in_a_clean() {
        let prob = make_simple_problem();
        let report = diagnose(&prob);
        let w = report.warnings.iter().any(|w| w.code == DiagnosticCode::ZeroRowInA);
        assert!(!w);
    }

    #[test]
    fn test_zero_row_in_a_warning() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[1, 1], &[0, 1], &[-1.0, -1.0], 2, 2).unwrap();
        let b = vec![0.0, -1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let report = diagnose(&prob);
        let w = report.warnings.iter().find(|w| w.code == DiagnosticCode::ZeroRowInA);
        assert!(w.is_some());
        assert_eq!(w.unwrap().severity, Severity::Warning);
        assert_eq!(w.unwrap().constraint_index, Some(0));
    }

    #[test]
    fn test_zero_row_in_a_error() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[1, 1], &[0, 1], &[-1.0, -1.0], 2, 2).unwrap();
        let b = vec![-1.0, -1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let report = diagnose(&prob);
        assert!(report.has_error);
        let w = report.warnings.iter().find(|w| w.code == DiagnosticCode::ZeroRowInA);
        assert!(w.is_some());
        assert_eq!(w.unwrap().severity, Severity::Error);
    }

    #[test]
    fn test_problem_size_always_present() {
        let prob = make_simple_problem();
        let report = diagnose(&prob);
        assert_eq!(report.info.n, 2);
        assert_eq!(report.info.m, 1);
        assert_eq!(report.info.nnz_q, 2);
        assert_eq!(report.info.nnz_a, 2);
        let w = report.warnings.iter().find(|w| w.code == DiagnosticCode::ProblemSize);
        assert!(w.is_some());
        assert_eq!(w.unwrap().severity, Severity::Info);
    }

    #[test]
    fn test_multiple_errors_combined() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-1.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(5.0, 1.0), (0.0, 1.0)];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let report = diagnose(&prob);
        assert!(report.has_error);
        let errors: Vec<_> = report.warnings.iter()
            .filter(|w| w.severity == Severity::Error)
            .collect();
        assert!(errors.len() >= 2);
    }
}
