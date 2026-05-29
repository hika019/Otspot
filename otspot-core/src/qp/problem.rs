//! QP問題のデータ構造定義。

use crate::problem::{ConstraintType, SolveStatus};
use crate::sparse::CscMatrix;

/// Quadratic term storage for a single QCQP constraint.
///
/// Uses COO (coordinate) format: symmetrized `(row, col, val)` triplets.
/// Memory is O(nnz), unlike `CscMatrix` which requires O(n) for `col_ptr`.
#[derive(Debug, Clone, Default)]
pub struct QcqpMatrix {
    /// Problem dimension (n×n symmetric matrix).
    pub n: usize,
    /// Symmetrized COO triplets: (row, col, value), 0-indexed.
    pub triplets: Vec<(usize, usize, f64)>,
}

impl QcqpMatrix {
    /// Creates an empty n×n matrix with no non-zero entries.
    pub fn new(n: usize) -> Self {
        Self { n, triplets: Vec::new() }
    }

    /// Returns the number of stored entries (after symmetrization).
    pub fn nnz(&self) -> usize {
        self.triplets.len()
    }
}

#[non_exhaustive]
#[derive(Debug)]
pub enum QpProblemError {
    DimensionMismatch(String),
    /// Non-finite coefficient (NaN or ±∞) in the named field at the given index.
    NonFiniteCoefficient { field: &'static str, index: usize },
    /// Invalid variable bound: NaN or lb > ub at the given index.
    InvalidBounds { index: usize, lb: f64, ub: f64 },
    /// A triplet (row, col) index exceeds the matrix dimension.
    TripletIndexOutOfBounds { constraint: usize, row: usize, col: usize, n: usize },
}

impl std::fmt::Display for QpProblemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QpProblemError::DimensionMismatch(msg) => write!(f, "dimension mismatch: {}", msg),
            QpProblemError::NonFiniteCoefficient { field, index } => {
                write!(f, "non-finite coefficient in {}: index {}", field, index)
            }
            QpProblemError::InvalidBounds { index, lb, ub } => {
                write!(f, "invalid bounds at index {}: lb={} > ub={} or NaN", index, lb, ub)
            }
            QpProblemError::TripletIndexOutOfBounds { constraint, row, col, n } => {
                write!(f, "triplet index out of bounds in constraint {}: ({},{}) >= n={}", constraint, row, col, n)
            }
        }
    }
}

impl std::error::Error for QpProblemError {}

/// min 1/2 x^T Q x + c^T x  s.t. Ax {<=,=,>=} b, lb <= x <= ub
///
/// When `quadratic_constraints` is non-empty the problem is a QCQP.
/// Entry `k` holds the symmetric `n×n` matrix `Q_k` for the quadratic part
/// of constraint `k`: `1/2 x^T Q_k x + a_k^T x {<=,=,>=} b_k`.
/// An empty `QcqpMatrix` at index `k` means constraint `k` has no quadratic part.
#[derive(Debug, Clone)]
pub struct QpProblem {
    pub q: CscMatrix,
    pub c: Vec<f64>,
    pub a: CscMatrix,
    pub b: Vec<f64>,
    pub bounds: Vec<(f64, f64)>,
    pub num_vars: usize,
    pub num_constraints: usize,
    pub constraint_types: Vec<ConstraintType>,
    /// Per-constraint quadratic matrices for QCQP, stored in COO format.
    ///
    /// Length is either 0 (pure QP/LP) or `num_constraints` (QCQP).
    /// Entry `k` holds symmetrized triplets for `Q_k`; empty means no quadratic part.
    pub quadratic_constraints: Vec<QcqpMatrix>,
    /// 目的関数値 = 1/2 x^T Q x + c^T x + obj_offset
    pub obj_offset: f64,
}

impl QpProblem {
    pub fn new(
        q: CscMatrix,
        c: Vec<f64>,
        a: CscMatrix,
        b: Vec<f64>,
        bounds: Vec<(f64, f64)>,
        constraint_types: Vec<ConstraintType>,
    ) -> Result<Self, QpProblemError> {
        let n = c.len();
        let m = b.len();
        if q.nrows != n || q.ncols != n {
            return Err(QpProblemError::DimensionMismatch(
                format!("Q must be {}x{}, got {}x{}", n, n, q.nrows, q.ncols)
            ));
        }
        if a.nrows != m || a.ncols != n {
            return Err(QpProblemError::DimensionMismatch(
                format!("A must be {}x{}, got {}x{}", m, n, a.nrows, a.ncols)
            ));
        }
        if bounds.len() != n {
            return Err(QpProblemError::DimensionMismatch(
                format!("bounds length must be {}, got {}", n, bounds.len())
            ));
        }
        if constraint_types.len() != m {
            return Err(QpProblemError::DimensionMismatch(
                format!("constraint_types length must be {}, got {}", m, constraint_types.len())
            ));
        }
        for (i, &v) in c.iter().enumerate() {
            if !v.is_finite() {
                return Err(QpProblemError::NonFiniteCoefficient { field: "c", index: i });
            }
        }
        for (i, &v) in b.iter().enumerate() {
            if !v.is_finite() {
                return Err(QpProblemError::NonFiniteCoefficient { field: "b", index: i });
            }
        }
        for (i, &v) in q.values.iter().enumerate() {
            if !v.is_finite() {
                return Err(QpProblemError::NonFiniteCoefficient { field: "Q", index: i });
            }
        }
        for (i, &v) in a.values.iter().enumerate() {
            if !v.is_finite() {
                return Err(QpProblemError::NonFiniteCoefficient { field: "A", index: i });
            }
        }
        for (i, &(lb, ub)) in bounds.iter().enumerate() {
            if lb.is_nan() || ub.is_nan() || lb > ub {
                return Err(QpProblemError::InvalidBounds { index: i, lb, ub });
            }
        }
        Ok(QpProblem { q, c, a, b, bounds, num_vars: n, num_constraints: m, constraint_types, quadratic_constraints: vec![], obj_offset: 0.0 })
    }

    /// Convenience constructor: all constraints are `≤`.
    #[doc(hidden)]
    pub fn new_all_le(
        q: CscMatrix,
        c: Vec<f64>,
        a: CscMatrix,
        b: Vec<f64>,
        bounds: Vec<(f64, f64)>,
    ) -> Result<Self, QpProblemError> {
        let m = b.len();
        Self::new(q, c, a, b, bounds, vec![ConstraintType::Le; m])
    }

    /// Q が全ゼロかどうかを検査する（LP退化ケース判定）
    pub fn is_zero_q(&self) -> bool {
        self.q.values.iter().all(|&v| v.abs() < 1e-12)
    }

    /// Returns `true` if the problem has at least one constraint with a non-zero quadratic term.
    ///
    /// Used as a guard: the QP/LP solver cannot handle QCQP and must reject such problems.
    pub fn has_qcqp_constraints(&self) -> bool {
        self.quadratic_constraints.iter().any(|q| !q.triplets.is_empty())
    }

    /// Set per-constraint quadratic matrices for QCQP, with validation.
    ///
    /// `qcs` must be either empty (pure QP) or have length equal to `num_constraints`.
    /// Each `QcqpMatrix` must have `n == num_vars`, finite values, and triplet indices
    /// within `[0, n)`.
    pub fn set_quadratic_constraints(&mut self, qcs: Vec<QcqpMatrix>) -> Result<(), QpProblemError> {
        if !qcs.is_empty() && qcs.len() != self.num_constraints {
            return Err(QpProblemError::DimensionMismatch(format!(
                "quadratic_constraints length must be 0 or {}, got {}",
                self.num_constraints, qcs.len()
            )));
        }
        for (k, qc) in qcs.iter().enumerate() {
            if qc.n != self.num_vars {
                return Err(QpProblemError::DimensionMismatch(format!(
                    "quadratic_constraints[{}].n must be {}, got {}",
                    k, self.num_vars, qc.n
                )));
            }
            for &(row, col, v) in &qc.triplets {
                if !v.is_finite() {
                    return Err(QpProblemError::NonFiniteCoefficient {
                        field: "quadratic_constraints",
                        index: k,
                    });
                }
                if row >= qc.n || col >= qc.n {
                    return Err(QpProblemError::TripletIndexOutOfBounds {
                        constraint: k,
                        row,
                        col,
                        n: qc.n,
                    });
                }
            }
        }
        self.quadratic_constraints = qcs;
        Ok(())
    }

    /// Q が対角行列かどうかを検査する
    pub fn is_diagonal_q(&self) -> bool {
        for col in 0..self.num_vars {
            let start = self.q.col_ptr[col];
            let end = self.q.col_ptr[col + 1];
            for k in start..end {
                let row = self.q.row_ind[k];
                if row != col && self.q.values[k].abs() > 1e-12 {
                    return false;
                }
            }
        }
        true
    }

}

impl crate::problem::SolverResult {
    pub fn infeasible() -> Self {
        crate::problem::SolverResult {
            status: SolveStatus::Infeasible,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],

            iterations: 0,
            ..Default::default()
        }
    }

    pub fn unbounded() -> Self {
        crate::problem::SolverResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            iterations: 0,
            ..Default::default()
        }
    }

    pub fn max_iterations(x: Vec<f64>, obj: f64, iters: usize) -> Self {
        crate::problem::SolverResult {
            status: SolveStatus::MaxIterations,
            objective: obj,
            solution: x,
            dual_solution: vec![],
            bound_duals: vec![],
            iterations: iters,
            ..Default::default()
        }
    }

    pub fn numerical_error() -> Self {
        crate::problem::SolverResult {
            status: SolveStatus::NumericalError,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],

            iterations: 0,
            ..Default::default()
        }
    }

    pub fn not_supported(msg: impl Into<String>) -> Self {
        crate::problem::SolverResult {
            status: SolveStatus::NotSupported(msg.into()),
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            iterations: 0,
            ..Default::default()
        }
    }
}

pub use crate::options::QpWarmStart;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ConstraintType;
    use crate::sparse::CscMatrix;

    fn make_qp(
        c: Vec<f64>,
        b: Vec<f64>,
        q_vals: Vec<f64>,
        a_vals: Vec<f64>,
        bounds: Vec<(f64, f64)>,
    ) -> Result<QpProblem, QpProblemError> {
        let n = c.len();
        let m = b.len();
        let q = if q_vals.is_empty() {
            CscMatrix::new(n, n)
        } else {
            let idx: Vec<usize> = (0..n).collect();
            crate::sparse::CscMatrix::from_triplets(&idx, &idx, &q_vals, n, n).unwrap()
        };
        let a = if a_vals.is_empty() {
            CscMatrix::new(m, n)
        } else {
            let rows = vec![0usize; n];
            let cols: Vec<usize> = (0..n).collect();
            crate::sparse::CscMatrix::from_triplets(&rows, &cols, &a_vals, m, n).unwrap()
        };
        let ct = vec![ConstraintType::Le; m];
        QpProblem::new(q, c, a, b, bounds, ct)
    }

    #[test]
    fn valid_qp_accepted() {
        let res = make_qp(
            vec![1.0, 2.0],
            vec![5.0],
            vec![],
            vec![1.0, 1.0],
            vec![(0.0, f64::INFINITY), (0.0, 10.0)],
        );
        assert!(res.is_ok());
    }

    // --- NaN / ±inf in c ---
    #[test]
    fn nan_in_c_rejected() {
        let cases = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        for bad in cases {
            let res = make_qp(vec![bad, 1.0], vec![5.0], vec![], vec![1.0, 1.0],
                              vec![(0.0, f64::INFINITY); 2]);
            assert!(
                matches!(res, Err(QpProblemError::NonFiniteCoefficient { field: "c", .. })),
                "expected NonFiniteCoefficient for c={bad}"
            );
        }
    }

    // --- NaN / ±inf in b ---
    #[test]
    fn nan_in_b_rejected() {
        let cases = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        for bad in cases {
            let res = make_qp(vec![1.0, 2.0], vec![bad], vec![], vec![1.0, 1.0],
                              vec![(0.0, f64::INFINITY); 2]);
            assert!(
                matches!(res, Err(QpProblemError::NonFiniteCoefficient { field: "b", .. })),
                "expected NonFiniteCoefficient for b={bad}"
            );
        }
    }

    // --- NaN / ±inf in Q values ---
    // Note: from_triplets drops NaN via DROP_TOL (NaN.abs() > tol == false).
    // Inject bad values directly to test validation independent of that path.
    #[test]
    fn nan_in_q_rejected() {
        let n = 2;
        let bad_vals = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        for bad in bad_vals {
            let mut q = CscMatrix::from_triplets(&[0], &[0], &[1.0], n, n).unwrap();
            q.values[0] = bad; // inject bad value directly
            let a = CscMatrix::new(1, n);
            let c = vec![1.0, 2.0];
            let b = vec![5.0];
            let bounds = vec![(0.0, f64::INFINITY); n];
            let ct = vec![ConstraintType::Le];
            let res = QpProblem::new(q, c, a, b, bounds, ct);
            assert!(
                matches!(res, Err(QpProblemError::NonFiniteCoefficient { field: "Q", .. })),
                "expected NonFiniteCoefficient for Q val={bad}"
            );
        }
    }

    // --- NaN / ±inf in A values ---
    #[test]
    fn nan_in_a_rejected() {
        let n = 2;
        let bad_vals = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        for bad in bad_vals {
            let q = CscMatrix::new(n, n);
            let mut a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap();
            a.values[0] = bad; // inject bad value directly
            let c = vec![1.0, 2.0];
            let b = vec![5.0];
            let bounds = vec![(0.0, f64::INFINITY); n];
            let ct = vec![ConstraintType::Le];
            let res = QpProblem::new(q, c, a, b, bounds, ct);
            assert!(
                matches!(res, Err(QpProblemError::NonFiniteCoefficient { field: "A", .. })),
                "expected NonFiniteCoefficient for A val={bad}"
            );
        }
    }

    // --- NaN in bounds ---
    #[test]
    fn nan_in_bounds_rejected() {
        let cases: Vec<(f64, f64)> = vec![
            (f64::NAN, 1.0),
            (0.0, f64::NAN),
            (f64::NAN, f64::NAN),
        ];
        for (lb, ub) in cases {
            let res = make_qp(
                vec![1.0, 2.0], vec![5.0], vec![], vec![1.0, 1.0],
                vec![(lb, ub), (0.0, f64::INFINITY)],
            );
            assert!(
                matches!(res, Err(QpProblemError::InvalidBounds { index: 0, .. })),
                "expected InvalidBounds for ({lb},{ub})"
            );
        }
    }

    // --- lb > ub ---
    #[test]
    fn lb_gt_ub_rejected() {
        let cases: Vec<(f64, f64)> = vec![
            (5.0, 1.0),
            (1.0, 0.0),
            (f64::INFINITY, f64::NEG_INFINITY),
            (0.1, 0.0),
        ];
        for (lb, ub) in cases {
            let res = make_qp(
                vec![1.0, 2.0], vec![5.0], vec![], vec![1.0, 1.0],
                vec![(lb, ub), (0.0, f64::INFINITY)],
            );
            assert!(
                matches!(res, Err(QpProblemError::InvalidBounds { .. })),
                "expected InvalidBounds for lb={lb} ub={ub}"
            );
        }
    }

    // --- inf bounds are valid ---
    #[test]
    fn inf_bounds_accepted() {
        let res = make_qp(
            vec![1.0, 2.0], vec![5.0], vec![], vec![1.0, 1.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, f64::INFINITY)],
        );
        assert!(res.is_ok(), "±inf bounds should be valid");
    }

    // --- dimension mismatch still caught ---
    #[test]
    fn dimension_mismatch_still_caught() {
        let n = 2;
        let q = CscMatrix::new(n, n);
        let a = CscMatrix::new(1, n);
        let c = vec![1.0, 2.0, 3.0]; // wrong length
        let b = vec![5.0];
        let bounds = vec![(0.0, f64::INFINITY); n];
        let ct = vec![ConstraintType::Le];
        let res = QpProblem::new(q, c, a, b, bounds, ct);
        assert!(matches!(res, Err(QpProblemError::DimensionMismatch(_))));
    }

    // --- set_quadratic_constraints ---

    #[test]
    fn set_quadratic_constraints_length_mismatch_rejected() {
        let mut prob = make_qp(
            vec![1.0, 2.0], vec![5.0], vec![], vec![1.0, 1.0],
            vec![(0.0, f64::INFINITY); 2],
        ).unwrap();
        // num_constraints = 1, so length 2 is invalid
        let qcs = vec![QcqpMatrix::new(2), QcqpMatrix::new(2)];
        let res = prob.set_quadratic_constraints(qcs);
        assert!(matches!(res, Err(QpProblemError::DimensionMismatch(_))));
    }

    #[test]
    fn set_quadratic_constraints_nan_rejected() {
        let mut prob = make_qp(
            vec![1.0, 2.0], vec![5.0], vec![], vec![1.0, 1.0],
            vec![(0.0, f64::INFINITY); 2],
        ).unwrap();
        let bad_vals = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        for bad in bad_vals {
            let mut qc = QcqpMatrix::new(2);
            qc.triplets.push((0, 0, bad));
            let res = prob.set_quadratic_constraints(vec![qc]);
            assert!(
                matches!(res, Err(QpProblemError::NonFiniteCoefficient { field: "quadratic_constraints", .. })),
                "expected NonFiniteCoefficient for val={bad}"
            );
        }
    }

    #[test]
    fn set_quadratic_constraints_empty_accepted() {
        let mut prob = make_qp(
            vec![1.0, 2.0], vec![5.0], vec![], vec![1.0, 1.0],
            vec![(0.0, f64::INFINITY); 2],
        ).unwrap();
        assert!(prob.set_quadratic_constraints(vec![]).is_ok());
    }

    #[test]
    fn set_quadratic_constraints_valid_accepted() {
        let mut prob = make_qp(
            vec![1.0, 2.0], vec![5.0], vec![], vec![1.0, 1.0],
            vec![(0.0, f64::INFINITY); 2],
        ).unwrap();
        let mut qc = QcqpMatrix::new(2);
        qc.triplets.push((0, 0, 2.0));
        qc.triplets.push((1, 1, 3.0));
        assert!(prob.set_quadratic_constraints(vec![qc]).is_ok());
    }

    // --- sentinel: n mismatch rejected (no-op rewrite → FAIL) ---
    #[test]
    fn set_quadratic_constraints_n_mismatch_rejected() {
        let mut prob = make_qp(
            vec![1.0, 2.0], vec![5.0], vec![], vec![1.0, 1.0],
            vec![(0.0, f64::INFINITY); 2],
        ).unwrap();
        // num_vars=2, but QcqpMatrix.n=3 — must be rejected
        let mut qc = QcqpMatrix::new(3);
        qc.triplets.push((0, 0, 1.0));
        let res = prob.set_quadratic_constraints(vec![qc]);
        assert!(
            matches!(res, Err(QpProblemError::DimensionMismatch(_))),
            "n mismatch must be rejected"
        );
    }

    // --- sentinel: OOB triplet row index rejected (no-op rewrite → FAIL) ---
    #[test]
    fn set_quadratic_constraints_triplet_row_oob_rejected() {
        let mut prob = make_qp(
            vec![1.0, 2.0], vec![5.0], vec![], vec![1.0, 1.0],
            vec![(0.0, f64::INFINITY); 2],
        ).unwrap();
        let mut qc = QcqpMatrix::new(2);
        qc.triplets.push((2, 0, 1.0)); // row=2 >= n=2
        let res = prob.set_quadratic_constraints(vec![qc]);
        assert!(
            matches!(res, Err(QpProblemError::TripletIndexOutOfBounds { row: 2, col: 0, n: 2, .. })),
            "row OOB must be rejected"
        );
    }

    // --- sentinel: OOB triplet col index rejected (no-op rewrite → FAIL) ---
    #[test]
    fn set_quadratic_constraints_triplet_col_oob_rejected() {
        let mut prob = make_qp(
            vec![1.0, 2.0], vec![5.0], vec![], vec![1.0, 1.0],
            vec![(0.0, f64::INFINITY); 2],
        ).unwrap();
        let mut qc = QcqpMatrix::new(2);
        qc.triplets.push((0, 5, 1.0)); // col=5 >= n=2
        let res = prob.set_quadratic_constraints(vec![qc]);
        assert!(
            matches!(res, Err(QpProblemError::TripletIndexOutOfBounds { col: 5, n: 2, .. })),
            "col OOB must be rejected"
        );
    }

    // --- sentinel: CscMatrix accessor returns correct data (no-op rewrite → FAIL) ---
    #[test]
    fn csc_accessor_returns_correct_data() {
        let rows = vec![0usize, 1];
        let cols = vec![0usize, 1];
        let vals = vec![3.0f64, 7.0];
        let m = CscMatrix::from_triplets(&rows, &cols, &vals, 2, 2).unwrap();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        assert_eq!(m.nnz(), 2);
        assert_eq!(m.col_ptr().len(), 3);
        assert_eq!(m.row_ind().len(), 2);
        assert_eq!(m.values().len(), 2);
        let (ri, vs) = m.get_column(0).unwrap();
        assert_eq!(ri, &[0]);
        assert_eq!(vs, &[3.0]);
        let (ri, vs) = m.get_column(1).unwrap();
        assert_eq!(ri, &[1]);
        assert_eq!(vs, &[7.0]);
    }
}
