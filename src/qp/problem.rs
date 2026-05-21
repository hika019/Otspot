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
}

impl std::fmt::Display for QpProblemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QpProblemError::DimensionMismatch(msg) => write!(f, "dimension mismatch: {}", msg),
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
        Ok(QpProblem { q, c, a, b, bounds, num_vars: n, num_constraints: m, constraint_types, quadratic_constraints: vec![], obj_offset: 0.0 })
    }

    /// 全制約 Le として構築するヘルパー。
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
}

pub use crate::options::QpWarmStart;
