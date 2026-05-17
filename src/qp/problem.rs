//! QP問題のデータ構造定義。

use crate::problem::{ConstraintType, SolveStatus};
use crate::sparse::CscMatrix;

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
        Ok(QpProblem { q, c, a, b, bounds, num_vars: n, num_constraints: m, constraint_types, obj_offset: 0.0 })
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

    #[deprecated(note = "IPMは等式ネイティブ化。build_extended_constraintsを使用")]
    #[allow(deprecated)]
    pub fn to_all_le(&self) -> (QpProblem, LeExpansionMap) {
        if self.constraint_types.iter().all(|ct| matches!(ct, ConstraintType::Le)) {
            return (self.clone(), LeExpansionMap::identity(self.num_constraints));
        }

        let n = self.num_vars;
        let m = self.num_constraints;

        let mut row_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
        for col in 0..n {
            for k in self.a.col_ptr[col]..self.a.col_ptr[col + 1] {
                let row = self.a.row_ind[k];
                row_entries[row].push((col, self.a.values[k]));
            }
        }

        let mut trip_rows: Vec<usize> = Vec::new();
        let mut trip_cols: Vec<usize> = Vec::new();
        let mut trip_vals: Vec<f64> = Vec::new();
        let mut new_b: Vec<f64> = Vec::new();
        let mut new_ct: Vec<ConstraintType> = Vec::new();
        let mut original_to_expanded: Vec<Vec<usize>> = Vec::with_capacity(m);
        let mut new_row = 0usize;

        for (i, ct) in self.constraint_types.iter().enumerate() {
            match ct {
                ConstraintType::Le => {
                    for &(col, val) in &row_entries[i] {
                        trip_rows.push(new_row);
                        trip_cols.push(col);
                        trip_vals.push(val);
                    }
                    new_b.push(self.b[i]);
                    new_ct.push(ConstraintType::Le);
                    original_to_expanded.push(vec![new_row]);
                    new_row += 1;
                }
                ConstraintType::Eq => {
                    for &(col, val) in &row_entries[i] {
                        trip_rows.push(new_row);
                        trip_cols.push(col);
                        trip_vals.push(val);
                    }
                    new_b.push(self.b[i]);
                    new_ct.push(ConstraintType::Le);
                    let row1 = new_row;
                    new_row += 1;
                    for &(col, val) in &row_entries[i] {
                        trip_rows.push(new_row);
                        trip_cols.push(col);
                        trip_vals.push(-val);
                    }
                    new_b.push(-self.b[i]);
                    new_ct.push(ConstraintType::Le);
                    let row2 = new_row;
                    new_row += 1;
                    original_to_expanded.push(vec![row1, row2]);
                }
                ConstraintType::Ge => {
                    for &(col, val) in &row_entries[i] {
                        trip_rows.push(new_row);
                        trip_cols.push(col);
                        trip_vals.push(-val);
                    }
                    new_b.push(-self.b[i]);
                    new_ct.push(ConstraintType::Le);
                    original_to_expanded.push(vec![new_row]);
                    new_row += 1;
                }
            }
        }

        let new_a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, new_row, n)
            .expect("to_all_le: CscMatrix::from_triplets failed");
        let mut new_problem = QpProblem::new(
            self.q.clone(),
            self.c.clone(),
            new_a,
            new_b,
            self.bounds.clone(),
            new_ct,
        ).expect("to_all_le: QpProblem::new failed");
        new_problem.obj_offset = self.obj_offset;

        (new_problem, LeExpansionMap { original_to_expanded })
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

    /// 対角要素のベクトルを返す（対角行列のみ有効）
    #[allow(dead_code)]
    pub fn diagonal_q(&self) -> Vec<f64> {
        let mut diag = vec![0.0; self.num_vars];
        for (col, diag_val) in diag.iter_mut().enumerate().take(self.num_vars) {
            let start = self.q.col_ptr[col];
            let end = self.q.col_ptr[col + 1];
            for k in start..end {
                if self.q.row_ind[k] == col {
                    *diag_val = self.q.values[k];
                }
            }
        }
        diag
    }
}

#[deprecated(note = "to_all_le()廃止に伴い不要")]
pub struct LeExpansionMap {
    pub original_to_expanded: Vec<Vec<usize>>,
}

#[allow(deprecated)]
impl LeExpansionMap {
    pub fn identity(m: usize) -> Self {
        Self {
            original_to_expanded: (0..m).map(|i| vec![i]).collect(),
        }
    }
}

#[deprecated(since = "0.1.0", note = "use SolverResult (LP/QP unified result type) instead")]
#[allow(unused)]
pub type QpResult = crate::problem::SolverResult;

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

/// Warm-start. 公開 API 互換性のため保持しているが現状 solve_qp_warm では未使用。
#[derive(Debug, Clone)]
pub struct QpWarmStart {
    pub initial_active_set: Vec<usize>,
    pub initial_point: Option<Vec<f64>>,
}
