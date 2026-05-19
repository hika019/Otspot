//! QP presolve types: status enum, postsolve metadata, and the public result struct.

use crate::linalg::ruiz::RuizScaler;
use crate::qp::QpProblem;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QpPresolveStatus {
    Feasible,
    Infeasible,
    Unbounded,
}

#[derive(Debug, Clone)]
pub(crate) enum QpPostsolveStep {
    FixedVar { idx: usize, val: f64 },
    /// Singleton Eq row `A[i,j]·x[j] = b[i]`; `row` lets postsolve recover `y[row]`.
    SingletonRow { row: usize, col: usize, val: f64 },
    EmptyCol { idx: usize, val: f64 },
    /// Per-row scaling factor used to unscale the dual after large-coefficient rescaling.
    LargeCoeffRowScale { row_scales: Vec<f64> },
}

pub(crate) struct QpPostsolveStack {
    pub(crate) steps: Vec<QpPostsolveStep>,
}

impl QpPostsolveStack {
    pub(crate) fn new() -> Self {
        Self { steps: Vec::new() }
    }
    pub(crate) fn push(&mut self, step: QpPostsolveStep) {
        self.steps.push(step);
    }
}

pub struct QpPresolveResult {
    pub reduced: QpProblem,
    /// orig col index → reduced col index (None = removed).
    pub col_map: Vec<Option<usize>>,
    pub col_map_inv: Vec<usize>,
    /// orig row index → reduced row index (None = removed).
    pub row_map: Vec<Option<usize>>,
    pub obj_offset: f64,
    pub q_linear_adjust: Vec<f64>,
    pub(crate) postsolve_stack: QpPostsolveStack,
    pub was_reduced: bool,
    pub orig_num_vars: usize,
    pub orig_num_constraints: usize,
    pub presolve_status: QpPresolveStatus,
    pub is_diagonal_q: bool,
    /// Number of independent variable blocks (1 = not separable).
    pub block_components: usize,
    pub ruiz_scaler: Option<RuizScaler>,
}

impl QpPresolveResult {
    pub fn no_reduction(prob: &QpProblem) -> Self {
        let n = prob.num_vars;
        let m = prob.num_constraints;
        QpPresolveResult {
            reduced: prob.clone(),
            col_map: (0..n).map(Some).collect(),
            col_map_inv: (0..n).collect(),
            row_map: (0..m).map(Some).collect(),
            obj_offset: prob.obj_offset,
            q_linear_adjust: vec![0.0; n],
            postsolve_stack: QpPostsolveStack::new(),
            was_reduced: false,
            orig_num_vars: n,
            orig_num_constraints: m,
            presolve_status: QpPresolveStatus::Feasible,
            is_diagonal_q: false,
            block_components: 1,
            ruiz_scaler: None,
        }
    }

    pub fn infeasible(prob: &QpProblem) -> Self {
        let mut r = Self::no_reduction(prob);
        r.presolve_status = QpPresolveStatus::Infeasible;
        r
    }

    pub fn unbounded(prob: &QpProblem) -> Self {
        let mut r = Self::no_reduction(prob);
        r.presolve_status = QpPresolveStatus::Unbounded;
        r
    }
}

/// Mutable working state threaded through every step. Owned values (not
/// references) so steps can mutate without lifetime gymnastics.
pub(super) struct Workspace {
    pub(super) c: Vec<f64>,
    pub(super) b: Vec<f64>,
    pub(super) c_comp: Vec<f64>,
    pub(super) b_comp: Vec<f64>,
    pub(super) bounds: Vec<(f64, f64)>,
    pub(super) removed_cols: Vec<bool>,
    pub(super) removed_rows: Vec<bool>,
    pub(super) obj_offset: f64,
    pub(super) obj_offset_comp: f64,
    pub(super) postsolve_stack: QpPostsolveStack,
    pub(super) row_entries: Vec<Vec<(usize, f64)>>,
}

impl Workspace {
    pub(super) fn from_problem(prob: &QpProblem) -> Self {
        let n = prob.num_vars;
        let m = prob.num_constraints;

        let mut row_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; m];
        for j in 0..n {
            let start = prob.a.col_ptr[j];
            let end = prob.a.col_ptr[j + 1];
            for idx in start..end {
                let row = prob.a.row_ind[idx];
                row_entries[row].push((j, prob.a.values[idx]));
            }
        }

        Self {
            c: prob.c.clone(),
            b: prob.b.clone(),
            c_comp: vec![0.0; n],
            b_comp: vec![0.0; m],
            bounds: prob.bounds.clone(),
            removed_cols: vec![false; n],
            removed_rows: vec![false; m],
            obj_offset: prob.obj_offset,
            obj_offset_comp: 0.0,
            postsolve_stack: QpPostsolveStack::new(),
            row_entries,
        }
    }
}
