//! Shared presolve types: postsolve step variants, public result/status,
//! per-transform flag toggles, and the mutable working state passed across steps.

use crate::problem::{ConstraintType, LpProblem};
use crate::tolerances::ZERO_TOL;

/// Inverse of one presolve operation. Pushed in forward order; postsolve
/// replays it in LIFO order.
#[derive(Debug, Clone)]
pub(crate) enum PostsolveStep {
    FixedVariable {
        orig_col: usize,
        value: f64,
    },
    EmptyColumn {
        orig_col: usize,
        value: f64,
    },
    EmptyRow {
        orig_row: usize,
    },
    /// Eq row reduced to a single variable.
    SingletonRow {
        orig_row: usize,
        orig_col: usize,
        value: f64,
    },
    RedundantConstraint {
        orig_row: usize,
    },
    BoundsTightened,
    /// Variable eliminated via a pivot Eq row. Shared by R6 (doubleton), R15 (free-var),
    /// and R5 (free-singleton-col). Postsolve restores
    ///   `orig_col = (rhs - Σ coeff_k * x_orig_other_k) / pivot`
    /// and recovers the row's dual via
    ///   `y_piv = (c_j_orig - Σ_{i ≠ piv_row} A_ij_orig * y_i) / pivot`.
    LinearSubstitution {
        orig_col: usize,
        orig_row: Option<usize>,
        pivot: f64,
        rhs: f64,
        others: Vec<(usize, f64)>,
        col_orig_entries: Vec<(usize, f64)>,
        c_orig: f64,
    },
}

/// Public presolve output: reduced LP plus the metadata postsolve needs.
pub struct PresolveResult {
    pub reduced_problem: LpProblem,
    pub(crate) postsolve_stack: Vec<PostsolveStep>,
    pub orig_num_vars: usize,
    pub orig_num_constraints: usize,
    pub col_map: Vec<Option<usize>>,
    pub row_map: Vec<Option<usize>>,
    pub was_reduced: bool,
    pub obj_offset: f64,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum PresolveStatus {
    Infeasible,
    Unbounded,
}

/// Per-transform on/off toggles. Default: all on. Sentinel tests flip individual
/// flags off to assert that disabling each path leaves a measurable artifact
/// (reduction count or runtime), proving the transform is not a no-op.
#[derive(Debug, Clone, Copy)]
pub struct PresolveFlags {
    pub enable_parallel_row: bool,
    pub enable_dup_dom_col: bool,
    pub enable_dual_fixing: bool,
}

impl Default for PresolveFlags {
    fn default() -> Self {
        Self {
            enable_parallel_row: true,
            enable_dup_dom_col: true,
            enable_dual_fixing: true,
        }
    }
}

impl PresolveFlags {
    pub fn all_off() -> Self {
        Self {
            enable_parallel_row: false,
            enable_dup_dom_col: false,
            enable_dual_fixing: false,
        }
    }
}

impl PresolveResult {
    pub fn no_reduction(problem: &LpProblem) -> Self {
        let n = problem.num_vars;
        let m = problem.num_constraints;
        PresolveResult {
            reduced_problem: problem.clone(),
            postsolve_stack: vec![],
            orig_num_vars: n,
            orig_num_constraints: m,
            col_map: (0..n).map(Some).collect(),
            row_map: (0..m).map(Some).collect(),
            was_reduced: false,
            obj_offset: 0.0,
        }
    }
}

/// Mutable working state shared by all per-step transforms.
///
/// `row_entries[i]` and `col_entries[j]` are dual representations of the same
/// sparse matrix and must be updated in lockstep.
pub(crate) struct PresolveState {
    pub(crate) row_entries: Vec<Vec<(usize, f64)>>,
    pub(crate) col_entries: Vec<Vec<(usize, f64)>>,
    pub(crate) b: Vec<f64>,
    pub(crate) bounds: Vec<(f64, f64)>,
    pub(crate) orig_bounds: Vec<(f64, f64)>,
    pub(crate) constraint_types: Vec<ConstraintType>,
    pub(crate) c: Vec<f64>,
    pub(crate) removed_cols: Vec<bool>,
    pub(crate) removed_rows: Vec<bool>,
    pub(crate) postsolve_stack: Vec<PostsolveStep>,
    pub(crate) obj_offset: f64,
}

impl PresolveState {
    pub(crate) fn from_problem(problem: &LpProblem) -> Self {
        let n = problem.num_vars;
        let m = problem.num_constraints;

        let mut row_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; m];
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        for j in 0..n {
            if let Ok((rows, vals)) = problem.a.get_column(j) {
                for (k, &row) in rows.iter().enumerate() {
                    let v = vals[k];
                    if v.abs() < ZERO_TOL {
                        continue;
                    }
                    row_entries[row].push((j, v));
                    col_entries[j].push((row, v));
                }
            }
        }

        PresolveState {
            row_entries,
            col_entries,
            b: problem.b.clone(),
            bounds: problem.bounds.clone(),
            orig_bounds: problem.bounds.clone(),
            constraint_types: problem.constraint_types.clone(),
            c: problem.c.clone(),
            removed_cols: vec![false; n],
            removed_rows: vec![false; m],
            postsolve_stack: Vec::new(),
            obj_offset: 0.0,
        }
    }

    pub(crate) fn active_row_entries(&self, i: usize) -> Vec<(usize, f64)> {
        self.row_entries[i]
            .iter()
            .filter(|&&(j, v)| !self.removed_cols[j] && v.abs() >= ZERO_TOL)
            .copied()
            .collect()
    }

    pub(crate) fn active_col_entries(&self, j: usize) -> Vec<(usize, f64)> {
        self.col_entries[j]
            .iter()
            .filter(|&&(i, v)| !self.removed_rows[i] && v.abs() >= ZERO_TOL)
            .copied()
            .collect()
    }

    /// Lookup of `A[i,j]` summed over duplicate entries (zero if absent or removed).
    pub(crate) fn coeff(&self, i: usize, j: usize) -> f64 {
        let mut s = 0.0;
        for &(jj, v) in &self.row_entries[i] {
            if jj == j && !self.removed_cols[jj] {
                s += v;
            }
        }
        s
    }

    /// `A[i,j] += delta`, merging duplicate entries and pruning if the result is ~0.
    pub(crate) fn add_to_entry(&mut self, i: usize, j: usize, delta: f64) {
        if delta.abs() < ZERO_TOL {
            return;
        }
        let mut found_row = false;
        for entry in self.row_entries[i].iter_mut() {
            if entry.0 == j {
                entry.1 += delta;
                found_row = true;
                break;
            }
        }
        if !found_row {
            self.row_entries[i].push((j, delta));
        }
        let mut found_col = false;
        for entry in self.col_entries[j].iter_mut() {
            if entry.0 == i {
                entry.1 += delta;
                found_col = true;
                break;
            }
        }
        if !found_col {
            self.col_entries[j].push((i, delta));
        }
        self.row_entries[i].retain(|&(jj, v)| jj != j || v.abs() >= ZERO_TOL);
        self.col_entries[j].retain(|&(ii, v)| ii != i || v.abs() >= ZERO_TOL);
    }
}
