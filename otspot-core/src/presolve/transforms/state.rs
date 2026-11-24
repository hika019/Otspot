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
        /// A[orig_row, orig_col] coefficient.
        coeff: f64,
        /// (row_k, A[row_k, orig_col]) for all rows k != orig_row that were active
        /// at elimination time. Used to recover y[orig_row] via stationarity.
        col_orig_entries: Vec<(usize, f64)>,
        /// Objective coefficient c[orig_col] at elimination time.
        c_orig: f64,
    },
    /// Le/Ge singleton row absorbed into a variable bound.
    #[allow(dead_code)]
    SingletonInequalityRow {
        orig_row: usize,
        orig_col: usize,
        coeff: f64,
        old_lb: f64,
        old_ub: f64,
        /// (row_k, A[row_k, orig_col]) for all rows k != orig_row that were active
        /// at elimination time. Used for stationarity-based dual recovery.
        col_orig_entries: Vec<(usize, f64)>,
        /// Objective coefficient c[orig_col] at elimination time.
        c_orig: f64,
    },
    RedundantConstraint {
        orig_row: usize,
    },
    /// Eq row removed by rank discovery (Step 12): it lies in the span of the
    /// other surviving Eq rows with a consistent RHS, so the feasible set is
    /// unchanged. Postsolve sets its dual to 0: the reduced dual stays
    /// dual-feasible for the original (stationarity rows are identical with
    /// y=0 added), the removed row binds always and Eq duals are sign-free
    /// (complementarity holds), and the dual objective is unchanged.
    DependentEquation {
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
    /// Forcing row: all variables forced to their contributing bounds.
    ForcingRow {
        orig_row: usize,
        fixed_vars: Vec<(usize, f64, f64, f64)>,
        /// (col, A[orig_row, col]) for all columns active at elimination time.
        /// Snapshot used in postsolve for stationarity-based dual recovery,
        /// bypassing the activity-based binding check that would read
        /// partially-restored LIFO solutions.
        row_orig_entries: Vec<(usize, f64)>,
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
    /// `true` when the fixpoint loop exhausted `presolve_max_pass` passes
    /// without reaching a stable fixpoint (i.e. the last pass still changed
    /// something). The emitted `reduced_problem` is valid either way — this
    /// only flags that further passes might have reduced it more. A
    /// `log::warn!` is emitted alongside this at the point of detection.
    pub pass_limit_hit: bool,
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
    /// Step 12: rank-based removal of linearly dependent Eq rows.
    pub enable_dependent_eq: bool,
}

impl Default for PresolveFlags {
    fn default() -> Self {
        Self {
            enable_parallel_row: true,
            enable_dup_dom_col: true,
            enable_dual_fixing: true,
            enable_dependent_eq: true,
        }
    }
}

impl PresolveFlags {
    pub fn all_off() -> Self {
        Self {
            enable_parallel_row: false,
            enable_dup_dom_col: false,
            enable_dual_fixing: false,
            enable_dependent_eq: false,
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
            obj_offset: problem.obj_offset,
            pass_limit_hit: false,
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
            let (rows, vals) = problem.a.column(j);
            for (k, &row) in rows.iter().enumerate() {
                let v = vals[k];
                if v.abs() < ZERO_TOL {
                    continue;
                }
                row_entries[row].push((j, v));
                col_entries[j].push((row, v));
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
        update_sparse_entry(&mut self.row_entries[i], j, delta);
        update_sparse_entry(&mut self.col_entries[j], i, delta);
    }
}

fn update_sparse_entry(entries: &mut Vec<(usize, f64)>, index: usize, delta: f64) {
    if let Some(pos) = entries.iter().position(|&(idx, _)| idx == index) {
        entries[pos].1 += delta;
        if entries[pos].1.abs() < ZERO_TOL {
            entries.remove(pos);
        }
    } else {
        entries.push((index, delta));
    }
}

// Scratch staging writes performed by the reduced-matrix rebuild so far on
// this thread (test builds only).
//
// `driver::build_reduced_result` is contractually a *direct* builder: it
// writes each emitted output entry exactly once, from the column side of
// the dual representation into the final CSC arrays. A COO/triplet staging
// detour would write every entry at least twice (driver staging plus
// `from_triplets`' internal staging), materializing an extra full copy of
// the matrix inside the window where the original problem and the dual
// representation are still resident. The sentinel test in this module
// asserts exactly one write per emitted entry, so a reintroduced staging
// array — or a builder that skipped staging and wrote nothing — fails.
#[cfg(test)]
thread_local! {
    pub(crate) static REBUILD_SCRATCH_WRITES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// Record `entries` scratch writes for the current rebuild (test builds
/// only; see [`REBUILD_SCRATCH_WRITES`]; release builds compile the call
/// site away).
#[cfg(test)]
pub(crate) fn count_rebuild_scratch_entry(entries: usize) {
    REBUILD_SCRATCH_WRITES.with(|c| c.set(c.get() + entries));
}

/// A rebuild is only meaningful (and only sentinel-testable) when it emits
/// matrix entries at all: the sentinel's write budget is `writes per entry`,
/// which needs at least one emitted entry to distinguish a direct builder
/// from a staging detour or a non-writing stub.
#[cfg(test)]
const SENTINEL_MIN_REBUILT_NNZ: usize = 1;

/// The write budget the sentinel enforces: exactly one scratch write per
/// emitted entry (direct build). `> 1` means a staging array reappeared.
#[cfg(test)]
const SENTINEL_SCRATCH_WRITES_PER_ENTRY: usize = 1;

#[cfg(test)]
mod rebuild_residency_sentinel {
    use super::{
        PresolveState, REBUILD_SCRATCH_WRITES, SENTINEL_MIN_REBUILT_NNZ,
        SENTINEL_SCRATCH_WRITES_PER_ENTRY,
    };
    use crate::problem::LpProblem;
    use otspot_num::sparse::CscMatrix;

    /// Sentinel (no-op-fail): `build_reduced_result` must be a direct
    /// builder — exactly [`SENTINEL_SCRATCH_WRITES_PER_ENTRY`] scratch write
    /// per emitted entry. A revert to the COO-triplet rebuild (driver
    /// staging + `from_triplets` staging, 2–3 writes per entry) or any
    /// builder that stops counting (0 writes) fails this equality without
    /// depending on allocator behavior or wall-clock timing.
    #[test]
    fn reduced_rebuild_writes_each_output_entry_exactly_once() {
        // Direct on the builder rather than through `run_presolve`: the
        // fixpoint's reductions are irrelevant to the write budget, and
        // small fixtures get folded away entirely by presolve heuristics.
        let a = CscMatrix::from_triplets(&[0, 1, 0, 1], &[0, 0, 1, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2)
            .unwrap();
        let problem = LpProblem::new(vec![1.0, 1.0], a, vec![1.0, 1.0]).unwrap();
        let st = PresolveState::from_problem(&problem);

        REBUILD_SCRATCH_WRITES.with(|c| c.set(0));
        let out = super::super::driver::build_reduced_result_for_test(&problem, st, 2, 2, false)
            .expect("feasible");
        let writes = REBUILD_SCRATCH_WRITES.with(|c| c.get());
        let emitted = out.reduced_problem.a.nnz();
        assert!(
            emitted >= SENTINEL_MIN_REBUILT_NNZ,
            "fixture must emit a non-empty matrix for the write budget to be meaningful"
        );
        assert_eq!(
            writes,
            emitted * SENTINEL_SCRATCH_WRITES_PER_ENTRY,
            "direct rebuild must write each emitted entry once; got {writes} writes \
             for {emitted} entries (a triplet staging detour doubles/triples this)"
        );
    }

    /// Independent-oracle pin for the direct builder's merge contract, which
    /// `CscMatrix::from_triplets` used to provide: per-column row indices in
    /// ascending order, duplicate `(row, col)` sums, `DROP_TOL` pruning of
    /// merged cancellations, and `ZERO_TOL` filtering of structural zeros —
    /// with the expected arrays hand-computed.
    #[test]
    fn direct_rebuild_merges_prunes_and_sorts_like_independent_oracle() {
        let a = CscMatrix::from_triplets(&[0, 1, 0, 1], &[0, 0, 1, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2)
            .unwrap();
        let problem = LpProblem::new(vec![1.0, 1.0], a.clone(), vec![1.0, 1.0]).unwrap();
        let mut st = PresolveState::from_problem(&problem);
        // Hand-built column contents: column 0 carries an unsorted order,
        // a duplicate pair (rows 0) that must merge to 5.0, and a duplicate
        // pair (rows 1) that cancels to 0.0 and must be pruned at DROP_TOL;
        // column 1 carries a sub-ZERO_TOL entry and an entry in a removed
        // row, both of which must vanish.
        st.col_entries[0] = vec![(1, 1.0), (1, -1.0), (0, 2.0), (0, 3.0)];
        st.col_entries[1] = vec![(1, 9.0), (0, 1e-13)];
        st.removed_rows[1] = true;

        let out = super::super::driver::build_reduced_result_for_test(&problem, st, 2, 2, false)
            .expect("feasible");
        let reduced = &out.reduced_problem.a;
        assert_eq!(reduced.nrows(), 1);
        assert_eq!(reduced.ncols(), 2);
        assert_eq!(reduced.col_ptr(), &[0, 1, 1], "column 1 fully pruned");
        assert_eq!(
            reduced.row_ind(),
            &[0],
            "merged column keeps one live entry"
        );
        assert_eq!(
            reduced.values(),
            &[5.0],
            "duplicate rows sum in place: 2.0 + 3.0"
        );
        assert!(out.was_reduced);
    }
}
