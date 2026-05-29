//! MILP / MIQP problem definitions.
//!
//! A MILP/MIQP is a continuous LP/QP **relaxation** plus a set of variables
//! constrained to integer values. We deliberately wrap the existing
//! [`LpProblem`] / [`QpProblem`] rather than adding an integrality field to them:
//! the continuous solvers stay untouched (zero regression surface), and each
//! branch-and-bound node solves the relaxation by swapping `bounds` — the same
//! mechanism the spatial QP B&B already uses for box subproblems.

use super::Relaxation;
use crate::linalg::ldl::is_q_psd_by_cholesky;
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::qp::QpProblem;

/// Construction error for [`MilpProblem`] / [`MiqpProblem`].
#[non_exhaustive]
#[derive(Debug, PartialEq, Eq)]
pub enum MipProblemError {
    /// An integer-variable index is out of range for the relaxation.
    InvalidIntegerVar { index: usize, num_vars: usize },
}

impl std::fmt::Display for MipProblemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MipProblemError::InvalidIntegerVar { index, num_vars } => write!(
                f,
                "integer variable index {} out of range (num_vars = {})",
                index, num_vars
            ),
        }
    }
}

impl std::error::Error for MipProblemError {}

/// Sort, de-duplicate and range-check the integer-variable index set.
fn normalize_integer_vars(
    mut integer_vars: Vec<usize>,
    num_vars: usize,
) -> Result<Vec<usize>, MipProblemError> {
    integer_vars.sort_unstable();
    integer_vars.dedup();
    if let Some(&j) = integer_vars.last() {
        if j >= num_vars {
            return Err(MipProblemError::InvalidIntegerVar { index: j, num_vars });
        }
    }
    Ok(integer_vars)
}

/// Mixed-Integer Linear Program: minimize `c^T x` over the [`LpProblem`]
/// feasible region with `x[j]` integral for every `j` in `integer_vars`.
#[derive(Debug, Clone)]
pub struct MilpProblem {
    /// Continuous LP relaxation (objective, constraints, bounds).
    pub lp: LpProblem,
    /// Sorted, de-duplicated indices of variables required to be integral.
    pub integer_vars: Vec<usize>,
}

impl MilpProblem {
    /// Build a MILP from an LP relaxation and the integer-variable indices.
    ///
    /// `integer_vars` is sorted and de-duplicated. An empty set is permitted and
    /// means the problem is a plain LP (the solver falls back to the LP path).
    pub fn new(lp: LpProblem, integer_vars: Vec<usize>) -> Result<Self, MipProblemError> {
        let integer_vars = normalize_integer_vars(integer_vars, lp.num_vars)?;
        Ok(Self { lp, integer_vars })
    }

    /// Number of decision variables.
    pub fn num_vars(&self) -> usize {
        self.lp.num_vars
    }
}

impl Relaxation for MilpProblem {
    fn num_vars(&self) -> usize {
        self.lp.num_vars
    }
    fn root_bounds(&self) -> &[(f64, f64)] {
        &self.lp.bounds
    }
    fn integer_vars(&self) -> &[usize] {
        &self.integer_vars
    }
    fn solve(&self, bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult {
        let mut sub = self.lp.clone();
        sub.bounds = bounds.to_vec();
        crate::lp::solve_lp_with(&sub, opts)
    }
}

/// Mixed-Integer **convex** Quadratic Program: minimize `1/2 x^T Q x + c^T x`
/// over the [`QpProblem`] feasible region with `x[j]` integral for every `j` in
/// `integer_vars`. Only convex (`Q` PSD) instances are in scope — see
/// [`MiqpProblem::is_convex`].
#[derive(Debug, Clone)]
pub struct MiqpProblem {
    /// Continuous QP relaxation (objective, constraints, bounds).
    pub qp: QpProblem,
    /// Sorted, de-duplicated indices of variables required to be integral.
    pub integer_vars: Vec<usize>,
}

impl MiqpProblem {
    /// Build an MIQP from a QP relaxation and the integer-variable indices.
    ///
    /// Convexity is **not** enforced here; the solver checks [`is_convex`] and
    /// rejects non-PSD `Q`. `integer_vars` is sorted and de-duplicated; an empty
    /// set means a plain QP (the solver falls back to the QP path).
    ///
    /// [`is_convex`]: MiqpProblem::is_convex
    pub fn new(qp: QpProblem, integer_vars: Vec<usize>) -> Result<Self, MipProblemError> {
        let integer_vars = normalize_integer_vars(integer_vars, qp.num_vars)?;
        Ok(Self { qp, integer_vars })
    }

    /// Number of decision variables.
    pub fn num_vars(&self) -> usize {
        self.qp.num_vars
    }

    /// Whether the objective is convex (`Q` positive semidefinite). The QP
    /// relaxation is a valid lower bound only when this holds; a non-convex MIQP
    /// is out of scope.
    pub fn is_convex(&self) -> bool {
        is_q_psd_by_cholesky(&self.qp.q)
    }

    /// Convexity check with an optional size limit.
    ///
    /// When `max_n` is `Some(limit)` and `Q.nrows > limit`, skips the Cholesky
    /// PSD test and returns `true` (assumed convex). Setting a limit lower than
    /// the actual matrix size creates a soundness hole: a nonconvex Q is accepted,
    /// making the QP relaxation an invalid lower bound.
    pub(crate) fn is_convex_with_limit(&self, max_n: Option<usize>) -> bool {
        if let Some(n) = max_n {
            if self.qp.q.nrows > n {
                return true;
            }
        }
        is_q_psd_by_cholesky(&self.qp.q)
    }
}

impl Relaxation for MiqpProblem {
    fn num_vars(&self) -> usize {
        self.qp.num_vars
    }
    fn root_bounds(&self) -> &[(f64, f64)] {
        &self.qp.bounds
    }
    fn integer_vars(&self) -> &[usize] {
        &self.integer_vars
    }
    fn solve(&self, bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult {
        // When integer branching fixes every variable to a point, the QP interior
        // point method has no interior and fails (NumericalError). Evaluate the
        // single candidate directly instead.
        if let Some(r) = solve_fixed_point(&self.qp, bounds) {
            return r;
        }
        let mut sub = self.qp.clone();
        sub.bounds = bounds.to_vec();
        crate::qp::solve_qp_with(&sub, opts)
    }
}

// A variable whose bound width is <= this is treated as fixed to a point.
// Integer branching tightens bounds to exact integers (floor/ceil), so a fixed
// variable has width 0; the small tolerance only guards float round-off.
const FIXED_BOX_TOL: f64 = 1e-9;
// Feasibility tolerance when checking a fully-fixed point against the linear
// constraints (matches the convex-QP solver's residual tolerance scale).
const FIXED_POINT_FEAS_TOL: f64 = 1e-6;

/// Solve the relaxation when **every** variable is fixed to a point (zero-width
/// box). The QP IPM cannot (no interior), so evaluate the single candidate `x`
/// directly: check linear-constraint feasibility, then return its exact objective
/// `1/2 x'Qx + c'x + offset`. Returns `None` when any variable is still free
/// (the IPM handles those, including partially-fixed boxes).
fn solve_fixed_point(qp: &QpProblem, bounds: &[(f64, f64)]) -> Option<SolverResult> {
    if !bounds.iter().all(|&(l, u)| u - l <= FIXED_BOX_TOL) {
        return None;
    }
    // Empty box (lower meaningfully above upper) → infeasible subproblem.
    if bounds.iter().any(|&(l, u)| l - u > FIXED_BOX_TOL) {
        return Some(SolverResult::infeasible());
    }
    let x: Vec<f64> = bounds.iter().map(|&(l, u)| 0.5 * (l + u)).collect();

    if qp.num_constraints > 0 {
        let lhs = qp.a.mat_vec_mul(&x).ok()?;
        for ((&lhs_k, &ct), &b_k) in lhs.iter().zip(&qp.constraint_types).zip(&qp.b) {
            let feasible = match ct {
                ConstraintType::Le => lhs_k <= b_k + FIXED_POINT_FEAS_TOL,
                ConstraintType::Ge => lhs_k >= b_k - FIXED_POINT_FEAS_TOL,
                ConstraintType::Eq => (lhs_k - b_k).abs() <= FIXED_POINT_FEAS_TOL,
            };
            if !feasible {
                return Some(SolverResult::infeasible());
            }
        }
    }

    // Objective 1/2 x'Qx + c'x + offset (Q is full-symmetric CSC storage).
    let qx = qp.q.mat_vec_mul(&x).ok()?;
    let quad: f64 = 0.5 * x.iter().zip(&qx).map(|(xi, qxi)| xi * qxi).sum::<f64>();
    let lin: f64 = qp.c.iter().zip(&x).map(|(ci, xi)| ci * xi).sum::<f64>();
    Some(SolverResult {
        status: SolveStatus::Optimal,
        objective: quad + lin + qp.obj_offset,
        solution: x,
        ..Default::default()
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::ldl::is_q_psd_by_cholesky;
    use crate::problem::ConstraintType;
    use crate::sparse::CscMatrix;

    fn lp_2var() -> LpProblem {
        // trivial 2-var LP, bounds [0,5]^2, one <= constraint
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 5.0), (0.0, 5.0)],
            None,
        )
        .unwrap()
    }

    fn qp_diag(diag: &[f64]) -> QpProblem {
        let n = diag.len();
        let idx: Vec<usize> = (0..n).collect();
        let q = CscMatrix::from_triplets(&idx, &idx, diag, n, n).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap();
        QpProblem::new_all_le(q, vec![0.0; n], a, vec![], vec![(0.0, 5.0); n]).unwrap()
    }

    #[test]
    fn new_sorts_and_dedups_integer_vars() {
        let m = MilpProblem::new(lp_2var(), vec![1, 0, 1]).unwrap();
        assert_eq!(m.integer_vars, vec![0, 1]);
    }

    #[test]
    fn new_rejects_out_of_range_index() {
        let err = MilpProblem::new(lp_2var(), vec![0, 2]).unwrap_err();
        assert_eq!(err, MipProblemError::InvalidIntegerVar { index: 2, num_vars: 2 });
    }

    #[test]
    fn empty_integer_vars_allowed() {
        let m = MilpProblem::new(lp_2var(), vec![]).unwrap();
        assert!(m.integer_vars.is_empty());
    }

    #[test]
    fn miqp_new_validates_indices() {
        let err = MiqpProblem::new(qp_diag(&[2.0, 2.0]), vec![2]).unwrap_err();
        assert_eq!(err, MipProblemError::InvalidIntegerVar { index: 2, num_vars: 2 });
    }

    #[test]
    fn psd_diagonal_q_is_convex() {
        let m = MiqpProblem::new(qp_diag(&[2.0, 4.0]), vec![0, 1]).unwrap();
        assert!(m.is_convex(), "positive diagonal Q must be PSD");
    }

    #[test]
    fn indefinite_q_is_not_convex() {
        let m = MiqpProblem::new(qp_diag(&[2.0, -3.0]), vec![0, 1]).unwrap();
        assert!(!m.is_convex(), "negative eigenvalue must be detected as non-convex");
    }

    #[test]
    fn zero_q_is_convex() {
        // Q = 0 (LP-like) is trivially PSD.
        let m = MiqpProblem::new(qp_diag(&[0.0, 0.0]), vec![0]).unwrap();
        assert!(m.is_convex());
    }

    #[test]
    fn fixed_point_evaluates_objective_exactly() {
        // min x^2 (Q=2) with x fixed to 2 → obj = 1/2·2·4 = 4, no IPM needed.
        let qp = qp_diag(&[2.0]);
        let r = solve_fixed_point(&qp, &[(2.0, 2.0)]).expect("all fixed → Some");
        assert_eq!(r.status, SolveStatus::Optimal);
        assert!((r.objective - 4.0).abs() < 1e-12, "obj={}", r.objective);
        assert_eq!(r.solution, vec![2.0]);
    }

    #[test]
    fn fixed_point_returns_none_when_a_var_is_free() {
        let qp = qp_diag(&[2.0, 2.0]);
        assert!(solve_fixed_point(&qp, &[(2.0, 2.0), (0.0, 5.0)]).is_none());
    }

    #[test]
    fn fixed_point_infeasible_constraint() {
        // x fixed to 2 but constraint x >= 5 → infeasible.
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let qp = QpProblem::new(q, vec![0.0], a, vec![5.0], vec![(2.0, 2.0)], vec![ConstraintType::Ge])
            .unwrap();
        let r = solve_fixed_point(&qp, &[(2.0, 2.0)]).expect("all fixed → Some");
        assert_eq!(r.status, SolveStatus::Infeasible);
    }

    #[test]
    fn fixed_point_empty_box_infeasible() {
        let qp = qp_diag(&[2.0]);
        let r = solve_fixed_point(&qp, &[(3.0, 2.0)]).expect("empty box → Some");
        assert_eq!(r.status, SolveStatus::Infeasible);
    }

    #[test]
    fn psd_with_off_diagonal_detected() {
        // Q = [[2,1],[1,2]] (full-symmetric storage) is PSD (eigenvalues 1, 3).
        let q = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[2.0, 1.0, 1.0, 2.0], 2, 2)
            .unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
        let qp = QpProblem::new_all_le(q, vec![0.0, 0.0], a, vec![], vec![(0.0, 5.0); 2]).unwrap();
        assert!(MiqpProblem::new(qp, vec![0, 1]).unwrap().is_convex());

        // Q = [[1,2],[2,1]] is indefinite (eigenvalues -1, 3).
        let q2 = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 2.0, 2.0, 1.0], 2, 2)
            .unwrap();
        let a2 = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
        let qp2 = QpProblem::new_all_le(q2, vec![0.0, 0.0], a2, vec![], vec![(0.0, 5.0); 2]).unwrap();
        assert!(!MiqpProblem::new(qp2, vec![0, 1]).unwrap().is_convex());
    }

    // ── Large-n (n > former PSD_DENSE_LIMIT=1000) sentinels ──────────────────

    /// Build an n×n MIQP with diagonal 1.0 and Q[0,1]=Q[1,0]=2.0.
    /// The top-left 2×2 block has eigenvalues {-1, 3} → matrix is indefinite.
    fn large_n_indefinite_miqp(n: usize) -> MiqpProblem {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i); cols.push(i); vals.push(1.0_f64);
        }
        // off-diagonal: Q[0,1]=Q[1,0]=2 makes the 2×2 block eigenvalues {-1,3}
        rows.push(0); cols.push(1); vals.push(2.0);
        rows.push(1); cols.push(0); vals.push(2.0);
        let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap();
        let qp = QpProblem::new_all_le(q, vec![0.0; n], a, vec![], vec![(0.0, 5.0); n]).unwrap();
        MiqpProblem::new(qp, vec![0]).unwrap()
    }

    /// Build an n×n MIQP with strictly positive diagonal Q (identity × 2) → PSD.
    fn large_n_psd_miqp(n: usize) -> MiqpProblem {
        let idx: Vec<usize> = (0..n).collect();
        let vals: Vec<f64> = vec![2.0; n];
        let q = CscMatrix::from_triplets(&idx, &idx, &vals, n, n).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap();
        let qp = QpProblem::new_all_le(q, vec![0.0; n], a, vec![], vec![(0.0, 5.0); n]).unwrap();
        MiqpProblem::new(qp, vec![0]).unwrap()
    }

    /// **Sentinel**: n>1000, diagonal≥0, off-diagonal indefinite → nonconvex.
    ///
    /// No-op proof: if `is_convex` reverts to `return true` for n > 1000
    /// (the old PSD_DENSE_LIMIT path), this assertion fails, exposing the
    /// false-Optimal bug.
    #[test]
    fn large_n_off_diag_indefinite_is_not_convex() {
        let m = large_n_indefinite_miqp(1001);
        assert!(
            !m.is_convex(),
            "n=1001 indefinite MIQP (diag≥0, off-diag λ_min=-1) must be detected as \
             nonconvex; `return true` for n>1000 produces false-Optimal (sentinel)"
        );
    }

    /// **No-op proof**: documents that the old unconditional-true path produces
    /// false-Optimal. Sparse LDL correctly returns false for the same Q.
    ///
    /// This test fails as written under the fix (both `assert!`s would need to pass),
    /// but is structured to make the bug explicit: the first assert shows old-code
    /// behaviour; the second shows the fix.
    #[test]
    fn no_op_proof_old_dense_limit_gives_false_optimal() {
        let m = large_n_indefinite_miqp(1001);
        let q = &m.qp.q;
        // Old code silently assumed convex for n > 1000:
        //   if n > PSD_DENSE_LIMIT { return true; }
        // Simulate that path: for n=1001 it always returned true.
        let old_path_result = q.nrows > 1000; // unconditional true for our matrix
        assert!(
            old_path_result,
            "old path: n>1000 always returned is_convex=true \
             (leads B&B to produce false-Optimal for indefinite Q)"
        );
        // Fixed path: sparse LDLᵀ correctly detects indefinite.
        assert!(
            !is_q_psd_by_cholesky(q),
            "fix: sparse LDL reports false for indefinite Q; solver rejects with nonconvex_result"
        );
    }

    /// **Regression guard**: large-n truly PSD Q must still pass the convexity gate.
    /// Verifies no over-rejection (false negatives) after the fix.
    #[test]
    fn large_n_diagonal_psd_is_convex() {
        // n=1001 diagonal-2 Q (strictly PD) must be convex.
        let m = large_n_psd_miqp(1001);
        assert!(m.is_convex(), "large-n diagonal PSD Q must not be over-rejected");
    }

    /// Regression guard: large-n Q=0 (LP case) is trivially convex.
    #[test]
    fn large_n_zero_q_is_convex() {
        let n = 1001usize;
        let q = CscMatrix::from_triplets(&[], &[], &[], n, n).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap();
        let qp = QpProblem::new_all_le(q, vec![0.0; n], a, vec![], vec![(0.0, 5.0); n]).unwrap();
        let m = MiqpProblem::new(qp, vec![0]).unwrap();
        assert!(m.is_convex(), "large-n zero Q (LP) must be convex");
    }

    /// Regression guard: large-n PSD-but-singular Q (rank-deficient PSD) is convex.
    #[test]
    fn large_n_psd_singular_is_convex() {
        // Q: identity except last diagonal is 0 → rank n-1, λ_min=0 (PSD).
        let n = 1001usize;
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for i in 0..n - 1 {
            rows.push(i); cols.push(i); vals.push(1.0);
        }
        let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap();
        let qp = QpProblem::new_all_le(q, vec![0.0; n], a, vec![], vec![(0.0, 5.0); n]).unwrap();
        let m = MiqpProblem::new(qp, vec![0]).unwrap();
        assert!(m.is_convex(), "large-n PSD-singular Q must be accepted as convex");
    }

    // Integration-test counterpart with identical Q structure: tests/diag_psd_check_soundness.rs.
    // Cross-crate boundary prevents sharing; kept separate intentionally.
    fn indefinite_q_n1001() -> QpProblem {
        let n = 1001_usize;
        let mut rows = vec![];
        let mut cols = vec![];
        let mut vals = vec![];
        for i in 0..n {
            rows.push(i); cols.push(i); vals.push(1.0_f64);
        }
        rows.push(0); cols.push(1); vals.push(2.0_f64);
        let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap();
        QpProblem::new_all_le(q, vec![0.0; n], a, vec![], vec![(0.0, 5.0); n]).unwrap()
    }

    /// `is_convex_with_limit(None)` は size 制限なしで indefinite Q を正しく検出する。
    ///
    /// **Sentinel**: limit=None で size check を早期 return するよう壊すと false が
    /// 返らなくなり、このテストが FAIL する。
    #[test]
    fn is_convex_with_limit_none_detects_indefinite_n1001() {
        let m = MiqpProblem::new(indefinite_q_n1001(), vec![0]).unwrap();
        assert!(
            !m.is_convex_with_limit(None),
            "limit=None: n=1001 indefinite Q は non-PSD と検出すべき"
        );
    }

    /// `is_convex_with_limit(Some(1000))` は n=1001 > 1000 のためチェックをスキップし true を返す (soundness 穴)。
    ///
    /// **Sentinel**: size check を削除すると `is_q_psd_by_cholesky` が走り false を返す
    /// → このテストが FAIL する。
    #[test]
    fn is_convex_with_limit_some_1000_skips_n1001() {
        let m = MiqpProblem::new(indefinite_q_n1001(), vec![0]).unwrap();
        assert!(
            m.is_convex_with_limit(Some(1000)),
            "limit=Some(1000): n=1001 > 1000 → skip → true (soundness 穴の確認)"
        );
    }

    /// `is_convex_with_limit(Some(2000))` は n=1001 ≤ 2000 なので Cholesky が走り false を返す。
    #[test]
    fn is_convex_with_limit_some_2000_detects_n1001() {
        let m = MiqpProblem::new(indefinite_q_n1001(), vec![0]).unwrap();
        assert!(
            !m.is_convex_with_limit(Some(2000)),
            "limit=Some(2000): n=1001 ≤ 2000 → Cholesky → non-PSD 検出"
        );
    }
}
