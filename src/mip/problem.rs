//! MILP / MIQP problem definitions (#14).
//!
//! A MILP/MIQP is a continuous LP/QP **relaxation** plus a set of variables
//! constrained to integer values. We deliberately wrap the existing
//! [`LpProblem`] / [`QpProblem`] rather than adding an integrality field to them:
//! the continuous solvers stay untouched (zero regression surface), and each
//! branch-and-bound node solves the relaxation by swapping `bounds` — the same
//! mechanism the spatial QP B&B already uses for box subproblems.

use super::Relaxation;
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::qp::QpProblem;
use crate::sparse::CscMatrix;

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
        q_is_psd(&self.qp.q, self.qp.num_vars)
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

// PSD-check tolerances. These mirror the QP convexity check
// (`qp::check_q_positive_semidefinite`); they are duplicated here to avoid
// touching the live QP solver module during MIP bring-up. Consolidating the two
// into one shared convexity check is a follow-up refactor.
const PSD_NEG_TOL_RATIO: f64 = 1e-6; // reject a diagonal more negative than ratio * max|Q|
const PSD_ABS_FLOOR: f64 = 1e-12; // floor for the negativity tolerance
// Diagonal regularization added before Cholesky. NOTE: this intentionally accepts
// Q whose smallest eigenvalue lies in `[-eps, 0)` as PSD — i.e. tiny negative
// eigenvalues within the regularization are *masked*. This keeps PSD-but-singular
// and round-off-perturbed convex Q solvable; a genuinely indefinite Q (eigenvalue
// below `-eps`) still produces a non-positive Cholesky pivot and is rejected.
const PSD_CHOL_EPS_RATIO: f64 = 1e-4; // eps = ratio * max|Q|, floored by PSD_EPS_FLOOR
const PSD_EPS_FLOOR: f64 = 1e-8; // floor for the regularization
const PSD_DENSE_LIMIT: usize = 1000; // skip the O(n^3) check above this size (assume convex)

/// Whether the symmetric matrix `Q` (full-symmetric CSC storage, `n x n`) is
/// positive semidefinite, within the tolerances above. A clearly indefinite `Q`
/// (a negative eigenvalue beyond the regularization) makes the regularized dense
/// Cholesky hit a non-positive pivot and returns `false`. PSD-but-singular `Q`
/// passes (the regularization keeps it convex-solvable).
fn q_is_psd(q: &CscMatrix, n: usize) -> bool {
    if n == 0 {
        return true;
    }
    let q_abs_max = q.values.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));

    // Quick reject: a diagonal entry that is meaningfully negative cannot be PSD.
    let neg_tol = (q_abs_max * PSD_NEG_TOL_RATIO).max(PSD_ABS_FLOOR);
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            if q.row_ind[k] == col && q.values[k] < -neg_tol {
                return false;
            }
        }
    }

    if n > PSD_DENSE_LIMIT {
        // O(n^3) dense Cholesky is too costly; assume convex (consistent with the
        // QP routing heuristic). Such a large integer QP is out of practical scope.
        return true;
    }

    let eps = (q_abs_max * PSD_CHOL_EPS_RATIO).max(PSD_EPS_FLOOR);
    let mut a = vec![0.0_f64; n * n];
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            if row <= col {
                let v = q.values[k];
                a[row * n + col] = v;
                if row != col {
                    a[col * n + row] = v;
                }
            }
        }
    }
    for i in 0..n {
        a[i * n + i] += eps;
    }

    // Dense L L^T factorization; a non-positive pivot means not PSD.
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 0.0 {
            return false;
        }
        let sqrt_d = d.sqrt();
        a[j * n + j] = sqrt_d;
        for i in (j + 1)..n {
            let mut l_ij = a[i * n + j];
            for k in 0..j {
                l_ij -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = l_ij / sqrt_d;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ConstraintType;

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
}
