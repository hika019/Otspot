//! Global optimisation of (possibly nonconvex) QCQP by spatial branch-and-bound
//! with McCormick relaxations of the bilinear/quadratic terms.
//!
//! Handles indefinite objective/constraint matrices. Bound finiteness is
//! required only for the §3.1 bound-requirement set `R` (variables whose lift
//! pairs enter a McCormick envelope or a rule-2-vetoed diagonal, see
//! [`NonconvexQcqp::validate`] and [`super::qcqp_classify`]); variables outside `R`
//! may stay free — the relaxation then omits their box rows and, for every
//! pair touching a non-finite endpoint, all envelope rows, so no `±inf`
//! coefficient ever reaches an LP. Termination on free variables is guarded
//! by the recession-ray gate on `Unbounded` relaxations (`global_core`).

use super::{ConeSpec, ConicOptions, ConicProblem};
use crate::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot_num::sparse::CscMatrix;

/// A (possibly nonconvex) quadratic constraint `(1/2) x^T P x + q^T x + r <= 0`.
#[derive(Debug, Clone)]
pub struct GQuadConstraint {
    /// Symmetric matrix `P` (`n x n`), may be indefinite.
    pub p: CscMatrix,
    /// Linear term.
    pub q: Vec<f64>,
    /// Constant.
    pub r: f64,
}

/// A (possibly nonconvex) QCQP. Bounds must be finite exactly for the
/// variables in the §3.1 bound-requirement set `R` (see [`Self::validate`]);
/// other variables may be free (`±inf`).
#[derive(Debug, Clone)]
pub struct NonconvexQcqp {
    /// Number of variables.
    pub n: usize,
    /// Objective matrix `P0` (symmetric, may be indefinite); `None` = linear.
    pub p0: Option<CscMatrix>,
    /// Objective linear term.
    pub q0: Vec<f64>,
    /// Quadratic constraints.
    pub quad: Vec<GQuadConstraint>,
    /// Linear inequalities `Gl x <= hl`.
    pub g_lin: CscMatrix,
    /// rhs.
    pub h_lin: Vec<f64>,
    /// Linear equalities `Ae x = be`.
    pub a_eq: CscMatrix,
    /// rhs.
    pub b_eq: Vec<f64>,
    /// Variable lower bounds (`-inf` = free, allowed only outside `R`).
    pub lb: Vec<f64>,
    /// Variable upper bounds (`inf` = free, allowed only outside `R`).
    pub ub: Vec<f64>,
}

impl NonconvexQcqp {
    /// Validate dimensional consistency and bound finiteness that
    /// `build_static`/`global_core` assume but never check themselves: a
    /// mismatched `q0`/`lb`/`ub`/`p0`/`quad[..]` dimension previously slice-
    /// copied or indexed out of bounds and panicked instead of failing as an
    /// ordinary invalid-input error (mirrors `ConicProblem::validate` /
    /// `MisocpProblem::validate`, PR #25 review INLINE-S).
    pub fn validate(&self) -> Result<(), String> {
        let n = self.n;
        if self.q0.len() != n {
            return Err(format!("q0 length {} != n {n}", self.q0.len()));
        }
        if self.lb.len() != n {
            return Err(format!("lb length {} != n {n}", self.lb.len()));
        }
        if self.ub.len() != n {
            return Err(format!("ub length {} != n {n}", self.ub.len()));
        }
        if let Some(p0) = &self.p0 {
            if p0.nrows() != n || p0.ncols() != n {
                return Err(format!(
                    "p0 shape ({}, {}) != (n, n) = ({n}, {n})",
                    p0.nrows(),
                    p0.ncols()
                ));
            }
        }
        for (k, qc) in self.quad.iter().enumerate() {
            if qc.p.nrows() != n || qc.p.ncols() != n {
                return Err(format!(
                    "quad[{k}].p shape ({}, {}) != (n, n) = ({n}, {n})",
                    qc.p.nrows(),
                    qc.p.ncols()
                ));
            }
            if qc.q.len() != n {
                return Err(format!("quad[{k}].q length {} != n {n}", qc.q.len()));
            }
        }
        if self.g_lin.ncols() != n {
            return Err(format!(
                "g_lin column count {} != n {n}",
                self.g_lin.ncols()
            ));
        }
        if self.g_lin.nrows() != self.h_lin.len() {
            return Err(format!(
                "g_lin row count {} != h_lin length {}",
                self.g_lin.nrows(),
                self.h_lin.len()
            ));
        }
        if self.a_eq.ncols() != n {
            return Err(format!("a_eq column count {} != n {n}", self.a_eq.ncols()));
        }
        if self.a_eq.nrows() != self.b_eq.len() {
            return Err(format!(
                "a_eq row count {} != b_eq length {}",
                self.a_eq.nrows(),
                self.b_eq.len()
            ));
        }
        // Decision D8 / plan §3.2: bound finiteness is required only for the
        // variables in the §3.1 bound-requirement set `R` (variables whose
        // lift pairs feed a McCormick envelope or a rule-2-vetoed diagonal).
        // A free variable outside `R` is exactly what plan §5 example 1 (PSD
        // ellipsoid rows over free `x`) must accept; the R-aware relaxation
        // (skip box rows on non-finite node bounds, emit no envelope rows for
        // non-finite pair endpoints) never hands `±inf` to the LP. `NaN`
        // bounds stay rejected for *every* variable: they are corrupt data,
        // not "free", and would poison node boxes with NaN widths.
        let req =
            super::qcqp_classify::classify_bound_requirements(n, self.p0.as_ref(), &self.quad)?;
        for (k, (&l, &u)) in self.lb.iter().zip(&self.ub).enumerate() {
            if l.is_nan() {
                return Err(format!("lb[{k}] is not finite: {l}"));
            }
            if u.is_nan() {
                return Err(format!("ub[{k}] is not finite: {u}"));
            }
            if req.required[k] && (!l.is_finite() || !u.is_finite()) {
                return Err(format!(
                    "lb[{k}] = {l} / ub[{k}] = {u}: variable is in the bound-requirement \
                     set R (its lift needs finite McCormick boxes)"
                ));
            }
            if l > u {
                return Err(format!("lb[{k}] = {l} > ub[{k}] = {u}"));
            }
        }
        // Numeric finiteness of the objective/constraint data itself: `build_static`
        // and `build_relax` copy every one of these fields straight into the node
        // relaxations `global_core` solves via `solve_relax_lp`, whose
        // `LpProblem::new_general` call is `.unwrap()`-ed on the assumption its own
        // finiteness check never fires -- a NaN/inf here (dimensionally valid, so
        // none of the checks above catch it) reaches that `.unwrap()` and panics
        // instead of failing as an ordinary invalid-input error (Codex review R3,
        // nonconvex.rs:58).
        super::find_non_finite("q0", &self.q0)?;
        if let Some(p0) = &self.p0 {
            super::find_non_finite("p0", p0.values())?;
        }
        for (k, qc) in self.quad.iter().enumerate() {
            super::find_non_finite(&format!("quad[{k}].p"), qc.p.values())?;
            super::find_non_finite(&format!("quad[{k}].q"), &qc.q)?;
            if !qc.r.is_finite() {
                return Err(format!("quad[{k}].r is not finite: {}", qc.r));
            }
        }
        super::find_non_finite("g_lin", self.g_lin.values())?;
        super::find_non_finite("h_lin", &self.h_lin)?;
        super::find_non_finite("a_eq", self.a_eq.values())?;
        super::find_non_finite("b_eq", &self.b_eq)?;
        Ok(())
    }
}

/// Result of global QCQP optimisation.
///
/// With an incumbent, `status` is `Timeout` (deadline hit), `Optimal` (search
/// exhausted, no node relaxation failed, and `gap <= gap_tol` — a certificate),
/// or `SuboptimalSolution` (node-limited, or exhausted but some relaxation
/// failed without a certificate, or the proven gap exceeds `gap_tol`).
///
/// Without an incumbent the status is the first applicable of `Timeout`,
/// `MaxIterations`, `NumericalError` (some region was never certified empty),
/// `Infeasible` (every region proven empty — only then is infeasibility a
/// certificate).
///
/// Either way, a relaxation `Unbounded` whose verified ray passes the plan
/// §3.2 recession verification against a base point feasible under the
/// original `feasible()` contract overrides the search: `status = Unbounded`,
/// `objective = -inf`, `x` = the verified base point. A ray that cannot be
/// certified instead yields `NotSupported(reason)` (never `numerical_failures`
/// counting; plan §3.2 line 111).
#[derive(Debug, Clone)]
pub struct GlobalResult {
    /// Termination status (see struct docs for the exact classification).
    pub status: SolveStatus,
    /// Best (incumbent) objective.
    pub objective: f64,
    /// Best solution.
    pub x: Vec<f64>,
    /// Nodes processed.
    pub nodes: usize,
    /// Optimality gap `incumbent - lower_bound` at termination, where
    /// `lower_bound` is the proven global lower bound: the minimum over
    /// terminal regions of their relaxation bounds and over unresolved
    /// regions (open stack nodes, failed relaxations) of their inherited
    /// parent bounds (`-inf` when the root itself is unresolved).
    pub gap: f64,
}

/// Spatial B&B options.
#[derive(Debug, Clone)]
pub struct GlobalOptions {
    /// Absolute optimality gap tolerance.
    pub gap_tol: f64,
    /// McCormick tightness tolerance (leaf when max term gap below this).
    pub bilinear_tol: f64,
    /// Feasibility tolerance for accepting incumbents.
    pub feas_tol: f64,
    /// Node limit.
    pub max_nodes: usize,
    /// Integrality tolerance (for mixed-integer variants).
    pub int_tol: f64,
}

impl Default for GlobalOptions {
    fn default() -> Self {
        Self {
            gap_tol: 1e-5,
            bilinear_tol: 1e-6,
            feas_tol: 1e-6,
            max_nodes: 50_000,
            int_tol: 1e-6,
        }
    }
}

impl GlobalOptions {
    /// Validate option ranges `global_core` assumes but never checks itself
    /// (same pattern as `misocp::BbOptions::validate`, PR #25 review "Reject
    /// NaN integrality tolerances before branching"). A non-finite `int_tol`
    /// makes every `frac_dist(..) <= tol` / `d > worst_if` comparison
    /// silently `false`, so `all_integral` never accepts a point (every
    /// incumbent is rejected) while `int_branch`-finding never fires either
    /// (no variable is ever picked to branch on) -- the search then runs dry
    /// without ever finding the true integer optimum.
    pub fn validate(&self) -> Result<(), String> {
        if !self.gap_tol.is_finite() || self.gap_tol < 0.0 {
            return Err(format!(
                "gap_tol must be finite and >= 0, got {}",
                self.gap_tol
            ));
        }
        if !self.bilinear_tol.is_finite() || self.bilinear_tol < 0.0 {
            return Err(format!(
                "bilinear_tol must be finite and >= 0, got {}",
                self.bilinear_tol
            ));
        }
        if !self.feas_tol.is_finite() || self.feas_tol < 0.0 {
            return Err(format!(
                "feas_tol must be finite and >= 0, got {}",
                self.feas_tol
            ));
        }
        if !self.int_tol.is_finite() || self.int_tol < 0.0 {
            return Err(format!(
                "int_tol must be finite and >= 0, got {}",
                self.int_tol
            ));
        }
        Ok(())
    }
}

/// `p`'s entry at `(row, col)`, or `0.0` if not stored explicitly:
/// `O(log deg(col))` via binary search on the column's sorted row indices.
/// Reads both triangular halves of a symmetric quadratic matrix without ever
/// materializing it as a dense `n x n` array (`to_dense_rows` was 10 GB at
/// `n = 100_000` and was rebuilt on every branch-and-bound node).
fn get_entry(p: &CscMatrix, row: usize, col: usize) -> f64 {
    let cp = p.col_ptr();
    let start = cp[col];
    let end = cp[col + 1];
    p.row_ind()[start..end]
        .binary_search(&row)
        .map(|pos| p.values()[start + pos])
        .unwrap_or(0.0)
}

/// Collect the symmetric index pairs `(i,j)` with `i<=j` that appear with a
/// nonzero coefficient in any quadratic matrix, scanning each matrix's own
/// CSC nonzeros directly (`O(nnz(P))` total, ascending by construction)
/// instead of a dense `n x n` bitset (`vec![false; n*n]` is 10 GB at
/// `n = 100_000`).
fn collect_pairs(qp: &NonconvexQcqp) -> Vec<(usize, usize)> {
    let mut seen: std::collections::BTreeSet<(usize, usize)> = std::collections::BTreeSet::new();
    let mut mark_matrix = |p: &CscMatrix| {
        let cp = p.col_ptr();
        let ri = p.row_ind();
        for j in 0..p.ncols() {
            for k in cp[j]..cp[j + 1] {
                let i = ri[k];
                seen.insert(if i <= j { (i, j) } else { (j, i) });
            }
        }
    };
    if let Some(p0) = &qp.p0 {
        mark_matrix(p0);
    }
    for qc in &qp.quad {
        mark_matrix(&qc.p);
    }
    seen.into_iter().collect()
}

/// `0.5 * x^T P x`.
fn quad_val(p: &CscMatrix, x: &[f64]) -> f64 {
    let px = p.mat_vec_mul(x).expect(
        "p.ncols() == x.len(): p is qp.p0/quad[k].p (n,n), validated by \
             NonconvexQcqp::validate() at global_core entry; x is always qp.n \
             long (res.x[..qp.n] truncation in global_core)",
    );
    0.5 * x.iter().zip(&px).map(|(a, b)| a * b).sum::<f64>()
}

struct RelaxResult {
    status: SolveStatus,
    objective: f64,
    x: Vec<f64>,
    /// 緩和 LP の検証済み recession ray (方向 + 実行可能基点、`[x, w]` の
    /// 元座標)。`Some` は `status == Unbounded` かつ LP simplex 側が
    /// Phase II ray ゲート通過後に元問題空間で独立検証したときのみ
    /// (`simplex::unbounded_ray::lp_ray_payload`、`SolverResult::ray_certificate`
    /// と同一契約・同一型)。未証明 Unbounded、presolve 縮約経由の判定、
    /// 基点の実行可能性検証失敗はすべて `None` (偽証なし)。
    /// 計画 §3.2 の recession 検証 (`verify_recession_unboundedness`) が消費する。
    ray: Option<crate::problem::RayCertificate>,
}

#[cfg(test)]
thread_local! {
    /// Fault injection for `solve_relax_lp` (test-only). Each relaxation solve
    /// pops the front entry: `Some(status)` short-circuits the LP and returns
    /// that status (no solution); `None` (or an exhausted queue) solves
    /// normally. `thread_local` so parallel tests cannot corrupt each other's
    /// plan (same rationale as `simplex::primal::PIVOT_OUT_BTRAN_COUNT`).
    static RELAX_STATUS_PLAN: std::cell::RefCell<std::collections::VecDeque<Option<SolveStatus>>> =
        const { std::cell::RefCell::new(std::collections::VecDeque::new()) };
}

#[cfg(test)]
thread_local! {
    /// Test-only cancel injection for the McCormick B&B's final-classification
    /// stop check (横展開: same "last region prunes stack empty, no
    /// re-check before declaring Infeasible" gap as `misocp::solve_misocp`).
    /// Counts calls to `test_maybe_cancel_before_final_classification`; once
    /// the count reaches `CANCEL_AFTER_FINAL_CLASSIFICATION_COMMIT`, flips the
    /// real `cancel_flag` `AtomicBool` a test threads through
    /// `ConicOptions::cancel_flag`.
    static FINAL_CLASSIFICATION_COMMIT_COUNT: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static CANCEL_AFTER_FINAL_CLASSIFICATION_COMMIT: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn test_maybe_cancel_before_final_classification(opts: &ConicOptions) {
    let count = FINAL_CLASSIFICATION_COMMIT_COUNT.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    if CANCEL_AFTER_FINAL_CLASSIFICATION_COMMIT.with(std::cell::Cell::get) == Some(count) {
        if let Some(flag) = &opts.cancel_flag {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// McCormick relaxations are pure LPs. Use the mature simplex LP path rather
/// than the generic conic IPM, which is intentionally small and aimed at SOC
/// blocks. The caller's wall-clock deadline is forwarded so a single node LP
/// cannot run past the B&B budget.
///
/// The shared `cancel_flag` is forwarded alongside the deadline for the same
/// reason: the B&B loop only re-reads `stop_requested()` *between* nodes, so a
/// cancel fired while one node relaxation is mid-solve would otherwise run to
/// completion, breaking `set_cancel_flag`'s cooperative-cancellation contract
/// on a single expensive node. `SolverOptions::cancel_flag` is checked at the
/// same cadence as the LP's wall-clock deadline (`external_stop_requested`).
///
/// Builds the combined equality+inequality matrix directly from `prob.a`/
/// `prob.g`'s own CSC nonzeros (`O(nnz(a) + nnz(g))`) instead of densifying
/// both to `Vec<Vec<f64>>` first -- `prob` is the per-node relaxation, so a
/// dense pass here repeated the same `O(n * m)` blowup `build_relax` is
/// fixed to avoid.
fn solve_relax_lp(
    prob: &ConicProblem,
    deadline: Option<std::time::Instant>,
    cancel_flag: Option<&std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> RelaxResult {
    #[cfg(test)]
    if let Some(status) = RELAX_STATUS_PLAN
        .with(|p| p.borrow_mut().pop_front())
        .flatten()
    {
        return RelaxResult {
            status,
            objective: f64::NAN,
            x: vec![],
            ray: None,
        };
    }
    let n = prob.n();
    let peq = prob.p();
    let mrows = prob.m();
    let mut rr = Vec::with_capacity(prob.a.nnz() + prob.g.nnz());
    let mut cc = Vec::with_capacity(prob.a.nnz() + prob.g.nnz());
    let mut vv = Vec::with_capacity(prob.a.nnz() + prob.g.nnz());
    let acp = prob.a.col_ptr();
    let ari = prob.a.row_ind();
    let ava = prob.a.values();
    for j in 0..n {
        for k in acp[j]..acp[j + 1] {
            rr.push(ari[k]);
            cc.push(j);
            vv.push(ava[k]);
        }
    }
    let gcp = prob.g.col_ptr();
    let gri = prob.g.row_ind();
    let gva = prob.g.values();
    for j in 0..n {
        for k in gcp[j]..gcp[j + 1] {
            rr.push(peq + gri[k]);
            cc.push(j);
            vv.push(gva[k]);
        }
    }
    let mut rhs = Vec::with_capacity(peq + mrows);
    let mut ctypes = Vec::with_capacity(peq + mrows);
    rhs.extend_from_slice(&prob.b);
    ctypes.extend(std::iter::repeat_n(ConstraintType::Eq, peq));
    rhs.extend_from_slice(&prob.h);
    ctypes.extend(std::iter::repeat_n(ConstraintType::Le, mrows));
    let a = CscMatrix::from_triplets(&rr, &cc, &vv, peq + mrows, n).unwrap();
    let lp = LpProblem::new_general(
        prob.c.clone(),
        a,
        rhs,
        ctypes,
        vec![(f64::NEG_INFINITY, f64::INFINITY); n],
        None,
    )
    .unwrap();
    let lp_opts = crate::options::SolverOptions {
        deadline,
        cancel_flag: cancel_flag.cloned(),
        ..Default::default()
    };
    let res = crate::lp::solve_lp_with(&lp, &lp_opts);
    RelaxResult {
        status: res.status,
        objective: res.objective,
        x: res.solution,
        ray: res.ray_certificate,
    }
}

/// Portions of the McCormick relaxation LP that do not depend on the current
/// box `[lb, ub]`: the objective, the equality matrix, and the rows coming
/// from the quadratic constraints and the original linear inequalities.
/// Building these once (before the branch-and-bound loop) instead of inside
/// `build_relax` turns the per-node rebuild into an `O(nnz)` clone of this
/// struct plus `O(n + pairs.len())` new box/McCormick rows, instead of
/// re-deriving everything from a dense `n x n` pass at every node.
struct StaticRelax {
    n: usize,
    pairs: Vec<(usize, usize)>,
    /// Objective coefficients over `[x, w]` (length `n + pairs.len()`).
    c: Vec<f64>,
    /// Equality matrix over `[x, w]` (`b.len() x (n + pairs.len())`).
    a: CscMatrix,
    b: Vec<f64>,
    /// Fixed inequality-row triplets over `[x, w]`: quadratic constraints
    /// followed by the original linear inequalities `Gl x <= hl`.
    fixed_ri: Vec<usize>,
    fixed_ci: Vec<usize>,
    fixed_vi: Vec<f64>,
    fixed_h: Vec<f64>,
    /// Per pair: rule-2 diagonal (`i == j && rule2_diag[i]`) -- emitted as
    /// the single under-estimator cut `w_ii >= 0` (the x̄ = 0 tangent), no
    /// interval/envelope rows (decision D2, plan §3.2: interval rows and
    /// McCormick coefficients would push `w_hi = ±inf`/`inf` coefficients
    /// into `h`/`A` for free variables).
    rule2_pair: Vec<bool>,
    /// Per pair: full interval + 4-facet McCormick envelope emitted (both
    /// endpoints have finite root bounds and the pair is not a rule-2
    /// diagonal, decision D3's "両端の箱が有限" emission condition).
    mc_pair: Vec<bool>,
    /// Indices of the `mc_pair` entries: the *only* branching candidates and
    /// leaf-gap terms (decision D4, plan §3.2 line 94). Rule-2 diagonals are
    /// excluded because the relaxation intentionally under-estimates their
    /// `w_ii` (`|w_ii − x_i²| > bilinear_tol` would fire forever), and
    /// envelope-less pairs are excluded because no box-based bound tightens
    /// them. Both endpoints of every `mc_pair` have finite root bounds, so
    /// every spatial branch variable lives in a finite box and the
    /// `0.5·(lb+ub)` fallback can never return `∓inf` and re-push an
    /// identical box (plan §0.1 line 16 infinite-loop path, §5 sentinel 3).
    mc_pairs: Vec<usize>,
    /// Raw §3.1 bound-requirement set `R` (per variable). Consumed by the
    /// `d_R = 0` assertion of the recession verification (plan §3.2 line
    /// 108) — the relaxation box rows should make it structural, the
    /// verifier re-checks it against the certificate it consumes.
    required: Vec<bool>,
}

/// `(pair_index, coeff)` for each McCormick pair with a nonzero `(1/2) x^T P
/// x` coefficient in `p`. Mirrors the pre-fix dense formula exactly
/// (`coeff = 0.5*d[i][i]` on the diagonal, `0.5*(d[i][j]+d[j][i])`
/// off-diagonal): `get_entry` replaces `d[i][j]` with an `O(log deg)` sparse
/// lookup, so the arithmetic (and its rounding) is bit-for-bit identical to
/// the dense reference.
fn quad_pair_coeffs(p: &CscMatrix, pairs: &[(usize, usize)]) -> Vec<(usize, f64)> {
    let mut out = Vec::new();
    for (pi, &(i, j)) in pairs.iter().enumerate() {
        let coeff = if i == j {
            0.5 * get_entry(p, i, i)
        } else {
            0.5 * (get_entry(p, i, j) + get_entry(p, j, i))
        };
        if coeff != 0.0 {
            out.push((pi, coeff));
        }
    }
    out
}

/// Precompute the node-invariant part of the relaxation once per solve (see
/// `StaticRelax`'s doc).
///
/// The §3.1 classification runs here (once per solve, `O(nnz)` + one
/// Cholesky per quadratic row) because every per-pair emission decision
/// (D2/D3) is node-invariant: `global_core` branches only on `mc_pairs`,
/// whose endpoints have finite root bounds, so a node box can never turn a
/// skipped pair into an emittable one or vice versa. A classification error
/// (mis-shaped public fields / non-finite coefficients) is unreachable past
/// `NonconvexQcqp::validate` at the `global_core` entry, which checks exactly
/// the dimensions and finiteness the classifier reads.
fn build_static(qp: &NonconvexQcqp) -> StaticRelax {
    let n = qp.n;
    let pairs = collect_pairs(qp);
    let npair = pairs.len();
    let nv = n + npair;

    let req = super::qcqp_classify::classify_bound_requirements(n, qp.p0.as_ref(), &qp.quad)
        .expect(
            "classification reads only dimensions/coefficients validated by \
             NonconvexQcqp::validate at the global_core entry",
        );
    // D3 emission condition "both endpoints' boxes are finite". Root finiteness
    // is node-invariant: branching only ever splits `mc_pairs` endpoints (D4),
    // which are exactly the variables with finite root bounds among boxed
    // ones, so a variable is never boxed into (or out of) finiteness.
    let var_boxed: Vec<bool> = (0..n)
        .map(|j| qp.lb[j].is_finite() && qp.ub[j].is_finite())
        .collect();
    let rule2_pair: Vec<bool> = pairs
        .iter()
        .map(|&(i, j)| i == j && req.rule2_diag[i])
        .collect();
    let mc_pair: Vec<bool> = pairs
        .iter()
        .zip(&rule2_pair)
        .map(|(&(i, j), &rule2)| !rule2 && var_boxed[i] && (i == j || var_boxed[j]))
        .collect();
    let mc_pairs: Vec<usize> = (0..npair).filter(|&pi| mc_pair[pi]).collect();

    let mut c = vec![0.0; nv];
    c[..n].copy_from_slice(&qp.q0);
    if let Some(p0) = &qp.p0 {
        for (pi, coeff) in quad_pair_coeffs(p0, &pairs) {
            c[n + pi] += coeff;
        }
    }

    let peq = qp.b_eq.len();
    let acp = qp.a_eq.col_ptr();
    let ari = qp.a_eq.row_ind();
    let ava = qp.a_eq.values();
    let mut a_ri = Vec::with_capacity(qp.a_eq.nnz());
    let mut a_ci = Vec::with_capacity(qp.a_eq.nnz());
    let mut a_vi = Vec::with_capacity(qp.a_eq.nnz());
    for j in 0..n {
        for k in acp[j]..acp[j + 1] {
            a_ri.push(ari[k]);
            a_ci.push(j);
            a_vi.push(ava[k]);
        }
    }
    let a = CscMatrix::from_triplets(&a_ri, &a_ci, &a_vi, peq, nv).unwrap();

    // quadratic constraints -> linear rows in [x, w].
    let mut fixed_ri = Vec::new();
    let mut fixed_ci = Vec::new();
    let mut fixed_vi = Vec::new();
    let mut fixed_h = Vec::new();
    for (row, qc) in qp.quad.iter().enumerate() {
        for (j, &v) in qc.q.iter().enumerate() {
            if v != 0.0 {
                fixed_ri.push(row);
                fixed_ci.push(j);
                fixed_vi.push(v);
            }
        }
        for (pi, coeff) in quad_pair_coeffs(&qc.p, &pairs) {
            fixed_ri.push(row);
            fixed_ci.push(n + pi);
            fixed_vi.push(coeff);
        }
        fixed_h.push(-qc.r);
    }
    // original linear inequalities Gl x <= hl.
    let row_base = qp.quad.len();
    let gcp = qp.g_lin.col_ptr();
    let gri = qp.g_lin.row_ind();
    let gva = qp.g_lin.values();
    for j in 0..n {
        for k in gcp[j]..gcp[j + 1] {
            fixed_ri.push(row_base + gri[k]);
            fixed_ci.push(j);
            fixed_vi.push(gva[k]);
        }
    }
    fixed_h.extend_from_slice(&qp.h_lin);

    StaticRelax {
        n,
        pairs,
        c,
        a,
        b: qp.b_eq.clone(),
        fixed_ri,
        fixed_ci,
        fixed_vi,
        fixed_h,
        rule2_pair,
        mc_pair,
        mc_pairs,
        required: req.required,
    }
}

/// Build the McCormick LP relaxation for the current box `[lb, ub]`.
///
/// Only the box rows and the McCormick envelope rows depend on the box;
/// everything else is `static_` (see its doc), so this clones the
/// precomputed fixed part (`O(nnz)`) and appends the per-node rows directly
/// as triplets, instead of re-deriving the whole relaxation from a dense
/// `n x n` pass at every branch-and-bound node (`O(n * (n + pairs.len()))`
/// per node -- the dominant B&B memory cost on large QCQPs).
///
/// Emission is driven by the §3.1 classification stored in `static_`
/// (plan §3.2, decisions D2/D3):
/// - Box rows are emitted exactly for variables whose node bounds are
///   finite. A non-finite bound must never reach `h` (`LpProblem::new_general`
///   rejects it and the call site `.unwrap()`s), and no envelope row is
///   emitted for a pair with a non-finite endpoint (`w_interval` and the
///   McCormick coefficients `xl_i, xu_j` would carry `±inf`/`NaN`).
///   *Deviation from decision D3's first sentence, recorded here per "迷ったら
///   保守側"*: D3 as worded also drops box rows for *finite*-bounded
///   variables outside `R`. For a scheme-B LP relaxation that turns every
///   currently-solvable problem with a linear objective term in such a
///   variable into an unbounded relaxation (the variable escapes the box it
///   genuinely has in the original problem; the recession verifier cannot
///   certify `Unbounded` for a problem with a finite optimum, and without an
///   incumbent `x̃` plan line 110 forces `NotSupported` -- a §6 "退化なし"
///   regression on the bounded qplib suites). Dropping those box rows
///   requires the Phase 2 incumbent/`x̃` + objective-cutoff machinery to
///   restore boundedness, so finite-bounded variables keep their box rows
///   until then; the ±inf hazard D3/plan §3.2 actually guards is fully
///   excluded by the finite-bound emission rule above.
/// - Rule-2 diagonal pairs emit only the under-estimator cut `w_ii >= 0`
///   (the x̄ = 0 tangent; every further tangent is a node-time OA cut, see
///   `TangentCut` in `global_core`). Interval and envelope rows are skipped
///   unconditionally (D2).
/// - All other pairs with finite endpoints emit the interval rows + 4
///   McCormick facets exactly as the dense reference does.
fn build_relax(static_: &StaticRelax, lb: &[f64], ub: &[f64]) -> ConicProblem {
    let n = static_.n;
    let pairs = &static_.pairs;
    let npair = pairs.len();
    let nv = n + npair;
    let m_fixed = static_.fixed_h.len();
    let boxed: Vec<bool> = (0..n)
        .map(|j| lb[j].is_finite() && ub[j].is_finite())
        .collect();
    let box_rows = 2 * boxed.iter().filter(|b| **b).count();
    // 1 row per rule-2 diagonal, 2 single-entry + 4 push_mc rows per mc pair.
    let mc_rows: usize = (0..npair)
        .map(|pi| {
            if static_.rule2_pair[pi] {
                1
            } else if static_.mc_pair[pi] {
                6
            } else {
                0
            }
        })
        .sum();
    let m = m_fixed + box_rows + mc_rows;

    let mut ri = static_.fixed_ri.clone();
    let mut ci = static_.fixed_ci.clone();
    let mut vi = static_.fixed_vi.clone();
    let mut h = static_.fixed_h.clone();
    // Capacity bound: an mc pair takes 2 + 4*3 = 14 triplets, a rule-2 pair
    // 1; 15 per pair is the uniform safe bound (reserve only, never asserted).
    let mc_nnz = 15 * npair;
    ri.reserve(box_rows + mc_nnz);
    ci.reserve(box_rows + mc_nnz);
    vi.reserve(box_rows + mc_nnz);
    h.reserve(box_rows + mc_rows);

    // variable box (finite-bounded variables only, see fn doc).
    let mut row = m_fixed;
    for j in 0..n {
        if !boxed[j] {
            continue;
        }
        ri.push(row);
        ci.push(j);
        vi.push(1.0);
        h.push(ub[j]);
        row += 1;
        ri.push(row);
        ci.push(j);
        vi.push(-1.0);
        h.push(-lb[j]);
        row += 1;
    }

    // McCormick envelopes for each pair, plus the interval of `w = x_i x_j`
    // over the box as explicit rows. For off-diagonal pairs the four envelope
    // rows are the convex hull of the bilinear graph, so the interval is
    // implied; for diagonal pairs (`w = x_k^2`) the two endpoint tangents do
    // NOT imply the interval lower bound (e.g. on `[-1, 1]` they only give
    // `w >= 2|x| - 1`, which dips to `-1` at `x = 0` instead of `0`), so the
    // relaxation of `x^2` terms is needlessly loose without it. Rule-2
    // diagonals instead carry only `w_ii >= 0` (D2): containment holds for
    // `w_ii >= x_i^2` regardless of sign (plan §3.1 note), the lift
    // coefficient of `w_ii` is `>= 0` everywhere a rule-2 diagonal appears
    // (that *is* rule 2), so nothing in the relaxation pushes `w_ii` up and
    // the constant tangent is the whole stationary envelope; OA cuts added at
    // relaxation solutions tighten the under-estimator further.
    for (pi, &(i, j)) in pairs.iter().enumerate() {
        let wj = n + pi;
        if static_.rule2_pair[pi] {
            // -w_ii <= 0  <=>  w_ii >= x_i^2's x̄=0 tangent (w >= 2·0·x - 0²).
            ri.push(row);
            ci.push(wj);
            vi.push(-1.0);
            h.push(0.0);
            row += 1;
            continue;
        }
        if !static_.mc_pair[pi] {
            continue;
        }
        let (xl_i, xu_i) = (lb[i], ub[i]);
        let (xl_j, xu_j) = (lb[j], ub[j]);
        let (w_lo, w_hi) = w_interval(xl_i, xu_i, xl_j, xu_j, i == j);

        ri.push(row);
        ci.push(wj);
        vi.push(1.0);
        h.push(w_hi);
        row += 1;
        ri.push(row);
        ci.push(wj);
        vi.push(-1.0);
        h.push(-w_lo);
        row += 1;

        // w >= xl_i x_j + xl_j x_i - xl_i xl_j  =>  -w + xl_j x_i + xl_i x_j <= xl_i xl_j
        push_mc(
            &mut ri,
            &mut ci,
            &mut vi,
            &mut h,
            &mut row,
            wj,
            i,
            j,
            -1.0,
            xl_j,
            xl_i,
            xl_i * xl_j,
        );
        // w >= xu_i x_j + xu_j x_i - xu_i xu_j
        push_mc(
            &mut ri,
            &mut ci,
            &mut vi,
            &mut h,
            &mut row,
            wj,
            i,
            j,
            -1.0,
            xu_j,
            xu_i,
            xu_i * xu_j,
        );
        // w <= xl_i x_j + xu_j x_i - xl_i xu_j  =>  w - xu_j x_i - xl_i x_j <= -xl_i xu_j
        push_mc(
            &mut ri,
            &mut ci,
            &mut vi,
            &mut h,
            &mut row,
            wj,
            i,
            j,
            1.0,
            -xu_j,
            -xl_i,
            -xl_i * xu_j,
        );
        // w <= xu_i x_j + xl_j x_i - xu_i xl_j
        push_mc(
            &mut ri,
            &mut ci,
            &mut vi,
            &mut h,
            &mut row,
            wj,
            i,
            j,
            1.0,
            -xl_j,
            -xu_i,
            -xu_i * xl_j,
        );
    }
    assert_eq!(row, m, "relaxation row-count invariant");

    let g = CscMatrix::from_triplets(&ri, &ci, &vi, m, nv).unwrap();
    ConicProblem {
        c: static_.c.clone(),
        a: static_.a.clone(),
        b: static_.b.clone(),
        g,
        h,
        cone: ConeSpec { l: m, soc: vec![] },
    }
}

/// Range of `w = x_i x_j` over the box `[xl_i, xu_i] x [xl_j, xu_j]`
/// (`diag`: the pair is `x_k^2`, whose range must include 0 when the box
/// straddles the origin and is bounded below by the smaller squared endpoint
/// otherwise — never by a corner product alone).
fn w_interval(xl_i: f64, xu_i: f64, xl_j: f64, xu_j: f64, diag: bool) -> (f64, f64) {
    if diag {
        let (a, b) = (xl_i * xl_i, xu_i * xu_i);
        let lo = if xl_i <= 0.0 && xu_i >= 0.0 {
            0.0
        } else {
            a.min(b)
        };
        (lo, a.max(b))
    } else {
        let corners = [xl_i * xl_j, xl_i * xu_j, xu_i * xl_j, xu_i * xu_j];
        let lo = corners.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = corners.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (lo, hi)
    }
}

/// Push one McCormick facet row as triplets (`row` is the running row
/// counter, incremented on return). Duplicate `(row, col)` triplets are
/// summed by `CscMatrix::from_triplets`, so pushing `xi_coef` and `xj_coef`
/// as two separate triplets at the same column when `i == j` reproduces the
/// dense reference's `r[i] += xi_coef; r[i] += xj_coef;` exactly.
#[allow(clippy::too_many_arguments)]
fn push_mc(
    ri: &mut Vec<usize>,
    ci: &mut Vec<usize>,
    vi: &mut Vec<f64>,
    h: &mut Vec<f64>,
    row: &mut usize,
    wj: usize,
    i: usize,
    j: usize,
    wcoef: f64,
    xi_coef: f64,
    xj_coef: f64,
    rhs: f64,
) {
    ri.push(*row);
    ci.push(wj);
    vi.push(wcoef);
    ri.push(*row);
    ci.push(i);
    vi.push(xi_coef);
    if i != j {
        ri.push(*row);
        ci.push(j);
        vi.push(xj_coef);
    } else {
        // For i==j both x-coefficients apply to the same variable.
        ri.push(*row);
        ci.push(i);
        vi.push(xj_coef);
    }
    h.push(rhs);
    *row += 1;
}

/// Feasibility check against the caller's literal `NonconvexQcqp` data.
/// `g_lin`/`a_eq`'s dot products with `x` are computed via `mat_vec_mul`
/// (`O(nnz)`) rather than by densifying either matrix first.
///
/// All residual comparisons use `!(v <= tol)`: overflowed activity turns
/// `inf/inf`-style dot products into NaN, and `v > tol` would silently
/// accept such an unverifiable point (conic/misocp.rs::conic_feasible is
/// the same-shaped gate).
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn feasible(qp: &NonconvexQcqp, x: &[f64], tol: f64) -> bool {
    for (k, &lbv) in qp.lb.iter().enumerate() {
        if !(lbv - tol <= x[k] && x[k] <= qp.ub[k] + tol) {
            return false;
        }
    }
    for qc in &qp.quad {
        let v = quad_val(&qc.p, x) + qc.q.iter().zip(x).map(|(a, b)| a * b).sum::<f64>() + qc.r;
        if !(v <= tol) {
            return false;
        }
    }
    let gv = qp.g_lin.mat_vec_mul(x).expect(
        "g_lin.ncols() == x.len() == n: guaranteed by NonconvexQcqp::validate() \
         at global_core entry (nonconvex.rs); x is always qp.n long",
    );
    for (i, &v) in gv.iter().enumerate() {
        if !(v <= qp.h_lin[i] + tol) {
            return false;
        }
    }
    let av = qp.a_eq.mat_vec_mul(x).expect(
        "a_eq.ncols() == x.len() == n: guaranteed by NonconvexQcqp::validate() \
         at global_core entry (nonconvex.rs); x is always qp.n long",
    );
    for (i, &v) in av.iter().enumerate() {
        if !((v - qp.b_eq[i]).abs() <= tol) {
            return false;
        }
    }
    true
}

fn objective(qp: &NonconvexQcqp, x: &[f64]) -> f64 {
    let mut o = qp.q0.iter().zip(x).map(|(a, b)| a * b).sum::<f64>();
    if let Some(p0) = &qp.p0 {
        o += quad_val(p0, x);
    }
    o
}

/// Globally optimise a nonconvex QCQP by spatial branch-and-bound.
pub fn solve_global_qcqp(
    qp: &NonconvexQcqp,
    opts: &ConicOptions,
    g: &GlobalOptions,
) -> GlobalResult {
    global_core(qp, &[], opts, g, None)
}

/// Globally optimise a nonconvex mixed-integer QCQP. `integers` lists variable
/// indices constrained to integer values; their bounds come from `qp.lb`/`qp.ub`.
/// Combines integer branching with spatial (McCormick) branching.
pub fn solve_global_miqcp(
    qp: &NonconvexQcqp,
    integers: &[usize],
    opts: &ConicOptions,
    g: &GlobalOptions,
) -> GlobalResult {
    global_core(qp, integers, opts, g, None)
}

fn frac_dist(v: f64) -> f64 {
    (v - v.round()).abs()
}

fn all_integral(x: &[f64], integers: &[usize], tol: f64) -> bool {
    integers.iter().all(|&k| frac_dist(x[k]) <= tol)
}

/// Absolute floating-point slack on the exhausted-search gap certificate:
/// nodes are bound-pruned at `lower >= incumbent - gap_tol`, so an exhausted
/// clean search proves `gap <= gap_tol` exactly in real arithmetic; the slack
/// only absorbs rounding in those comparisons (scaled by `1 + |incumbent|`).
const GAP_CERT_SLACK: f64 = 1e-12;

/// Accepts `x` as a new incumbent if, after rounding its integer components
/// to `xr`, it is still feasible at the caller's own `tol` -- never a looser
/// hardcoded floor. Rounding moves an integer coordinate by up to `int_tol`,
/// which can push a tight/equality row's residual up by the same amount; a
/// caller requesting a tighter `tol` than that floor (e.g. IPM `eps = 1e-8`)
/// must have the rounded point re-checked at `tol`, not silently waved
/// through at `1e-6` (PR #25 review, nonconvex.rs).
fn accept(
    qp: &NonconvexQcqp,
    integers: &[usize],
    int_tol: f64,
    x: &[f64],
    incumbent: &mut f64,
    inc_x: &mut Vec<f64>,
    tol: f64,
) {
    if feasible(qp, x, tol) && all_integral(x, integers, int_tol) {
        // Round integers for a clean reported solution.
        let mut xr = x.to_vec();
        for &k in integers {
            xr[k] = xr[k].round();
        }
        if feasible(qp, &xr, tol) {
            let ov = objective(qp, &xr);
            if ov < *incumbent {
                *incumbent = ov;
                *inc_x = xr;
            }
        }
    }
}

/// Reject an out-of-range `integers` index before the search tree is built.
///
/// `solve_global_miqcp` takes `integers` as a bare `&[usize]`, not a field on
/// a validated struct, so nothing else ever checks it against `qp.n`: the
/// first root-node relaxation indexes `x[k]` for every `k` in `integers`
/// (`global_core`'s per-node fractional-branching scan), so `k >= qp.n`
/// panics on the very first node rather than failing as an ordinary
/// invalid-input error (Codex review R3, nonconvex.rs:763).
fn validate_integers(n: usize, integers: &[usize]) -> Result<(), String> {
    for (k, &j) in integers.iter().enumerate() {
        if j >= n {
            return Err(format!("integers[{k}] = {j} out of range (n = {n})"));
        }
    }
    Ok(())
}

/// Seed for `global_core`: an initial incumbent and the root box to search.
///
/// Plan §3.2 line 112 (decision D7): the search starts at `incumbent =
/// +inf`, so an upper bound produced by a bound-free local solver (Phase 2)
/// would never reach prune or gap reporting. Phase 2 also derives bounds that
/// it deliberately keeps out of `qp.lb`/`qp.ub` (plan §4.2 line 136: rewriting
/// them would reject incumbents inside `feas_tol`), so the derived root box
/// travels with the seed. This milestone only wires the plumbing; the public
/// entry points pass `None` (current behaviour), and the CHANGELOG entry for
/// the public signature change ships with the Phase 2 / M5 wiring.
#[derive(Debug, Clone)]
struct GlobalSeed {
    /// Feasible point `x̃` (re-validated by `global_core` at `feas_tol`
    /// before use -- a corrupt seed fails as `NotSupported`, never as a
    /// trusted incumbent).
    x: Vec<f64>,
    /// Objective value at `x`; cross-checked against `objective(qp, x)` and
    /// the incumbent is always the recomputed value (a stale/mismatched `f`
    /// fails the seed, it is never trusted as a lower upper-bound).
    f: f64,
    root_lb: Vec<f64>,
    root_ub: Vec<f64>,
}

/// Zero-decision tolerance for the §3.2 recession verification, after the ray
/// direction's `x` block is normalised to `‖d‖_inf = 1`: residuals are
/// compared against `1e-9 * (1 + ‖data‖_inf)`, the same relative backward-
/// error scale the simplex ray gate uses when it certifies `RayCertificate`
/// (`simplex::unbounded_ray`), so the nonconvex-side check is neither weaker
/// nor stricter than the certificate it consumes.
const RAY_RECESSION_ZERO_TOL: f64 = 1e-9;

/// Descent threshold on `q0^T d` for the §3.2 objective condition (plan line
/// 107, `q0^T d < 0`): with `‖d‖_inf = 1` a directional slope below
/// `1e-9` is indistinguishable from zero at the certificate tolerance, so
/// only strictly-negatives beyond this slack certify unbounded descent.
const RAY_DESCENT_TOL: f64 = 1e-9;

/// `‖M‖_inf` (max absolute row sum) over a CSC matrix, `O(nnz)`.
fn max_abs_row_sum(m: &CscMatrix) -> f64 {
    let cp = m.col_ptr();
    let ri = m.row_ind();
    let va = m.values();
    let mut rows = vec![0.0f64; m.nrows().max(1)];
    for j in 0..m.ncols() {
        for k in cp[j]..cp[j + 1] {
            rows[ri[k]] += va[k].abs();
        }
    }
    rows.into_iter().fold(0.0f64, f64::max)
}

/// Verify that a relaxation recession ray certifies *original-problem*
/// unboundedness (plan §3.2 lines 100–111, decision D6). `ray` is the
/// simplex-verified certificate in relaxation coordinates `[x, w]`; only its
/// `x` block `d` matters, normalised to `‖d‖_inf = 1`. All coefficients are
/// the **original** rows (plan line 103: a row convex in the relaxation is
/// still verified, never waived).
///
/// Checks (each `Err`-reasoned; never a panic):
/// - `d_R = 0` (plan line 108: structural from the box rows, asserted here);
/// - integer variables `d_k = 0` (a ray direction must live in the integer
///   lattice; `0` is the only recession direction it admits);
/// - bounds: finite `lb_j` forces `d_j >= 0`, finite `ub_j` forces `d_j <= 0`;
/// - linear rows `G d <= 0`, `A_eq d = 0`;
/// - every quadratic row (PSD or not, `Eq` already split to `±P` upstream so
///   both sides give `q^T d <= 0` and hence `q^T d = 0` for free, plan 109):
///   `P_k d = 0` and `q_k^T d <= 0`. Soundness: `P_k d = 0` makes
///   `f_k(x̃ + t d) = f_k(x̃) + t·q_k^T d` exactly (the `t²` term
///   `d^T P_k d = d^T(P_k d)` and the cross term `(P_k d)^T x̃` both vanish),
///   so feasibility degrades at most linearly and stays `<= 0`;
/// - objective `P0 d = 0` and `q0^T d <= -RAY_DESCENT_TOL`, so
///   `f0(x̃ + t d) -> -inf`.
///
/// Interpretation record: plan line 105 ("規則 2 の変数: `d_i = 0`") is a
/// device of the line-108 support argument to kill the `d^T P_k d` terms.
/// Applied literally to *vacuously* rule-2 variables (variables with no
/// quadratic support at all are vacuously rule 2 per §3.1), it would force
/// `d = 0` on every free variable and make the plan-line-111 `Unbounded`
/// return unreachable even for `min -x s.t. x <= 1`-type rays -- the literal
/// reading contradicts the plan's own return contract. The load-bearing
/// content (`d^T P_k d = 0` on every row) is enforced *directly* by the `P_k
/// d = 0` checks above, which subsume line 105 for every variable with
/// quadratic support (the only ones line 108's argument speaks about) and
/// change nothing else. Conservative side unchanged: anything not certified
/// stays `NotSupported`.
fn verify_recession_unboundedness(
    qp: &NonconvexQcqp,
    integers: &[usize],
    static_: &StaticRelax,
    ray: &crate::problem::RayCertificate,
) -> Result<(), String> {
    let n = qp.n;
    let nv = n + static_.pairs.len();
    if ray.direction.len() != nv || ray.base_point.len() != nv {
        return Err(format!(
            "ray certificate dimension {} / {} != relaxation dimension {nv}",
            ray.direction.len(),
            ray.base_point.len()
        ));
    }
    let norm = ray.direction[..n]
        .iter()
        .fold(0.0f64, |a: f64, &v| a.max(v.abs()));
    if norm == 0.0 || norm.is_nan() {
        return Err(
            "ray direction has a zero x-block (only lifted w coordinates drift): \
             no recession direction of the original problem to verify"
                .to_string(),
        );
    }
    let d: Vec<f64> = ray.direction[..n].iter().map(|&v| v / norm).collect();
    let zero = |scale: f64| RAY_RECESSION_ZERO_TOL * (1.0 + scale);

    for (j, &dj) in d.iter().enumerate() {
        if static_.required[j] && dj.abs() > zero(0.0) {
            return Err(format!(
                "ray moves bound-required variable {j} (d[{j}] = {dj}): the box rows \
                 should have forced d_R = 0 (plan §3.2 line 108)"
            ));
        }
        if qp.lb[j].is_finite() && dj < -zero(0.0) {
            return Err(format!(
                "ray decreases bounded variable {j} (d[{j}] = {dj} < 0 with lb = {})",
                qp.lb[j]
            ));
        }
        if qp.ub[j].is_finite() && dj > zero(0.0) {
            return Err(format!(
                "ray increases bounded variable {j} (d[{j}] = {dj} > 0 with ub = {})",
                qp.ub[j]
            ));
        }
    }
    for &k in integers {
        if d[k].abs() > zero(0.0) {
            return Err(format!(
                "ray moves integer variable {k} (d[{k}] = {}): the integer lattice \
                 admits no nonzero recession direction on it",
                d[k]
            ));
        }
    }

    let check_le = |name: &str, vals: &[f64], scale: f64| -> Result<(), String> {
        for (i, &v) in vals.iter().enumerate() {
            if v > zero(scale) {
                return Err(format!("{name} row {i} grows along the ray ({v} > 0)"));
            }
        }
        Ok(())
    };
    let check_eq0 = |name: &str, vals: &[f64], scale: f64| -> Result<(), String> {
        for (i, &v) in vals.iter().enumerate() {
            if v.abs() > zero(scale) {
                return Err(format!("{name} row {i} is not flat along the ray ({v})"));
            }
        }
        Ok(())
    };

    check_le(
        "g_lin",
        &qp.g_lin
            .mat_vec_mul(&d)
            .expect("g_lin.ncols() == n: NonconvexQcqp::validate ran at global_core entry"),
        max_abs_row_sum(&qp.g_lin),
    )?;
    check_eq0(
        "a_eq",
        &qp.a_eq
            .mat_vec_mul(&d)
            .expect("a_eq.ncols() == n: NonconvexQcqp::validate ran at global_core entry"),
        max_abs_row_sum(&qp.a_eq),
    )?;

    for (k, qc) in qp.quad.iter().enumerate() {
        let pd =
            qc.p.mat_vec_mul(&d)
                .expect("quad[k].p is (n, n): NonconvexQcqp::validate ran at global_core entry");
        let scale = max_abs_row_sum(&qc.p);
        check_eq0(&format!("quad[{k}].p d"), &pd, scale)?;
        let qd: f64 = qc.q.iter().zip(&d).map(|(a, b)| a * b).sum();
        let qscale = qc.q.iter().fold(0.0f64, |a: f64, &v| a.max(v.abs()));
        if qd > zero(qscale) {
            return Err(format!(
                "quad[{k}] grows along the ray (q^T d = {qd} > 0): the row is not a \
                 recession inequality in this direction"
            ));
        }
    }
    if let Some(p0) = &qp.p0 {
        let pd = p0
            .mat_vec_mul(&d)
            .expect("p0 is (n, n): validated at global_core entry");
        check_eq0("p0 d", &pd, max_abs_row_sum(p0))?;
    }
    let q0d: f64 = qp.q0.iter().zip(&d).map(|(a, b)| a * b).sum();
    if q0d > -RAY_DESCENT_TOL {
        return Err(format!(
            "objective does not descend along the ray (q0^T d = {q0d}, need < -{RAY_DESCENT_TOL})"
        ));
    }
    Ok(())
}

/// McCormick spatial branch-and-bound core shared by [`solve_global_qcqp`] /
/// [`solve_global_miqcp`].
///
/// Certificate-acceptance contract (横展開, mirrors
/// `misocp::solve_misocp`): both the no-incumbent `Infeasible` classification
/// and the incumbent `certified` -> `Optimal` classification defer to
/// `Timeout` when an external stop is observed at that instant, even if the
/// underlying region prunes were themselves un-raced. Unlike the QP IPPMM /
/// conic IPM per-certificate checks (where `Optimal` is independently
/// KKT/residual-verified and therefore exempt), a B&B tree's exhaustive-
/// search `Infeasible` and `Optimal` both rest on the identical deterministic
/// "search space fully pruned" signal -- there is no basis to trust one and
/// distrust the other, so both defer symmetrically.
fn global_core(
    qp: &NonconvexQcqp,
    integers: &[usize],
    opts: &ConicOptions,
    g: &GlobalOptions,
    seed: Option<&GlobalSeed>,
) -> GlobalResult {
    if let Err(e) = qp
        .validate()
        .and_then(|()| opts.validate())
        .and_then(|()| g.validate())
        .and_then(|()| validate_integers(qp.n, integers))
    {
        return GlobalResult {
            status: SolveStatus::NotSupported(e),
            objective: f64::NAN,
            x: vec![],
            nodes: 0,
            gap: f64::INFINITY,
        };
    }
    let static_ = build_static(qp);
    let mut incumbent = f64::INFINITY;
    let mut inc_x: Vec<f64> = Vec::new();
    // Root box: the seed's derived box when present (plan §4.2 line 136
    // keeps derived bounds out of `qp.lb`/`qp.ub`), else the problem box.
    let mut root_box = (qp.lb.clone(), qp.ub.clone());
    if let Some(s) = seed {
        let bad = if s.x.len() != qp.n || s.root_lb.len() != qp.n || s.root_ub.len() != qp.n {
            Some("seed x/root_lb/root_ub length != n".to_string())
        } else if s
            .root_lb
            .iter()
            .zip(&s.root_ub)
            .any(|(l, u)| l.is_nan() || u.is_nan() || l > u)
        {
            // Non-finite (`±inf`) box sides are allowed -- they are what the
            // declared box itself carries for free variables; NaN is not.
            Some("seed root box must satisfy lb <= ub (no NaN sides)".to_string())
        } else if s
            .root_lb
            .iter()
            .zip(&s.root_ub)
            .zip(qp.lb.iter().zip(&qp.ub))
            .any(|((&l, &u), (&pl, &pu))| l < pl || u > pu)
        {
            // The derived box must refine, never widen, the declared box (a
            // widened side would admit points the public `feasible()`
            // contract rejects).
            Some("seed root box widens the declared box".to_string())
        } else if !feasible(qp, &s.x, g.feas_tol) || !all_integral(&s.x, integers, g.int_tol) {
            Some("seed x is not feasible for the original problem at feas_tol".to_string())
        } else {
            let fx = objective(qp, &s.x);
            ((s.f - fx).abs() > 1e-9 * (1.0 + fx.abs())).then(|| {
                format!(
                    "seed objective mismatch: seed.f = {} vs recomputed {fx}",
                    s.f
                )
            })
        };
        if let Some(e) = bad {
            return GlobalResult {
                status: SolveStatus::NotSupported(format!("invalid global seed: {e}")),
                objective: f64::NAN,
                x: vec![],
                nodes: 0,
                gap: f64::INFINITY,
            };
        }
        incumbent = objective(qp, &s.x);
        inc_x = s.x.clone();
        root_box = (s.root_lb.clone(), s.root_ub.clone());
    }
    let mut nodes = 0usize;
    let mut limited = false;
    let mut timed_out = false;
    let mut numerical_failures = 0usize;
    // Proven global lower bound: min over resolved terminal regions of their
    // relaxation bounds (`+inf` = every region proven empty so far). Regions
    // left unresolved (break, failed relaxation) contribute their inherited
    // parent bound instead. Stack entries carry that inherited bound
    // (`-inf` at the root).
    let mut lower_bound = f64::INFINITY;

    let mut stack = vec![(root_box.0, root_box.1, f64::NEG_INFINITY)];

    while let Some((lb, ub, inherited)) = stack.pop() {
        // `stop_requested` covers both the wall-clock deadline and the
        // shared cancel flag (`ConicOptions::cancel_flag`); a deadline-only
        // check here left a cancel fired mid-search unhonored until the
        // search happened to exhaust itself or hit `max_nodes` (PR #25
        // review INLINE-D) -- the pre-launch guard in `qcqp_route.rs` only
        // covers a flag already set before this loop starts, not one set
        // while it runs.
        if opts.stop_requested() {
            timed_out = true;
            lower_bound = lower_bound.min(inherited);
            break;
        }
        if nodes >= g.max_nodes {
            limited = true;
            lower_bound = lower_bound.min(inherited);
            break;
        }
        nodes += 1;
        let relax = build_relax(&static_, &lb, &ub);
        let res = solve_relax_lp(&relax, opts.deadline, opts.cancel_flag.as_ref());
        match res.status {
            SolveStatus::Optimal => {}
            // Empty relaxation => the region holds no feasible point: a
            // certified fathom (no bound contribution; the region's "bound"
            // is +inf).
            SolveStatus::Infeasible => continue,
            SolveStatus::Timeout => {
                timed_out = true;
                lower_bound = lower_bound.min(inherited);
                break;
            }
            // Plan §3.2 lines 98-111 (decision D6): an unbounded relaxation
            // cannot be fixed by branching (the recession direction lives on
            // free variables and survives into every child), so branch-free
            // disposition only: consume the simplex-verified ray against the
            // ORIGINAL problem data, and require a base point `x̃` that the
            // original `feasible()` contract accepts -- the relaxation
            // solution's x-projection is a McCormick point, not a substitute
            // (plan line 110). Verified + base present => genuinely
            // Unbounded (return, objective -inf); anything else =>
            // NotSupported with the reason. Never `numerical_failures`: an
            // unbounded LP proved something ("the relaxation is
            // unbounded"), it is not a numerical artefact.
            SolveStatus::Unbounded => {
                let verdict = match (&res.ray, inc_x.is_empty()) {
                    (None, _) => Err("relaxation LP reported Unbounded without a verified ray \
                         certificate (unproven/presolve-reduced verdict): cannot \
                         certify original-problem unboundedness"
                        .to_string()),
                    (Some(_), true) => Err("relaxation unbounded with a verified ray, but no \
                         original-problem feasible base point x̃ is at hand (the \
                         relaxation solution's x-projection is a McCormick point, \
                         plan §3.2 line 110): unboundedness stays unproven"
                        .to_string()),
                    (Some(ray), false) => {
                        verify_recession_unboundedness(qp, integers, &static_, ray)
                    }
                };
                return match verdict {
                    Ok(()) => GlobalResult {
                        status: SolveStatus::Unbounded,
                        objective: f64::NEG_INFINITY,
                        x: inc_x,
                        nodes,
                        gap: f64::INFINITY,
                    },
                    Err(reason) => GlobalResult {
                        status: SolveStatus::NotSupported(reason),
                        objective: incumbent,
                        x: inc_x,
                        nodes,
                        gap: f64::INFINITY,
                    },
                };
            }
            // NumericalError, MaxIterations, SuboptimalSolution, …: nothing
            // was proven about the region. Prune it — there is no trustworthy
            // point to branch on — but record the failure so the search never
            // claims a certificate, and keep the region's inherited bound in
            // the global bound. (`Unbounded` is handled by its own arm above
            // since G2b: with free variables the lifted polytope is no longer
            // a bounded box, so unboundedness is a real verdict to consume,
            // not a numerical artefact to count.)
            _ => {
                numerical_failures += 1;
                lower_bound = lower_bound.min(inherited);
                continue;
            }
        }
        // A child region is contained in its parent, so its true minimum is
        // never below the parent's proven bound; `max` keeps the inherited
        // bound when the child LP value dips below it numerically.
        let lower = res.objective.max(inherited);
        if lower >= incumbent - g.gap_tol {
            lower_bound = lower_bound.min(lower);
            continue; // bound prune (certified: region cannot beat incumbent by > gap_tol)
        }
        let x = res.x[..qp.n].to_vec();

        accept(
            qp,
            integers,
            g.int_tol,
            &x,
            &mut incumbent,
            &mut inc_x,
            g.feas_tol,
        );

        // (1) integer branching first: pick the most fractional integer var.
        let mut int_branch = None;
        let mut worst_if = g.int_tol;
        for &k in integers {
            let d = frac_dist(x[k]);
            if d > worst_if {
                worst_if = d;
                int_branch = Some(k);
            }
        }
        if let Some(k) = int_branch {
            let fl = x[k].floor();
            let ce = x[k].ceil();
            // Children cover every integer point of the region, so the
            // region's bound lives on in their inherited bounds; a skipped
            // side holds no integer point (certified empty).
            // down: ub_k = floor
            if fl >= lb[k] - 1e-9 {
                let lb_d = lb.clone();
                let mut ub_d = ub.clone();
                ub_d[k] = fl;
                stack.push((lb_d, ub_d, lower));
            }
            // up: lb_k = ceil
            if ce <= ub[k] + 1e-9 {
                let mut lb_u = lb.clone();
                let ub_u = ub.clone();
                lb_u[k] = ce;
                stack.push((lb_u, ub_u, lower));
            }
            continue;
        }

        // (2) Spatial branching on the worst McCormick term gap.
        //
        // Candidates are exactly `static_.mc_pairs` (decision D4): pairs with
        // emitted interval + 4-facet envelopes. Rule-2 diagonals are excluded
        // -- their relaxation `w_ii` sits at the under-estimator by design
        // (the x̄ = 0 tangent plus OA cuts), so counting `|w_ii − x_i²|` as a
        // "gap" would branch forever, and on a free `x_i` the fallback split
        // `0.5·(lb+ub)` returns `∓inf` and re-pushes the identical box
        // (plan §0.1 line 16). Plan §3.2 line 97's "only variables in R":
        // every `mc_pair` endpoint is in `R` or is a D1-exempt rule-1 support
        // variable (finite raw bounds by the D1 filter), and rule-2-vetoed
        // diagonals are in `R`; all candidates therefore live in finite boxes,
        // which is the load-bearing property pinned by the sentinel test.
        //
        // Decision D5 (recorded): no `w_ii := x_i²` projection is applied to
        // the leaf gap or incumbent test. The incumbent machinery is
        // original-problem based throughout (`feasible(qp, x, ·)` and
        // `objective(qp, x)` read `x` only), and rule-2 pairs are excluded
        // from the gap by D4, so a projection would have no consumer.
        let mut worst_gap = 0.0;
        let mut branch_var = None;
        for &pi in &static_.mc_pairs {
            let &(i, j) = &static_.pairs[pi];
            let wv = res.x[qp.n + pi];
            let gap = (wv - x[i] * x[j]).abs();
            if gap > worst_gap {
                worst_gap = gap;
                branch_var = Some(if (ub[i] - lb[i]) >= (ub[j] - lb[j]) {
                    i
                } else {
                    j
                });
            }
        }

        if worst_gap <= g.bilinear_tol {
            // Leaf: the relaxation is tight, so `lower` is (numerically) the
            // region's true minimum and stands as its terminal bound. Uses
            // the caller's own `feas_tol`, unchanged -- same as the per-node
            // `accept` above; an unexplained `* 100.0` here previously let a
            // leaf point violating a constraint by up to 100x the requested
            // tolerance be reported as feasible (PR #25 review INLINE-Q).
            accept(
                qp,
                integers,
                g.int_tol,
                &x,
                &mut incumbent,
                &mut inc_x,
                g.feas_tol,
            );
            lower_bound = lower_bound.min(lower);
            continue;
        }

        let mut pushed = false;
        if let Some(k) = branch_var {
            let mid = x[k].clamp(lb[k], ub[k]);
            let split = if (mid - lb[k]).abs() < 1e-9 || (ub[k] - mid).abs() < 1e-9 {
                0.5 * (lb[k] + ub[k])
            } else {
                mid
            };
            let lb1 = lb.clone();
            let mut ub1 = ub.clone();
            ub1[k] = split;
            let mut lb2 = lb.clone();
            let ub2 = ub.clone();
            lb2[k] = split;
            if ub1[k] - lb1[k] > 1e-9 {
                stack.push((lb1, ub1, lower));
                pushed = true;
            }
            if ub2[k] - lb2[k] > 1e-9 {
                stack.push((lb2, ub2, lower));
                pushed = true;
            }
        }
        if !pushed {
            // Box too thin to split further: terminal with the (possibly
            // loose) relaxation bound.
            lower_bound = lower_bound.min(lower);
        }
    }

    // Regions still open when the search stopped keep their inherited bounds.
    for (_, _, inherited) in &stack {
        lower_bound = lower_bound.min(*inherited);
    }

    #[cfg(test)]
    test_maybe_cancel_before_final_classification(opts);

    if inc_x.is_empty() {
        // No incumbent: `Infeasible` is a certificate, so it requires a fully
        // exhausted search in which every region was proven empty. A
        // timed-out / node-limited search proved nothing; neither did one
        // with failed relaxations or with uncertified terminal regions
        // (finite `lower_bound` = some region had a valid relaxation but no
        // feasible point was recovered from it).
        let status = if timed_out {
            SolveStatus::Timeout
        } else if limited {
            SolveStatus::MaxIterations
        } else if numerical_failures > 0 || lower_bound < f64::INFINITY {
            SolveStatus::NumericalError
        } else if opts.stop_requested() {
            // 証明受理直前の stop check: 関数 doc の contract 参照
            SolveStatus::Timeout
        } else {
            SolveStatus::Infeasible
        };
        return GlobalResult {
            status,
            objective: f64::INFINITY,
            x: vec![],
            nodes,
            gap: f64::INFINITY,
        };
    }
    let gap = (incumbent - lower_bound).max(0.0);
    let certified = !timed_out
        && !limited
        && numerical_failures == 0
        && gap <= g.gap_tol + GAP_CERT_SLACK * (1.0 + incumbent.abs());
    GlobalResult {
        // With an incumbent, a node/gap-limited search is `SuboptimalSolution`
        // (the incumbent is valid but unproven), matching `misocp::solve_misocp`
        // and `mip::solve_miqp`. `MaxIterations` is reserved for the
        // no-incumbent node-limited case above. `certified` shares the same
        // stop-check backstop as the no-incumbent `Infeasible` branch above
        // (contract: 関数 doc 参照, P2-A).
        status: if timed_out || (certified && opts.stop_requested()) {
            SolveStatus::Timeout
        } else if certified {
            SolveStatus::Optimal
        } else {
            SolveStatus::SuboptimalSolution
        },
        objective: incumbent,
        x: inc_x,
        nodes,
        gap,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// Queue `plan` as the per-node relaxation-status fault plan, run `f`, then
    /// clear it (so a panic inside `f` cannot leak state into other tests).
    fn with_relax_plan<R>(plan: Vec<Option<SolveStatus>>, f: impl FnOnce() -> R) -> R {
        RELAX_STATUS_PLAN.with(|p| *p.borrow_mut() = VecDeque::from(plan));
        let out = f();
        RELAX_STATUS_PLAN.with(|p| p.borrow_mut().clear());
        out
    }

    /// PR #25 review (nonconvex.rs, "Honor feasibility tolerance after
    /// integer rounding"): `accept`'s post-rounding feasibility recheck must
    /// use the caller's own `tol`, not a hardcoded `1e-6` floor.
    ///
    /// `x0` is an "integer" variable pinned by an equality to `5 + 3e-7`: it
    /// is within `int_tol = 1e-6` of the integer 5 (so the pre-round
    /// `all_integral` gate passes), but rounding to `5.0` leaves an equality
    /// residual of `3e-7` -- inside a hardcoded `1e-6` floor but outside a
    /// caller-requested `tol = 1e-8`. The independent oracle here is exact
    /// arithmetic on the single equality row: no integer satisfies
    /// `x0 = 5 + 3e-7` to `1e-8`, so `xr = [5.0]` must be rejected and the
    /// incumbent must stay untouched.
    ///
    /// Sentinel: restoring the old `tol.max(1e-6)` floor inside `accept`
    /// makes this FAIL (incumbent becomes `5.0`, `inc_x = [5.0]`).
    #[test]
    fn accept_honors_caller_tolerance_after_rounding_not_a_fixed_floor() {
        let n = 1usize;
        let qp = NonconvexQcqp {
            n,
            p0: None,
            q0: vec![1.0],
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap(),
            b_eq: vec![5.0 + 3e-7],
            lb: vec![0.0],
            ub: vec![10.0],
        };
        let x = vec![5.0 + 3e-7];
        let mut incumbent = f64::INFINITY;
        let mut inc_x: Vec<f64> = Vec::new();
        accept(&qp, &[0], 1e-6, &x, &mut incumbent, &mut inc_x, 1e-8);
        assert!(
            incumbent.is_infinite() && inc_x.is_empty(),
            "rounding x0 to 5.0 leaves a 3e-7 equality residual, which must \
             fail the caller's tol=1e-8 rather than a hardcoded 1e-6 floor: \
             incumbent={incumbent}, inc_x={inc_x:?}"
        );
    }

    /// 非 finite な x はどの閾の比較でも偽になり「実行可能」と誤認される。
    /// sentinel: bounds 比較を `x[k] < lbv - tol || x[k] > ub[k] + tol` に戻すと
    /// NaN が素通りして true を返し FAIL する。
    #[test]
    fn feasible_rejects_non_finite_input() {
        let n = 1usize;
        let qp = NonconvexQcqp {
            n,
            p0: None,
            q0: vec![1.0],
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
            lb: vec![0.0],
            ub: vec![10.0],
        };
        assert!(!feasible(&qp, &[f64::NAN], 1e-6));
        assert!(!feasible(&qp, &[f64::INFINITY], 1e-6));
        assert!(feasible(&qp, &[1.0], 1e-6));
    }

    /// 行 activity の overflow (`inf + -inf`) が quad の拘束式内で NaN になり、
    /// NaN の残差が `v > tol` を素通りして実行可能と誤認される欠陥 (conic
    /// feasible ゲートの NaN fail-open) を検証。x 自体は有限。
    ///
    /// v = 0.5 * xᵀpx + qᵀx + r = 0.5·(1e155·1e155) + (-1e154)·1e155
    ///   = 0.5·inf + (-inf) = NaN。手計算オラクル。
    /// sentinel: quad 比較を `v > tol` に戻すと NaN が素通りして FAIL。
    #[test]
    fn feasible_rejects_nan_quad_activity() {
        let n = 1usize;
        let p = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let qp = NonconvexQcqp {
            n,
            p0: None,
            q0: vec![1.0],
            quad: vec![GQuadConstraint {
                p,
                q: vec![-1e154],
                r: 0.0,
            }],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
            lb: vec![0.0],
            ub: vec![f64::INFINITY],
        };
        assert!(
            !feasible(&qp, &[1e155], 1e-6),
            "quadratic activity overflows to NaN (0.5·inf + -inf); must not be accepted as feasible"
        );
        assert!(
            f64::is_nan(
                quad_val(&qp.quad[0].p, &[1e155])
                    + qp.quad[0]
                        .q
                        .iter()
                        .zip(&[1e155])
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                    + qp.quad[0].r
            ),
            "test precondition: the activity really does evaluate to NaN"
        );
    }

    /// `p` is 3x3 but `x` has 2 entries. Must panic, not silently zero-fill;
    /// reverting `quad_val` to the old zero-fill-on-Err fallback makes this
    /// FAIL (silently returns `0.0` instead).
    #[test]
    #[should_panic(expected = "p.ncols() == x.len()")]
    fn quad_val_panics_on_dimension_mismatch() {
        let p = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1.0], 3, 3).unwrap();
        let x = vec![1.0, 2.0];
        let _ = quad_val(&p, &x);
    }

    /// `g_lin` is built with 3 columns while `qp.n`/`x.len()` is 2, bypassing
    /// `NonconvexQcqp::validate()`. Must panic, not silently zero-fill;
    /// reverting to the old zero-fill-on-Err fallback makes this FAIL.
    #[test]
    #[should_panic(expected = "g_lin.ncols() == x.len() == n")]
    fn feasible_panics_on_g_lin_dimension_mismatch() {
        let n = 2usize;
        let qp = NonconvexQcqp {
            n,
            p0: None,
            q0: vec![0.0; n],
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, 3).unwrap(), // wrong: n=2, cols=3
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
            lb: vec![0.0; n],
            ub: vec![1.0; n],
        };
        let x = vec![0.5, 0.5];
        let _ = feasible(&qp, &x, 1e-6);
    }

    /// Same rationale as `feasible_panics_on_g_lin_dimension_mismatch`, for
    /// the equality branch (reached after `g_lin` passes).
    #[test]
    #[should_panic(expected = "a_eq.ncols() == x.len() == n")]
    fn feasible_panics_on_a_eq_dimension_mismatch() {
        let n = 2usize;
        let qp = NonconvexQcqp {
            n,
            p0: None,
            q0: vec![0.0; n],
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, 3).unwrap(), // wrong: n=2, cols=3
            b_eq: vec![],
            lb: vec![0.0; n],
            ub: vec![1.0; n],
        };
        let x = vec![0.5, 0.5];
        let _ = feasible(&qp, &x, 1e-6);
    }

    /// PR #25 review INLINE-Q: the leaf `accept` call in `global_core`
    /// (`worst_gap <= g.bilinear_tol`) passed `g.feas_tol * 100.0` instead of
    /// `g.feas_tol`, 100x looser than the per-node `accept` call just above
    /// it and than the caller's requested tolerance.
    ///
    /// `x0` is a single integer variable pinned by an equality to `5 +
    /// 5e-5`: `int_tol = 1e-3` accepts it as "integral" (no integer branch
    /// fires), and with zero bilinear pairs (`quad`/`p0` both empty) every
    /// node is immediately a spatial leaf, so `global_core` reaches the leaf
    /// `accept` call at the root. Rounding to `5.0` leaves a `5e-5` equality
    /// residual: 50x `feas_tol = 1e-6` (inside the old `100x` leaf
    /// loosening, which would have accepted it) but outside `feas_tol`
    /// itself (which must reject it).
    ///
    /// Sentinel: restoring the leaf's `g.feas_tol * 100.0` makes this FAIL
    /// (the point is accepted, `res.x = [5.0]`, `res.status = Optimal`).
    #[test]
    fn leaf_accept_no_longer_loosens_feas_tol_by_100x() {
        let n = 1usize;
        let qp = NonconvexQcqp {
            n,
            p0: None,
            q0: vec![0.0],
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap(),
            b_eq: vec![5.0 + 5e-5],
            lb: vec![0.0],
            ub: vec![10.0],
        };
        let g = GlobalOptions {
            feas_tol: 1e-6,
            int_tol: 1e-3,
            ..GlobalOptions::default()
        };
        let res = solve_global_miqcp(&qp, &[0], &ConicOptions::default(), &g);
        assert!(
            res.x.is_empty(),
            "a leaf point violating the equality by 5e-5 (50x feas_tol=1e-6, \
             inside the old 100x-loosened leaf tolerance of 1e-4) must not \
             be accepted as feasible: {res:?}"
        );
        assert_eq!(res.status, SolveStatus::NumericalError, "{res:?}");
    }

    /// PR #25 review pattern applied here too ("Reject NaN integrality
    /// tolerances before branching", originally flagged on
    /// `misocp::BbOptions`): `GlobalOptions::int_tol = NaN` must be rejected
    /// up front. Every `frac_dist(..) <= tol` / `d > worst_if` comparison
    /// against a `NaN` tolerance is silently `false`, so `all_integral` never
    /// accepts any point (blocking every incumbent) while the fractional-
    /// variable finder never picks a branch variable either -- the search
    /// runs dry without ever reaching the true integer optimum.
    ///
    /// `min x0*x1` s.t. `x0, x1` integer in `[-2, 2]` has global optimum `-4`
    /// at `(-2, 2)`/`(2, -2)` (mirrors `nonconvex_miqcp_integer_bilinear` in
    /// `conic::tests`); with `int_tol = NaN` a sound search must reject the
    /// options rather than silently report an incumbent-less failure.
    ///
    /// Sentinel: dropping the `g.validate()` call in `global_core` makes this
    /// FAIL with `NumericalError` (the search finds no incumbent) instead of
    /// `NotSupported`.
    #[test]
    fn global_miqcp_nan_int_tol_is_rejected_not_silently_broken() {
        let n = 2usize;
        let p = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[1.0, 1.0], n, n).unwrap();
        let qp = NonconvexQcqp {
            n,
            p0: Some(p),
            q0: vec![0.0, 0.0],
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
            lb: vec![-2.0, -2.0],
            ub: vec![2.0, 2.0],
        };
        let g = GlobalOptions {
            int_tol: f64::NAN,
            ..GlobalOptions::default()
        };
        let res = solve_global_miqcp(&qp, &[0, 1], &ConicOptions::default(), &g);
        assert!(
            matches!(res.status, SolveStatus::NotSupported(_)),
            "NaN int_tol must be rejected, got {res:?}"
        );
        assert!(res.x.is_empty());
    }

    /// PR #25 review INLINE-S: `NonconvexQcqp` had no `validate()`, so
    /// `build_static`'s unchecked `c[..n].copy_from_slice(&qp.q0)` (and the
    /// analogous `lb`/`ub`/`p0`/`quad[..]` dimension assumptions) panicked on
    /// a mismatched public struct instead of failing cleanly. Every branch
    /// independently exercised against a minimally-valid `n = 2` baseline.
    #[test]
    fn validate_rejects_every_mismatched_dimension() {
        let base = || NonconvexQcqp {
            n: 2,
            p0: None,
            q0: vec![0.0, 0.0],
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap(),
            b_eq: vec![],
            lb: vec![0.0, 0.0],
            ub: vec![1.0, 1.0],
        };
        assert!(base().validate().is_ok(), "baseline must be valid");

        let mut bad = base();
        bad.q0 = vec![0.0];
        assert!(bad.validate().is_err(), "short q0 must be rejected");

        let mut bad = base();
        bad.lb = vec![0.0];
        assert!(bad.validate().is_err(), "short lb must be rejected");

        let mut bad = base();
        bad.ub = vec![1.0];
        assert!(bad.validate().is_err(), "short ub must be rejected");

        let mut bad = base();
        bad.p0 = Some(CscMatrix::from_triplets(&[], &[], &[], 3, 3).unwrap());
        assert!(bad.validate().is_err(), "mis-shaped p0 must be rejected");

        let mut bad = base();
        bad.quad = vec![GQuadConstraint {
            p: CscMatrix::from_triplets(&[], &[], &[], 3, 3).unwrap(),
            q: vec![0.0, 0.0],
            r: 0.0,
        }];
        assert!(
            bad.validate().is_err(),
            "mis-shaped quad[].p must be rejected"
        );

        let mut bad = base();
        bad.quad = vec![GQuadConstraint {
            p: CscMatrix::from_triplets(&[], &[], &[], 2, 2).unwrap(),
            q: vec![0.0],
            r: 0.0,
        }];
        assert!(bad.validate().is_err(), "short quad[].q must be rejected");

        let mut bad = base();
        bad.g_lin = CscMatrix::from_triplets(&[], &[], &[], 0, 3).unwrap();
        assert!(bad.validate().is_err(), "mis-shaped g_lin must be rejected");

        let mut bad = base();
        bad.g_lin = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 2).unwrap();
        assert!(
            bad.validate().is_err(),
            "g_lin row count without matching h_lin must be rejected"
        );

        let mut bad = base();
        bad.a_eq = CscMatrix::from_triplets(&[], &[], &[], 0, 3).unwrap();
        assert!(bad.validate().is_err(), "mis-shaped a_eq must be rejected");

        let mut bad = base();
        bad.a_eq = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 2).unwrap();
        assert!(
            bad.validate().is_err(),
            "a_eq row count without matching b_eq must be rejected"
        );

        let mut bad = base();
        bad.lb = vec![f64::NAN, 0.0];
        assert!(bad.validate().is_err(), "NaN lb must be rejected");

        let mut bad = base();
        bad.lb = vec![5.0, 0.0];
        bad.ub = vec![1.0, 1.0];
        assert!(bad.validate().is_err(), "lb > ub must be rejected");
    }

    /// Decision D8 replacement pair for the old "non-finite ub must be
    /// rejected" assertion (removed per the delete-and-add rule): finiteness
    /// is now demanded exactly on `R`.
    ///
    /// Sentinel (plan §5 sentinel 1 direction): with `x0` in `R` (an
    /// indefinite constraint row touches it) a free `x0` must still be
    /// rejected naming variable 0; a free `x1` outside `R` must be accepted.
    /// Reverting `validate` to the old all-finite rule fails the accept half;
    /// dropping the `R` gate fails the reject half.
    #[test]
    fn validate_requires_finite_bounds_only_on_r() {
        // quad[0]: -x0^2 + x0 <= 0 (composed diagonal -1 < 0 -> rule-2 veto
        // marks x0; nothing touches x1) => R = {x0}.
        let base = || NonconvexQcqp {
            n: 2,
            p0: None,
            q0: vec![0.0, 0.0],
            quad: vec![GQuadConstraint {
                p: CscMatrix::from_triplets(&[0], &[0], &[-2.0], 2, 2).unwrap(),
                q: vec![1.0, 0.0],
                r: 0.0,
            }],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap(),
            b_eq: vec![],
            lb: vec![0.0, 0.0],
            ub: vec![1.0, 1.0],
        };
        assert!(base().validate().is_ok());

        let mut free_r = base();
        free_r.lb = vec![f64::NEG_INFINITY, 0.0];
        free_r.ub = vec![f64::INFINITY, 1.0];
        let err = free_r.validate().unwrap_err();
        assert!(
            err.contains("lb[0]") && err.contains('R'),
            "free x0 in R must be rejected by name: {err}"
        );

        let mut free_out = base();
        free_out.lb = vec![0.0, f64::NEG_INFINITY];
        free_out.ub = vec![1.0, f64::INFINITY];
        assert!(
            free_out.validate().is_ok(),
            "free x1 outside R must be accepted"
        );

        // NaN bounds are corrupt data, not "free": rejected everywhere.
        let mut nan_free = base();
        nan_free.ub = vec![1.0, f64::NAN];
        assert!(
            nan_free.validate().is_err(),
            "NaN ub outside R rejected too"
        );
    }

    /// Plan §5 sentinel 1 (validate level): plan §3.1 example 1 — free
    /// `x`, PSD ellipsoid-style rows — must clear `validate` (the classifier
    /// exempts PSD rows, so `R` is empty). Full end-to-end (no
    /// `NotSupported("... not finite ...")`) is pinned in the M5 suite.
    #[test]
    fn validate_accepts_psd_rows_over_free_variables() {
        // min x1  s.t.  x0^2 + x1^2 - 1 <= 0, x free.
        let n = 2usize;
        let qp = NonconvexQcqp {
            n,
            p0: None,
            q0: vec![0.0, 1.0],
            quad: vec![GQuadConstraint {
                p: CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap(),
                q: vec![0.0, 0.0],
                r: -1.0,
            }],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
            lb: vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
            ub: vec![f64::INFINITY, f64::INFINITY],
        };
        qp.validate()
            .expect("PSD rows over free x must validate (R empty)");
    }

    /// Companion to `validate_rejects_every_mismatched_dimension`: the public
    /// entry path (`global_core`, reached via `solve_global_qcqp`/
    /// `solve_global_miqcp`) must surface a mismatched `NonconvexQcqp` as a
    /// clean `NotSupported` result, not panic. Pre-fix, `q0` shorter than `n`
    /// panicked inside `build_static`'s `copy_from_slice`.
    ///
    /// Sentinel: dropping the `qp.validate()` call in `global_core` makes
    /// this FAIL by panicking (not by a different `status`).
    #[test]
    fn mismatched_dims_returns_not_supported_not_a_panic() {
        let qp = NonconvexQcqp {
            n: 2,
            p0: None,
            q0: vec![0.0], // one short of n=2
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap(),
            b_eq: vec![],
            lb: vec![0.0, 0.0],
            ub: vec![1.0, 1.0],
        };
        let res = solve_global_qcqp(&qp, &ConicOptions::default(), &GlobalOptions::default());
        assert!(
            matches!(res.status, SolveStatus::NotSupported(_)),
            "mismatched q0 length must be a clean NotSupported, got {res:?}"
        );
        assert!(res.x.is_empty());
    }

    #[test]
    fn public_entry_rejects_non_finite_data_and_integer_indices() {
        let base = || NonconvexQcqp {
            n: 1,
            p0: None,
            q0: vec![0.0],
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap(),
            b_eq: vec![],
            lb: vec![0.0],
            ub: vec![1.0],
        };
        let mut non_finite = base();
        non_finite.q0[0] = f64::NAN;
        let res = solve_global_qcqp(
            &non_finite,
            &ConicOptions::default(),
            &GlobalOptions::default(),
        );
        assert!(
            matches!(res.status, SolveStatus::NotSupported(_)),
            "{res:?}"
        );

        let res = solve_global_miqcp(
            &base(),
            &[1],
            &ConicOptions::default(),
            &GlobalOptions::default(),
        );
        assert!(
            matches!(res.status, SolveStatus::NotSupported(_)),
            "{res:?}"
        );
        assert_eq!(res.nodes, 0);
    }

    /// `min x0 + x1  s.t.  x0*x1 >= 1,  x in [0.1,3]^2` (optimum 2 at (1,1)).
    /// Requires deep spatial branching (~99 nodes), so a fault injected past
    /// the incumbent node still leaves plenty of the tree unresolved.
    fn hyperbola() -> NonconvexQcqp {
        let n = 2usize;
        // (1/2) x^T P x with P = [[0,-1],[-1,0]] = -x0*x1; constraint -x0*x1 + 1 <= 0.
        let p = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[-1.0, -1.0], n, n).unwrap();
        NonconvexQcqp {
            n,
            p0: None,
            q0: vec![1.0, 1.0],
            quad: vec![GQuadConstraint {
                p,
                q: vec![0.0, 0.0],
                r: 1.0,
            }],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
            lb: vec![0.1, 0.1],
            ub: vec![3.0, 3.0],
        }
    }

    /// `min x0^2  over  x0 in [-1,1]`, expressed for the spatial solver.
    /// Objective `p0 = [[2]]` gives `(1/2)*2*x^2 = w`, so the relaxation
    /// minimises `w` directly — a value oracle independent of which x-vertex
    /// the LP lands on.
    fn min_sq() -> NonconvexQcqp {
        let n = 1usize;
        let p = CscMatrix::from_triplets(&[0], &[0], &[2.0], n, n).unwrap();
        NonconvexQcqp {
            n,
            p0: Some(p),
            q0: vec![0.0],
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
            lb: vec![-1.0],
            ub: vec![1.0],
        }
    }

    /// Sentinel. The rule-2 diagonal relaxation of `w = x0^2` over `[-1,1]`
    /// bounds `w` below by the x̄ = 0 tangent row `w >= 0` only: no
    /// interval rows, no endpoint secant/tangent envelope (decision D2).
    /// Hand oracle: `x^2 >= 0`, so the tight relaxation minimum of `w` is
    /// exactly 0. Removing the tangent row makes `w` free and the relaxation
    /// Unbounded; adding the old envelope rows back is caught by
    /// `rule2_pair_emits_exactly_one_pair_row` below.
    #[test]
    fn mccormick_w_box_bounds_diagonal_square_below() {
        let qp = min_sq();
        let static_ = build_static(&qp);
        assert_eq!(static_.pairs, vec![(0, 0)]);
        let relax = build_relax(&static_, &qp.lb, &qp.ub);
        let res = solve_relax_lp(&relax, None, None);
        assert_eq!(res.status, SolveStatus::Optimal, "{:?}", res.status);
        assert!(
            res.objective.abs() < 1e-7,
            "relaxed min w must be 0 (true x^2 >= 0), got {}",
            res.objective
        );
    }

    /// Sentinel (decision D2, plan §3.2): a rule-2 diagonal pair contributes
    /// exactly one relaxation row per pair -- the constant under-estimator
    /// cut `-w_ii <= 0` -- and never the interval rows or the 4 McCormick
    /// facets. Reverting D2 (envelope rows back for rule-2 pairs) raises the
    /// row count and fails.
    #[test]
    fn rule2_pair_emits_exactly_one_pair_row() {
        let qp = min_sq();
        let static_ = build_static(&qp);
        assert_eq!(static_.rule2_pair, vec![true]);
        assert_eq!(static_.mc_pair, vec![false]);
        let relax = build_relax(&static_, &qp.lb, &qp.ub);
        // m = m_fixed(0) + box(2) + tangent(1).
        assert_eq!(relax.cone.l, 3, "box rows + one tangent row");
        assert_eq!(relax.h, vec![1.0, 1.0, 0.0], "ub, -lb, tangent rhs 0");
    }

    /// Sentinel (decision D3 / plan §3.2): non-finite bounds never reach the
    /// relaxation's `h`, and a pair with a non-finite endpoint emits no
    /// interval/envelope rows (its `w` column stays unconstrained except by
    /// the lifted fixed rows -- a valid over-relaxation, since the lifted
    /// rows contain the graph exactly). Reverting to the unconditional box /
    /// pair loops pushes `±inf` into `h` and panics the LP constructor.
    #[test]
    fn non_finite_bounds_and_pairs_never_reach_h() {
        // min x1  s.t.  -x0^2 + x1 <= 0;  x0 in [-2, 2] (rule-2-vetoed
        // diagonal -> R), x1 free (linear + this row's linear part).
        let n = 2usize;
        let p = CscMatrix::from_triplets(&[0], &[0], &[-2.0], n, n).unwrap();
        let qp = NonconvexQcqp {
            n,
            p0: None,
            q0: vec![0.0, 1.0],
            quad: vec![GQuadConstraint {
                p,
                q: vec![0.0, 1.0],
                r: 0.0,
            }],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
            lb: vec![-2.0, f64::NEG_INFINITY],
            ub: vec![2.0, f64::INFINITY],
        };
        let static_ = build_static(&qp);
        // pair (0,0): diagonal lift coeff -1 -> rule 2 vetoed, x0 finite -> mc.
        assert_eq!(static_.rule2_pair, vec![false]);
        assert_eq!(static_.mc_pair, vec![true]);
        let relax = build_relax(&static_, &qp.lb, &qp.ub);
        // m = fixed quad(1) + box(2: x0 only) + envelope(6).
        assert_eq!(relax.cone.l, 9, "no box rows for the free x1");
        assert!(
            relax.h.iter().all(|v| v.is_finite()),
            "±inf must never reach h: {:?}",
            relax.h
        );
    }

    /// Sentinel (decision D3): an off-diagonal pair whose endpoints are free
    /// contributes no interval/McCormick rows; only its lifted fixed rows
    /// remain (finite rhs), so the relaxation stays constructible.
    #[test]
    fn free_off_diagonal_pair_emits_no_envelope() {
        // -x0*x1 + 1 <= 0 over free x (both pair endpoints free).
        let n = 2usize;
        let p = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[-1.0, -1.0], n, n).unwrap();
        let qp = NonconvexQcqp {
            n,
            p0: None,
            q0: vec![0.0, 0.0],
            quad: vec![GQuadConstraint {
                p,
                q: vec![0.0, 0.0],
                r: 1.0,
            }],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
            lb: [f64::NEG_INFINITY; 2].to_vec(),
            ub: [f64::INFINITY; 2].to_vec(),
        };
        let static_ = build_static(&qp);
        assert_eq!(static_.mc_pair, vec![false]);
        let relax = build_relax(&static_, &qp.lb, &qp.ub);
        assert_eq!(relax.cone.l, 1, "only the lifted quad row survives");
        assert_eq!(relax.h, vec![-1.0], "h = -r, all finite");
    }

    /// Sentinel (decision D4, plan §5 test 3): a rule-2 diagonal `w_ii` must
    /// never be a spatial branching candidate. Fixture: `min x0^2 − x1  s.t.
    /// x0 − x1 = 0`, `x1 in [0, 5]`, `x0` free — the lifted pair (0,0) is a
    /// rule-2 diagonal with a *free* box. Pre-D4 (branching scanned all
    /// pairs), the root relaxation lands at `x0 = x1 = 5, w = 0`, the term gap
    /// `|w − x0²| = 25` selects `x0` as branch variable, the relaxation point
    /// sits exactly at the child box's finite end, the fallback split
    /// `0.5·(lb+ub) = 0.5·(5 + inf)` is `+inf`, and the re-pushed child has
    /// the identical `[5, inf]` box — the search consumes the whole node
    /// budget without progress (plan §0.1 line 16). Post-D4 the root is an
    /// immediate leaf: exactly one node is processed.
    #[test]
    fn rule2_diagonal_is_never_a_branch_candidate() {
        let n = 2usize;
        let p = CscMatrix::from_triplets(&[0], &[0], &[2.0], n, n).unwrap();
        let qp = NonconvexQcqp {
            n,
            p0: Some(p),
            q0: vec![0.0, -1.0],
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap(),
            b_eq: vec![0.0],
            lb: vec![f64::NEG_INFINITY, 0.0],
            ub: vec![f64::INFINITY, 5.0],
        };
        let static_ = build_static(&qp);
        assert_eq!(
            static_.mc_pairs,
            Vec::<usize>::new(),
            "the only pair is a rule-2 diagonal (free x0, diagonal coeff +1)"
        );
        let g = GlobalOptions {
            max_nodes: 50,
            ..GlobalOptions::default()
        };
        let res = solve_global_qcqp(&qp, &ConicOptions::default(), &g);
        assert_eq!(
            res.nodes, 1,
            "root must leaf without any spatial branch (pre-D4 this burned the \
             50-node budget re-pushing identical infinite boxes): {res:?}"
        );
        assert!(
            !matches!(res.status, SolveStatus::NotSupported(_)),
            "{res:?}"
        );
    }

    /// Companion pin for `rule2_diagonal_is_never_a_branch_candidate`: with a
    /// genuine McCormick pair present, branching proceeds on its (finite-box)
    /// endpoints while the rule-2 diagonal stays out of the candidate set.
    #[test]
    fn branch_candidates_are_the_envelope_pairs_only() {
        // min x0^2 + x0*x1 s.t. x in [-1,2]^2: pair (0,0) rule 2, pair (0,1)
        // off-diagonal (indefinite objective -> R = {x0, x1}, both boxed).
        let n = 2usize;
        let p = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[2.0, 1.0, 1.0], n, n).unwrap();
        let qp = NonconvexQcqp {
            n,
            p0: Some(p),
            q0: vec![0.0, 0.0],
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
            lb: vec![-1.0, -1.0],
            ub: vec![2.0, 2.0],
        };
        let static_ = build_static(&qp);
        // pairs sorted ascending: (0,0) then (0,1).
        assert_eq!(static_.rule2_pair, vec![true, false]);
        assert_eq!(static_.mc_pairs, vec![1], "only the off-diagonal pair");
        assert!(static_.mc_pairs.iter().all(|&pi| {
            let (i, j) = static_.pairs[pi];
            qp.lb[i].is_finite()
                && qp.ub[i].is_finite()
                && qp.lb[j].is_finite()
                && qp.ub[j].is_finite()
        }));
    }

    /// Sentinel. `solve_relax_lp` must forward the caller's deadline to the
    /// LP path; an already-expired deadline therefore stops the LP with
    /// `Timeout`. Reverting to `SolverOptions::default()` (no deadline) solves
    /// the relaxation to `Optimal`, ignoring the timeout.
    #[test]
    fn relaxation_lp_honors_expired_deadline() {
        let qp = hyperbola();
        let static_ = build_static(&qp);
        let relax = build_relax(&static_, &qp.lb, &qp.ub);
        let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let res = solve_relax_lp(&relax, Some(past), None);
        assert_eq!(
            res.status,
            SolveStatus::Timeout,
            "expired deadline must abort the node LP, got {:?}",
            res.status
        );
    }

    /// Sentinel. `solve_relax_lp` must forward the caller's `cancel_flag` to the
    /// LP path; an already-fired flag therefore stops the LP with `Timeout`.
    /// Reverting to `cancel_flag: None` solves the relaxation to `Optimal`.
    #[test]
    fn relaxation_lp_honors_fired_cancel_flag() {
        let qp = hyperbola();
        let static_ = build_static(&qp);
        let relax = build_relax(&static_, &qp.lb, &qp.ub);
        let fired = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let res = solve_relax_lp(&relax, None, Some(&fired));
        assert_eq!(
            res.status,
            SolveStatus::Timeout,
            "fired cancel_flag must abort the node LP, got {:?}",
            res.status
        );
    }

    /// G2a wiring sentinel: an Unbounded relaxation LP must surface its
    /// verified recession ray (`RelaxResult::ray`, direction + base point in
    /// the relaxation's own coordinates). `solve_relax_lp` runs with default
    /// options (presolve ON), so the fixture must be one presolve cannot
    /// decide itself: `min −x1 −x2 s.t. x1 − x2 ≤ 0` (x free) is unbounded
    /// along `d = (1, 1)` (`d1 − d2 = 0 ≤ 0`, `cᵀd = −2 < 0`). Reverting the
    /// `ray: res.ray_certificate` forwarding (or the LP-side payload minting)
    /// makes `ray` None and fails this test.
    #[test]
    fn relaxation_lp_unbounded_surfaces_verified_ray() {
        let g = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, -1.0], 1, 2).unwrap();
        let relax = ConicProblem {
            c: vec![-1.0, -1.0],
            a: CscMatrix::new(0, 2),
            b: vec![],
            g,
            h: vec![0.0],
            cone: ConeSpec { l: 1, soc: vec![] },
        };
        let res = solve_relax_lp(&relax, None, None);
        assert_eq!(
            res.status,
            SolveStatus::Unbounded,
            "min −x1−x2 s.t. x1 − x2 ≤ 0 must be Unbounded, got {:?}",
            res.status
        );
        let ray = res
            .ray
            .expect("verified Unbounded relaxation must carry the ray payload");
        let d = &ray.direction;
        assert_eq!(d.len(), 2);
        assert_eq!(ray.base_point.len(), 2);
        // Independent recomputation from the hand-written problem data only.
        assert!(
            d[0] - d[1] <= 1e-6,
            "hand oracle recession: d1 − d2 ≤ 0, got {d:?}"
        );
        assert!(-d[0] - d[1] < -1e-6, "hand oracle descent: −d1−d2 < 0");
        assert!(
            ray.base_point[0] - ray.base_point[1] <= 1e-6,
            "base must satisfy x1 − x2 ≤ 0, got {:?}",
            ray.base_point
        );
    }

    /// Fixture for the §3.2 unbounded-verdict tests: `min −x0  s.t.  x1 <= x0`
    /// and `x1 <= 1`, with the coupling row carried as a quadratic row with
    /// `P = 0, q = (−1, 1), r = 0`; `x0` free, `x1 <= 1`. The original
    /// problem is genuinely unbounded along `d = e0` (`(x0, x1) = (t, 0)`,
    /// objective `-t -> -inf`). A `P = 0` row stores no quadratic support, so
    /// nothing lifts: `pairs` is empty, the relaxation is the LP itself
    /// (`R = ∅`), and every expectation below is hand-computable. The coupling
    /// row is deliberately not separable per column (LP presolve withholds
    /// the ray certificate for trivially reduced rows -- G2a contract "no
    /// false certificate"), so this LP surfaces the verified payload.
    fn free_ray_qp() -> NonconvexQcqp {
        let n = 2usize;
        NonconvexQcqp {
            n,
            p0: None,
            q0: vec![-1.0, 0.0],
            quad: vec![GQuadConstraint {
                p: CscMatrix::from_triplets(&[], &[], &[], n, n).unwrap(),
                q: vec![-1.0, 1.0],
                r: 0.0,
            }],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
            lb: vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
            ub: vec![f64::INFINITY, 1.0],
        }
    }

    /// Plan §3.2 line 111 / §5 sentinel 5 (Unbounded half): with a seed base
    /// point `x̃` (Phase 2's local-solver role, wired via the D7 seed), the
    /// verified relaxation ray passes the recession verification and the
    /// search must terminate immediately with `Unbounded`, `objective = -inf`
    /// and the base point -- not burn nodes, not report `SuboptimalSolution`
    /// (pre-D6, `Unbounded` fell into the `numerical_failures` arm and the
    /// seed incumbent surfaced as `SuboptimalSolution`).
    #[test]
    fn verified_ray_with_seed_returns_unbounded() {
        let qp = free_ray_qp();
        let seed = GlobalSeed {
            x: vec![0.0, 0.0],
            f: 0.0,
            root_lb: vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
            root_ub: vec![f64::INFINITY, 1.0],
        };
        // The seed root box must refine (never widen) the declared box, so
        // the declared x0 side is given a hair of room for the seed box.
        let mut qp_boxed = qp.clone();
        qp_boxed.lb[0] = 0.0;
        qp_boxed.ub[0] = 0.0;
        let tight = GlobalSeed {
            root_lb: vec![0.0, -1e-9],
            root_ub: vec![0.0, 1.0],
            ..seed.clone()
        };
        // (a) declared box free along x0: root relaxation Unbounded, seed
        //     verifies -> Unbounded verdict at node 1.
        let res = global_core(
            &qp,
            &[],
            &ConicOptions::default(),
            &GlobalOptions::default(),
            Some(&seed),
        );
        assert_eq!(res.status, SolveStatus::Unbounded, "{res:?}");
        assert_eq!(res.objective, f64::NEG_INFINITY);
        assert_eq!(res.x, vec![0.0, 0.0], "x must be the verified base x̃");
        assert_eq!(res.nodes, 1, "the verdict must not burn nodes: {res:?}");
        // (b) a box that pins the ray direction makes the LP bounded: the
        //     seed incumbent is reported normally (search proves Optimal).
        //     The declared box here is `[0, 0]` on x0, so the seed from (a)
        //     refines it; optimum is the truncation `x0 = 0, x1 = 1` at −0?
        //     objective `-x0 = 0` everywhere feasible -> incumbent 0, and the
        //     relaxation lands there: leaf gap 0, Optimal.
        let res_boxed = global_core(
            &qp_boxed,
            &[],
            &ConicOptions::default(),
            &GlobalOptions::default(),
            Some(&tight),
        );
        assert!(
            matches!(
                res_boxed.status,
                SolveStatus::Optimal | SolveStatus::SuboptimalSolution
            ),
            "{res_boxed:?}"
        );
        assert!(
            (res_boxed.objective - 0.0).abs() <= 1e-6,
            "truncated optimum is 0 (x0 pinned): {res_boxed:?}"
        );
    }

    /// Plan §3.2 line 110 / §5 sentinel 5 (NotSupported half, no base point):
    /// the same unbounded relaxation without any `x̃` must terminate at the
    /// first node with a *reasoned* `NotSupported` -- never an infinite loop
    /// (branching cannot remove the ray, plan line 99, and the pre-D6 code
    /// counted this as a numerical failure instead of a verdict).
    #[test]
    fn unbounded_without_base_point_is_reasoned_not_supported() {
        let qp = free_ray_qp();
        let res = solve_global_qcqp(
            &qp,
            &ConicOptions::default(),
            &GlobalOptions {
                max_nodes: 200,
                ..GlobalOptions::default()
            },
        );
        match &res.status {
            SolveStatus::NotSupported(reason) => {
                assert!(
                    reason.contains("base point") && reason.contains("x̃"),
                    "reason must name the missing x̃: {reason}"
                );
            }
            other => panic!("expected reasoned NotSupported, got {other:?} in {res:?}"),
        }
        assert_eq!(res.nodes, 1, "must verdict at the first node: {res:?}");
    }

    /// Plan line 103 in action: `min −x s.t. x² <= 1` (plan line 100's
    /// example). The true set is bounded, but the finite relaxation carries
    /// only the `x̄ = 0` tangent `w >= 0`, so its polytope IS unbounded; the
    /// ray's `P_k d = 2d ≠ 0` must veto the `Unbounded` claim (a false
    /// unboundedness certificate is the worst possible error mode). The M4 OA
    /// retry removes this ray at a far point; until then the disposition is
    /// the reasoned `NotSupported`, with `nodes == 1` pinning termination.
    #[test]
    fn bounded_set_with_unbounded_relaxation_is_not_false_unbounded() {
        let n = 1usize;
        let qp = NonconvexQcqp {
            n,
            p0: None,
            q0: vec![-1.0],
            quad: vec![GQuadConstraint {
                p: CscMatrix::from_triplets(&[0], &[0], &[2.0], n, n).unwrap(),
                q: vec![0.0],
                r: -1.0,
            }],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
            lb: vec![f64::NEG_INFINITY],
            ub: vec![f64::INFINITY],
        };
        let seed = GlobalSeed {
            x: vec![0.0],
            f: 0.0,
            root_lb: vec![f64::NEG_INFINITY],
            root_ub: vec![f64::INFINITY],
        };
        let res = global_core(
            &qp,
            &[],
            &ConicOptions::default(),
            &GlobalOptions {
                max_nodes: 200,
                ..GlobalOptions::default()
            },
            Some(&seed),
        );
        assert!(
            !matches!(res.status, SolveStatus::Unbounded),
            "false unboundedness certificate: {res:?}"
        );
        match &res.status {
            SolveStatus::NotSupported(reason) => {
                // Either disposition is sound here: the simplex ray gate may
                // withhold the certificate outright (e.g. the `w` block is
                // presolve-reduced, so the verdict arrives unproven -- the
                // `None`-ray reason), or the certificate arrives and the
                // `P_k d = 0` check vetoes it (names `quad[0]`). What must
                // never happen is the `Unbounded` certificate itself.
                assert!(
                    reason.contains("ray certificate") || reason.contains("quad[0]"),
                    "reason must name the withheld certificate or the vetoing row: {reason}"
                );
            }
            other => panic!("expected reasoned NotSupported, got {other:?} in {res:?}"),
        }
        assert_eq!(res.nodes, 1, "must verdict at the first node: {res:?}");
    }

    /// Hand-oracle battery for `verify_recession_unboundedness` (independent
    /// of the LP solver: the rays below are computed on paper from
    /// `free_ray_qp`'s data and fed directly, so the verifier's accept half
    /// cannot be "explained" by a solver bug in either direction).
    #[test]
    fn recession_verifier_hand_oracle_cases() {
        let qp = free_ray_qp();
        let static_ = build_static(&qp);
        let cert = |d: &[f64]| crate::problem::RayCertificate {
            direction: d.to_vec(),
            base_point: vec![0.0, 0.0],
        };
        // Accept: d = e0 (relaxation dim 2: pair (0,0) lifted by collect_pairs
        // from the quad row's diagonal support? P = 0 stores no diagonal ->
        // no pair, nv = n = 2).
        assert!(
            static_.pairs.is_empty(),
            "P = 0 rows lift nothing: {:?}",
            static_.pairs
        );
        verify_recession_unboundedness(&qp, &[], &static_, &cert(&[1.0, 0.0]))
            .expect("genuine recession direction must verify");
        // Reject: the row `-x0 + x1 <= 0` grows along d = -e0 (q^T d = 1).
        let err =
            verify_recession_unboundedness(&qp, &[], &static_, &cert(&[-1.0, 0.0])).unwrap_err();
        assert!(err.contains("grows along the ray"), "{err}");
        // Reject: d = -(1,1) is a genuine recession direction (the row stays
        // flat, both bounds allow it) but the objective ASCENDS (q0^T d = 1).
        let err =
            verify_recession_unboundedness(&qp, &[], &static_, &cert(&[-1.0, -1.0])).unwrap_err();
        assert!(err.contains("does not descend"), "{err}");
        // Reject: ray increases the upper-bounded x1 (finite ub forces
        // d1 <= 0). Note `(1, -1)` would NOT be rejected -- decreasing a
        // half-bounded variable is a genuine recession direction.
        let err =
            verify_recession_unboundedness(&qp, &[], &static_, &cert(&[1.0, 1.0])).unwrap_err();
        assert!(err.contains("bounded variable 1"), "{err}");
        // Reject: zero x-block (only w drifts is meaningless downstream).
        let err =
            verify_recession_unboundedness(&qp, &[], &static_, &cert(&[0.0, 0.0])).unwrap_err();
        assert!(err.contains("zero x-block"), "{err}");
        // Reject: movement on an `R` variable (`-x0^2 + x0 <= 0` puts x0 in
        // R via the rule-2 veto; the box rows should have made d_0 = 0).
        let r_qp = NonconvexQcqp {
            quad: vec![GQuadConstraint {
                p: CscMatrix::from_triplets(&[0], &[0], &[-2.0], 2, 2).unwrap(),
                q: vec![1.0, 0.0],
                r: 0.0,
            }],
            lb: vec![0.0, f64::NEG_INFINITY],
            ub: vec![1.0, f64::INFINITY],
            ..free_ray_qp()
        };
        let r_static = build_static(&r_qp);
        assert!(r_static.required[0] && !r_static.required[1]);
        // `r_qp` lifts the (0,0) diagonal (P = -2 stores support): relaxation
        // dimension is 3, so the certificate carries a trailing w slot.
        let r_cert = crate::problem::RayCertificate {
            direction: vec![1.0, 1.0, 0.0],
            base_point: vec![0.0; 3],
        };
        let err = verify_recession_unboundedness(&r_qp, &[], &r_static, &r_cert).unwrap_err();
        assert!(err.contains("bound-required variable 0"), "{err}");
        // Scale invariance: 1e6 * e0 normalises onto the same verdict.
        verify_recession_unboundedness(&qp, &[], &static_, &cert(&[1e6, 0.0]))
            .expect("the certificate direction is scale-free");
    }

    /// Sentinel. A failed root relaxation (no certificate) leaves the whole
    /// feasible region unexplored: the search must report `NumericalError`, not
    /// a false `Infeasible`. Reverting to the blanket `continue` (no failure
    /// tracking) falls through to the empty-incumbent `Infeasible` branch.
    #[test]
    fn failed_root_relaxation_is_not_false_infeasible() {
        let qp = hyperbola();
        let res = with_relax_plan(vec![Some(SolveStatus::NumericalError)], || {
            solve_global_qcqp(&qp, &ConicOptions::default(), &GlobalOptions::default())
        });
        assert_eq!(res.status, SolveStatus::NumericalError, "{res:?}");
        assert!(res.x.is_empty());
    }

    /// Sentinel. An exhausted search with an incumbent but a node relaxation
    /// that failed without a certificate is not a global-optimality proof: it
    /// must report `SuboptimalSolution`, never `Optimal`. Node 81 (well past the
    /// incumbent node, well before the ~99-node exhaustion) is forced to fail.
    /// Reverting the failure tracking + gap-gated status makes it claim
    /// `Optimal` (status was hardcoded to `Optimal` whenever an incumbent
    /// existed and the search was not node/time limited).
    #[test]
    fn incumbent_with_failed_node_is_not_optimal() {
        let qp = hyperbola();
        let mut plan = vec![None; 80];
        plan.extend(std::iter::repeat_n(Some(SolveStatus::NumericalError), 512));
        let res = with_relax_plan(plan, || {
            solve_global_qcqp(&qp, &ConicOptions::default(), &GlobalOptions::default())
        });
        assert_eq!(res.status, SolveStatus::SuboptimalSolution, "{res:?}");
        assert!(
            (res.objective - 2.0).abs() < 5e-3,
            "incumbent must survive, got {}",
            res.objective
        );
    }

    /// P2 contract. A node-limited search that already holds an incumbent is
    /// `SuboptimalSolution` (valid but unproven), never `Optimal` and never
    /// `MaxIterations` (that is reserved for the incumbent-less case). This
    /// matches `misocp::solve_misocp` / `mip::solve_miqp`. `max_nodes = 80`
    /// stops between the incumbent node (~60) and full exhaustion (~99).
    #[test]
    fn node_limited_with_incumbent_is_suboptimal() {
        let qp = hyperbola();
        let g = GlobalOptions {
            max_nodes: 80,
            ..GlobalOptions::default()
        };
        let res = solve_global_qcqp(&qp, &ConicOptions::default(), &g);
        assert_eq!(res.status, SolveStatus::SuboptimalSolution, "{res:?}");
        assert_eq!(res.nodes, 80, "must stop at the node limit");
        assert!(
            (res.objective - 2.0).abs() < 5e-3,
            "incumbent must be present, got {}",
            res.objective
        );
    }

    /// PR #25 review INLINE-D: the B&B loop only checked `opts.deadline`,
    /// never `stop_requested()` (deadline *or* `cancel_flag`), so a cancel
    /// fired mid-search went unhonored until node/deadline exhaustion -- the
    /// pre-launch guard in `qp::qcqp_route::solve_qcqp_via_conic` only covers
    /// a flag already set *before* this loop starts, not one checked inside
    /// it. With `deadline: None` and the flag already set, `hyperbola`'s full
    /// ~99-node search (see `exhausted_clean_search_certifies_optimal_with_zero_gap`,
    /// same `qp`/`GlobalOptions::default()`) must stop at the very first
    /// iteration instead of running to completion.
    #[test]
    fn cancel_flag_is_honored_inside_the_bnb_loop() {
        let qp = hyperbola();
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let opts = ConicOptions {
            cancel_flag: Some(cancel),
            ..ConicOptions::default()
        };
        let res = solve_global_qcqp(&qp, &opts, &GlobalOptions::default());
        assert_eq!(res.status, SolveStatus::Timeout, "{res:?}");
        assert_eq!(res.nodes, 0, "must stop before processing any node");
    }

    /// 横展開 (キャンセル後の false Infeasible): purely linear,
    /// contradictory constraints (`x0 <= -1` and `x0 >= 0`) -- the root
    /// relaxation LP is itself infeasible, so the single root region is
    /// pruned (`SolveStatus::Infeasible => continue` in `global_core`) and
    /// the stack empties immediately, with no per-node race involved.
    /// Cancel racing in right after that prune (stack now empty, about to
    /// declare the global search Infeasible) must still yield `Timeout`.
    ///
    /// Sentinel: reverting the `opts.stop_requested()` branch added right
    /// before `SolveStatus::Infeasible` in `global_core`'s final
    /// classification makes this FAIL with `Infeasible` instead of
    /// `Timeout`.
    #[test]
    fn cancel_race_before_final_infeasible_classification_prefers_timeout() {
        let n = 1usize;
        let qp = NonconvexQcqp {
            n,
            p0: None,
            q0: vec![0.0],
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, -1.0], 2, n).unwrap(),
            h_lin: vec![-1.0, 0.0],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
            lb: vec![-10.0],
            ub: vec![10.0],
        };

        FINAL_CLASSIFICATION_COMMIT_COUNT.with(|c| c.set(0));
        CANCEL_AFTER_FINAL_CLASSIFICATION_COMMIT.with(|c| c.set(Some(1)));

        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let opts = ConicOptions {
            cancel_flag: Some(std::sync::Arc::clone(&cancel)),
            ..ConicOptions::default()
        };
        let res = solve_global_qcqp(&qp, &opts, &GlobalOptions::default());
        CANCEL_AFTER_FINAL_CLASSIFICATION_COMMIT.with(|c| c.set(None));

        assert_eq!(
            res.status,
            SolveStatus::Timeout,
            "cancel racing in right before the final Infeasible classification must yield Timeout, got {res:?}"
        );
        assert!(res.x.is_empty());
    }

    /// Baseline/control for the sentinel above: without cancel injection,
    /// the same contradictory-constraint fixture must still report the
    /// genuine Infeasible certificate.
    #[test]
    fn no_cancel_still_reports_infeasible_via_global_core() {
        let n = 1usize;
        let qp = NonconvexQcqp {
            n,
            p0: None,
            q0: vec![0.0],
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, -1.0], 2, n).unwrap(),
            h_lin: vec![-1.0, 0.0],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
            lb: vec![-10.0],
            ub: vec![10.0],
        };
        let res = solve_global_qcqp(&qp, &ConicOptions::default(), &GlobalOptions::default());
        assert_eq!(
            res.status,
            SolveStatus::Infeasible,
            "baseline (no cancel): got {:?}",
            res.status
        );
    }

    /// P2-A (レビュー指摘): `global_core` の最終分類は Infeasible と
    /// Optimal (`certified`) の両方が同一の決定論的シグナルに支えられており、
    /// 一方だけ backstop すると非対称になる。`hyperbola` (通常は
    /// `exhausted_clean_search_certifies_optimal_with_zero_gap` で Optimal を
    /// 返す fixture) で、最終分類直前に cancel が観測されれば incumbent が
    /// 真に最適であっても Timeout を優先すること。
    ///
    /// Sentinel: `global_core` の `certified && opts.stop_requested()` 分岐を
    /// revert すると `Optimal` が返り FAIL する。
    #[test]
    fn cancel_race_before_final_optimal_classification_prefers_timeout() {
        let qp = hyperbola();

        FINAL_CLASSIFICATION_COMMIT_COUNT.with(|c| c.set(0));
        CANCEL_AFTER_FINAL_CLASSIFICATION_COMMIT.with(|c| c.set(Some(1)));

        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let opts = ConicOptions {
            cancel_flag: Some(std::sync::Arc::clone(&cancel)),
            ..ConicOptions::default()
        };
        let res = solve_global_qcqp(&qp, &opts, &GlobalOptions::default());
        CANCEL_AFTER_FINAL_CLASSIFICATION_COMMIT.with(|c| c.set(None));

        assert_eq!(
            res.status,
            SolveStatus::Timeout,
            "cancel racing in right before the final Optimal classification must yield Timeout even though the incumbent is truly optimal, got {res:?}"
        );
        assert!(
            !res.x.is_empty(),
            "incumbent must still be reported under Timeout"
        );
    }

    /// Positive control (companion to the sentinel above). A clean, fully exhausted search proves global
    /// optimality: `Optimal` with a certified near-zero gap.
    #[test]
    fn exhausted_clean_search_certifies_optimal_with_zero_gap() {
        let qp = hyperbola();
        let res = solve_global_qcqp(&qp, &ConicOptions::default(), &GlobalOptions::default());
        assert_eq!(res.status, SolveStatus::Optimal, "{res:?}");
        assert!((res.objective - 2.0).abs() < 5e-3, "obj={}", res.objective);
        assert!(
            res.gap <= GlobalOptions::default().gap_tol + 1e-9,
            "gap must be within tol, got {}",
            res.gap
        );
    }

    // -----------------------------------------------------------------
    // Independent oracle: the pre-fix dense implementation, kept
    // self-contained here purely as a reference for
    // `sparse_relax_matches_dense_reference` below. Production code must
    // never call these again.
    // -----------------------------------------------------------------

    fn reference_dense(a: &CscMatrix) -> Vec<Vec<f64>> {
        a.to_dense_rows()
    }

    fn reference_collect_pairs(qp: &NonconvexQcqp) -> Vec<(usize, usize)> {
        let n = qp.n;
        let mut seen = vec![false; n * n];
        let mark = |i: usize, j: usize, seen: &mut Vec<bool>| {
            let (a, b) = if i <= j { (i, j) } else { (j, i) };
            seen[a * n + b] = true;
        };
        let scan = |p: &CscMatrix, seen: &mut Vec<bool>| {
            let d = reference_dense(p);
            for i in 0..n {
                for j in 0..n {
                    if d[i][j] != 0.0 {
                        mark(i, j, seen);
                    }
                }
            }
        };
        if let Some(p0) = &qp.p0 {
            scan(p0, &mut seen);
        }
        for qc in &qp.quad {
            scan(&qc.p, &mut seen);
        }
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in i..n {
                if seen[i * n + j] {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }

    fn reference_add_quad_coeffs(
        d: &[Vec<f64>],
        n: usize,
        pairs: &[(usize, usize)],
        row: &mut [f64],
    ) {
        for (pi, &(i, j)) in pairs.iter().enumerate() {
            let wj = n + pi;
            let coeff = if i == j {
                0.5 * d[i][i]
            } else {
                0.5 * (d[i][j] + d[j][i])
            };
            if coeff != 0.0 {
                row[wj] += coeff;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_push_mc(
        rows: &mut Vec<Vec<f64>>,
        h: &mut Vec<f64>,
        nv: usize,
        wj: usize,
        i: usize,
        j: usize,
        wcoef: f64,
        xi_coef: f64,
        xj_coef: f64,
        rhs: f64,
    ) {
        let mut r = vec![0.0; nv];
        r[wj] += wcoef;
        r[i] += xi_coef;
        if i != j {
            r[j] += xj_coef;
        } else {
            r[i] += xj_coef;
        }
        rows.push(r);
        h.push(rhs);
    }

    fn reference_build_relax(
        qp: &NonconvexQcqp,
        pairs: &[(usize, usize)],
        lb: &[f64],
        ub: &[f64],
    ) -> ConicProblem {
        // G2b contract change (documented at PR-description level): the dense
        // oracle's independence claim covers the ASSEMBLY arithmetic (dense
        // row scans, coefficient sums, row layout) -- it predates and knows
        // nothing about the sparse triplet path. The §3.1 classification that
        // now drives emission (which rows exist at all) is the shared
        // *specification* under test, verified independently in
        // `qcqp_classify::tests` (hand-computed rules 1-3 / §5 examples) and
        // `tests/qcqp_classify_qplib_phase0.rs` (external Gershgorin-eigvalsh
        // reconciliation on QPLIB data). Sharing it here keeps the oracle a
        // dense-assembly checker rather than a stale copy of the old
        // "every pair boxed and enveloped" shape, which plan §3.2 explicitly
        // replaces.
        let n = qp.n;
        let npair = pairs.len();
        let nv = n + npair;
        let req =
            super::super::qcqp_classify::classify_bound_requirements(n, qp.p0.as_ref(), &qp.quad)
                .unwrap();
        let boxed: Vec<bool> = (0..n)
            .map(|j| lb[j].is_finite() && ub[j].is_finite())
            .collect();
        let rule2_pair: Vec<bool> = pairs
            .iter()
            .map(|&(i, j)| i == j && req.rule2_diag[i])
            .collect();
        let mut c = vec![0.0; nv];
        c[..n].copy_from_slice(&qp.q0);
        if let Some(p0) = &qp.p0 {
            let d = reference_dense(p0);
            reference_add_quad_coeffs(&d, n, pairs, &mut c);
        }

        let ae = reference_dense(&qp.a_eq);
        let peq = qp.b_eq.len();
        let mut at = (Vec::new(), Vec::new(), Vec::new());
        for (i, row) in ae.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    at.0.push(i);
                    at.1.push(j);
                    at.2.push(v);
                }
            }
        }
        let a = CscMatrix::from_triplets(&at.0, &at.1, &at.2, peq, nv).unwrap();

        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut h: Vec<f64> = Vec::new();

        for qc in &qp.quad {
            let mut row = vec![0.0; nv];
            for (j, &v) in qc.q.iter().enumerate() {
                row[j] += v;
            }
            let d = reference_dense(&qc.p);
            reference_add_quad_coeffs(&d, n, pairs, &mut row);
            rows.push(row);
            h.push(-qc.r);
        }
        let gl = reference_dense(&qp.g_lin);
        for (i, row) in gl.iter().enumerate() {
            let mut r = vec![0.0; nv];
            r[..n].copy_from_slice(row);
            rows.push(r);
            h.push(qp.h_lin[i]);
        }
        for j in 0..n {
            if !boxed[j] {
                continue;
            }
            let mut r = vec![0.0; nv];
            r[j] = 1.0;
            rows.push(r);
            h.push(ub[j]);
            let mut r2 = vec![0.0; nv];
            r2[j] = -1.0;
            rows.push(r2);
            h.push(-lb[j]);
        }
        for (pi, &(i, j)) in pairs.iter().enumerate() {
            let wj = n + pi;
            if rule2_pair[pi] {
                // Under-estimator tangent at x̄ = 0: -w_ii <= 0 (D2).
                let mut rt = vec![0.0; nv];
                rt[wj] = -1.0;
                rows.push(rt);
                h.push(0.0);
                continue;
            }
            if !(boxed[i] && (i == j || boxed[j])) {
                continue;
            }
            let (xl_i, xu_i) = (lb[i], ub[i]);
            let (xl_j, xu_j) = (lb[j], ub[j]);
            let (w_lo, w_hi) = w_interval(xl_i, xu_i, xl_j, xu_j, i == j);
            let mut r_hi = vec![0.0; nv];
            r_hi[wj] = 1.0;
            rows.push(r_hi);
            h.push(w_hi);
            let mut r_lo = vec![0.0; nv];
            r_lo[wj] = -1.0;
            rows.push(r_lo);
            h.push(-w_lo);
            reference_push_mc(
                &mut rows,
                &mut h,
                nv,
                wj,
                i,
                j,
                -1.0,
                xl_j,
                xl_i,
                xl_i * xl_j,
            );
            reference_push_mc(
                &mut rows,
                &mut h,
                nv,
                wj,
                i,
                j,
                -1.0,
                xu_j,
                xu_i,
                xu_i * xu_j,
            );
            reference_push_mc(
                &mut rows,
                &mut h,
                nv,
                wj,
                i,
                j,
                1.0,
                -xu_j,
                -xl_i,
                -xl_i * xu_j,
            );
            reference_push_mc(
                &mut rows,
                &mut h,
                nv,
                wj,
                i,
                j,
                1.0,
                -xl_j,
                -xu_i,
                -xu_i * xl_j,
            );
        }

        let m = rows.len();
        let mut gt = (Vec::new(), Vec::new(), Vec::new());
        for (ri, row) in rows.iter().enumerate() {
            for (ci, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    gt.0.push(ri);
                    gt.1.push(ci);
                    gt.2.push(v);
                }
            }
        }
        let g = CscMatrix::from_triplets(&gt.0, &gt.1, &gt.2, m, nv).unwrap();
        ConicProblem {
            c,
            a,
            b: qp.b_eq.clone(),
            g,
            h,
            cone: ConeSpec { l: m, soc: vec![] },
        }
    }

    fn assert_csc_eq(a: &CscMatrix, b: &CscMatrix, label: &str) {
        assert_eq!(a.nrows(), b.nrows(), "{label}: nrows");
        assert_eq!(a.ncols(), b.ncols(), "{label}: ncols");
        assert_eq!(a.col_ptr(), b.col_ptr(), "{label}: col_ptr");
        assert_eq!(a.row_ind(), b.row_ind(), "{label}: row_ind");
        assert_eq!(a.values(), b.values(), "{label}: values");
    }

    fn assert_relax_eq(sparse: &ConicProblem, reference: &ConicProblem, label: &str) {
        assert_eq!(sparse.c, reference.c, "{label}: c");
        assert_eq!(sparse.b, reference.b, "{label}: b");
        assert_eq!(sparse.h, reference.h, "{label}: h");
        assert_eq!(sparse.cone.l, reference.cone.l, "{label}: cone.l");
        assert_eq!(sparse.cone.soc, reference.cone.soc, "{label}: cone.soc");
        assert_csc_eq(&sparse.a, &reference.a, &format!("{label}: a"));
        assert_csc_eq(&sparse.g, &reference.g, &format!("{label}: g"));
    }

    fn csc(rows: &[Vec<f64>], nrows: usize, ncols: usize) -> CscMatrix {
        let mut ri = Vec::new();
        let mut ci = Vec::new();
        let mut vi = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    ri.push(i);
                    ci.push(j);
                    vi.push(v);
                }
            }
        }
        CscMatrix::from_triplets(&ri, &ci, &vi, nrows, ncols).unwrap()
    }

    /// Sentinel (independent-oracle equivalence): the sparse `build_static` +
    /// `build_relax` must construct byte-for-byte the same relaxation as the
    /// pre-fix dense reference above, across a spread of objective/constraint
    /// shapes (diagonal-only, off-diagonal, asymmetric-storage quadratic
    /// matrices, straddling/non-straddling boxes, existing equalities/
    /// inequalities, and a narrowed mid-branch-and-bound box). Reintroducing
    /// a row/column indexing mistake in the sparse rewrite changes `c`/`a`/
    /// `g`/`h` here and fails this test without needing a full LP solve.
    #[test]
    fn sparse_relax_matches_dense_reference() {
        struct Case {
            name: &'static str,
            qp: NonconvexQcqp,
            lb: Vec<f64>,
            ub: Vec<f64>,
        }

        let n2 = 2usize;
        let diag_only = NonconvexQcqp {
            n: n2,
            p0: Some(csc(&[vec![4.0, 0.0], vec![0.0, 0.5]], n2, n2)),
            q0: vec![0.0, 0.0],
            quad: vec![],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n2).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n2).unwrap(),
            b_eq: vec![],
            lb: vec![-1.0, -2.0],
            ub: vec![1.0, 2.0],
        };

        let n3 = 3usize;
        // p0: (0,0)=4, (0,1)=(1,0)=1 (symmetric), (2,2)=0.5.
        let p0 = csc(
            &[
                vec![4.0, 1.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.5],
            ],
            n3,
            n3,
        );
        // qc1.p: ONLY (0,2)=2.0 stored (no mirrored (2,0)) -- asymmetric
        // storage, exercising `get_entry`'s missing-mirror default of 0.
        let qc1_p = CscMatrix::from_triplets(&[0], &[2], &[2.0], n3, n3).unwrap();
        // qc2.p: (1,1)=2.0 (diag), (1,2)=(2,1)=-1.0 (symmetric off-diag).
        let qc2_p =
            CscMatrix::from_triplets(&[1, 1, 2], &[1, 2, 1], &[2.0, -1.0, -1.0], n3, n3).unwrap();
        let g_lin = csc(&[vec![1.0, 1.0, 0.0], vec![0.0, 1.0, 1.0]], 2, n3);
        let a_eq = csc(&[vec![1.0, 0.0, 1.0]], 1, n3);
        let mixed = NonconvexQcqp {
            n: n3,
            p0: Some(p0),
            q0: vec![0.1, -0.2, 0.0],
            quad: vec![
                GQuadConstraint {
                    p: qc1_p,
                    q: vec![0.5, 0.0, 0.0],
                    r: -3.0,
                },
                GQuadConstraint {
                    p: qc2_p,
                    q: vec![0.0, 1.0, 0.5],
                    r: 1.0,
                },
            ],
            g_lin,
            h_lin: vec![5.0, 4.0],
            a_eq,
            b_eq: vec![2.0],
            // variable 2's box [1,5] does not straddle 0 (diag non-straddling
            // branch of `w_interval`); variables 0/1 do straddle.
            lb: vec![-2.0, -1.0, 1.0],
            ub: vec![3.0, 4.0, 5.0],
        };

        let hb = hyperbola();
        // G2b shapes: rule-2 diagonal in the objective (PSD-exempt p0 so its
        // +x0^2 diagonal stays rule 2), an envelope-less off-diagonal pair
        // with a free endpoint, and a mixed finite/free box.
        let rule2_free = NonconvexQcqp {
            n: n2,
            p0: Some(csc(&[vec![2.0, 0.0], vec![0.0, 0.0]], n2, n2)),
            q0: vec![0.0, 1.0],
            quad: vec![GQuadConstraint {
                p: csc(&[vec![0.0, -1.0], vec![-1.0, 0.0]], n2, n2),
                q: vec![0.0, 0.0],
                r: 1.0,
            }],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n2).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n2).unwrap(),
            b_eq: vec![],
            lb: vec![-2.0, f64::NEG_INFINITY],
            ub: vec![2.0, f64::INFINITY],
        };
        let cases = vec![
            Case {
                name: "diagonal_only_straddling_box",
                qp: diag_only.clone(),
                lb: diag_only.lb.clone(),
                ub: diag_only.ub.clone(),
            },
            Case {
                name: "off_diagonal_bilinear_constraint",
                qp: hb.clone(),
                lb: hb.lb.clone(),
                ub: hb.ub.clone(),
            },
            Case {
                name: "off_diagonal_narrowed_box_mid_bnb",
                qp: hb.clone(),
                lb: vec![0.5, 0.2],
                ub: vec![2.0, 1.5],
            },
            Case {
                name: "mixed_asymmetric_p0_qc_glin_aeq",
                qp: mixed.clone(),
                lb: mixed.lb.clone(),
                ub: mixed.ub.clone(),
            },
            Case {
                name: "mixed_narrowed_box_mid_bnb",
                qp: mixed.clone(),
                lb: vec![-0.5, 1.0, 2.0],
                ub: vec![1.0, 3.0, 4.0],
            },
            Case {
                name: "rule2_diagonal_and_free_endpoint_pair",
                qp: rule2_free.clone(),
                lb: rule2_free.lb.clone(),
                ub: rule2_free.ub.clone(),
            },
            Case {
                name: "rule2_diagonal_narrowed_box_mid_bnb",
                qp: rule2_free.clone(),
                lb: vec![-1.0, f64::NEG_INFINITY],
                ub: vec![1.5, f64::INFINITY],
            },
        ];

        for case in &cases {
            let static_ = build_static(&case.qp);
            let sparse = build_relax(&static_, &case.lb, &case.ub);
            let ref_pairs = reference_collect_pairs(&case.qp);
            assert_eq!(static_.pairs, ref_pairs, "{}: pairs", case.name);
            let reference = reference_build_relax(&case.qp, &ref_pairs, &case.lb, &case.ub);
            assert_relax_eq(&sparse, &reference, case.name);
        }
    }
}
