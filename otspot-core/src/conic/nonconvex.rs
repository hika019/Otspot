//! Global optimisation of (possibly nonconvex) QCQP by spatial branch-and-bound
//! with McCormick relaxations of the bilinear/quadratic terms.
//!
//! Handles indefinite objective/constraint matrices. Requires finite variable
//! bounds `[lb, ub]` (needed for valid McCormick envelopes and termination).

use super::{ConeSpec, ConicOptions, ConicProblem};
use crate::problem::{ConstraintType, LpProblem, SolveStatus};
use crate::sparse::CscMatrix;

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

/// A nonconvex QCQP with finite variable bounds.
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
    /// Variable lower bounds (finite).
    pub lb: Vec<f64>,
    /// Variable upper bounds (finite).
    pub ub: Vec<f64>,
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
    let px = p.mat_vec_mul(x).unwrap_or_else(|_| vec![0.0; x.len()]);
    0.5 * x.iter().zip(&px).map(|(a, b)| a * b).sum::<f64>()
}

struct RelaxResult {
    status: SolveStatus,
    objective: f64,
    x: Vec<f64>,
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

/// McCormick relaxations are pure LPs. Use the mature simplex LP path rather
/// than the generic conic IPM, which is intentionally small and aimed at SOC
/// blocks. The caller's wall-clock deadline is forwarded so a single node LP
/// cannot run past the B&B budget.
///
/// Builds the combined equality+inequality matrix directly from `prob.a`/
/// `prob.g`'s own CSC nonzeros (`O(nnz(a) + nnz(g))`) instead of densifying
/// both to `Vec<Vec<f64>>` first -- `prob` is the per-node relaxation, so a
/// dense pass here repeated the same `O(n * m)` blowup `build_relax` is
/// fixed to avoid.
fn solve_relax_lp(prob: &ConicProblem, deadline: Option<std::time::Instant>) -> RelaxResult {
    #[cfg(test)]
    if let Some(status) = RELAX_STATUS_PLAN
        .with(|p| p.borrow_mut().pop_front())
        .flatten()
    {
        return RelaxResult {
            status,
            objective: f64::NAN,
            x: vec![],
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
        ..Default::default()
    };
    let res = crate::lp::solve_lp_with(&lp, &lp_opts);
    RelaxResult {
        status: res.status,
        objective: res.objective,
        x: res.solution,
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
fn build_static(qp: &NonconvexQcqp) -> StaticRelax {
    let n = qp.n;
    let pairs = collect_pairs(qp);
    let npair = pairs.len();
    let nv = n + npair;

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
    }
}

/// Build the McCormick LP relaxation for the current box `[lb, ub]`.
///
/// Only the box rows (`2n`) and the McCormick envelope rows (`6` per pair)
/// depend on the box; everything else is `static_` (see its doc), so this
/// clones the precomputed fixed part (`O(nnz)`) and appends the per-node
/// rows directly as triplets, instead of re-deriving the whole relaxation
/// from a dense `n x n` pass at every branch-and-bound node (`O(n * (n +
/// pairs.len()))` per node -- the dominant B&B memory cost on large QCQPs).
fn build_relax(static_: &StaticRelax, lb: &[f64], ub: &[f64]) -> ConicProblem {
    let n = static_.n;
    let pairs = &static_.pairs;
    let npair = pairs.len();
    let nv = n + npair;
    let m_fixed = static_.fixed_h.len();
    let m = m_fixed + 2 * n + 6 * npair;

    let mut ri = static_.fixed_ri.clone();
    let mut ci = static_.fixed_ci.clone();
    let mut vi = static_.fixed_vi.clone();
    let mut h = static_.fixed_h.clone();
    let box_rows = 2 * n;
    let mc_rows = 6 * npair;
    // 2 single-entry rows + 4 push_mc rows (always 3 triplets each) per pair.
    let mc_nnz = 2 * npair + 12 * npair;
    ri.reserve(box_rows + mc_nnz);
    ci.reserve(box_rows + mc_nnz);
    vi.reserve(box_rows + mc_nnz);
    h.reserve(box_rows + mc_rows);

    // variable box.
    let mut row = m_fixed;
    for j in 0..n {
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
    // relaxation of `x^2` terms is needlessly loose without it.
    for (pi, &(i, j)) in pairs.iter().enumerate() {
        let wj = n + pi;
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
fn feasible(qp: &NonconvexQcqp, x: &[f64], tol: f64) -> bool {
    for (k, &lbv) in qp.lb.iter().enumerate() {
        if x[k] < lbv - tol || x[k] > qp.ub[k] + tol {
            return false;
        }
    }
    for qc in &qp.quad {
        let v = quad_val(&qc.p, x) + qc.q.iter().zip(x).map(|(a, b)| a * b).sum::<f64>() + qc.r;
        if v > tol {
            return false;
        }
    }
    let gv = qp
        .g_lin
        .mat_vec_mul(x)
        .unwrap_or_else(|_| vec![0.0; qp.h_lin.len()]);
    for (i, &v) in gv.iter().enumerate() {
        if v > qp.h_lin[i] + tol {
            return false;
        }
    }
    let av = qp
        .a_eq
        .mat_vec_mul(x)
        .unwrap_or_else(|_| vec![0.0; qp.b_eq.len()]);
    for (i, &v) in av.iter().enumerate() {
        if (v - qp.b_eq[i]).abs() > tol {
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
    global_core(qp, &[], opts, g)
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
    global_core(qp, integers, opts, g)
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

fn global_core(
    qp: &NonconvexQcqp,
    integers: &[usize],
    opts: &ConicOptions,
    g: &GlobalOptions,
) -> GlobalResult {
    let static_ = build_static(qp);
    let mut incumbent = f64::INFINITY;
    let mut inc_x: Vec<f64> = Vec::new();
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

    let mut stack = vec![(qp.lb.clone(), qp.ub.clone(), f64::NEG_INFINITY)];

    let accept = |x: &[f64], incumbent: &mut f64, inc_x: &mut Vec<f64>, tol: f64| {
        if feasible(qp, x, tol) && all_integral(x, integers, g.int_tol) {
            // Round integers for a clean reported solution.
            let mut xr = x.to_vec();
            for &k in integers {
                xr[k] = xr[k].round();
            }
            if feasible(qp, &xr, tol.max(1e-6)) {
                let ov = objective(qp, &xr);
                if ov < *incumbent {
                    *incumbent = ov;
                    *inc_x = xr;
                }
            }
        }
    };

    while let Some((lb, ub, inherited)) = stack.pop() {
        if opts
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
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
        let res = solve_relax_lp(&relax, opts.deadline);
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
            // NumericalError, MaxIterations, SuboptimalSolution, Unbounded
            // (the lifted polytope is a bounded box, so an "unbounded" LP is
            // a numerical artefact), …: nothing was proven about the region.
            // Prune it — there is no trustworthy point to branch on — but
            // record the failure so the search never claims a certificate,
            // and keep the region's inherited bound in the global bound.
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

        accept(&x, &mut incumbent, &mut inc_x, g.feas_tol);

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

        // (2) spatial branching on the worst McCormick term gap.
        let mut worst_gap = 0.0;
        let mut branch_var = None;
        for (pi, &(i, j)) in static_.pairs.iter().enumerate() {
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
            // region's true minimum and stands as its terminal bound.
            accept(&x, &mut incumbent, &mut inc_x, g.feas_tol * 100.0);
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
        // no-incumbent node-limited case above.
        status: if timed_out {
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

    /// #30 sentinel. The McCormick relaxation of `w = x0^2` over `[-1,1]`
    /// bounds `w` below by the two endpoint tangents (`w >= 2x-1`, `w >= -2x-1`),
    /// which meet at `-1` at `x = 0`. The explicit `w`-interval row adds the
    /// true bound `w >= 0`. Hand oracle: `x^2 >= 0`, so the tight relaxation
    /// minimum of `w` is exactly 0. Reverting the `w`-box rows drops it to -1.
    #[test]
    fn mccormick_w_box_bounds_diagonal_square_below() {
        let qp = min_sq();
        let static_ = build_static(&qp);
        assert_eq!(static_.pairs, vec![(0, 0)]);
        let relax = build_relax(&static_, &qp.lb, &qp.ub);
        let res = solve_relax_lp(&relax, None);
        assert_eq!(res.status, SolveStatus::Optimal, "{:?}", res.status);
        assert!(
            res.objective.abs() < 1e-7,
            "relaxed min w must be 0 (true x^2 >= 0), got {}",
            res.objective
        );
    }

    /// #18 sentinel. `solve_relax_lp` must forward the caller's deadline to the
    /// LP path; an already-expired deadline therefore stops the LP with
    /// `Timeout`. Reverting to `SolverOptions::default()` (no deadline) solves
    /// the relaxation to `Optimal`, ignoring the timeout.
    #[test]
    fn relaxation_lp_honors_expired_deadline() {
        let qp = hyperbola();
        let static_ = build_static(&qp);
        let relax = build_relax(&static_, &qp.lb, &qp.ub);
        let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let res = solve_relax_lp(&relax, Some(past));
        assert_eq!(
            res.status,
            SolveStatus::Timeout,
            "expired deadline must abort the node LP, got {:?}",
            res.status
        );
    }

    /// #19 sentinel. A failed root relaxation (no certificate) leaves the whole
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

    /// #20 sentinel. An exhausted search with an incumbent but a node relaxation
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

    /// #20 positive control. A clean, fully exhausted search proves global
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
        let n = qp.n;
        let npair = pairs.len();
        let nv = n + npair;
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
