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
/// With an incumbent, `status` is `Timeout` (deadline hit), `MaxIterations`
/// (node-limited), `Optimal` (search exhausted, no node relaxation failed,
/// and `gap <= gap_tol` — a certificate), or `SuboptimalSolution` (search
/// exhausted but some relaxation failed without a certificate, or the proven
/// gap exceeds `gap_tol`). Without an incumbent: `Timeout` > `MaxIterations`
/// > `NumericalError` (some region was never certified empty) > `Infeasible`
/// (every region proven empty — only then is infeasibility a certificate).
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

fn dense(a: &CscMatrix) -> Vec<Vec<f64>> {
    a.to_dense_rows()
}

/// Collect the symmetric index pairs `(i,j)` with `i<=j` that appear with a
/// nonzero coefficient in any quadratic matrix.
fn collect_pairs(qp: &NonconvexQcqp) -> Vec<(usize, usize)> {
    let n = qp.n;
    let mut seen = vec![false; n * n];
    let mark = |i: usize, j: usize, seen: &mut Vec<bool>| {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        seen[a * n + b] = true;
    };
    let scan = |p: &CscMatrix, seen: &mut Vec<bool>| {
        let d = dense(p);
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
    let gd = dense(&prob.g);
    let ad = dense(&prob.a);
    let mut rows = Vec::with_capacity(prob.p() + prob.m());
    let mut rhs = Vec::with_capacity(prob.p() + prob.m());
    let mut ctypes = Vec::with_capacity(prob.p() + prob.m());
    for (i, row) in ad.iter().enumerate() {
        rows.push(row.clone());
        rhs.push(prob.b[i]);
        ctypes.push(ConstraintType::Eq);
    }
    for (i, row) in gd.iter().enumerate() {
        rows.push(row.clone());
        rhs.push(prob.h[i]);
        ctypes.push(ConstraintType::Le);
    }
    let mut rr = Vec::new();
    let mut cc = Vec::new();
    let mut vv = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                rr.push(i);
                cc.push(j);
                vv.push(v);
            }
        }
    }
    let a = CscMatrix::from_triplets(&rr, &cc, &vv, rows.len(), n).unwrap();
    let lp = LpProblem::new_general(
        prob.c.clone(),
        a,
        rhs,
        ctypes,
        vec![(f64::NEG_INFINITY, f64::INFINITY); n],
        None,
    )
    .unwrap();
    let mut lp_opts = crate::options::SolverOptions::default();
    lp_opts.deadline = deadline;
    let res = crate::lp::solve_lp_with(&lp, &lp_opts);
    RelaxResult {
        status: res.status,
        objective: res.objective,
        x: res.solution,
    }
}

/// Build the McCormick LP relaxation for the current box `[lb, ub]`.
fn build_relax(
    qp: &NonconvexQcqp,
    pairs: &[(usize, usize)],
    lb: &[f64],
    ub: &[f64],
) -> ConicProblem {
    let n = qp.n;
    let npair = pairs.len();
    let nv = n + npair;
    // objective coefficients on [x, w].
    let mut c = vec![0.0; nv];
    c[..n].copy_from_slice(&qp.q0);
    if let Some(p0) = &qp.p0 {
        let d = dense(p0);
        add_quad_coeffs(&d, n, pairs, &mut c);
    }

    // equalities on x (padded).
    let ae = dense(&qp.a_eq);
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

    // quadratic constraints -> linear in [x, w].
    for qc in &qp.quad {
        let mut row = vec![0.0; nv];
        for (j, &v) in qc.q.iter().enumerate() {
            row[j] += v;
        }
        let d = dense(&qc.p);
        add_quad_coeffs(&d, n, pairs, &mut row);
        rows.push(row);
        h.push(-qc.r);
    }
    // original linear inequalities Gl x <= hl.
    let gl = dense(&qp.g_lin);
    for (i, row) in gl.iter().enumerate() {
        let mut r = vec![0.0; nv];
        r[..n].copy_from_slice(row);
        rows.push(r);
        h.push(qp.h_lin[i]);
    }
    // variable box.
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
        let mut r_hi = vec![0.0; nv];
        r_hi[wj] = 1.0;
        rows.push(r_hi);
        h.push(w_hi);
        let mut r_lo = vec![0.0; nv];
        r_lo[wj] = -1.0;
        rows.push(r_lo);
        h.push(-w_lo);
        // w >= xl_i x_j + xl_j x_i - xl_i xl_j  =>  -w + xl_j x_i + xl_i x_j <= xl_i xl_j
        push_mc(
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
        // w >= xu_i x_j + xu_j x_i - xu_i xu_j
        push_mc(
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
        // w <= xl_i x_j + xu_j x_i - xl_i xu_j  =>  w - xu_j x_i - xl_i x_j <= -xl_i xu_j
        push_mc(
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
        // w <= xu_i x_j + xl_j x_i - xu_i xl_j
        push_mc(
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

/// Add `(1/2) x^T P x` coefficients onto a linear row over `[x, w]`.
fn add_quad_coeffs(d: &[Vec<f64>], n: usize, pairs: &[(usize, usize)], row: &mut [f64]) {
    for (pi, &(i, j)) in pairs.iter().enumerate() {
        let wj = n + pi;
        let coeff = if i == j {
            0.5 * d[i][i]
        } else {
            // symmetric: P_ij + P_ji = 2 P_ij, times 1/2 => P_ij (use both).
            0.5 * (d[i][j] + d[j][i])
        };
        if coeff != 0.0 {
            row[wj] += coeff;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_mc(
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
        // For i==j both x-coefficients apply to the same variable.
        r[i] += xj_coef;
    }
    rows.push(r);
    h.push(rhs);
}

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
    let gl = dense(&qp.g_lin);
    for (i, row) in gl.iter().enumerate() {
        let v: f64 = row.iter().zip(x).map(|(a, b)| a * b).sum();
        if v > qp.h_lin[i] + tol {
            return false;
        }
    }
    let ae = dense(&qp.a_eq);
    for (i, row) in ae.iter().enumerate() {
        let v: f64 = row.iter().zip(x).map(|(a, b)| a * b).sum();
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
    let pairs = collect_pairs(qp);
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
        let relax = build_relax(qp, &pairs, &lb, &ub);
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
        for (pi, &(i, j)) in pairs.iter().enumerate() {
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
        status: if timed_out {
            SolveStatus::Timeout
        } else if limited {
            SolveStatus::MaxIterations
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
        let pairs = collect_pairs(&qp);
        assert_eq!(pairs, vec![(0, 0)]);
        let relax = build_relax(&qp, &pairs, &qp.lb, &qp.ub);
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
        let pairs = collect_pairs(&qp);
        let relax = build_relax(&qp, &pairs, &qp.lb, &qp.ub);
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
}
