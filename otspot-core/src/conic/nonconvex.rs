//! Global optimisation of (possibly nonconvex) QCQP by spatial branch-and-bound
//! with McCormick relaxations of the bilinear/quadratic terms.
//!
//! Handles indefinite objective/constraint matrices. Requires finite variable
//! bounds `[lb, ub]` (needed for valid McCormick envelopes and termination).

use super::{solve_socp, ConeSpec, ConicOptions, ConicProblem};
use crate::problem::SolveStatus;
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
#[derive(Debug, Clone)]
pub struct GlobalResult {
    /// Status: `Optimal` (proven within gap), `MaxIterations` (node-limited with
    /// an incumbent), or `Infeasible`.
    pub status: SolveStatus,
    /// Best (incumbent) objective.
    pub objective: f64,
    /// Best solution.
    pub x: Vec<f64>,
    /// Nodes processed.
    pub nodes: usize,
    /// Optimality gap `incumbent - best_bound` at termination.
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
}

impl Default for GlobalOptions {
    fn default() -> Self {
        Self {
            gap_tol: 1e-5,
            bilinear_tol: 1e-6,
            feas_tol: 1e-6,
            max_nodes: 50_000,
        }
    }
}

fn dense(a: &CscMatrix) -> Vec<Vec<f64>> {
    let mut d = vec![vec![0.0; a.ncols()]; a.nrows()];
    let cp = a.col_ptr();
    let ri = a.row_ind();
    let va = a.values();
    for j in 0..a.ncols() {
        for k in cp[j]..cp[j + 1] {
            d[ri[k]][j] = va[k];
        }
    }
    d
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

struct Relax {
    prob: ConicProblem,
    npair: usize,
}

/// Build the McCormick LP relaxation for the current box `[lb, ub]`.
fn build_relax(
    qp: &NonconvexQcqp,
    pairs: &[(usize, usize)],
    pair_idx: &std::collections::HashMap<(usize, usize), usize>,
    lb: &[f64],
    ub: &[f64],
) -> Relax {
    let n = qp.n;
    let npair = pairs.len();
    let nv = n + npair;
    // objective coefficients on [x, w].
    let mut c = vec![0.0; nv];
    c[..n].copy_from_slice(&qp.q0);
    if let Some(p0) = &qp.p0 {
        let d = dense(p0);
        add_quad_coeffs(&d, n, pairs, pair_idx, &mut c);
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
        add_quad_coeffs(&d, n, pairs, pair_idx, &mut row);
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
    // McCormick envelopes for each pair.
    for (pi, &(i, j)) in pairs.iter().enumerate() {
        let wj = n + pi;
        let (xl_i, xu_i) = (lb[i], ub[i]);
        let (xl_j, xu_j) = (lb[j], ub[j]);
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
    let prob = ConicProblem {
        c,
        a,
        b: qp.b_eq.clone(),
        g,
        h,
        cone: ConeSpec { l: m, soc: vec![] },
    };
    Relax { prob, npair }
}

/// Add `(1/2) x^T P x` coefficients onto a linear row over `[x, w]`.
fn add_quad_coeffs(
    d: &[Vec<f64>],
    n: usize,
    pairs: &[(usize, usize)],
    pair_idx: &std::collections::HashMap<(usize, usize), usize>,
    row: &mut [f64],
) {
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
        let _ = pair_idx;
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
    let pairs = collect_pairs(qp);
    let mut pair_idx = std::collections::HashMap::new();
    for (pi, &pr) in pairs.iter().enumerate() {
        pair_idx.insert(pr, pi);
    }
    let mut incumbent = f64::INFINITY;
    let mut inc_x: Vec<f64> = Vec::new();
    let mut nodes = 0usize;
    let mut best_bound = f64::NEG_INFINITY;
    let mut limited = false;

    let mut stack = vec![(qp.lb.clone(), qp.ub.clone())];
    let mut frontier_min = f64::INFINITY;

    while let Some((lb, ub)) = stack.pop() {
        if nodes >= g.max_nodes {
            limited = true;
            break;
        }
        nodes += 1;
        let relax = build_relax(qp, &pairs, &pair_idx, &lb, &ub);
        let res = solve_socp(&relax.prob, opts);
        if res.status != SolveStatus::Optimal {
            continue;
        }
        let lower = res.objective;
        if lower >= incumbent - g.gap_tol {
            continue; // bound prune
        }
        frontier_min = frontier_min.min(lower);
        let x = res.x[..qp.n].to_vec();

        // Try incumbent from the relaxation point (project handled by feasibility).
        if feasible(qp, &x, g.feas_tol) {
            let ov = objective(qp, &x);
            if ov < incumbent {
                incumbent = ov;
                inc_x = x.clone();
            }
        }

        // Worst McCormick term gap.
        let mut worst_gap = 0.0;
        let mut branch_var = None;
        for (pi, &(i, j)) in pairs.iter().enumerate() {
            let wv = res.x[qp.n + pi];
            let gap = (wv - x[i] * x[j]).abs();
            if gap > worst_gap {
                worst_gap = gap;
                // branch on the wider-range variable of the pair.
                branch_var = Some(if (ub[i] - lb[i]) >= (ub[j] - lb[j]) {
                    i
                } else {
                    j
                });
            }
        }
        let _ = relax.npair;

        if worst_gap <= g.bilinear_tol {
            // Relaxation is tight here: x is (near) feasible for the true model.
            if feasible(qp, &x, g.feas_tol * 100.0) {
                let ov = objective(qp, &x);
                if ov < incumbent {
                    incumbent = ov;
                    inc_x = x.clone();
                }
            }
            continue;
        }

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
                stack.push((lb1, ub1));
            }
            if ub2[k] - lb2[k] > 1e-9 {
                stack.push((lb2, ub2));
            }
        }
    }

    best_bound = best_bound.max(frontier_min);
    if inc_x.is_empty() {
        return GlobalResult {
            status: SolveStatus::Infeasible,
            objective: f64::INFINITY,
            x: vec![],
            nodes,
            gap: f64::INFINITY,
        };
    }
    let gap = (incumbent - best_bound).abs();
    GlobalResult {
        status: if limited {
            SolveStatus::MaxIterations
        } else {
            SolveStatus::Optimal
        },
        objective: incumbent,
        x: inc_x,
        nodes,
        gap,
    }
}
