//! Mixed-integer SOCP / QCQP via branch-and-bound over the conic relaxation.
//!
//! Branching adds integer variable bounds as nonnegative-orthant rows to the
//! relaxation `G x + s = h`, then re-solves with the SOCP interior-point method.

use super::qcqp::{to_conic, QcqpProblem};
use super::{solve_socp, ConeSpec, ConicOptions, ConicProblem};
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

/// A mixed-integer SOCP: a base [`ConicProblem`] plus integrality on selected
/// variables with explicit (finite) bounds used to bound the search.
#[derive(Debug, Clone)]
pub struct MisocpProblem {
    /// Continuous relaxation.
    pub base: ConicProblem,
    /// Indices of integer-constrained variables.
    pub integers: Vec<usize>,
    /// Lower bounds aligned with `integers`.
    pub int_lb: Vec<f64>,
    /// Upper bounds aligned with `integers`.
    pub int_ub: Vec<f64>,
}

/// Result of a mixed-integer conic solve.
#[derive(Debug, Clone)]
pub struct MisocpResult {
    /// Status (`Optimal`, `Infeasible`, or `MaxIterations` when node-limited).
    pub status: SolveStatus,
    /// Best objective found.
    pub objective: f64,
    /// Best solution.
    pub x: Vec<f64>,
    /// Branch-and-bound nodes processed.
    pub nodes: usize,
}

/// Branch-and-bound configuration.
#[derive(Debug, Clone)]
pub struct BbOptions {
    /// Integrality tolerance.
    pub int_tol: f64,
    /// Node limit.
    pub max_nodes: usize,
    /// Optimality gap tolerance for pruning.
    pub gap_tol: f64,
}

impl Default for BbOptions {
    fn default() -> Self {
        Self {
            int_tol: 1e-6,
            max_nodes: 20_000,
            gap_tol: 1e-6,
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

/// Build a relaxation with per-node integer bounds `[lb_j, ub_j]` appended as
/// orthant rows (kept contiguous before the SOC blocks).
fn build_relaxation(
    base: &ConicProblem,
    lb: &[f64],
    ub: &[f64],
    integers: &[usize],
) -> ConicProblem {
    let n = base.n();
    let gd = dense(&base.g);
    let l = base.cone.l;
    // Existing orthant rows [0..l), SOC rows [l..m).
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut h: Vec<f64> = Vec::new();
    for i in 0..l {
        rows.push(gd[i].clone());
        h.push(base.h[i]);
    }
    // Bound rows: x_j <= ub  and  -x_j <= -lb.
    for (k, &j) in integers.iter().enumerate() {
        let mut r = vec![0.0; n];
        r[j] = 1.0;
        rows.push(r);
        h.push(ub[k]);
        let mut r2 = vec![0.0; n];
        r2[j] = -1.0;
        rows.push(r2);
        h.push(-lb[k]);
    }
    let new_l = rows.len();
    for i in l..base.m() {
        rows.push(gd[i].clone());
        h.push(base.h[i]);
    }
    let m = rows.len();
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
    let g = CscMatrix::from_triplets(&ri, &ci, &vi, m, n).unwrap();
    ConicProblem {
        c: base.c.clone(),
        a: base.a.clone(),
        b: base.b.clone(),
        g,
        h,
        cone: ConeSpec {
            l: new_l,
            soc: base.cone.soc.clone(),
        },
    }
}

/// Solve a mixed-integer SOCP by branch-and-bound (depth-first, best-bound
/// pruning). Minimises `c^T x`.
pub fn solve_misocp(prob: &MisocpProblem, opts: &ConicOptions, bb: &BbOptions) -> MisocpResult {
    let mut incumbent_obj = f64::INFINITY;
    let mut incumbent_x: Vec<f64> = Vec::new();
    let mut nodes = 0usize;
    let mut node_limited = false;

    // Stack of (lb, ub) per integer.
    let mut stack: Vec<(Vec<f64>, Vec<f64>)> = vec![(prob.int_lb.clone(), prob.int_ub.clone())];

    while let Some((lb, ub)) = stack.pop() {
        if nodes >= bb.max_nodes {
            node_limited = true;
            break;
        }
        nodes += 1;
        let relax = build_relaxation(&prob.base, &lb, &ub, &prob.integers);
        let res = solve_socp(&relax, opts);
        match res.status {
            SolveStatus::Optimal => {}
            SolveStatus::Unbounded => {
                return MisocpResult {
                    status: SolveStatus::Unbounded,
                    objective: f64::NEG_INFINITY,
                    x: vec![],
                    nodes,
                };
            }
            _ => continue, // infeasible / numerical: prune
        }
        // Bound pruning.
        if res.objective >= incumbent_obj - bb.gap_tol {
            continue;
        }
        // Find most-fractional integer.
        let mut branch: Option<(usize, f64)> = None;
        let mut worst = bb.int_tol;
        for (k, &j) in prob.integers.iter().enumerate() {
            let v = res.x[j];
            let frac = v - v.floor();
            let dist = frac.min(1.0 - frac);
            if dist > worst {
                worst = dist;
                branch = Some((k, v));
            }
        }
        match branch {
            None => {
                // Integer-feasible: accept.
                if res.objective < incumbent_obj {
                    incumbent_obj = res.objective;
                    incumbent_x = res.x.clone();
                }
            }
            Some((k, v)) => {
                let fl = v.floor();
                let ce = v.ceil();
                // Down child: ub_k = floor(v).
                let lb_d = lb.clone();
                let mut ub_d = ub.clone();
                ub_d[k] = fl;
                if lb_d[k] <= ub_d[k] + 1e-9 {
                    stack.push((lb_d, ub_d));
                }
                // Up child: lb_k = ceil(v).
                let mut lb_u = lb.clone();
                let ub_u = ub.clone();
                lb_u[k] = ce;
                if lb_u[k] <= ub_u[k] + 1e-9 {
                    stack.push((lb_u, ub_u));
                }
            }
        }
    }

    if incumbent_x.is_empty() {
        return MisocpResult {
            status: SolveStatus::Infeasible,
            objective: f64::INFINITY,
            x: vec![],
            nodes,
        };
    }
    MisocpResult {
        status: if node_limited {
            SolveStatus::MaxIterations
        } else {
            SolveStatus::Optimal
        },
        objective: incumbent_obj,
        x: incumbent_x,
        nodes,
    }
}

/// Solve a mixed-integer QCQP: reformulate to a conic problem and run the
/// mixed-integer SOCP branch-and-bound. `integers` index the original QCQP
/// variables (`0..qp.n`); `lb`/`ub` are aligned finite bounds.
pub fn solve_miqcp(
    qp: &QcqpProblem,
    integers: &[usize],
    lb: &[f64],
    ub: &[f64],
    opts: &ConicOptions,
    bb: &BbOptions,
) -> MisocpResult {
    let (base, _nvar, _convexity_unproven) = match to_conic(qp) {
        Ok(v) => v,
        Err(e) => {
            return MisocpResult {
                status: SolveStatus::NotSupported(e),
                objective: f64::NAN,
                x: vec![],
                nodes: 0,
            }
        }
    };
    let mp = MisocpProblem {
        base,
        integers: integers.to_vec(),
        int_lb: lb.to_vec(),
        int_ub: ub.to_vec(),
    };
    let mut res = solve_misocp(&mp, opts, bb);
    // Truncate epigraph variable if present.
    if res.x.len() > qp.n {
        res.x.truncate(qp.n);
    }
    res
}
