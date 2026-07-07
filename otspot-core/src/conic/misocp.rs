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
///
/// `status` distinguishes a proven conclusion from an inconclusive search —
/// see [`solve_misocp`] for the full state table (no incumbent × {proven,
/// node-limited, timed out, numerically failed} and incumbent × {proven,
/// node-limited/failed, timed out}).
#[derive(Debug, Clone)]
pub struct MisocpResult {
    /// Solve status; see the type-level doc for the full state table.
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
    /// Wall-clock deadline for the branch-and-bound loop, checked once per
    /// node before the relaxation is solved. `None` disables the check
    /// (bounded only by `max_nodes`). Independent of [`ConicOptions::deadline`],
    /// which bounds each node's own interior-point iterations; callers should
    /// normally set both to the same instant.
    pub deadline: Option<std::time::Instant>,
}

impl Default for BbOptions {
    fn default() -> Self {
        Self {
            int_tol: 1e-6,
            max_nodes: 20_000,
            gap_tol: 1e-6,
            deadline: None,
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

/// Build a relaxation with per-node integer bounds `[lb_j, ub_j]`. Strict
/// bounds are appended as orthant rows (kept contiguous before the SOC
/// blocks); a fixed variable (`lb_j == ub_j`) becomes an equality row
/// `x_j = lb_j` instead — the row pair `x_j <= v`, `-x_j <= -v` has no
/// strictly feasible slack, which stalls the interior-point method.
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
    let ad = dense(&base.a);
    let mut eq_rows: Vec<Vec<f64>> = ad.to_vec();
    let mut b = base.b.clone();
    for (k, &j) in integers.iter().enumerate() {
        if lb[k] == ub[k] {
            // Fixed by branching: x_j = lb_k as an equality row.
            let mut r = vec![0.0; n];
            r[j] = 1.0;
            eq_rows.push(r);
            b.push(lb[k]);
            continue;
        }
        // Bound rows: x_j <= ub  and  -x_j <= -lb.
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
    let to_csc = |rows: &[Vec<f64>]| {
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
        CscMatrix::from_triplets(&ri, &ci, &vi, rows.len(), n).unwrap()
    };
    ConicProblem {
        c: base.c.clone(),
        a: to_csc(&eq_rows),
        b,
        g: to_csc(&rows),
        h,
        cone: ConeSpec {
            l: new_l,
            soc: base.cone.soc.clone(),
        },
    }
}

/// Solve a mixed-integer SOCP by branch-and-bound (depth-first, best-bound
/// pruning). Minimises `c^T x`.
///
/// Node outcomes are classified, not collapsed into one "prune" bucket:
/// `Infeasible` prunes only with a Farkas certificate (`infeas_cert`);
/// `Unbounded` propagates only with a verified improving ray
/// (`primal_ray` -- a restricted relaxation can only be unbounded if the
/// root is too); anything else (`NumericalError`, an unproven
/// `Infeasible`/`Unbounded`, `MaxIterations`, `NotSupported`) is a
/// **numerical failure**, pruned but counted so an empty search never
/// proves false infeasibility. `Timeout` on a node stops the whole search.
///
/// Final status without an incumbent: `Timeout` > `MaxIterations` >
/// `NumericalError` > `Infeasible` (every leaf infeasible or pruned).
/// With an incumbent: `Timeout` > `Optimal` (full exhaustion, **no**
/// numerical failures) > `SuboptimalSolution` (mirrors `mip::solve_miqp`).
pub fn solve_misocp(prob: &MisocpProblem, opts: &ConicOptions, bb: &BbOptions) -> MisocpResult {
    let mut incumbent_obj = f64::INFINITY;
    let mut incumbent_x: Vec<f64> = Vec::new();
    let mut nodes = 0usize;
    let mut node_limited = false;
    let mut timed_out = false;
    let mut numerical_failures = 0usize;

    // Stack of (lb, ub) per integer.
    let mut stack: Vec<(Vec<f64>, Vec<f64>)> = vec![(prob.int_lb.clone(), prob.int_ub.clone())];

    while let Some((lb, ub)) = stack.pop() {
        if bb.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            timed_out = true;
            break;
        }
        if nodes >= bb.max_nodes {
            node_limited = true;
            break;
        }
        nodes += 1;
        let relax = build_relaxation(&prob.base, &lb, &ub, &prob.integers);
        let res = solve_socp(&relax, opts);
        match res.status {
            SolveStatus::Optimal => {}
            SolveStatus::Unbounded if res.primal_ray.is_some() => {
                // Verified improving ray: a restricted relaxation can only be
                // unbounded if the root is too.
                return MisocpResult {
                    status: SolveStatus::Unbounded,
                    objective: f64::NEG_INFINITY,
                    x: vec![],
                    nodes,
                };
            }
            SolveStatus::Infeasible if res.infeas_cert.is_some() => continue, // proven: prune
            SolveStatus::Timeout => {
                timed_out = true;
                break;
            }
            _ => {
                // Unproven Infeasible, NumericalError, MaxIterations, NotSupported, …:
                // no certified conclusion. Prune (can't branch on an untrustworthy
                // point) but flag the search as inconclusive.
                numerical_failures += 1;
                continue;
            }
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

    let proven = !timed_out && !node_limited && numerical_failures == 0;

    if incumbent_x.is_empty() {
        let status = if timed_out {
            SolveStatus::Timeout
        } else if node_limited {
            SolveStatus::MaxIterations
        } else if numerical_failures > 0 {
            SolveStatus::NumericalError
        } else {
            debug_assert!(proven);
            SolveStatus::Infeasible
        };
        return MisocpResult {
            status,
            objective: f64::INFINITY,
            x: vec![],
            nodes,
        };
    }
    MisocpResult {
        status: if proven {
            SolveStatus::Optimal
        } else if timed_out {
            SolveStatus::Timeout
        } else {
            SolveStatus::SuboptimalSolution
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
