//! Mixed-integer SOCP / QCQP via branch-and-bound over the conic relaxation.
//!
//! Branching adds integer variable bounds as nonnegative-orthant rows to the
//! relaxation `G x + s = h`, then re-solves with the SOCP interior-point method.

use super::equil::Equilibrator;
use super::qcqp::{to_conic, QcqpProblem};
use super::{ipm, ConeSpec, ConicOptions, ConicProblem};
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

impl MisocpProblem {
    /// Validate `base` plus the integer branching data's consistency against
    /// it: `build_relaxation` indexes `int_lb`/`int_ub` by position and `base`
    /// by `integers[k]` with no bounds checks of its own, so a length
    /// mismatch or an out-of-range index previously indexed out of bounds and
    /// panicked (PR #25 review 38, 39) instead of failing as an ordinary
    /// invalid-input `NotSupported`.
    pub fn validate(&self) -> Result<(), String> {
        self.base.validate()?;
        if self.int_lb.len() != self.integers.len() {
            return Err(format!(
                "int_lb length {} != integers length {}",
                self.int_lb.len(),
                self.integers.len()
            ));
        }
        if self.int_ub.len() != self.integers.len() {
            return Err(format!(
                "int_ub length {} != integers length {}",
                self.int_ub.len(),
                self.integers.len()
            ));
        }
        let n = self.base.n();
        for (k, &j) in self.integers.iter().enumerate() {
            if j >= n {
                return Err(format!("integers[{k}] = {j} out of range (n = {n})"));
            }
        }
        for (k, &v) in self.int_lb.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("int_lb[{k}] is not finite: {v}"));
            }
        }
        for (k, &v) in self.int_ub.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("int_ub[{k}] is not finite: {v}"));
            }
        }
        Ok(())
    }
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

impl BbOptions {
    /// Validate option ranges the branch-and-bound loop assumes but never
    /// checks itself (PR #25 review, "Reject NaN integrality tolerances
    /// before branching"). A non-finite `int_tol` (in particular `NaN`)
    /// makes every `dist > worst` fractional-vs-integer comparison in
    /// `solve_misocp` silently `false` (`NaN` comparisons are never `true`),
    /// so the search never finds a branching variable and accepts *any*
    /// relaxation point -- however fractional -- as a proven integer-feasible
    /// incumbent.
    pub fn validate(&self) -> Result<(), String> {
        if !self.int_tol.is_finite() || self.int_tol < 0.0 {
            return Err(format!(
                "int_tol must be finite and >= 0, got {}",
                self.int_tol
            ));
        }
        if !self.gap_tol.is_finite() || self.gap_tol < 0.0 {
            return Err(format!(
                "gap_tol must be finite and >= 0, got {}",
                self.gap_tol
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only override of `solve_misocp`'s mid-loop deadline check
    /// (test-only, mirrors `nonconvex::RELAX_STATUS_PLAN`'s fault-injection
    /// pattern). `Some(n)` deterministically forces the deadline branch to
    /// fire once `nodes >= n`, regardless of `BbOptions::deadline` or
    /// wall-clock jitter; `None` (the default) leaves the real deadline check
    /// as the sole trigger. B&B is single-threaded and depends only on
    /// `prob`/`opts` (not on `bb.deadline`, which the node relaxations never
    /// see), so two runs with identical `prob`/`opts` visit the exact same
    /// node sequence -- this lets a test reproduce "deadline hit mid-search,
    /// incumbent kept" deterministically instead of racing a wall clock
    /// against CPU/OS jitter (see `misocp_mid_search_deadline_keeps_incumbent`).
    /// `thread_local` so parallel tests cannot corrupt each other's setting.
    pub(super) static DEADLINE_AFTER_NODE: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// RAII guard resetting `DEADLINE_AFTER_NODE` back to `None` on drop,
/// including on unwind (mirrors `ipm::ForceTimeoutGuard`; same manual-reset-
/// leaks-on-panic risk, since both are a `set(Some(..))` .. `set(None)` pair
/// bracketing a `solve_misocp` call in a test).
#[cfg(test)]
pub(super) struct DeadlineAfterNodeGuard;

#[cfg(test)]
impl DeadlineAfterNodeGuard {
    pub(super) fn new(after_node: usize) -> Self {
        DEADLINE_AFTER_NODE.with(|c| c.set(Some(after_node)));
        Self
    }
}

#[cfg(test)]
impl Drop for DeadlineAfterNodeGuard {
    fn drop(&mut self) {
        DEADLINE_AFTER_NODE.with(|c| c.set(None));
    }
}

#[cfg(test)]
thread_local! {
    /// Per-node relaxation-status fault injection (test-only, mirrors
    /// `nonconvex::RELAX_STATUS_PLAN`). Each visited node pops the front entry:
    /// `Some(status)` overrides the node relaxation's status and drops any
    /// certificate, so a plan entry of `Some(NumericalError)` deterministically
    /// makes that one node fail *without* a certificate (the `_ =>` arm counts
    /// it as a `numerical_failure`) regardless of the IPM's iteration numerics;
    /// `None` (or an exhausted queue) leaves the real solve untouched. This
    /// reproduces "an incumbent exists but one node failed inconclusively, so
    /// the search is not exhaustive" without calibrating a fragile per-node
    /// iteration budget. `thread_local` so parallel tests cannot corrupt each
    /// other's plan.
    pub(super) static NODE_STATUS_PLAN:
        std::cell::RefCell<std::collections::VecDeque<Option<SolveStatus>>> =
        const { std::cell::RefCell::new(std::collections::VecDeque::new()) };
}

/// Build a relaxation with per-node integer bounds `[lb_j, ub_j]`. Strict
/// bounds are appended as orthant rows (kept contiguous before the SOC
/// blocks); a fixed variable (`lb_j == ub_j`) becomes an equality row
/// `x_j = lb_j` instead — the row pair `x_j <= v`, `-x_j <= -v` has no
/// strictly feasible slack, which stalls the interior-point method.
///
/// Constructs the new `G`/`A` directly from `base`'s CSC nonzeros
/// (`O(nnz(base.g) + nnz(base.a) + border rows)`) instead of densifying
/// `base` to `Vec<Vec<f64>>` and re-extracting the nonzeros — every
/// branch-and-bound node calls this, so an `O(n*m)` dense pass here repeats
/// at every node and dominates B&B memory on any large MISOCP regardless of
/// how few integer variables are actually being branched on (see
/// `otspot-core/tests/memory_budget.rs`'s MISOCP route fence).
fn build_relaxation(
    base: &ConicProblem,
    lb: &[f64],
    ub: &[f64],
    integers: &[usize],
) -> ConicProblem {
    let n = base.n();
    let l = base.cone.l;
    let m = base.m();

    let mut n_free_borders = 0usize;
    let mut n_fixed = 0usize;
    for (k, _) in integers.iter().enumerate() {
        if lb[k] == ub[k] {
            n_fixed += 1;
        } else {
            n_free_borders += 1;
        }
    }
    let new_l = l + 2 * n_free_borders;
    let new_m = new_l + (m - l);

    // G: orthant rows [0, l) keep their row index; SOC rows [l, m) shift by
    // the number of inserted border rows; the border rows fill [l, new_l).
    let g_nnz = base.g.nnz();
    let mut g_ri = Vec::with_capacity(g_nnz + 2 * n_free_borders);
    let mut g_ci = Vec::with_capacity(g_nnz + 2 * n_free_borders);
    let mut g_vi = Vec::with_capacity(g_nnz + 2 * n_free_borders);
    let cp = base.g.col_ptr();
    let ri = base.g.row_ind();
    let va = base.g.values();
    for j in 0..n {
        for k in cp[j]..cp[j + 1] {
            let i = ri[k];
            let shifted = if i < l { i } else { i + (new_l - l) };
            g_ri.push(shifted);
            g_ci.push(j);
            g_vi.push(va[k]);
        }
    }
    let mut h = vec![0.0; new_m];
    h[..l].copy_from_slice(&base.h[..l]);
    h[new_l..new_m].copy_from_slice(&base.h[l..m]);
    let mut g_row = l;
    for (k, &j) in integers.iter().enumerate() {
        if lb[k] == ub[k] {
            continue;
        }
        // Bound rows: x_j <= ub  and  -x_j <= -lb.
        g_ri.push(g_row);
        g_ci.push(j);
        g_vi.push(1.0);
        h[g_row] = ub[k];
        g_row += 1;
        g_ri.push(g_row);
        g_ci.push(j);
        g_vi.push(-1.0);
        h[g_row] = -lb[k];
        g_row += 1;
    }
    assert_eq!(g_row, new_l);
    let g = CscMatrix::from_triplets(&g_ri, &g_ci, &g_vi, new_m, n).unwrap();

    // A: existing equality rows keep their row index; each fixed integer
    // appends one equality row `x_j = lb_k`.
    let p = base.p();
    let new_p = p + n_fixed;
    let a_nnz = base.a.nnz();
    let mut a_ri = Vec::with_capacity(a_nnz + n_fixed);
    let mut a_ci = Vec::with_capacity(a_nnz + n_fixed);
    let mut a_vi = Vec::with_capacity(a_nnz + n_fixed);
    let acp = base.a.col_ptr();
    let ari = base.a.row_ind();
    let ava = base.a.values();
    for j in 0..n {
        for k in acp[j]..acp[j + 1] {
            a_ri.push(ari[k]);
            a_ci.push(j);
            a_vi.push(ava[k]);
        }
    }
    let mut b = base.b.clone();
    b.reserve(n_fixed);
    let mut a_row = p;
    for (k, &j) in integers.iter().enumerate() {
        if lb[k] != ub[k] {
            continue;
        }
        // Fixed by branching: x_j = lb_k as an equality row.
        a_ri.push(a_row);
        a_ci.push(j);
        a_vi.push(1.0);
        b.push(lb[k]);
        a_row += 1;
    }
    assert_eq!(a_row, new_p);
    let a = CscMatrix::from_triplets(&a_ri, &a_ci, &a_vi, new_p, n).unwrap();

    ConicProblem {
        c: base.c.clone(),
        a,
        b,
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
///
/// Convexity caveat: [`MisocpProblem`] carries no `convexity_unproven` flag, so
/// this entry point cannot detect a base problem whose SOC blocks came from a
/// clamped (only-approximate) QCQP Cholesky. Both callers gate that upstream
/// ([`solve_miqcp`] and the Model layer refuse a clamped result before building
/// the `MisocpProblem`); a future direct caller must perform the same check.
///
/// Node outcomes are classified, not collapsed into one "prune" bucket:
/// `Infeasible` prunes only with a Farkas certificate (`infeas_cert`);
/// `Unbounded` with a verified improving ray (`primal_ray`) propagates only
/// once every integer variable is already fixed in this node (otherwise the
/// ray says nothing about integer-feasibility and the node is bisected
/// further instead); anything else (`NumericalError`, an unproven
/// `Infeasible`/`Unbounded`, `MaxIterations`, `NotSupported`) is a
/// **numerical failure**, pruned but counted so an empty search never
/// proves false infeasibility. `Timeout` on a node stops the whole search.
///
/// Final status without an incumbent: `Timeout` > `MaxIterations` >
/// `NumericalError` > `Infeasible`. With an incumbent: `Timeout` > `Optimal`
/// (full exhaustion, no numerical failures) > `SuboptimalSolution`.
pub fn solve_misocp(prob: &MisocpProblem, opts: &ConicOptions, bb: &BbOptions) -> MisocpResult {
    if let Err(e) = prob
        .validate()
        .and_then(|()| opts.validate())
        .and_then(|()| bb.validate())
    {
        return MisocpResult {
            status: SolveStatus::NotSupported(e),
            objective: f64::NAN,
            x: vec![],
            nodes: 0,
        };
    }
    let mut incumbent_obj = f64::INFINITY;
    let mut incumbent_x: Vec<f64> = Vec::new();
    let mut nodes = 0usize;
    let mut node_limited = false;
    let mut timed_out = false;
    let mut numerical_failures = 0usize;

    // Equilibrate `base` once for the whole tree instead of
    // per-node: every node shares the same column scale `d`, so a node's
    // bound/fixing rows are built directly in scaled space (`lb/d[j]`,
    // `ub/d[j]`) via the unmodified `build_relaxation`, and the O(sweeps *
    // nnz) Ruiz cost is paid once, not once per (potentially thousands of)
    // B&B nodes.
    let equil = Equilibrator::compute(&prob.base);
    let scaled_base = equil.scale_problem(&prob.base);

    // Stack of (lb, ub) per integer, kept in *original* units (integrality
    // and fractional branching are properties of `x_j`, not the scaled
    // `x_j / d[j]`).
    let mut stack: Vec<(Vec<f64>, Vec<f64>)> = vec![(prob.int_lb.clone(), prob.int_ub.clone())];

    while let Some((lb, ub)) = stack.pop() {
        let deadline_hit = bb.deadline.is_some_and(|d| std::time::Instant::now() >= d);
        // Test-only: force the same branch deterministically at a chosen
        // node count instead of a wall-clock instant (see `DEADLINE_AFTER_NODE`).
        #[cfg(test)]
        let deadline_hit =
            deadline_hit || DEADLINE_AFTER_NODE.with(|c| c.get().is_some_and(|n| nodes >= n));
        if deadline_hit {
            timed_out = true;
            break;
        }
        if nodes >= bb.max_nodes {
            node_limited = true;
            break;
        }
        nodes += 1;
        let lb_scaled: Vec<f64> = prob
            .integers
            .iter()
            .zip(&lb)
            .map(|(&j, &v)| v / equil.d[j])
            .collect();
        let ub_scaled: Vec<f64> = prob
            .integers
            .iter()
            .zip(&ub)
            .map(|(&j, &v)| v / equil.d[j])
            .collect();
        let relax = build_relaxation(&scaled_base, &lb_scaled, &ub_scaled, &prob.integers);
        // Internal call (bypasses `solve_socp`): no re-equilibration per
        // node, and `y`/`z`/`s` stay in scaled space since this loop only
        // ever reads `primal_ray`/`infeas_cert`'s *presence* (a proof of the
        // relaxation's infeasibility/unboundedness, true regardless of the
        // reparametrization -- feasibility is scale-invariant), never a
        // certificate's numeric value.
        // A branch-and-bound relaxation node is a tightly-bounded, low-degree
        // subproblem: the Mehrotra starting-point balancing that helps the
        // high-degree continuous root SOCP instead overshoots here and stalls
        // the last iterations at the precision floor (see `ipm::starting_point`),
        // so the node is solved from the plain data-driven start (`balance =
        // false`).
        let mut res = ipm::solve(&relax, opts, false);
        // Test-only: force this node's status (dropping certificates) to inject
        // a deterministic, non-certificate node failure (see `NODE_STATUS_PLAN`).
        #[cfg(test)]
        if let Some(forced) = NODE_STATUS_PLAN
            .with(|p| p.borrow_mut().pop_front())
            .flatten()
        {
            res.status = forced;
            res.infeas_cert = None;
            res.primal_ray = None;
        }
        for (xi, &dj) in res.x.iter_mut().zip(&equil.d) {
            *xi *= dj;
        }
        res.objective = prob.base.c.iter().zip(&res.x).map(|(a, b)| a * b).sum();
        match res.status {
            SolveStatus::Optimal => {}
            SolveStatus::Unbounded if res.primal_ray.is_some() => {
                // A verified ray only certifies the *relaxation*'s recession
                // cone holds a `c^T d < 0` direction `d` whose component on every
                // integer variable is ~0 (node box/fixing rows), so it extends
                // any feasible node point to -infinity — but says nothing about
                // whether the node's polytope holds an integer-feasible point:
                // an equality elsewhere can pin an "integer" variable to a
                // fractional value while the integrality-blind relaxation stays
                // unbounded. Only once every integer variable is fixed
                // (`lb[k] == ub[k]`) does any feasible node point carry integer
                // values, certifying the MI problem unbounded. Otherwise bisect
                // the widest free integer (finite `int_lb`/`int_ub` via
                // `validate`, terminating) and fall to `Infeasible` if leaves empty.
                if let Some(k) = (0..lb.len()).find(|&k| lb[k] != ub[k]) {
                    let mid = ((lb[k] + ub[k]) / 2.0).floor();
                    let mut ub_d = ub.clone();
                    ub_d[k] = mid;
                    if lb[k] <= ub_d[k] + 1e-9 {
                        stack.push((lb.clone(), ub_d));
                    }
                    let mut lb_u = lb.clone();
                    lb_u[k] = mid + 1.0;
                    if lb_u[k] <= ub[k] + 1e-9 {
                        stack.push((lb_u, ub.clone()));
                    }
                    continue;
                }
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
    let (base, _nvar, convexity_unproven) = match to_conic(qp) {
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
    // Branch-and-bound soundness requires every node's conic relaxation to be
    // a *valid* relaxation of the QCQP: bounds and pruning all rest on it. When
    // `to_conic` clamped a negative jitter-band pivot (`convexity_unproven`),
    // the reformulation is only approximate, so every node's dual bound is
    // untrustworthy and no prune/incumbent-bound is sound. There is no
    // mixed-integer spatial fallback wired here, so refuse rather than certify
    // an unproven bound — matching the continuous route's
    // `is_clean_convex_outcome` gate and the SOC-constrained Model path, both
    // of which reject `convexity_unproven` for the same reason.
    if convexity_unproven {
        return MisocpResult {
            status: SolveStatus::NotSupported(
                "MIQCP has an indefinite (Cholesky jitter-band) quadratic; the \
                 conic relaxation is only approximate, so branch-and-bound bounds \
                 are not sound"
                    .to_string(),
            ),
            objective: f64::NAN,
            x: vec![],
            nodes: 0,
        };
    }
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
    // Recompute the true QCQP objective from `x` rather than trusting the
    // conic relaxation's `objective` (PR #25 review 29): when the objective
    // is quadratic, `to_conic` minimizes an epigraph variable `t` bounding
    // `(1/2) x^T P0 x + q0^T x`, and the reported conic objective is that `t`,
    // not the caller's literal `P0`/`q0` evaluated at `x`. On a node-limited
    // or timed-out incumbent the two can differ, so recomputing from `x` — as
    // the continuous `solve_qcqp` entry point already does — keeps the MISOCP
    // and continuous/MIQCP-objective conventions aligned. (Clamped/`unproven`
    // reformulations are refused above, so `t` here is always an exact-PSD
    // epigraph bound rather than a clamped-curvature one.)
    if !res.x.is_empty() {
        res.objective = qcqp_true_objective(qp, &res.x);
    }
    res
}

/// Independent recomputation of `(1/2) x^T P0 x + q0^T x` from the caller's
/// literal `QcqpProblem` data (not the conic epigraph variable -- see
/// `solve_miqcp`).
fn qcqp_true_objective(qp: &QcqpProblem, x: &[f64]) -> f64 {
    let mut obj: f64 = qp.q0.iter().zip(x).map(|(q, xi)| q * xi).sum();
    if let Some(p0) = &qp.p0 {
        let px = p0.mat_vec_mul(x).expect(
            "p0.ncols() == x.len() == qp.n: guaranteed by to_conic's validate_dims, \
             which solve_miqcp() always runs, giving p0.ncols() == qp.n. \
             solve_misocp's x is always either empty or >= qp.n long (the SOCP \
             relaxation retains at least the qp.n original columns); the \
             is_empty() guard at the call site (solve_miqcp) skips the empty \
             case, and its conditional truncate (only fires when len > qp.n) \
             brings any longer x down to exactly qp.n",
        );
        obj += 0.5 * x.iter().zip(&px).map(|(xi, pxi)| xi * pxi).sum::<f64>();
    }
    obj
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `p0` is 3x3 but `qp.n`/`x.len()` is 2, bypassing `to_conic`'s
    /// `validate_dims`. Must panic, not silently zero-fill; reverting to the
    /// old zero-fill-on-Err fallback makes this FAIL.
    #[test]
    #[should_panic(expected = "p0.ncols() == x.len() == qp.n")]
    fn qcqp_true_objective_panics_on_dimension_mismatch() {
        let qp = QcqpProblem {
            n: 2,
            p0: Some(
                CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1.0], 3, 3).unwrap(),
            ),
            q0: vec![0.0, 0.0],
            quad: vec![],
            g_lin: CscMatrix::new(0, 2),
            h_lin: vec![],
            a_eq: CscMatrix::new(0, 2),
            b_eq: vec![],
        };
        let x = vec![1.0, 2.0];
        let _ = qcqp_true_objective(&qp, &x);
    }

    /// Pre-fix reference: densify `base.g`/`base.a` to `Vec<Vec<f64>>`, build
    /// the bound/fixing rows as dense row vectors, then re-extract nonzeros
    /// into a fresh `CscMatrix`. This is the exact algorithm `build_relaxation`
    /// used before the sparse rewrite (O(n*m) per B&B node); it is kept here,
    /// self-contained, purely as an independent oracle for
    /// `build_relaxation_matches_dense_reference` below -- production code
    /// must never call this again.
    fn reference_dense_build_relaxation(
        base: &ConicProblem,
        lb: &[f64],
        ub: &[f64],
        integers: &[usize],
    ) -> ConicProblem {
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

        let n = base.n();
        let gd = dense(&base.g);
        let l = base.cone.l;
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
                let mut r = vec![0.0; n];
                r[j] = 1.0;
                eq_rows.push(r);
                b.push(lb[k]);
                continue;
            }
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

    fn assert_csc_eq(a: &CscMatrix, b: &CscMatrix, label: &str) {
        assert_eq!(a.nrows(), b.nrows(), "{label}: nrows");
        assert_eq!(a.ncols(), b.ncols(), "{label}: ncols");
        assert_eq!(a.col_ptr(), b.col_ptr(), "{label}: col_ptr");
        assert_eq!(a.row_ind(), b.row_ind(), "{label}: row_ind");
        assert_eq!(a.values(), b.values(), "{label}: values");
    }

    fn assert_relaxation_eq(sparse: &ConicProblem, reference: &ConicProblem, label: &str) {
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

    /// Sentinel (independent-oracle equivalence): the sparse `build_relaxation`
    /// must construct byte-for-byte the same `ConicProblem` as the pre-fix
    /// dense reference, across a spread of `(l, soc, p)` shapes and
    /// free/fixed/mixed branching patterns. Reintroducing any row-index or
    /// column-index mistake in the sparse rewrite (e.g. forgetting to shift
    /// the SOC block by the number of inserted border rows) changes `g`/`a`'s
    /// nonzero pattern here and fails this test without needing a full SOCP
    /// solve.
    #[test]
    fn build_relaxation_matches_dense_reference() {
        struct Case {
            name: &'static str,
            base: ConicProblem,
            lb: Vec<f64>,
            ub: Vec<f64>,
            integers: Vec<usize>,
        }
        let cases = vec![
            Case {
                name: "no_integers",
                base: ConicProblem {
                    c: vec![1.0, -1.0],
                    a: csc(&[vec![1.0, 1.0]], 1, 2),
                    b: vec![2.0],
                    g: csc(&[vec![0.0, 0.0], vec![-1.0, 0.0], vec![0.0, -1.0]], 3, 2),
                    h: vec![1.0, 0.0, 0.0],
                    cone: ConeSpec { l: 0, soc: vec![3] },
                },
                lb: vec![],
                ub: vec![],
                integers: vec![],
            },
            Case {
                name: "free_borders_only_l_zero",
                base: ConicProblem {
                    c: vec![-1.0, -1.0],
                    a: CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap(),
                    b: vec![],
                    g: csc(&[vec![0.0, 0.0], vec![-1.0, 0.0], vec![0.0, -1.0]], 3, 2),
                    h: vec![2.0, 0.0, 0.0],
                    cone: ConeSpec { l: 0, soc: vec![3] },
                },
                lb: vec![0.0, 0.0],
                ub: vec![2.0, 2.0],
                integers: vec![0, 1],
            },
            Case {
                name: "free_borders_with_existing_l",
                base: ConicProblem {
                    c: vec![1.0, 0.5, -2.0],
                    a: csc(&[vec![1.0, 1.0, 1.0]], 1, 3),
                    b: vec![3.0],
                    g: csc(
                        &[
                            vec![1.0, 0.0, 0.0],
                            vec![0.0, 0.0, 0.0],
                            vec![-1.0, 0.0, 0.0],
                            vec![0.0, -1.0, 0.0],
                        ],
                        4,
                        3,
                    ),
                    h: vec![5.0, 1.0, 0.0, 0.0],
                    cone: ConeSpec { l: 1, soc: vec![3] },
                },
                lb: vec![0.0, -1.0],
                ub: vec![4.0, 4.0],
                integers: vec![0, 2],
            },
            Case {
                name: "fixed_only",
                base: ConicProblem {
                    c: vec![1.0],
                    a: CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap(),
                    b: vec![],
                    g: csc(&[vec![1.0]], 1, 1),
                    h: vec![2.0e10],
                    cone: ConeSpec { l: 1, soc: vec![] },
                },
                lb: vec![1.0e10],
                ub: vec![1.0e10],
                integers: vec![0],
            },
            Case {
                name: "mixed_fixed_and_free_with_prior_equalities",
                base: ConicProblem {
                    c: vec![1.0, 2.0, 3.0, 4.0],
                    a: csc(&[vec![1.0, 0.0, 1.0, 0.0], vec![0.0, 1.0, 0.0, 1.0]], 2, 4),
                    b: vec![1.0, 2.0],
                    g: csc(
                        &[
                            vec![1.0, 0.0, 0.0, 0.0],
                            vec![0.0, 0.0, 0.0, 0.0],
                            vec![-1.0, 0.0, 0.0, 0.0],
                            vec![0.0, -1.0, 0.0, 0.0],
                            vec![0.0, 0.0, -1.0, 0.0],
                        ],
                        5,
                        4,
                    ),
                    h: vec![10.0, 0.0, 0.0, 0.0, 0.0],
                    cone: ConeSpec { l: 1, soc: vec![4] },
                },
                lb: vec![0.0, 3.0, -5.0],
                ub: vec![7.0, 3.0, 5.0],
                integers: vec![0, 1, 3],
            },
        ];

        for case in &cases {
            let sparse = build_relaxation(&case.base, &case.lb, &case.ub, &case.integers);
            let reference =
                reference_dense_build_relaxation(&case.base, &case.lb, &case.ub, &case.integers);
            assert_relaxation_eq(&sparse, &reference, case.name);
        }
    }
}
