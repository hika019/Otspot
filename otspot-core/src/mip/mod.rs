//! Mixed-integer programming (MILP / MIQP) via branch-and-bound.
//!
//! MILP (LP relaxation) and convex MIQP (QP relaxation) share one generic driver
//! (`solve_mip_with_stats`); the per-node solver is abstracted by `Relaxation`.
//! Pruning (`qp::global::pruning`) is reused from the spatial QP B&B. Non-convex
//! MIQP is out of scope and reported as [`SolveStatus::NonConvex`].

pub(crate) mod branch;
pub(crate) mod conflict;
pub(crate) mod cut_pool;
pub(crate) mod cuts;
pub(crate) mod effort;
pub(crate) mod heuristics;
pub(crate) mod node;
pub(crate) mod parallel;
pub(crate) mod presolve;
mod problem;
pub(crate) mod queue;
pub(crate) mod stats;
pub(crate) mod symmetry;

pub use problem::{MilpProblem, MipProblemError, MiqpProblem};
pub use stats::MipStats;

use crate::options::{MipBranching, MipConfig, SolverOptions, WarmStartBasis};
use crate::problem::certificate::BoundGapCertificate;
use crate::problem::{ConstraintType, SolveStatus, SolverResult, TimingBreakdown};
use crate::qp::global::pruning::{should_prune, within_gap};
use otspot_num::linalg::timeout::deadline_reached;
use otspot_num::sparse::CscMatrix;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use branch::{
    branch_bounds, is_integer_feasible, select_branching_variable,
    select_branching_variable_reliability, split_integer_box, strong_branch_candidates,
    widest_splittable_integer, PseudocostState,
};
use node::MipNode;
use queue::{NodeQueue, DIVE_FREQUENCY, DIVE_FREQUENCY_NO_INCUMBENT, MAX_DIVE_DEPTH};

/// A continuous relaxation the MIP branch-and-bound driver can solve over
/// arbitrary variable bounds. MILP uses an LP relaxation; convex MIQP uses a
/// QP one. Branching tightens the bounds, so the same driver works for both —
/// only the relaxation solver differs.
pub(crate) trait Relaxation {
    fn num_vars(&self) -> usize;
    fn root_bounds(&self) -> &[(f64, f64)];
    fn integer_vars(&self) -> &[usize];
    /// Solve the relaxation with `bounds` substituted for the original bounds.
    /// `opts` already has multistart / global_optimization stripped and the
    /// deadline fixed by the driver.
    fn solve(&self, bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult;
    /// Whether the driver should disable presolve on every B&B *node* solve.
    /// True for MILP: each node re-solves the same LP with only bounds tightened,
    /// so per-node presolve is redundant and its variable renumbering drops the
    /// propagated warm-start basis. False (default) for MIQP, whose IPM relies on
    /// presolve's Ruiz scaling for per-node conditioning.
    fn skip_node_presolve(&self) -> bool {
        false
    }
    /// Whether repeated LP Ruiz scaling can be skipped on root-child relaxations
    /// whose matrix/objective are unchanged and only bounds differ.
    fn can_skip_repeated_lp_scaling(&self) -> bool {
        false
    }
    /// Return constraint data for per-node bound propagation, or `None` to skip it.
    ///
    /// MILP returns `Some((&A, &b, &constraint_types))` so the B&B driver can call
    /// [`presolve::tighten_bounds_at_node`] before each LP solve. MIQP returns `None`
    /// because it relies on per-node Ruiz scaling (presolve) for conditioning instead.
    fn propagation_data(&self) -> Option<(&CscMatrix, &[f64], &[ConstraintType])> {
        None
    }

    /// In-tree cut separation hook. Default: no-op (returns `(None, 0, false)`).
    ///
    /// MILP overrides this to re-separate GMI/MIR from the node LP relaxation and
    /// return a cut-tightened result when the node bound improves. Cuts bake in
    /// the node's branching-tightened bounds, so they are valid only within this
    /// node's subtree: separation is **node-local** (a fresh pool per call, never
    /// reused at other nodes) and the cut rows are not propagated to children.
    /// `bounds` are the node's bounds; `res` is its (Optimal) relaxation result.
    /// `max_iters` bounds the simplex iterations this attempt may spend (see
    /// `effort::separation_iter_budget`); a round is skipped once the
    /// remaining allowance drops below the per-dimension useful minimum (see
    /// `cuts::separate_tree_cuts`). The first `u64` in the return is the
    /// total *real* simplex iterations actually spent across all rounds of
    /// this attempt; the second is [`cuts::tree_cut_construction_surcharge`]'s
    /// fixed-cost overhead accrued the same way — kept separate so the
    /// caller can route it to [`MipStats::tree_cut_overhead_iters`] rather
    /// than [`MipStats::tree_cut_iters`] (see that field's doc for why
    /// merging them regressed unrelated gates). Both are reported whether or
    /// not a cut was accepted, so the caller can charge the gate even on a
    /// dry attempt. The `bool` is whether this call actually attempted
    /// separation (passed the node-selection interval) — P3-B: kept explicit
    /// rather than inferred from the iteration counts, since a real attempt
    /// whose LP solves all happen to need zero simplex iterations (e.g. an
    /// already-optimal starting basis) would otherwise be indistinguishable
    /// from a skipped one, silently miscounting the dry-streak backoff.
    fn separate_tree_cuts(
        &self,
        _bounds: &[(f64, f64)],
        _res: &SolverResult,
        _mask: &[bool],
        _opts: &SolverOptions,
        _depth: usize,
        _node_index: usize,
        _max_iters: u64,
    ) -> (Option<SolverResult>, u64, u64, bool) {
        (None, 0, 0, false)
    }

    /// Run the RINS heuristic: fix integer variables where the LP relaxation and
    /// the incumbent agree and solve a sub-MIP over the remaining variables.
    /// Returns `(None, 0, 0)` for MIQP (default) or when RINS is disabled/skipped.
    /// The first `u64` is the sub-MIP's `nodes_processed`; the second is the
    /// sub-MIP's own recursive `total_simplex_iters` (see `effort`). Both are
    /// reported whenever a sub-MIP solve was actually attempted (independent
    /// of whether the result was usable) so hidden sub-MIP work is never lost.
    ///
    /// `iter_budget` is RINS's own remaining share of `effort::
    /// total_simplex_iters` (`effort::rins_iter_budget`), computed by the
    /// caller *after* `effort::may_run_rins` approves the call — the
    /// implementation caps the sub-MIP's own `MipConfig::max_lp_iters` at
    /// `min(heuristics::SUB_MIP_MAX_LP_ITERS, iter_budget)` (Codex review,
    /// P1: approval alone did not cap the size of the approved work), or
    /// skips the call outright when `iter_budget` is below
    /// `heuristics::SUB_MIP_MIN_LP_ITERS` (markshare_4_0 regression fix: a
    /// truncated call still pays full per-call setup overhead for a sub-MIP
    /// too small to search anything useful).
    fn run_rins(
        &self,
        _x_lp: &[f64],
        _x_inc: &[f64],
        _cfg: &MipConfig,
        _deadline: &Option<std::time::Instant>,
        _iter_budget: u64,
        _opts: &SolverOptions,
    ) -> (Option<SolverResult>, u64, u64) {
        (None, 0, 0)
    }

    /// Run the RENS heuristic: round a node LP relaxation by fixing integral
    /// components and restricting fractional ones to `{floor, ceil}`, then solve
    /// the small sub-MIP. Returns `(None, 0, 0)` for MIQP (default) or when
    /// disabled/skipped. See [`Relaxation::run_rins`] for the `u64` meanings
    /// and the `iter_budget` contract.
    fn run_rens(
        &self,
        _x_lp: &[f64],
        _cfg: &MipConfig,
        _deadline: &Option<std::time::Instant>,
        _iter_budget: u64,
        _opts: &SolverOptions,
    ) -> (Option<SolverResult>, u64, u64) {
        (None, 0, 0)
    }

    /// Run the local-branching heuristic: add a Hamming-distance ≤ k cut on the
    /// binary variables around the incumbent and solve the neighborhood sub-MIP.
    /// Returns `(None, 0, 0)` for MIQP (default) or when disabled/skipped. See
    /// [`Relaxation::run_rins`] for the `u64` meanings and the `iter_budget`
    /// contract.
    fn run_local_branching(
        &self,
        _x_inc: &[f64],
        _cfg: &MipConfig,
        _deadline: &Option<std::time::Instant>,
        _iter_budget: u64,
        _opts: &SolverOptions,
    ) -> (Option<SolverResult>, u64, u64) {
        (None, 0, 0)
    }
}

// Test-only observability: on a trivial root problem (few/no constraints),
// `tighten_bounds_with_probing` / `symmetry::break_symmetry` can genuinely
// complete in under 1us, at which point `Duration::as_micros()` truncation
// makes `root_probing_us`/`root_symmetry_us` read back as `0` — indistinguishable
// from "never populated". Racing real operation speed against that 1us
// truncation floor is what made `root_probing_and_symmetry_us_are_populated`
// flaky (see its doc). This flag makes the two measured spans below spin
// until a small, fixed floor has elapsed, deterministically clearing the
// truncation boundary regardless of machine speed or scheduler jitter.
// `#[cfg(test)]`-only, zero footprint in production builds.
#[cfg(test)]
thread_local! {
    static FORCE_MIN_ROOT_TIMING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Floor (in microseconds) `test_force_min_elapsed` spins past when
/// `FORCE_MIN_ROOT_TIMING` is set. Comfortably above the 1us truncation
/// boundary so the forced floor can never itself land exactly on it.
#[cfg(test)]
const FORCED_MIN_ELAPSED_US: u128 = 5;

#[cfg(test)]
fn test_force_min_elapsed(t0: Instant) {
    if FORCE_MIN_ROOT_TIMING.with(std::cell::Cell::get) {
        while t0.elapsed().as_micros() < FORCED_MIN_ELAPSED_US {
            std::hint::spin_loop();
        }
    }
}

#[cfg(not(test))]
#[inline(always)]
fn test_force_min_elapsed(_: Instant) {}

/// Solve a MILP to (relative) ε-optimality via branch-and-bound.
pub fn solve_milp(problem: &MilpProblem, options: &SolverOptions, cfg: &MipConfig) -> SolverResult {
    solve_milp_with_stats(problem, options, cfg).0
}

/// Like [`solve_milp`] but also returns search statistics (test sentinel hook).
///
/// Returns `(NumericalError, default stats)` immediately if `options` fails
/// validation (invalid tolerance, zero threads, etc.).
pub fn solve_milp_with_stats(
    problem: &MilpProblem,
    options: &SolverOptions,
    cfg: &MipConfig,
) -> (SolverResult, MipStats) {
    if options.validate().is_err() {
        return (SolverResult::numerical_error(), MipStats::default());
    }
    if let Err(e) = validate_integer_vars(&problem.integer_vars, problem.lp.num_vars) {
        return (
            SolverResult::not_supported(e.to_string()),
            MipStats::default(),
        );
    }
    // Establish a shared deadline before FP so that FP and B&B draw from the same
    // budget.  Without this, each LP in FP gets a fresh `timeout_secs` window and
    // `solve_mip_core` resets the clock again — allowing up to (MAX_FP_ITER + 1)×
    // timeout consumption.  If the caller already set an explicit deadline, honour it.
    let deadline = options.deadline.or_else(|| {
        options
            .timeout_secs
            .map(|s| Instant::now() + Duration::from_secs_f64(s))
    });
    let mut opts_with_dl = options.clone();
    opts_with_dl.deadline = deadline;
    opts_with_dl.timeout_secs = None;

    // MILP-specific root presolve: coefficient propagation tightens integer bounds.
    // Presolve is skipped when there are no integer variables (pure LP fallback is
    // handled inside the generic driver). Non-empty integer_vars with infeasible
    // integer rounding return early here before entering the B&B.
    if !problem.integer_vars.is_empty() {
        let mask = integer_mask(problem.lp.num_vars, &problem.integer_vars);
        // Root presolve: multi-pass propagation + probing tightens integer bounds.
        let mut tightened = problem.lp.bounds.clone();
        let presolve_t0 = std::time::Instant::now();
        let presolve_ok = presolve::tighten_bounds_with_probing(
            &problem.lp.a,
            &problem.lp.b,
            &problem.lp.constraint_types,
            &mut tightened,
            &problem.integer_vars,
            deadline,
        );
        test_force_min_elapsed(presolve_t0);
        let presolve_elapsed = presolve_t0.elapsed();
        let presolve_ms = presolve_elapsed.as_secs_f64() * 1000.0;
        let presolve_us = presolve_elapsed.as_micros().min(u128::from(u64::MAX)) as u64;
        let problem_bt: MilpProblem = match presolve_ok {
            None => {
                // Infeasibility detected at presolve. Report the presolve time as
                // relaxation_time_infeasible_ms so callers see a nonzero infeasibility cost,
                // and as root_probing_us (Codex review, P2) so this early return's cost is
                // attributed the same way as the non-infeasible path below (`stats.root_
                // probing_us = presolve_us;`) — without it, `milp_solve`'s `attribution_
                // covered_us_root_inclusive` silently undercounts whenever probing proves
                // infeasibility and takes non-negligible time.
                let stats = MipStats {
                    relaxation_time_infeasible_ms: presolve_ms,
                    root_probing_us: presolve_us,
                    ..Default::default()
                };
                return (SolverResult::infeasible(), stats);
            }
            Some(_) if tightened != problem.lp.bounds => {
                let mut lp_bt = problem.lp.clone();
                lp_bt.bounds = tightened;
                MilpProblem {
                    lp: lp_bt,
                    integer_vars: problem.integer_vars.clone(),
                }
            }
            Some(_) => problem.clone(),
        };
        // Static symmetry breaking: append lex-leader ordering rows for orbits
        // of interchangeable binary variables. The rows preserve at least one
        // optimal representative per orbit (objective unchanged) while shrinking
        // the search tree, so the whole downstream pipeline (FP, cuts, B&B) may
        // operate on the reduced symmetric space.
        let symmetry_t0 = Instant::now();
        let problem_bt = if cfg.symmetry {
            symmetry::break_symmetry(&problem_bt)
        } else {
            problem_bt
        };
        test_force_min_elapsed(symmetry_t0);
        let symmetry_us = symmetry_t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        // Run the feasibility pump on the original (bound-tightened) LP before
        // augmenting with cuts.  FP must see the unmodified constraint structure
        // so that the LP pump LPs and the final validation both use the original
        // bounds and Le/Ge rows, not the GMI cut rows added below.
        let fp_t0 = Instant::now();
        let fp_inc = heuristics::feasibility_pump::run_feasibility_pump(
            &problem_bt.lp,
            &problem_bt.integer_vars,
            cfg.integer_feas_tol,
            &opts_with_dl,
        );
        let fp_us = fp_t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        // Root GMI cuts tighten the LP relaxation without removing any
        // integer-feasible point, so the optimum is unchanged while the tree
        // shrinks. The added rows leave `num_vars` (hence `mask`) untouched.
        let cut_t0 = Instant::now();
        let effective = if cfg.cuts {
            cuts::add_root_cuts(&problem_bt, &opts_with_dl, cfg)
        } else {
            problem_bt
        };
        let root_cut_us = cut_t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        let (res, mut stats) = solve_mip_dispatch(&effective, &opts_with_dl, cfg, mask, fp_inc);
        stats.fp_us = fp_us;
        stats.root_cut_us = root_cut_us;
        stats.root_probing_us = presolve_us;
        stats.root_symmetry_us = symmetry_us;
        return (res, stats);
    }
    solve_mip_with_stats(problem, &opts_with_dl, cfg)
}

/// Solve a **convex** MIQP to (relative) ε-optimality via branch-and-bound.
///
/// Each node solves a convex QP relaxation (IP-PMM). A non-PSD `Q` (non-convex
/// MIQP) is out of scope: the QP relaxation would not be a valid lower bound, so
/// the solver returns [`SolveStatus::NonConvex`] rather than a silently wrong
/// answer. Use `solve_qp_global` for non-convex continuous QP.
pub fn solve_miqp(problem: &MiqpProblem, options: &SolverOptions, cfg: &MipConfig) -> SolverResult {
    solve_miqp_with_stats(problem, options, cfg).0
}

/// Like [`solve_miqp`] but also returns search statistics (test sentinel hook).
///
/// Returns `(NumericalError, default stats)` immediately if `options` fails
/// validation (invalid tolerance, zero threads, etc.).
pub fn solve_miqp_with_stats(
    problem: &MiqpProblem,
    options: &SolverOptions,
    cfg: &MipConfig,
) -> (SolverResult, MipStats) {
    if options.validate().is_err() {
        return (SolverResult::numerical_error(), MipStats::default());
    }
    // Central structural check: `solve_fixed_point` (the all-integer-fixed
    // B&B leaf) indexes `quadratic_constraints[k]` for `k < num_constraints`,
    // so a non-empty vector shorter than `num_constraints` (direct
    // public-field assignment on `problem.qp`, bypassing the setter) would
    // panic. `QpProblem::validate` is the shared source of this invariant.
    if problem.qp.validate().is_err() {
        return (SolverResult::numerical_error(), MipStats::default());
    }
    if let Err(e) = validate_integer_vars(&problem.integer_vars, problem.qp.num_vars) {
        return (
            SolverResult::not_supported(e.to_string()),
            MipStats::default(),
        );
    }
    if !problem.is_convex() {
        return (nonconvex_result(), MipStats::default());
    }
    // MIQP root presolve: bound tightening via coefficient propagation.
    // `tighten_bounds_linear` ignores Q (quadratic term) and operates on the
    // same linear constraints as MILP, so it is valid for convex MIQP too.
    if !problem.integer_vars.is_empty() {
        let n = problem.qp.num_vars;
        let mask = integer_mask(n, &problem.integer_vars);
        match presolve::tighten_bounds_linear(
            n,
            &problem.qp.a,
            &problem.qp.b,
            &problem.qp.constraint_types,
            &problem.qp.bounds,
            &mask,
        ) {
            None => return (SolverResult::infeasible(), MipStats::default()),
            Some(tightened) if tightened != problem.qp.bounds => {
                let mut qp_bt = problem.qp.clone();
                qp_bt.bounds = tightened;
                let problem_bt = MiqpProblem {
                    qp: qp_bt,
                    integer_vars: problem.integer_vars.clone(),
                };
                return solve_mip_core(&problem_bt, options, cfg, mask, None);
            }
            Some(_) => {
                return solve_mip_core(problem, options, cfg, mask, None);
            }
        }
    }
    solve_mip_with_stats(problem, options, cfg)
}

/// Generic branch-and-bound driver shared by MILP (LP relaxation) and convex
/// MIQP (QP relaxation). The only difference between the two is the relaxation
/// solver, abstracted by [`Relaxation`].
fn solve_mip_with_stats<R: Relaxation>(
    problem: &R,
    options: &SolverOptions,
    cfg: &MipConfig,
) -> (SolverResult, MipStats) {
    let mask = integer_mask(problem.num_vars(), problem.integer_vars());
    solve_mip_core(problem, options, cfg, mask, None)
}

/// Solve candidate LPs for strong branching and return per-variable scores.
///
/// For each candidate variable `j`, solves down and up child LPs, records the
/// objective improvements as pseudocost observations, and returns the combined
/// branching score keyed by `j`.  Candidates that fail to solve to Optimal are
/// skipped (no pseudocost update, no score entry).
#[allow(clippy::too_many_arguments)]
fn measure_strong_branch_scores<R: Relaxation>(
    problem: &R,
    parent_bounds: &[(f64, f64)],
    parent_sol: &[f64],
    parent_obj: f64,
    candidates: &[usize],
    j_to_k: &HashMap<usize, usize>,
    shared: &SolverOptions,
    parent_warm_start: Option<&WarmStartBasis>,
    pc: &mut PseudocostState,
    stats: &mut MipStats,
) -> HashMap<usize, f64> {
    let mut scores = HashMap::new();
    stats.strong_branch_calls += 1;
    stats.strong_branch_candidates += candidates.len();
    let mut sb_opts = shared.clone();
    sb_opts.warm_start = parent_warm_start.cloned();
    if problem.can_skip_repeated_lp_scaling() {
        sb_opts.use_ruiz_scaling = false;
    }
    let mut sb_retry_opts = sb_opts.clone();
    sb_retry_opts.use_ruiz_scaling = true;
    // Cold variants for children that change the standard-form column layout
    // (infinite→finite bound): the parent basis indices no longer match those
    // columns, so the warm start must be dropped — same guard the real-children
    // path applies. Without this, strong-branch scores for one-sided integer
    // variables are computed from a layout-mismatched warm start.
    let mut sb_opts_cold = sb_opts.clone();
    sb_opts_cold.warm_start = None;
    let mut sb_retry_opts_cold = sb_retry_opts.clone();
    sb_retry_opts_cold.warm_start = None;

    for &j in candidates {
        let v = parent_sol[j];
        let (down_bounds, up_bounds) = branch::branch_bounds(parent_bounds, j, v);
        let (down_o, down_ro) = if bound_layout_changes(parent_bounds, &down_bounds, j) {
            (&sb_opts_cold, &sb_retry_opts_cold)
        } else {
            (&sb_opts, &sb_retry_opts)
        };
        let (up_o, up_ro) = if bound_layout_changes(parent_bounds, &up_bounds, j) {
            (&sb_opts_cold, &sb_retry_opts_cold)
        } else {
            (&sb_opts, &sb_retry_opts)
        };

        let t0 = Instant::now();
        let r_down = solve_relaxation_with_scaling_retry(problem, &down_bounds, down_o, down_ro);
        let r_up = solve_relaxation_with_scaling_retry(problem, &up_bounds, up_o, up_ro);
        stats.strong_branch_lp_solves += 2;
        stats.strong_branch_us = stats
            .strong_branch_us
            .saturating_add(t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64);
        stats.strong_branch_iters = stats
            .strong_branch_iters
            .saturating_add(r_down.iterations as u64)
            .saturating_add(r_up.iterations as u64);

        let down_ok = matches!(
            r_down.status,
            SolveStatus::Optimal | SolveStatus::Infeasible
        );
        let up_ok = matches!(r_up.status, SolveStatus::Optimal | SolveStatus::Infeasible);

        if !down_ok || !up_ok {
            continue;
        }

        let d_down = if r_down.status == SolveStatus::Infeasible {
            f64::INFINITY
        } else {
            (r_down.objective - parent_obj).max(0.0)
        };
        let d_up = if r_up.status == SolveStatus::Infeasible {
            f64::INFINITY
        } else {
            (r_up.objective - parent_obj).max(0.0)
        };

        // `j` is drawn from `candidates`, which `strong_branch_candidates` builds
        // by iterating the same `integer_vars` slice that `j_to_k` was built
        // from (`j_to_k = integer_vars.iter().enumerate().map(|(k, &j)| (j, k))`),
        // so `j` is always a key of `j_to_k`.
        let &k = j_to_k
            .get(&j)
            .expect("j is drawn from integer_vars, which built j_to_k");
        let f_down = v - v.floor();
        let f_up = v.ceil() - v;
        if d_down.is_finite() && f_down > 1e-12 {
            pc.record_down(k, d_down / f_down);
        }
        if d_up.is_finite() && f_up > 1e-12 {
            pc.record_up(k, d_up / f_up);
        }

        let score = branch::pseudocost_score(d_down, d_up);
        scores.insert(j, score);
    }
    scores
}

fn solve_relaxation_with_scaling_retry<R: Relaxation>(
    problem: &R,
    bounds: &[(f64, f64)],
    fast_opts: &SolverOptions,
    retry_opts: &SolverOptions,
) -> SolverResult {
    let res = problem.solve(bounds, fast_opts);
    if fast_opts.use_ruiz_scaling
        || !needs_scaled_retry(&res)
        || deadline_reached(fast_opts.deadline)
    {
        return res;
    }
    // Codex review (P1): the retry re-solves the *same* relaxation the fast
    // attempt already spent `res.iterations` on, so it must not receive a
    // fresh copy of `fast_opts.max_iters` — that cap bounds this call's total
    // simplex work (`solve_node_relaxation` sets it to the node's remaining
    // share of `MipConfig::max_lp_iters`; see `SolverOptions::max_iters`).
    // Handing the retry the same limit again would let one node/strong-branch
    // candidate spend up to 2x its iteration allowance whenever the fast
    // (unscaled) attempt exhausts its budget without converging (Stalled /
    // MaxIterations, both in `needs_scaled_retry`). Charge the fast attempt's
    // iterations against the shared cap first, and skip the retry outright
    // once nothing remains — an unscaled attempt that stalled at exactly the
    // node's remaining budget gains nothing from a differently-scaled retry
    // that would immediately hit the same (now zero) cap.
    let mut capped_retry_opts;
    let retry_opts = match fast_opts.max_iters {
        Some(cap) => {
            let remaining = cap.saturating_sub(res.iterations as u64);
            if remaining == 0 {
                return res;
            }
            capped_retry_opts = retry_opts.clone();
            capped_retry_opts.max_iters = Some(remaining);
            &capped_retry_opts
        }
        None => retry_opts,
    };
    let mut retry = problem.solve(bounds, retry_opts);
    retry.timing_breakdown = combine_timing(res.timing_breakdown, retry.timing_breakdown);
    // P3-F: the first (unscaled) attempt's simplex iterations would otherwise
    // be silently dropped — `effort::total_simplex_iters` and friends rely on
    // `res.iterations` to attribute cost, and both attempts genuinely ran.
    retry.iterations = retry.iterations.saturating_add(res.iterations);
    retry
}

fn needs_scaled_retry(res: &SolverResult) -> bool {
    matches!(
        res.status,
        SolveStatus::Timeout
            | SolveStatus::NumericalError
            | SolveStatus::SuboptimalSolution
            | SolveStatus::Stalled
            | SolveStatus::MaxIterations
    )
}

fn combine_timing(
    first: Option<TimingBreakdown>,
    second: Option<TimingBreakdown>,
) -> Option<TimingBreakdown> {
    match (first, second) {
        (None, None) => None,
        (Some(t), None) | (None, Some(t)) => Some(t),
        (Some(a), Some(b)) => Some(TimingBreakdown {
            presolve_us: a.presolve_us.saturating_add(b.presolve_us),
            solve_us: a.solve_us.saturating_add(b.solve_us),
            postsolve_us: a.postsolve_us.saturating_add(b.postsolve_us),
            ipm_factorize_us: a.ipm_factorize_us.saturating_add(b.ipm_factorize_us),
            ipm_solve_us: a.ipm_solve_us.saturating_add(b.ipm_solve_us),
            ipm_reg_retries: a.ipm_reg_retries.saturating_add(b.ipm_reg_retries),
            ipm_used_iterative: a.ipm_used_iterative || b.ipm_used_iterative,
            postsolve_map_us: a.postsolve_map_us.saturating_add(b.postsolve_map_us),
            postsolve_lsq_us: a.postsolve_lsq_us.saturating_add(b.postsolve_lsq_us),
            postsolve_recovery_us: a
                .postsolve_recovery_us
                .saturating_add(b.postsolve_recovery_us),
            postsolve_refine_us: a.postsolve_refine_us.saturating_add(b.postsolve_refine_us),
            postsolve_krylov_ir_us: a
                .postsolve_krylov_ir_us
                .saturating_add(b.postsolve_krylov_ir_us),
        }),
    }
}

/// Core B&B driver that accepts a precomputed `integer_mask` to avoid
/// recomputing it when the caller (e.g. `solve_milp_with_stats`) already has it.
///
/// `initial_incumbent` is an optional integer-feasible solution found by a
/// pre-B&B heuristic (e.g., feasibility pump). When provided it is adopted as
/// the starting incumbent so B&B can immediately prune nodes whose relaxation
/// bound is already within the gap tolerance.
/// [`prepare_mip_search`]'s early-return variant: the plain-LP/QP solve
/// result for a problem with no integer variables. Boxed (`clippy::result_
/// large_err`): `(SolverResult, MipStats)` is large enough to otherwise
/// bloat every `Result` returned from this rarely-taken path.
type MipSearchEarlyReturn = Box<(SolverResult, MipStats)>;

/// Builds the shared per-node `SolverOptions` and computes the search
/// deadline for `solve_mip_core`. Returns `Err` early (with the plain-LP/QP
/// solve result) when `problem` has no integer variables — no B&B needed.
fn prepare_mip_search<R: Relaxation>(
    problem: &R,
    options: &SolverOptions,
) -> Result<(SolverOptions, Option<Instant>, MipStats), MipSearchEarlyReturn> {
    let stats = MipStats {
        approx_bounds_bytes_per_node: problem.num_vars() * 2 * std::mem::size_of::<f64>(),
        root_lp_bound: f64::NEG_INFINITY,
        ..MipStats::default()
    };

    let deadline = options.deadline.or_else(|| {
        options
            .timeout_secs
            .map(|s| Instant::now() + Duration::from_secs_f64(s))
    });
    let mut shared = options.clone();
    shared.deadline = deadline;
    shared.timeout_secs = None;
    shared.multistart = None;
    shared.global_optimization = None;

    if problem.integer_vars().is_empty() {
        return Err(Box::new((
            problem.solve(problem.root_bounds(), &shared),
            stats,
        )));
    }

    shared.recover_warm_start_basis = true;
    shared.use_lp_crash_basis = false;
    shared.warm_start = None;
    if problem.skip_node_presolve() {
        shared.presolve = false;
    }
    Ok((shared, deadline, stats))
}

/// In-tree GMI/MIR cut re-separation for one node (gated by
/// [`may_separate_tree_cuts`]; MILP overrides `separate_tree_cuts`, MIQP is a
/// no-op). Returns `res` unchanged unless separation found a cut-tightened
/// result with a better bound, in which case that replaces it.
#[allow(clippy::too_many_arguments)]
fn maybe_apply_tree_cut_separation<R: Relaxation>(
    problem: &R,
    cfg: &MipConfig,
    stats: &mut MipStats,
    mask: &[bool],
    node: &MipNode,
    solve_bounds: &[(f64, f64)],
    node_options: &SolverOptions,
    res: SolverResult,
) -> SolverResult {
    if !may_separate_tree_cuts(cfg, stats)
        || !matches!(res.status, SolveStatus::Optimal)
        || res.solution.is_empty()
    {
        return res;
    }
    let tree_cut_t0 = Instant::now();
    let max_iters = effort::separation_iter_budget(stats);
    let (separated, sep_iters, sep_overhead_iters, attempted) = problem.separate_tree_cuts(
        solve_bounds,
        &res,
        mask,
        node_options,
        node.depth,
        stats.nodes_processed,
        max_iters,
    );
    stats.tree_cut_us = stats
        .tree_cut_us
        .saturating_add(tree_cut_t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64);
    stats.tree_cut_iters = stats.tree_cut_iters.saturating_add(sep_iters);
    stats.tree_cut_overhead_iters = stats
        .tree_cut_overhead_iters
        .saturating_add(sep_overhead_iters);
    record_separation_attempt(stats, attempted, separated.is_some());
    if let Some(improved) = separated {
        stats.tree_cut_rounds += 1;
        return improved;
    }
    res
}

/// Read-only search context shared by every node-loop iteration: the problem
/// configuration, the per-node `SolverOptions` template and the derived index
/// tables. Fixed for the whole solve, so a parallel worker can borrow it.
pub(crate) struct SearchCtx<'a> {
    cfg: &'a MipConfig,
    /// Per-node `SolverOptions` template built by [`prepare_mip_search`].
    shared: &'a SolverOptions,
    mask: &'a [bool],
    integer_vars: &'a [usize],
    j_to_k: &'a HashMap<usize, usize>,
    deadline: Option<Instant>,
    root_bounds: &'a [(f64, f64)],
    use_reliability: bool,
}

/// Mutable search state advanced by [`run_node`]. The serial driver owns one;
/// each parallel worker owns its own, with the incumbent and the conflict
/// clauses backed by process-wide shared handles (see [`MipState`] and
/// [`conflict::ConflictStore`]) and the remaining fields reduced at join.
pub(crate) struct SearchState {
    /// Serial: the whole open-node set. Parallel: the worker's private dive
    /// stack plus the not-yet-published children of the node it just
    /// processed (drained into the shared pool by the worker loop).
    q: NodeQueue,
    state: MipState,
    stats: MipStats,
    pc: PseudocostState,
    conflicts: conflict::ConflictStore,
    open_lb: f64,
    had_open: bool,
    proof_uncertain: bool,
    deadline_stop: bool,
    maxnodes_stop: bool,
    unbounded: bool,
    nodes_since_dive: usize,
    dive_start_depth: usize,
}

impl SearchState {
    fn new(
        stats: MipStats,
        n_int: usize,
        state: MipState,
        conflicts: conflict::ConflictStore,
    ) -> Self {
        Self {
            q: NodeQueue::new(),
            state,
            stats,
            pc: PseudocostState::new(n_int),
            conflicts,
            open_lb: f64::INFINITY,
            had_open: false,
            proof_uncertain: false,
            deadline_stop: false,
            maxnodes_stop: false,
            unbounded: false,
            nodes_since_dive: 0,
            dive_start_depth: 0,
        }
    }
}

/// Whether the node loop should process another node or stop.
pub(crate) enum NodeLoop {
    Continue,
    Break,
}

/// Applies one node's [`process_node_outcome`] result: prunes, records an
/// open lower bound, or pushes branched/split children onto `s.q`. Always the
/// last step of a node-loop iteration, so it also charges
/// `node_loop_other_us` for this iteration on every path (matching every
/// other exit point in the loop).
fn apply_node_outcome<R: Relaxation>(
    problem: &R,
    mut node: MipNode,
    res: &SolverResult,
    ctx: &SearchCtx<'_>,
    s: &mut SearchState,
    is_root: bool,
    iter_t0: Instant,
    iter_before: MipStats,
) {
    let trusted = matches!(res.status, SolveStatus::Optimal) && !res.solution.is_empty();
    match process_node_outcome(problem, &mut node, res, trusted, ctx, s, is_root) {
        NodeAction::Skip { end_dive } => {
            if end_dive && s.q.is_diving() {
                s.q.end_dive();
            }
        }
        NodeAction::OpenLb {
            node_lb,
            uncertain,
            end_dive,
        } => {
            s.open_lb = s.open_lb.min(node_lb);
            s.had_open = true;
            if uncertain {
                s.proof_uncertain = true;
            }
            if end_dive && s.q.is_diving() {
                s.q.end_dive();
            }
        }
        NodeAction::PushChildren {
            node_lb,
            down,
            up,
            kind,
            end_dive,
        } => {
            if end_dive && s.q.is_diving() {
                s.q.end_dive();
            }
            match kind {
                ChildKind::Branched {
                    jb,
                    res_obj,
                    jb_val,
                    down_ws,
                    up_ws,
                } => {
                    s.q.push(
                        node.child_branched(down, node_lb, down_ws, jb, false, res_obj, jb_val),
                    );
                    s.q.push(node.child_branched(up, node_lb, up_ws, jb, true, res_obj, jb_val));
                }
                ChildKind::Split => {
                    s.q.push(node.child(down, node_lb));
                    s.q.push(node.child(up, node_lb));
                }
            }
        }
    }
    flush_loop_other(&mut s.stats, iter_t0, iter_before);
}

/// Outcome of [`dispatch_relaxation_status`]: whether the node loop should
/// keep processing this node (`Proceed`), skip to the next node
/// (`Continue`), or stop the search (`Break`).
enum StatusDispatch {
    Proceed,
    Continue,
    Break,
}

/// Dispatches on a node relaxation's `SolveStatus` before separation/branching
/// runs. `Infeasible` prunes and learns a conflict clause; `Unbounded` and
/// `Timeout` stop the search (folding the node's bound into the open region
/// for `Timeout`); `MaxIterations`/`SuboptimalSolution` do the same when
/// `cfg.max_lp_iters` is what actually stopped this node's own solve (see the
/// `cfg.max_lp_iters` check below); any other status proceeds to
/// separation/branching as normal.
#[allow(clippy::too_many_arguments)]
fn dispatch_relaxation_status(
    res: &SolverResult,
    node: &MipNode,
    q: &mut NodeQueue,
    stats: &mut MipStats,
    cfg: &MipConfig,
    conflicts: &mut conflict::ConflictStore,
    root_bounds: &[(f64, f64)],
    open_lb: &mut f64,
    had_open: &mut bool,
    deadline_stop: &mut bool,
    unbounded: &mut bool,
) -> StatusDispatch {
    match res.status {
        SolveStatus::Infeasible => {
            stats.pruned += 1;
            if q.is_diving() {
                q.end_dive();
            }
            let learn_t0 = Instant::now();
            conflicts.learn(&node.var_bounds, root_bounds);
            stats.conflict_us = stats
                .conflict_us
                .saturating_add(learn_t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64);
            stats.conflict_clauses_learned = conflicts.len();
            StatusDispatch::Continue
        }
        SolveStatus::Unbounded => {
            *unbounded = true;
            if q.is_diving() {
                q.end_dive();
            }
            StatusDispatch::Break
        }
        SolveStatus::Timeout => {
            *open_lb = open_lb.min(node.lower_bound);
            *had_open = true;
            *deadline_stop = true;
            if q.is_diving() {
                q.end_dive();
            }
            StatusDispatch::Break
        }
        // Codex round 3 (P2): `cfg.max_lp_iters` (sub-MIP-only, see
        // `solve_node_relaxation`'s doc) is "treated identically to deadline
        // expiry" per `check_stop_conditions`'s doc — but that gate only
        // re-checks the budget at the *next* pop. A node whose own
        // relaxation is what pushes `lp_iters_total` past the cap honestly
        // reports `MaxIterations`/`SuboptimalSolution` (never `Timeout`,
        // since no wall-clock deadline fired — see `stop_status`), so
        // without this arm it fell through to `Proceed` and, if this was the
        // last node left in the queue, the loop would exit with no stop flag
        // ever set: a budget-truncated search silently finalized as if the
        // tree had been fully explored. Scoped to `cfg.max_lp_iters` being
        // both set and actually exhausted, so an unrelated cycling/plateau
        // bail on the (uncapped) top-level search still proceeds as before.
        SolveStatus::MaxIterations | SolveStatus::SuboptimalSolution
            if cfg
                .max_lp_iters
                .is_some_and(|limit| stats.lp_iters_total >= limit) =>
        {
            *open_lb = open_lb.min(node.lower_bound);
            *had_open = true;
            *deadline_stop = true;
            if q.is_diving() {
                q.end_dive();
            }
            StatusDispatch::Break
        }
        _ => StatusDispatch::Proceed,
    }
}

/// Solves one node's LP/QP relaxation (with the scaling-retry fallback) and
/// folds the timing/scale/fallback deltas into `stats` via
/// `accumulate_node_stats`. Returns the relaxation result and the per-node
/// `SolverOptions` used (needed again by tree-cut separation).
///
/// When `cfg.max_lp_iters` is set (RINS/RENS/local-branching sub-MIPs; `None`
/// for the top-level search), this node's `SolverOptions::max_iters` is set
/// to the *remaining* share of that budget (`limit - stats.lp_iters_total` so
/// far). Without this, `check_stop_conditions`'s `max_lp_iters` gate — checked
/// once per popped node, before this call — cannot stop a single node whose
/// own relaxation blows past the entire remaining budget by itself; the
/// per-solve cap here makes that node's own simplex loop enforce it instead
/// (`dual_advanced::bounded_core`'s bland-mode loops honor `max_iters`
/// alongside `deadline`, returning `Stalled` rather than grinding on).
///
/// `is_root` (`node.depth == 0`) selects the root vs. descendant stats
/// buckets and the repeated-scaling skip. It is exactly the predicate the
/// previous `root_solved` latch computed: the first relaxation any search
/// solves is the root (the queue starts seeded with it alone, and every
/// pre-solve `continue` at the root empties the queue), and no other node
/// has depth 0. Stating it per node rather than latching it makes the
/// buckets well-defined when several workers process nodes concurrently.
fn solve_node_relaxation<R: Relaxation>(
    problem: &R,
    shared: &SolverOptions,
    cfg: &MipConfig,
    solve_bounds: &[(f64, f64)],
    node: &MipNode,
    stats: &mut MipStats,
) -> (SolverResult, SolverOptions) {
    let is_root = node.depth == 0;
    let scale_before = crate::presolve::scaling::lp_scale_profile_snapshot();
    let fallback_before = crate::simplex::dual_advanced::fallback_profile_snapshot();
    let t0 = Instant::now();
    let mut node_options = shared.clone();
    if !is_root && problem.can_skip_repeated_lp_scaling() {
        node_options.use_ruiz_scaling = false;
    }
    if let Some(ref ws) = node.warm_start {
        node_options.warm_start = Some(ws.clone());
    }
    if let Some(limit) = cfg.max_lp_iters {
        node_options.max_iters = Some(limit.saturating_sub(stats.lp_iters_total));
    }
    let res = if node_options.use_ruiz_scaling {
        problem.solve(solve_bounds, &node_options)
    } else {
        let mut retry_options = node_options.clone();
        retry_options.use_ruiz_scaling = true;
        solve_relaxation_with_scaling_retry(problem, solve_bounds, &node_options, &retry_options)
    };
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let scale_delta = crate::presolve::scaling::lp_scale_profile_delta(
        scale_before,
        crate::presolve::scaling::lp_scale_profile_snapshot(),
    );
    let fallback_delta = crate::simplex::dual_advanced::fallback_profile_delta(
        fallback_before,
        crate::simplex::dual_advanced::fallback_profile_snapshot(),
    );
    accumulate_node_stats(
        stats,
        elapsed_ms,
        &scale_delta,
        &fallback_delta,
        &res,
        is_root,
        node.depth,
    );
    (res, node_options)
}

/// Per-node bound propagation (MILP only; MIQP's `propagation_data` returns
/// `None`, so this is always `Ok(None)` for MIQP). `Err(())` means
/// propagation proved the node's bounds infeasible; the caller must count it
/// pruned and `continue`. `node_propagation_us` is charged exactly once
/// either way.
fn propagate_node_bounds<R: Relaxation>(
    problem: &R,
    node: &MipNode,
    mask: &[bool],
    stats: &mut MipStats,
) -> Result<Option<Vec<(f64, f64)>>, ()> {
    let propagation_t0 = Instant::now();
    let result = if let Some((prop_a, prop_b, prop_ct)) = problem.propagation_data() {
        presolve::tighten_bounds_at_node(
            problem.num_vars(),
            prop_a,
            prop_b,
            prop_ct,
            &node.var_bounds,
            mask,
        )
        .map(Some)
    } else {
        Ok(None)
    };
    stats.node_propagation_us = stats.node_propagation_us.saturating_add(
        propagation_t0
            .elapsed()
            .as_micros()
            .min(u128::from(u64::MAX)) as u64,
    );
    if result.is_err() {
        stats.pruned += 1;
        stats.propagation_pruned += 1;
    }
    result
}

/// Node-loop stop conditions (deadline / `max_nodes` / deterministic
/// `max_lp_iters`), checked once per popped node before any other
/// processing. Returns `true` when the loop must stop; on stop, folds
/// `node`'s bound into the open region, sets the matching stop flag, ends
/// any active dive, and flushes `node_loop_other_us` for this iteration.
///
/// The `max_lp_iters` stop (Phase 1d, P1-B) is a fixed cap on this solve's
/// own node-relaxation simplex iterations, set on RINS/RENS/local-branching
/// sub-MIP configs (see `heuristics::SUB_MIP_MAX_LP_ITERS`) so their
/// termination point does not depend on wall-clock timing; it is treated
/// identically to deadline expiry.
/// `nodes_done` is the node count `cfg.max_nodes` is compared against: the
/// serial driver passes its own `stats.nodes_processed`; a parallel worker
/// passes the search-wide total, so the cap bounds the whole tree rather than
/// each worker's private share.
fn check_stop_conditions(
    ctx: &SearchCtx<'_>,
    node: &MipNode,
    s: &mut SearchState,
    nodes_done: usize,
    iter_t0: Instant,
    iter_before: MipStats,
) -> bool {
    let stop_reason = if deadline_reached(ctx.deadline) {
        Some(false)
    } else if nodes_done >= ctx.cfg.max_nodes {
        Some(true)
    } else if ctx
        .cfg
        .max_lp_iters
        .is_some_and(|limit| s.stats.lp_iters_total >= limit)
    {
        Some(false)
    } else {
        None
    };
    let Some(is_maxnodes) = stop_reason else {
        return false;
    };
    s.open_lb = s.open_lb.min(node.lower_bound);
    s.had_open = true;
    if is_maxnodes {
        s.maxnodes_stop = true;
    } else {
        s.deadline_stop = true;
    }
    if s.q.is_diving() {
        s.q.end_dive();
    }
    flush_loop_other(&mut s.stats, iter_t0, iter_before);
    true
}

/// Process exactly one popped node: stop-condition checks, dive bookkeeping,
/// bound/conflict pruning, the relaxation solve, in-tree separation and the
/// branching decision. Shared verbatim by the serial driver and by every
/// parallel worker (`parallel::run_workers`), so the two search modes can
/// never drift apart in what a node *means*; they differ only in where the
/// open nodes live and how the per-worker results are reduced.
///
/// `nodes_done` — see [`check_stop_conditions`].
fn run_node<R: Relaxation>(
    problem: &R,
    node: MipNode,
    ctx: &SearchCtx<'_>,
    s: &mut SearchState,
    nodes_done: usize,
) -> NodeLoop {
    // Snapshot for `node_loop_other_us`: MipStats is Copy, so this is a
    // cheap per-iteration baseline. Every named bucket below is measured
    // by Instant and accumulated into `stats`; `flush_loop_other` at each
    // exit point below charges whatever wall time this iteration spent
    // outside those buckets to `node_loop_other_us`.
    let iter_t0 = Instant::now();
    let iter_before = s.stats;
    if check_stop_conditions(ctx, &node, s, nodes_done, iter_t0, iter_before) {
        return NodeLoop::Break;
    }

    // --- Dive management: start a dive every DIVE_FREQUENCY best-bound pops ---
    if !s.q.is_diving() {
        s.nodes_since_dive += 1;
        let freq = if s.state.incumbent_obj.is_none() {
            DIVE_FREQUENCY_NO_INCUMBENT
        } else {
            DIVE_FREQUENCY
        };
        if s.nodes_since_dive >= freq {
            s.nodes_since_dive = 0;
            s.dive_start_depth = node.depth;
            s.q.start_dive();
        }
    }

    // --- Incumbent-bound pruning and conflict pruning ---
    if let Some(inc) = s.state.incumbent_obj {
        if should_prune(node.lower_bound, Some(inc), ctx.cfg.gap_tol) {
            s.stats.pruned += 1;
            if s.q.is_diving() {
                s.q.end_dive();
            }
            flush_loop_other(&mut s.stats, iter_t0, iter_before);
            return NodeLoop::Continue;
        }
    }
    let conflict_check_t0 = Instant::now();
    let node_is_conflicted = node.depth > 0 && s.conflicts.is_conflicted(&node.var_bounds);
    s.stats.conflict_us = s.stats.conflict_us.saturating_add(
        conflict_check_t0
            .elapsed()
            .as_micros()
            .min(u128::from(u64::MAX)) as u64,
    );
    if node_is_conflicted {
        s.stats.pruned += 1;
        s.stats.conflict_pruned += 1;
        flush_loop_other(&mut s.stats, iter_t0, iter_before);
        return NodeLoop::Continue;
    }

    let tightened = match propagate_node_bounds(problem, &node, ctx.mask, &mut s.stats) {
        Ok(t) => t,
        Err(()) => {
            flush_loop_other(&mut s.stats, iter_t0, iter_before);
            return NodeLoop::Continue;
        }
    };
    let solve_bounds: &[(f64, f64)] = tightened.as_deref().unwrap_or(&node.var_bounds);

    let is_root = node.depth == 0;
    let (mut res, node_options) = solve_node_relaxation(
        problem,
        ctx.shared,
        ctx.cfg,
        solve_bounds,
        &node,
        &mut s.stats,
    );
    // Must run unconditionally for every processed node (Codex review,
    // P2): `nodes_processed` was already incremented above regardless of
    // this node's outcome, but `maybe_apply_tree_cut_separation` below is
    // skipped entirely on Infeasible/Unbounded/Timeout dispatch. Checking
    // the reset only inside that skipped path let a boundary node that
    // happened to be Infeasible push the periodic reset a full interval
    // late (`is_multiple_of` never matches again until the next
    // multiple).
    maybe_reset_separation_dry_streak(&mut s.stats);

    match dispatch_relaxation_status(
        &res,
        &node,
        &mut s.q,
        &mut s.stats,
        ctx.cfg,
        &mut s.conflicts,
        ctx.root_bounds,
        &mut s.open_lb,
        &mut s.had_open,
        &mut s.deadline_stop,
        &mut s.unbounded,
    ) {
        StatusDispatch::Continue => {
            flush_loop_other(&mut s.stats, iter_t0, iter_before);
            return NodeLoop::Continue;
        }
        StatusDispatch::Break => {
            flush_loop_other(&mut s.stats, iter_t0, iter_before);
            return NodeLoop::Break;
        }
        StatusDispatch::Proceed => {}
    }

    res = maybe_apply_tree_cut_separation(
        problem,
        ctx.cfg,
        &mut s.stats,
        ctx.mask,
        &node,
        solve_bounds,
        &node_options,
        res,
    );

    apply_node_outcome(problem, node, &res, ctx, s, is_root, iter_t0, iter_before);
    NodeLoop::Continue
}

/// Owned inputs a [`SearchCtx`] borrows: the per-node options template, the
/// integer-variable index tables, the root bounds and the seed statistics.
type PreparedSearch = (
    SolverOptions,
    Vec<usize>,
    HashMap<usize, usize>,
    Vec<(f64, f64)>,
    MipStats,
);

/// Build the owned inputs the serial and parallel drivers both borrow from.
fn prepare_search_inputs<R: Relaxation>(
    problem: &R,
    options: &SolverOptions,
) -> Result<PreparedSearch, MipSearchEarlyReturn> {
    let (shared, _deadline, stats) = prepare_mip_search(problem, options)?;
    let integer_vars = problem.integer_vars().to_vec();
    let j_to_k: HashMap<usize, usize> = integer_vars
        .iter()
        .enumerate()
        .map(|(k, &j)| (j, k))
        .collect();
    let root_bounds = problem.root_bounds().to_vec();
    Ok((shared, integer_vars, j_to_k, root_bounds, stats))
}

/// Route a branch-and-bound search to the serial or the parallel driver.
///
/// The parallel driver is opt-in and requires **both**:
///
/// * `options.threads >= 2` — `threads = 1` (the default) must stay on the
///   serial driver, which is deterministic node-for-node. Reproducibility of
///   the search trajectory is what the benchmark suite's per-problem
///   regression diffs are built on, so it is not something a default may
///   trade away;
/// * `cfg.max_lp_iters == None` — that budget (set only on RINS / RENS /
///   local-branching sub-MIPs, see `heuristics::SUB_MIP_MAX_LP_ITERS`) exists
///   precisely to make a sub-MIP's stopping point independent of timing, so a
///   caller asking for it gets the serial driver whatever `threads` says.
///   Sub-MIPs already force `threads = 1` on their own options; this is the
///   structural guarantee behind that convention.
pub(crate) fn solve_mip_dispatch<R: Relaxation + Sync>(
    problem: &R,
    options: &SolverOptions,
    cfg: &MipConfig,
    mask: Vec<bool>,
    initial_incumbent: Option<SolverResult>,
) -> (SolverResult, MipStats) {
    if options.threads >= 2 && cfg.max_lp_iters.is_none() {
        return parallel::solve_mip_parallel(
            problem,
            options,
            cfg,
            mask,
            initial_incumbent,
            options.threads,
        );
    }
    solve_mip_core(problem, options, cfg, mask, initial_incumbent)
}

fn solve_mip_core<R: Relaxation>(
    problem: &R,
    options: &SolverOptions,
    cfg: &MipConfig,
    mask: Vec<bool>,
    initial_incumbent: Option<SolverResult>,
) -> (SolverResult, MipStats) {
    let (shared, integer_vars, j_to_k, root_bounds, stats) =
        match prepare_search_inputs(problem, options) {
            Ok(v) => v,
            Err(early_return) => return *early_return,
        };
    let ctx = SearchCtx {
        cfg,
        shared: &shared,
        mask: &mask,
        integer_vars: &integer_vars,
        j_to_k: &j_to_k,
        deadline: shared.deadline,
        root_bounds: &root_bounds,
        use_reliability: cfg.branching == MipBranching::Reliability,
    };

    let mut s = SearchState::new(
        stats,
        integer_vars.len(),
        MipState::new(),
        conflict::ConflictStore::new(),
    );
    if let Some(inc) = initial_incumbent {
        if s.state.consider(&inc) {
            s.stats.incumbent_updates += 1;
            s.stats.fp_incumbent_found = true;
        }
    }
    s.q.push(MipNode::root(
        problem.root_bounds().to_vec(),
        f64::NEG_INFINITY,
    ));

    while let Some(node) = s.q.pop() {
        let nodes_done = s.stats.nodes_processed;
        if matches!(
            run_node(problem, node, &ctx, &mut s, nodes_done),
            NodeLoop::Break
        ) {
            break;
        }
    }

    let outcome = SearchOutcome::from(&s);
    finalize_mip_result(problem, cfg, s.q, s.state, s.stats, outcome)
}

/// Terminal flags reduced out of one or more [`SearchState`]s for
/// [`finalize_mip_result`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct SearchOutcome {
    open_lb: f64,
    had_open: bool,
    unbounded: bool,
    deadline_stop: bool,
    maxnodes_stop: bool,
    proof_uncertain: bool,
}

impl SearchOutcome {
    /// The identity element of [`SearchOutcome::absorb`]: nothing explored,
    /// nothing left open, no stop requested.
    fn empty() -> Self {
        Self {
            open_lb: f64::INFINITY,
            had_open: false,
            unbounded: false,
            deadline_stop: false,
            maxnodes_stop: false,
            proof_uncertain: false,
        }
    }

    fn from(s: &SearchState) -> Self {
        Self {
            open_lb: s.open_lb,
            had_open: s.had_open,
            unbounded: s.unbounded,
            deadline_stop: s.deadline_stop,
            maxnodes_stop: s.maxnodes_stop,
            proof_uncertain: s.proof_uncertain,
        }
    }

    /// Reduce a worker's terminal flags into the search-wide outcome: the
    /// open lower bound is the minimum over workers (any region one worker
    /// left open is open for the search), and every stop/uncertainty flag is
    /// a disjunction.
    fn absorb(&mut self, other: &Self) {
        self.open_lb = self.open_lb.min(other.open_lb);
        self.had_open |= other.had_open;
        self.unbounded |= other.unbounded;
        self.deadline_stop |= other.deadline_stop;
        self.maxnodes_stop |= other.maxnodes_stop;
        self.proof_uncertain |= other.proof_uncertain;
    }
}

/// Post-loop finalization shared by every `solve_mip_core` exit path:
/// closes out any active dive, short-circuits on `Unbounded`, computes the
/// remaining open lower bound, and — from the surviving incumbent (if any) —
/// decides `Optimal` (with a [`BoundGapCertificate`]) vs `Timeout` vs
/// `SuboptimalSolution`, or defers to [`finalize_no_incumbent`].
fn finalize_mip_result<R: Relaxation>(
    problem: &R,
    cfg: &MipConfig,
    mut q: NodeQueue,
    mut state: MipState,
    stats: MipStats,
    outcome: SearchOutcome,
) -> (SolverResult, MipStats) {
    let SearchOutcome {
        open_lb,
        had_open,
        unbounded,
        deadline_stop,
        maxnodes_stop,
        proof_uncertain,
    } = outcome;
    if q.is_diving() {
        q.end_dive();
    }
    if unbounded {
        return (SolverResult::unbounded(), stats);
    }

    let remaining_lb = match q.best_lower_bound() {
        Some(b) => open_lb.min(b),
        None => open_lb,
    };
    let interrupted = deadline_stop || maxnodes_stop;
    // Codex review (P0 follow-up): `remaining_lb == +inf` means either (a)
    // fully resolved (`open_lb` untouched, `had_open` never fired — the same
    // condition `finalize_no_incumbent` uses for `fully_resolved` below),
    // trivially gap-closed; or (b) a corrupted node folded a non-finite bound
    // into `open_lb` with `had_open == true`, which is NOT a proof.
    // `within_gap`'s symmetric `is_finite()` guard (P0 fix) rejects both
    // uniformly, so (a) needs this short-circuit ahead of it — mirrors
    // `qp::global::solve_qp_global_with_stats`'s `!halted_early` branch,
    // which already bypasses `within_gap` under the equivalent conditions.
    // Sentinel: `tests::fully_resolved_search_still_proves_optimal_without_
    // open_region` — reverting this demotes a complete search's incumbent
    // from `Optimal` to `SuboptimalSolution`.
    let fully_resolved = !interrupted && !had_open && q.is_empty();

    match state.incumbent.take() {
        Some(mut inc) => {
            let inc_obj = state.incumbent_obj.expect("incumbent objective set");
            let proven = !proof_uncertain
                && (fully_resolved || within_gap(inc_obj, remaining_lb, cfg.gap_tol));
            inc.solution = round_integers(inc.solution, problem.integer_vars());
            inc.status = if proven {
                let effective_lb = remaining_lb.min(inc_obj);
                let scale = 1.0_f64.max(inc_obj.abs());
                let gap_rel = (inc_obj - effective_lb) / scale;
                inc.bound_gap_cert = Some(BoundGapCertificate::new(
                    inc_obj,
                    effective_lb,
                    gap_rel,
                    cfg.gap_tol,
                ));
                SolveStatus::Optimal
            } else if deadline_stop {
                SolveStatus::Timeout
            } else {
                SolveStatus::SuboptimalSolution
            };
            (inc, stats)
        }
        None => (
            finalize_no_incumbent(interrupted, had_open, q.is_empty(), deadline_stop),
            stats,
        ),
    }
}

// ---------------------------------------------------------------------------
// B&B node processing helpers
// ---------------------------------------------------------------------------

/// Action the B&B loop should take after processing a node's outcome.
enum NodeAction {
    /// Skip node (pruned or integer-feasible leaf). End dive when `end_dive`.
    Skip { end_dive: bool },
    /// Record open lower bound (max-depth or proof-uncertain region).
    /// Mark `proof_uncertain` when `uncertain`. End dive when `end_dive`.
    OpenLb {
        node_lb: f64,
        uncertain: bool,
        end_dive: bool,
    },
    /// Push two child nodes. End dive when `end_dive`.
    PushChildren {
        node_lb: f64,
        down: Vec<(f64, f64)>,
        up: Vec<(f64, f64)>,
        kind: ChildKind,
        end_dive: bool,
    },
}

/// Metadata distinguishing trusted-Optimal branches from non-Optimal bisections.
enum ChildKind {
    /// Trusted (Optimal) branch: children carry warm-start bases and pseudocost metadata.
    Branched {
        jb: usize,
        res_obj: f64,
        /// Parent LP solution value at `jb`; used to compute per-unit pseudocost.
        jb_val: f64,
        down_ws: Option<WarmStartBasis>,
        up_ws: Option<WarmStartBasis>,
    },
    /// Non-Optimal bisection on widest integer interval; no warm-start or metadata.
    Split,
}

/// Accumulate per-node timing and profiling counters into `stats`.
fn accumulate_node_stats(
    stats: &mut MipStats,
    elapsed_ms: f64,
    scale_delta: &crate::presolve::scaling::LpScaleProfileSnapshot,
    fallback_delta: &crate::simplex::dual_advanced::SimplexFallbackSnapshot,
    res: &SolverResult,
    is_root: bool,
    node_depth: usize,
) {
    stats.nodes_processed += 1;
    stats.max_depth_seen = stats.max_depth_seen.max(node_depth);
    stats.relaxation_time_total_ms += elapsed_ms;
    if is_root {
        stats.relaxation_time_root_ms = elapsed_ms;
        stats.lp_scale_us_root += scale_delta.scale_us;
        stats.lp_scale_calls_root += scale_delta.calls;
        if let Some(tb) = res.timing_breakdown {
            stats.lp_solve_us_root += tb.solve_us;
        }
    } else {
        stats.relaxation_time_desc_ms += elapsed_ms;
        stats.lp_scale_us_desc += scale_delta.scale_us;
        stats.lp_scale_calls_desc += scale_delta.calls;
        if let Some(tb) = res.timing_breakdown {
            stats.lp_solve_us_desc += tb.solve_us;
        }
    }
    stats.fallback_ub_violation_out_of_scope += fallback_delta.ub_violation_out_of_scope;
    stats.fallback_phase1_bound_violation += fallback_delta.phase1_bound_violation;
    stats.fallback_crash_infeasible += fallback_delta.crash_infeasible;
    match res.status {
        SolveStatus::Optimal => stats.relaxation_time_optimal_ms += elapsed_ms,
        SolveStatus::Infeasible => stats.relaxation_time_infeasible_ms += elapsed_ms,
        _ => {}
    }
    if let Some(tb) = res.timing_breakdown {
        stats.lp_presolve_us_total += tb.presolve_us;
        stats.lp_solve_us_total += tb.solve_us;
        stats.lp_postsolve_us_total += tb.postsolve_us;
    }
    stats.lp_iters_total = stats.lp_iters_total.saturating_add(res.iterations as u64);
}

/// Charge the wall time elapsed since `iter_t0` that is not already
/// attributed to any other named `MipStats` bucket touched during this B&B
/// loop iteration to `node_loop_other_us`.
///
/// `before` is a per-iteration snapshot (`MipStats` is `Copy`) taken at the
/// top of the loop; every field compared here is only ever grown via
/// `saturating_add` within an iteration, so the deltas are non-negative by
/// construction. This is a "self time" computation (total iteration time
/// minus the time already booked to named children), not a global residual
/// against the whole search wall clock, so it stays meaningful even though
/// several of its subtrahends (tree_cut_us, rins_us, ...) are simultaneously
/// new Phase 0 counters.
fn flush_loop_other(stats: &mut MipStats, iter_t0: Instant, before: MipStats) {
    let iter_us = iter_t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
    let relax_delta_us = ((stats.relaxation_time_total_ms - before.relaxation_time_total_ms)
        * 1_000.0)
        .max(0.0) as u64;
    let attributed = stats
        .node_propagation_us
        .saturating_sub(before.node_propagation_us)
        .saturating_add(relax_delta_us)
        .saturating_add(stats.tree_cut_us.saturating_sub(before.tree_cut_us))
        .saturating_add(stats.rins_us.saturating_sub(before.rins_us))
        .saturating_add(stats.rens_us.saturating_sub(before.rens_us))
        .saturating_add(
            stats
                .local_branching_us
                .saturating_sub(before.local_branching_us),
        )
        .saturating_add(
            stats
                .branch_select_us
                .saturating_sub(before.branch_select_us),
        )
        .saturating_add(stats.conflict_us.saturating_sub(before.conflict_us));
    stats.node_loop_other_us = stats
        .node_loop_other_us
        .saturating_add(iter_us.saturating_sub(attributed));
}

/// Whether in-tree GMI/MIR cut re-separation should run at this node: static
/// config gate (`cfg.tree_cuts`) AND `effort::may_run_separation`'s
/// deterministic simplex-iteration-share/dry-streak gate. Extracted from the
/// B&B loop's separation call site so the gate itself is directly
/// unit-testable (see `mip::tests::separation_iter_share_is_enforced`).
fn may_separate_tree_cuts(cfg: &MipConfig, stats: &MipStats) -> bool {
    cfg.tree_cuts && effort::may_run_separation(stats)
}

/// Update `stats.tree_cut_dry_streak` (see `effort::SEPARATION_DRY_STREAK_LIMIT`)
/// after one in-tree separation call. `attempted = false` means the
/// node-selection interval skipped this call entirely (not a real dry
/// attempt — P3-B: this is the explicit flag from `separate_tree_cuts`, not
/// inferred from the iteration count, which can legitimately be 0 for a real
/// attempt), so only a real attempt (rounds actually ran) updates the streak.
/// `accepted` additionally resets the streak on any incumbent improvement
/// elsewhere in the loop (P3-A) — see `maybe_reset_separation_dry_streak` for
/// the periodic, attempt-independent reset. Extracted so the bookkeeping is
/// directly unit-testable (see
/// `mip::tests::separation_dry_streak_is_tracked_and_reset`).
fn record_separation_attempt(stats: &mut MipStats, attempted: bool, accepted: bool) {
    if !attempted {
        return;
    }
    if accepted {
        stats.tree_cut_dry_streak = 0;
    } else {
        stats.tree_cut_dry_streak = stats.tree_cut_dry_streak.saturating_add(1);
    }
}

/// Periodically resets `tree_cut_dry_streak` to 0 regardless of whether
/// separation is currently attempting anything, so a dry streak accumulated
/// early in a long search does not disable separation *permanently* — the
/// tree shape (and hence what separation would see) changes substantially
/// over hundreds of nodes. See
/// `effort::SEPARATION_DRY_STREAK_RESET_NODE_INTERVAL` for the interval's
/// derivation. (P3-A; the complementary incumbent-improvement reset happens
/// inline wherever `stats.incumbent_updates` is incremented.)
fn maybe_reset_separation_dry_streak(stats: &mut MipStats) {
    if stats.nodes_processed > 0
        && stats
            .nodes_processed
            .is_multiple_of(effort::SEPARATION_DRY_STREAK_RESET_NODE_INTERVAL)
    {
        stats.tree_cut_dry_streak = 0;
    }
}

/// Select the variable to branch on for an Optimal relaxation solution.
///
/// With `use_reliability`, runs strong-branching trials for candidates with
/// insufficient pseudocost observations, then falls back to reliability
/// pseudocost selection. Without it, delegates to the configured heuristic
/// (most-infeasible, least-infeasible, etc.).
#[allow(clippy::too_many_arguments)]
fn pick_branch_var<R: Relaxation>(
    problem: &R,
    node_bounds: &[(f64, f64)],
    sol: &[f64],
    obj: f64,
    mask: &[bool],
    integer_vars: &[usize],
    j_to_k: &HashMap<usize, usize>,
    shared: &SolverOptions,
    cfg: &MipConfig,
    deadline: &Option<Instant>,
    parent_warm_start: Option<&WarmStartBasis>,
    pc: &mut PseudocostState,
    stats: &mut MipStats,
    use_reliability: bool,
) -> usize {
    if use_reliability {
        let sb_cands = strong_branch_candidates(sol, mask, integer_vars, cfg.integer_feas_tol, pc);
        let strong_scores = if !sb_cands.is_empty()
            && !deadline_reached(*deadline)
            && effort::may_run_strong_branch(stats)
        {
            measure_strong_branch_scores(
                problem,
                node_bounds,
                sol,
                obj,
                &sb_cands,
                j_to_k,
                shared,
                parent_warm_start,
                pc,
                stats,
            )
        } else {
            HashMap::new()
        };
        // An empty `strong_scores` behaves identically to `None` in
        // `select_branching_variable_reliability` (every lookup misses,
        // falling back to `pc.score`), so branching on emptiness here was
        // dead code — always passing `Some` is equivalent and simpler.
        select_branching_variable_reliability(
            sol,
            mask,
            integer_vars,
            cfg.integer_feas_tol,
            pc,
            Some(&strong_scores),
        )
        .expect("non-integer-feasible Optimal relaxation has a fractional integer var")
    } else {
        select_branching_variable(sol, mask, cfg.integer_feas_tol, cfg.branching)
            .expect("non-integer-feasible Optimal relaxation has a fractional integer var")
    }
}

/// Attempt a RINS heuristic call and update stats/incumbent when it improves.
fn try_rins<R: Relaxation>(
    problem: &R,
    stats: &mut MipStats,
    state: &mut MipState,
    cfg: &MipConfig,
    deadline: &Option<Instant>,
    opts: &SolverOptions,
    rel_sol: &[f64],
) {
    if !cfg.rins_enabled
        || !stats
            .nodes_processed
            .is_multiple_of(heuristics::rins::RINS_INTERVAL)
        || state.incumbent_obj.is_none()
        || !effort::may_run_rins(stats)
    {
        return;
    }
    let rins_res = {
        let inc_sol = match state.incumbent {
            Some(ref inc) => &inc.solution,
            None => return,
        };
        stats.rins_calls += 1;
        let rins_t0 = Instant::now();
        let (res, sub_mip_nodes, sub_mip_iters) = problem.run_rins(
            rel_sol,
            inc_sol,
            cfg,
            deadline,
            effort::rins_iter_budget(stats),
            opts,
        );
        stats.rins_us = stats
            .rins_us
            .saturating_add(rins_t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64);
        stats.rins_iters = stats.rins_iters.saturating_add(sub_mip_iters);
        stats.sub_mip_nodes_total = stats.sub_mip_nodes_total.saturating_add(sub_mip_nodes);
        res
    };
    if let Some(res) = rins_res {
        if state.consider(&res) {
            stats.incumbent_updates += 1;
            stats.rins_improvements += 1;
            stats.tree_cut_dry_streak = 0;
        }
    }
}

/// Attempt a RENS heuristic call on the node LP relaxation and update
/// stats/incumbent when it yields an improving feasible point. Unlike RINS,
/// RENS does not require an existing incumbent (it manufactures one).
fn try_rens<R: Relaxation>(
    problem: &R,
    stats: &mut MipStats,
    state: &mut MipState,
    cfg: &MipConfig,
    deadline: &Option<Instant>,
    opts: &SolverOptions,
    rel_sol: &[f64],
) {
    // Checked before the (state-mutating) `should_try` decision below so a
    // denied budget never consumes the guaranteed first-attempt flag: RENS
    // still gets that guaranteed attempt on a later call once the budget
    // allows it.
    if !cfg.rens_enabled || !effort::may_run_rens(stats) {
        return;
    }
    let should_try = if state.incumbent_obj.is_none() {
        if !state.rens_first_incumbent_attempted {
            state.rens_first_incumbent_attempted = true;
            true
        } else {
            stats
                .nodes_processed
                .is_multiple_of(heuristics::rens::RENS_INTERVAL_WITH_INCUMBENT)
        }
    } else {
        stats
            .nodes_processed
            .is_multiple_of(heuristics::rens::RENS_INTERVAL_WITH_INCUMBENT)
    };
    if !should_try {
        return;
    }
    stats.rens_calls += 1;
    let rens_t0 = Instant::now();
    let (rens_res, sub_mip_nodes, sub_mip_iters) = problem.run_rens(
        rel_sol,
        cfg,
        deadline,
        effort::rens_iter_budget(stats),
        opts,
    );
    stats.rens_us = stats
        .rens_us
        .saturating_add(rens_t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64);
    stats.rens_iters = stats.rens_iters.saturating_add(sub_mip_iters);
    stats.sub_mip_nodes_total = stats.sub_mip_nodes_total.saturating_add(sub_mip_nodes);
    if let Some(res) = rens_res {
        if state.consider(&res) {
            stats.incumbent_updates += 1;
            stats.rens_improvements += 1;
            stats.tree_cut_dry_streak = 0;
        }
    }
}

/// Attempt a local-branching heuristic call around the current incumbent and
/// update stats/incumbent when it strictly improves.
fn try_local_branching<R: Relaxation>(
    problem: &R,
    stats: &mut MipStats,
    state: &mut MipState,
    cfg: &MipConfig,
    deadline: &Option<Instant>,
    opts: &SolverOptions,
) {
    if !cfg.local_branching_enabled
        || !stats
            .nodes_processed
            .is_multiple_of(heuristics::local_branching::LOCAL_BRANCHING_INTERVAL)
        || !effort::may_run_local_branching(stats)
    {
        return;
    }
    let lb_res = {
        let inc_sol = match state.incumbent {
            Some(ref inc) => &inc.solution,
            None => return,
        };
        stats.local_branching_calls += 1;
        let lb_t0 = Instant::now();
        let (res, sub_mip_nodes, sub_mip_iters) = problem.run_local_branching(
            inc_sol,
            cfg,
            deadline,
            effort::local_branching_iter_budget(stats),
            opts,
        );
        stats.local_branching_us = stats
            .local_branching_us
            .saturating_add(lb_t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64);
        stats.local_branching_iters = stats.local_branching_iters.saturating_add(sub_mip_iters);
        stats.sub_mip_nodes_total = stats.sub_mip_nodes_total.saturating_add(sub_mip_nodes);
        res
    };
    if let Some(res) = lb_res {
        if state.consider(&res) {
            stats.incumbent_updates += 1;
            stats.local_branching_improvements += 1;
            stats.tree_cut_dry_streak = 0;
        }
    }
}

/// Determine what the B&B loop should do after a node's relaxation result.
///
/// Handles both trusted (Optimal) and non-trusted paths: pseudocost updates,
/// integer-feasibility detection, RINS, max-depth, reduced-cost fixing,
/// branching variable selection, and child bound computation.
fn process_node_outcome<R: Relaxation>(
    problem: &R,
    node: &mut MipNode,
    res: &SolverResult,
    trusted: bool,
    ctx: &SearchCtx<'_>,
    s: &mut SearchState,
    is_root: bool,
) -> NodeAction {
    let SearchCtx {
        cfg,
        shared,
        mask,
        integer_vars,
        j_to_k,
        deadline,
        use_reliability,
        ..
    } = *ctx;
    let deadline = &deadline;
    let dive_start_depth = s.dive_start_depth;
    let (state, stats, pc) = (&mut s.state, &mut s.stats, &mut s.pc);
    if trusted {
        // Pseudocost update: record per-unit cost (delta / fractionality) so
        // that score() can correctly predict gains at different fractionalities.
        if use_reliability {
            if let Some(jb) = node.branch_var {
                // `node.branch_var` is only ever set (via `child_branched`) to the
                // `jb` returned by `pick_branch_var`, which always returns a `j`
                // that is a key of `j_to_k` (either from `integer_vars` directly,
                // via `select_branching_variable_reliability`, or via the
                // `mask`-gated `select_branching_variable`, where `mask` and
                // `j_to_k` are built from the same `integer_vars`).
                let &k = j_to_k
                    .get(&jb)
                    .expect("node.branch_var is always a key of j_to_k");
                let delta = (res.objective - node.parent_obj).max(0.0);
                let v = node.branch_parent_val;
                if node.branch_up {
                    let f_up = v.ceil() - v;
                    if f_up > 1e-12 {
                        pc.record_up(k, delta / f_up);
                    }
                } else {
                    let f_down = v - v.floor();
                    if f_down > 1e-12 {
                        pc.record_down(k, delta / f_down);
                    }
                }
            }
        }
        if is_root {
            stats.root_lp_bound = res.objective;
        }
        let node_lb = node.lower_bound.max(res.objective);

        // Post-solve bound pruning.
        if let Some(inc) = state.incumbent_obj {
            if should_prune(node_lb, Some(inc), cfg.gap_tol) {
                stats.pruned += 1;
                return NodeAction::Skip { end_dive: false };
            }
        }

        // Integer-feasible leaf.
        if is_integer_feasible(&res.solution, mask, cfg.integer_feas_tol) {
            // Codex review (P2, follow-up to `within_gap`'s false-Optimal fix):
            // a `res` that reaches here claims a trusted (`Optimal`)
            // integer-feasible leaf, but if its objective/solution is
            // non-finite it is a corrupt candidate, not proof this region is
            // exhausted. `MipState::consider` already refuses to *adopt* it
            // (its own `is_finite_candidate()` guard), but silently `Skip`ping
            // the node regardless would still discard the region as if fully
            // resolved — with no other open node left, `finalize_no_incumbent`
            // could then report a false `Infeasible` (its `fully_resolved`
            // check only looks at `had_open`/`queue_empty`, not at whether a
            // candidate was rejected here). Fold it into the open region
            // instead, via `node.lower_bound` (the last *trustworthy* bound —
            // not `node_lb`, which folds in the corrupt `res.objective`).
            if !res.is_finite_candidate() {
                return NodeAction::OpenLb {
                    node_lb: node.lower_bound,
                    uncertain: true,
                    end_dive: true,
                };
            }
            if state.consider(res) {
                stats.incumbent_updates += 1;
                stats.tree_cut_dry_streak = 0;
            }
            return NodeAction::Skip { end_dive: true };
        }

        // Primal heuristics on this fractional node: RENS rounds the LP point to
        // manufacture an incumbent; RINS and local branching refine an existing one.
        try_rens(problem, stats, state, cfg, deadline, shared, &res.solution);
        try_rins(problem, stats, state, cfg, deadline, shared, &res.solution);
        try_local_branching(problem, stats, state, cfg, deadline, shared);

        // Max-depth limit.
        if node.depth + 1 > cfg.max_depth {
            return NodeAction::OpenLb {
                node_lb,
                uncertain: false,
                end_dive: true,
            };
        }

        // End dive when it has descended MAX_DIVE_DEPTH levels below the pivot.
        let end_dive = node.depth >= dive_start_depth + MAX_DIVE_DEPTH;

        // Reduced-cost fixing: tighten integer-variable bounds for children.
        if let Some(inc_obj) = state.incumbent_obj {
            stats.rc_vars_fixed +=
                reduced_cost_fixing(res, inc_obj, &mut node.var_bounds, problem.integer_vars());
        }

        // Select branching variable and compute child bounds.
        let branch_select_t0 = Instant::now();
        let jb = pick_branch_var(
            problem,
            &node.var_bounds,
            &res.solution,
            res.objective,
            mask,
            integer_vars,
            j_to_k,
            shared,
            cfg,
            deadline,
            res.warm_start_basis.as_ref(),
            pc,
            stats,
            use_reliability,
        );
        stats.branch_select_us = stats.branch_select_us.saturating_add(
            branch_select_t0
                .elapsed()
                .as_micros()
                .min(u128::from(u64::MAX)) as u64,
        );
        let (down, up) = branch_bounds(&node.var_bounds, jb, res.solution[jb]);
        let child_ws = res.warm_start_basis.clone();
        let down_ws = if bound_layout_changes(&node.var_bounds, &down, jb) {
            None
        } else {
            child_ws.clone()
        };
        let up_ws = if bound_layout_changes(&node.var_bounds, &up, jb) {
            None
        } else {
            child_ws
        };
        NodeAction::PushChildren {
            node_lb,
            down,
            up,
            kind: ChildKind::Branched {
                jb,
                res_obj: res.objective,
                jb_val: res.solution[jb],
                down_ws,
                up_ws,
            },
            end_dive,
        }
    } else {
        // Non-Optimal relaxation: bisect on the widest integer interval.
        let node_lb = node.lower_bound;
        match widest_splittable_integer(&node.var_bounds, mask) {
            Some(_) if node.depth + 1 > cfg.max_depth => NodeAction::OpenLb {
                node_lb,
                uncertain: true,
                end_dive: true,
            },
            Some(jb) => {
                let end_dive = node.depth >= dive_start_depth + MAX_DIVE_DEPTH;
                let (down, up) = split_integer_box(&node.var_bounds, jb);
                NodeAction::PushChildren {
                    node_lb,
                    down,
                    up,
                    kind: ChildKind::Split,
                    end_dive,
                }
            }
            None => NodeAction::OpenLb {
                node_lb,
                uncertain: true,
                end_dive: true,
            },
        }
    }
}

/// Classify the terminal status when no integer-feasible incumbent was found.
///
/// `Infeasible` may be claimed **only** when the whole tree was resolved: the
/// queue is empty, there was no budget interruption, and **no region was left
/// unexplored** (`had_open == false`). An unexplored region (a depth/budget limit,
/// or an unsolvable no-interior relaxation that could not be bisected) means we
/// cannot prove infeasibility, so a no-solution status is returned instead — never
/// a silent false `Infeasible`.
fn finalize_no_incumbent(
    interrupted: bool,
    had_open: bool,
    queue_empty: bool,
    deadline_stop: bool,
) -> SolverResult {
    let fully_resolved = !interrupted && !had_open && queue_empty;
    if fully_resolved {
        SolverResult::infeasible()
    } else if deadline_stop {
        no_solution_result(SolveStatus::Timeout)
    } else {
        no_solution_result(SolveStatus::MaxIterations)
    }
}

/// Reject an `integer_vars` index out of range for `num_vars` before it
/// reaches `integer_mask`'s `assert!`.
///
/// `MilpProblem`/`MiqpProblem` are public structs with a `pub integer_vars`
/// field: `new()` validates it via `normalize_integer_vars`, but a caller can
/// build the struct with a literal (bypassing `new()` entirely, all fields
/// `pub`) or mutate `integer_vars` afterward, so `solve_milp`/`solve_miqp`
/// must re-check it themselves at the solve entry rather than trust
/// construction-time validation -- the same defense already applied to
/// `MisocpProblem::integers` and `NonconvexQcqp`'s `integers` parameter
/// (Codex review R3 horizontal expansion, nonconvex.rs:763).
fn validate_integer_vars(integer_vars: &[usize], num_vars: usize) -> Result<(), MipProblemError> {
    if let Some(&j) = integer_vars.iter().find(|&&j| j >= num_vars) {
        return Err(MipProblemError::InvalidIntegerVar { index: j, num_vars });
    }
    Ok(())
}

/// Boolean mask of length `num_vars`; `true` where the variable is integral.
pub(crate) fn integer_mask(num_vars: usize, integer_vars: &[usize]) -> Vec<bool> {
    let mut mask = vec![false; num_vars];
    for &j in integer_vars {
        assert!(
            j < num_vars,
            "integer variable index {} out of range for {} variables",
            j,
            num_vars
        );
        mask[j] = true;
    }
    mask
}

/// A result tagging a non-convex MIQP/MIQCP (non-PSD `Q` or nonconvex
/// quadratic constraint) as out of scope.
fn nonconvex_result() -> SolverResult {
    SolverResult {
        status: SolveStatus::NonConvex(
            "convex MIQP/MIQCP only: Q is not PSD or a quadratic constraint is nonconvex"
                .to_string(),
        ),
        objective: f64::INFINITY,
        solution: vec![],
        ..Default::default()
    }
}

/// Round the integer components of `sol` to exact integers (relaxation noise removal).
fn round_integers(mut sol: Vec<f64>, integer_vars: &[usize]) -> Vec<f64> {
    for &j in integer_vars {
        assert!(
            j < sol.len(),
            "integer variable index {} out of range for solution length {}",
            j,
            sol.len()
        );
        sol[j] = sol[j].round();
    }
    sol
}

/// Returns `true` when tightening var `j`'s bound changes the standard-form
/// column layout vs the parent. An infinite bound becoming finite (ub: ∞→boxed,
/// or lb: free→lower-bounded) changes the number of structural columns or adds
/// a UB constraint row, making the parent basis index-incompatible.
fn bound_layout_changes(
    parent_bounds: &[(f64, f64)],
    child_bounds: &[(f64, f64)],
    j: usize,
) -> bool {
    let (p_lb, p_ub) = parent_bounds[j];
    let (c_lb, c_ub) = child_bounds[j];
    (p_ub.is_infinite() && c_ub.is_finite()) || (p_lb.is_infinite() && c_lb.is_finite())
}

/// A result carrying no usable solution, tagged with `status`.
fn no_solution_result(status: SolveStatus) -> SolverResult {
    SolverResult {
        status,
        objective: f64::INFINITY,
        solution: vec![],
        ..Default::default()
    }
}

/// Maximum distance from a variable value to a bound for it to be considered "at" that bound.
///
/// Used in reduced-cost fixing to decide whether `x[j]` is at its lower or upper bound.
/// Matches typical LP primal feasibility tolerances.
const BOUND_AT_TOL: f64 = 1e-8;

/// Apply reduced-cost fixing to variable bounds in preparation for branching.
///
/// For each integer variable `j` where the LP value is at a bound and the
/// reduced cost exceeds the MIP gap, `x[j]` cannot improve on the incumbent
/// by moving away from that bound — so we fix it there:
///
/// - `x[j] ≈ lb[j]` and `rc[j] > gap`:  fix `x[j] = ceil(lb)`
/// - `x[j] ≈ ub[j]` and `-rc[j] > gap`: fix `x[j] = floor(ub)`
///
/// Bounds are rounded to the nearest feasible integer before fixing. If the
/// rounded bounds are inconsistent (`ceil(lb) > floor(ub)`), the fix is skipped.
///
/// `gap = incumbent_obj - lp_result.objective` (must be positive).
/// Returns the number of variables fixed.
pub(crate) fn reduced_cost_fixing(
    lp_result: &SolverResult,
    incumbent_obj: f64,
    node_bounds: &mut [(f64, f64)],
    integer_vars: &[usize],
) -> usize {
    if lp_result.reduced_costs.is_empty() || lp_result.solution.is_empty() {
        return 0;
    }
    assert_eq!(
        lp_result.reduced_costs.len(),
        node_bounds.len(),
        "reduced-cost fixing requires one reduced cost per variable"
    );
    assert_eq!(
        lp_result.solution.len(),
        node_bounds.len(),
        "reduced-cost fixing requires one solution value per variable"
    );
    let gap = incumbent_obj - lp_result.objective;
    if gap <= 0.0 {
        return 0;
    }
    let rc = &lp_result.reduced_costs;
    let x = &lp_result.solution;
    let mut count = 0usize;
    for &j in integer_vars {
        assert!(
            j < node_bounds.len(),
            "integer variable index {} out of range for {} variables",
            j,
            node_bounds.len()
        );
        let (lb, ub) = node_bounds[j];
        if (lb - ub).abs() < BOUND_AT_TOL {
            continue; // already fixed
        }
        let int_lb = lb.ceil();
        let int_ub = ub.floor();
        if int_lb > int_ub + BOUND_AT_TOL {
            continue; // empty integer range after rounding
        }
        let xj = x[j];
        let rcj = rc[j];
        if (xj - lb).abs() <= BOUND_AT_TOL && rcj > gap {
            node_bounds[j] = (int_lb, int_lb);
            count += 1;
        } else if (xj - ub).abs() <= BOUND_AT_TOL && -rcj > gap {
            node_bounds[j] = (int_ub, int_ub);
            count += 1;
        }
    }
    count
}

/// Incumbent (best integer-feasible upper bound) tracking.
///
/// With `shared = None` (serial search) this is a plain local best. A
/// parallel worker instead carries a handle onto the search-wide incumbent:
/// `consider` arbitrates against it under a lock, and `sync_shared` refreshes
/// the local view. The local objective is then always a *past* value of the
/// shared one, which only ever decreases — so the bound comparisons this
/// state feeds (`should_prune`) can only be too conservative, never prune a
/// region that still holds the optimum.
struct MipState {
    incumbent: Option<SolverResult>,
    incumbent_obj: Option<f64>,
    rens_first_incumbent_attempted: bool,
    shared: Option<std::sync::Arc<parallel::SharedIncumbent>>,
}

impl MipState {
    fn new() -> Self {
        Self {
            incumbent: None,
            incumbent_obj: None,
            rens_first_incumbent_attempted: false,
            shared: None,
        }
    }

    /// A worker-local view backed by the search-wide incumbent `shared`.
    fn shared(shared: std::sync::Arc<parallel::SharedIncumbent>) -> Self {
        let mut s = Self::new();
        s.shared = Some(shared);
        s.sync_shared();
        s
    }

    /// Adopt `res` as the new incumbent if it strictly improves the objective.
    /// Returns `true` when the incumbent changed.
    ///
    /// Codex review (P2, follow-up to the `within_gap` false-Optimal fix):
    /// rejects a candidate whose `objective`/`solution` isn't
    /// `is_finite_candidate()` outright, regardless of `status`. Every one of
    /// this function's 5 call sites (the "trusted" B&B leaf, the FP-seeded
    /// `initial_incumbent`, RINS, RENS, local branching) trusts its own
    /// upstream contract to only hand a genuine solution here; this is the
    /// single point that actually re-verifies it before a result becomes "the
    /// incumbent" — without it, a poisoned candidate is merely *reported*
    /// honestly instead of falsely as `Optimal` (the earlier `within_gap`
    /// fix), but is still wrongly *adopted*, e.g. as `SuboptimalSolution` with
    /// a non-finite objective, violating that status's documented "verified
    /// feasible point" contract.
    ///
    /// Integration note (parallel B&B merge): checked *before* the
    /// `self.shared` delegation so a poisoned candidate is rejected
    /// uniformly whether this search is serial or parallel — the parallel
    /// path also gets its own independent guard in
    /// `SharedIncumbent::consider` (the search-wide `initial_incumbent` seed
    /// calls that directly, bypassing this function entirely).
    fn consider(&mut self, res: &SolverResult) -> bool {
        if !res.is_finite_candidate() {
            return false;
        }
        if let Some(shared) = self.shared.clone() {
            let improved = shared.consider(res);
            self.sync_shared();
            return improved;
        }
        let better = match self.incumbent_obj {
            None => true,
            Some(o) => res.objective < o,
        };
        if better {
            self.incumbent_obj = Some(res.objective);
            self.incumbent = Some(res.clone());
        }
        better
    }

    /// Refresh the local view from the search-wide incumbent. No-op for a
    /// serial search, and lock-free unless the shared value actually improved.
    fn sync_shared(&mut self) {
        let Some(shared) = &self.shared else { return };
        if let Some((obj, res)) = shared.take_better_than(self.incumbent_obj) {
            self.incumbent_obj = Some(obj);
            self.incumbent = Some(res);
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_parallel;
