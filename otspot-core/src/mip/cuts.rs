//! Root cutting planes for MILP: GMI, MIR, cover, clique, and implied-bound cuts.
//!
//! Cuts tighten the LP relaxation without removing integer-feasible points.
//! Uses the primal-simplex standard form (`build_standard_form`) so every
//! nonbasic sits at 0, simplifying the tableau row formulae.
//!
//! Pipeline per round:
//!   1. Solve the LP relaxation (primal simplex, no presolve).
//!   2. For each fractional integer basic, form the tableau row and emit a cut.
//!   3. Back-substitute to original variables (`G·x >= rhs`).
//!
//! After the GMI/MIR rounds, a structural cut phase runs cover, clique, and
//! implied-bound cuts directly from the constraint matrix and LP solution.

use crate::basis::{BasisManager, LuBasis};
use crate::options::{
    MipConfig, SimplexMethod, SolverOptions, WarmStartBasis, DEFAULT_MAX_CUT_ROUNDS,
};
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::simplex::{build_standard_form, StandardForm};
use crate::tolerances::{feas_rel_tol, ZERO_TOL};
use otspot_num::linalg::timeout::deadline_reached;
use otspot_num::sparse::CscMatrix;

use super::cut_pool::{Cut, CutPool};
use super::problem::MilpProblem;

mod flow_cover;
mod knapsack_cover;

/// Maximum cuts added per round.
const MAX_CUTS_PER_ROUND: usize = 64;
/// Relative LP-bound improvement below which cut rounds stop.
const MIN_OBJ_IMPROVEMENT_REL: f64 = 1e-4;
/// Reject a cut whose coefficient magnitudes span more than this ratio.
const GMI_MAX_COEF_DYNAMISM: f64 = 1e9;
/// Fraction of the remaining solve budget root cut generation may consume.
const CUT_TIME_FRACTION: f64 = 0.3;

/// In-tree separation runs at nodes whose depth is a multiple of this. Cuts near
/// a subtree's root close gaps that propagate to all its descendants, so effort
/// concentrates there rather than at every (mostly redundant) node.
const TREE_CUT_DEPTH_INTERVAL: usize = 4;
/// In-tree separation also runs every this-many processed nodes regardless of
/// depth, so long dives still receive periodic re-tightening.
const TREE_CUT_NODE_INTERVAL: usize = 32;
/// Minimum relative node-bound gain for a re-separated node LP to be accepted.
/// Below this the cuts did not meaningfully tighten the bound, so the original
/// (warm-startable) result is kept to avoid pessimising children with a
/// basis-less re-solve. Mirrors the root [`MIN_OBJ_IMPROVEMENT_REL`] threshold.
const MIN_TREE_CUT_GAIN_REL: f64 = 1e-4;
/// Maximum separation rounds per node. Each round re-solves the node LP and adds
/// a GMI (even) / MIR (odd) batch; kept small so a separating node costs only a
/// few extra LP solves. Rounds also stop early when the bound stalls.
const TREE_CUT_MAX_ROUNDS: usize = 4;
/// Minimum useful `max_iters` for a single cold cut-LP solve within
/// `separate_tree_cuts`, as a multiple of `dim = lp.num_vars +
/// lp.num_constraints`: a round whose remaining allowance is below this
/// threshold is skipped (deferred to a later attempt) rather than run.
///
/// A cold primal simplex from an all-slack start empirically needs on the
/// order of one to a few pivots per dimension for a well-conditioned LP, so
/// 4x is below what even a normal (non-pathological), non-warm-started
/// resolve of the current cut-augmented LP typically needs — an allowance
/// smaller than this cannot realistically reach `Optimal`, so attempting the
/// solve only pays cold-solve setup cost for a result that will be discarded
/// (`cut_res.status != SolveStatus::Optimal` breaks the round immediately
/// below).
///
/// markshare_4_0 regression fix: this constant previously *inflated* a
/// too-small remaining allowance up to this floor (`max(remaining, dim *
/// mult)`) so every approved attempt still ran at least one full round
/// regardless of how little the share budget actually granted. As RINS/RENS/
/// local-branching (see `heuristics::SUB_MIP_MIN_LP_ITERS`), that let a
/// round attempt overshoot its nominal share by design; here it also meant
/// every eligible node paid at least `2 * dim * 4` iterations of guaranteed
/// overhead once approved, however small its share. Using the same threshold
/// as a *skip* condition instead — matching the sub-MIP pattern — bounds a
/// round to only run when it can realistically finish, deferring the rest to
/// a later attempt once the (growing) share ceiling clears this threshold
/// again.
const TREE_CUT_MIN_SOLVE_ITER_DIM_MULT: u64 = 4;

/// `dim * TREE_CUT_MIN_SOLVE_ITER_DIM_MULT` — see that constant's doc.
fn tree_cut_min_useful_iters(lp: &LpProblem) -> u64 {
    let dim = (lp.num_vars + lp.num_constraints) as u64;
    dim.saturating_mul(TREE_CUT_MIN_SOLVE_ITER_DIM_MULT)
}

/// `lp.num_vars + lp.num_constraints`: same `dim` convention as
/// [`tree_cut_min_useful_iters`], and the pre-expansion proxy this module's
/// fixed-cost surcharge (below) uses for `build_standard_form(lp)`'s own
/// `m + n_total`. Reading the two struct fields directly, this is free —
/// unlike calling `build_standard_form` itself, which is exactly the cost
/// the surcharge is charging *for*, not a way to measure it. It is a fair
/// proxy, not an exact substitute: UB-row expansion adds at most `num_vars`
/// rows and free-variable splitting at most `num_vars` columns, so `sf.m +
/// sf.n_total` is bounded by `dim + 2 * lp.num_vars <= 3 * dim` — the two
/// track each other within a constant factor.
fn tree_cut_dim(lp: &LpProblem) -> u64 {
    (lp.num_vars + lp.num_constraints) as u64
}

/// Iteration-equivalent cost charged by [`tree_cut_construction_surcharge`]
/// per unit of [`tree_cut_dim`] per `build_standard_form`-equivalent
/// construction is `TREE_CUT_BUILD_ITER_COST_PER_DIM /
/// TREE_CUT_BUILD_ITER_COST_DIVISOR` = 1/4.
///
/// A single simplex iteration's dominant cost is a sparse triangular solve
/// (FTRAN/BTRAN) over the current LU factors, touching on the order of `m`
/// (basis size) nonzeros; `build_standard_form` allocates and populates
/// arrays of that same order (variable shifts, UB rows, slack columns) —
/// the same *order* of work as one iteration's triangular solve, which is
/// what fixes the numerator at 1 rather than some other order-of-magnitude
/// constant: the bug this surcharge fixes is the iteration budget *ignoring*
/// this cost entirely, so any nonzero, order-correct charge restores it to
/// the accounting.
///
/// The `/ 4` divisor is the one number here actually fit to data, not
/// derived from the complexity argument above (which only pins the order of
/// magnitude, not the constant): coefficient 1 (no divisor) reduced
/// `markshare_4_0`'s accepted-round count enough to fix its regression
/// (Optimal 507.3s) but, measured directly, also broke a small/fast search
/// unrelated to `markshare_4_0`'s scale — `gt2 --timeout 60` went from a
/// deterministic 100-node `Optimal` to a 3,744-node `Timeout` purely from
/// this surcharge's magnitude (before the `MipStats::tree_cut_overhead_
/// iters` isolation fix below existed to explain *why*: at coefficient 1 the
/// surcharge was simply too large per round for a fixed-point search this
/// small). `1/4` was the value found, by direct measurement of both
/// instances together, to land `markshare_4_0`'s accepted-round count at
/// 30,251–31,269 — matching its own pre-warm-start cold-equivalent round
/// count (31,268) — while `gt2 --timeout 60` ×3 stays at the deterministic
/// 100-node `Optimal` (its separation now self-gates to 0 accepted rounds
/// entirely, at this instance's dimension, rather than over-firing).
const TREE_CUT_BUILD_ITER_COST_PER_DIM: u64 = 1;
const TREE_CUT_BUILD_ITER_COST_DIVISOR: u64 = 4;

/// `build_standard_form`-equivalent construction count charged per LP per
/// call site — a structural count fixed by this module's own call graph
/// (`separate_tree_cuts`, `generate_round`, `extend_basis_for_new_rows`,
/// `tree_cut_resolve`), not data-dependent. Applied via
/// [`tree_cut_construction_surcharge`] at each call site immediately after
/// it actually runs, so a round that breaks early (e.g. `generate_round`
/// finds no cuts and never reaches the round-end validate) is charged only
/// for the constructions it actually performed.
///
/// - Round-start warm re-solve (round > 0, via [`tree_cut_resolve`]): its
///   own shape-check `build_standard_form` call plus the warm solve's own
///   internal `simplex::entry` construction (`presolve: false`, so exactly
///   one) = 2 ([`TREE_CUT_BUILDS_ROUND_START_WARM`]).
/// - Round-start cold bootstrap (round 0, via [`solve_cut_lp`]): only the
///   cold solve's own internal construction (no separate shape check) = 1
///   ([`TREE_CUT_BUILDS_ROUND_START_COLD`]).
/// - [`generate_round`]'s own build, charged every round that reaches it
///   (whether or not it finds a cut) = 1
///   ([`TREE_CUT_BUILDS_GENERATE_ROUND`]).
/// - Round-end validate ([`extend_basis_for_new_rows`]'s build, plus
///   [`tree_cut_resolve`]'s shape-check build, plus the warm solve's own
///   internal construction) = 3 ([`TREE_CUT_BUILDS_ROUND_END`]).
///
/// markshare_4_0 regression fix (Phase 3b): warm-started re-solves cut each
/// round's *iteration* cost by roughly 10x, but not this per-round fixed
/// cost, which the iteration budget was blind to — with rounds no longer
/// throttled by iterations, `TREE_CUT_MAX_ROUNDS`-bounded but far more
/// frequent cheap rounds fit inside the same `effort::separation_iter_
/// budget`, ~3x more on `markshare_4_0` (31,268 → 92,711 accepted rounds in
/// a 1000s run), whose accumulated fixed cost alone consumed 80% of wall
/// clock (`tree_cut_us_pct_wall` 68.65% → 79.95%) despite fewer nodes
/// processed overall (1,813,460 → 1,328,286) — Optimal (921.6s) regressed
/// to Timeout (1000s).
const TREE_CUT_BUILDS_ROUND_START_WARM: u64 = 2;
const TREE_CUT_BUILDS_ROUND_START_COLD: u64 = 1;
const TREE_CUT_BUILDS_GENERATE_ROUND: u64 = 1;
const TREE_CUT_BUILDS_ROUND_END: u64 = 3;

/// Deterministic fixed-cost surcharge (iteration-equivalent units, added to
/// `iters_spent` in [`separate_tree_cuts`]) for `n_builds`
/// `build_standard_form`-equivalent constructions of `lp`'s standard form.
/// See [`TREE_CUT_BUILD_ITER_COST_PER_DIM`] and the `TREE_CUT_BUILDS_*`
/// constants for the per-unit cost and call-count derivations.
fn tree_cut_construction_surcharge(lp: &LpProblem, n_builds: u64) -> u64 {
    n_builds
        .saturating_mul(tree_cut_dim(lp))
        .saturating_mul(TREE_CUT_BUILD_ITER_COST_PER_DIM)
        / TREE_CUT_BUILD_ITER_COST_DIVISOR
}

/// A generated cut `coeffs · x >= rhs` over the original variable space.
struct CutRow {
    coeffs: Vec<f64>,
    rhs: f64,
}

/// Classification of an original variable's structural standard-form column.
#[derive(Clone, Copy)]
enum StructKind {
    LbShift,
    UbOnly,
    FreeSplit,
}

/// Per-structural-column metadata (length `n_shifted`).
#[derive(Clone, Copy)]
struct StructCol {
    var: usize,
    offset: f64,
    kind: StructKind,
    integral: bool,
}

/// What a standard-form slack column measures, in original variables.
#[derive(Clone, Copy)]
enum SlackKind {
    ConstraintLe(usize),
    ConstraintGe(usize),
    UbRow(usize),
}

/// Which cutting-plane formula to apply.
#[derive(Clone, Copy)]
enum CutKind {
    Gmi,
    Mir,
}

/// Append GMI and MIR cuts at the root, returning the augmented problem.
///
/// Rounds alternate GMI (even) / MIR (odd). After the tableau-based rounds,
/// a structural phase adds cover, clique, and implied-bound cuts.
/// Multi-round generation uses Ge rows internally (preserving the original
/// simplex path). After all rounds, added Ge cut rows are converted to Le
/// (`−g·x ≤ −rhs`) before returning, so B&B node solves use slack variables
/// (coeff +1) rather than surplus variables (coeff −1).
pub(crate) fn add_root_cuts(
    milp: &MilpProblem,
    options: &SolverOptions,
    cfg: &MipConfig,
) -> MilpProblem {
    let integer_mask = super::integer_mask(milp.lp.num_vars, &milp.integer_vars);
    let max_rounds = if cfg.max_cut_rounds == 0 {
        DEFAULT_MAX_CUT_ROUNDS
    } else {
        cfg.max_cut_rounds
    };

    // `committed` is always an LP that solves cleanly; cuts are committed only
    // after the candidate is re-validated with the *same* solver the B&B uses.
    let m_orig = milp.lp.num_constraints;
    let mut committed = milp.lp.clone();
    let mut prev_obj: Option<f64> = None;
    let max_total_cuts = m_orig.max(MAX_CUTS_PER_ROUND);
    let mut total_cuts = 0usize;

    let cut_deadline = options.deadline.map(|d| {
        let now = std::time::Instant::now();
        now + d.saturating_duration_since(now).mul_f64(CUT_TIME_FRACTION)
    });

    // GMI / MIR tableau-based rounds.
    for round_idx in 0..max_rounds {
        if deadline_reached(cut_deadline) {
            break;
        }
        let res = solve_cut_lp(&committed, options, cut_deadline, None);
        if res.status != SolveStatus::Optimal {
            break;
        }
        let Some(ws) = res.warm_start_basis.as_ref() else {
            break;
        };
        if let Some(po) = prev_obj {
            let scale = 1.0_f64.max(po.abs());
            if (res.objective - po).abs() <= MIN_OBJ_IMPROVEMENT_REL * scale {
                break;
            }
        }
        prev_obj = Some(res.objective);

        let kind = if round_idx % 2 == 0 {
            CutKind::Gmi
        } else {
            CutKind::Mir
        };
        let cuts = generate_round(&committed, &integer_mask, &res.solution, &ws.basis, kind);
        if cuts.is_empty() {
            break;
        }
        let candidate = append_ge_rows_with_integer_mask(&committed, &cuts, &integer_mask);
        let check = solve_validate(&candidate, options, cut_deadline, None);
        if check.status != SolveStatus::Optimal {
            break;
        }
        committed = candidate;
        total_cuts += cuts.len();
        if total_cuts >= max_total_cuts {
            break;
        }
    }

    // Structural cut phase: cover, clique, implied-bound.
    if total_cuts < max_total_cuts && !deadline_reached(cut_deadline) {
        let res = solve_cut_lp(&committed, options, cut_deadline, None);
        if res.status == SolveStatus::Optimal {
            let budget = max_total_cuts.saturating_sub(total_cuts);
            let mut structural: Vec<CutRow> = Vec::new();
            structural.extend(generate_cover_cuts(
                &committed,
                &integer_mask,
                &res.solution,
            ));
            structural.extend(generate_clique_cuts(
                &committed,
                &integer_mask,
                &res.solution,
            ));
            structural.extend(generate_implied_bound_cuts(
                &committed,
                &integer_mask,
                &res.solution,
            ));
            structural.extend(flow_cover::generate_flow_cover_cuts(
                &committed,
                &integer_mask,
                &res.solution,
            ));
            structural.extend(knapsack_cover::generate_lifted_knapsack_cover_cuts(
                &committed,
                &integer_mask,
                &res.solution,
            ));
            structural.truncate(budget);
            if !structural.is_empty() {
                let candidate =
                    append_ge_rows_with_integer_mask(&committed, &structural, &integer_mask);
                let check = solve_validate(&candidate, options, cut_deadline, None);
                if check.status == SolveStatus::Optimal {
                    committed = candidate;
                }
            }
        }
    }

    // Convert added Ge cut rows to Le before handing to B&B.
    let lp = convert_cuts_to_le_with_integer_mask(committed, m_orig, &integer_mask);

    let le_check = solve_validate(&lp, options, cut_deadline, None);
    let final_lp = if le_check.status == SolveStatus::Optimal {
        lp
    } else {
        milp.lp.clone()
    };

    MilpProblem {
        lp: final_lp,
        integer_vars: milp.integer_vars.clone(),
    }
}

/// `max_iters` is `None` for root cut generation (`add_root_cuts`, which has
/// no per-call iteration budget concept — bounded only by `cut_deadline`) and
/// `Some(remaining)` for in-tree separation (`separate_tree_cuts`), whose
/// caller must bound this single cold solve's own simplex work: without it, a
/// solve with `presolve: false` and no warm start can run past its round's
/// share of `effort::separation_iter_budget` on its own (Codex review, P1 —
/// confirmed by `dcmulti`'s single 27s cut-LP solve under Phase 1c re-bench).
fn solve_validate(
    lp: &LpProblem,
    options: &SolverOptions,
    deadline: Option<std::time::Instant>,
    max_iters: Option<u64>,
) -> crate::problem::SolverResult {
    let opts = SolverOptions {
        presolve: false,
        recover_warm_start_basis: false,
        warm_start: None,
        warm_start_lp: None,
        deadline,
        timeout_secs: None,
        max_iters,
        primal_tol: options.primal_tol,
        dual_tol: options.dual_tol,
        threads: options.threads,
        tolerance: options.tolerance,
        cancel_flag: options.cancel_flag.clone(),
        ..SolverOptions::default()
    };
    crate::lp::solve_lp_with(lp, &opts)
}

/// Builds the [`SolverOptions`] for [`solve_cut_lp`]. Split out (mirroring
/// [`tree_cut_warm_options`]) so the options themselves — including
/// `tolerance`, easy to silently drop since `solve_cut_lp` has no other
/// caller-visible effect of it on a well-scaled LP — are directly testable
/// rather than only inferable from a solve's output.
fn solve_cut_lp_options(
    options: &SolverOptions,
    deadline: Option<std::time::Instant>,
    max_iters: Option<u64>,
) -> SolverOptions {
    SolverOptions {
        presolve: false,
        simplex_method: SimplexMethod::Primal,
        recover_warm_start_basis: true,
        warm_start: None,
        warm_start_lp: None,
        deadline,
        timeout_secs: None,
        max_iters,
        primal_tol: options.primal_tol,
        dual_tol: options.dual_tol,
        threads: options.threads,
        tolerance: options.tolerance,
        cancel_flag: options.cancel_flag.clone(),
        ..SolverOptions::default()
    }
}

/// See [`solve_validate`] for the `max_iters` contract shared by both cut-LP
/// solve helpers.
fn solve_cut_lp(
    lp: &LpProblem,
    options: &SolverOptions,
    deadline: Option<std::time::Instant>,
    max_iters: Option<u64>,
) -> crate::problem::SolverResult {
    let opts = solve_cut_lp_options(options, deadline, max_iters);
    crate::lp::solve_lp_with(lp, &opts)
}

/// Builds the [`SolverOptions`] for [`solve_tree_cut_warm`]: `DualAdvanced`
/// (not `Primal`, which never consults `warm_start` at all — see
/// `primal::two_phase_simplex`'s only use of it, gating the *crash basis*,
/// not warm-starting) with `disable_bounded_dispatch: true` and `warm_basis`
/// as the starting basis.
///
/// `disable_bounded_dispatch` is required, not optional: this module's
/// tableau (`generate_round`, [`extend_basis_for_new_rows`]) is built from
/// `build_standard_form`'s `sf.m`-shaped space (upper bounds as extra rows).
/// `dual_advanced`'s bounded fast path uses the smaller `build_bounded_
/// standard_form` space (`bsf.m`) for any LP with a finite upper bound —
/// true for every in-tree separation candidate in an all-boxed MILP — and
/// silently rejects an `sf.m`-shaped warm start via its own `warm.basis.len()
/// == bsf.m` guard, falling back to a cold solve whose returned basis is
/// then in the *wrong* (`bsf.m`) space for the next round's
/// `generate_round`/`extend_basis_for_new_rows` call, which return no cuts
/// rather than erroring (see `disable_bounded_dispatch`'s own doc, and the
/// `gt2` `cuts_empty` 1.2%→15.1% regression this caused before the option
/// existed).
fn tree_cut_warm_options(
    options: &SolverOptions,
    deadline: Option<std::time::Instant>,
    max_iters: Option<u64>,
    warm_basis: Vec<usize>,
) -> SolverOptions {
    SolverOptions {
        presolve: false,
        simplex_method: SimplexMethod::DualAdvanced,
        disable_bounded_dispatch: true,
        recover_warm_start_basis: true,
        warm_start: Some(WarmStartBasis {
            basis: warm_basis,
            x_b: Vec::new(),
        }),
        warm_start_lp: None,
        deadline,
        timeout_secs: None,
        max_iters,
        primal_tol: options.primal_tol,
        dual_tol: options.dual_tol,
        threads: options.threads,
        tolerance: options.tolerance,
        cancel_flag: options.cancel_flag.clone(),
        ..SolverOptions::default()
    }
}

/// Warm-started re-solve of an in-tree separation LP from `warm_basis` — a
/// basis for `lp`'s own `build_standard_form` space, either carried forward
/// unchanged (no new rows since it was last valid) or extended by
/// [`extend_basis_for_new_rows`] (this round's new cut rows appended).
///
/// See [`tree_cut_warm_options`] for why this must dispatch through
/// `DualAdvanced` with `disable_bounded_dispatch: true` rather than
/// [`solve_cut_lp`]'s cold `SimplexMethod::Primal`. Adding rows never
/// changes the objective or any existing column, so every already-optimal
/// reduced cost is untouched (dual-feasible); the new rows' own surplus
/// columns start primal-infeasible (a cut is generated because the current
/// vertex violates it) — exactly the situation dual simplex resolves in a
/// handful of pivots rather than a full re-solve.
fn solve_tree_cut_warm(
    lp: &LpProblem,
    options: &SolverOptions,
    deadline: Option<std::time::Instant>,
    max_iters: Option<u64>,
    warm_basis: Vec<usize>,
) -> crate::problem::SolverResult {
    let opts = tree_cut_warm_options(options, deadline, max_iters, warm_basis);
    crate::lp::solve_lp_with(lp, &opts)
}

/// Extends a `prev_basis` (valid for the LP `candidate` had *before* this
/// round's `k` new Ge rows were appended) into `candidate`'s larger standard
/// form, by taking each new row's own surplus column as its basic variable.
///
/// `append_ge_rows_with_integer_mask` always appends new rows after all of
/// `candidate`'s existing *real* rows, and adding rows never introduces new
/// *structural* columns (variable bounds are unchanged) — only one new
/// slack/surplus column per non-`Eq` new row (ours are always `Ge`).
///
/// `build_standard_form` additionally appends one implicit `Le` row *after
/// all real rows* for every variable with both bounds finite ("UB rows"),
/// each consuming its own slack column — for an all-boxed-integer MILP
/// (e.g. `mas76`, all-binary) this is not a rare edge case, it is most of
/// `n_total`. Inserting `k` new real rows shifts every one of those UB-row
/// slack columns later by `k`; a `prev_basis` entry referencing one (very
/// likely, since `m` basic slots are shared between structural columns and
/// slacks, so many optimal bases include at least one UB-row slack) would
/// otherwise alias a *different* column in `candidate`'s larger form,
/// producing a basis matrix that is usually singular. Column indices below
/// `boundary` (structural + old real-row slacks, whose relative row-scan
/// order is unaffected by appending rows after them) are copied as-is;
/// indices at or above it (UB-row slacks) are shifted by `k` to their new
/// position. `k == 0` is the identity map (used when re-solving `committed`
/// itself at a later round's start, with no new rows since `prev_basis`).
fn extend_basis_for_new_rows(candidate: &LpProblem, prev_basis: &[usize], k: usize) -> Vec<usize> {
    let sf = build_standard_form(candidate);
    let n_real_slack = candidate
        .constraint_types
        .iter()
        .filter(|&&ct| ct != ConstraintType::Eq)
        .count();
    let boundary = sf.n_shifted + n_real_slack - k;
    let mut extended: Vec<usize> = prev_basis
        .iter()
        .map(|&idx| if idx < boundary { idx } else { idx + k })
        .collect();
    extended.extend(boundary..boundary + k);
    extended
}

/// Warm-solves `lp` from `warm_basis` via [`solve_tree_cut_warm`], falling
/// back to a cold [`solve_cut_lp`] bootstrap when the *result* cannot be
/// trusted to seed the next round: the solve reached `Optimal` but the
/// basis it returned has a different length than `build_standard_form(lp)
/// .m`.
///
/// Pure defense, not a reachable path today: with `tree_cut_warm_options`
/// hardcoding `disable_bounded_dispatch: true`, every warm solve here runs
/// in `lp`'s own `build_standard_form` space, so this mismatch cannot fire
/// through this module's own call sites (see
/// `tree_cut_warm_options_dispatches_dual_advanced_with_disabled_bounded_path`
/// and `separate_tree_cuts_accepts_legacy_warm_start_without_singular_
/// fallback`, which cover *that* contract directly and fail if it
/// regresses). It guards instead against a future regression in
/// `dual_advanced` itself re-opening this gap: when it silently did pre-fix
/// (via the bounded fast path's smaller space), the mismatched basis was
/// not rejected loudly — `generate_round`'s own `basis.len() != sf.m` guard
/// just returned zero cuts every round after, degrading in-tree separation
/// into a silent no-op (`gt2`'s `cuts_empty` 1.2% → 15.1%). The
/// `debug_assert!` turns any recurrence into an immediate test/debug-build
/// failure; the runtime fallback keeps release builds correct — a cold
/// re-solve, not silence — at the cost of one extra solve for that round. A
/// non-`Optimal` warm status is returned as-is, with no cold retry:
/// [`separate_tree_cuts`] already ends the round on any non-`Optimal`
/// result, so retrying here would (at best) waste a solve that cannot
/// change the outcome, and (for a status like `Infeasible`, a certificate
/// rather than a resource limit) would risk quietly overriding a real
/// answer with a different one from a different starting basis — including
/// self-"healing" a possible false-`Infeasible` misdetection, whose
/// investigation is explicitly out of scope for this change (see the
/// task's verification item measuring it, not fixing it here).
fn tree_cut_resolve(
    lp: &LpProblem,
    options: &SolverOptions,
    deadline: Option<std::time::Instant>,
    max_iters: Option<u64>,
    warm_basis: Vec<usize>,
) -> crate::problem::SolverResult {
    let expected_m = build_standard_form(lp).m;
    let res = solve_tree_cut_warm(lp, options, deadline, max_iters, warm_basis);
    if res.status != SolveStatus::Optimal {
        return res;
    }
    let shape_ok = res
        .warm_start_basis
        .as_ref()
        .is_some_and(|ws| ws.basis.len() == expected_m);
    debug_assert!(
        shape_ok,
        "tree-cut warm solve returned a basis whose length does not match \
         build_standard_form(lp).m; disable_bounded_dispatch should prevent this"
    );
    if shape_ok {
        return res;
    }
    let mut cold = solve_cut_lp(lp, options, deadline, max_iters);
    cold.iterations = cold.iterations.saturating_add(res.iterations);
    cold
}

fn generate_round(
    lp: &LpProblem,
    integer_mask: &[bool],
    x_star: &[f64],
    basis: &[usize],
    kind: CutKind,
) -> Vec<CutRow> {
    assert_eq!(
        x_star.len(),
        lp.num_vars,
        "cut separation requires one LP value per variable"
    );
    let sf = build_standard_form(lp);
    // Both guards are the historical phase-2 failure mode made loud
    // (Codex review, P2-5): a shape-mismatched `basis` (e.g. from the
    // bounded fast path's smaller space, see `disable_bounded_dispatch`'s
    // doc) used to make this function silently return no cuts every round,
    // masking `gt2`'s `cuts_empty` 1.2% → 15.1% regression rather than
    // surfacing it. With `tree_cut_warm_options` hardcoding `disable_
    // bounded_dispatch: true`, every caller's `basis` should already be in
    // `lp`'s own `build_standard_form` space, so these should never fire —
    // the `debug_assert!`s turn a recurrence into an immediate test/
    // debug-build failure instead of a silent empty-cuts round.
    if basis.len() != sf.m {
        debug_assert!(
            false,
            "generate_round: basis.len()={} does not match build_standard_form(lp).m={}",
            basis.len(),
            sf.m
        );
        return Vec::new();
    }
    if basis.iter().any(|&j| j >= sf.n_total) {
        debug_assert!(
            false,
            "generate_round: basis contains an index >= sf.n_total={}",
            sf.n_total
        );
        return Vec::new();
    }

    let frac_tol = feas_rel_tol();
    let struct_cols = classify_struct_cols(&sf, integer_mask);
    let slack_kinds = classify_slack_cols(lp, &sf);
    let rows = row_lists(&lp.a, lp.num_constraints);

    let mut in_basis = vec![false; sf.n_total];
    for &j in basis {
        in_basis[j] = true;
    }

    let Ok(mut lu) = LuBasis::new_timed(&sf.a, basis, 0, None) else {
        return Vec::new();
    };
    let mut beta = sf.b.clone();
    lu.ftran_dense(&mut beta);

    let mut sources: Vec<(usize, f64)> = Vec::new();
    for (i, &col) in basis.iter().enumerate() {
        if col >= sf.n_shifted {
            continue;
        }
        let sc = struct_cols[col];
        if !sc.integral {
            continue;
        }
        let b = beta[i];
        let f0 = b - b.floor();
        if f0 <= frac_tol || f0 >= 1.0 - frac_tol {
            continue;
        }
        sources.push((i, (f0 - 0.5).abs()));
    }
    sources.sort_by(|a, b| a.1.total_cmp(&b.1));
    sources.truncate(MAX_CUTS_PER_ROUND);

    let mut cuts = Vec::new();
    for (i, _) in sources {
        if let Some(cut) = build_cut(
            &sf,
            &mut lu,
            &beta,
            i,
            &in_basis,
            &struct_cols,
            &slack_kinds,
            &rows,
            lp,
            x_star,
            frac_tol,
            kind,
        ) {
            cuts.push(cut);
        }
    }
    cuts
}

#[allow(clippy::too_many_arguments)]
fn build_cut(
    sf: &StandardForm,
    lu: &mut LuBasis,
    beta: &[f64],
    i: usize,
    in_basis: &[bool],
    struct_cols: &[StructCol],
    slack_kinds: &[Option<SlackKind>],
    rows: &[Vec<(usize, f64)>],
    lp: &LpProblem,
    x_star: &[f64],
    frac_tol: f64,
    kind: CutKind,
) -> Option<CutRow> {
    let f0 = {
        let b = beta[i];
        b - b.floor()
    };
    let one_minus_f0 = 1.0 - f0;

    let mut rho = vec![0.0; sf.m];
    rho[i] = 1.0;
    lu.btran_dense(&mut rho);

    let mut g = vec![0.0; lp.num_vars];
    let mut d = 0.0_f64;

    for j in 0..sf.n_total {
        if in_basis[j] {
            continue;
        }
        let alpha = column_dot(&sf.a, j, &rho);
        if alpha.abs() <= ZERO_TOL {
            continue;
        }
        let integral = j < sf.n_shifted && struct_cols[j].integral;
        let effective_kind = if matches!(kind, CutKind::Mir) && j >= sf.n_shifted {
            CutKind::Gmi
        } else {
            kind
        };
        let gamma = match effective_kind {
            CutKind::Gmi => gmi_coeff(alpha, f0, one_minus_f0, integral),
            CutKind::Mir => mir_coeff(alpha, f0, one_minus_f0, integral),
        };
        if gamma <= ZERO_TOL {
            continue;
        }
        if !accumulate_column(
            j,
            gamma,
            sf,
            struct_cols,
            slack_kinds,
            rows,
            lp,
            &mut g,
            &mut d,
        ) {
            return None;
        }
    }

    let rhs = 1.0 - d;
    finalize_cut(g, rhs, x_star, frac_tol)
}

fn gmi_coeff(alpha: f64, f0: f64, one_minus_f0: f64, integral: bool) -> f64 {
    if integral {
        let f = (alpha - alpha.floor()).clamp(0.0, 1.0);
        if f <= f0 {
            f / f0
        } else {
            (1.0 - f) / one_minus_f0
        }
    } else if alpha > 0.0 {
        alpha / f0
    } else {
        -alpha / one_minus_f0
    }
}

/// MIR coefficient — identical to GMI for all cases.
fn mir_coeff(alpha: f64, f0: f64, one_minus_f0: f64, integral: bool) -> f64 {
    gmi_coeff(alpha, f0, one_minus_f0, integral)
}

#[allow(clippy::too_many_arguments)]
fn accumulate_column(
    j: usize,
    gamma: f64,
    sf: &StandardForm,
    struct_cols: &[StructCol],
    slack_kinds: &[Option<SlackKind>],
    rows: &[Vec<(usize, f64)>],
    lp: &LpProblem,
    g: &mut [f64],
    d: &mut f64,
) -> bool {
    if j < sf.n_shifted {
        let sc = struct_cols[j];
        match sc.kind {
            StructKind::LbShift => {
                g[sc.var] += gamma;
                *d += gamma * (-sc.offset);
            }
            StructKind::UbOnly => {
                g[sc.var] -= gamma;
                *d += gamma * sc.offset;
            }
            StructKind::FreeSplit => return false,
        }
    } else {
        match slack_kinds[j - sf.n_shifted] {
            Some(SlackKind::ConstraintLe(r)) => {
                *d += gamma * lp.b[r];
                for &(c, v) in &rows[r] {
                    g[c] -= gamma * v;
                }
            }
            Some(SlackKind::ConstraintGe(r)) => {
                *d += gamma * (-lp.b[r]);
                for &(c, v) in &rows[r] {
                    g[c] += gamma * v;
                }
            }
            Some(SlackKind::UbRow(p)) => {
                let ub = lp.bounds[p].1;
                g[p] -= gamma;
                *d += gamma * ub;
            }
            None => return false,
        }
    }
    true
}

fn column_dot(a: &CscMatrix, j: usize, rho: &[f64]) -> f64 {
    let (rows, vals) = a.column(j);
    rows.iter()
        .zip(vals)
        .map(|(&r, &v)| v * rho[r])
        .sum::<f64>()
}

fn finalize_cut(g: Vec<f64>, rhs: f64, x_star: &[f64], frac_tol: f64) -> Option<CutRow> {
    assert_eq!(
        g.len(),
        x_star.len(),
        "cut violation evaluation requires matching cut and LP solution dimensions"
    );
    if !rhs.is_finite() || g.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut max_abs = 0.0_f64;
    let mut min_abs = f64::INFINITY;
    for &v in &g {
        let a = v.abs();
        if a > ZERO_TOL {
            max_abs = max_abs.max(a);
            min_abs = min_abs.min(a);
        }
    }
    if max_abs <= ZERO_TOL {
        return None;
    }
    if max_abs / min_abs > GMI_MAX_COEF_DYNAMISM {
        return None;
    }
    let lhs: f64 = g.iter().zip(x_star).map(|(&gi, &xi)| gi * xi).sum();
    let violation = rhs - lhs;
    if violation <= frac_tol * (1.0 + rhs.abs()) {
        return None;
    }
    Some(CutRow { coeffs: g, rhs })
}

fn classify_struct_cols(sf: &StandardForm, integer_mask: &[bool]) -> Vec<StructCol> {
    let mut cols = vec![
        StructCol {
            var: 0,
            offset: 0.0,
            kind: StructKind::LbShift,
            integral: false,
        };
        sf.n_shifted
    ];
    for (p, info) in sf.orig_var_info.iter().enumerate() {
        let is_int = integer_mask[p];
        if info.new_vars.len() == 2 {
            for &(idx, _) in &info.new_vars {
                cols[idx] = StructCol {
                    var: p,
                    offset: 0.0,
                    kind: StructKind::FreeSplit,
                    integral: false,
                };
            }
            continue;
        }
        let (idx, coeff) = info.new_vars[0];
        let kind = if coeff > 0.0 {
            StructKind::LbShift
        } else {
            StructKind::UbOnly
        };
        let shift_integral = (info.offset - info.offset.round()).abs() <= ZERO_TOL;
        cols[idx] = StructCol {
            var: p,
            offset: info.offset,
            kind,
            integral: is_int && shift_integral,
        };
    }
    cols
}

fn classify_slack_cols(lp: &LpProblem, sf: &StandardForm) -> Vec<Option<SlackKind>> {
    let n_slack = sf.n_total - sf.n_shifted;
    let mut kinds = vec![None; n_slack];

    let ub_row_vars: Vec<usize> = (0..lp.num_vars)
        .filter(|&p| {
            let (lo, hi) = lp.bounds[p];
            lo.is_finite() && hi.is_finite()
        })
        .collect();

    let mut s = 0usize;
    for (r, &ct) in lp.constraint_types.iter().enumerate() {
        match ct {
            ConstraintType::Le => {
                kinds[s] = Some(SlackKind::ConstraintLe(r));
                s += 1;
            }
            ConstraintType::Ge => {
                kinds[s] = Some(SlackKind::ConstraintGe(r));
                s += 1;
            }
            ConstraintType::Eq => {}
        }
    }
    for &p in &ub_row_vars {
        assert!(
            s < n_slack,
            "UB-row count exceeds slack column count: s={s} >= n_slack={n_slack}"
        );
        kinds[s] = Some(SlackKind::UbRow(p));
        s += 1;
    }
    debug_assert_eq!(s, n_slack, "slack count mismatch vs standard form");
    kinds
}

fn row_lists(a: &CscMatrix, num_rows: usize) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); num_rows];
    for c in 0..a.ncols() {
        let (rs, vs) = a.column(c);
        for (&r, &v) in rs.iter().zip(vs) {
            rows[r].push((c, v));
        }
    }
    rows
}

#[cfg(test)]
fn append_ge_rows(lp: &LpProblem, cuts: &[CutRow]) -> LpProblem {
    append_ge_rows_with_integer_mask(lp, cuts, &[])
}

fn append_ge_rows_with_integer_mask(
    lp: &LpProblem,
    cuts: &[CutRow],
    integer_mask: &[bool],
) -> LpProblem {
    let m_old = lp.num_constraints;
    let n = lp.num_vars;
    let m_new = m_old + cuts.len();

    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
    for c in 0..lp.a.ncols() {
        let (rs, vs) = lp.a.column(c);
        for (&r, &v) in rs.iter().zip(vs) {
            trip_rows.push(r);
            trip_cols.push(c);
            trip_vals.push(v);
        }
    }
    for (k, cut) in cuts.iter().enumerate() {
        let r = m_old + k;
        for (col, &v) in cut.coeffs.iter().enumerate() {
            if v.abs() > ZERO_TOL {
                trip_rows.push(r);
                trip_cols.push(col);
                trip_vals.push(v);
            }
        }
    }
    let a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_new, n)
        .expect("cut-augmented A is well-formed");

    let mut b = lp.b.clone();
    let mut ctypes = lp.constraint_types.clone();
    for cut in cuts {
        b.push(cut.rhs);
        ctypes.push(ConstraintType::Ge);
    }

    let bounds = normalize_near_empty_bounds(&lp.bounds, integer_mask);
    let mut out = LpProblem::new_general(lp.c.clone(), a, b, ctypes, bounds, lp.name.clone())
        .expect("cut-augmented LP is valid");
    out.obj_offset = lp.obj_offset;
    out
}

fn normalize_near_empty_bounds(bounds: &[(f64, f64)], integer_mask: &[bool]) -> Vec<(f64, f64)> {
    bounds
        .iter()
        .enumerate()
        .map(|(j, &(lb, ub))| {
            if lb <= ub {
                return (lb, ub);
            }
            if lb - ub > ZERO_TOL {
                return (lb, ub);
            }
            if integer_mask.get(j).copied().unwrap_or(false) {
                let int_lb = lb.ceil();
                let int_ub = ub.floor();
                if int_lb <= int_ub {
                    (int_lb, int_ub)
                } else if (lb - int_ub).abs() <= ZERO_TOL {
                    (int_ub, int_ub)
                } else if (int_lb - ub).abs() <= ZERO_TOL {
                    (int_lb, int_lb)
                } else {
                    (lb, ub)
                }
            } else {
                let fixed = 0.5 * (lb + ub);
                (fixed, fixed)
            }
        })
        .collect()
}

#[cfg(test)]
fn convert_cuts_to_le(lp: LpProblem, m_orig: usize) -> LpProblem {
    convert_cuts_to_le_with_integer_mask(lp, m_orig, &[])
}

fn convert_cuts_to_le_with_integer_mask(
    lp: LpProblem,
    m_orig: usize,
    integer_mask: &[bool],
) -> LpProblem {
    if lp.num_constraints == m_orig {
        return lp;
    }
    let m_total = lp.num_constraints;
    let n = lp.num_vars;

    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
    for c in 0..lp.a.ncols() {
        let (rs, vs) = lp.a.column(c);
        for (&r, &v) in rs.iter().zip(vs) {
            trip_rows.push(r);
            trip_cols.push(c);
            trip_vals.push(if r >= m_orig { -v } else { v });
        }
    }
    let a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_total, n)
        .expect("cut-Le conversion is well-formed");

    let mut b = lp.b[..m_orig].to_vec();
    let mut ctypes = lp.constraint_types[..m_orig].to_vec();
    for i in m_orig..m_total {
        b.push(-lp.b[i]);
        ctypes.push(ConstraintType::Le);
    }

    let bounds = normalize_near_empty_bounds(&lp.bounds, integer_mask);
    let mut out = LpProblem::new_general(lp.c.clone(), a, b, ctypes, bounds, lp.name.clone())
        .expect("cut-Le LP is valid");
    out.obj_offset = lp.obj_offset;
    out
}

// ── Structural cuts (cover, clique, implied bound) ──────────────────────────

/// Returns `true` when variable `j` is a binary integer (bounds within [0,1]).
fn is_binary(j: usize, integer_mask: &[bool], bounds: &[(f64, f64)]) -> bool {
    j < integer_mask.len()
        && integer_mask[j]
        && bounds[j].0 >= -ZERO_TOL
        && bounds[j].1 <= 1.0 + ZERO_TOL
}

/// Cover cuts for 0-1 knapsack Le constraints.
///
/// For each Le row whose support is entirely non-negative binary, finds a
/// minimal cover C (Σ_{j∈C} a_j > b) and emits −Σ_{j∈C} x_j ≥ −(|C|−1).
fn generate_cover_cuts(lp: &LpProblem, integer_mask: &[bool], x_star: &[f64]) -> Vec<CutRow> {
    assert_eq!(
        x_star.len(),
        lp.num_vars,
        "cover separation requires one LP value per variable"
    );
    let frac_tol = feas_rel_tol();
    let rows = row_lists(&lp.a, lp.num_constraints);
    let mut cuts = Vec::new();

    'row: for r in 0..lp.num_constraints {
        if cuts.len() >= MAX_CUTS_PER_ROUND {
            break;
        }
        if lp.constraint_types[r] != ConstraintType::Le {
            continue;
        }
        let b = lp.b[r];
        if b <= ZERO_TOL {
            continue;
        }
        let row = &rows[r];
        if row.len() < 2 {
            continue;
        }
        // Every nonzero entry must be a positive-coefficient binary variable.
        for &(j, v) in row {
            if v <= ZERO_TOL || !is_binary(j, integer_mask, &lp.bounds) {
                continue 'row;
            }
        }

        // Sort by coefficient descending; greedy cover.
        let mut sorted: Vec<(usize, f64)> = row.to_vec();
        sorted.sort_by(|a, b_| b_.1.total_cmp(&a.1));

        let mut cover: Vec<usize> = Vec::new();
        let mut sum = 0.0_f64;
        for &(j, v) in &sorted {
            cover.push(j);
            sum += v;
            if sum > b {
                break;
            }
        }
        if sum <= b {
            continue; // all variables needed; no cover exists
        }

        // Minimise: remove smallest-coefficient elements that keep sum > b.
        let mut k = cover.len();
        while k > 0 {
            k -= 1;
            let j = cover[k];
            let coeff_j = sorted
                .iter()
                .find(|&&(jj, _)| jj == j)
                .map_or(0.0, |&(_, v)| v);
            if sum - coeff_j > b {
                sum -= coeff_j;
                cover.swap_remove(k);
            }
        }

        if cover.len() < 2 {
            continue;
        }

        let mut g = vec![0.0; lp.num_vars];
        for &j in &cover {
            g[j] = -1.0;
        }
        let rhs = -((cover.len() - 1) as f64);
        if let Some(cut) = finalize_cut(g, rhs, x_star, frac_tol) {
            cuts.push(cut);
        }
    }
    cuts
}

/// Clique cuts via global pairwise conflict graph.
///
/// Two binary variables i, j "conflict" when some Le row has a_i + a_j > b_r
/// (both being 1 would violate that constraint). We build this conflict graph
/// from all rows, then for each fractional binary variable greedily extend to a
/// clique in the conflict graph. Cliques of size ≥ 3 that the LP solution
/// violates (Σ x_star[j] > 1) are emitted as −Σ_{j∈clique} x_j ≥ −1.
fn generate_clique_cuts(lp: &LpProblem, integer_mask: &[bool], x_star: &[f64]) -> Vec<CutRow> {
    assert_eq!(
        x_star.len(),
        lp.num_vars,
        "clique separation requires one LP value per variable"
    );
    let frac_tol = feas_rel_tol();
    let rows = row_lists(&lp.a, lp.num_constraints);
    let n = lp.num_vars;

    // Build conflict adjacency: conflicts[i] contains all j that conflict with i.
    let mut conflicts: Vec<Vec<usize>> = vec![Vec::new(); n];
    'row: for r in 0..lp.num_constraints {
        if lp.constraint_types[r] != ConstraintType::Le {
            continue;
        }
        let b = lp.b[r];
        if b <= ZERO_TOL {
            continue;
        }
        let row = &rows[r];
        if row.len() < 2 {
            continue;
        }
        // Pairwise conflict a_i + a_j > b is only valid when the residual
        // min-activity of all other variables is ≥ 0. Require every entry to be
        // a positive-coefficient binary variable (same guard as cover cuts).
        for &(j, v) in row {
            if v <= ZERO_TOL || !is_binary(j, integer_mask, &lp.bounds) {
                continue 'row;
            }
        }
        let bin_entries: Vec<(usize, f64)> = row.to_vec();
        for pi in 0..bin_entries.len() {
            for pj in (pi + 1)..bin_entries.len() {
                let (i, ai) = bin_entries[pi];
                let (j, aj) = bin_entries[pj];
                if ai + aj > b + ZERO_TOL {
                    if !conflicts[i].contains(&j) {
                        conflicts[i].push(j);
                    }
                    if !conflicts[j].contains(&i) {
                        conflicts[j].push(i);
                    }
                }
            }
        }
    }

    // For each fractional binary variable, greedily grow a clique in the conflict
    // graph and emit a cut if the LP solution violates Σ x_j ≤ 1.
    let mut cuts = Vec::new();
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();

    for seed in 0..n {
        if cuts.len() >= MAX_CUTS_PER_ROUND {
            break;
        }
        if !is_binary(seed, integer_mask, &lp.bounds) {
            continue;
        }
        let x_seed = x_star[seed];
        if x_seed <= frac_tol {
            continue; // seed is at zero, no incentive to include
        }
        if conflicts[seed].is_empty() {
            continue;
        }

        // Start clique with seed; extend by adding neighbours that conflict with all.
        let mut clique: Vec<usize> = vec![seed];
        // Candidate: neighbours of seed, sorted by x_star descending (greedy).
        let mut candidates: Vec<usize> = conflicts[seed].clone();
        candidates.sort_by(|&a, &b| {
            let xa = x_star[a];
            let xb = x_star[b];
            xb.total_cmp(&xa)
        });
        for cand in candidates {
            // cand must conflict with every current clique member.
            if clique.iter().all(|&m| conflicts[cand].contains(&m)) {
                clique.push(cand);
            }
        }

        if clique.len() < 3 {
            continue;
        }

        // Deduplicate.
        let mut key_vec = clique.clone();
        key_vec.sort_unstable();
        let key: u64 = key_vec.iter().take(5).fold(0u64, |acc, &j| {
            acc.wrapping_mul(1_000_003).wrapping_add(j as u64 + 1)
        });
        if !seen.insert(key) {
            continue;
        }

        let mut g = vec![0.0; lp.num_vars];
        for &j in &clique {
            g[j] = -1.0;
        }
        if let Some(cut) = finalize_cut(g, -1.0, x_star, frac_tol) {
            cuts.push(cut);
        }
    }
    cuts
}

/// Implied bound cuts derived from constraint activity bounds.
///
/// For integer variable i, the continuous implied bound is rounded to the
/// tightest integer value (floor for upper bounds, ceil for lower bounds).
/// This creates a violation gap when the LP solution is fractional between
/// the integer implied bound and the variable's original bound.
///
/// Le row, integer var i with a_i > 0:
///   implied_ub_int = floor((b − activity_min_without_i) / a_i)
///   Cut: −x_i ≥ −implied_ub_int
///
/// Ge row, integer var i with a_i > 0:
///   implied_lb_int = ceil((b − activity_max_without_i) / a_i)
///   Cut: x_i ≥ implied_lb_int
fn generate_implied_bound_cuts(
    lp: &LpProblem,
    integer_mask: &[bool],
    x_star: &[f64],
) -> Vec<CutRow> {
    assert_eq!(
        x_star.len(),
        lp.num_vars,
        "implied-bound separation requires one LP value per variable"
    );
    use crate::tolerances::INT_ROUND_TOL;
    let frac_tol = feas_rel_tol();
    let rows = row_lists(&lp.a, lp.num_constraints);
    let mut cuts = Vec::new();

    'row: for r in 0..lp.num_constraints {
        if cuts.len() >= MAX_CUTS_PER_ROUND {
            break;
        }
        let b = lp.b[r];
        let row = &rows[r];
        if row.is_empty() {
            continue;
        }

        match lp.constraint_types[r] {
            ConstraintType::Le => {
                // activity_min: a_j * lb_j (a_j > 0) + a_j * ub_j (a_j < 0)
                let mut activity_min = 0.0_f64;
                for &(j, v) in row {
                    let (lb, ub) = lp.bounds[j];
                    let bound = if v > 0.0 { lb } else { ub };
                    if !bound.is_finite() {
                        continue 'row;
                    }
                    activity_min += v * bound;
                }

                for &(i, ai) in row {
                    if cuts.len() >= MAX_CUTS_PER_ROUND {
                        break;
                    }
                    if ai <= ZERO_TOL || !integer_mask[i] {
                        continue;
                    }
                    let (lb_i, ub_i) = lp.bounds[i];
                    if !ub_i.is_finite() {
                        continue;
                    }
                    let lb_i_val = if lb_i.is_finite() { lb_i } else { continue };
                    // Remove var i's own contribution (positive coeff → used lb_i).
                    let activity_min_i = activity_min - ai * lb_i_val;
                    let raw_ub = (b - activity_min_i) / ai;
                    // Floor to integer (INT_ROUND_TOL guards floating-point drift).
                    let implied_ub = (raw_ub + INT_ROUND_TOL).floor();
                    if implied_ub >= ub_i - frac_tol {
                        continue; // not tighter than current bound
                    }
                    let mut g = vec![0.0; lp.num_vars];
                    g[i] = -1.0;
                    if let Some(cut) = finalize_cut(g, -implied_ub, x_star, frac_tol) {
                        cuts.push(cut);
                    }
                }
            }
            ConstraintType::Ge => {
                // activity_max: a_j * ub_j (a_j > 0) + a_j * lb_j (a_j < 0)
                let mut activity_max = 0.0_f64;
                for &(j, v) in row {
                    let (lb, ub) = lp.bounds[j];
                    let bound = if v > 0.0 { ub } else { lb };
                    if !bound.is_finite() {
                        continue 'row;
                    }
                    activity_max += v * bound;
                }

                for &(i, ai) in row {
                    if cuts.len() >= MAX_CUTS_PER_ROUND {
                        break;
                    }
                    if ai <= ZERO_TOL || !integer_mask[i] {
                        continue;
                    }
                    let (lb_i, ub_i) = lp.bounds[i];
                    if !lb_i.is_finite() {
                        continue;
                    }
                    let ub_i_val = if ub_i.is_finite() { ub_i } else { continue };
                    let activity_max_i = activity_max - ai * ub_i_val;
                    let raw_lb = (b - activity_max_i) / ai;
                    // Ceil to integer.
                    let implied_lb = (raw_lb - INT_ROUND_TOL).ceil();
                    if implied_lb <= lb_i + frac_tol {
                        continue;
                    }
                    let mut g = vec![0.0; lp.num_vars];
                    g[i] = 1.0;
                    if let Some(cut) = finalize_cut(g, implied_lb, x_star, frac_tol) {
                        cuts.push(cut);
                    }
                }
            }
            ConstraintType::Eq => {}
        }
    }
    cuts
}

// ── In-tree separation ──────────────────────────────────────────────────────

/// Whether in-tree separation should run at this node (depth- or count-gated).
fn tree_cut_node_selected(depth: usize, node_index: usize) -> bool {
    (depth > 0 && depth.is_multiple_of(TREE_CUT_DEPTH_INTERVAL))
        || (node_index > 0 && node_index.is_multiple_of(TREE_CUT_NODE_INTERVAL))
}

/// Re-separate GMI/MIR cuts from a B&B node's LP relaxation and return a
/// cut-tightened result when its bound improves by at least
/// [`MIN_TREE_CUT_GAIN_REL`], else `None`.
///
/// **Soundness (node-local).** Cuts derive from the node tableau and bake in
/// branching-tightened bounds, so they are valid only inside this node's
/// subtree. [`CutPool`] is created fresh per call and never stored across
/// nodes or propagated to children — B&B receives only this node's tightened
/// bound/solution, a valid lower bound for the subtree.
///
/// Mirrors root [`add_root_cuts`] but node-local: bootstrap → (generate →
/// pool-filter → append (Ge) → warm re-solve → warm re-solve)*, stopping
/// when the bound stalls. Only the very first solve (round 0's bootstrap) is
/// cold: `node_res`'s own `warm_start_basis` (from whatever dispatch solved
/// the node relaxation) is not known to be in this module's own
/// `build_standard_form` space — it may have come from `dual_advanced`'s
/// bounded fast path, a smaller basis space this module's tableau cannot
/// use (see [`tree_cut_warm_options`]'s `disable_bounded_dispatch`) — so
/// [`solve_cut_lp`] always cold-bootstraps once to obtain a basis in the
/// right space, which every later warm solve then carries forward. Every
/// round after round 0 re-solves `committed` itself (warm, from the
/// previous round's own basis — no new rows since, so this should
/// re-verify optimality in about one iteration) to derive this round's
/// cut-generation source, rather than reusing the previous round's
/// already-in-hand result object directly: the two are mathematically the
/// same LP, but deriving the source from an explicit fresh solve each round
/// (through the same shape-guarded [`tree_cut_resolve`] as the round's own
/// validate step below) means a corrupted or stale basis cannot silently
/// propagate for more than one round.
///
/// `max_iters` bounds simplex iterations across rounds (see
/// `effort::separation_iter_budget`): each individual solve within a round
/// is skipped — ending this attempt — once the remaining allowance drops
/// below [`tree_cut_min_useful_iters`]'s per-dimension minimum, rather than
/// being attempted with a too-small `SolverOptions::max_iters` (see that
/// function's doc: `TREE_CUT_MIN_SOLVE_ITER_DIM_MULT`'s markshare_4_0 note).
/// Otherwise the solve's own `max_iters` is exactly the remaining allowance,
/// bounding a single solve that would otherwise spend arbitrarily more than
/// what remains of this attempt's budget on its own (confirmed by
/// `dcmulti`'s single 27s cold cut-LP solve under Phase 1c re-bench).
/// Real simplex iterations (the first `u64`) and the [`tree_cut_construction_
/// surcharge`] fixed-cost overhead (the second `u64`, iteration-equivalent
/// units, not real simplex work) are tracked and returned separately —
/// see [`MipStats::tree_cut_overhead_iters`](super::MipStats::
/// tree_cut_overhead_iters) for why the caller must not merge them into the
/// same counter. Within this call both still count against the same
/// `max_iters` allowance: `remaining` at each round boundary is `max_iters`
/// minus the combined real-plus-overhead spend so far, so a round that would
/// exceed the *true* per-round cost (real work plus its own accounting
/// overhead) is skipped exactly as if it were all real iterations. Returns
/// the iterations actually spent plus whether this call passed the
/// node-selection interval and attempted separation (independent of the
/// iteration count, which can legitimately be 0 for a real attempt).
pub(crate) fn separate_tree_cuts(
    node_lp: &LpProblem,
    integer_mask: &[bool],
    options: &SolverOptions,
    node_res: &SolverResult,
    depth: usize,
    node_index: usize,
    max_iters: u64,
) -> (Option<SolverResult>, u64, u64, bool) {
    if !tree_cut_node_selected(depth, node_index) {
        return (None, 0, 0, false);
    }
    let base_obj = node_res.objective;
    // Fresh per-node pool: cuts are valid only in this subtree (see soundness note).
    let mut pool = CutPool::new();
    let mut committed = node_lp.clone();
    let mut accepted: Option<SolverResult> = None;
    let mut prev_obj = base_obj;
    let mut iters_spent: u64 = 0;
    let mut overhead_spent: u64 = 0;
    // `build_standard_form(committed)`-shaped basis carried warm from round
    // to round; `None` until round 0's cold bootstrap succeeds.
    let mut basis: Option<Vec<usize>> = None;

    for round_idx in 0..TREE_CUT_MAX_ROUNDS {
        // Codex review (P1) / markshare_4_0 follow-up: pass the *actual*
        // remaining allowance as this solve's own `max_iters` — a solve with
        // no other cap could otherwise spend arbitrarily more than what
        // remains of this attempt's `max_iters` budget on its own (confirmed
        // by `dcmulti`'s single 27s cold cut-LP solve under Phase 1c
        // re-bench). Skip the round entirely (rather than inflating the cap)
        // once the remaining allowance is below what a solve realistically
        // needs — see `tree_cut_min_useful_iters`'s doc.
        let remaining = max_iters.saturating_sub(iters_spent.saturating_add(overhead_spent));
        if remaining < tree_cut_min_useful_iters(&committed) {
            break;
        }
        let is_bootstrap = basis.is_none();
        let cut_res = match basis.take() {
            None => solve_cut_lp(&committed, options, options.deadline, Some(remaining)),
            Some(prev_basis) => tree_cut_resolve(
                &committed,
                options,
                options.deadline,
                Some(remaining),
                prev_basis,
            ),
        };
        iters_spent = iters_spent.saturating_add(cut_res.iterations as u64);
        // `is_bootstrap` selects the same branch `cut_res` above just took:
        // `solve_cut_lp` (no shape-check build of its own — see
        // `TREE_CUT_BUILDS_ROUND_START_COLD`) on round 0, `tree_cut_resolve`
        // (with its own shape-check build — `TREE_CUT_BUILDS_ROUND_START_
        // WARM`) every round after.
        let round_start_builds = if is_bootstrap {
            TREE_CUT_BUILDS_ROUND_START_COLD
        } else {
            TREE_CUT_BUILDS_ROUND_START_WARM
        };
        overhead_spent = overhead_spent.saturating_add(tree_cut_construction_surcharge(
            &committed,
            round_start_builds,
        ));
        if cut_res.status != SolveStatus::Optimal {
            break;
        }
        let Some(ws) = cut_res.warm_start_basis.as_ref() else {
            break;
        };
        if cut_res.solution.is_empty() {
            break;
        }
        let x_star = cut_res.solution.clone();
        let round_basis = ws.basis.clone();

        let kind = if round_idx % 2 == 0 {
            CutKind::Gmi
        } else {
            CutKind::Mir
        };
        let cuts = generate_round(&committed, integer_mask, &x_star, &round_basis, kind);
        // Charged unconditionally, before checking `cuts.is_empty()` below:
        // `generate_round` always builds `committed`'s standard form first,
        // even on a round that ends up finding nothing worth cutting.
        overhead_spent = overhead_spent.saturating_add(tree_cut_construction_surcharge(
            &committed,
            TREE_CUT_BUILDS_GENERATE_ROUND,
        ));
        if cuts.is_empty() {
            break;
        }
        let pool_candidates: Vec<Cut> = cuts
            .into_iter()
            .map(|c| Cut {
                coeffs: c.coeffs,
                rhs: c.rhs,
                sense: ConstraintType::Ge,
            })
            .collect();
        let selected = pool.separate_round(pool_candidates, &x_star);
        if selected.is_empty() {
            break;
        }

        let rows: Vec<CutRow> = selected
            .into_iter()
            .map(|c| CutRow {
                coeffs: c.coeffs,
                rhs: c.rhs,
            })
            .collect();
        let k = rows.len();
        let candidate = append_ge_rows_with_integer_mask(&committed, &rows, integer_mask);
        // Re-check the remaining allowance between the two solves of this
        // round: the cut-generation solve above may have already spent some
        // (or all) of it, and this validate solve is itself a second solve
        // subject to the same skip threshold. If too little remains to
        // realistically finish it, abandon this round rather than run a
        // truncated validate solve — the round's cuts are discarded (not
        // committed), matching the "skip, don't truncate" pattern.
        let remaining = max_iters.saturating_sub(iters_spent.saturating_add(overhead_spent));
        if remaining < tree_cut_min_useful_iters(&candidate) {
            break;
        }
        let warm_basis = extend_basis_for_new_rows(&candidate, &round_basis, k);
        let check = tree_cut_resolve(
            &candidate,
            options,
            options.deadline,
            Some(remaining),
            warm_basis,
        );
        iters_spent = iters_spent.saturating_add(check.iterations as u64);
        // Charged against `candidate` (not `committed`): `extend_basis_for_
        // new_rows` and `tree_cut_resolve` above both build `candidate`'s
        // own (larger, +k rows) standard form, not `committed`'s.
        overhead_spent = overhead_spent.saturating_add(tree_cut_construction_surcharge(
            &candidate,
            TREE_CUT_BUILDS_ROUND_END,
        ));
        if check.status != SolveStatus::Optimal || check.solution.is_empty() {
            break;
        }
        let Some(check_ws) = check.warm_start_basis.clone() else {
            break;
        };
        committed = candidate;
        basis = Some(check_ws.basis);
        let obj = check.objective;
        accepted = Some(check);

        // Stop once the bound stops improving meaningfully.
        let scale = 1.0_f64.max(prev_obj.abs());
        if obj <= prev_obj + MIN_TREE_CUT_GAIN_REL * scale {
            break;
        }
        prev_obj = obj;
    }

    // Accept only a meaningful tightening over the node's existing bound. The
    // returned result carries no warm-start basis (its augmented layout would not
    // match child node solves, which use the original constraint structure).
    let Some(mut res) = accepted else {
        return (None, iters_spent, overhead_spent, true);
    };
    let scale = 1.0_f64.max(base_obj.abs());
    if res.objective <= base_obj + MIN_TREE_CUT_GAIN_REL * scale {
        return (None, iters_spent, overhead_spent, true);
    }
    res.warm_start_basis = None;
    (Some(res), iters_spent, overhead_spent, true)
}

#[cfg(test)]
mod tests;
