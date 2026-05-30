//! Bounded dual simplex core (BFRT-aware) consuming `BoundedStandardForm`.
//!
//! Companion to `core.rs`. Unlike the legacy core (which sees variable upper
//! bounds as auxiliary `x_j + s = u_j` rows), this loop keeps the upper bound
//! as a non-basic state flag (`at_upper[j]`) and routes entering-variable
//! selection through `bfrt_select_entering` so non-basic columns whose bound
//! switch absorbs less infeasibility than a normal pivot can flip lb↔ub
//! mid-iter instead of forcing a small dual step.
//!
//! BFRT is unconditionally applied; there is no user-facing disable flag.
//!
//! Maros (2003) §7.6 reference algorithm:
//! - leaving pricing detects rows where `x_B[r]` violates a bound;
//! - BFRT returns `(entering, theta, flips)` — `flips` ⊂ non-basic columns
//!   whose bound switch is consumed by the dual step;
//! - flip apply: `x_B -= u_k · α_k` (lb→ub) or `+= u_k · α_k` (ub→lb);
//! - pivot equation gains a `+u_q` correction at the leaving row when the
//!   entering column is currently at its upper bound (the "符号反転 pivot"):
//!   `x_B[r]_new = step + (u_q if at_upper[q] else 0)`
//!   derived from the column-swap update with q's non-basic value u_q being
//!   removed from the effective RHS as q enters the basis.
//!
//! Scope: dual-phase iteration only. The driver (`solve_bounded_dual`) assumes
//! the LP enters with a primal-feasible RHS after cost perturbation (Le-only,
//! `num_artificial == 0`) and is exercised in tests via warm-start states with
//! synthetic primal infeasibility. Phase 2 primal + solution / dual recovery
//! + cold/warm production wiring live in follow-up tasks.
//!
//! ## Outcome reporting (`BoundedOutcome`)
//!
//! The loop returns its own `BoundedOutcome` instead of the shared
//! `SimplexOutcome`. The reason is the third terminal state — ub-violation —
//! that is genuinely "fall back to legacy" rather than "ran out of time".
//! Multiplexing both onto `Timeout` made wiring-layer dispatch ambiguous and
//! made sentinel assertions on Timeout meaningless. `UbViolationOutOfScope`
//! carries the offending row/objective so callers can either retry on the
//! legacy core or report a definite status.
//!
//! ## Bound-violation handling
//!
//! The `bound_flip` BFRT primitive is documented for the lb-violation leaving
//! direction (`x_B[r] < 0`). When `x_B[r] > u_{basis[r]}` (ub overshoot) the
//! symmetric BFRT requires mirroring `trow` and a parallel pivot adjustment
//! that is outside this phase's scope — the loop detects ub-violation and
//! returns `BoundedOutcome::UbViolationOutOfScope` so the wiring layer can
//! fall back to legacy `core.rs`. The cold-start path used by production
//! wiring enters with `x_B = b ≥ 0` so this branch is never triggered there.

use crate::basis::{BasisManager, LuBasis};
use crate::options::SolverOptions;
use crate::problem::LpProblem;
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::PIVOT_TOL;
use std::sync::atomic::Ordering;

use super::super::dual_common::{basic_obj, compute_dual_vars_into, compute_reduced_costs_into, recompute_gamma_truth};
use super::super::pricing::DualLeavingStrategy;
use super::super::standard_form::{BoundedStandardForm, SimplexOutcome};
use super::bound_flip::{bfrt_select_entering, bump_bfrt_flip_invocations, ColBound};

/// Terminal status of the bounded dual simplex iteration.
///
/// Distinct from the shared `SimplexOutcome`: the `UbViolationOutOfScope`
/// variant lets the wiring layer route to the legacy core deterministically
/// without confusing genuine deadlines with "this loop doesn't handle that
/// state". `Timeout`/`SingularBasis` retain their usual meaning.
#[derive(Debug)]
pub(crate) enum BoundedOutcome {
    /// Phase 1 dual optimal (perturbed cost). Fields carry the perturbed
    /// objective and dual variables; production code discards them (Phase 2
    /// primal re-optimises from scratch). Tests use them for dual-recovery
    /// smoke tests.
    #[allow(dead_code)]
    Optimal(f64, Vec<f64>),
    Unbounded,
    /// Deadline or hard iteration cap. Carries the latest objective.
    Timeout(f64),
    SingularBasis,
    /// `x_B[r] > u_{basis[r]}` reached without an lb-violation candidate.
    /// `row` is the offending basis row. Callers fall back to legacy `core.rs`.
    /// Field is read by tests; production callers use `{ .. }` to fall through.
    #[allow(dead_code)]
    UbViolationOutOfScope { row: usize },
}

#[cfg(test)]
thread_local! {
    /// Test-only hook: when `true`, the production `iterate` skips the
    /// `x_B -= alpha_flip · weight` update inside the flip loop while still
    /// toggling `at_upper[k]`. Sentinels flip this on/off to prove the flip
    /// apply is load-bearing without maintaining a parallel test copy of the
    /// iteration loop (which masked the production logic from no-op probes).
    ///
    /// Lives behind `#[cfg(test)]` so release binaries carry neither the
    /// thread-local slot nor the branch (verified via `nm` on the release
    /// `rlib` — symbol must be absent).
    static FLIP_APPLY_DISABLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn set_flip_apply_disabled(v: bool) {
    FLIP_APPLY_DISABLE.with(|c| c.set(v));
}

#[cfg(test)]
fn flip_apply_disabled() -> bool {
    FLIP_APPLY_DISABLE.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn flip_apply_disabled() -> bool {
    false
}

/// Hard iteration cap shared by both bounded dual and bounded primal Phase 2 loops.
/// Guarantees termination even when pricing degenerates or deadline is None (unit tests).
const SIMPLEX_ITER_HARD_CAP: usize = 1_000_000;

/// Hard cap for the bounded dual simplex loop (alias of [`SIMPLEX_ITER_HARD_CAP`]).
/// Guarantees termination when deadline is absent (unit tests) or pricing degenerates.
const BOUNDED_DUAL_ITER_HARD_CAP: usize = SIMPLEX_ITER_HARD_CAP;

/// Internal state of the bounded dual simplex iteration. Built from
/// `BoundedStandardForm` (cold) or hand-populated by tests / warm-start
/// callers, and consumed by `iterate`.
///
/// Field invariants:
/// - `basis.len() == m`, `x_b.len() == m`.
/// - `at_upper.len() == is_basic.len() == reduced_costs.len() == n_total`.
/// - For each basic column `j ∈ basis`, `is_basic[j] == true`,
///   `at_upper[j] == false` (basic vars have no bound state).
/// - Non-basic vars: `at_upper[j]` indicates current bound; the non-basic
///   value is `0` (lb) or `upper_bounds[j]` (ub).
/// - `x_b[i] = (B^{-1} (b − Σ_{k at_upper non-basic} u_k · a_k))[i]`,
///   reflecting the flip-applied effective RHS.
pub(crate) struct BoundedDualState {
    pub(crate) basis: Vec<usize>,
    pub(crate) at_upper: Vec<bool>,
    pub(crate) x_b: Vec<f64>,
    pub(crate) reduced_costs: Vec<f64>,
    pub(crate) is_basic: Vec<bool>,
    pub(crate) iterations: usize,
}

impl BoundedDualState {
    /// Cold-start state from a `BoundedStandardForm`: slacks in the basis,
    /// every non-basic variable at its lower bound, `x_B = b`, dual feasibility
    /// achieved by cost perturbation `c̃_j = max(c_j, 0)` evaluated at `y = 0`.
    ///
    /// Caller must supply Ruiz-scaled `(a, b)` — the LU factorization happens
    /// in `iterate` so the state alone is decoupled from `BasisManager`.
    pub(crate) fn cold(bsf: &BoundedStandardForm, b_scaled: &[f64]) -> Self {
        let m = bsf.m;
        let n_total = bsf.n_total;
        assert_eq!(bsf.initial_basis.len(), m);
        let mut is_basic = vec![false; n_total];
        for &j in bsf.initial_basis.iter() {
            if j < n_total {
                is_basic[j] = true;
            }
        }
        Self {
            basis: bsf.initial_basis.clone(),
            at_upper: vec![false; n_total],
            x_b: b_scaled.to_vec(),
            reduced_costs: vec![0.0; n_total],
            is_basic,
            iterations: 0,
        }
    }
}

/// Per-iteration scratch buffers. Allocated once and reused across iters.
struct IterBuffers {
    rho: Vec<f64>,
    trow: Vec<f64>,
    alpha: Vec<f64>,
    alpha_flip: Vec<f64>,
    sigma: Vec<f64>,
    col_bounds: Vec<ColBound>,
    y: Vec<f64>,
}

impl IterBuffers {
    fn new(m: usize, n_total: usize, upper_bounds: &[f64]) -> Self {
        let col_bounds = (0..n_total)
            .map(|j| ColBound {
                upper: upper_bounds[j],
                at_upper: false,
            })
            .collect();
        Self {
            rho: vec![0.0; m],
            trow: vec![0.0; n_total],
            alpha: vec![0.0; m],
            alpha_flip: vec![0.0; m],
            sigma: vec![0.0; m],
            col_bounds,
            y: vec![0.0; m],
        }
    }
}

/// Entry point: drives a cold-start bounded dual simplex on a Le-only
/// `BoundedStandardForm`. Caller supplies the Ruiz-scaled `(a, b, c)` so the
/// scaling lives at the same layer as `cold_start_advanced` does today.
///
/// Returns `(outcome, state)` so warm-start sequels can inspect the final
/// basis / `at_upper`. The reported objective in `Optimal(obj, y)` uses the
/// perturbed cost; the wiring layer is responsible for re-evaluating with the
/// original `c` once Phase 2 primal completes.
pub(crate) fn solve_bounded_dual(
    bsf: &BoundedStandardForm,
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    options: &SolverOptions,
    ubs: &[f64],
    leaving: &mut dyn DualLeavingStrategy,
) -> (BoundedOutcome, BoundedDualState) {
    let state = BoundedDualState::cold(bsf, b);
    iterate(state, bsf, a, c, options, ubs, leaving)
}

/// Inner iteration loop. Accepts a pre-populated state — tests use this to
/// inject synthetic primal infeasibilities; production cold/warm-start callers
/// supply the matching basis. Cost perturbation is applied here so callers
/// don't have to pre-perturb `c`.
///
/// `ubs` is the effective per-column upper bound slice used for bound-violation
/// checks and flip weights. Pass `&bsf.upper_bounds` when the matrices are
/// unscaled; pass Ruiz-scaled bounds (`u_j / col_scale[j]`) when scaled.
pub(crate) fn iterate(
    mut state: BoundedDualState,
    bsf: &BoundedStandardForm,
    a: &CscMatrix,
    c: &[f64],
    options: &SolverOptions,
    ubs: &[f64],
    leaving: &mut dyn DualLeavingStrategy,
) -> (BoundedOutcome, BoundedDualState) {
    let m = bsf.m;
    let n_total = bsf.n_total;
    debug_assert_eq!(state.basis.len(), m);
    debug_assert_eq!(state.x_b.len(), m);
    debug_assert_eq!(state.at_upper.len(), n_total);
    debug_assert_eq!(state.is_basic.len(), n_total);

    let mut basis_mgr = match LuBasis::new(a, &state.basis, options.max_etas) {
        Ok(bm) => bm,
        Err(crate::error::SolverError::SingularBasis { .. }) => {
            return (BoundedOutcome::SingularBasis, state);
        }
        Err(_) => {
            let obj = bounded_obj(c, &state.basis, &state.x_b, &state.at_upper, &state.is_basic, ubs);
            return (BoundedOutcome::Timeout(obj), state);
        }
    };

    // Early-exit before O(m²) γ init; prevents budget overrun on large warm-start solves.
    if options.deadline.is_some_and(|d| std::time::Instant::now() >= d)
        || options.cancel_flag.as_ref().is_some_and(|f| f.load(Ordering::Relaxed))
    {
        let obj = bounded_obj(c, &state.basis, &state.x_b, &state.at_upper, &state.is_basic, ubs);
        return (BoundedOutcome::Timeout(obj), state);
    }

    let needs_sigma = leaving.needs_sigma();
    if needs_sigma {
        match recompute_gamma_truth(
            &mut basis_mgr, m, options.deadline,
            options.cancel_flag.as_deref(),
        ) {
            None => {
                let obj = bounded_obj(c, &state.basis, &state.x_b, &state.at_upper, &state.is_basic, ubs);
                return (BoundedOutcome::Timeout(obj), state);
            }
            Some(gamma_truth) => leaving.set_initial_gamma(&gamma_truth),
        }
    }

    // Cost perturbation: c̃_j = max(c_j, 0). With slack initial basis (y = 0)
    // every reduced cost is ≥ 0 ⇒ dual feasible. The perturbation is local
    // to this loop; the caller restores the original cost in Phase 2 primal.
    let c_perturbed: Vec<f64> = c.iter().map(|&v| v.max(0.0)).collect();

    let mut buf = IterBuffers::new(m, n_total, ubs);

    // Initial reduced costs (r_j = c̃_j − y^T a_j with y = B^{-T} c̃_B).
    compute_reduced_costs_into(
        a,
        &c_perturbed,
        &mut basis_mgr,
        &state.is_basic,
        n_total,
        &state.basis,
        &mut buf.y,
        &mut state.reduced_costs,
    );

    loop {
        state.iterations = state.iterations.saturating_add(1);
        if state.iterations > BOUNDED_DUAL_ITER_HARD_CAP {
            let obj = bounded_obj(c, &state.basis, &state.x_b, &state.at_upper, &state.is_basic, ubs);
            return (BoundedOutcome::Timeout(obj), state);
        }
        let timed_out = options.deadline.is_some_and(|d| std::time::Instant::now() >= d);
        let cancelled = options
            .cancel_flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed));
        if timed_out || cancelled {
            let obj = bounded_obj(c, &state.basis, &state.x_b, &state.at_upper, &state.is_basic, ubs);
            return (BoundedOutcome::Timeout(obj), state);
        }

        // ub-violation scan (separate from lb-violation leaving selection).
        let mut ub_violation_row: Option<usize> = None;
        for i in 0..m {
            let xi = state.x_b[i];
            let ub_i = ubs[state.basis[i]];
            if ub_i.is_finite() && xi > ub_i + options.primal_tol {
                ub_violation_row.get_or_insert(i);
            }
        }
        let leaving_row = leaving.select_leaving(&state.x_b, options.primal_tol, &state.basis);
        if leaving_row.is_none() {
            if let Some(row) = ub_violation_row {
                // Out-of-scope for this loop: distinct from Timeout so the
                // wiring layer can route to the legacy core deterministically.
                return (BoundedOutcome::UbViolationOutOfScope { row }, state);
            }
            let obj = basic_obj(c, &state.basis, &state.x_b);
            let mut y = vec![0.0; m];
            compute_dual_vars_into(&c_perturbed, &mut basis_mgr, &state.basis, &mut y);
            return (BoundedOutcome::Optimal(obj, y), state);
        }
        let r = leaving_row.unwrap();

        // BTRAN ρ = B^{-T} e_r.
        for slot in buf.rho.iter_mut() {
            *slot = 0.0;
        }
        buf.rho[r] = 1.0;
        let mut rho_sv = SparseVec::from_dense(&buf.rho);
        basis_mgr.btran(&mut rho_sv);
        rho_sv.to_dense_into(&mut buf.rho);

        // σ = B^{-1} ρ (needed by DSE after_pivot weight update).
        if needs_sigma {
            let mut sigma_sv = SparseVec::from_dense(&buf.rho);
            basis_mgr.ftran(&mut sigma_sv);
            sigma_sv.to_dense_into(&mut buf.sigma);
        }

        // PRICE trow[j] = ρ^T a_j on non-basic columns.
        for j in 0..n_total {
            if state.is_basic[j] {
                buf.trow[j] = 0.0;
                continue;
            }
            let (rows, vals) = a.get_column(j).unwrap();
            let mut dot = 0.0;
            for (k, &row) in rows.iter().enumerate() {
                dot += buf.rho[row] * vals[k];
            }
            buf.trow[j] = dot;
        }

        // Refresh `col_bounds.at_upper` (uppers themselves never change).
        for j in 0..n_total {
            buf.col_bounds[j].at_upper = state.at_upper[j];
        }

        let leaving_residual = state.x_b[r]; // negative; BFRT uses |·|
        let bfrt = match bfrt_select_entering(
            &buf.trow,
            &state.reduced_costs,
            &state.is_basic,
            &buf.col_bounds,
            n_total,
            PIVOT_TOL,
            leaving_residual,
        ) {
            None => {
                // No compatible non-basic column ⇒ dual unbounded ⇒ primal
                // infeasible (matches `core.rs` convention).
                return (BoundedOutcome::Unbounded, state);
            }
            Some(res) => res,
        };

        // Apply flips: each non-entering bypassed breakpoint switches its
        // bound. x_B picks up Δx_N[k] · α_k per flip — flip from lb (0) to ub
        // (u_k) adds +u_k to x_N[k]; ub→lb adds −u_k. x_B := x_B − α_k · Δ.
        // The `flip_apply_disabled` gate exists for test no-op proofs only and
        // is a const `false` in release builds (eliminated by the optimizer).
        let apply_flip = !flip_apply_disabled();
        for &k in &bfrt.flips {
            let u_k = ubs[k];
            debug_assert!(
                u_k.is_finite(),
                "BFRT must not return infinite-upper flips"
            );
            if apply_flip {
                ftran_column(a, &mut basis_mgr, k, m, &mut buf.alpha_flip);
                let direction = if state.at_upper[k] { -1.0 } else { 1.0 };
                let weight = direction * u_k;
                for i in 0..m {
                    state.x_b[i] -= buf.alpha_flip[i] * weight;
                }
            }
            state.at_upper[k] = !state.at_upper[k];
        }

        let entering_col = bfrt.entering_col;
        let theta = bfrt.theta;
        let entering_at_upper = state.at_upper[entering_col];

        // FTRAN α_q = B^{-1} a_q.
        ftran_column(a, &mut basis_mgr, entering_col, m, &mut buf.alpha);
        let pivot_element = buf.alpha[r];
        if pivot_element.abs() < PIVOT_TOL {
            // Numerically unstable pivot — refactor and recompute reduced
            // costs. Matches the legacy core's recovery path.
            basis_mgr.refactor_if_needed_timed(a, &state.basis, options.deadline);
            if basis_mgr.refactor_failed {
                if basis_mgr.singular_basis {
                    return (BoundedOutcome::SingularBasis, state);
                }
                let obj = bounded_obj(c, &state.basis, &state.x_b, &state.at_upper, &state.is_basic, ubs);
                return (BoundedOutcome::Timeout(obj), state);
            }
            leaving.after_refactor(m);
            if needs_sigma {
                match recompute_gamma_truth(
                    &mut basis_mgr, m, options.deadline,
                    options.cancel_flag.as_deref(),
                ) {
                    None => {
                        let obj = bounded_obj(c, &state.basis, &state.x_b, &state.at_upper, &state.is_basic, ubs);
                        return (BoundedOutcome::Timeout(obj), state);
                    }
                    Some(gamma_truth) => leaving.set_initial_gamma(&gamma_truth),
                }
            }
            compute_reduced_costs_into(
                a,
                &c_perturbed,
                &mut basis_mgr,
                &state.is_basic,
                n_total,
                &state.basis,
                &mut buf.y,
                &mut state.reduced_costs,
            );
            continue;
        }

        // Standard column-swap pivot update of x_B. The at-upper entering
        // correction (`+u_q`) accounts for q's non-basic value being subtracted
        // from the effective RHS as q transitions into the basis (derivation
        // in module doc).
        let step = state.x_b[r] / pivot_element;
        for i in 0..m {
            state.x_b[i] -= buf.alpha[i] * step;
        }
        state.x_b[r] = step;
        if entering_at_upper {
            let u_q = ubs[entering_col];
            debug_assert!(u_q.is_finite(), "at_upper entering must be finite");
            state.x_b[r] += u_q;
        }
        // Defensive clamp; matches `core.rs`.
        for val in state.x_b.iter_mut() {
            if val.abs() < options.clamp_tol {
                *val = 0.0;
            }
        }

        // Reduced-cost increment: r_j_new = r_j − θ trow[j] for non-basic j.
        // The leaving column σ becomes non-basic with r_σ = −θ.
        let leaving_col = state.basis[r];
        for j in 0..n_total {
            if !state.is_basic[j] {
                state.reduced_costs[j] -= theta * buf.trow[j];
            }
        }
        if leaving_col < n_total {
            state.reduced_costs[leaving_col] = -theta;
        }

        // Basis bookkeeping: q enters as basic (its previous at_upper flag
        // is cleared — basic vars carry no bound state). σ leaves to its lb
        // (lb-violation leaving direction ⇒ σ → 0).
        state.is_basic[entering_col] = true;
        state.at_upper[entering_col] = false;
        if leaving_col < n_total {
            state.is_basic[leaving_col] = false;
            state.at_upper[leaving_col] = false;
        }

        // Push the column swap through the LU.
        let (col_rows, col_vals) = a.get_column(entering_col).unwrap();
        let alpha_sv = SparseVec {
            indices: col_rows.to_vec(),
            values: col_vals.to_vec(),
            len: m,
        };
        let mut alpha_sv_for_update = alpha_sv;
        basis_mgr.ftran(&mut alpha_sv_for_update);
        basis_mgr.update(entering_col, r, &alpha_sv_for_update);
        state.basis[r] = entering_col;

        leaving.after_pivot(r, &buf.alpha, &buf.sigma, pivot_element);

        // Refactor + reduced-cost refresh on the LU's request (eta cap).
        if basis_mgr.needs_refactor() {
            basis_mgr.refactor_if_needed_timed(a, &state.basis, options.deadline);
            if basis_mgr.refactor_failed {
                if basis_mgr.singular_basis {
                    return (BoundedOutcome::SingularBasis, state);
                }
                let obj = bounded_obj(c, &state.basis, &state.x_b, &state.at_upper, &state.is_basic, ubs);
                return (BoundedOutcome::Timeout(obj), state);
            }
            leaving.after_refactor(m);
            if needs_sigma {
                match recompute_gamma_truth(
                    &mut basis_mgr, m, options.deadline,
                    options.cancel_flag.as_deref(),
                ) {
                    None => {
                        let obj = bounded_obj(c, &state.basis, &state.x_b, &state.at_upper, &state.is_basic, ubs);
                        return (BoundedOutcome::Timeout(obj), state);
                    }
                    Some(gamma_truth) => leaving.set_initial_gamma(&gamma_truth),
                }
            }
            compute_reduced_costs_into(
                a,
                &c_perturbed,
                &mut basis_mgr,
                &state.is_basic,
                n_total,
                &state.basis,
                &mut buf.y,
                &mut state.reduced_costs,
            );
        }
    }
}

/// FTRAN a column of `a` and dump into `out` (length `m`). Wraps the
/// `SparseVec` boilerplate that every FTRAN site repeats.
fn ftran_column(
    a: &CscMatrix,
    basis_mgr: &mut LuBasis,
    col: usize,
    m: usize,
    out: &mut [f64],
) {
    let (rows, vals) = a.get_column(col).unwrap();
    let mut sv = SparseVec {
        indices: rows.to_vec(),
        values: vals.to_vec(),
        len: m,
    };
    basis_mgr.ftran(&mut sv);
    sv.to_dense_into(out);
}

// ── test-only hook: disable the at_upper correction in extract_solution_bounded ──

#[cfg(test)]
thread_local! {
    /// When true, `extract_solution_bounded` skips the non-basic-at-upper
    /// contribution. Used by the no-op proof sentinel to confirm the correction
    /// is load-bearing. Release builds see a const-false inlined by the optimizer.
    static AT_UPPER_APPLY_DISABLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn set_at_upper_apply_disabled(v: bool) {
    AT_UPPER_APPLY_DISABLE.with(|c| c.set(v));
}

#[cfg(test)]
fn at_upper_apply_disabled() -> bool {
    AT_UPPER_APPLY_DISABLE.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn at_upper_apply_disabled() -> bool {
    false
}

// ── bounded solution / dual recovery ──────────────────────────────────────────

/// Recover the full primal vector from a bounded dual terminal state.
///
/// Unlike `extract_solution` (which sets every non-basic value to 0), this
/// function accounts for variables that are non-basic **at their upper bound**:
/// - basic j = basis[i]: `x_new[j] = x_b[i] * col_scale[j]`
/// - non-basic at lb (`at_upper[j] = false`): `x_new[j] = 0`
/// - non-basic at ub (`at_upper[j] = true`): `x_new[j] = upper_bounds[j]`
///   (no col_scale needed: `upper_bounds` lives in the pre-Ruiz-scale space,
///   so the scale factors cancel)
///
/// The result is mapped back to original variables via `orig_var_info`
/// with double-double arithmetic for free-variable split cancellation.
pub(crate) fn extract_solution_bounded(
    bsf: &BoundedStandardForm,
    state: &BoundedDualState,
    col_scale: &[f64],
) -> Vec<f64> {
    use twofloat::TwoFloat;
    let mut x_new = vec![0.0f64; bsf.n_shifted];

    for i in 0..bsf.m {
        let j = state.basis[i];
        if j < bsf.n_shifted {
            let scale = col_scale.get(j).copied().unwrap_or(1.0);
            x_new[j] = state.x_b[i] * scale;
        }
    }

    if !at_upper_apply_disabled() {
        for j in 0..bsf.n_shifted {
            if !state.is_basic[j] && state.at_upper[j] {
                // upper_bounds[j] is in original (pre-scale) space; col_scale cancels.
                x_new[j] = bsf.upper_bounds[j];
            }
        }
    }

    let mut solution = vec![0.0f64; bsf.n_orig];
    for (orig_j, sol_j) in solution.iter_mut().enumerate() {
        let info = &bsf.orig_var_info[orig_j];
        let mut value = TwoFloat::from(info.offset);
        for &(new_idx, coeff) in &info.new_vars {
            value += TwoFloat::new_mul(coeff, x_new[new_idx]);
        }
        *sol_j = f64::from(value);
    }
    solution
}

/// Recover original-problem duals, reduced costs, and slack from a bounded
/// dual terminal state. Mirrors `extract_dual_info` but operates on
/// `BoundedStandardForm` (no UB rows ⇒ `bsf.m == m_orig`).
pub(crate) fn extract_dual_info_bounded(
    bsf: &BoundedStandardForm,
    problem: &LpProblem,
    y_std: &[f64],
    solution: &[f64],
    row_scale: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m_orig = bsf.m;
    let n_orig = bsf.n_orig;

    let mut dual_solution = vec![0.0f64; m_orig];
    for i in 0..m_orig {
        let sign = if bsf.row_negated[i] { -1.0 } else { 1.0 };
        let rs = row_scale.get(i).copied().unwrap_or(1.0);
        dual_solution[i] = sign * rs * y_std[i];
    }

    let mut slack = problem.b.clone();
    for (j, &sol_j) in solution.iter().enumerate().take(n_orig) {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                slack[row] -= vals[k] * sol_j;
            }
        }
    }

    let mut reduced_costs = problem.c.clone();
    for (j, rc_j) in reduced_costs.iter_mut().enumerate().take(n_orig) {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                *rc_j -= dual_solution[row] * vals[k];
            }
        }
    }

    (dual_solution, reduced_costs, slack)
}

// ── bounded primal Phase 2 ─────────────────────────────────────────────────

const PHASE2_PRIMAL_ITER_CAP: usize = SIMPLEX_ITER_HARD_CAP;

/// Objective including non-basic at-upper-bound contributions.
///
/// `basic_obj` only sums over basic variables. For the bounded simplex, the
/// full objective is c_B^T x_B + Σ_{j non-basic at ub} c_j · u_j. The basic
/// filter is defensive: the invariant `at_upper[j] ⇒ !is_basic[j]` is
/// maintained by `iterate` / `phase2_primal_bounded`, but skipping basic vars
/// keeps `bounded_obj` correct even if a future caller violates that.
fn bounded_obj(
    c: &[f64],
    basis: &[usize],
    x_b: &[f64],
    at_upper: &[bool],
    is_basic: &[bool],
    ubs: &[f64],
) -> f64 {
    let basic: f64 = basis.iter().zip(x_b.iter()).map(|(&j, &v)| c[j] * v).sum();
    let at_ub: f64 = at_upper
        .iter()
        .enumerate()
        .filter(|&(j, &flag)| flag && !is_basic.get(j).copied().unwrap_or(false))
        .map(|(j, _)| c.get(j).copied().unwrap_or(0.0) * ubs.get(j).copied().unwrap_or(0.0))
        .sum();
    basic + at_ub
}

/// Drive primal Phase 2 from a primal-feasible `BoundedDualState`.
///
/// Caller supplies the state produced by `solve_bounded_dual` (perturbed-cost
/// dual phase) and the **original** cost vector `c`. The function minimizes
/// the original objective while maintaining primal feasibility, handling
/// variables at their upper bound via bounded-primal ratio test.
///
/// Pricing: non-basic at lb enters if `rc < 0`; non-basic at ub enters if
/// `rc > 0` (reversed, because decreasing from ub reduces the objective).
/// Ratio test: leaving variable hits either its lb or ub; entering variable
/// may flip to its opposite bound without a basis change (step = `u_q`).
///
/// Returns `(SimplexOutcome, BoundedDualState)` so the caller can extract the
/// solution and dual variables from the terminal state.
/// `ubs` must match the Ruiz-scaling space of `a` and `c` (pass
/// `&bsf.upper_bounds` for unscaled, or scaled bounds from `scale_upper_bounds`).
pub(crate) fn phase2_primal_bounded(
    bsf: &BoundedStandardForm,
    mut state: BoundedDualState,
    a: &CscMatrix,
    c: &[f64],
    options: &SolverOptions,
    iters: &mut usize,
    ubs: &[f64],
) -> (SimplexOutcome, BoundedDualState) {
    let m = bsf.m;
    let n_total = bsf.n_total;

    let mut basis_mgr = match LuBasis::new(a, &state.basis, options.max_etas) {
        Ok(bm) => bm,
        Err(_) => return (SimplexOutcome::SingularBasis, state),
    };

    let mut y = vec![0.0f64; m];
    let mut rc = vec![0.0f64; n_total];
    let mut alpha = vec![0.0f64; m];

    loop {
        *iters = iters.saturating_add(1);
        if *iters > PHASE2_PRIMAL_ITER_CAP {
            return (SimplexOutcome::Timeout(bounded_obj(c, &state.basis, &state.x_b, &state.at_upper, &state.is_basic, ubs)), state);
        }
        if options.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return (SimplexOutcome::Timeout(bounded_obj(c, &state.basis, &state.x_b, &state.at_upper, &state.is_basic, ubs)), state);
        }

        compute_reduced_costs_into(a, c, &mut basis_mgr, &state.is_basic, n_total,
                                   &state.basis, &mut y, &mut rc);

        // Pricing: most-improving non-basic (Dantzig rule).
        // at_lb: enter if rc < 0 (increasing x improves min objective)
        // at_ub: enter if rc > 0 (decreasing x improves min objective)
        let mut best_score = PIVOT_TOL;
        let mut entering: Option<usize> = None;
        for j in 0..n_total {
            if state.is_basic[j] {
                continue;
            }
            let score = if state.at_upper[j] { rc[j] } else { -rc[j] };
            if score > best_score {
                best_score = score;
                entering = Some(j);
            }
        }

        let q = match entering {
            None => {
                let obj = bounded_obj(
                    c, &state.basis, &state.x_b, &state.at_upper, &state.is_basic, ubs,
                );
                return (SimplexOutcome::Optimal(obj, y), state);
            }
            Some(q) => q,
        };

        let from_ub = state.at_upper[q];
        // dir = +1: entering from lb (x_q increases); dir = -1: from ub (decreases).
        let dir = if from_ub { -1.0f64 } else { 1.0 };

        ftran_column(a, &mut basis_mgr, q, m, &mut alpha);

        // Bounded ratio test.
        // Update rule: x_b[i] += delta[i] * theta, where delta[i] = -alpha[i] * dir.
        // Leaving at lb: delta[r] < 0, theta = x_b[r] / (-delta[r]) = x_b[r] * dir / alpha[r]...
        // Equivalent formulation (cleaner): treat "effective pivot column" as alpha[i]*dir.
        // Leaving at lb: eff_alpha[r] > PIVOT_TOL, theta = x_b[r] / eff_alpha[r]
        // Leaving at ub: eff_alpha[r] < -PIVOT_TOL, theta = (ub_r - x_b[r]) / (-eff_alpha[r])
        let mut min_step = f64::INFINITY;
        let mut leaving_row: Option<usize> = None;
        let mut leaving_at_ub = false;

        for i in 0..m {
            let eff = alpha[i] * dir;
            let xi = state.x_b[i];
            let ub_i = ubs[state.basis[i]];

            if eff > PIVOT_TOL {
                let step = xi / eff;
                if step < min_step {
                    min_step = step.max(0.0);
                    leaving_row = Some(i);
                    leaving_at_ub = false;
                }
            } else if eff < -PIVOT_TOL && ub_i.is_finite() {
                let step = (ub_i - xi) / (-eff);
                if step < min_step {
                    min_step = step.max(0.0);
                    leaving_row = Some(i);
                    leaving_at_ub = true;
                }
            }
        }

        // Check if entering variable hits its own opposite bound first.
        let ub_q = ubs[q];
        if ub_q.is_finite() && ub_q < min_step {
            // Flip entering var to opposite bound; no basis change.
            bump_bfrt_flip_invocations();
            for i in 0..m {
                state.x_b[i] -= alpha[i] * dir * ub_q;
            }
            state.at_upper[q] = !from_ub;
            basis_mgr.refactor_if_needed_timed(a, &state.basis, options.deadline);
            if basis_mgr.refactor_failed {
                return if basis_mgr.singular_basis {
                    (SimplexOutcome::SingularBasis, state)
                } else {
                    (SimplexOutcome::Timeout(bounded_obj(c, &state.basis, &state.x_b, &state.at_upper, &state.is_basic, ubs)), state)
                };
            }
            continue;
        }

        let r = match leaving_row {
            None => return (SimplexOutcome::Unbounded, state),
            Some(r) => r,
        };

        let theta = min_step;
        let leaving_col = state.basis[r];

        // Update x_b: all rows by -alpha[i]*dir*theta, then override row r.
        for i in 0..m {
            state.x_b[i] -= alpha[i] * dir * theta;
        }
        state.x_b[r] = if from_ub { ub_q - theta } else { theta };

        // Clamp near-zero (matches existing cores).
        for v in state.x_b.iter_mut() {
            if v.abs() < options.clamp_tol {
                *v = 0.0;
            }
        }

        // Bound state of outgoing / incoming columns.
        state.at_upper[leaving_col] = leaving_at_ub;
        state.at_upper[q] = false; // entering is now basic; basic vars carry no bound state
        state.is_basic[leaving_col] = false;
        state.is_basic[q] = true;
        state.basis[r] = q;

        // Eta update.
        let (cr, cv) = a.get_column(q).unwrap();
        let mut alpha_sv = SparseVec { indices: cr.to_vec(), values: cv.to_vec(), len: m };
        basis_mgr.ftran(&mut alpha_sv);
        basis_mgr.update(q, r, &alpha_sv);

        if basis_mgr.needs_refactor() {
            basis_mgr.refactor_if_needed_timed(a, &state.basis, options.deadline);
            if basis_mgr.refactor_failed {
                return if basis_mgr.singular_basis {
                    (SimplexOutcome::SingularBasis, state)
                } else {
                    (SimplexOutcome::Timeout(bounded_obj(c, &state.basis, &state.x_b, &state.at_upper, &state.is_basic, ubs)), state)
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SolverOptions;
    use crate::problem::{ConstraintType, LpProblem};
    use crate::simplex::pricing::MostInfeasibleLeaving;
    use crate::simplex::dual_advanced::bound_flip::{
        bfrt_flip_invocations, reset_bfrt_flip_invocations,
    };
    use crate::simplex::standard_form::build_bounded_standard_form;
    use crate::sparse::CscMatrix;

    /// Algebraic invariant tolerance — generous because injected warm-start
    /// states walk the loop through many BTRAN/FTRAN rounds where rounding
    /// accumulates. A tight 1e-8 would false-positive on long Timeout runs.
    const INVARIANT_TOL: f64 = 1e-6;

    /// RAII guard for the test-only `FLIP_APPLY_DISABLE` hook. Avoids leaking
    /// the disabled state across tests if an assertion unwinds.
    struct FlipApplyGuard;
    impl FlipApplyGuard {
        fn disabled() -> Self {
            set_flip_apply_disabled(true);
            Self
        }
    }
    impl Drop for FlipApplyGuard {
        fn drop(&mut self) {
            set_flip_apply_disabled(false);
        }
    }

    /// Small boxed-var LP with `c̃ = max(c,0) ≡ 0` (every `c` is negative).
    /// The dual phase on the cost-perturbed LP has *all reduced costs zero*,
    /// which keeps the loop in a degenerate stall once an lb-violation is
    /// injected — useful for cold-start sanity but *not* for Optimal-only
    /// convergence assertions. Used for dimension / immediate-terminate tests.
    ///
    ///     min  -x0 - x1
    ///     s.t.  x0 + x1 ≤ 6
    ///           x0 - x1 ≤ 2
    ///           0 ≤ x0 ≤ 4, 0 ≤ x1 ≤ 4
    /// Original optimum: x0=2, x1=4, obj=-6 (unused here — the cost-perturbed
    /// dual phase, not the original LP, drives the loop).
    fn lp_boxed_2x2_degenerate() -> LpProblem {
        let rows = vec![0, 0, 1, 1];
        let cols = vec![0, 1, 0, 1];
        let vals = vec![1.0, 1.0, 1.0, -1.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 2, 2).unwrap();
        let b = vec![6.0, 2.0];
        let c = vec![-1.0, -1.0];
        let ctypes = vec![ConstraintType::Le, ConstraintType::Le];
        let bounds = vec![(0.0, 4.0), (0.0, 4.0)];
        LpProblem::new_general(c, a, b, ctypes, bounds, None).unwrap()
    }

    /// Mixed bounds: boxed + half-finite + fixed. Fixed vars have ub = 0
    /// after the lb-shift, so BFRT must early-skip (weight = 0). All `c < 0`
    /// → degenerate dual phase (`c̃ = 0`); used to exercise the
    /// fixed-variable handling, not optimality.
    fn lp_mixed_bounds_degenerate() -> LpProblem {
        let n = 4;
        let m = 2;
        let rows = vec![0, 0, 0, 0, 1, 1];
        let cols = vec![0, 1, 2, 3, 0, 1];
        let vals = vec![1.0, 1.0, 1.0, 1.0, 2.0, 1.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        let b = vec![10.0, 8.0];
        let c = vec![-1.0, -2.0, -1.0, 0.0];
        let ctypes = vec![ConstraintType::Le, ConstraintType::Le];
        let bounds = vec![
            (0.0, 3.0),
            (0.0, f64::INFINITY),
            (0.0, 5.0),
            (2.0, 2.0),
        ];
        LpProblem::new_general(c, a, b, ctypes, bounds, None).unwrap()
    }

    /// Fixture descriptor: an LP and a list of synthetic primal
    /// infeasibilities to inject into the cold-start `x_B` before running
    /// `iterate`. Positive costs (`c̃ = max(c, 0) = c`) keep reduced costs
    /// dual-feasible at cold start so BFRT has multiple breakpoints to walk
    /// when the loop tries to recover from the injection.
    struct InvariantFixture {
        name: &'static str,
        problem: LpProblem,
        /// (row, magnitude > 0) pairs — `state.x_b[row] = -magnitude`.
        inject_negative_x_b: Vec<(usize, f64)>,
    }

    /// 1-row, 2 boxed positive-cost vars — minimal shape that still walks
    /// past one breakpoint when the residual is large enough.
    ///     min  x0 + 2 x1
    ///     s.t. x0 + x1 ≤ 5
    ///          0 ≤ x0 ≤ 4, 0 ≤ x1 ≤ 3
    fn fixture_one_row_two_boxed() -> InvariantFixture {
        let rows = vec![0, 0];
        let cols = vec![0, 1];
        let vals = vec![1.0, 1.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 1, 2).unwrap();
        let b = vec![5.0];
        let c = vec![1.0, 2.0];
        let ctypes = vec![ConstraintType::Le];
        let bounds = vec![(0.0, 4.0), (0.0, 3.0)];
        let problem = LpProblem::new_general(c, a, b, ctypes, bounds, None).unwrap();
        InvariantFixture {
            name: "one_row_two_boxed",
            problem,
            inject_negative_x_b: vec![(0, 3.0)],
        }
    }

    /// 2 rows, 3 boxed vars with distinct positive costs and tight upper
    /// bounds. Designed so BFRT crosses multiple breakpoints during the
    /// infeasibility recovery (`|residual| > u_0 · trow[0]`).
    ///     min  x0 + 3 x1 + 5 x2
    ///     s.t. x0 + x1 + x2 ≤ 7
    ///          0.5 x0 + x1 + 2 x2 ≤ 6
    ///          0 ≤ x0 ≤ 2, 0 ≤ x1 ≤ 2, 0 ≤ x2 ≤ 1
    fn fixture_two_rows_three_boxed() -> InvariantFixture {
        let rows = vec![0, 0, 0, 1, 1, 1];
        let cols = vec![0, 1, 2, 0, 1, 2];
        let vals = vec![1.0, 1.0, 1.0, 0.5, 1.0, 2.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 2, 3).unwrap();
        let b = vec![7.0, 6.0];
        let c = vec![1.0, 3.0, 5.0];
        let ctypes = vec![ConstraintType::Le, ConstraintType::Le];
        let bounds = vec![(0.0, 2.0), (0.0, 2.0), (0.0, 1.0)];
        let problem = LpProblem::new_general(c, a, b, ctypes, bounds, None).unwrap();
        InvariantFixture {
            name: "two_rows_three_boxed",
            problem,
            inject_negative_x_b: vec![(0, 3.0)],
        }
    }

    /// Reconstruct the full primal vector from `(basis, x_b, at_upper)` and
    /// compute the algebraic residual `A · x_full − b_effective`. The flip-
    /// apply step is exactly what keeps this residual at zero: when `at_upper`
    /// toggles for column k, x_B must absorb `± u_k · B^{-1} a_k` so that
    /// `A · x_full` stays equal to `b_effective = b` (where `b_effective` is
    /// the original problem RHS — every flip is offset by a corresponding x_B
    /// move).
    ///
    /// Returns the max absolute component of the residual vector. A correct
    /// `iterate` keeps this at `O(numerical noise)`; the no-op flip apply
    /// leaves a residual of `Σ_k u_k · a_k` (one per executed flip) and so
    /// blows past `INVARIANT_TOL` after the first flip.
    fn basis_rhs_residual(
        state: &BoundedDualState,
        bsf: &BoundedStandardForm,
    ) -> f64 {
        let mut x_full = vec![0.0; bsf.n_total];
        for (pos, &j) in state.basis.iter().enumerate() {
            x_full[j] = state.x_b[pos];
        }
        for j in 0..bsf.n_total {
            if state.at_upper[j] && !state.is_basic[j] {
                x_full[j] = bsf.upper_bounds[j];
            }
        }
        let mut residual = vec![0.0; bsf.m];
        for j in 0..bsf.n_total {
            let xj = x_full[j];
            if xj == 0.0 {
                continue;
            }
            if let Ok((rows, vals)) = bsf.a.get_column(j) {
                for (k, &row) in rows.iter().enumerate() {
                    residual[row] += vals[k] * xj;
                }
            }
        }
        for i in 0..bsf.m {
            residual[i] -= bsf.b[i];
        }
        residual.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
    }

    #[test]
    fn cold_state_from_bsf_has_consistent_dimensions() {
        let lp = lp_boxed_2x2_degenerate();
        let bsf = build_bounded_standard_form(&lp);
        let state = BoundedDualState::cold(&bsf, &bsf.b);
        assert_eq!(state.basis.len(), bsf.m);
        assert_eq!(state.x_b.len(), bsf.m);
        assert_eq!(state.at_upper.len(), bsf.n_total);
        assert_eq!(state.is_basic.len(), bsf.n_total);
        for j in 0..bsf.n_shifted {
            assert!(!state.is_basic[j]);
            assert!(!state.at_upper[j]);
        }
        assert_eq!(state.x_b, bsf.b);
    }

    /// Cold-start dual phase on Le-only b≥0 input terminates immediately
    /// (x_B = b ≥ 0 already primal-feasible). No iterations beyond the
    /// optimality probe.
    #[test]
    fn cold_dual_le_only_terminates_immediately() {
        let lp = lp_boxed_2x2_degenerate();
        let bsf = build_bounded_standard_form(&lp);
        let opts = SolverOptions::default();
        let (outcome, state) = solve_bounded_dual(&bsf, &bsf.a, &bsf.b, &bsf.c, &opts, &bsf.upper_bounds, &mut MostInfeasibleLeaving);
        match outcome {
            BoundedOutcome::Optimal(_, _) => {}
            other => panic!("expected Optimal, got {:?}", other),
        }
        assert_eq!(state.iterations, 1);
    }

    /// Fixed variable (lb=ub ⇒ shifted upper=0) is handled by BFRT: the
    /// weight contribution is 0, so no flip-set inflation. Drives the
    /// "BFRT early skip" path. Outcome can be Optimal or Timeout depending
    /// on cycling; this test asserts only that the loop does not panic or
    /// return a logically impossible status.
    #[test]
    fn fixed_variable_does_not_break_iteration() {
        let lp = lp_mixed_bounds_degenerate();
        let bsf = build_bounded_standard_form(&lp);
        let mut state = BoundedDualState::cold(&bsf, &bsf.b);
        state.x_b[0] = -0.5;
        let opts = SolverOptions {
            deadline: Some(std::time::Instant::now() + std::time::Duration::from_millis(500)),
            ..SolverOptions::default()
        };
        let (outcome, _state) = iterate(state, &bsf, &bsf.a, &bsf.c, &opts, &bsf.upper_bounds, &mut MostInfeasibleLeaving);
        match outcome {
            BoundedOutcome::Optimal(_, _) => {}
            BoundedOutcome::Timeout(_) => {}
            BoundedOutcome::UbViolationOutOfScope { .. } => {}
            other => panic!("unexpected outcome {:?}", other),
        }
    }

    /// Compute the same `iterate` residual twice — once with the production
    /// flip apply, once with `FLIP_APPLY_DISABLE` set — and return the per-
    /// fixture residual pair, the executed flip count, and the pre-iterate
    /// residual. Caller asserts the algebraic relation between them.
    ///
    /// Phase-1-only design constraint: with `c̃ = max(c, 0)` and an injected
    /// `x_B[r] < 0` the dual loop cannot in general re-establish primal
    /// feasibility — full anti-cycling (Bland fallback, lex perturbation)
    /// lives in `core.rs` and is reused in follow-up wiring tasks. What the
    /// loop *must* maintain regardless of outcome is the column-update
    /// invariant `A · x_full = b_effective` (where `b_effective ≡ b` because
    /// injection is the only perturbation), which the flip apply preserves
    /// exactly and the no-op breaks by `Σ_k u_k · a_k` per executed flip.
    fn measure_iterate_residual(
        fx: &InvariantFixture,
        deadline_ms: u64,
        flip_disabled: bool,
    ) -> (f64, f64, u64) {
        reset_bfrt_flip_invocations();
        let bsf = build_bounded_standard_form(&fx.problem);
        let mut state = BoundedDualState::cold(&bsf, &bsf.b);
        for &(row, mag) in &fx.inject_negative_x_b {
            state.x_b[row] = -mag;
        }
        let pre_residual = basis_rhs_residual(&state, &bsf);
        let opts = SolverOptions {
            deadline: Some(
                std::time::Instant::now() + std::time::Duration::from_millis(deadline_ms),
            ),
            ..SolverOptions::default()
        };
        let _guard = if flip_disabled {
            Some(FlipApplyGuard::disabled())
        } else {
            None
        };
        let (_outcome, post) = iterate(state, &bsf, &bsf.a, &bsf.c, &opts, &bsf.upper_bounds, &mut MostInfeasibleLeaving);
        let post_residual = basis_rhs_residual(&post, &bsf);
        let flips = bfrt_flip_invocations();
        (pre_residual, post_residual, flips)
    }

    /// **Correctness sentinel (table-driven, multi-fixture).** For every
    /// fixture: run `iterate`, recompute the algebraic residual
    /// `‖A · x_full − b‖_∞`, and verify the production loop preserves it
    /// (i.e. residual_post ≈ residual_pre). The flip apply line
    /// `x_B -= alpha_flip · weight` is exactly the algebra that keeps this
    /// invariant — if the loop toggles `at_upper[k]` without absorbing the
    /// `u_k · B^{-1} a_k` change into `x_B`, the residual drifts by
    /// `u_k · a_k` per flip.
    ///
    /// Multi-fixture: 1-row/2-boxed and 2-row/3-boxed shapes, distinct
    /// upper bounds and reduced-cost gradients, share the same assertion.
    /// Companion no-op proof below.
    #[test]
    fn flip_apply_preserves_basis_rhs_invariant() {
        let fixtures = [fixture_one_row_two_boxed(), fixture_two_rows_three_boxed()];
        let mut at_least_one_flip = false;
        for fx in &fixtures {
            // Short deadline keeps accumulated FP noise small (iterate runs
            // ~50 k pivots/10 ms; per-pivot BTRAN/FTRAN drift < 1e-12).
            let (pre, post, flips) = measure_iterate_residual(fx, 10, false);
            let drift = (post - pre).abs();
            assert!(
                drift < INVARIANT_TOL,
                "{}: production iterate drifted the algebraic invariant by \
                 {drift:.3e} (pre={pre:.3e}, post={post:.3e}, flips={flips}) \
                 — flip apply is no longer preserving A·x_full = b",
                fx.name,
            );
            if flips > 0 {
                at_least_one_flip = true;
            }
        }
        assert!(
            at_least_one_flip,
            "no fixture exercised a BFRT flip — the sentinel proves nothing \
             about the flip apply path"
        );
    }

    /// **No-op proof for `flip_apply_preserves_basis_rhs_invariant`.** With
    /// the `FLIP_APPLY_DISABLE` hook engaged, `iterate` toggles `at_upper[k]`
    /// without updating `x_B`. At least one fixture must drift the invariant
    /// past `INVARIANT_TOL`; otherwise the correctness sentinel would pass
    /// on a broken flip apply (the pilot87/speed-f2/speed-b1 anti-pattern).
    #[test]
    fn flip_apply_preserves_basis_rhs_invariant_noop_proof() {
        let fixtures = [fixture_one_row_two_boxed(), fixture_two_rows_three_boxed()];
        let mut max_drift = 0.0_f64;
        let mut max_drift_fixture = "<none>";
        let mut total_flips = 0;
        for fx in &fixtures {
            let (pre, post, flips) = measure_iterate_residual(fx, 10, true);
            let drift = (post - pre).abs();
            total_flips += flips;
            if drift > max_drift {
                max_drift = drift;
                max_drift_fixture = fx.name;
            }
        }
        assert!(
            total_flips > 0,
            "no BFRT flip happened under FLIP_APPLY_DISABLE either — the \
             fixture set does not exercise the flip path at all"
        );
        assert!(
            max_drift > INVARIANT_TOL,
            "no-op flip apply produced max drift {max_drift:.3e} (fixture \
             '{max_drift_fixture}', total_flips={total_flips}) ≤ {INVARIANT_TOL:.0e} \
             — the production correctness sentinel could not have detected \
             the broken flip apply"
        );
    }

    /// Effectiveness sentinel: BFRT flip count strictly > 0 after a residual
    /// that spans multiple breakpoints. Pairs with `flip_apply_preserves_basis_rhs_invariant`
    /// which verifies the apply step is load-bearing on the same fixtures.
    /// Strengthened to also confirm the algebraic invariant is preserved
    /// across the flips (a count without the apply update would silently
    /// pass — pilot87 anti-pattern).
    #[test]
    fn bfrt_flip_count_positive_when_residual_spans_breakpoints() {
        let fx = fixture_two_rows_three_boxed();
        let (pre, post, flips) = measure_iterate_residual(&fx, 10, false);
        assert!(
            flips >= 1,
            "expected BFRT flip count ≥ 1, got {flips} — fixture no longer \
             exercises BFRT"
        );
        let drift = (post - pre).abs();
        assert!(
            drift < INVARIANT_TOL,
            "{}: invariant drifted by {drift:.3e} despite {flips} flips — \
             flip apply not preserving A·x_full = b",
            fx.name,
        );
    }

    /// Inject a single lb-violation and verify the loop makes **measurable
    /// progress**: BFRT must be invoked at least once and the invariant must
    /// remain intact (so the loop's pivots/flips are algebraically correct,
    /// even if anti-cycling eventually halts it with a Timeout). Pure
    /// Optimal-only convergence requires Bland fallback / lex perturbation
    /// which is in `core.rs` and out of scope here.
    #[test]
    fn inject_lb_violation_makes_progress_boxed() {
        let fx = fixture_two_rows_three_boxed();
        let (pre, post, flips) = measure_iterate_residual(&fx, 10, false);
        assert!(
            flips > 0,
            "{}: zero BFRT invocations — loop did no flip work",
            fx.name,
        );
        let drift = (post - pre).abs();
        assert!(
            drift < INVARIANT_TOL,
            "{}: invariant drifted by {drift:.3e} during {flips} flips — \
             pivot / flip algebra broken",
            fx.name,
        );
    }

    /// UbViolationOutOfScope must be reachable: inject `x_B[r] > u_basis[r]`
    /// with no lb-violation present. The loop must return the specialised
    /// variant (not `Timeout`) so the wiring layer can route deterministically.
    #[test]
    fn ub_violation_returns_specialised_outcome() {
        let fx = fixture_two_rows_three_boxed();
        let bsf = build_bounded_standard_form(&fx.problem);
        let mut state = BoundedDualState::cold(&bsf, &bsf.b);
        // The slack basis column for row 0 has upper = +∞ in the bounded form
        // (slacks have no finite ub by construction). To trigger the
        // ub-violation branch we need a basis row whose basic column carries a
        // finite ub; pivot a structural boxed var into the basis manually by
        // overwriting state.basis[0] = 2 (col index of x2, ub=1), set x_b[0]
        // beyond x2's ub, and leave x_b[1] feasible. No lb violation ⇒
        // pricing exits the loop at the ub-violation check.
        let target_col = 2; // x2 with ub = 1.0 (post-shift).
        assert!(bsf.upper_bounds[target_col].is_finite());
        state.basis[0] = target_col;
        state.is_basic[target_col] = true;
        // Keep the second basis row's slack basic; clear the original slack
        // for row 0 from the basic flag set (the LU will see an inconsistent
        // basis but iterate must still terminate on the pricing check before
        // FTRAN gets invoked).
        let prev_slack = bsf.initial_basis[0];
        if prev_slack != target_col {
            state.is_basic[prev_slack] = false;
        }
        state.x_b[0] = bsf.upper_bounds[target_col] + 1.5; // strictly above ub
        state.x_b[1] = state.x_b[1].max(0.0);
        let opts = SolverOptions {
            deadline: Some(std::time::Instant::now() + std::time::Duration::from_millis(200)),
            ..SolverOptions::default()
        };
        let (outcome, _post) = iterate(state, &bsf, &bsf.a, &bsf.c, &opts, &bsf.upper_bounds, &mut MostInfeasibleLeaving);
        match outcome {
            BoundedOutcome::UbViolationOutOfScope { row, .. } => {
                assert_eq!(row, 0);
            }
            // Singular-basis is also acceptable: the synthetic basis swap
            // may not factor. Either way, Timeout would mean the ub-violation
            // detection failed to short-circuit, which is the regression we
            // are guarding against.
            BoundedOutcome::SingularBasis => {}
            other => panic!(
                "expected UbViolationOutOfScope or SingularBasis, got {:?}",
                other
            ),
        }
    }

    // ── extract_solution_bounded / extract_dual_info_bounded tests ────────

    /// RAII guard for `AT_UPPER_APPLY_DISABLE`.
    struct AtUpperApplyGuard;
    impl AtUpperApplyGuard {
        fn disabled() -> Self {
            set_at_upper_apply_disabled(true);
            Self
        }
    }
    impl Drop for AtUpperApplyGuard {
        fn drop(&mut self) {
            set_at_upper_apply_disabled(false);
        }
    }

    /// Table-driven fixture for extract_solution_bounded.
    ///
    /// Each row: expected solution after manually placing at least one
    /// non-basic variable at its upper bound (at_upper = true).
    struct ExtractFixture {
        name: &'static str,
        problem: LpProblem,
        /// Columns to flip to at_upper in the cold-start state.
        flip_to_upper: Vec<usize>,
        /// Expected solution after the flip.
        expected: Vec<f64>,
    }

    /// Fixture: lb=0, ub=∞ — no variable at upper; extract_solution_bounded
    /// must match extract_solution on the same cold-start state.
    fn fixture_unbounded_compat() -> ExtractFixture {
        // min x0 + 2 x1, x0 + x1 ≤ 3, x0,x1 ≥ 0 (no UB)
        // Optimal (perturbed = original since c > 0): x0=x1=0
        let rows = vec![0, 0];
        let cols = vec![0, 1];
        let vals = vec![1.0, 1.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 1, 2).unwrap();
        let problem = LpProblem::new_general(
            vec![1.0, 2.0],
            a,
            vec![3.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        ).unwrap();
        ExtractFixture {
            name: "unbounded_compat",
            problem,
            flip_to_upper: vec![], // nothing at upper
            expected: vec![0.0, 0.0],
        }
    }

    /// Fixture: lb=0, ub=1 — x0 manually placed at ub=1.
    fn fixture_boxed_ub1() -> ExtractFixture {
        // min x0 + x1, x0 + x1 ≤ 5, 0 ≤ x0 ≤ 1, 0 ≤ x1 ≤ 1
        // build_bounded: n_shifted=2, upper_bounds=[1,1,∞]
        // manual: at_upper[0]=true (x0=1), x1=0 (at lb)
        // expected: x0=1, x1=0
        let rows = vec![0, 0];
        let cols = vec![0, 1];
        let vals = vec![1.0, 1.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 1, 2).unwrap();
        let problem = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 1.0), (0.0, 1.0)],
            None,
        ).unwrap();
        ExtractFixture {
            name: "boxed_ub1",
            problem,
            flip_to_upper: vec![0], // x0 at ub=1
            expected: vec![1.0, 0.0],
        }
    }

    /// Fixture: lb=−5, ub=5 — shifted variable at ub=10 (post-shift), orig=5.
    fn fixture_nonzero_lb() -> ExtractFixture {
        // min x0 + x1, x0 + x1 ≤ 5, -5 ≤ x0 ≤ 5, -5 ≤ x1 ≤ 5
        // build_bounded: x0_shifted = x0 + 5, ub_shifted=10, offset=-5
        // manual: at_upper[0]=true (x0_shifted=10 → x0=5)
        // expected: x0=5, x1=-5 (x1 at lb default)
        let rows = vec![0, 0];
        let cols = vec![0, 1];
        let vals = vec![1.0, 1.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 1, 2).unwrap();
        let problem = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(-5.0, 5.0), (-5.0, 5.0)],
            None,
        ).unwrap();
        ExtractFixture {
            name: "nonzero_lb",
            problem,
            flip_to_upper: vec![0], // x0_shifted at ub=10 → x0=5
            expected: vec![5.0, -5.0],
        }
    }

    /// Build a `BoundedDualState` that reflects `flip_to_upper` columns being
    /// at their upper bounds. Adjusts `x_b` (single basic slack) so that
    /// `A * x_full = b` holds.
    fn state_with_flips(bsf: &BoundedStandardForm, flip_cols: &[usize]) -> BoundedDualState {
        let mut state = BoundedDualState::cold(bsf, &bsf.b);
        for &j in flip_cols {
            assert!(!state.is_basic[j], "can only flip non-basic columns");
            assert!(bsf.upper_bounds[j].is_finite(), "flip target must have finite ub");
            state.at_upper[j] = true;
            // Adjust x_b[i] for each basic row: x_b[i] -= upper_bounds[j] * A[i,j]
            let (rows, vals) = bsf.a.get_column(j).unwrap();
            for (k, &row) in rows.iter().enumerate() {
                if row < bsf.m {
                    state.x_b[row] -= vals[k] * bsf.upper_bounds[j];
                }
            }
        }
        state
    }

    /// Table-driven correctness: three bound patterns each check that the
    /// expected original-variable value is recovered after placing one or
    /// more variables at their upper bound.
    #[test]
    fn extract_solution_bounded_multi_fixture() {
        let fixtures = [
            fixture_unbounded_compat(),
            fixture_boxed_ub1(),
            fixture_nonzero_lb(),
        ];
        const EPS: f64 = 1e-10;
        for fx in &fixtures {
            let bsf = build_bounded_standard_form(&fx.problem);
            let state = state_with_flips(&bsf, &fx.flip_to_upper);
            let sol = extract_solution_bounded(&bsf, &state, &[]);
            assert_eq!(
                sol.len(),
                fx.expected.len(),
                "{}: solution length mismatch",
                fx.name
            );
            for (i, (&got, &want)) in sol.iter().zip(fx.expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < EPS,
                    "{}: solution[{}] = {got:.6e}, expected {want:.6e}",
                    fx.name,
                    i
                );
            }
        }
    }

    /// Equivalence: for the unbounded-compat fixture (all at_upper false),
    /// extract_solution_bounded must give the same result as the unbounded
    /// `extract_solution` on the same state — they only diverge when at_upper
    /// is true.
    #[test]
    fn extract_solution_bounded_matches_unbounded_when_no_at_upper() {
        use crate::simplex::primal::extract_solution;
        use crate::simplex::standard_form::build_standard_form;
        const EPS: f64 = 1e-10;
        let fx = fixture_unbounded_compat();
        let bsf = build_bounded_standard_form(&fx.problem);
        let sf = build_standard_form(&fx.problem);
        // Run bounded dual (terminates immediately; no at_upper set).
        let opts = SolverOptions::default();
        let (outcome, state) = solve_bounded_dual(&bsf, &bsf.a, &bsf.b, &bsf.c, &opts, &bsf.upper_bounds, &mut MostInfeasibleLeaving);
        assert!(
            matches!(outcome, BoundedOutcome::Optimal(..)),
            "expected Optimal, got {:?}", outcome
        );
        // All non-basics must be at lb for the equivalence to hold.
        let any_upper = state.at_upper.iter().enumerate()
            .any(|(j, &u)| u && !state.is_basic[j]);
        assert!(!any_upper, "unexpected at_upper set in unbounded-compat fixture");

        let sol_bounded = extract_solution_bounded(&bsf, &state, &[]);
        // For the unscaled standard form, basis and x_b come directly from bsf initial state.
        let sol_std = extract_solution(&sf, &state.basis, &state.x_b, &[]);
        for (i, (&a, &b)) in sol_bounded.iter().zip(sol_std.iter()).enumerate() {
            assert!(
                (a - b).abs() < EPS,
                "bounded[{}]={a:.3e} vs unbounded[{}]={b:.3e}", i, i
            );
        }
    }

    /// No-op proof: disabling the at_upper correction in
    /// extract_solution_bounded causes the boxed-ub1 sentinel to produce a
    /// wrong solution. The no-op result must differ from the correct result
    /// by more than EPS.
    #[test]
    fn extract_solution_bounded_noop_proof() {
        const EPS: f64 = 1e-6;
        let fx = fixture_boxed_ub1();
        let bsf = build_bounded_standard_form(&fx.problem);
        let state = state_with_flips(&bsf, &fx.flip_to_upper);

        let sol_correct = extract_solution_bounded(&bsf, &state, &[]);
        let sol_noop = {
            let _guard = AtUpperApplyGuard::disabled();
            extract_solution_bounded(&bsf, &state, &[])
        };

        // Correct: x0=1 (at ub). No-op: x0=0 (at_upper correction skipped).
        assert!(
            (sol_correct[0] - 1.0).abs() < EPS,
            "correct solution[0] should be 1.0, got {}", sol_correct[0]
        );
        assert!(
            sol_noop[0].abs() < EPS,
            "noop solution[0] should be 0.0 (correction disabled), got {}", sol_noop[0]
        );
        let max_diff = sol_correct.iter().zip(sol_noop.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff > EPS,
            "no-op proof FAILED: correct and noop solutions are identical (diff={max_diff:.3e}) \
             — the at_upper correction is not load-bearing in this fixture"
        );
    }

    /// extract_dual_info_bounded basic smoke: row-negated flag inversion and
    /// slack computation from the same fixture used in the equivalence test.
    #[test]
    fn extract_dual_info_bounded_smoke() {
        use crate::simplex::standard_form::extract_dual_info;
        use crate::simplex::standard_form::build_standard_form;
        let fx = fixture_unbounded_compat();
        let bsf = build_bounded_standard_form(&fx.problem);
        let sf = build_standard_form(&fx.problem);
        let opts = SolverOptions::default();
        let (outcome, state) = solve_bounded_dual(&bsf, &bsf.a, &bsf.b, &bsf.c, &opts, &bsf.upper_bounds, &mut MostInfeasibleLeaving);
        let (_, y_std) = match outcome {
            BoundedOutcome::Optimal(obj, y) => (obj, y),
            other => panic!("expected Optimal, got {:?}", other),
        };
        let solution = extract_solution_bounded(&bsf, &state, &[]);
        let (dual_b, rc_b, slack_b) = extract_dual_info_bounded(
            &bsf, &fx.problem, &y_std, &solution, &[],
        );
        // Compare with the legacy path on the equivalent standard form.
        let (dual_s, rc_s, slack_s) = extract_dual_info(
            &sf, &fx.problem, &y_std[..sf.m.min(y_std.len())], &solution, &[],
        );
        // For a Le-only non-negated problem the row_negated flags are all false;
        // the dual vectors must match entry for entry.
        const EPS: f64 = 1e-8;
        for i in 0..dual_b.len() {
            assert!(
                (dual_b[i] - dual_s[i]).abs() < EPS,
                "dual[{}]: bounded={:.3e}, std={:.3e}", i, dual_b[i], dual_s[i]
            );
        }
        for j in 0..rc_b.len() {
            assert!(
                (rc_b[j] - rc_s[j]).abs() < EPS,
                "rc[{}]: bounded={:.3e}, std={:.3e}", j, rc_b[j], rc_s[j]
            );
        }
        for i in 0..slack_b.len() {
            assert!(
                (slack_b[i] - slack_s[i]).abs() < EPS,
                "slack[{}]: bounded={:.3e}, std={:.3e}", i, slack_b[i], slack_s[i]
            );
        }
    }

    // ── phase2_primal_bounded tests ────────────────────────────────────────

    /// End-to-end Phase 2: start from bounded dual Optimal (perturbed costs),
    /// run phase2_primal_bounded with original costs, verify the known optimal.
    ///
    /// LP: min -x0 - x1, x0+x1 ≤ 6, x0-x1 ≤ 2, 0 ≤ x0 ≤ 4, 0 ≤ x1 ≤ 4
    /// Known optimal: x0=4, x1=2, obj=-6.
    #[test]
    fn phase2_primal_bounded_reaches_known_optimal() {
        let lp = lp_boxed_2x2_degenerate();
        let bsf = build_bounded_standard_form(&lp);
        let opts = SolverOptions::default();
        // Bounded dual: c̃ = max(c,0) = [0,0] → terminates immediately with slack basis.
        let (dual_outcome, dual_state) = solve_bounded_dual(&bsf, &bsf.a, &bsf.b, &bsf.c, &opts, &bsf.upper_bounds, &mut MostInfeasibleLeaving);
        assert!(matches!(dual_outcome, BoundedOutcome::Optimal(..)));

        let mut iters = 0usize;
        let (p2_outcome, p2_state) = phase2_primal_bounded(
            &bsf, dual_state, &bsf.a, &bsf.c, &opts, &mut iters, &bsf.upper_bounds,
        );
        assert!(
            matches!(p2_outcome, SimplexOutcome::Optimal(..)),
            "Phase 2 did not reach Optimal: {:?}", p2_outcome
        );
        let sol = extract_solution_bounded(&bsf, &p2_state, &[]);
        let obj: f64 = lp.c.iter().zip(sol.iter()).map(|(c, x)| c * x).sum();
        assert!(
            (obj - (-6.0)).abs() < 1e-6,
            "expected obj=-6.0, got {obj:.6e}"
        );
        assert!(
            (sol[0] - 4.0).abs() < 1e-6 && (sol[1] - 2.0).abs() < 1e-6,
            "expected x=(4,2), got ({:.3e},{:.3e})", sol[0], sol[1]
        );
        assert!(iters > 0, "phase2 should have made at least one iteration");
    }

    /// Phase 2 with no original-cost improvement needed: perturbed ≡ original
    /// (all c ≥ 0). The loop must return Optimal on the very first pricing pass.
    #[test]
    fn phase2_primal_bounded_noop_when_already_optimal() {
        let fx = fixture_one_row_two_boxed(); // c = [1, 2] (all positive)
        let bsf = build_bounded_standard_form(&fx.problem);
        let opts = SolverOptions::default();
        let (dual_outcome, dual_state) = solve_bounded_dual(&bsf, &bsf.a, &bsf.b, &bsf.c, &opts, &bsf.upper_bounds, &mut MostInfeasibleLeaving);
        assert!(matches!(dual_outcome, BoundedOutcome::Optimal(..)));
        let mut iters = 0usize;
        let (p2_outcome, _) = phase2_primal_bounded(
            &bsf, dual_state, &bsf.a, &bsf.c, &opts, &mut iters, &bsf.upper_bounds,
        );
        assert!(
            matches!(p2_outcome, SimplexOutcome::Optimal(..)),
            "expected Optimal, got {:?}", p2_outcome
        );
        // c = [1,2] ≥ 0 so c̃ = c; dual already optimal for original costs.
        assert_eq!(iters, 1, "should terminate after one pricing pass (no improvement)");
    }

    // ── bounded_obj Timeout sentinel ─────────────────────────────────────────

    /// Sentinel: `iterate` Phase 1 dual Timeout returns the bounded objective,
    /// including non-basic at-upper-bound contributions.
    ///
    /// Table-driven over two fixtures. Each pre-sets one structural variable at
    /// its upper bound via `state_with_flips`, then triggers a deadline Timeout
    /// on the first iteration (already-expired deadline). The test asserts
    /// `returned_obj ≈ bounded_obj` within 1e-10.
    ///
    /// No-op proof is embedded: `|bounded_obj − basic_obj| ≥ MIN_CONTRIBUTION`
    /// guarantees the test fails if the Timeout path were reverted to
    /// `basic_obj` (diff > MIN_CONTRIBUTION >> 1e-10).
    #[test]
    fn phase1_dual_timeout_obj_matches_bounded_obj() {
        const EPS: f64 = 1e-10;
        const MIN_CONTRIBUTION: f64 = 0.5;

        let p1_problem = fixture_one_row_two_boxed().problem;
        let p2_problem = fixture_two_rows_three_boxed().problem;
        let bsfs = [
            ("one_row_two_boxed_x1_at_upper",    build_bounded_standard_form(&p1_problem), 1usize),
            ("two_rows_three_boxed_x0_at_upper",  build_bounded_standard_form(&p2_problem), 0usize),
        ];

        for (name, bsf, flip_col) in &bsfs {
            let state = state_with_flips(bsf, &[*flip_col]);

            let exp_bounded = bounded_obj(
                &bsf.c, &state.basis, &state.x_b,
                &state.at_upper, &state.is_basic, &bsf.upper_bounds,
            );
            let exp_basic = basic_obj(&bsf.c, &state.basis, &state.x_b);

            assert!(
                (exp_bounded - exp_basic).abs() >= MIN_CONTRIBUTION,
                "{name}: fixture degenerate — bounded_obj={exp_bounded:.6e} \
                 basic_obj={exp_basic:.6e} differ by {:.3e} < {MIN_CONTRIBUTION:.1e}",
                (exp_bounded - exp_basic).abs()
            );

            let deadline = std::time::Instant::now()
                - std::time::Duration::from_millis(1);
            let opts = SolverOptions { deadline: Some(deadline), ..SolverOptions::default() };
            let (outcome, _) = iterate(state, bsf, &bsf.a, &bsf.c, &opts, &bsf.upper_bounds, &mut MostInfeasibleLeaving);
            match outcome {
                BoundedOutcome::Timeout(obj) => {
                    assert!(
                        (obj - exp_bounded).abs() < EPS,
                        "{name}: Phase 1 dual Timeout obj={obj:.6e} differs from \
                         bounded_obj={exp_bounded:.6e} by {:.3e}; \
                         basic_obj={exp_basic:.6e} — at_upper contributions missing",
                        (obj - exp_bounded).abs()
                    );
                }
                other => panic!("{name}: expected Timeout (expired deadline), got {other:?}"),
            }
        }
    }

    /// Sentinel: `phase2_primal_bounded` Timeout returns the bounded objective,
    /// including non-basic at-upper-bound contributions.
    ///
    /// Pre-sets x0 at ub=1 in a primal-feasible state and forces a deadline
    /// Timeout on the first iteration. Asserts returned obj ≈ bounded_obj.
    /// No-op proof embedded: diff from basic_obj ≥ MIN_CONTRIBUTION.
    #[test]
    fn phase2_primal_timeout_obj_matches_bounded_obj() {
        const EPS: f64 = 1e-10;
        const MIN_CONTRIBUTION: f64 = 0.5;

        let problem = fixture_boxed_ub1().problem;
        let bsf = build_bounded_standard_form(&problem);
        let state = state_with_flips(&bsf, &[0]); // x0 at ub=1

        let exp_bounded = bounded_obj(
            &bsf.c, &state.basis, &state.x_b,
            &state.at_upper, &state.is_basic, &bsf.upper_bounds,
        );
        let exp_basic = basic_obj(&bsf.c, &state.basis, &state.x_b);

        assert!(
            (exp_bounded - exp_basic).abs() >= MIN_CONTRIBUTION,
            "fixture degenerate — bounded_obj={exp_bounded:.6e} \
             basic_obj={exp_basic:.6e} differ by {:.3e} < {MIN_CONTRIBUTION:.1e}",
            (exp_bounded - exp_basic).abs()
        );

        let deadline = std::time::Instant::now() - std::time::Duration::from_millis(1);
        let opts = SolverOptions { deadline: Some(deadline), ..SolverOptions::default() };
        let mut iters = 0usize;
        let (outcome, _) = phase2_primal_bounded(
            &bsf, state, &bsf.a, &bsf.c, &opts, &mut iters, &bsf.upper_bounds,
        );
        match outcome {
            SimplexOutcome::Timeout(obj) => {
                assert!(
                    (obj - exp_bounded).abs() < EPS,
                    "Phase 2 primal Timeout obj={obj:.6e} differs from \
                     bounded_obj={exp_bounded:.6e} by {:.3e}; \
                     basic_obj={exp_basic:.6e} — at_upper contributions missing",
                    (obj - exp_bounded).abs()
                );
            }
            other => panic!("expected Timeout (expired deadline), got {other:?}"),
        }
    }

    /// **Sentinel:** expired deadline before O(m²) γ init must return `Timeout`
    /// immediately without entering the iteration loop.
    ///
    /// Uses `DualSteepestEdgeLeaving` (needs_sigma = true) so the pre-loop
    /// deadline guard is the only early-exit path for `recompute_gamma_truth`.
    /// No-op proof: with a live deadline the same LP must NOT return Timeout
    /// (it converges immediately because b ≥ 0 is already primal-feasible).
    #[test]
    fn dse_expired_deadline_returns_timeout_before_gamma_init() {
        use crate::simplex::dual_advanced::steepest_edge::DualSteepestEdgeLeaving;
        const EPS: f64 = 1e-10;

        let fx = fixture_one_row_two_boxed();
        let bsf = build_bounded_standard_form(&fx.problem);
        // Cold-start state: x_B = b ≥ 0 → already primal feasible. Without a
        // deadline the loop terminates Optimal on the first pricing probe.
        let state_fresh = || BoundedDualState::cold(&bsf, &bsf.b);

        // ── production run (live deadline) must NOT be Timeout ──────────────
        let opts_live = SolverOptions {
            deadline: Some(std::time::Instant::now() + std::time::Duration::from_millis(500)),
            ..SolverOptions::default()
        };
        let (live_outcome, _) = iterate(
            state_fresh(), &bsf, &bsf.a, &bsf.c, &opts_live, &bsf.upper_bounds,
            &mut DualSteepestEdgeLeaving::new(bsf.m),
        );
        match live_outcome {
            BoundedOutcome::Optimal(_, _) => {}
            other => panic!("no-op proof: expected Optimal with live deadline, got {other:?}"),
        }

        // ── sentinel (expired deadline) must return Timeout immediately ─────
        let expired = std::time::Instant::now() - std::time::Duration::from_millis(1);
        let opts_expired = SolverOptions { deadline: Some(expired), ..SolverOptions::default() };
        let (outcome, state_out) = iterate(
            state_fresh(), &bsf, &bsf.a, &bsf.c, &opts_expired, &bsf.upper_bounds,
            &mut DualSteepestEdgeLeaving::new(bsf.m),
        );
        match outcome {
            BoundedOutcome::Timeout(obj) => {
                let exp = bounded_obj(
                    &bsf.c, &state_out.basis, &state_out.x_b,
                    &state_out.at_upper, &state_out.is_basic, &bsf.upper_bounds,
                );
                assert!(
                    (obj - exp).abs() < EPS,
                    "Timeout obj={obj:.6e} ≠ bounded_obj={exp:.6e}; delta={:.3e}",
                    (obj - exp).abs()
                );
                assert_eq!(
                    state_out.iterations, 0,
                    "expected 0 iterations (early-exit before loop), got {}",
                    state_out.iterations
                );
            }
            other => panic!("expected Timeout with expired deadline, got {other:?}"),
        }
    }
}
