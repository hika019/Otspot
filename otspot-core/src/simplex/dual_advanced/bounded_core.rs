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

use crate::linalg::timeout::deadline_reached;
use crate::basis::{BasisManager, LuBasis};
use crate::error::SolverError;
use crate::options::SolverOptions;
use crate::problem::LpProblem;
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::{feas_rel_tol, PIVOT_TOL};
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::super::dual_common::{
    basic_obj, compute_dual_vars_into, made_progress_with_floor, recompute_gamma_truth,
    NO_PROGRESS_MIN, NO_PROGRESS_TRIGGER_FACTOR,
};
use super::super::pricing::{DualLeavingStrategy, CAP_MULT_OF_M, GAMMA_FLOOR};
use super::super::standard_form::{BoundedStandardForm, SimplexOutcome};
use super::super::trace::IterTrace;
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
    UbViolationOutOfScope {
        row: usize,
    },
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

    /// Test-only hook: when `true`, `primal_simplex_aug` starts in Bland mode
    /// (smallest-index entering / strict-min-ratio leaving) from iteration 0.
    /// Sentinels flip this to prove that the anti-cycling rule reaches the *same*
    /// optimum as Devex pricing — i.e. Bland breaks degenerate stalls without
    /// changing the LP solution. Lives behind `#[cfg(test)]` (absent in release).
    static FORCE_BLAND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn set_flip_apply_disabled(v: bool) {
    FLIP_APPLY_DISABLE.with(|c| c.set(v));
}

#[cfg(test)]
fn flip_apply_disabled() -> bool {
    FLIP_APPLY_DISABLE.with(|c| c.get())
}

#[cfg(test)]
pub(crate) fn set_primal_force_bland(v: bool) {
    FORCE_BLAND.with(|c| c.set(v));
}

#[cfg(test)]
fn primal_force_bland() -> bool {
    FORCE_BLAND.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn primal_force_bland() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn flip_apply_disabled() -> bool {
    false
}

/// Deadline check interval for the RC inner loop.
///
/// `Instant::now()` 実測 48ns/call。per-col 呼び出しは大規模問題で支配的:
///
/// - pds-80 (n=426k): 426k × 48ns ≈ 20ms/iter (RC pass 全コストを clock read が占有)
/// - dfl001 (n=12k): 12k × 48ns ≈ 0.6ms/iter (RC コストの ~33%)
///
/// chunk=512 で per-RC-pass の check 数を `ceil(n/512)` に削減:
///
/// - pds-80: 833 checks × 48ns ≈ 0.04ms (オーバーヘッド無視可)
///
/// deadline 超過は最大 INTERVAL=512 列分 (~100µs @ 観測 throughput) で
/// ソルバ全体の deadline (秒〜分単位) 内で許容範囲。
///
/// 実測 speedup (timeout=60s bench):
///
/// - pds-80   bounded-aug:  64 → 144 iter/s (2.25x)
/// - pds-30   bounded-aug: 177 → 371 iter/s (~2.1x)
/// - ken-18   bounded-aug: 236 → 355 iter/s (~1.5x)
/// - dfl001   bounded-aug: 556 → 625 iter/s (1.12x)
const DEADLINE_CHECK_INTERVAL: usize = 512;

/// Cyclic partial-pricing window size for the bounded primal cores
/// (`phase2_primal_bounded` / `primal_simplex_aug`).
///
/// Each pricing pass reduces-cost-prices at most this many non-basic columns,
/// starting from a rotating cursor (`state.price_start`), instead of all
/// `n_price`. The full scan is *deferred*, never skipped: optimality is only
/// declared after a complete `n_price` sweep finds no improving column (see
/// `partial_price_entering`). On large network LPs (pds/ken/dfl001) the
/// per-iteration reduced-cost scan (`yᵀaⱼ` over every non-basic column)
/// dominates wall time; windowing it cuts the per-iteration price cost while
/// preserving the exact optimality test.
///
/// For `n_price ≤ PARTIAL_PRICE_CHUNK` the window covers every column, so small
/// and medium LPs price fully (identical behaviour to the pre-partial cores).
///
/// Value: placeholder pending bench calibration (dfl001/ken-18 net throughput
/// vs. the 105/109 PASS set). A fixed absolute window gives the largest
/// relative reduction exactly where the scan dominates (large `n_price`), and
/// keeps the entering candidate sample large enough to avoid an iteration-count
/// blow-up. Tune against `timeout=1000, eps=1e-6` before treating as final.
const PARTIAL_PRICE_CHUNK: usize = 2048;

/// Read `OTSPOT_PP_CHUNK` from the environment once; `None` means "not set / invalid".
///
/// Cached via `OnceLock` so repeated calls are a single pointer load.
/// Intended for bench calibration only — production default remains
/// `PARTIAL_PRICE_CHUNK`.
fn env_pp_chunk() -> Option<usize> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Option<usize>> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("OTSPOT_PP_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
    })
}

/// Effective partial-pricing window for a given `n_price`, clamped to
/// `[1, n_price]`.
///
/// Priority (highest first):
/// 1. `#[cfg(test)]` thread-local override (unit-test sentinels).
/// 2. `OTSPOT_PP_CHUNK` environment variable (bench calibration).
/// 3. `PARTIAL_PRICE_CHUNK` named constant (production default).
fn partial_price_chunk(n_price: usize) -> usize {
    let base = {
        #[cfg(test)]
        {
            let o = PARTIAL_PRICE_CHUNK_OVERRIDE.with(|c| c.get());
            if o != 0 {
                o
            } else if let Some(v) = env_pp_chunk() {
                v
            } else {
                PARTIAL_PRICE_CHUNK
            }
        }
        #[cfg(not(test))]
        {
            env_pp_chunk().unwrap_or(PARTIAL_PRICE_CHUNK)
        }
    };
    base.clamp(1, n_price.max(1))
}

// ── partial-pricing test hooks (compiled out of release) ──────────────────────
#[cfg(test)]
thread_local! {
    /// Override for `PARTIAL_PRICE_CHUNK` (0 = use the production constant).
    static PARTIAL_PRICE_CHUNK_OVERRIDE: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    /// When `true`, `partial_price_entering` declares Optimal after the FIRST
    /// window even if the full `n_price` sweep is incomplete — the broken
    /// behaviour the false-optimal no-op proof must catch.
    static PARTIAL_PRICE_SINGLE_WINDOW: std::cell::Cell<bool> = const {
        std::cell::Cell::new(false)
    };
    /// Running count of columns actually reduced-cost-priced across pricing
    /// passes. Sentinels snapshot the delta to assert partial < full.
    static PARTIAL_PRICE_COLS_SCANNED: std::cell::Cell<u64> = const {
        std::cell::Cell::new(0)
    };
}

#[cfg(test)]
pub(crate) fn set_partial_price_chunk_override(v: usize) {
    PARTIAL_PRICE_CHUNK_OVERRIDE.with(|c| c.set(v));
}

#[cfg(test)]
pub(crate) fn set_partial_price_single_window(v: bool) {
    PARTIAL_PRICE_SINGLE_WINDOW.with(|c| c.set(v));
}

#[cfg(test)]
pub(crate) fn reset_partial_price_cols_scanned() {
    PARTIAL_PRICE_COLS_SCANNED.with(|c| c.set(0));
}

#[cfg(test)]
pub(crate) fn partial_price_cols_scanned() -> u64 {
    PARTIAL_PRICE_COLS_SCANNED.with(|c| c.get())
}

#[cfg(test)]
fn partial_price_single_window() -> bool {
    PARTIAL_PRICE_SINGLE_WINDOW.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn partial_price_single_window() -> bool {
    false
}

// Counts deadline checks issued inside `compute_reduced_costs_into_timed` (test-only).
//
// `thread_local!` 化により、並列 test 実行 (`--test-threads 3`) で他 test の
// 同関数呼び出しが counter を汚染するのを防ぐ。sentinel test は自スレッドの
// snapshot 差分だけを見るので false FAIL が起きない。
//
// For a problem with `n_price` columns, the timed RC loop issues
// `ceil(n_price / DEADLINE_CHECK_INTERVAL)` checks — not `n_price` checks.
// Sentinel asserts the count is strictly less than `n_price` on a problem
// where `n_price > DEADLINE_CHECK_INTERVAL`; reverting to per-column checks
// makes the count equal `n_price`, failing the assertion (no-op FAIL).
#[cfg(test)]
thread_local! {
    pub(crate) static RC_DEADLINE_CHECK_COUNT: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

fn compute_reduced_costs_into_timed(
    a: &CscMatrix,
    c: &[f64],
    basis_mgr: &mut LuBasis,
    is_basic: &[bool],
    n_price: usize,
    basis: &[usize],
    y_buf: &mut [f64],
    rc_out: &mut [f64],
    deadline: Option<Instant>,
) -> bool {
    if deadline_reached(deadline) {
        return false;
    }
    compute_dual_vars_into(c, basis_mgr, basis, y_buf);
    compute_reduced_costs_window(a, c, is_basic, y_buf, rc_out, 0, n_price, deadline)
}

/// Reduced costs `r_k = c_k − yᵀa_k` for the column window `[start, end)` only,
/// written into `rc_out[start..end]` (basic columns zeroed). `y` is precomputed
/// (`B^{-T}c_B`); the caller computes it once per pricing pass and reuses it
/// across windows of the same basis. Deadline is checked per
/// `DEADLINE_CHECK_INTERVAL` chunk, not per column. Returns `false` on deadline.
#[allow(clippy::too_many_arguments)]
fn compute_reduced_costs_window(
    a: &CscMatrix,
    c: &[f64],
    is_basic: &[bool],
    y: &[f64],
    rc_out: &mut [f64],
    start: usize,
    end: usize,
    deadline: Option<Instant>,
) -> bool {
    let mut j = start;
    while j < end {
        // Check deadline once per chunk, not per column.
        if deadline_reached(deadline) {
            return false;
        }
        #[cfg(test)]
        RC_DEADLINE_CHECK_COUNT.with(|c| c.set(c.get() + 1));
        let chunk_end = (j + DEADLINE_CHECK_INTERVAL).min(end);
        for k in j..chunk_end {
            if is_basic[k] {
                rc_out[k] = 0.0;
            } else {
                let (rows, vals) = a.get_column(k).unwrap();
                let mut ya = 0.0;
                for (ri, &row) in rows.iter().enumerate() {
                    ya += y[row] * vals[ri];
                }
                rc_out[k] = c[k] - ya;
            }
        }
        j = chunk_end;
    }
    true
}

/// Outcome of a partial-pricing pass.
enum PartialPrice {
    /// `entering` won pricing; `next_start` is where the next pass should begin.
    Entering { entering: usize, next_start: usize },
    /// A full `n_price` sweep found no improving column → Optimal. `next_start`
    /// is reset so a warm re-entry begins fresh.
    Optimal { next_start: usize },
    /// Deadline expired mid-window.
    Deadline,
}

/// Cyclic partial pricing for the bounded primal cores.
///
/// Starting at `price_start`, prices the reduced costs of successive
/// `partial_price_chunk(n_price)`-sized windows (wrapping around `n_price`),
/// scoring each non-basic column via `score`. The first window that contains an
/// improving column wins; among that window's columns the best score is chosen
/// (Dantzig-within-window). `next_start` advances past the chosen window so the
/// next pass continues from fresh territory.
///
/// **False-optimal invariant (load-bearing):** `Optimal` is returned ONLY after
/// a complete `n_price` sweep (every column priced under the *current* basis)
/// produced no improving column. Windowing merely defers that sweep across
/// pivots; an improving column in a not-yet-scanned window can never trigger
/// Optimal. The `partial_price_single_window` test hook breaks exactly this rule
/// (declares Optimal after one window) so the no-op proof can detect a
/// regression that prices only a single window before declaring optimality.
///
/// `score(j, rc_j) -> Option<f64>`: `None` = column `j` is not improving;
/// `Some(s)` = improving with score `s` (larger wins). `rc_j` is freshly priced.
fn partial_price_entering<F>(
    a: &CscMatrix,
    c: &[f64],
    is_basic: &[bool],
    y: &[f64],
    rc_out: &mut [f64],
    n_price: usize,
    price_start: usize,
    deadline: Option<Instant>,
    mut score: F,
) -> PartialPrice
where
    F: FnMut(usize, f64) -> Option<f64>,
{
    if n_price == 0 {
        return PartialPrice::Optimal { next_start: 0 };
    }
    let chunk = partial_price_chunk(n_price);
    let single_window = partial_price_single_window();
    let mut start = price_start % n_price;
    let mut scanned = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    let mut entering: Option<usize> = None;

    while scanned < n_price {
        // Contiguous window: capped by chunk, by columns left in the sweep, and
        // by the distance to the end of the array (wrap handled by resetting
        // `start` to 0 below). Capping by the remaining-sweep count keeps every
        // column priced at most once per sweep when `price_start != 0`.
        let remaining = n_price - scanned;
        let seg_len = chunk.min(remaining).min(n_price - start);
        let end = start + seg_len;

        if !compute_reduced_costs_window(a, c, is_basic, y, rc_out, start, end, deadline) {
            return PartialPrice::Deadline;
        }
        #[cfg(test)]
        PARTIAL_PRICE_COLS_SCANNED.with(|cnt| cnt.set(cnt.get() + seg_len as u64));

        for j in start..end {
            if is_basic[j] {
                continue;
            }
            if let Some(s) = score(j, rc_out[j]) {
                if s > best_score {
                    best_score = s;
                    entering = Some(j);
                }
            }
        }

        scanned += seg_len;
        start = if end >= n_price { 0 } else { end };

        if let Some(entering) = entering {
            return PartialPrice::Entering {
                entering,
                next_start: start,
            };
        }
        // Test-only: stop after the first window to PROVE the full-sweep
        // requirement is load-bearing. `false` (and branch-eliminated) in release.
        if single_window {
            break;
        }
    }

    PartialPrice::Optimal { next_start: start }
}

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
    /// Cyclic partial-pricing cursor for the bounded primal cores: the next
    /// pricing pass begins its window scan here and advances it past the window
    /// that yields an entering column. Carried in the state so it persists
    /// across iterations (and Phase I → Phase II). Unused by the dual loop.
    pub(crate) price_start: usize,
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
            price_start: 0,
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

    let mut basis_mgr =
        match LuBasis::new_timed(a, &state.basis, options.max_etas, options.deadline) {
            Ok(bm) => bm,
            Err(crate::error::SolverError::SingularBasis { .. }) => {
                return (BoundedOutcome::SingularBasis, state);
            }
            Err(_) => {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
        };

    // Early-exit before O(m²) γ init; prevents budget overrun on large warm-start solves.
    if deadline_reached(options.deadline)
        || options
            .cancel_flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed))
    {
        let obj = bounded_obj(
            c,
            &state.basis,
            &state.x_b,
            &state.at_upper,
            &state.is_basic,
            ubs,
        );
        return (BoundedOutcome::Timeout(obj), state);
    }

    let needs_sigma = leaving.needs_sigma();
    if needs_sigma {
        match recompute_gamma_truth(
            &mut basis_mgr,
            m,
            options.deadline,
            options.cancel_flag.as_deref(),
        ) {
            None => {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
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
    if !compute_reduced_costs_into_timed(
        a,
        &c_perturbed,
        &mut basis_mgr,
        &state.is_basic,
        n_total,
        &state.basis,
        &mut buf.y,
        &mut state.reduced_costs,
        options.deadline,
    ) {
        let obj = bounded_obj(
            c,
            &state.basis,
            &state.x_b,
            &state.at_upper,
            &state.is_basic,
            ubs,
        );
        return (BoundedOutcome::Timeout(obj), state);
    }

    // Anti-cycling: track progress; switch to Bland's rule when stalled.
    let k_trigger = (NO_PROGRESS_TRIGGER_FACTOR * m).max(NO_PROGRESS_MIN);
    let mut best_infeas = leaving.progress_metric(&state.x_b, &state.basis);
    let mut iters_since_progress: usize = 0;
    let mut bland_mode = false;
    let mut trace = IterTrace::new("bounded-dual");

    loop {
        state.iterations = state.iterations.saturating_add(1);
        let timed_out = deadline_reached(options.deadline);
        let cancelled = options
            .cancel_flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::Relaxed));
        if timed_out || cancelled {
            let obj = bounded_obj(
                c,
                &state.basis,
                &state.x_b,
                &state.at_upper,
                &state.is_basic,
                ubs,
            );
            return (BoundedOutcome::Timeout(obj), state);
        }

        if let Some(t) = trace.as_mut() {
            let obj = bounded_obj(
                c,
                &state.basis,
                &state.x_b,
                &state.at_upper,
                &state.is_basic,
                ubs,
            );
            t.log(state.iterations, obj, &state.basis, bland_mode);
        }

        // ub-violation scan (separate from lb-violation leaving selection).
        let mut ub_violation_row: Option<usize> = None;
        for i in 0..m {
            if deadline_reached(options.deadline) {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
            let xi = state.x_b[i];
            let ub_i = ubs[state.basis[i]];
            if ub_i.is_finite() && xi > ub_i + options.primal_tol {
                ub_violation_row.get_or_insert(i);
            }
        }
        let leaving_row = if bland_mode {
            leaving.bland_leaving(&state.x_b, options.primal_tol, &state.basis)
        } else {
            leaving.select_leaving(&state.x_b, options.primal_tol, &state.basis)
        };
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
            if deadline_reached(options.deadline) {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
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
            if deadline_reached(options.deadline) {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
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
            debug_assert!(u_k.is_finite(), "BFRT must not return infinite-upper flips");
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
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
            leaving.after_refactor(m);
            if !compute_reduced_costs_into_timed(
                a,
                &c_perturbed,
                &mut basis_mgr,
                &state.is_basic,
                n_total,
                &state.basis,
                &mut buf.y,
                &mut state.reduced_costs,
                options.deadline,
            ) {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
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
            if deadline_reached(options.deadline) {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
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

        // Anti-cycling progress check. If no improvement for k_trigger iters, enter
        // Bland mode (smallest-index leaving rule) so the loop terminates finitely.
        if !bland_mode {
            let current = leaving.progress_metric(&state.x_b, &state.basis);
            if made_progress_with_floor(best_infeas, current, 0.0) {
                best_infeas = current;
                iters_since_progress = 0;
            } else {
                iters_since_progress += 1;
                if iters_since_progress >= k_trigger {
                    bland_mode = true;
                }
            }
        }

        // Refactor + reduced-cost refresh on the LU's request (eta cap).
        if basis_mgr.needs_refactor() {
            basis_mgr.refactor_if_needed_timed(a, &state.basis, options.deadline);
            if basis_mgr.refactor_failed {
                if basis_mgr.singular_basis {
                    return (BoundedOutcome::SingularBasis, state);
                }
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
            leaving.after_refactor(m);
            if !compute_reduced_costs_into_timed(
                a,
                &c_perturbed,
                &mut basis_mgr,
                &state.is_basic,
                n_total,
                &state.basis,
                &mut buf.y,
                &mut state.reduced_costs,
                options.deadline,
            ) {
                let obj = bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                );
                return (BoundedOutcome::Timeout(obj), state);
            }
        }
    }
}

/// FTRAN a column of `a` and dump into `out` (length `m`). Wraps the
/// `SparseVec` boilerplate that every FTRAN site repeats.
fn ftran_column(a: &CscMatrix, basis_mgr: &mut LuBasis, col: usize, m: usize, out: &mut [f64]) {
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

// ── test-only hook: force zero alpha_sv in primal simplex LU update ───────────

#[cfg(test)]
thread_local! {
    /// When true, the primal simplex LU update replaces `SparseVec::from_dense(&alpha)`
    /// with an empty sparse vector. This makes `add_eta_sparse` see pivot_element = 0
    /// → inv_pivot = ∞, corrupting subsequent FTRAN/BTRAN operations. Used by the
    /// no-op proof sentinels to confirm the dense-to-sparse conversion is load-bearing.
    /// Release builds see a const-false inlined by the optimizer.
    static PRIMAL_ALPHA_SV_DISABLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn set_primal_alpha_sv_disabled(v: bool) {
    PRIMAL_ALPHA_SV_DISABLE.with(|c| c.set(v));
}

#[cfg(test)]
fn primal_alpha_sv_disabled() -> bool {
    PRIMAL_ALPHA_SV_DISABLE.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn primal_alpha_sv_disabled() -> bool {
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
    project_reduced_costs_to_active_bounds(problem, solution, &mut reduced_costs);

    (dual_solution, reduced_costs, slack)
}

fn active_at_bound(x: f64, bound: f64) -> bool {
    bound.is_finite() && (x - bound).abs() <= feas_rel_tol() * (1.0 + x.abs() + bound.abs())
}

fn project_reduced_costs_to_active_bounds(
    problem: &LpProblem,
    solution: &[f64],
    reduced_costs: &mut [f64],
) {
    for (j, rc) in reduced_costs.iter_mut().enumerate().take(problem.num_vars) {
        if j >= solution.len() {
            continue;
        }
        let (lb, ub) = problem.bounds[j];
        let at_lb = active_at_bound(solution[j], lb);
        let at_ub = active_at_bound(solution[j], ub);
        *rc = if at_lb && !at_ub {
            rc.max(0.0)
        } else if at_ub && !at_lb {
            rc.min(0.0)
        } else if at_lb && at_ub {
            *rc
        } else {
            0.0
        };
    }
}

// ── bounded primal Phase 2 ─────────────────────────────────────────────────

/// Objective including non-basic at-upper-bound contributions.
///
/// Full objective: c_B^T x_B + Σ_{j non-basic at ub} c_j · u_j.
/// Invariant: `at_upper[j] ⇒ !is_basic[j]`, maintained by `iterate` /
/// `phase2_primal_bounded`. `debug_assert` traps violations in debug/test
/// builds; release builds rely on callers maintaining the invariant.
fn bounded_obj(
    c: &[f64],
    basis: &[usize],
    x_b: &[f64],
    at_upper: &[bool],
    is_basic: &[bool],
    ubs: &[f64],
) -> f64 {
    debug_assert_eq!(
        at_upper.len(),
        is_basic.len(),
        "at_upper/is_basic length mismatch"
    );
    debug_assert_eq!(at_upper.len(), c.len(), "at_upper/c length mismatch");
    debug_assert_eq!(at_upper.len(), ubs.len(), "at_upper/ubs length mismatch");
    debug_assert_eq!(basis.len(), x_b.len(), "basis/x_b length mismatch");
    let basic: f64 = basis.iter().zip(x_b.iter()).map(|(&j, &v)| c[j] * v).sum();
    let at_ub: f64 = at_upper
        .iter()
        .enumerate()
        .filter(|&(_, &flag)| flag)
        .inspect(|&(j, _)| {
            debug_assert!(
                !is_basic[j],
                "invariant at_upper[j] => !is_basic[j] violated at j={j}"
            )
        })
        .map(|(j, _)| c[j] * ubs[j])
        .sum();
    basic + at_ub
}

/// Outcome of the bounded (two-sided) ratio test.
#[cfg_attr(test, derive(Debug))]
enum BoundedLeave {
    /// Entering variable reaches its own opposite bound before any basic
    /// variable; flip it without a basis change (step = `ub_q`).
    Flip,
    /// Basic variable in `row` leaves at its lower (`at_ub = false`) or upper
    /// (`at_ub = true`) bound; `step` is the primal step length.
    Pivot { row: usize, at_ub: bool, step: f64 },
    /// No basic variable blocks the step and the entering bound is infinite.
    Unbounded,
}

/// Running best leaving candidate for the two-sided Harris ratio test: largest
/// pivot `|eff|`, ties (within `PIVOT_TOL`) broken by Bland's rule (smallest
/// basic index). Tracks the bound side and step so the chosen row carries them.
#[derive(Default)]
struct LeaveCand {
    row: Option<usize>,
    at_ub: bool,
    step: f64,
    best_pivot_abs: f64,
}

impl LeaveCand {
    fn relax(&mut self, i: usize, pivot_abs: f64, at_ub: bool, step: f64, basis: &[usize]) {
        if pivot_abs > self.best_pivot_abs + PIVOT_TOL {
            self.best_pivot_abs = pivot_abs;
            self.row = Some(i);
            self.at_ub = at_ub;
            self.step = step;
        } else if (pivot_abs - self.best_pivot_abs).abs() <= PIVOT_TOL {
            match self.row {
                None => {
                    self.row = Some(i);
                    self.at_ub = at_ub;
                    self.step = step;
                }
                Some(prev) if basis[i] < basis[prev] => {
                    self.row = Some(i);
                    self.at_ub = at_ub;
                    self.step = step;
                }
                _ => {}
            }
        }
    }
}

/// Two-sided Harris ratio test for the bounded primal cores.
///
/// `eff[i] = alpha[i] · dir` is the effective pivot column. A basic variable
/// leaves at its lower bound when `eff[i] > floor` (decreasing toward 0) or at
/// its upper bound when `eff[i] < -floor` with finite `ub_i` (increasing toward
/// `ub_i`). The entering variable instead flips at its own bound `ub_q`.
///
/// Pass 1 computes the feasibility-preserving step
/// `θ = min_i (room_i + feas_tol) / |eff_i|` (capped by `ub_q`); pass 2 selects,
/// among rows whose *true* ratio is within `θ`, the largest pivot `|eff_i|`,
/// breaking ties by Bland's rule. Choosing the largest pivot — rather than the
/// strict-min-ratio row — keeps the basis well-conditioned under degeneracy,
/// where many rows share a zero ratio and a strict-min rule would repeatedly
/// pick near-zero pivots until the LU factorization turns singular. This mirrors
/// the one-sided `primal::ratio_test::select_leaving_feasibility_preserving`.
///
/// Phase I artificial preference: when `art_threshold = Some(t)` and the tie-band
/// (rows with true ratio ≤ θ) contains an artificial basic variable
/// (`basis[i] >= t`), the leaving row is taken among the artificials only. This
/// is the standard HiGHS/GLPK Phase I min-ratio preference — on a degenerate
/// vertex many structural rows share the tie and would otherwise leave the
/// artificials stranded at tiny-step pivots (dfl001 slow tail). θ is unchanged,
/// so the step stays feasibility-preserving; only the choice among equally
/// eligible rows differs.
fn select_leaving_bounded(
    alpha: &[f64],
    dir: f64,
    x_b: &[f64],
    basis: &[usize],
    ubs: &[f64],
    ub_q: f64,
    m: usize,
    floor: f64,
    feas_tol: f64,
    art_threshold: Option<usize>,
) -> BoundedLeave {
    let mut theta = f64::INFINITY;
    let mut min_true = f64::INFINITY;
    for i in 0..m {
        let eff = alpha[i] * dir;
        let xi = x_b[i];
        let ub_i = ubs[basis[i]];
        if eff > floor {
            theta = theta.min((xi + feas_tol) / eff);
            min_true = min_true.min(xi / eff);
        } else if eff < -floor && ub_i.is_finite() {
            let neg = -eff;
            theta = theta.min((ub_i - xi + feas_tol) / neg);
            min_true = min_true.min((ub_i - xi) / neg);
        }
    }

    // Entering bound binds strictly first → flip (preserves "pivot on ties",
    // never flips past a degenerate blocking row whose true ratio is 0).
    if ub_q.is_finite() && ub_q < min_true {
        return BoundedLeave::Flip;
    }
    // Never step past the entering variable's own bound.
    if ub_q.is_finite() {
        theta = theta.min(ub_q);
    }
    if !theta.is_finite() {
        return BoundedLeave::Unbounded;
    }

    // Pass 2: among rows with true ratio ≤ θ, take the largest pivot (Bland
    // tie-break). `best_art` tracks the same over artificial rows only; when an
    // artificial sits in the tie-band it is preferred (Phase I, see above).
    let mut best = LeaveCand::default();
    let mut best_art = LeaveCand::default();
    for i in 0..m {
        let eff = alpha[i] * dir;
        let xi = x_b[i];
        let ub_i = ubs[basis[i]];
        let (true_ratio, at_ub, pivot_abs) = if eff > floor {
            (xi / eff, false, eff)
        } else if eff < -floor && ub_i.is_finite() {
            ((ub_i - xi) / (-eff), true, -eff)
        } else {
            continue;
        };
        if true_ratio <= theta {
            let step = true_ratio.max(0.0);
            best.relax(i, pivot_abs, at_ub, step, basis);
            if art_threshold.is_some_and(|t| basis[i] >= t) {
                best_art.relax(i, pivot_abs, at_ub, step, basis);
            }
        }
    }

    let chosen = if best_art.row.is_some() { best_art } else { best };
    match chosen.row {
        Some(row) => BoundedLeave::Pivot {
            row,
            at_ub: chosen.at_ub,
            step: chosen.step,
        },
        None => BoundedLeave::Unbounded,
    }
}

/// Practical Bland leaving: minimum-ratio within a `PIVOT_TOL` tolerance band,
/// ties broken by smallest basic-variable index.
///
/// Used by `primal_simplex_aug` once a degenerate stall triggers anti-cycling.
/// Unlike `select_leaving_bounded` (largest-pivot Harris, chosen for LU
/// conditioning), this selects the smallest-basis-index row among those whose
/// ratio lies in `[min_ratio, min_ratio + PIVOT_TOL]`. Paired with Bland
/// entering (smallest improving column index) it breaks degenerate cycling in
/// practice.
///
/// The `PIVOT_TOL` band deviates from strict Bland: exact Bland finiteness
/// requires the strict minimum ratio; the band can admit additional candidates
/// beyond that strict minimum, so the theoretical finiteness guarantee is
/// weakened in proportion to `PIVOT_TOL`. Conditioning is sacrificed
/// deliberately; Bland mode is a transient escape, not the steady-state pricing.
fn select_leaving_bland_bounded(
    alpha: &[f64],
    dir: f64,
    x_b: &[f64],
    basis: &[usize],
    ubs: &[f64],
    ub_q: f64,
    m: usize,
    floor: f64,
) -> BoundedLeave {
    let mut min_ratio = f64::INFINITY;
    for i in 0..m {
        let eff = alpha[i] * dir;
        let xi = x_b[i];
        let ub_i = ubs[basis[i]];
        if eff > floor {
            min_ratio = min_ratio.min(xi / eff);
        } else if eff < -floor && ub_i.is_finite() {
            min_ratio = min_ratio.min((ub_i - xi) / (-eff));
        }
    }

    if ub_q.is_finite() && ub_q < min_ratio {
        return BoundedLeave::Flip;
    }
    if ub_q.is_finite() {
        min_ratio = min_ratio.min(ub_q);
    }
    if !min_ratio.is_finite() {
        return BoundedLeave::Unbounded;
    }

    // Among rows achieving the minimum ratio (within PIVOT_TOL), Bland selects
    // the smallest basic-variable index — never the largest pivot.
    let mut leaving: Option<usize> = None;
    let mut leaving_at_ub = false;
    let mut chosen_step = 0.0f64;
    for i in 0..m {
        let eff = alpha[i] * dir;
        let xi = x_b[i];
        let ub_i = ubs[basis[i]];
        let (true_ratio, at_ub) = if eff > floor {
            (xi / eff, false)
        } else if eff < -floor && ub_i.is_finite() {
            ((ub_i - xi) / (-eff), true)
        } else {
            continue;
        };
        if true_ratio <= min_ratio + PIVOT_TOL {
            match leaving {
                None => {
                    leaving = Some(i);
                    leaving_at_ub = at_ub;
                    chosen_step = true_ratio.max(0.0);
                }
                Some(prev) if basis[i] < basis[prev] => {
                    leaving = Some(i);
                    leaving_at_ub = at_ub;
                    chosen_step = true_ratio.max(0.0);
                }
                _ => {}
            }
        }
    }

    match leaving {
        Some(row) => BoundedLeave::Pivot {
            row,
            at_ub: leaving_at_ub,
            step: chosen_step,
        },
        None => BoundedLeave::Unbounded,
    }
}

/// Bland entering for `primal_simplex_aug`: the smallest structural-column
/// index whose reduced cost is improving. Scanning from index 0 (rather than the
/// Devex / partial-pricing order) is what gives Bland its anti-cycling guarantee.
/// Reduced cost is recomputed directly from the current duals `y`; artificials
/// `[n_struct, n_aug)` are never priced.
fn bland_entering(
    a: &CscMatrix,
    c: &[f64],
    is_basic: &[bool],
    at_upper: &[bool],
    y: &[f64],
    n_struct: usize,
    floor: f64,
) -> Option<usize> {
    for j in 0..n_struct {
        if is_basic[j] {
            continue;
        }
        let (rows, vals) = a.get_column(j).unwrap();
        let mut rc = c[j];
        for (k, &row) in rows.iter().enumerate() {
            rc -= vals[k] * y[row];
        }
        let violation = if at_upper[j] { rc } else { -rc };
        if violation > floor {
            return Some(j);
        }
    }
    None
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

    let timeout_obj = |state: &BoundedDualState| {
        SimplexOutcome::Timeout(bounded_obj(
            c,
            &state.basis,
            &state.x_b,
            &state.at_upper,
            &state.is_basic,
            ubs,
        ))
    };
    if deadline_reached(options.deadline) {
        return (timeout_obj(&state), state);
    }

    let mut basis_mgr =
        match LuBasis::new_timed(a, &state.basis, options.max_etas, options.deadline) {
            Ok(bm) => bm,
            Err(SolverError::DeadlineExceeded) => return (timeout_obj(&state), state),
            Err(_) => return (SimplexOutcome::SingularBasis, state),
        };

    let mut y = vec![0.0f64; m];
    let mut rc = vec![0.0f64; n_total];
    let mut alpha = vec![0.0f64; m];
    let mut trace = IterTrace::new("bounded-primal");

    loop {
        *iters = iters.saturating_add(1);
        if deadline_reached(options.deadline) {
            return (
                SimplexOutcome::Timeout(bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                )),
                state,
            );
        }

        if let Some(t) = trace.as_mut() {
            let obj = bounded_obj(
                c,
                &state.basis,
                &state.x_b,
                &state.at_upper,
                &state.is_basic,
                ubs,
            );
            t.log(*iters, obj, &state.basis, false);
        }

        // Dual vars (reused across partial-pricing windows of this basis).
        if deadline_reached(options.deadline) {
            return (timeout_obj(&state), state);
        }
        compute_dual_vars_into(c, &mut basis_mgr, &state.basis, &mut y);

        // Pricing: cyclic partial pricing, most-improving within the first
        // improving window (Dantzig rule).
        // at_lb: enter if rc < 0 (increasing x improves min objective)
        // at_ub: enter if rc > 0 (decreasing x improves min objective)
        let q = {
            let at_upper = &state.at_upper;
            match partial_price_entering(
                a,
                c,
                &state.is_basic,
                &y,
                &mut rc,
                n_total,
                state.price_start,
                options.deadline,
                |j, rc_j| {
                    let score = if at_upper[j] { rc_j } else { -rc_j };
                    (score > PIVOT_TOL).then_some(score)
                },
            ) {
                PartialPrice::Deadline => return (timeout_obj(&state), state),
                PartialPrice::Optimal { next_start } => {
                    state.price_start = next_start;
                    let obj = bounded_obj(
                        c,
                        &state.basis,
                        &state.x_b,
                        &state.at_upper,
                        &state.is_basic,
                        ubs,
                    );
                    return (SimplexOutcome::Optimal(obj, y), state);
                }
                PartialPrice::Entering {
                    entering,
                    next_start,
                } => {
                    state.price_start = next_start;
                    entering
                }
            }
        };

        let from_ub = state.at_upper[q];
        // dir = +1: entering from lb (x_q increases); dir = -1: from ub (decreases).
        let dir = if from_ub { -1.0f64 } else { 1.0 };

        ftran_column(a, &mut basis_mgr, q, m, &mut alpha);

        if deadline_reached(options.deadline) {
            return (
                SimplexOutcome::Timeout(bounded_obj(
                    c,
                    &state.basis,
                    &state.x_b,
                    &state.at_upper,
                    &state.is_basic,
                    ubs,
                )),
                state,
            );
        }

        // Two-sided Harris ratio test (largest pivot within the feasibility
        // tolerance) — strict-min-ratio selection would pick near-zero pivots
        // under degeneracy and drive the LU basis singular.
        let ub_q = ubs[q];
        let (r, leaving_at_ub, theta) = match select_leaving_bounded(
            &alpha,
            dir,
            &state.x_b,
            &state.basis,
            ubs,
            ub_q,
            m,
            PIVOT_TOL,
            options.primal_tol,
            None,
        ) {
            BoundedLeave::Flip => {
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
                        (
                            SimplexOutcome::Timeout(bounded_obj(
                                c,
                                &state.basis,
                                &state.x_b,
                                &state.at_upper,
                                &state.is_basic,
                                ubs,
                            )),
                            state,
                        )
                    };
                }
                continue;
            }
            BoundedLeave::Unbounded => return (SimplexOutcome::Unbounded, state),
            BoundedLeave::Pivot { row, at_ub, step } => (row, at_ub, step),
        };

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

        // Eta update: reuse the already-computed B^{-1} a_q (dense `alpha`) —
        // sparsifying avoids a second redundant FTRAN on the original column.
        let alpha_sv = if primal_alpha_sv_disabled() {
            SparseVec { indices: vec![], values: vec![], len: m }
        } else {
            SparseVec::from_dense(&alpha)
        };
        basis_mgr.update(q, r, &alpha_sv);

        if basis_mgr.needs_refactor() {
            basis_mgr.refactor_if_needed_timed(a, &state.basis, options.deadline);
            if basis_mgr.refactor_failed {
                return if basis_mgr.singular_basis {
                    (SimplexOutcome::SingularBasis, state)
                } else {
                    (
                        SimplexOutcome::Timeout(bounded_obj(
                            c,
                            &state.basis,
                            &state.x_b,
                            &state.at_upper,
                            &state.is_basic,
                            ubs,
                        )),
                        state,
                    )
                };
            }
        }
    }
}

// ── augmented bounded primal (Eq + UB Phase I / Phase II) ─────────────────────

// Eq+UB dispatch counter — sentinel tests only.
#[cfg(test)]
thread_local! {
    static EQ_UB_DISPATCH_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_eq_ub_dispatch_count() {
    EQ_UB_DISPATCH_COUNT.with(|c| c.set(0));
}

#[cfg(test)]
pub(crate) fn eq_ub_dispatch_count() -> u64 {
    EQ_UB_DISPATCH_COUNT.with(|c| c.get())
}

#[cfg(test)]
pub(super) fn bump_eq_ub_dispatch_count() {
    EQ_UB_DISPATCH_COUNT.with(|c| c.set(c.get().saturating_add(1)));
}

#[cfg(not(test))]
#[inline(always)]
pub(super) fn bump_eq_ub_dispatch_count() {}

/// Bounded primal simplex on the augmented matrix `[bsf.a | I_art]` with
/// pricing restricted to the first `n_struct` columns (structural + slacks).
/// Artificial columns at `[n_struct, n_aug)` may be basic but never priced —
/// Phase I drives them out via leaving pivots; Phase II keeps any remaining
/// at value 0 (the Phase I optimality check rejects nonzero residuals).
///
/// State sizing:
/// - `state.basis.len() == bsf.m`, `state.x_b.len() == bsf.m`.
/// - `state.at_upper.len() == state.is_basic.len() == n_aug` (caller pre-sized).
///
/// Cost `c_aug` has length `n_aug`. Phase I passes `[0; n_struct] ++ [1; n_art]`;
/// Phase II passes `[c_scaled; n_struct] ++ [0; n_art]`.
///
/// Returns the primal `SimplexOutcome` (`Optimal(obj, y)` etc.). The objective
/// includes the at-upper contribution from `ubs_aug` and is computed against
/// `c_aug` as supplied (Phase I: artificial sum; Phase II: original objective).
#[allow(clippy::too_many_arguments)]
pub(crate) fn bounded_primal_phase1(
    a_aug: &CscMatrix,
    c_aug: &[f64],
    ubs_aug: &[f64],
    n_struct: usize,
    state: &mut BoundedDualState,
    options: &SolverOptions,
    iters: &mut usize,
) -> SimplexOutcome {
    // Phase I minimizes Σartificial: prefer driving artificials out of the basis
    // (artificials live at columns `[n_struct, n_aug)`).
    primal_simplex_aug(
        a_aug,
        c_aug,
        ubs_aug,
        n_struct,
        state,
        options,
        iters,
        Some(n_struct),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bounded_primal_phase2_aug(
    a_aug: &CscMatrix,
    c_aug: &[f64],
    ubs_aug: &[f64],
    n_struct: usize,
    state: &mut BoundedDualState,
    options: &SolverOptions,
    iters: &mut usize,
) -> SimplexOutcome {
    // Phase II optimizes the original objective; no artificial preference.
    primal_simplex_aug(a_aug, c_aug, ubs_aug, n_struct, state, options, iters, None)
}

#[allow(clippy::too_many_arguments)]
fn primal_simplex_aug(
    a_aug: &CscMatrix,
    c_aug: &[f64],
    ubs_aug: &[f64],
    n_struct: usize,
    state: &mut BoundedDualState,
    options: &SolverOptions,
    iters: &mut usize,
    art_threshold: Option<usize>,
) -> SimplexOutcome {
    let m = state.basis.len();
    let n_aug = state.at_upper.len();
    debug_assert_eq!(state.x_b.len(), m);
    debug_assert_eq!(state.is_basic.len(), n_aug);
    debug_assert_eq!(ubs_aug.len(), n_aug);
    debug_assert_eq!(c_aug.len(), n_aug);
    debug_assert!(n_struct <= n_aug);

    let timeout_obj = |st: &BoundedDualState| {
        SimplexOutcome::Timeout(bounded_obj(
            c_aug,
            &st.basis,
            &st.x_b,
            &st.at_upper,
            &st.is_basic,
            ubs_aug,
        ))
    };
    if deadline_reached(options.deadline) {
        return timeout_obj(state);
    }

    let mut basis_mgr =
        match LuBasis::new_timed(a_aug, &state.basis, options.max_etas, options.deadline) {
            Ok(bm) => bm,
            Err(SolverError::DeadlineExceeded) => return timeout_obj(state),
            Err(_) => return SimplexOutcome::SingularBasis,
        };

    let mut y = vec![0.0f64; m];
    let mut rc = vec![0.0f64; n_struct];
    let mut alpha = vec![0.0f64; m];
    // Devex pricing mirrors the standard primal core: scaled reduced-cost
    // scores avoid repeatedly choosing dense/degenerate columns by raw RC alone.
    let mut devex_weights = vec![1.0f64; n_struct];
    let mut trace = IterTrace::new("bounded-aug-primal");

    // Anti-cycling: after `k_trigger` consecutive degenerate (step ≈ 0) pivots
    // — the signature of a cycle — switch to Bland's rule (smallest improving
    // column / smallest min-ratio row), which terminates finitely. Revert to
    // Devex once a productive step resumes so steady-state pricing stays fast.
    let k_trigger = (NO_PROGRESS_TRIGGER_FACTOR * m).max(NO_PROGRESS_MIN);
    let step_zero_threshold = PIVOT_TOL * (m as f64).max(1.0);
    let force_bland = primal_force_bland();
    let mut iters_since_progress: usize = 0;
    let mut bland_mode = force_bland;

    loop {
        *iters = iters.saturating_add(1);
        if deadline_reached(options.deadline)
            || options
                .cancel_flag
                .as_ref()
                .is_some_and(|f| f.load(Ordering::Relaxed))
        {
            return timeout_obj(state);
        }

        if let Some(t) = trace.as_mut() {
            let obj = bounded_obj(
                c_aug,
                &state.basis,
                &state.x_b,
                &state.at_upper,
                &state.is_basic,
                ubs_aug,
            );
            t.log(*iters, obj, &state.basis, bland_mode);
        }

        // Dual vars (reused across partial-pricing windows of this basis).
        if deadline_reached(options.deadline) {
            return timeout_obj(state);
        }
        compute_dual_vars_into(c_aug, &mut basis_mgr, &state.basis, &mut y);

        // Entering selection. Bland mode (anti-cycling) scans from column 0 for
        // the smallest improving index; the steady-state path uses cyclic partial
        // pricing with Devex weighting (most-improving within the first improving
        // window). Artificials at `[n_struct, n_aug)` are never priced in either.
        let q = if bland_mode {
            match bland_entering(
                a_aug,
                c_aug,
                &state.is_basic,
                &state.at_upper,
                &y,
                n_struct,
                PIVOT_TOL,
            ) {
                Some(j) => j,
                None => {
                    let obj = bounded_obj(
                        c_aug,
                        &state.basis,
                        &state.x_b,
                        &state.at_upper,
                        &state.is_basic,
                        ubs_aug,
                    );
                    return SimplexOutcome::Optimal(obj, y);
                }
            }
        } else {
            let at_upper = &state.at_upper;
            match partial_price_entering(
                a_aug,
                c_aug,
                &state.is_basic,
                &y,
                &mut rc,
                n_struct,
                state.price_start,
                options.deadline,
                |j, rc_j| {
                    let violation = if at_upper[j] { rc_j } else { -rc_j };
                    if violation <= PIVOT_TOL {
                        return None;
                    }
                    let gamma = devex_weights[j].max(GAMMA_FLOOR);
                    Some(violation / gamma.sqrt())
                },
            ) {
                PartialPrice::Deadline => return timeout_obj(state),
                PartialPrice::Optimal { next_start } => {
                    state.price_start = next_start;
                    let obj = bounded_obj(
                        c_aug,
                        &state.basis,
                        &state.x_b,
                        &state.at_upper,
                        &state.is_basic,
                        ubs_aug,
                    );
                    return SimplexOutcome::Optimal(obj, y);
                }
                PartialPrice::Entering {
                    entering,
                    next_start,
                } => {
                    state.price_start = next_start;
                    entering
                }
            }
        };

        let from_ub = state.at_upper[q];
        let dir = if from_ub { -1.0f64 } else { 1.0 };

        ftran_column(a_aug, &mut basis_mgr, q, m, &mut alpha);

        // Two-sided Harris ratio test (largest pivot within the feasibility
        // tolerance) — strict-min-ratio selection would pick near-zero pivots
        // under degeneracy and drive the LU basis singular (grow22 Phase II).
        let ub_q = ubs_aug[q];
        let leave = if bland_mode {
            select_leaving_bland_bounded(
                &alpha,
                dir,
                &state.x_b,
                &state.basis,
                ubs_aug,
                ub_q,
                m,
                PIVOT_TOL,
            )
        } else {
            select_leaving_bounded(
                &alpha,
                dir,
                &state.x_b,
                &state.basis,
                ubs_aug,
                ub_q,
                m,
                PIVOT_TOL,
                options.primal_tol,
                art_threshold,
            )
        };
        let (r, leaving_at_ub, theta) = match leave {
            BoundedLeave::Flip => {
                bump_bfrt_flip_invocations();
                if let Some(t) = trace.as_mut() {
                    t.note_flip();
                }
                for i in 0..m {
                    state.x_b[i] -= alpha[i] * dir * ub_q;
                }
                state.at_upper[q] = !from_ub;
                // A flip moves x by a full bound width — genuine progress; reset
                // the degenerate-stall counter and leave Bland mode.
                iters_since_progress = 0;
                if !force_bland {
                    bland_mode = false;
                }
                basis_mgr.refactor_if_needed_timed(a_aug, &state.basis, options.deadline);
                if basis_mgr.refactor_failed {
                    return if basis_mgr.singular_basis {
                        SimplexOutcome::SingularBasis
                    } else {
                        timeout_obj(state)
                    };
                }
                continue;
            }
            BoundedLeave::Unbounded => return SimplexOutcome::Unbounded,
            BoundedLeave::Pivot { row, at_ub, step } => (row, at_ub, step),
        };
        if let Some(t) = trace.as_mut() {
            t.note_pivot(theta, options.primal_tol);
        }

        // Anti-cycling bookkeeping. A near-zero step makes no progress; once
        // `k_trigger` such steps accumulate, switch to Bland to break a possible
        // cycle. A productive step resets the counter and reverts to Devex.
        if theta > step_zero_threshold {
            iters_since_progress = 0;
            if !force_bland {
                bland_mode = false;
            }
        } else {
            iters_since_progress = iters_since_progress.saturating_add(1);
            if iters_since_progress >= k_trigger {
                bland_mode = true;
            }
        }
        let leaving_col = state.basis[r];

        for i in 0..m {
            state.x_b[i] -= alpha[i] * dir * theta;
        }
        state.x_b[r] = if from_ub { ub_q - theta } else { theta };
        for v in state.x_b.iter_mut() {
            if v.abs() < options.clamp_tol {
                *v = 0.0;
            }
        }

        state.at_upper[leaving_col] = leaving_at_ub;
        state.at_upper[q] = false;
        state.is_basic[leaving_col] = false;
        state.is_basic[q] = true;
        state.basis[r] = q;

        // Eta update: reuse dense `alpha` (already B^{-1} a_q from ftran_column above);
        // sparsifying avoids a second redundant FTRAN on the original column.
        let alpha_sv = if primal_alpha_sv_disabled() {
            SparseVec { indices: vec![], values: vec![], len: m }
        } else {
            SparseVec::from_dense(&alpha)
        };
        let norm_sq: f64 = alpha.iter().map(|&v| v * v).sum();
        let mut gamma_leaving = 1.0;
        if leaving_col < n_struct {
            gamma_leaving = devex_weights[leaving_col];
            let pivot = alpha[r];
            if pivot.abs() > PIVOT_TOL {
                let cap = CAP_MULT_OF_M * (m as f64).max(1.0);
                let new_weight = (norm_sq / (pivot * pivot))
                    .min(cap)
                    .max(GAMMA_FLOOR);
                devex_weights[leaving_col] = devex_weights[leaving_col].max(new_weight);
                gamma_leaving = devex_weights[leaving_col];
            }
        }
        if q < n_struct {
            devex_weights[q] = if norm_sq > GAMMA_FLOOR {
                (gamma_leaving / norm_sq).max(1.0)
            } else {
                1.0
            };
        }
        basis_mgr.update(q, r, &alpha_sv);

        if basis_mgr.needs_refactor() {
            basis_mgr.refactor_if_needed_timed(a_aug, &state.basis, options.deadline);
            if basis_mgr.refactor_failed {
                return if basis_mgr.singular_basis {
                    SimplexOutcome::SingularBasis
                } else {
                    timeout_obj(state)
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
    use crate::simplex::dual_advanced::bound_flip::{
        bfrt_flip_invocations, reset_bfrt_flip_invocations,
    };
    use crate::simplex::pricing::MostInfeasibleLeaving;
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
        let bounds = vec![(0.0, 3.0), (0.0, f64::INFINITY), (0.0, 5.0), (2.0, 2.0)];
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
    fn basis_rhs_residual(state: &BoundedDualState, bsf: &BoundedStandardForm) -> f64 {
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
        let (outcome, state) = solve_bounded_dual(
            &bsf,
            &bsf.a,
            &bsf.b,
            &bsf.c,
            &opts,
            &bsf.upper_bounds,
            &mut MostInfeasibleLeaving,
        );
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
        let (outcome, _state) = iterate(
            state,
            &bsf,
            &bsf.a,
            &bsf.c,
            &opts,
            &bsf.upper_bounds,
            &mut MostInfeasibleLeaving,
        );
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
        let (_outcome, post) = iterate(
            state,
            &bsf,
            &bsf.a,
            &bsf.c,
            &opts,
            &bsf.upper_bounds,
            &mut MostInfeasibleLeaving,
        );
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
        let (outcome, _post) = iterate(
            state,
            &bsf,
            &bsf.a,
            &bsf.c,
            &opts,
            &bsf.upper_bounds,
            &mut MostInfeasibleLeaving,
        );
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

    #[test]
    fn reduced_cost_projection_zeros_inactive_bound_duals() {
        let a = CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[1.0, 1.0, 1.0], 1, 3).unwrap();
        let problem = LpProblem::new_general(
            vec![0.0; 3],
            a,
            vec![10.0],
            vec![ConstraintType::Le],
            vec![(0.0, 100.0), (0.0, 5.0), (0.0, 5.0)],
            None,
        )
        .unwrap();
        let solution = vec![2.0, 0.0, 5.0];
        let mut rc = vec![-1.0e-8, -2.0, 3.0];

        project_reduced_costs_to_active_bounds(&problem, &solution, &mut rc);

        assert_eq!(
            rc[0], 0.0,
            "inactive upper-bound roundoff must not become a positive z_ub"
        );
        assert_eq!(
            rc[1], 0.0,
            "negative lower-bound reduced cost is an active z_lb and must be projected to zero"
        );
        assert_eq!(
            rc[2], 0.0,
            "positive upper-bound reduced cost must not be hidden by z_lb"
        );
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
        )
        .unwrap();
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
        )
        .unwrap();
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
        )
        .unwrap();
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
            assert!(
                bsf.upper_bounds[j].is_finite(),
                "flip target must have finite ub"
            );
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
        let (outcome, state) = solve_bounded_dual(
            &bsf,
            &bsf.a,
            &bsf.b,
            &bsf.c,
            &opts,
            &bsf.upper_bounds,
            &mut MostInfeasibleLeaving,
        );
        assert!(
            matches!(outcome, BoundedOutcome::Optimal(..)),
            "expected Optimal, got {:?}",
            outcome
        );
        // All non-basics must be at lb for the equivalence to hold.
        let any_upper = state
            .at_upper
            .iter()
            .enumerate()
            .any(|(j, &u)| u && !state.is_basic[j]);
        assert!(
            !any_upper,
            "unexpected at_upper set in unbounded-compat fixture"
        );

        let sol_bounded = extract_solution_bounded(&bsf, &state, &[]);
        // For the unscaled standard form, basis and x_b come directly from bsf initial state.
        let sol_std = extract_solution(&sf, &state.basis, &state.x_b, &[]);
        for (i, (&a, &b)) in sol_bounded.iter().zip(sol_std.iter()).enumerate() {
            assert!(
                (a - b).abs() < EPS,
                "bounded[{}]={a:.3e} vs unbounded[{}]={b:.3e}",
                i,
                i
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
            "correct solution[0] should be 1.0, got {}",
            sol_correct[0]
        );
        assert!(
            sol_noop[0].abs() < EPS,
            "noop solution[0] should be 0.0 (correction disabled), got {}",
            sol_noop[0]
        );
        let max_diff = sol_correct
            .iter()
            .zip(sol_noop.iter())
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
        use crate::simplex::standard_form::build_standard_form;
        use crate::simplex::standard_form::extract_dual_info;
        let fx = fixture_unbounded_compat();
        let bsf = build_bounded_standard_form(&fx.problem);
        let sf = build_standard_form(&fx.problem);
        let opts = SolverOptions::default();
        let (outcome, state) = solve_bounded_dual(
            &bsf,
            &bsf.a,
            &bsf.b,
            &bsf.c,
            &opts,
            &bsf.upper_bounds,
            &mut MostInfeasibleLeaving,
        );
        let (_, y_std) = match outcome {
            BoundedOutcome::Optimal(obj, y) => (obj, y),
            other => panic!("expected Optimal, got {:?}", other),
        };
        let solution = extract_solution_bounded(&bsf, &state, &[]);
        let (dual_b, rc_b, slack_b) =
            extract_dual_info_bounded(&bsf, &fx.problem, &y_std, &solution, &[]);
        // Compare with the legacy path on the equivalent standard form.
        let (dual_s, rc_s, slack_s) = extract_dual_info(
            &sf,
            &fx.problem,
            &y_std[..sf.m.min(y_std.len())],
            &solution,
            &[],
        );
        // For a Le-only non-negated problem the row_negated flags are all false;
        // the dual vectors must match entry for entry.
        const EPS: f64 = 1e-8;
        for i in 0..dual_b.len() {
            assert!(
                (dual_b[i] - dual_s[i]).abs() < EPS,
                "dual[{}]: bounded={:.3e}, std={:.3e}",
                i,
                dual_b[i],
                dual_s[i]
            );
        }
        for j in 0..rc_b.len() {
            assert!(
                (rc_b[j] - rc_s[j]).abs() < EPS,
                "rc[{}]: bounded={:.3e}, std={:.3e}",
                j,
                rc_b[j],
                rc_s[j]
            );
        }
        for i in 0..slack_b.len() {
            assert!(
                (slack_b[i] - slack_s[i]).abs() < EPS,
                "slack[{}]: bounded={:.3e}, std={:.3e}",
                i,
                slack_b[i],
                slack_s[i]
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
        let (dual_outcome, dual_state) = solve_bounded_dual(
            &bsf,
            &bsf.a,
            &bsf.b,
            &bsf.c,
            &opts,
            &bsf.upper_bounds,
            &mut MostInfeasibleLeaving,
        );
        assert!(matches!(dual_outcome, BoundedOutcome::Optimal(..)));

        let mut iters = 0usize;
        let (p2_outcome, p2_state) = phase2_primal_bounded(
            &bsf,
            dual_state,
            &bsf.a,
            &bsf.c,
            &opts,
            &mut iters,
            &bsf.upper_bounds,
        );
        assert!(
            matches!(p2_outcome, SimplexOutcome::Optimal(..)),
            "Phase 2 did not reach Optimal: {:?}",
            p2_outcome
        );
        let sol = extract_solution_bounded(&bsf, &p2_state, &[]);
        let obj: f64 = lp.c.iter().zip(sol.iter()).map(|(c, x)| c * x).sum();
        assert!(
            (obj - (-6.0)).abs() < 1e-6,
            "expected obj=-6.0, got {obj:.6e}"
        );
        assert!(
            (sol[0] - 4.0).abs() < 1e-6 && (sol[1] - 2.0).abs() < 1e-6,
            "expected x=(4,2), got ({:.3e},{:.3e})",
            sol[0],
            sol[1]
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
        let (dual_outcome, dual_state) = solve_bounded_dual(
            &bsf,
            &bsf.a,
            &bsf.b,
            &bsf.c,
            &opts,
            &bsf.upper_bounds,
            &mut MostInfeasibleLeaving,
        );
        assert!(matches!(dual_outcome, BoundedOutcome::Optimal(..)));
        let mut iters = 0usize;
        let (p2_outcome, _) = phase2_primal_bounded(
            &bsf,
            dual_state,
            &bsf.a,
            &bsf.c,
            &opts,
            &mut iters,
            &bsf.upper_bounds,
        );
        assert!(
            matches!(p2_outcome, SimplexOutcome::Optimal(..)),
            "expected Optimal, got {:?}",
            p2_outcome
        );
        // c = [1,2] ≥ 0 so c̃ = c; dual already optimal for original costs.
        assert_eq!(
            iters, 1,
            "should terminate after one pricing pass (no improvement)"
        );
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
            (
                "one_row_two_boxed_x1_at_upper",
                build_bounded_standard_form(&p1_problem),
                1usize,
            ),
            (
                "two_rows_three_boxed_x0_at_upper",
                build_bounded_standard_form(&p2_problem),
                0usize,
            ),
        ];

        for (name, bsf, flip_col) in &bsfs {
            let state = state_with_flips(bsf, &[*flip_col]);

            let exp_bounded = bounded_obj(
                &bsf.c,
                &state.basis,
                &state.x_b,
                &state.at_upper,
                &state.is_basic,
                &bsf.upper_bounds,
            );
            let exp_basic = basic_obj(&bsf.c, &state.basis, &state.x_b);

            assert!(
                (exp_bounded - exp_basic).abs() >= MIN_CONTRIBUTION,
                "{name}: fixture degenerate — bounded_obj={exp_bounded:.6e} \
                 basic_obj={exp_basic:.6e} differ by {:.3e} < {MIN_CONTRIBUTION:.1e}",
                (exp_bounded - exp_basic).abs()
            );

            let deadline = std::time::Instant::now() - std::time::Duration::from_millis(1);
            let opts = SolverOptions {
                deadline: Some(deadline),
                ..SolverOptions::default()
            };
            let (outcome, _) = iterate(
                state,
                bsf,
                &bsf.a,
                &bsf.c,
                &opts,
                &bsf.upper_bounds,
                &mut MostInfeasibleLeaving,
            );
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
            &bsf.c,
            &state.basis,
            &state.x_b,
            &state.at_upper,
            &state.is_basic,
            &bsf.upper_bounds,
        );
        let exp_basic = basic_obj(&bsf.c, &state.basis, &state.x_b);

        assert!(
            (exp_bounded - exp_basic).abs() >= MIN_CONTRIBUTION,
            "fixture degenerate — bounded_obj={exp_bounded:.6e} \
             basic_obj={exp_basic:.6e} differ by {:.3e} < {MIN_CONTRIBUTION:.1e}",
            (exp_bounded - exp_basic).abs()
        );

        let deadline = std::time::Instant::now() - std::time::Duration::from_millis(1);
        let opts = SolverOptions {
            deadline: Some(deadline),
            ..SolverOptions::default()
        };
        let mut iters = 0usize;
        let (outcome, _) = phase2_primal_bounded(
            &bsf,
            state,
            &bsf.a,
            &bsf.c,
            &opts,
            &mut iters,
            &bsf.upper_bounds,
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

    /// Sentinel: `bounded_obj` panics in debug mode when a variable is
    /// simultaneously in `at_upper` and `is_basic` (invariant violation).
    ///
    /// No-op proof: removing the `debug_assert!(!is_basic[j], ...)` inside
    /// `bounded_obj` makes this test NOT panic → `#[should_panic]` causes FAIL.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "invariant at_upper")]
    fn bounded_obj_invariant_violation_panics_in_debug() {
        let c = vec![1.0, 2.0];
        let basis = vec![0usize];
        let x_b = vec![0.0];
        let at_upper = vec![false, true]; // var 1 at upper
        let is_basic = vec![true, true]; // var 1 ALSO basic → invariant violation
        let ubs = vec![5.0, 3.0];
        let _ = bounded_obj(&c, &basis, &x_b, &at_upper, &is_basic, &ubs);
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
            state_fresh(),
            &bsf,
            &bsf.a,
            &bsf.c,
            &opts_live,
            &bsf.upper_bounds,
            &mut DualSteepestEdgeLeaving::new(bsf.m),
        );
        match live_outcome {
            BoundedOutcome::Optimal(_, _) => {}
            other => panic!("no-op proof: expected Optimal with live deadline, got {other:?}"),
        }

        // ── sentinel (expired deadline) must return Timeout immediately ─────
        let expired = std::time::Instant::now() - std::time::Duration::from_millis(1);
        let opts_expired = SolverOptions {
            deadline: Some(expired),
            ..SolverOptions::default()
        };
        let (outcome, state_out) = iterate(
            state_fresh(),
            &bsf,
            &bsf.a,
            &bsf.c,
            &opts_expired,
            &bsf.upper_bounds,
            &mut DualSteepestEdgeLeaving::new(bsf.m),
        );
        match outcome {
            BoundedOutcome::Timeout(obj) => {
                let exp = bounded_obj(
                    &bsf.c,
                    &state_out.basis,
                    &state_out.x_b,
                    &state_out.at_upper,
                    &state_out.is_basic,
                    &bsf.upper_bounds,
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

    /// Reference implementation of the OLD strict-min-ratio bounded ratio test
    /// (`step < min_step`, first eligible row wins on a tie). Used only to prove
    /// the sentinel below bites: reverting `select_leaving_bounded` to this rule
    /// reselects the near-zero pivot under degeneracy.
    fn old_strict_min_leaving(
        alpha: &[f64],
        dir: f64,
        x_b: &[f64],
        basis: &[usize],
        ubs: &[f64],
        ub_q: f64,
        m: usize,
    ) -> Option<usize> {
        let mut min_step = f64::INFINITY;
        let mut leaving_row: Option<usize> = None;
        for i in 0..m {
            let eff = alpha[i] * dir;
            let xi = x_b[i];
            let ub_i = ubs[basis[i]];
            if eff > PIVOT_TOL {
                let step = (xi / eff).max(0.0);
                if step < min_step {
                    min_step = step;
                    leaving_row = Some(i);
                }
            } else if eff < -PIVOT_TOL && ub_i.is_finite() {
                let step = ((ub_i - xi) / (-eff)).max(0.0);
                if step < min_step {
                    min_step = step;
                    leaving_row = Some(i);
                }
            }
        }
        if ub_q.is_finite() && ub_q < min_step {
            return None; // flip
        }
        leaving_row
    }

    /// Sentinel (no-op proof) for the grow22 regression: under full degeneracy
    /// (every basic variable at a zero ratio) the bounded ratio test must select
    /// the row with the *largest* pivot for numerical stability. The previous
    /// strict-min-ratio rule kept the first eligible row regardless of pivot
    /// magnitude; on grow22's all-`Eq` Phase II this repeatedly chose pivots of
    /// order 1e-8, accumulating LU error until the basis turned singular and the
    /// solve returned NumericalError.
    ///
    /// Three rows share ratio 0 with pivots `10·PIVOT_TOL` (row 0, just above
    /// the eligibility floor), 50 (row 1), and 1.0 (row 2).
    /// `select_leaving_bounded` must pick row 1 (largest |pivot|). Reverting to
    /// `old_strict_min_leaving` picks row 0 (the tiny pivot), so the
    /// `assert_eq!(row, 1)` below FAILs — proving the fix is load-bearing.
    #[test]
    fn select_leaving_bounded_picks_large_pivot_under_degeneracy() {
        let tiny = PIVOT_TOL * 10.0; // eligible (> floor) but ill-conditioned
        let alpha = [tiny, 50.0, 1.0];
        let dir = 1.0;
        let x_b = [0.0, 0.0, 0.0]; // fully degenerate vertex
        let basis = [0usize, 1, 2];
        let ubs = [f64::INFINITY; 3];
        let ub_q = f64::INFINITY;
        let m = 3;

        match select_leaving_bounded(
            &alpha, dir, &x_b, &basis, &ubs, ub_q, m, PIVOT_TOL, PIVOT_TOL, None,
        ) {
            BoundedLeave::Pivot { row, step, .. } => {
                assert_eq!(row, 1, "must pick the largest pivot (row 1, |α|=50)");
                assert_eq!(step, 0.0, "degenerate vertex ⇒ zero step");
            }
            other => panic!("expected a Pivot, got {other:?}"),
        }

        // No-op proof: the old strict-min rule selects the tiny-pivot row 0.
        let old = old_strict_min_leaving(&alpha, dir, &x_b, &basis, &ubs, ub_q, m);
        assert_eq!(
            old,
            Some(0),
            "old strict-min rule must select the tiny-pivot row 0 (proves the sentinel bites)"
        );
    }

    /// The upper-bound leaving branch must also obey largest-pivot selection.
    /// Two basic variables sit exactly at their upper bound (room 0, ratio 0)
    /// with increasing direction; pivots are 1e-8 (row 0) and 8.0 (row 1).
    #[test]
    fn select_leaving_bounded_picks_large_pivot_at_upper_bound() {
        // dir·alpha < 0 ⇒ basic variable increases toward its ub.
        let alpha = [-(PIVOT_TOL * 10.0), -8.0];
        let dir = 1.0;
        let x_b = [5.0, 5.0]; // already at ub ⇒ room 0 ⇒ degenerate
        let basis = [0usize, 1];
        let ubs = [5.0, 5.0];
        let ub_q = f64::INFINITY;
        let m = 2;

        match select_leaving_bounded(
            &alpha, dir, &x_b, &basis, &ubs, ub_q, m, PIVOT_TOL, PIVOT_TOL, None,
        ) {
            BoundedLeave::Pivot { row, at_ub, .. } => {
                assert_eq!(row, 1, "must pick the largest ub-pivot (row 1, |α|=8)");
                assert!(at_ub, "leaving variable hits its upper bound");
            }
            other => panic!("expected a Pivot, got {other:?}"),
        }
    }

    /// Phase I artificial preference (ON vs OFF) for `select_leaving_bounded`.
    /// The tie-band (ratio ≤ θ) holds a structural row (basis 2, larger pivot)
    /// and an artificial row (basis 5 ≥ threshold, smaller pivot), both at the
    /// degenerate vertex (ratio 0). `art_threshold = Some(5)` drives the
    /// artificial out; `None` keeps the largest-pivot structural choice. θ —
    /// hence the step (0 here) and feasibility — is identical for both.
    #[test]
    fn select_leaving_bounded_phase1_prefers_artificial() {
        let alpha = [4.0, 1.0];
        let dir = 1.0;
        let x_b = [0.0, 0.0];
        let basis = [2usize, 5usize];
        let ubs = [f64::INFINITY; 6];
        let ub_q = f64::INFINITY;
        let m = 2;

        // OFF: largest pivot (structural row 0).
        let off = select_leaving_bounded(
            &alpha, dir, &x_b, &basis, &ubs, ub_q, m, PIVOT_TOL, PIVOT_TOL, None,
        );
        match off {
            BoundedLeave::Pivot { row, .. } => assert_eq!(row, 0, "OFF picks structural row 0"),
            other => panic!("expected Pivot, got {other:?}"),
        }

        // ON: artificial row 1 (basis 5 ≥ 5) leaves first.
        let on = select_leaving_bounded(
            &alpha,
            dir,
            &x_b,
            &basis,
            &ubs,
            ub_q,
            m,
            PIVOT_TOL,
            PIVOT_TOL,
            Some(5),
        );
        match on {
            BoundedLeave::Pivot { row, step, .. } => {
                assert_eq!(row, 1, "ON drives the artificial (row 1) out first");
                assert_eq!(step, 0.0, "degenerate vertex ⇒ zero step (θ unchanged)");
            }
            other => panic!("expected Pivot, got {other:?}"),
        }
    }

    /// No-op proof: an artificial row OUTSIDE the tie-band (ratio 1 ≫ θ) is never
    /// preferred — the preference reorders only within θ and never changes the
    /// min ratio. `Some(threshold)` matches `None` here.
    #[test]
    fn select_leaving_bounded_phase1_artificial_outside_band_noop() {
        let alpha = [1.0, 1.0];
        let dir = 1.0;
        let x_b = [0.0, 1.0]; // row 0 ratio 0 (in band), row 1 artificial ratio 1
        let basis = [2usize, 5usize];
        let ubs = [f64::INFINITY; 6];
        let ub_q = f64::INFINITY;
        let m = 2;

        let off = select_leaving_bounded(
            &alpha, dir, &x_b, &basis, &ubs, ub_q, m, PIVOT_TOL, PIVOT_TOL, None,
        );
        let on = select_leaving_bounded(
            &alpha,
            dir,
            &x_b,
            &basis,
            &ubs,
            ub_q,
            m,
            PIVOT_TOL,
            PIVOT_TOL,
            Some(5),
        );
        let off_row = match off {
            BoundedLeave::Pivot { row, .. } => row,
            other => panic!("expected Pivot, got {other:?}"),
        };
        let on_row = match on {
            BoundedLeave::Pivot { row, .. } => row,
            other => panic!("expected Pivot, got {other:?}"),
        };
        assert_eq!(on_row, 0, "out-of-band artificial must NOT be selected");
        assert_eq!(on_row, off_row, "preference is a no-op outside the tie-band");
    }

    /// Sentinel: `compute_reduced_costs_into_timed` must issue at most
    /// `ceil(n / DEADLINE_CHECK_INTERVAL)` deadline checks, not one per column.
    ///
    /// Build a problem with `n_price > DEADLINE_CHECK_INTERVAL` (here n=4
    /// columns, but we call the function directly with a synthetic problem whose
    /// column count exceeds the interval). We inject `n_synthetic >> INTERVAL`
    /// columns and assert:
    ///   checks < n_synthetic
    /// Reverting to per-column checks makes `checks == n_synthetic`, failing the
    /// assertion (no-op FAIL).
    ///
    /// The test also verifies correctness: the chunked loop must produce the
    /// same reduced costs as the reference scalar loop.
    #[test]
    fn rc_timed_deadline_checks_are_chunked_not_per_column() {
        // Build a synthetic problem with n_synthetic >> DEADLINE_CHECK_INTERVAL columns.
        // We need a real CscMatrix and LuBasis; use a diagonal identity basis for simplicity.
        let n_synthetic = DEADLINE_CHECK_INTERVAL * 4 + 100; // >> DEADLINE_CHECK_INTERVAL
        let m = 3usize;

        // Diagonal m×m identity matrix (first m columns).  Remaining columns are
        // unit vectors re-using the first m columns — ensures no column is empty.
        let mut rows_t: Vec<usize> = Vec::new();
        let mut cols_t: Vec<usize> = Vec::new();
        let mut vals_t: Vec<f64> = Vec::new();
        for j in 0..n_synthetic {
            let row = j % m;
            rows_t.push(row);
            cols_t.push(j);
            vals_t.push(1.0);
        }
        let a = CscMatrix::from_triplets(&rows_t, &cols_t, &vals_t, m, n_synthetic).unwrap();
        let basis: Vec<usize> = (0..m).collect();
        let c: Vec<f64> = (0..n_synthetic).map(|j| j as f64).collect();
        let is_basic: Vec<bool> = (0..n_synthetic).map(|j| j < m).collect();
        let mut y_buf = vec![0.0f64; m];
        let mut rc_out = vec![0.0f64; n_synthetic];

        let opts = SolverOptions {
            max_etas: 50,
            ..SolverOptions::default()
        };
        let mut basis_mgr = LuBasis::new_timed(&a, &basis, opts.max_etas, None).unwrap();

        // Snapshot before the call. thread_local counter なので並列 test 中の
        // 他 test の bounded simplex 呼び出しは現スレッドの値に影響しない。
        let before = RC_DEADLINE_CHECK_COUNT.with(|c| c.get());

        let deadline_far = Some(std::time::Instant::now() + std::time::Duration::from_secs(60));
        let ok = compute_reduced_costs_into_timed(
            &a,
            &c,
            &mut basis_mgr,
            &is_basic,
            n_synthetic,
            &basis,
            &mut y_buf,
            &mut rc_out,
            deadline_far,
        );
        assert!(ok, "RC compute must succeed (deadline is far)");

        let after = RC_DEADLINE_CHECK_COUNT.with(|c| c.get());
        let checks = after - before;
        let max_expected = n_synthetic.div_ceil(DEADLINE_CHECK_INTERVAL);

        assert!(
            checks <= max_expected,
            "chunked RC loop issued {checks} deadline checks for n={n_synthetic}, \
             expected ≤ {max_expected} (= ceil(n/INTERVAL)). \
             Reverting to per-column checks would make this > {max_expected}."
        );
        assert!(
            checks < n_synthetic,
            "must issue far fewer deadline checks than columns: \
             {checks} checks for {n_synthetic} columns. \
             Per-column regression detected."
        );

        // Correctness: verify rc_out matches the reference scalar computation.
        // y = B^{-T} c_B; for diagonal identity basis, y = c_B = c[0..m].
        // rc[j] = c[j] - y^T a_j = c[j] - y[j%m] * 1.0 for non-basic j.
        for j in m..n_synthetic {
            let expected = c[j] - c[j % m];
            assert!(
                (rc_out[j] - expected).abs() < 1e-10,
                "rc[{j}] = {} expected {expected} (correctness check)",
                rc_out[j]
            );
        }
    }

    // ── sentinel: single-FTRAN primal update ─────────────────────────────────

    /// RAII guard for `PRIMAL_ALPHA_SV_DISABLE`. Restores the flag on drop so
    /// a panicking test cannot leak the disabled state to sibling tests.
    struct PrimalAlphaSvGuard;
    impl PrimalAlphaSvGuard {
        fn disabled() -> Self {
            set_primal_alpha_sv_disabled(true);
            Self
        }
    }
    impl Drop for PrimalAlphaSvGuard {
        fn drop(&mut self) {
            set_primal_alpha_sv_disabled(false);
        }
    }

    /// (a) Numerical equivalence: `SparseVec::from_dense(&alpha)` matches the
    /// result of applying `basis_mgr.ftran` directly to the original column data.
    /// This is the invariant that makes the single-FTRAN path correct.
    ///
    /// Algebraic no-op proof embedded: a zero sparse vector does NOT match the
    /// correct FTRAN result — proving that using `from_dense` is non-trivially
    /// different from using an empty vector.
    #[test]
    fn primal_ftran_alpha_sv_numerical_equiv() {
        use crate::basis::BasisManager;

        // 2-row LP: slacks as initial basis. Structural cols 0,1 have
        // non-trivial FTRAN images (not just identity columns).
        let m = 2;
        // Matrix [1 1 1 0; 0.5 1 0 1] (struct + slacks)
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 0, 1],
            &[0, 1, 0, 1, 2, 3],
            &[1.0, 1.0, 0.5, 1.0, 1.0, 1.0],
            m,
            4,
        )
        .unwrap();
        let basis = vec![2usize, 3]; // slack basis
        let opts = SolverOptions { max_etas: 50, ..SolverOptions::default() };
        let mut basis_mgr = LuBasis::new_timed(&a, &basis, opts.max_etas, None).unwrap();

        for col in 0..2usize {
            // Dense alpha via ftran_column (what the modified loop uses).
            let mut alpha = vec![0.0f64; m];
            ftran_column(&a, &mut basis_mgr, col, m, &mut alpha);

            // Sparse alpha via the old code path (raw column → ftran).
            let (cr, cv) = a.get_column(col).unwrap();
            let mut alpha_sv_old = SparseVec {
                indices: cr.to_vec(),
                values: cv.to_vec(),
                len: m,
            };
            basis_mgr.ftran(&mut alpha_sv_old);

            // New code path: from_dense.
            let alpha_sv_new = SparseVec::from_dense(&alpha);

            // Must match within 1e-10 (both compute B^{-1} a_q; rounding is negligible).
            let d_old = alpha_sv_old.to_dense();
            let d_new = alpha_sv_new.to_dense();
            for i in 0..m {
                assert!(
                    (d_old[i] - d_new[i]).abs() < 1e-10,
                    "col {col} row {i}: old_ftran={:.6e} from_dense={:.6e}",
                    d_old[i],
                    d_new[i]
                );
            }

            // Algebraic no-op proof: a zero sparse vector must differ from the
            // correct FTRAN result, proving `from_dense` is non-trivially correct.
            let zero_sv = SparseVec { indices: vec![], values: vec![], len: m };
            let d_zero = zero_sv.to_dense();
            let max_diff: f64 = (0..m)
                .map(|i| (d_old[i] - d_zero[i]).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff > 1e-6,
                "col {col}: no-op proof FAILED — zero sv matches FTRAN result (max_diff={max_diff:.3e}); \
                 choose a column with non-trivial FTRAN image"
            );
        }
    }

    /// (b) Optimality invariant for `phase2_primal_bounded` (Site 2): the LP
    ///     must reach the known optimal with the single-FTRAN path active.
    ///
    /// LP: min -x0-x1, x0+x1≤6, x0-x1≤2, 0≤x0≤4, 0≤x1≤4
    /// Known optimal: x0=4, x1=2, obj=-6.
    #[test]
    fn phase2_primal_bounded_single_ftran_reaches_known_optimal() {
        let lp = lp_boxed_2x2_degenerate();
        let bsf = build_bounded_standard_form(&lp);
        let opts = SolverOptions::default();
        let (dual_outcome, dual_state) = solve_bounded_dual(
            &bsf,
            &bsf.a,
            &bsf.b,
            &bsf.c,
            &opts,
            &bsf.upper_bounds,
            &mut MostInfeasibleLeaving,
        );
        assert!(matches!(dual_outcome, BoundedOutcome::Optimal(..)));

        let mut iters = 0usize;
        let (outcome, p2_state) = phase2_primal_bounded(
            &bsf,
            dual_state,
            &bsf.a,
            &bsf.c,
            &opts,
            &mut iters,
            &bsf.upper_bounds,
        );
        match outcome {
            SimplexOutcome::Optimal(obj, _) => {
                assert!(
                    (obj - (-6.0)).abs() < 1e-6,
                    "expected obj=-6.0 (x0=4,x1=2), got {obj:.6e}"
                );
            }
            other => panic!("expected Optimal, got {other:?}"),
        }
        let sol = extract_solution_bounded(&bsf, &p2_state, &[]);
        assert!(
            (sol[0] - 4.0).abs() < 1e-6 && (sol[1] - 2.0).abs() < 1e-6,
            "expected (x0,x1)=(4,2), got ({:.3e},{:.3e})",
            sol[0],
            sol[1]
        );
        assert!(iters > 0, "phase2 must make at least one iteration");
    }

    /// (c) No-op proof for `phase2_primal_bounded` (Site 2): with
    /// `PRIMAL_ALPHA_SV_DISABLE` forced on, the LU update receives a zero sparse
    /// vector (inv_pivot = ∞), corrupting the eta matrix. The corrupt BTRAN in the
    /// next iteration produces ∞ → NaN reduced costs → wrong pivot selection →
    /// wrong final x_b → wrong objective (-4 instead of -6).
    ///
    /// Sentinel design: if the fix is REVERTED, the hook has no effect and the
    /// solve returns the correct obj = -6 → the assertion below FAILS.
    #[test]
    fn phase2_primal_bounded_single_ftran_noop_proof() {
        let lp = lp_boxed_2x2_degenerate();
        let bsf = build_bounded_standard_form(&lp);
        let opts = SolverOptions::default();
        let (dual_outcome, dual_state) = solve_bounded_dual(
            &bsf,
            &bsf.a,
            &bsf.b,
            &bsf.c,
            &opts,
            &bsf.upper_bounds,
            &mut MostInfeasibleLeaving,
        );
        assert!(matches!(dual_outcome, BoundedOutcome::Optimal(..)));

        let _guard = PrimalAlphaSvGuard::disabled();
        let mut iters = 0usize;
        let (outcome, _) = phase2_primal_bounded(
            &bsf,
            dual_state,
            &bsf.a,
            &bsf.c,
            &opts,
            &mut iters,
            &bsf.upper_bounds,
        );
        let obj = match outcome {
            SimplexOutcome::Optimal(o, _) => o,
            // Non-optimal outcomes also demonstrate that the corrupt eta broke the solve.
            _ => return,
        };
        assert!(
            (obj - (-6.0)).abs() > 1e-3,
            "no-op proof FAILED: corrupt (zero) alpha_sv still yields correct obj={obj:.6e}; \
             the from_dense conversion is NOT load-bearing (fix had no effect on this path)"
        );
    }

    /// (b') Optimality invariant for `bounded_primal_phase1` / `primal_simplex_aug`
    ///     (Site 3): Phase I on a 2-row Eq LP must drive art_sum to zero.
    ///
    /// LP: x0+x1=4, x0+2*x1=5, 0≤x0≤3, 0≤x1≤∞.
    /// Unique solution: x0=3, x1=1. Phase I optimal iff art_sum ≈ 0.
    #[test]
    fn primal_simplex_aug_single_ftran_reaches_feasible() {
        // Augmented matrix: [x0 x1 art0 art1], 2 rows.
        // art0 for row 0 (x0+x1=4), art1 for row 1 (x0+2x1=5).
        let a_aug = CscMatrix::from_triplets(
            &[0, 1, 0, 1, 0, 1],
            &[0, 0, 1, 1, 2, 3],
            &[1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
            2,
            4,
        )
        .unwrap();
        let n_struct = 2usize;
        let ubs_aug = vec![3.0f64, f64::INFINITY, f64::INFINITY, f64::INFINITY];
        let c_p1 = vec![0.0f64, 0.0, 1.0, 1.0]; // minimize art_sum

        let mut state = BoundedDualState {
            basis: vec![2usize, 3],
            at_upper: vec![false; 4],
            x_b: vec![4.0f64, 5.0],
            reduced_costs: vec![0.0; 4],
            is_basic: vec![false, false, true, true],
            iterations: 0,
            price_start: 0,
        };
        let opts = SolverOptions::default();
        let mut iters = 0usize;
        let outcome = bounded_primal_phase1(&a_aug, &c_p1, &ubs_aug, n_struct, &mut state, &opts, &mut iters);

        match outcome {
            SimplexOutcome::Optimal(art_sum, _) => {
                assert!(
                    art_sum.abs() < 1e-6,
                    "Phase I must reach art_sum≈0 (LP feasible); got {art_sum:.6e}"
                );
            }
            other => panic!("expected Phase I Optimal, got {other:?}"),
        }
        assert!(iters > 0, "Phase I must make at least one pivot");
    }

    /// (c') No-op proof for `primal_simplex_aug` (Site 3): with
    /// `PRIMAL_ALPHA_SV_DISABLE` forced on, the corrupt eta causes wrong BTRAN
    /// in the next iteration, producing NaN reduced costs. All NaN violations are
    /// treated as non-entering, so Phase I exits early with a residual artificial
    /// (art_sum > 0 — feasible LP declared infeasible-looking).
    ///
    /// Sentinel design: if the fix is REVERTED, the hook has no effect and Phase I
    /// correctly reaches art_sum ≈ 0 → the assertion below FAILS.
    #[allow(clippy::single_match)]
    #[test]
    fn primal_simplex_aug_single_ftran_noop_proof() {
        let a_aug = CscMatrix::from_triplets(
            &[0, 1, 0, 1, 0, 1],
            &[0, 0, 1, 1, 2, 3],
            &[1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
            2,
            4,
        )
        .unwrap();
        let n_struct = 2usize;
        let ubs_aug = vec![3.0f64, f64::INFINITY, f64::INFINITY, f64::INFINITY];
        let c_p1 = vec![0.0f64, 0.0, 1.0, 1.0];

        let mut state = BoundedDualState {
            basis: vec![2usize, 3],
            at_upper: vec![false; 4],
            x_b: vec![4.0f64, 5.0],
            reduced_costs: vec![0.0; 4],
            is_basic: vec![false, false, true, true],
            iterations: 0,
            price_start: 0,
        };
        let opts = SolverOptions::default();
        let mut iters = 0usize;

        let _guard = PrimalAlphaSvGuard::disabled();
        let outcome = bounded_primal_phase1(&a_aug, &c_p1, &ubs_aug, n_struct, &mut state, &opts, &mut iters);

        match outcome {
            SimplexOutcome::Optimal(art_sum, _) => {
                assert!(
                    art_sum > 1e-6,
                    "no-op proof FAILED: corrupt (zero) alpha_sv still reaches art_sum≈0 ({art_sum:.6e}); \
                     the from_dense conversion is NOT load-bearing (fix had no effect on this path)"
                );
            }
            // Any non-Optimal outcome also demonstrates corruption.
            _ => {}
        }
    }

    // ── anti-degeneracy (Bland) sentinels ────────────────────────────────────

    /// RAII guard for the test-only `FORCE_BLAND` hook (restores on unwind).
    struct ForceBlandGuard;
    impl ForceBlandGuard {
        fn on() -> Self {
            set_primal_force_bland(true);
            Self
        }
    }
    impl Drop for ForceBlandGuard {
        fn drop(&mut self) {
            set_primal_force_bland(false);
        }
    }

    /// Degenerate Eq+UB augmented Phase I instance shared by the Bland sentinels.
    ///
    /// LP: min x0+x1+x2  s.t. x0+x1=1, x1+x2=1, 0≤xi≤1. Unique optimum x=(0,1,0),
    /// obj=1; degenerate because a single basic x1 covers both rows (x0,x2 sit at
    /// 0). Returns `(a_aug, ubs_aug, n_struct)` with artificials in cols [3,4].
    fn degenerate_phase1_aug() -> (CscMatrix, Vec<f64>, usize) {
        // cols: x0,x1,x2,art0,art1.  x1 spans both rows; artificials are identity.
        let a_aug = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 0, 1],
            &[0, 1, 1, 2, 3, 4],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            2,
            5,
        )
        .unwrap();
        let ubs_aug = vec![1.0f64, 1.0, 1.0, f64::INFINITY, f64::INFINITY];
        (a_aug, ubs_aug, 3)
    }

    fn fresh_phase1_state() -> BoundedDualState {
        BoundedDualState {
            basis: vec![3usize, 4],
            at_upper: vec![false; 5],
            x_b: vec![1.0f64, 1.0],
            reduced_costs: vec![0.0; 5],
            is_basic: vec![false, false, false, true, true],
            iterations: 0,
            price_start: 0,
        }
    }

    /// Run both phases of the bounded primal core on `degenerate_phase1_aug` and
    /// return the true objective `c·x` at the optimum.
    fn solve_degenerate_via_core(opts: &SolverOptions) -> f64 {
        let (a_aug, mut ubs_aug, n_struct) = degenerate_phase1_aug();
        let c_p1 = vec![0.0f64, 0.0, 0.0, 1.0, 1.0];
        let mut state = fresh_phase1_state();
        let mut iters = 0usize;
        match bounded_primal_phase1(&a_aug, &c_p1, &ubs_aug, n_struct, &mut state, opts, &mut iters)
        {
            SimplexOutcome::Optimal(art_sum, _) => {
                assert!(art_sum.abs() < 1e-9, "Phase I art_sum {art_sum:.3e} ≠ 0");
            }
            other => panic!("Phase I expected Optimal, got {other:?}"),
        }
        // Pin artificials out for Phase II, then minimise the true objective.
        for col in n_struct..ubs_aug.len() {
            ubs_aug[col] = 0.0;
        }
        let c_p2 = vec![1.0f64, 1.0, 1.0, 0.0, 0.0];
        match bounded_primal_phase2_aug(
            &a_aug, &c_p2, &ubs_aug, n_struct, &mut state, opts, &mut iters,
        ) {
            SimplexOutcome::Optimal(_, _) => {}
            other => panic!("Phase II expected Optimal, got {other:?}"),
        }
        bounded_obj(
            &c_p2,
            &state.basis,
            &state.x_b,
            &state.at_upper,
            &state.is_basic,
            &ubs_aug,
        )
    }

    /// Forcing Bland from iteration 0 reaches the *same* optimum as Devex on the
    /// degenerate LP — anti-cycling must never change the solution. No-op proof:
    /// a Bland rule that stepped past a blocking row (or mis-priced) would land
    /// on a different objective, breaking the `obj_bland == obj_devex == 1`
    /// assertion. Routes through the production core (no presolve short-circuit).
    #[test]
    fn force_bland_reaches_same_optimum_as_devex() {
        let opts = SolverOptions::default();
        let obj_devex = solve_degenerate_via_core(&opts);
        let obj_bland = {
            let _g = ForceBlandGuard::on();
            solve_degenerate_via_core(&opts)
        };
        assert!(
            (obj_devex - 1.0).abs() < 1e-9,
            "Devex optimum must be 1.0, got {obj_devex:.9}"
        );
        assert!(
            (obj_bland - obj_devex).abs() < 1e-9,
            "Bland changed the optimum: devex={obj_devex:.9} bland={obj_bland:.9}"
        );
    }

    /// Bland leaving picks the smallest basic-variable index among min-ratio
    /// ties — *not* the largest pivot. No-op proof: reverting
    /// `select_leaving_bland_bounded` to the largest-pivot rule (==
    /// `select_leaving_bounded`) makes the Bland row equal the Harris row, so the
    /// `bland_row == 1` / `assert_ne!` checks FAIL.
    #[test]
    fn bland_leaving_breaks_ties_by_smallest_index() {
        // Two rows tie at ratio 1.0: row 0 has the larger pivot (2.0), row 1 the
        // smaller basic index (3 < 5).
        let alpha = [2.0f64, 1.0];
        let x_b = [2.0f64, 1.0];
        let basis = [5usize, 3];
        let ubs = vec![f64::INFINITY; 6];
        let inf = f64::INFINITY;

        let bland_row = match select_leaving_bland_bounded(
            &alpha, 1.0, &x_b, &basis, &ubs, inf, 2, PIVOT_TOL,
        ) {
            BoundedLeave::Pivot { row, .. } => row,
            other => panic!("expected Bland Pivot, got {other:?}"),
        };
        let harris_row = match select_leaving_bounded(
            &alpha, 1.0, &x_b, &basis, &ubs, inf, 2, PIVOT_TOL, 1e-9, None,
        ) {
            BoundedLeave::Pivot { row, .. } => row,
            other => panic!("expected Harris Pivot, got {other:?}"),
        };
        assert_eq!(bland_row, 1, "Bland must take the smallest-index row (basis 3)");
        assert_eq!(harris_row, 0, "Harris must take the largest-pivot row (pivot 2.0)");
        assert_ne!(bland_row, harris_row, "the two rules must diverge on this tie");
    }

    /// Bland entering returns the smallest *improving* column index, skipping
    /// non-improving columns — never the most-improving. No-op proof: an
    /// argmax-violation rule would return column 2 (violation 5 > 1), failing the
    /// `Some(0)` / `Some(1)` assertions.
    #[test]
    fn bland_entering_returns_smallest_improving_index() {
        // 1 row, 3 cols, all coeff 1, zero duals ⇒ rc_j = c_j.
        let a = CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[1.0, 1.0, 1.0], 1, 3).unwrap();
        let y = [0.0f64];
        let is_basic = [false, false, false];
        let at_upper = [false, false, false];

        // cols 0 and 2 improving at lb (rc<0), col 2 most-improving.
        let c0 = [-1.0f64, 0.5, -5.0];
        assert_eq!(
            bland_entering(&a, &c0, &is_basic, &at_upper, &y, 3, PIVOT_TOL),
            Some(0),
            "smallest improving index is 0, not the most-improving col 2"
        );

        // col 0 non-improving (rc>0 at lb) ⇒ skip to col 1.
        let c1 = [5.0f64, -1.0, -5.0];
        assert_eq!(
            bland_entering(&a, &c1, &is_basic, &at_upper, &y, 3, PIVOT_TOL),
            Some(1),
            "must skip non-improving col 0 and take col 1"
        );
    }

    // ── partial pricing sentinels ────────────────────────────────────────────

    /// Override `PARTIAL_PRICE_CHUNK` for the lifetime of the guard. Restores the
    /// production constant (override = 0) on drop, even across a panic unwind.
    struct PartialPriceChunkGuard;
    impl PartialPriceChunkGuard {
        fn set(chunk: usize) -> Self {
            set_partial_price_chunk_override(chunk);
            Self
        }
    }
    impl Drop for PartialPriceChunkGuard {
        fn drop(&mut self) {
            set_partial_price_chunk_override(0);
        }
    }

    /// Engage the broken single-window mode (declares Optimal after one window
    /// without the full sweep). Restores the correct behaviour on drop.
    struct PartialPriceSingleWindowGuard;
    impl PartialPriceSingleWindowGuard {
        fn enabled() -> Self {
            set_partial_price_single_window(true);
            Self
        }
    }
    impl Drop for PartialPriceSingleWindowGuard {
        fn drop(&mut self) {
            set_partial_price_single_window(false);
        }
    }

    /// Window override that forces a single full-sweep window on any LP in this
    /// file (`≥ n_price`, clamped by `partial_price_chunk`).
    const FULL_PRICE: usize = 1_000_000;

    /// Boxed LP whose unique optimum places `x1` at its UPPER bound, so Phase 2
    /// must execute a primal bound-flip (`BoundedLeave::Flip`).
    /// min −2x0 − 3x1 ; x0 + x1 ≤ 5 ; 0 ≤ x0 ≤ 2, 0 ≤ x1 ≤ 4.
    /// Optimum: x0 = 1, x1 = 4, obj = −14.
    fn lp2_boxed_flip() -> LpProblem {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        LpProblem::new_general(
            vec![-2.0, -3.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 2.0), (0.0, 4.0)],
            None,
        )
        .unwrap()
    }

    /// Run `lp`'s Phase 2 from its bounded-dual terminal state with the given
    /// partial-pricing chunk override. Returns the objective recomputed from the
    /// recovered solution against `lp.c` (independent of the solver's reported
    /// obj), the iteration count, and the Phase-2 primal bound-flip count.
    fn solve_phase2_obj(lp: &LpProblem, chunk: usize) -> (f64, usize, u64) {
        let bsf = build_bounded_standard_form(lp);
        let opts = SolverOptions::default();
        let (dual_outcome, dual_state) = solve_bounded_dual(
            &bsf,
            &bsf.a,
            &bsf.b,
            &bsf.c,
            &opts,
            &bsf.upper_bounds,
            &mut MostInfeasibleLeaving,
        );
        assert!(matches!(dual_outcome, BoundedOutcome::Optimal(..)));
        let _chunk = PartialPriceChunkGuard::set(chunk);
        reset_bfrt_flip_invocations();
        let mut iters = 0usize;
        let (outcome, state) = phase2_primal_bounded(
            &bsf,
            dual_state,
            &bsf.a,
            &bsf.c,
            &opts,
            &mut iters,
            &bsf.upper_bounds,
        );
        let flips = bfrt_flip_invocations();
        assert!(
            matches!(outcome, SimplexOutcome::Optimal(..)),
            "phase2 not Optimal: {outcome:?}"
        );
        let sol = extract_solution_bounded(&bsf, &state, &[]);
        let obj: f64 = lp.c.iter().zip(sol.iter()).map(|(c, x)| c * x).sum();
        (obj, iters, flips)
    }

    /// **Same-obj sentinel + BFRT-flip preservation.** Partial pricing (1-column
    /// windows) and full pricing must reach the SAME externally-known optimum on
    /// boxed/degenerate LPs, and any fixture whose full-pricing solve executes a
    /// primal bound-flip must still flip under partial pricing (the flip path is
    /// not dropped by windowing the entering scan). Breaking partial pricing into
    /// a single-window false-optimal (or dropping flips) diverges the objective.
    #[test]
    fn partial_pricing_phase2_matches_full_and_preserves_flips() {
        let cases: [(LpProblem, f64); 2] = [
            (lp_boxed_2x2_degenerate(), -6.0),
            (lp2_boxed_flip(), -14.0),
        ];
        let mut any_flip = false;
        for (lp, known) in &cases {
            let (obj_full, _, flips_full) = solve_phase2_obj(lp, FULL_PRICE);
            let (obj_partial, _, flips_partial) = solve_phase2_obj(lp, 1);
            assert!(
                (obj_full - known).abs() < 1e-6,
                "full pricing obj {obj_full:.6e} != known {known:.6e}"
            );
            assert!(
                (obj_partial - known).abs() < 1e-6,
                "partial pricing obj {obj_partial:.6e} != known {known:.6e}"
            );
            assert!(
                (obj_partial - obj_full).abs() < 1e-9,
                "partial {obj_partial:.6e} != full {obj_full:.6e}"
            );
            if flips_full > 0 {
                assert!(
                    flips_partial > 0,
                    "primal bound-flip dropped under partial pricing \
                     (full flips={flips_full}, partial flips={flips_partial})"
                );
                any_flip = true;
            }
        }
        assert!(
            any_flip,
            "no fixture exercised a primal bound-flip — flip coverage is vacuous"
        );
    }

    /// **False-optimal no-op proof (the most important sentinel).** Optimality
    /// must be declared ONLY after a full `n_price` sweep finds no improving
    /// column. At the cold Phase-2 vertex of this LP the first column (x0) is
    /// non-improving while x1 is improving and at_ub at the optimum. With
    /// 1-column windows starting at `price_start = 0`, the full-sweep (correct)
    /// run skips the empty first window and prices x1, reaching the true optimum
    /// −6; the single-window (broken) run prices only the first window, finds
    /// nothing, and WRONGLY declares Optimal at the cold objective 0.
    ///
    /// The broken obj must differ from the known optimum; if the full-sweep
    /// requirement were removed, the correct run would also miss −6 and its
    /// assertion would fail.
    #[test]
    fn partial_pricing_false_optimal_requires_full_sweep_noop_proof() {
        const KNOWN: f64 = -6.0;
        let lp = LpProblem::new_general(
            vec![1.0, -2.0],
            CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap(),
            vec![4.0],
            vec![ConstraintType::Le],
            vec![(0.0, 3.0), (0.0, 3.0)],
            None,
        )
        .unwrap();

        let solve = |single_window: bool| -> f64 {
            let bsf = build_bounded_standard_form(&lp);
            let opts = SolverOptions::default();
            let (_d, dual_state) = solve_bounded_dual(
                &bsf,
                &bsf.a,
                &bsf.b,
                &bsf.c,
                &opts,
                &bsf.upper_bounds,
                &mut MostInfeasibleLeaving,
            );
            let _chunk = PartialPriceChunkGuard::set(1);
            let _sw = single_window.then(PartialPriceSingleWindowGuard::enabled);
            let mut iters = 0usize;
            let (out, state) = phase2_primal_bounded(
                &bsf,
                dual_state,
                &bsf.a,
                &bsf.c,
                &opts,
                &mut iters,
                &bsf.upper_bounds,
            );
            assert!(
                matches!(out, SimplexOutcome::Optimal(..)),
                "expected Optimal, got {out:?}"
            );
            let sol = extract_solution_bounded(&bsf, &state, &[]);
            lp.c.iter().zip(sol.iter()).map(|(c, x)| c * x).sum()
        };

        let correct = solve(false);
        let broken = solve(true);
        assert!(
            (correct - KNOWN).abs() < 1e-6,
            "full-sweep partial pricing must reach {KNOWN}, got {correct:.6e}"
        );
        assert!(
            (broken - KNOWN).abs() > 1e-3,
            "no-op proof FAILED: single-window (no full sweep) still reached the \
             optimum (got {broken:.6e}); the full-sweep optimality requirement is \
             not load-bearing — a false-optimal would slip through"
        );
    }

    /// **Pricing-scan reduction (effectiveness, with no-op anchor).** Full
    /// pricing reduced-cost-prices every column every iteration (exactly
    /// `iters * n_price`). Partial pricing must price strictly fewer — it stops
    /// each non-final iteration at the first improving window. The full-pricing
    /// `== iters*n_price` equality is the no-op anchor: if windowing were
    /// disabled, the partial run would hit the same product and the strict-`<`
    /// assertion would fail.
    #[test]
    fn partial_pricing_reduces_per_iteration_scan() {
        let lp = lp_boxed_2x2_degenerate();
        let bsf = build_bounded_standard_form(&lp);
        let n_price = bsf.n_total;
        let opts = SolverOptions::default();

        let run = |chunk: usize| -> (u64, usize) {
            let (_d, dual_state) = solve_bounded_dual(
                &bsf,
                &bsf.a,
                &bsf.b,
                &bsf.c,
                &opts,
                &bsf.upper_bounds,
                &mut MostInfeasibleLeaving,
            );
            let _g = PartialPriceChunkGuard::set(chunk);
            reset_partial_price_cols_scanned();
            let mut iters = 0usize;
            let (out, _s) = phase2_primal_bounded(
                &bsf,
                dual_state,
                &bsf.a,
                &bsf.c,
                &opts,
                &mut iters,
                &bsf.upper_bounds,
            );
            assert!(matches!(out, SimplexOutcome::Optimal(..)));
            (partial_price_cols_scanned(), iters)
        };

        let (scanned_partial, iters_partial) = run(1);
        let (scanned_full, iters_full) = run(FULL_PRICE);

        assert_eq!(
            scanned_full,
            (iters_full * n_price) as u64,
            "full pricing must price all {n_price} cols on each of {iters_full} iters"
        );
        assert!(
            scanned_partial < (iters_partial * n_price) as u64,
            "partial pricing scanned {scanned_partial} cols over {iters_partial} iters \
             (≥ full {iters_partial}*{n_price}); windowing is not active"
        );
    }

    /// Partial pricing must also preserve correctness on the augmented Phase I
    /// core (`primal_simplex_aug`). Eq LP x0+x1=4, x0+2x1=5, 0≤x0≤3, x1≥0;
    /// Phase I must drive the artificial sum to zero under both full and
    /// partial pricing.
    #[test]
    fn partial_pricing_aug_phase1_matches_full() {
        let a_aug = CscMatrix::from_triplets(
            &[0, 1, 0, 1, 0, 1],
            &[0, 0, 1, 1, 2, 3],
            &[1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
            2,
            4,
        )
        .unwrap();
        let n_struct = 2usize;
        let ubs_aug = vec![3.0f64, f64::INFINITY, f64::INFINITY, f64::INFINITY];
        let c_p1 = vec![0.0f64, 0.0, 1.0, 1.0];
        let opts = SolverOptions::default();

        let build_state = || BoundedDualState {
            basis: vec![2usize, 3],
            at_upper: vec![false; 4],
            x_b: vec![4.0f64, 5.0],
            reduced_costs: vec![0.0; 4],
            is_basic: vec![false, false, true, true],
            iterations: 0,
            price_start: 0,
        };

        let run = |chunk: usize| -> f64 {
            let _g = PartialPriceChunkGuard::set(chunk);
            let mut state = build_state();
            let mut iters = 0usize;
            match bounded_primal_phase1(&a_aug, &c_p1, &ubs_aug, n_struct, &mut state, &opts, &mut iters)
            {
                SimplexOutcome::Optimal(art_sum, _) => art_sum,
                other => panic!("expected Phase I Optimal, got {other:?}"),
            }
        };

        let art_full = run(FULL_PRICE);
        let art_partial = run(1);
        assert!(art_full.abs() < 1e-6, "full Phase I art_sum {art_full:.6e}");
        assert!(
            art_partial.abs() < 1e-6,
            "partial Phase I art_sum {art_partial:.6e} — partial pricing broke Phase I feasibility"
        );
    }
}
