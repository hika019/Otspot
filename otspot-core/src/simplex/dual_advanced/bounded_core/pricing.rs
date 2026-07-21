//! Reduced-cost pricing for bounded primal/dual simplex cores.

use super::super::super::dual_common::compute_dual_vars_into;
use crate::basis::LuBasis;
use crate::linalg::timeout::deadline_reached;
use crate::sparse::CscMatrix;
use std::time::Instant;

/// Deadline check interval for the RC inner loop.
///
/// `Instant::now()` costs ~48ns/call; per-col checks dominate on large LPs.
/// chunk=512 reduces checks to `ceil(n/512)`, making overhead negligible.
/// Max deadline overshoot is ~512 cols (~100µs), well within solver-level
/// deadlines. Measured 1.5–2.25× speedup on large network LPs.
pub(super) const DEADLINE_CHECK_INTERVAL: usize = 512;

/// Cyclic partial-pricing window size for bounded primal cores.
///
/// Each pass prices at most this many non-basic columns from a rotating cursor.
/// Optimality requires a full `n_price` sweep with no improving column (see
/// `partial_price_entering`). For `n_price ≤ PARTIAL_PRICE_CHUNK`, pricing is
/// full (identical to pre-partial cores). On large network LPs the RC scan
/// dominates wall time; windowing cuts per-iter cost while preserving the
/// exact optimality test.
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
pub(crate) fn set_partial_price_chunk_override(v: usize) -> usize {
    PARTIAL_PRICE_CHUNK_OVERRIDE.with(|c| c.replace(v))
}

#[cfg(test)]
pub(crate) fn set_partial_price_single_window(v: bool) -> bool {
    PARTIAL_PRICE_SINGLE_WINDOW.with(|c| c.replace(v))
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

pub(super) fn compute_reduced_costs_into_timed(
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
pub(super) enum PartialPrice {
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
/// Prices `partial_price_chunk`-sized windows from `price_start`, wrapping
/// around `n_price`. First window with an improving column wins (Dantzig
/// within window). `Optimal` requires a full `n_price` sweep with no
/// improving column — windowing defers that sweep across pivots but never
/// short-circuits it.
///
/// `score(j, rc_j) -> Option<f64>`: `None` = not improving;
/// `Some(s)` = improving with score `s` (larger wins).
pub(super) fn partial_price_entering<F>(
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
        if single_window {
            break;
        }
    }

    PartialPrice::Optimal { next_start: start }
}
