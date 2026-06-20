//! Reduced-cost pricing for bounded primal/dual simplex cores.

use crate::linalg::timeout::deadline_reached;
use super::super::super::dual_common::compute_dual_vars_into;
use crate::basis::LuBasis;
use crate::sparse::CscMatrix;
use std::time::Instant;

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
pub(super) const DEADLINE_CHECK_INTERVAL: usize = 512;

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
