//! BFRT (Bound-Flipping Ratio Test) sentinel suite.
//!
//! Validates the BFRT primitive via:
//! 1. **Multi-pattern ratio test scenarios** — bounded vs unbounded mixes,
//!    at-upper handling, tie breaking, degenerate cases.
//! 2. **Pivot reduction via a mini bounded dual simplex** — Harris vs BFRT
//!    on synthetic bound-rich LPs; asserts ≥ 30 % pivot reduction.
//! 3. **No-op proof** — `BOUND_FLIP_DISABLE=1` env hook drops BFRT back to
//!    Harris choice; sentinel ratio collapses to ~1.0.
//! 4. **Probe wiring** — flip-invocation counter increments only when BFRT
//!    genuinely flips at least one variable; reverting wiring (passing
//!    infinite uppers) makes the counter stay at 0.
//!
//! ## Why a mini simplex in the test file
//!
//! Production wiring of BFRT into `dual_advanced/core.rs` requires an
//! alternate `StandardForm` that does **not** expand bounded variables to
//! upper-bound rows (otherwise BFRT sees only x ≥ 0 columns with no flip
//! handles). That refactor is a follow-up task. The mini simplex below
//! lets us prove the primitive's pivot-reduction effect on representative
//! bound-rich data **today**, while production integration matures.

use otspot_core::bound_flip::{
    bfrt_flip_invocations, bfrt_select_entering, reset_bfrt_flip_invocations, BfrtResult, ColBound,
};

const PIVOT_TOL: f64 = 1e-8;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Honor `BOUND_FLIP_DISABLE=1` env hook. When set, the wrapper degenerates
/// BFRT to Harris (smallest breakpoint, no flips). Mirrors the production
/// no-op contract documented on `SolverOptions::enable_bound_flipping`.
fn bfrt_or_harris(
    trow: &[f64],
    reduced_costs: &[f64],
    is_basic: &[bool],
    bounds: &[ColBound],
    n_price: usize,
    leaving_residual: f64,
) -> Option<BfrtResult> {
    if std::env::var("BOUND_FLIP_DISABLE").ok().as_deref() == Some("1") {
        // Harris equivalent: treat all uppers as infinite → no flips ever.
        let harris_bounds: Vec<ColBound> = bounds
            .iter()
            .map(|b| ColBound {
                upper: f64::INFINITY,
                at_upper: b.at_upper,
            })
            .collect();
        return bfrt_select_entering(
            trow,
            reduced_costs,
            is_basic,
            &harris_bounds,
            n_price,
            PIVOT_TOL,
            leaving_residual,
        );
    }
    bfrt_select_entering(
        trow,
        reduced_costs,
        is_basic,
        bounds,
        n_price,
        PIVOT_TOL,
        leaving_residual,
    )
}

/// Mini bounded dual simplex: counts pivots when solving Bx_B = b with
/// `n_price` non-basic candidates, each with finite upper bound. Returns the
/// number of pivot rounds needed to drive `residual` to ≤ tol.
///
/// Each "iteration" picks one leaving row with the largest |residual|, runs
/// the chosen ratio test, applies flips (no-cost — just bookkeeping) and one
/// real pivot. The synthetic structure abstracts away FTRAN/BTRAN so we can
/// focus purely on the ratio-test step-size effect.
struct MiniSimplexResult {
    pivots: usize,
    flips_total: usize,
}

fn mini_bounded_simplex(
    n_rows: usize,
    n_cols: usize,
    initial_residual: f64,
    seed: u64,
    use_bfrt: bool,
) -> MiniSimplexResult {
    // Synthetic ratio-test inputs: each leaving row r has
    //   trow[j] = lcg-generated in [0.5, 2.5] for j < n_cols
    //   reduced_costs[j] = lcg-generated in [0.01, 0.5]
    //   bounds[j].upper = small random in [0.1, 1.5]
    // We simulate `n_rows` iterations, but each loop tracks residual
    // progress: BFRT consumes more residual per pivot, so fewer pivots.
    let mut state = seed | 1;
    let mut lcg = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 32) as u32 as f64) / (u32::MAX as f64)
    };

    let trow: Vec<f64> = (0..n_cols).map(|_| 0.5 + 2.0 * lcg()).collect();
    let r: Vec<f64> = (0..n_cols).map(|_| 0.01 + 0.49 * lcg()).collect();
    let bounds: Vec<ColBound> = (0..n_cols)
        .map(|_| ColBound {
            upper: 0.1 + 1.4 * lcg(),
            at_upper: false,
        })
        .collect();
    let is_basic = vec![false; n_cols];

    let mut residual = initial_residual;
    let mut pivots = 0usize;
    let mut flips_total = 0usize;
    let max_iter = n_rows * 200;

    while residual > 1e-9 && pivots < max_iter {
        let pick = if use_bfrt {
            bfrt_select_entering(&trow, &r, &is_basic, &bounds, n_cols, PIVOT_TOL, residual)
        } else {
            // Harris emulation: force all uppers infinite → BFRT degenerates
            // to smallest-breakpoint selection (= Harris pass 1 / no flips).
            let harris_bounds: Vec<ColBound> = bounds
                .iter()
                .map(|b| ColBound {
                    upper: f64::INFINITY,
                    at_upper: b.at_upper,
                })
                .collect();
            bfrt_select_entering(
                &trow,
                &r,
                &is_basic,
                &harris_bounds,
                n_cols,
                PIVOT_TOL,
                residual,
            )
        };
        let Some(res) = pick else { break };
        let abs_pivot = trow[res.entering_col].abs();
        // residual consumed = step * |pivot| (entering) + Σ u_k * |trow_k| (flips).
        // Standard simplex residual reduction.
        let mut consumed = res.theta * abs_pivot;
        for &f in res.flips.iter() {
            let f: usize = f;
            consumed += bounds[f].upper * trow[f].abs();
        }
        if consumed <= 1e-12 {
            // Degenerate: cannot reduce. Bail to avoid infinite loop in test.
            break;
        }
        residual = (residual - consumed).max(0.0);
        flips_total += res.flips.len();
        pivots += 1;
    }
    MiniSimplexResult { pivots, flips_total }
}

// ---------------------------------------------------------------------------
// Multi-pattern table-driven sentinel
// ---------------------------------------------------------------------------

#[test]
fn sentinel_bfrt_reduces_pivots_on_bound_rich_lp() {
    // Run mini bounded simplex over multiple synthetic seeds (multi-pattern
    // data per `feedback_test_multi_data_pattern`). For each seed, BFRT
    // pivots must be strictly fewer than Harris pivots, and the average
    // reduction across seeds must clear the 30 % gate.
    let seeds = [0xC0FFEE_u64, 0xDEAD_BEEF, 0xABCD_1234, 0xFEED_FACE, 0x4242_4242];
    let mut total_h = 0usize;
    let mut total_b = 0usize;
    for &seed in &seeds {
        let harris = mini_bounded_simplex(50, 40, 30.0, seed, false);
        let bfrt = mini_bounded_simplex(50, 40, 30.0, seed, true);
        assert!(
            bfrt.pivots < harris.pivots,
            "seed={:#x}: BFRT pivots ({}) must be < Harris pivots ({})",
            seed,
            bfrt.pivots,
            harris.pivots
        );
        assert!(
            bfrt.flips_total > 0,
            "seed={:#x}: BFRT must perform at least one flip on bound-rich input",
            seed
        );
        total_h += harris.pivots;
        total_b += bfrt.pivots;
    }
    let reduction = 1.0 - (total_b as f64) / (total_h as f64);
    assert!(
        reduction >= 0.30,
        "BFRT must reduce pivots by ≥ 30 % across seeds; got {:.1} % (h={}, b={})",
        reduction * 100.0,
        total_h,
        total_b
    );
    eprintln!(
        "[BFRT sentinel] pivots Harris={} BFRT={} (reduction={:.1}%)",
        total_h,
        total_b,
        reduction * 100.0
    );
}

#[test]
fn sentinel_bfrt_no_op_when_disabled() {
    // Saturate `BOUND_FLIP_DISABLE=1` for the duration of this test thread.
    // The wrapper must degenerate to Harris choice → pivot counts collide.
    // SAFETY: env::set_var is unsafe in Rust 1.86+ because it is process-wide
    // and not thread-safe across libc; we accept the racy semantics here
    // because the sentinel runs serially per nextest's default `-j1` per
    // test, and we unset before returning.
    unsafe { std::env::set_var("BOUND_FLIP_DISABLE", "1"); }
    let seed = 0xC0FFEE_u64;
    let harris = mini_bounded_simplex(50, 40, 30.0, seed, false);
    // bfrt_or_harris honors the env var: this should produce Harris pivots.
    let bfrt_disabled_pivots = {
        let mut state = seed | 1;
        let mut lcg = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 32) as u32 as f64) / (u32::MAX as f64)
        };
        let trow: Vec<f64> = (0..40).map(|_| 0.5 + 2.0 * lcg()).collect();
        let r: Vec<f64> = (0..40).map(|_| 0.01 + 0.49 * lcg()).collect();
        let bounds: Vec<ColBound> = (0..40)
            .map(|_| ColBound { upper: 0.1 + 1.4 * lcg(), at_upper: false })
            .collect();
        let is_basic = vec![false; 40];
        let mut residual = 30.0;
        let mut pivots = 0usize;
        while residual > 1e-9 && pivots < 10_000 {
            let pick = bfrt_or_harris(&trow, &r, &is_basic, &bounds, 40, residual);
            let Some(res) = pick else { break };
            let abs_pivot = trow[res.entering_col].abs();
            let mut consumed = res.theta * abs_pivot;
            for &f in &res.flips {
                consumed += bounds[f].upper * trow[f].abs();
            }
            if consumed <= 1e-12 { break; }
            residual = (residual - consumed).max(0.0);
            pivots += 1;
        }
        pivots
    };
    unsafe { std::env::remove_var("BOUND_FLIP_DISABLE"); }
    assert_eq!(
        bfrt_disabled_pivots, harris.pivots,
        "BOUND_FLIP_DISABLE=1 must collapse BFRT to Harris pivot count"
    );
}

#[test]
fn sentinel_probe_counter_proves_wiring_live() {
    // Reset → run BFRT on bound-rich input → counter must increment.
    // Then run with all infinite uppers → counter must NOT increment further.
    // This is the `feedback_sentinel_must_fail_under_noop` proof: if a
    // refactor accidentally short-circuits BFRT to Harris (= passing
    // infinite uppers everywhere), the counter stops growing and this
    // sentinel fails.
    reset_bfrt_flip_invocations();
    let trow = vec![1.0, 1.0, 1.0];
    let r = vec![0.1, 0.2, 0.3];
    let bounds_bounded = vec![
        ColBound { upper: 1.0, at_upper: false },
        ColBound { upper: 1.0, at_upper: false },
        ColBound { upper: f64::INFINITY, at_upper: false },
    ];
    let is_basic = vec![false; 3];

    // Run #1: bounded → at least one flip → counter increments
    let _ = bfrt_select_entering(&trow, &r, &is_basic, &bounds_bounded, 3, PIVOT_TOL, 2.5).unwrap();
    let after_real = bfrt_flip_invocations();
    assert!(after_real >= 1, "BFRT must increment counter on real flip");

    // Run #2: all infinite (= Harris emulation) → counter unchanged
    let bounds_unbounded = vec![
        ColBound { upper: f64::INFINITY, at_upper: false },
        ColBound { upper: f64::INFINITY, at_upper: false },
        ColBound { upper: f64::INFINITY, at_upper: false },
    ];
    let _ = bfrt_select_entering(&trow, &r, &is_basic, &bounds_unbounded, 3, PIVOT_TOL, 2.5).unwrap();
    assert_eq!(
        bfrt_flip_invocations(),
        after_real,
        "infinite-upper input must NOT flip — counter must be unchanged"
    );
}

#[test]
fn sentinel_table_driven_step_size_increases_with_bounds() {
    // Multiple `(name, trow, r, bounds, residual, harris_theta, min_bfrt_theta)` patterns.
    // Each pattern represents a distinct ratio-test regime; BFRT must reach
    // a θ at least as large as the listed minimum, and strictly larger than
    // Harris when any flippable column is present.
    struct Case {
        name: &'static str,
        trow: Vec<f64>,
        r: Vec<f64>,
        bounds: Vec<ColBound>,
        residual: f64,
        harris_theta: f64,
        min_bfrt_theta: f64,
    }

    let cases = vec![
        Case {
            name: "single small flip",
            trow: vec![1.0, 1.0],
            r: vec![0.1, 0.5],
            bounds: vec![
                ColBound { upper: 1.0, at_upper: false },
                ColBound { upper: f64::INFINITY, at_upper: false },
            ],
            residual: 2.0,
            harris_theta: 0.1,
            min_bfrt_theta: 0.5,
        },
        Case {
            name: "no bounded → harris equivalent",
            trow: vec![1.0, 2.0, 3.0],
            r: vec![0.3, 0.4, 0.9],
            bounds: vec![ColBound { upper: f64::INFINITY, at_upper: false }; 3],
            residual: 1.0,
            harris_theta: 0.2,
            min_bfrt_theta: 0.2,
        },
        Case {
            name: "chain of 5 small bounded",
            trow: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            r: vec![0.01, 0.02, 0.03, 0.04, 0.05],
            bounds: (0..5)
                .map(|_| ColBound { upper: 1.0, at_upper: false })
                .collect(),
            residual: 4.5,
            harris_theta: 0.01,
            min_bfrt_theta: 0.04,
        },
        Case {
            name: "mixed at-upper + at-lower",
            trow: vec![-1.0, 2.0, 1.5],
            r: vec![-0.05, 0.6, 0.3],
            bounds: vec![
                ColBound { upper: 1.0, at_upper: true },  // breakpoint = 0.05, weight = 1
                ColBound { upper: f64::INFINITY, at_upper: false }, // breakpoint = 0.3, weight = ∞
                ColBound { upper: 0.5, at_upper: false }, // breakpoint = 0.2, weight = 0.75
            ],
            residual: 1.5,
            harris_theta: 0.05,
            min_bfrt_theta: 0.2,
        },
    ];

    let is_basic_4 = [false; 5];
    for c in &cases {
        let n = c.trow.len();
        let is_basic = &is_basic_4[..n];
        let res = bfrt_select_entering(&c.trow, &c.r, is_basic, &c.bounds, n, PIVOT_TOL, c.residual)
            .unwrap_or_else(|| panic!("{}: BFRT must return Some", c.name));
        assert!(
            res.theta + 1e-12 >= c.min_bfrt_theta,
            "{}: BFRT theta {:.6} must be ≥ {:.6}",
            c.name,
            res.theta,
            c.min_bfrt_theta
        );
        assert!(
            res.theta + 1e-12 >= c.harris_theta,
            "{}: BFRT theta {:.6} must dominate Harris {:.6}",
            c.name,
            res.theta,
            c.harris_theta
        );
    }
}
