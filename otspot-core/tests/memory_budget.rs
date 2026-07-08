// Synthetic problem builders index into parallel row/col/val vecs by a
// derived offset, so the numeric loops read more clearly as index loops than
// as iterator chains.
#![allow(clippy::needless_range_loop)]

//! Memory-budget fence for solve paths that must stay O(nnz), not O(n^2).
//!
//! QPLIB_8547-class sparse problems (n in the 1e5-1e6 range, nnz = O(n)) hit an
//! unconditional dense expansion on the conic bridge and drove RSS to 10-15 GB
//! (OOM). That was a code bug, not a missing test: nothing in the suite would
//! have caught an O(n^2) (or worse) allocation on a path that is supposed to
//! be linear in problem size. This file is the general fence for that failure
//! class: synthetic sparse problems with a known closed-form optimum, solved
//! through the real public entry points, with peak allocation measured by a
//! process-wide counting allocator and asserted against a budget.
//!
//! The allocator is `#[global_allocator]` for this binary only (integration
//! test binaries are separate processes), so it does not affect unit tests
//! compiled into the library. It uses atomics rather than thread-locals
//! (contrast `otspot_io`'s single-threaded `peak_alloc`) because these solve
//! paths use rayon/faer internal parallelism, and a real O(n^2) regression on
//! a worker thread must count toward the peak. Only allocations routed through
//! the Rust global allocator are visible: thread stacks and direct/FFI
//! `malloc`/`mmap` are not counted.
//!
//! Under `cargo test` all three routes run as threads of one process and
//! share `CURRENT`/`PEAK`, so each test holds `MEM_TEST_LOCK` across its
//! whole build+solve+measure window — serialization is self-enforced,
//! independent of the runner. (nextest runs one process per test, so
//! cross-test contamination cannot happen there; its `memory-budget`
//! test-group only caps concurrent RSS, see `nextest.toml`.)

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

struct TrackingAlloc;

static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static BASELINE_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CURRENT_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            if new_size >= layout.size() {
                grow(new_size - layout.size());
            } else {
                CURRENT_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

#[inline]
fn grow(delta: usize) {
    let now = CURRENT_BYTES.fetch_add(delta, Ordering::Relaxed) + delta;
    PEAK_BYTES.fetch_max(now, Ordering::Relaxed);
}

#[global_allocator]
static TRACKING_ALLOC: TrackingAlloc = TrackingAlloc;

/// Serializes the build+solve+measure window of each test so that under
/// `cargo test` (threads in one process) tests cannot inflate each other's
/// shared counters. A panicking holder (failed assertion) poisons the lock;
/// the counters are plain atomics with no invariant to protect, so the
/// guard is recovered via `into_inner`.
static MEM_TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock_measurement() -> MutexGuard<'static, ()> {
    MEM_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Rebase peak tracking to the current live-allocation level. Call
/// immediately before the section under measurement.
fn reset_peak() {
    let cur = CURRENT_BYTES.load(Ordering::Relaxed);
    BASELINE_BYTES.store(cur, Ordering::Relaxed);
    PEAK_BYTES.store(cur, Ordering::Relaxed);
}

/// Peak live-allocation bytes above the last `reset_peak()` baseline.
fn peak_bytes() -> usize {
    PEAK_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(BASELINE_BYTES.load(Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// Detection-power injection (verification aid, always off)
// ---------------------------------------------------------------------------

/// Flip to `true` locally to confirm a route's budget actually has teeth: it
/// injects one dense `k x k` allocation (same shape as the historical
/// `CscMatrix::from_triplets(..., n, n)`-per-slot bug class) inside the
/// measured window, sized to `2x` the route's own budget. Verified manually
/// during development that all three routes FAIL with this `true` (see
/// commit message for the measured peaks) and must stay `false` in committed
/// code — this is a revert-checked sentinel, not a runtime switch.
///
/// `k` is derived from the budget rather than hardcoded to the route's real
/// `n` because the real `n` (up to 1e6) would make a literal `n x n` bomb an
/// 8 TB allocation — physically impossible to execute for verification, and
/// not the point: the point is confirming the *tracking + assertion* catches
/// a dense blow-up at all, not reproducing the exact historical byte count.
const INJECT_QUADRATIC_BLOWUP: bool = false;

#[inline]
fn maybe_inject_dense_blowup(budget_bytes: usize) {
    if INJECT_QUADRATIC_BLOWUP {
        let target_bytes = budget_bytes as u128 * 2;
        let k = ((target_bytes / 8) as f64).sqrt().ceil() as usize;
        let bomb: Vec<Vec<f64>> = vec![vec![0.0_f64; k]; k];
        std::hint::black_box(&bomb);
    }
}

// ---------------------------------------------------------------------------
// LP route: banded equality LP, n = 2*M, nnz = 2*M
// ---------------------------------------------------------------------------

use otspot_core::lp::solve_lp_with;
use otspot_core::options::SolverOptions;
use otspot_core::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot_core::sparse::CscMatrix;

/// Rows: `M`. Vars: `N_LP = 2*M` (`x_i` at `i`, slack `s_i` at `M+i`).
/// Matches the QPLIB_8547 scale class (n in 1e5-1e6, nnz = O(n)).
const LP_M: usize = 500_000;
const LP_N: usize = 2 * LP_M;

/// Measured peak (release, `cargo nextest run --release`, this machine):
/// 296.3 MB. Budget = measured x3 margin (CLAUDE.md convention). An O(n^2)
/// dense fallback at this scale would need `LP_N^2 * 8` bytes ~= 8 TB, so this
/// budget fails hard on any quadratic-or-worse regression while giving ample
/// slack for legitimate linear-factor changes (allocator fragmentation,
/// presolve buffers, etc.).
const LP_PEAK_BUDGET_BYTES: usize = 900 * 1024 * 1024;

/// `x_i + s_i = 1`, `0 <= x_i, s_i <= 1`, minimize `c_x(i) x_i + c_s(i) s_i`.
/// Each row is an independent 2-var LP: since `c_x(i) < 0 < c_s(i)` always,
/// the optimum is always the corner `x_i=1, s_i=0` — a hand-provable, not
/// solver-derived, fact used as the independent oracle below.
fn build_banded_lp(m: usize) -> (LpProblem, f64) {
    let n = 2 * m;
    let mut rows = Vec::with_capacity(n);
    let mut cols = Vec::with_capacity(n);
    let mut vals = Vec::with_capacity(n);
    let mut c = vec![0.0; n];
    let mut expected_obj = 0.0_f64;
    for i in 0..m {
        rows.push(i);
        cols.push(i);
        vals.push(1.0);
        rows.push(i);
        cols.push(m + i);
        vals.push(1.0);

        let c_x = -1.0 - ((i % 17) as f64) * 0.01; // in [-1.16, -1.0]
        let c_s = 0.5 + ((i % 11) as f64) * 0.02; // in [0.5, 0.7]
        c[i] = c_x;
        c[m + i] = c_s;
        expected_obj += c_x; // corner x_i=1, s_i=0 always wins
    }
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).expect("banded LP matrix");
    let b = vec![1.0; m];
    let bounds = vec![(0.0, 1.0); n];
    let lp = LpProblem::new_general(c, a, b, vec![ConstraintType::Eq; m], bounds, None)
        .expect("banded LP problem");
    (lp, expected_obj)
}

/// Peak allocation fence for the LP simplex path on a large sparse problem.
///
/// **No-op failure guarantee**: flipping `INJECT_QUADRATIC_BLOWUP` to `true`
/// makes this FAIL (verified manually: peak jumps to 1823.2 MB against the
/// 900 MB budget — a ~1.8 GB dense `k x k` bomb sized to 2x the budget).
#[test]
fn lp_route_peak_within_budget() {
    let _serial = lock_measurement();
    let (lp, expected_obj) = build_banded_lp(LP_M);

    reset_peak();
    let r = solve_lp_with(&lp, &SolverOptions::default());
    maybe_inject_dense_blowup(LP_PEAK_BUDGET_BYTES);
    let peak = peak_bytes();
    eprintln!(
        "lp_route peak={peak} bytes ({:.2} MB)",
        peak as f64 / 1_048_576.0
    );

    assert_eq!(r.status, SolveStatus::Optimal, "LP route: expected Optimal");
    let rel_err = (r.objective - expected_obj).abs() / expected_obj.abs().max(1.0);
    assert!(
        rel_err < 1e-6,
        "LP route: obj={:.6e} expected={:.6e} rel_err={:.3e}",
        r.objective,
        expected_obj,
        rel_err
    );
    assert!(
        peak <= LP_PEAK_BUDGET_BYTES,
        "LP route peak {:.1} MB exceeds {:.1} MB budget (n={LP_N}) — \
         check for an unconditional dense expansion on the simplex/presolve path",
        peak as f64 / 1_048_576.0,
        LP_PEAK_BUDGET_BYTES as f64 / 1_048_576.0
    );
}

// ---------------------------------------------------------------------------
// QP IPM route: diagonal Hessian + banded equality constraints, n = 2*M
// ---------------------------------------------------------------------------

use otspot_core::qp::solve_qp_with;
use otspot_core::QpProblem;

/// Rows: `M`. Vars: `N_QP = 2*M`. Diagonal `Q` mirrors the DCL-type QPLIB
/// Hessian (separable quadratic objective, sparse linear constraints) that
/// originally OOM'd through the conic bridge.
const QP_M: usize = 250_000;
const QP_N: usize = 2 * QP_M;

/// Measured peak (release, `cargo nextest run --release`, this machine):
/// 823.5 MB. Budget = measured x3 margin, same rationale as
/// `LP_PEAK_BUDGET_BYTES`. An O(n^2) KKT/dense fallback at this scale would
/// need `QP_N^2 * 8` bytes ~= 2 TB — this budget fails hard on that class of
/// regression while tolerating legitimate linear-factor IPM overhead
/// (multiple KKT solves per iteration, line-search buffers).
const QP_PEAK_BUDGET_BYTES: usize = 2_560 * 1024 * 1024;

/// `x_i + s_i = 1`, `0 <= x_i, s_i <= 1`, minimize
/// `1/2 q(i) x_i^2 + 1/2 q(M+i) s_i^2 + c(i) x_i + c(M+i) s_i`.
/// Substituting `s_i = 1 - x_i` reduces each row to a strictly convex 1-D
/// quadratic in `x_i`; the closed-form stationary point (clipped to `[0,1]`)
/// is the independent oracle used below, not anything the solver computes.
fn build_diag_qp(m: usize) -> (QpProblem, f64) {
    let n = 2 * m;
    let q_coeff = |k: usize| 1.0 + (k % 5) as f64 * 0.1; // in [1.0, 1.4], > 0
    let c_coeff = |k: usize| -1.0 - (k % 13) as f64 * 0.01;

    let mut qrows = Vec::with_capacity(n);
    let mut qcols = Vec::with_capacity(n);
    let mut qvals = Vec::with_capacity(n);
    let mut c = vec![0.0; n];
    for k in 0..n {
        qrows.push(k);
        qcols.push(k);
        qvals.push(q_coeff(k));
        c[k] = c_coeff(k);
    }
    let q = CscMatrix::from_triplets(&qrows, &qcols, &qvals, n, n).expect("diag Q");

    let mut rows = Vec::with_capacity(n);
    let mut cols = Vec::with_capacity(n);
    let mut vals = Vec::with_capacity(n);
    let mut expected_obj = 0.0_f64;
    for i in 0..m {
        rows.push(i);
        cols.push(i);
        vals.push(1.0);
        rows.push(i);
        cols.push(m + i);
        vals.push(1.0);

        let a = q_coeff(i);
        let b = q_coeff(m + i);
        let ca = c_coeff(i);
        let cb = c_coeff(m + i);
        let x_star = ((b - ca + cb) / (a + b)).clamp(0.0, 1.0);
        expected_obj += 0.5 * a * x_star * x_star
            + 0.5 * b * (1.0 - x_star).powi(2)
            + ca * x_star
            + cb * (1.0 - x_star);
    }
    let a_mat = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).expect("banded QP A");
    let b_vec = vec![1.0; m];
    let bounds = vec![(0.0, 1.0); n];
    let qp = QpProblem::new(q, c, a_mat, b_vec, bounds, vec![ConstraintType::Eq; m])
        .expect("diag QP problem");
    (qp, expected_obj)
}

/// Peak allocation fence for the QP IPM path on a large sparse problem.
///
/// **No-op failure guarantee**: flipping `INJECT_QUADRATIC_BLOWUP` to `true`
/// makes this FAIL (verified manually: peak jumps to 5134.2 MB against the
/// 2560 MB budget).
#[test]
fn qp_ipm_route_peak_within_budget() {
    let _serial = lock_measurement();
    let (qp, expected_obj) = build_diag_qp(QP_M);

    reset_peak();
    let r = solve_qp_with(&qp, &SolverOptions::default());
    maybe_inject_dense_blowup(QP_PEAK_BUDGET_BYTES);
    let peak = peak_bytes();
    eprintln!(
        "qp_route peak={peak} bytes ({:.2} MB)",
        peak as f64 / 1_048_576.0
    );

    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "QP IPM route: expected Optimal (strictly convex Q)"
    );
    let rel_err = (r.objective - expected_obj).abs() / expected_obj.abs().max(1.0);
    assert!(
        rel_err < 1e-4,
        "QP IPM route: obj={:.6e} expected={:.6e} rel_err={:.3e}",
        r.objective,
        expected_obj,
        rel_err
    );
    assert!(
        peak <= QP_PEAK_BUDGET_BYTES,
        "QP IPM route peak {:.1} MB exceeds {:.1} MB budget (n={QP_N}) — \
         check for an unconditional dense KKT expansion",
        peak as f64 / 1_048_576.0,
        QP_PEAK_BUDGET_BYTES as f64 / 1_048_576.0
    );
}

// ---------------------------------------------------------------------------
// Conic SOCP route: many small independent second-order cones
// ---------------------------------------------------------------------------

use otspot_core::conic::{solve_socp, ConeSpec, ConicOptions, ConicProblem};

/// `conic::ipm::solve` (Phase 3a, conic-oom) solves a sparse augmented
/// quasidefinite KKT system (`conic::kkt`) built directly from `A`/`G`'s CSC
/// storage -- no dense `A`, `G`, or KKT matrix of any size is ever
/// materialized (unlike the pre-Phase-3a `csc_to_dense`-based path this
/// fence originally pinned, which was O((n+m)^2) by construction). `m` many
/// small independent SOC blocks contribute `O(sum d_i^2)` fill to the KKT's
/// `W^2` block, which for `NUM_BLOCKS` blocks of fixed dimension is `O(n)`
/// overall, so `NUM_BLOCKS` can now be scaled up (unlike the old dense
/// fence's deliberately-small size) to make this a genuine O(nnz) fence.
const NUM_BLOCKS: usize = 10_000;
const CONIC_N: usize = 3 * NUM_BLOCKS;

/// Measured peak (release, `cargo test --release --test memory_budget
/// conic_socp_route_peak_within_budget -- --nocapture`, this machine):
/// 27.6 MB at n=30000, m=30000, p=10000. Budget = measured x3 margin
/// (CLAUDE.md convention). This *is* now an O(nnz) fence: Phase 3a
/// (conic-oom) replaced the dense `A`/`G`/KKT densification this fence used
/// to pin (`csc_to_dense` calls, now removed) with the sparse augmented
/// quasidefinite system in `conic::kkt` (no dense intermediate of any size
/// anywhere in the solve path). A dense-KKT fallback at this scale would
/// need `(n+p+m)^2 * 8` bytes = `(70000)^2 * 8` ~= 39.2 GB -- physically
/// unrunnable on ordinary CI hardware, so this fence cannot be verified by
/// literally reverting and re-running (see `INJECT_QUADRATIC_BLOWUP` for the
/// revert-detection mechanism used instead, verified below).
const CONIC_PEAK_BUDGET_BYTES: usize = 83 * 1024 * 1024;

/// `NUM_BLOCKS` independent 3-dim SOC blocks: cone `x0 >= sqrt(x1^2+x2^2)`
/// via `G = -I`, `h = 0` (so `s = x`). Equality `x0_b = 1` fixes the cone
/// apex; minimizing `-x1_b` then drives `x1_b` to the cone boundary at
/// `x1_b=1, x2_b=0` (any nonzero `x2_b` would only tighten headroom on
/// `x1_b` for no benefit). Closed-form optimum: `-1` per block.
fn build_many_soc_socp(num_blocks: usize) -> (ConicProblem, f64) {
    let n = 3 * num_blocks;
    let mut g_rows = Vec::with_capacity(n);
    let mut g_cols = Vec::with_capacity(n);
    let mut g_vals = Vec::with_capacity(n);
    for i in 0..n {
        g_rows.push(i);
        g_cols.push(i);
        g_vals.push(-1.0);
    }
    let g = CscMatrix::from_triplets(&g_rows, &g_cols, &g_vals, n, n).expect("block SOC G");
    let h = vec![0.0; n];

    let mut c = vec![0.0; n];
    let mut a_rows = Vec::with_capacity(num_blocks);
    let mut a_cols = Vec::with_capacity(num_blocks);
    let mut a_vals = Vec::with_capacity(num_blocks);
    for blk in 0..num_blocks {
        c[3 * blk + 1] = -1.0;
        a_rows.push(blk);
        a_cols.push(3 * blk);
        a_vals.push(1.0);
    }
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, num_blocks, n)
        .expect("block SOC equality A");
    let b = vec![1.0; num_blocks];

    let problem = ConicProblem {
        c,
        a,
        b,
        g,
        h,
        cone: ConeSpec {
            l: 0,
            soc: vec![3; num_blocks],
        },
    };
    (problem, -(num_blocks as f64))
}

/// Peak allocation fence for the conic SOCP path (see `NUM_BLOCKS` doc
/// comment: an O(nnz) fence as of Phase 3a / conic-oom).
///
/// **No-op failure guarantee**: flipping `INJECT_QUADRATIC_BLOWUP` to `true`
/// makes this FAIL (verified manually: peak jumps to 166.9 MB against the
/// 83.0 MB budget).
#[test]
fn conic_socp_route_peak_within_budget() {
    let _serial = lock_measurement();
    let (problem, expected_obj) = build_many_soc_socp(NUM_BLOCKS);

    reset_peak();
    let r = solve_socp(&problem, &ConicOptions::default());
    maybe_inject_dense_blowup(CONIC_PEAK_BUDGET_BYTES);
    let peak = peak_bytes();
    eprintln!(
        "conic_route peak={peak} bytes ({:.2} MB)",
        peak as f64 / 1_048_576.0
    );

    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "conic SOCP route: expected Optimal"
    );
    let rel_err = (r.objective - expected_obj).abs() / expected_obj.abs().max(1.0);
    assert!(
        rel_err < 1e-4,
        "conic SOCP route: obj={:.6e} expected={:.6e} rel_err={:.3e}",
        r.objective,
        expected_obj,
        rel_err
    );
    assert!(
        peak <= CONIC_PEAK_BUDGET_BYTES,
        "conic SOCP route peak {:.1} MB exceeds {:.1} MB budget (n={CONIC_N}, {NUM_BLOCKS} blocks)",
        peak as f64 / 1_048_576.0,
        CONIC_PEAK_BUDGET_BYTES as f64 / 1_048_576.0
    );
}

// ---------------------------------------------------------------------------
// Conic SOCP route: one huge second-order cone (Phase 3b rank-1 border)
// ---------------------------------------------------------------------------

/// Single-SOC dimension for the border-representation fence. Matches the
/// QPLIB DCQ shape (the QCQP->SOCP bridge emits one SOC of dimension `n+2`
/// per quadratic term; QPLIB_8585 has `n = 99,999`). Any dimension at or
/// above `conic::cone::SOC_BORDER_MIN_DIM` takes the same `O(d)` border
/// path; this size makes the O(d^2) alternative unambiguous: a dense
/// `d x d` `W^2` block would need `d^2 * 8` bytes = 80 GB.
const HUGE_SOC_D: usize = 100_000;

/// Measured peak (release, `cargo test --release --test memory_budget
/// conic_single_huge_soc_peak_within_budget -- --nocapture`, this machine):
/// 78.4 MB at `d = 100,000` (Optimal, 7 iterations). Budget = measured x3
/// margin (CLAUDE.md convention), rounded to 240 MB. Linear in `d` by
/// construction: the KKT skeleton stores `O(d)` entries for the cone (a
/// diagonal, one dense border column of length `d`, one single-entry
/// column, two corners) instead of the dense representation's `d(d+1)/2`,
/// and the aux columns are pinned after AMD so `L`'s fill stays `O(nnz)`
/// (`kkt::amd_pinned_aux`). The dense alternative's *first* allocation
/// alone (`d(d+1)/2 * 8` = 40 GB) exceeds this budget by ~170x, so any
/// regression to it fails instantly (in practice it aborts the process on
/// the allocation itself -- observed during development when
/// `w2_values_col_major`'s capacity still counted border cones).
const HUGE_SOC_PEAK_BUDGET_BYTES: usize = 240 * 1024 * 1024;

/// `min -x1` s.t. `x0 = 1`, `x in Q_d` via `G = -I`, `h = 0`. Hand-provable
/// optimum: `x1 = 1` on the cone boundary, objective `-1` (same family as
/// `build_many_soc_socp`, one giant block instead of many small ones).
fn build_single_huge_soc_socp(d: usize) -> (ConicProblem, f64) {
    let n = d;
    let idx: Vec<usize> = (0..n).collect();
    let g = CscMatrix::from_triplets(&idx, &idx, &vec![-1.0; n], n, n).expect("huge SOC G");
    let h = vec![0.0; n];
    let mut c = vec![0.0; n];
    c[1] = -1.0;
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).expect("huge SOC A");
    let b = vec![1.0];
    let problem = ConicProblem {
        c,
        a,
        b,
        g,
        h,
        cone: ConeSpec { l: 0, soc: vec![d] },
    };
    (problem, -1.0)
}

/// Peak allocation fence for a single `d = 100,000` second-order cone
/// through the real `solve_socp` entry point -- the Phase 3b
/// rank-1-border KKT representation (`cone::visit_border_pattern`) is the
/// only reason this can run at all: the pre-3b dense `W^2` block for this
/// cone alone would be an 80 GB allocation (see `HUGE_SOC_D`).
///
/// **No-op failure guarantee**: flipping `INJECT_QUADRATIC_BLOWUP` to `true`
/// makes this FAIL (verified manually, same mechanism as the other three
/// routes); independently, lowering `cone::SOC_BORDER_MIN_DIM`'s routing
/// (reverting `visit_w2_pattern`'s border skip) reintroduces the literal
/// `d(d+1)/2`-entry skeleton, which at this `d` aborts on allocation long
/// before the budget assert -- both failure modes are loud.
#[test]
fn conic_single_huge_soc_peak_within_budget() {
    let _serial = lock_measurement();
    let (problem, expected_obj) = build_single_huge_soc_socp(HUGE_SOC_D);

    reset_peak();
    let r = solve_socp(&problem, &ConicOptions::default());
    maybe_inject_dense_blowup(HUGE_SOC_PEAK_BUDGET_BYTES);
    let peak = peak_bytes();
    eprintln!(
        "conic_huge_soc peak={peak} bytes ({:.2} MB), status={:?}, iters={}",
        peak as f64 / 1_048_576.0,
        r.status,
        r.iterations
    );

    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "huge single-SOC route: expected Optimal"
    );
    let rel_err = (r.objective - expected_obj).abs() / expected_obj.abs().max(1.0);
    assert!(
        rel_err < 1e-4,
        "huge single-SOC route: obj={:.6e} expected={:.6e} rel_err={:.3e}",
        r.objective,
        expected_obj,
        rel_err
    );
    assert!(
        peak <= HUGE_SOC_PEAK_BUDGET_BYTES,
        "huge single-SOC route peak {:.1} MB exceeds {:.1} MB budget (d={HUGE_SOC_D})",
        peak as f64 / 1_048_576.0,
        HUGE_SOC_PEAK_BUDGET_BYTES as f64 / 1_048_576.0
    );
}

// ---------------------------------------------------------------------------
// MISOCP route: branch-and-bound over many independent SOC blocks
// ---------------------------------------------------------------------------

use otspot_core::{solve_misocp, BbOptions, MisocpProblem};

/// Padding SOC blocks (dim 3 each), decoupled from the branching subproblem
/// (disjoint rows/columns): `n = 1 (branching var) + 3 * MISOCP_NUM_BLOCKS`,
/// a few-thousand-dimension MISOCP with `nnz = O(n)`.
const MISOCP_NUM_BLOCKS: usize = 2_000;

/// `build_relaxation` (`otspot_core::conic::misocp`) used to densify `base.g`
/// and `base.a` to `Vec<Vec<f64>>` on *every* branch-and-bound node,
/// regardless of how few integer variables were actually being branched on:
/// `O(n*m)` per node, repeated at every node. That is the class this fences.
///
/// Measured peak (release, `cargo test --release --test memory_budget
/// conic_misocp_route_peak_within_budget -- --nocapture`, this machine):
/// 5.95 MB at `n = 6001`, 3 B&B nodes (root + down + up, matching
/// `misocp_exhaustive_search_without_failures_is_optimal`'s node trace).
/// Budget = measured x3 margin (CLAUDE.md convention), rounded up.
///
/// **No-op failure guarantee, verified two ways**: (1) `INJECT_QUADRATIC_BLOWUP`
/// (below) makes this FAIL; (2) unlike the conic SOCP fences above, literally
/// reverting `build_relaxation` to its pre-fix `Vec<Vec<f64>>`-densifying
/// form is runnable at this `n` and was done manually during development:
/// peak jumped to 733.9 MB (vs. this budget's 18.0 MB, and the fixed
/// implementation's own 5.95 MB) -- the pre-fix code densifies `base.g` and
/// `base.a` (each `n x m = 6001 x 6001`, `~288 MB` undropped at once) on
/// every one of the 3 nodes, since every node rebuilds the whole relaxation
/// regardless of how few integer variables are actually being branched on.
const MISOCP_PEAK_BUDGET_BYTES: usize = 18 * 1024 * 1024;

/// `min x + sum_b(-x1_b)` s.t. `x >= 0.5`, integer `x in [0, 2]`, over `n`
/// independent SOC blocks `b`: `x0_b` fixed to `1` by an equality row, cone
/// `x0_b >= sqrt(x1_b^2 + x2_b^2)`. The branching subproblem is exactly
/// `conic::tests::half_int_lp` (root relaxation fractional at `x = 0.5`; the
/// down child `x <= 0` conflicts with `x >= 0.5` and is Farkas-infeasible;
/// the up child `x >= 1` gives the integer optimum `x = 1`) -- already
/// proven correct by `misocp_exhaustive_search_without_failures_is_optimal`.
/// The blocks share no row or column with `x`, so by separability the
/// overall optimum is the sum of the two independent subproblems' optima:
/// `1 - num_blocks` (each block's hand-provable optimum is `-1`, same
/// argument as `build_many_soc_socp` above).
fn build_misocp_with_many_blocks(num_blocks: usize) -> (MisocpProblem, f64) {
    let n = 1 + 3 * num_blocks;
    let m = 1 + 3 * num_blocks;

    // l=1 row: -x <= -0.5  (x >= 0.5), column 0 is the branching variable.
    let mut g_rows = vec![0usize];
    let mut g_cols = vec![0usize];
    let mut g_vals = vec![-1.0];
    let mut h = vec![-0.5];
    for b in 0..num_blocks {
        let base_col = 1 + 3 * b;
        for k in 0..3 {
            g_rows.push(1 + 3 * b + k);
            g_cols.push(base_col + k);
            g_vals.push(-1.0);
            h.push(0.0);
        }
    }
    let g = CscMatrix::from_triplets(&g_rows, &g_cols, &g_vals, m, n).expect("misocp G");

    let mut c = vec![0.0; n];
    c[0] = 1.0;
    let mut a_rows = Vec::with_capacity(num_blocks);
    let mut a_cols = Vec::with_capacity(num_blocks);
    let mut a_vals = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let base_col = 1 + 3 * b;
        c[base_col + 1] = -1.0;
        a_rows.push(b);
        a_cols.push(base_col);
        a_vals.push(1.0);
    }
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, num_blocks, n).expect("misocp A");
    let b_vec = vec![1.0; num_blocks];

    let base = ConicProblem {
        c,
        a,
        b: b_vec,
        g,
        h,
        cone: ConeSpec {
            l: 1,
            soc: vec![3; num_blocks],
        },
    };
    let prob = MisocpProblem {
        base,
        integers: vec![0],
        int_lb: vec![0.0],
        int_ub: vec![2.0],
    };
    (prob, 1.0 - num_blocks as f64)
}

/// Peak allocation fence for the MISOCP branch-and-bound path
/// (`otspot_core::conic::misocp::build_relaxation`): the relaxation rebuilt
/// at every node must stay `O(nnz)`, not `O(n*m)`.
///
/// **No-op failure guarantee**: flipping `INJECT_QUADRATIC_BLOWUP` to `true`
/// makes this FAIL (same mechanism as the other three routes; sized to `2x`
/// this route's budget).
#[test]
fn conic_misocp_route_peak_within_budget() {
    let _serial = lock_measurement();
    let (problem, expected_obj) = build_misocp_with_many_blocks(MISOCP_NUM_BLOCKS);

    reset_peak();
    let r = solve_misocp(&problem, &ConicOptions::default(), &BbOptions::default());
    maybe_inject_dense_blowup(MISOCP_PEAK_BUDGET_BYTES);
    let peak = peak_bytes();
    eprintln!(
        "misocp_route peak={peak} bytes ({:.2} MB), nodes={}",
        peak as f64 / 1_048_576.0,
        r.nodes
    );

    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "MISOCP route: expected Optimal"
    );
    assert_eq!(r.nodes, 3, "MISOCP route: expected root + down + up nodes");
    let rel_err = (r.objective - expected_obj).abs() / expected_obj.abs().max(1.0);
    assert!(
        rel_err < 1e-4,
        "MISOCP route: obj={:.6e} expected={:.6e} rel_err={:.3e}",
        r.objective,
        expected_obj,
        rel_err
    );
    assert!(
        (r.x[0] - 1.0).abs() < 1e-6,
        "MISOCP route: branching var x[0]={} want 1.0",
        r.x[0]
    );
    assert!(
        peak <= MISOCP_PEAK_BUDGET_BYTES,
        "MISOCP route peak {:.1} MB exceeds {:.1} MB budget (n={}, {MISOCP_NUM_BLOCKS} blocks) — \
         check for an unconditional dense expansion in build_relaxation",
        peak as f64 / 1_048_576.0,
        MISOCP_PEAK_BUDGET_BYTES as f64 / 1_048_576.0,
        1 + 3 * MISOCP_NUM_BLOCKS
    );
}

// ---------------------------------------------------------------------------
// Nonconvex QCQP route: spatial branch-and-bound over a diagonal Hessian
// ---------------------------------------------------------------------------

use otspot_core::conic::{solve_global_qcqp, GlobalOptions, NonconvexQcqp};

/// `collect_pairs` used to allocate a `vec![false; n*n]` bitset once per
/// solve, and `build_relax` used to densify `p0`/`a_eq`/`qc.p`/`g_lin` to
/// `Vec<Vec<f64>>` on *every* branch-and-bound node (`O(n * (n +
/// pairs.len()))` per node). Matches the QPLIB DCQ scale class (n up to
/// ~1e5) at a size where the dense reference (`n^2 * 8` bytes per matrix,
/// ~288 MB at this `n`) is still measurable without aborting the process --
/// same `n` as the MISOCP fence above, for a direct before/after comparison.
const NCQCQP_N: usize = 6_001;

/// Measured peak (release, `cargo test --release --test memory_budget
/// conic_nonconvex_route_peak_within_budget -- --nocapture`, this machine):
/// 31.0 MB at `n = 6001`, 1 B&B node (root only). Budget = measured x3
/// margin (CLAUDE.md convention), rounded up.
///
/// **No-op failure guarantee, verified two ways**: (1) `INJECT_QUADRATIC_BLOWUP`
/// (below) makes this FAIL; (2) reverting `collect_pairs`/`solve_relax_lp`/
/// `build_relax` to their pre-fix dense form was done manually during
/// development (`git stash` the sparse rewrite, keep this test) and run
/// under `systemd-run --user --scope -p MemoryMax=8G`: it was OOM-killed
/// within a minute (journal: "A process of this unit has been killed by the
/// OOM killer") -- `p0`/`a_eq`/`g_lin` densify to `6001 x 6001` (~288 MB
/// each), and `solve_relax_lp`'s own dense pass over the *built* relaxation
/// (`m x nv` ~= 54000 x 12000) is ~5 GB by itself, dwarfing this budget by
/// construction rather than by a measurable finite peak.
const NCQCQP_PEAK_BUDGET_BYTES: usize = 96 * 1024 * 1024;

/// Diagonal strictly-convex objective `sum x_i^2` (`p0` diagonal, `nnz =
/// n`), an equality chain pinning every `x_i` to `0` (`x_0 = 0`, then `x_i -
/// x_{i-1} = 0` for `i >= 1`, `nnz(a_eq) = O(n)`), and a redundant banded
/// inequality `x_i + x_{i+1} <= 10` (`nnz(g_lin) = O(n)`), over box `[-1,
/// 1]^n`. The chain makes `x = 0` the *only* feasible point regardless of
/// the objective; at `x = 0` every diagonal McCormick pair's explicit
/// `w`-interval lower bound is `0` (the box straddles `0`) and the strictly
/// positive objective coefficient on `w_i` drives it to that bound exactly,
/// so the root relaxation is already tight -- hand-provable optimum `0` at
/// `x = 0`, zero spatial branching (a single node).
fn build_diag_nonconvex_qcqp(n: usize) -> (NonconvexQcqp, f64) {
    let mut p0_r = Vec::with_capacity(n);
    let mut p0_c = Vec::with_capacity(n);
    let mut p0_v = Vec::with_capacity(n);
    for i in 0..n {
        p0_r.push(i);
        p0_c.push(i);
        p0_v.push(2.0); // 0.5 * 2 * x_i^2 = x_i^2
    }
    let p0 = CscMatrix::from_triplets(&p0_r, &p0_c, &p0_v, n, n).expect("diag P0");

    let mut ae_r = vec![0usize];
    let mut ae_c = vec![0usize];
    let mut ae_v = vec![1.0];
    for i in 1..n {
        ae_r.push(i);
        ae_c.push(i);
        ae_v.push(1.0);
        ae_r.push(i);
        ae_c.push(i - 1);
        ae_v.push(-1.0);
    }
    let a_eq = CscMatrix::from_triplets(&ae_r, &ae_c, &ae_v, n, n).expect("chain A_eq");
    let b_eq = vec![0.0; n];

    let mut gl_r = Vec::with_capacity(2 * (n - 1));
    let mut gl_c = Vec::with_capacity(2 * (n - 1));
    let mut gl_v = Vec::with_capacity(2 * (n - 1));
    for i in 0..n - 1 {
        gl_r.push(i);
        gl_c.push(i);
        gl_v.push(1.0);
        gl_r.push(i);
        gl_c.push(i + 1);
        gl_v.push(1.0);
    }
    let g_lin = CscMatrix::from_triplets(&gl_r, &gl_c, &gl_v, n - 1, n).expect("banded G_lin");
    let h_lin = vec![10.0; n - 1];

    let qp = NonconvexQcqp {
        n,
        p0: Some(p0),
        q0: vec![0.0; n],
        quad: vec![],
        g_lin,
        h_lin,
        a_eq,
        b_eq,
        lb: vec![-1.0; n],
        ub: vec![1.0; n],
    };
    (qp, 0.0)
}

/// Peak allocation fence for the nonconvex spatial-B&B QCQP path
/// (`otspot_core::conic::nonconvex`): both `collect_pairs` and the per-node
/// `build_relax` must stay `O(nnz)`, not `O(n^2)` / `O(n * (n + pairs))`.
///
/// **No-op failure guarantee**: flipping `INJECT_QUADRATIC_BLOWUP` to `true`
/// makes this FAIL (same mechanism as the other routes; sized to `2x` this
/// route's budget).
#[test]
fn conic_nonconvex_route_peak_within_budget() {
    let _serial = lock_measurement();
    let (qp, expected_obj) = build_diag_nonconvex_qcqp(NCQCQP_N);

    reset_peak();
    let r = solve_global_qcqp(&qp, &ConicOptions::default(), &GlobalOptions::default());
    maybe_inject_dense_blowup(NCQCQP_PEAK_BUDGET_BYTES);
    let peak = peak_bytes();
    eprintln!(
        "nonconvex_route peak={peak} bytes ({:.2} MB), nodes={}",
        peak as f64 / 1_048_576.0,
        r.nodes
    );

    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "nonconvex QCQP route: expected Optimal"
    );
    assert_eq!(
        r.nodes, 1,
        "nonconvex QCQP route: root relaxation must already be tight (no spatial branching)"
    );
    let rel_err = (r.objective - expected_obj).abs() / expected_obj.abs().max(1.0);
    assert!(
        rel_err < 1e-6,
        "nonconvex QCQP route: obj={:.6e} expected={:.6e} rel_err={:.3e}",
        r.objective,
        expected_obj,
        rel_err
    );
    for (idx, &xi) in r.x.iter().enumerate() {
        assert!(
            xi.abs() < 1e-6,
            "nonconvex QCQP route: x[{idx}]={xi} want 0"
        );
    }
    assert!(
        peak <= NCQCQP_PEAK_BUDGET_BYTES,
        "nonconvex QCQP route peak {:.1} MB exceeds {:.1} MB budget (n={NCQCQP_N}) — \
         check for an unconditional dense expansion in collect_pairs/build_relax",
        peak as f64 / 1_048_576.0,
        NCQCQP_PEAK_BUDGET_BYTES as f64 / 1_048_576.0
    );
}
