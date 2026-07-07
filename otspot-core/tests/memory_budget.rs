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

/// `conic::ipm::solve` densifies `A` and `G` (and the KKT system) up front
/// regardless of sparsity (see `csc_to_dense` calls at the top of `solve`),
/// so this route is currently O((n+m)^2) by construction, not O(nnz) — that
/// densification is a separate, already-tracked issue (the sparse-KKT
/// alternative diverges at small n per commit 09dba4cf), not something this
/// fence should paper over. `NUM_BLOCKS` is deliberately small: this is a
/// regression fence pinned to the *current* dense behavior (catch further
/// blow-ups, e.g. an accidental extra dense copy per iteration), not an
/// O(nnz) fence like the LP/QP routes above.
const NUM_BLOCKS: usize = 800;
const CONIC_N: usize = 3 * NUM_BLOCKS;

/// Measured peak (release, `cargo nextest run --release`, this machine):
/// 226.2 MB at n=2400 (already three-digit MB at this small n — direct
/// evidence the dense path would reach GB scale by n in the low 10^4s, per
/// the module doc comment). Budget = measured x3 margin. Because the path is
/// already dense this is a regression fence against *further* densification
/// (e.g. a redundant dense copy per IPM iteration), not an O(nnz) proof.
const CONIC_PEAK_BUDGET_BYTES: usize = 680 * 1024 * 1024;

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

/// Peak allocation fence for the conic SOCP path — a small-n regression
/// fence, not an O(nnz) fence (see `NUM_BLOCKS` doc comment).
///
/// **No-op failure guarantee**: flipping `INJECT_QUADRATIC_BLOWUP` to `true`
/// makes this FAIL (verified manually: peak jumps to 1360.5 MB against the
/// 680 MB budget).
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
