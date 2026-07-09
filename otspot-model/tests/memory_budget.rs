//! Memory-budget fence for the Model DSL conic bridge (`add_qc_le` /
//! `add_soc_le` -> `solve_qcqp_internal` / `append_soc` in `model.rs`).
//!
//! Before the sparse rewrite, every row handed to the conic solver (linear
//! `<=`/`>=`/`==` constraint rows, variable-bound rows, and SOC rows) was
//! densified into a `Vec<f64>` of length `n` (or `nvar`) before being
//! re-triplet'd into a `CscMatrix`. For a model with `O(n)` sparse
//! constraints this is an `O(n * rows)` allocation instead of `O(nnz)` — the
//! same bug class as the QPLIB conic-bridge OOM (`otspot-core/tests/
//! memory_budget.rs`) and the nonconvex QCQP B&B fix (Task #8), just one
//! layer up in the Model DSL.
//!
//! This fence builds a synthetic sparse conic Model through the public
//! `otspot_model::Model` API only (no internal access), with a closed-form
//! independent-oracle optimum, and asserts the real `model.solve()` call
//! stays within an `O(nnz)` peak-allocation budget.
//!
//! The counting global allocator mirrors `otspot-core/tests/memory_budget.rs`
//! (see that file's module doc for the rationale: process-wide atomics,
//! `#[global_allocator]` scoped to this test binary only, `MEM_TEST_LOCK`
//! serializing the measured window under `cargo test`'s single-process
//! thread model).

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

/// Serializes the build+solve+measure window (see `otspot-core/tests/
/// memory_budget.rs`'s module doc for the full rationale).
static MEM_TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock_measurement() -> MutexGuard<'static, ()> {
    MEM_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn reset_peak() {
    let cur = CURRENT_BYTES.load(Ordering::Relaxed);
    BASELINE_BYTES.store(cur, Ordering::Relaxed);
    PEAK_BYTES.store(cur, Ordering::Relaxed);
}

fn peak_bytes() -> usize {
    PEAK_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(BASELINE_BYTES.load(Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// Detection-power injection (verification aid, always off)
// ---------------------------------------------------------------------------

/// Flip to `true` locally to confirm the budget below actually has teeth: it
/// injects one dense `k x k` allocation inside the measured window, sized to
/// `2x` the budget. Verified manually during development that the test FAILS
/// with this `true` (see the commit message for the measured peak) — must
/// stay `false` in committed code, mirroring `otspot-core/tests/
/// memory_budget.rs`'s `INJECT_QUADRATIC_BLOWUP`.
const INJECT_DENSE_BLOWUP: bool = false;

#[inline]
fn maybe_inject_dense_blowup(budget_bytes: usize) {
    if INJECT_DENSE_BLOWUP {
        let target_bytes = budget_bytes as u128 * 2;
        let k = ((target_bytes / 8) as f64).sqrt().ceil() as usize;
        let bomb: Vec<Vec<f64>> = vec![vec![0.0_f64; k]; k];
        std::hint::black_box(&bomb);
    }
}

// ---------------------------------------------------------------------------
// Synthetic conic-bridge Model: many independent SOC blocks, each with a
// linear Eq/Ge/Le constraint and finite variable bounds, so the fence
// exercises `split_linear_rows`, `bounds_to_ineq_rows`, and `append_soc`
// together at scale.
// ---------------------------------------------------------------------------

use otspot_core::problem::SolveStatus;
use otspot_model::Model;

/// Blocks of 3 vars each: `n = 3 * NUM_BLOCKS`. Matches the "thousands of
/// variables, sparse constraints" scale this fence targets — large enough
/// that an `O(n * rows)` dense revert is unambiguous (rows are themselves
/// `O(NUM_BLOCKS)`, so dense cost is `O(NUM_BLOCKS^2)`), small enough that a
/// manual revert-confirmation run does not thrash the machine.
const NUM_BLOCKS: usize = 1_500;

/// Per block `b`: vars `x0_b in [0, 100]` (cone bound), `x1_b, x2_b in
/// [-10, 10]`.
/// Constraints: `x0_b == 1` (Eq, fixes the cone apex), `x0_b - x1_b >= 0`
/// (Ge), `x1_b + x2_b <= 15` (Le, slack — never binds), and the SOC cone
/// `x0_b >= sqrt(x1_b^2 + x2_b^2)`.
/// Objective: minimize `sum(-x1_b)`.
///
/// Closed-form optimum (hand-derived, independent of the solver): the cone
/// with apex `x0_b = 1` bounds `x1_b <= 1` (at `x2_b = 0`, the norm-maximizing
/// point for a fixed `x1_b`); the `Ge` row gives the same bound
/// (`x1_b <= x0_b = 1`) and is tight there; the bounds (`x1_b <= 10`) and
/// `Le` row (`x1_b + x2_b <= 15`) are slack. So `x1_b = 1, x2_b = 0` per
/// block, objective `= -NUM_BLOCKS`.
fn build_conic_bridge_model(num_blocks: usize) -> (Model, f64) {
    let mut m = Model::new("conic-bridge-memory-fence");
    let mut obj = otspot_model::Expression::from_constant(0.0);
    for _ in 0..num_blocks {
        let x0 = m.add_var("x0", 0.0, 100.0);
        let x1 = m.add_var("x1", -10.0, 10.0);
        let x2 = m.add_var("x2", -10.0, 10.0);

        m.add_constraint((1.0 * x0).eq_constraint(1.0));
        m.add_constraint((1.0 * x0 - 1.0 * x1).geq(0.0));
        m.add_constraint((1.0 * x1 + 1.0 * x2).leq(15.0));
        m.add_soc_le(vec![1.0 * x1, 1.0 * x2], 1.0 * x0);

        obj = obj - 1.0 * x1;
    }
    m.minimize(obj);
    (m, -(num_blocks as f64))
}

/// Measured peak (release, `cargo test -p otspot-model --test memory_budget
/// --release -- --nocapture`, this machine): 12.21 MB / 0.106s at
/// `NUM_BLOCKS = 1500` (`n = 4500`). Budget = measured x3 margin (CLAUDE.md
/// convention). Manually confirmed against the pre-fix dense-row conic
/// bridge (`git stash` on `model.rs` only, same test): 1291.73 MB / 0.582s —
/// a ~106x memory / ~5.5x time regression this fence now catches.
const CONIC_BRIDGE_PEAK_BUDGET_BYTES: usize = 40 * 1024 * 1024;

/// **No-op failure guarantee**: flipping `INJECT_DENSE_BLOWUP` to `true`
/// makes this FAIL (see that const's doc comment).
#[test]
fn model_conic_bridge_peak_within_budget() {
    let _serial = lock_measurement();
    let (mut model, expected_obj) = build_conic_bridge_model(NUM_BLOCKS);

    reset_peak();
    let t0 = std::time::Instant::now();
    let result = model.solve();
    let elapsed = t0.elapsed();
    maybe_inject_dense_blowup(CONIC_BRIDGE_PEAK_BUDGET_BYTES);
    let peak = peak_bytes();
    eprintln!(
        "model_conic_bridge peak={peak} bytes ({:.2} MB), solve_time={:.3}s",
        peak as f64 / 1_048_576.0,
        elapsed.as_secs_f64()
    );

    let r = result.expect("conic bridge memory fence: solve() must succeed");
    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "conic bridge memory fence: expected Optimal"
    );
    let rel_err = (r.objective_value - expected_obj).abs() / expected_obj.abs().max(1.0);
    assert!(
        rel_err < 1e-4,
        "conic bridge memory fence: obj={:.6e} expected={:.6e} rel_err={:.3e}",
        r.objective_value,
        expected_obj,
        rel_err
    );
    assert!(
        peak <= CONIC_BRIDGE_PEAK_BUDGET_BYTES,
        "conic bridge memory fence: peak {:.1} MB exceeds {:.1} MB budget (NUM_BLOCKS={NUM_BLOCKS})",
        peak as f64 / 1_048_576.0,
        CONIC_BRIDGE_PEAK_BUDGET_BYTES as f64 / 1_048_576.0
    );
}
