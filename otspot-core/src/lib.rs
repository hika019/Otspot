//! # otspot — 数理最適化ソルバー
//!
//! 線形計画法（LP）・二次計画法（QP）と混合整数問題（MILP / MIQP）を解く Rust ソルバークレート。
//! LP は改訂単体法（Revised Simplex）、QP は内点法（IPM / IP-PMM）を核とし、
//! 実行不可能・非有界の判定と完全な主双対情報の出力に対応する。
//!
//! ## 主要モジュール
//!
//! | モジュール | 役割 |
//! |-----------|------|
//! | [`sparse`] | CSC 形式の疎行列・疎ベクトル演算 |
//! | [`problem`] | 問題定義（`LpProblem` / `QpProblem`、`SolveStatus`、`SolverResult`） |
//! | [`lp`] | LP 求解エントリポイント（`solve_lp_with`） |
//! | [`qp`] | 内点法ソルバー（QP、IPM / IP-PMM） |
//! | [`mip`] | 混合整数ソルバー（MILP / MIQP、branch-and-bound） |
//! | [`options`] | `SolverOptions`、`Tolerance` |
//!
//! ## 使用例
//!
//! MPS ファイルから LP 問題を読み込んで解く (via the `otspot` facade):
//!
//! ```rust,ignore
//! use std::path::Path;
//! use otspot::io::mps;
//!
//! let prob = mps::parse_mps_file(Path::new("problem.mps")).expect("MPS読み込み失敗");
//! let result = otspot_core::solve(&prob);
//! println!("最適値: {:?}", result);
//! ```

pub mod error;
pub use error::SolverError;
pub use error::MpsError;
#[doc(hidden)]
pub mod presolve;
pub mod sparse;
pub mod problem;
pub(crate) mod simplex;
// Internal parsers compiled only under cfg(test), used by otspot-core's own
// integration-style tests (e.g. qp::ipm_solver diagnostics). These are
// source-duplicates of otspot-io's canonical, published parsers and are not
// part of the production library. Full removal is tracked separately (the
// ipm_solver tests depend on crate-internal IPM diagnostics).
#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod io;
pub(crate) mod basis;
pub mod tolerances;
pub mod options;
pub use options::{
    BranchingStrategy, DualPricing, GlobalOptimizationConfig, LpWarmStart, MipBranching, MipConfig,
    SolverOptions, Tolerance, WarmStartBasis,
};
pub mod qp;
pub mod mip;
pub mod lp;
#[doc(hidden)]
pub mod linalg;

#[cfg(test)]
pub(crate) mod test_kkt;

/// Thread-local peak-allocation tracker for memory sentinel tests.
///
/// Wraps the system allocator and records, per-thread, the maximum net bytes
/// concurrently live above a caller-defined baseline.  Using thread-local
/// storage means tests running in parallel on different threads do not
/// interfere with each other's measurements.
///
/// Usage in a test:
/// ```ignore
/// crate::peak_alloc::begin();
/// do_heavy_work();
/// let peak = crate::peak_alloc::peak_bytes();
/// assert!(peak <= MAX_BYTES, "peak {peak} exceeds limit");
/// ```
#[cfg(test)]
pub(crate) mod peak_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        /// Net bytes allocated on this thread (alloc - dealloc).
        static CURRENT: Cell<isize> = const { Cell::new(0) };
        /// Baseline captured by `begin()`; delta is measured above this.
        static BASELINE: Cell<isize> = const { Cell::new(0) };
        /// Maximum (CURRENT - BASELINE) observed since last `begin()`.
        static PEAK_DELTA: Cell<isize> = const { Cell::new(0) };
    }

    /// Record the current allocation level as the baseline for this thread.
    pub fn begin() {
        CURRENT.with(|c| BASELINE.with(|b| b.set(c.get())));
        PEAK_DELTA.with(|p| p.set(0));
    }

    /// Peak bytes allocated above the baseline captured by `begin()`.
    pub fn peak_bytes() -> usize {
        PEAK_DELTA.with(|p| p.get().max(0) as usize)
    }

    /// Net live bytes above the baseline captured by `begin()`, right now
    /// (not the peak). Used by accumulation sentinels to assert that memory
    /// returns to baseline after each unit of work is dropped.
    pub fn current_bytes() -> isize {
        CURRENT.with(|c| c.get()) - BASELINE.with(|b| b.get())
    }

    #[inline]
    fn update(delta: isize) {
        CURRENT.with(|c| {
            let new = c.get() + delta;
            c.set(new);
            let above = new - BASELINE.with(|b| b.get());
            PEAK_DELTA.with(|p| {
                if above > p.get() {
                    p.set(above);
                }
            });
        });
    }

    pub struct TrackingAlloc;

    unsafe impl GlobalAlloc for TrackingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = System.alloc(layout);
            if !ptr.is_null() {
                update(layout.size() as isize);
            }
            ptr
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let ptr = System.alloc_zeroed(layout);
            if !ptr.is_null() {
                update(layout.size() as isize);
            }
            ptr
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);
            update(-(layout.size() as isize));
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let new_ptr = System.realloc(ptr, layout, new_size);
            if !new_ptr.is_null() {
                update(new_size as isize - layout.size() as isize);
            }
            new_ptr
        }
    }
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOC: peak_alloc::TrackingAlloc = peak_alloc::TrackingAlloc;

// --- re-export: ユーザーが最も使う型を最短パスで ---
pub use sparse::CscMatrix;
pub use problem::{SolveRoute, SolveStats, SolveStatus};
pub use problem::certificate::{
    FarkasCertificate, IncompleteReason, NotProven, OptimalCertificate,
    SolveOutcome, UnboundedRayCertificate,
};
pub use qp::certificate::prove_optimal;
pub use qp::{solve_qp, solve_qp_global, solve_qp_with, QpProblem, SolverResult, QpWarmStart};
pub use mip::{
    solve_milp, solve_milp_with_stats, solve_miqp, solve_miqp_with_stats, MilpProblem,
    MipProblemError, MipStats, MiqpProblem,
};
pub use lp::solve_lp_with;
pub use simplex::{solve, solve_with};

/// Internal BFRT (Bound-Flipping Ratio Test) primitives for integration tests.
/// Deferred for removal until typed pipeline (#15) restructures the simplex tree.
#[doc(hidden)]
pub mod bound_flip {
    pub use crate::simplex::dual_advanced::bound_flip::{
        bfrt_flip_invocations, bfrt_select_entering, reset_bfrt_flip_invocations,
        BfrtResult, ColBound,
    };
}
pub use qp::{diagnose, DiagnosticReport, DiagnosticWarning, DiagnosticCode, Severity, ProblemInfo};

/// RAII guard that disables a production sentinel for the duration of its lifetime.
///
/// On construction: calls `enable` to disable the sentinel.
/// On drop: calls `restore` to re-enable the sentinel.
/// Panic-safe: `restore` runs even if the guarded closure panics.
pub(crate) struct ScopedDisable<D: Fn()> {
    restore: D,
}

impl<D: Fn()> ScopedDisable<D> {
    pub(crate) fn new<E: Fn()>(enable: E, restore: D) -> Self {
        enable();
        ScopedDisable { restore }
    }
}

impl<D: Fn()> Drop for ScopedDisable<D> {
    fn drop(&mut self) {
        (self.restore)();
    }
}

/// Apply the LP KKT optimality guard to a solver result.
///
/// Exposed for integration-test sentinel load-bearing proofs. Runs full
/// KKT+dual_sign verification via `prove_optimal_lp`; demotes false-Optimal
/// to `SuboptimalSolution`. Non-Optimal results pass through unchanged.
#[doc(hidden)]
pub fn apply_lp_primal_guard(
    result: crate::problem::SolverResult,
    problem: &crate::problem::LpProblem,
) -> crate::problem::SolverResult {
    crate::qp::certificate::guard_lp_optimal(result, problem)
}
