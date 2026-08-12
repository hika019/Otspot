//! `SolverOptions::threads` を faer `Par` へ橋渡しし、その並列度を専用 rayon
//! プールへ閉じ込めるヘルパ。
//!
//! # `Par` だけでは上限にならない (faer 0.24.4 実測)
//! `Par::Rayon(n)` を渡しても faer は `n` を守らない。内部の `spindle` は
//! `ROOT` 未設定時にグローバル rayon プールへフォールバックする経路を持ち
//! (`spindle::for_each_raw_imp` の `into_par_iter()` 分岐)、そこでの並列度は
//! グローバルプールのサイズ = `available_parallelism()` になる。8 コア機で
//! `threads = 2` を指定した dense QP (n=700, m=40) の solve 中、同時 runnable
//! なスレッド数は **10** に達した (`threads = 8` 指定時と同値)。計測手順は
//! `tests/thread_budget.rs`。
//!
//! したがって上限を成立させるのは [`with_solver_pool`] であり、`Par` の値では
//! ない。プール内では `rayon::current_num_threads() == threads` になるため
//! spindle のフォールバック経路もそのプール上で走り、同じ計測が 10 → 3
//! (ワーカー 2 + サンプラ 1) に下がる。
//!
//! # プールのライフサイクル
//! 所有するプールは **直近の 1 サイズだけ**。同サイズは再利用し、budget 変更時は
//! cache 所有権を新プールへ移す。旧プールは実行中 solve が `Arc` を持つ間のみ
//! 存続し、最後の利用終了で worker を retire する。これにより同一 budget の生成
//! コストを避けつつ、可変 budget の長寿命 process でも worker が累積しない。
//!
//! # MILP との関係
//! MILP 分枝限定法 (`otspot_core::mip::parallel`) はこの経路を使わない。
//! `std::thread::scope` で厳密に `threads` 本のワーカーを持ち、ワーカー内の
//! `SolverOptions::threads` を 1 に落とすので、そちらは存在数・同時実行数の
//! 双方が構造的に上限として保証される。

use faer::Par;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, OnceLock};

/// The one process-owned solver pool. Replacing this entry retires the
/// previous budget after any in-flight [`Arc`] clones finish.
static SOLVER_POOL: OnceLock<Mutex<Option<CachedSolverPool>>> = OnceLock::new();

struct CachedSolverPool {
    threads: usize,
    pool: Arc<rayon::ThreadPool>,
}

/// Run `f` with every rayon-backed task it spawns confined to `threads`
/// workers, handing it the faer [`Par`] that matches that confinement.
///
/// The `Par` is produced here rather than by the caller because the two are
/// one decision: a `Par::Rayon(n)` that is *not* running inside a size-`n`
/// pool does not bound anything (see the module doc). The three cases:
///
/// * `threads <= 1` → `f(Par::Seq)` inline on the calling thread. No pool, no
///   handoff, byte-identical to not calling this at all. This is the default
///   path (`SolverOptions::threads` defaults to 1).
/// * a pool is available → `pool.install(|| f(Par::Rayon(threads)))`.
/// * the pool could not be built → `f(Par::Seq)` inline. **Not** `Par::Rayon`:
///   without the pool that value would send faer straight back to the global
///   pool, i.e. the exact overrun this function exists to prevent. Running
///   sequentially under-uses the budget, which is a slowdown; exceeding it is
///   a broken guarantee, so the fallback takes the slowdown.
pub fn with_solver_pool<R: Send>(threads: usize, f: impl Send + FnOnce(Par) -> R) -> R {
    if threads <= 1 {
        return f(Par::Seq);
    }
    match solver_thread_pool(threads) {
        Some(pool) => pool.install(move || f(solver_par_from_threads(threads))),
        None => f(Par::Seq),
    }
}

/// Returns the process-owned rayon pool of size `threads`, building it on first
/// use or replacing the previously cached budget.
///
/// `None` when the pool could not be built (the OS refused the threads).
/// Callers must then fall back to sequential execution — see
/// [`with_solver_pool`] for why widening to the global pool is not an option.
pub fn solver_thread_pool(threads: usize) -> Option<Arc<rayon::ThreadPool>> {
    let cached = SOLVER_POOL.get_or_init(|| Mutex::new(None));
    let mut guard = cached.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = guard.as_ref() {
        if entry.threads == threads {
            return Some(Arc::clone(&entry.pool));
        }
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .ok()?,
    );
    *guard = Some(CachedSolverPool {
        threads,
        pool: Arc::clone(&pool),
    });
    Some(pool)
}

/// `SolverOptions::threads` を faer `Par` に変換する。
///
/// - `threads == 0` または `1` → `Par::Seq` (シリアル)
/// - `threads >= 2`           → `Par::Rayon(threads)` (rayon 並列)
///
/// `threads == 0` は input sanitization 用 (内部で 1 に補正)。
///
/// **この値だけでは並列度の上限にならない** — 必ず [`with_solver_pool`] が
/// 用意したプールの中で使うこと (理由はモジュール doc)。
pub fn solver_par_from_threads(threads: usize) -> Par {
    let n = threads.max(1);
    if n == 1 {
        Par::Seq
    } else {
        // NonZeroUsize::new は n >= 1 で必ず Some
        Par::Rayon(NonZeroUsize::new(n).expect("n >= 1 guaranteed by max(1)"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threads_zero_yields_seq() {
        assert_eq!(solver_par_from_threads(0), Par::Seq);
    }

    #[test]
    fn threads_one_yields_seq() {
        assert_eq!(solver_par_from_threads(1), Par::Seq);
    }

    #[test]
    fn threads_n_yields_rayon_n() {
        for n in [2usize, 4, 8, 16, 64] {
            let par = solver_par_from_threads(n);
            match par {
                Par::Rayon(k) => assert_eq!(k.get(), n, "threads={n}"),
                Par::Seq => panic!("threads={n} should yield Rayon, got Seq"),
            }
        }
    }

    /// `threads <= 1` must not touch a pool at all: the closure runs on the
    /// caller's own thread and is told to stay sequential.
    #[test]
    fn budget_of_one_runs_inline_and_sequential() {
        let caller = std::thread::current().id();
        for threads in [0usize, 1] {
            let (par, tid) = with_solver_pool(threads, |par| (par, std::thread::current().id()));
            assert_eq!(par, Par::Seq, "threads={threads}");
            assert_eq!(tid, caller, "threads={threads} must not hand off");
        }
    }

    /// Inside the pool the closure sees exactly the budget it asked for, both
    /// as the faer `Par` and as rayon's own view of the pool it is running in.
    #[test]
    fn pooled_run_sees_the_requested_width() {
        for threads in [2usize, 3] {
            let (par, width) = with_solver_pool(threads, |par| (par, rayon::current_num_threads()));
            match par {
                Par::Rayon(k) => assert_eq!(k.get(), threads),
                Par::Seq => panic!("threads={threads} should be pooled"),
            }
            assert_eq!(
                width, threads,
                "rayon must see the dedicated pool, not the global one"
            );
        }
    }

    /// The same budget reuses one pool; changing budget replaces the sole
    /// cache-owned pool and retires the old one after in-flight users finish.
    ///
    /// Sentinel: reverting to the leaked per-budget HashMap leaves `old`
    /// upgradeable after every local strong reference is dropped.
    #[test]
    fn changing_budget_retires_the_previous_cached_pool() {
        let a = solver_thread_pool(5).expect("pool");
        let b = solver_thread_pool(5).expect("pool");
        assert!(
            Arc::ptr_eq(&a, &b),
            "the same budget must reuse one cached pool"
        );
        let old = Arc::downgrade(&a);
        let other = solver_thread_pool(6).expect("pool");
        assert!(
            !Arc::ptr_eq(&a, &other),
            "different budgets must not share a pool"
        );
        assert_eq!(a.current_num_threads(), 5);
        assert_eq!(other.current_num_threads(), 6);
        drop(a);
        drop(b);
        assert!(
            old.upgrade().is_none(),
            "the previous pool must retire once its in-flight users finish"
        );
    }
}
