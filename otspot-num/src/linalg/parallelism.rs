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
//! プールはサイズごとにプロセス内へキャッシュされ、以後の solve が
//! 再利用する。したがって **同じサイズの初回 solve だけが `threads` 本を新規
//! 生成する**。2 回目以降の生成コストはゼロで、MIQP のノードごと QP 解のように
//! 高頻度に呼ばれる経路でも問題にならない。
//!
//! # MILP との関係
//! MILP 分枝限定法 (`otspot_core::mip::parallel`) はこの経路を使わない。
//! `std::thread::scope` で厳密に `threads` 本のワーカーを持ち、ワーカー内の
//! `SolverOptions::threads` を 1 に落とすので、そちらは存在数・同時実行数の
//! 双方が構造的に上限として保証される。

use faer::Par;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::{Mutex, OnceLock};

/// Dedicated rayon pools, one per distinct thread budget, reused for the
/// lifetime of the process.
///
/// The pools are **leaked** (`Box::leak`), so their threads live until the
/// process exits and never appear in a leak checker's "freed" column. That is
/// deliberate and bounded: an entry is created only for a thread budget that
/// was actually requested, so the footprint is
/// `sum over distinct budgets n of n threads` — in practice one entry, since a
/// program picks a thread count once. Leaking buys a `&'static ThreadPool`
/// that any solve can `install` into without refcount traffic, and avoids the
/// alternative of tearing a pool down and rebuilding it per solve.
static SOLVER_POOLS: OnceLock<Mutex<HashMap<usize, &'static rayon::ThreadPool>>> = OnceLock::new();

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

/// The process-wide rayon pool of size `threads`, building it on first use.
///
/// `None` when the pool could not be built (the OS refused the threads).
/// Callers must then fall back to sequential execution — see
/// [`with_solver_pool`] for why widening to the global pool is not an option.
pub fn solver_thread_pool(threads: usize) -> Option<&'static rayon::ThreadPool> {
    let pools = SOLVER_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = pools.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&pool) = guard.get(&threads) {
        return Some(pool);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok()?;
    let pool: &'static rayon::ThreadPool = Box::leak(Box::new(pool));
    guard.insert(threads, pool);
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

    /// Pools are cached per size: the same budget reuses one pool (so repeated
    /// solves do not respawn threads), and different budgets get different
    /// pools (so one size cannot silently serve another).
    #[test]
    fn pools_are_cached_per_size() {
        let a = solver_thread_pool(5).expect("pool");
        let b = solver_thread_pool(5).expect("pool");
        assert!(
            std::ptr::eq(a, b),
            "the same budget must reuse one cached pool"
        );
        let other = solver_thread_pool(6).expect("pool");
        assert!(
            !std::ptr::eq(a, other),
            "different budgets must not share a pool"
        );
        assert_eq!(a.current_num_threads(), 5);
        assert_eq!(other.current_num_threads(), 6);
    }
}
