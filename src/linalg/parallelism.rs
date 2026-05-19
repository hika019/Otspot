//! SolverOptions::threads を faer `Par` に橋渡しするヘルパ。
//!
//! threads=1 → `Par::Seq` (= 既存挙動完全互換)、
//! threads>=2 → `Par::Rayon(NonZero(threads))` (= faer 内部 supernodal Cholesky / LDL を
//! グローバル rayon thread-pool で並列化)。
//!
//! 単発 LP/QP solve の per-call parallelism 配線点。
//! multistart 並列とは独立 (multistart 中は各 inner solve に threads=1 が渡る)。

use faer::Par;
use std::num::NonZeroUsize;

/// `SolverOptions::threads` を faer `Par` に変換する。
///
/// - `threads == 0` または `1` → `Par::Seq` (シリアル)
/// - `threads >= 2`           → `Par::Rayon(threads)` (rayon 並列)
///
/// `threads == 0` は input sanitization 用 (内部で 1 に補正)。
/// faer の `Par::Rayon` は `NonZeroUsize` を要求するため、ここで安全に変換する。
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
}
