//! タイムアウト + キャンセル管理
//!
//! `TimeoutCtx` は全ソルバーで共通のタイムアウト管理ヘルパー。

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

/// タイムアウト + キャンセルを一元管理するヘルパー
pub struct TimeoutCtx {
    pub deadline: Option<Instant>,
    pub cancel: Arc<AtomicBool>,
}

impl TimeoutCtx {
    /// Construct from solver-independent primitives.
    pub fn new(
        deadline: Option<Instant>,
        timeout_secs: Option<f64>,
        cancel: Option<Arc<AtomicBool>>,
    ) -> Self {
        let deadline = deadline.or_else(|| {
            timeout_secs.map(|seconds| Instant::now() + Duration::from_secs_f64(seconds))
        });
        let cancel = cancel.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        Self { deadline, cancel }
    }

    #[inline]
    pub fn should_stop(&self) -> bool {
        self.cancel.load(Ordering::Relaxed) || self.deadline.is_some_and(|d| Instant::now() >= d)
    }
}

/// Returns `true` if `deadline` is set and the current time has passed it.
#[inline]
pub fn deadline_reached(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() >= d)
}
