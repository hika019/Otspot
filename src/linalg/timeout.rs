//! タイムアウト + キャンセル管理
//!
//! `TimeoutCtx` は AS/IPM 全ソルバーで共通のタイムアウト管理ヘルパー。
//! `SolverOptions` から一度だけ構築し、各ソルバーループで `should_stop()` を呼ぶ。

use crate::options::SolverOptions;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

/// タイムアウト + キャンセルを一元管理するヘルパー
pub(crate) struct TimeoutCtx {
    pub(crate) deadline: Option<Instant>,
    pub(crate) cancel: Arc<AtomicBool>,
}

impl TimeoutCtx {
    /// SolverOptions からコンテキストを構築する（最初の1回のみ呼ぶ）
    pub(crate) fn from_options(opts: &SolverOptions) -> Self {
        let deadline = opts.deadline.or_else(|| {
            opts.timeout_secs
                .map(|s| Instant::now() + Duration::from_secs_f64(s))
        });
        let cancel = opts
            .cancel_flag
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        Self { deadline, cancel }
    }

    #[inline]
    pub(crate) fn should_stop(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
            || self.deadline.is_some_and(|d| Instant::now() >= d)
    }
}
