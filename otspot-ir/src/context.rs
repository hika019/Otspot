//! Cross-cutting solve controls.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use otspot_num::SolveControl;

#[derive(Debug, Clone, Default)]
pub struct SolveContext {
    deadline: Option<Instant>,
    cancel: Option<Arc<AtomicBool>>,
}

impl SolveContext {
    pub fn with_deadline(deadline: Instant) -> Self {
        Self {
            deadline: Some(deadline),
            cancel: None,
        }
    }

    pub fn with_timeout(timeout: Duration) -> Self {
        Self::with_deadline(Instant::now() + timeout)
    }

    pub fn with_cancel_flag(mut self, cancel: Arc<AtomicBool>) -> Self {
        self.cancel = Some(cancel);
        self
    }

    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    pub fn control(&self) -> SolveControl<'_> {
        SolveControl {
            deadline: self.deadline,
            cancel: self.cancel.as_deref(),
        }
    }
}
