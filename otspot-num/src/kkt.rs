//! Common linear/KKT backend contracts.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::{CscMatrixView, NumericError};

/// Deadline and cancellation state passed to every expensive numerical call.
#[derive(Clone, Copy, Default)]
pub struct SolveControl<'a> {
    pub deadline: Option<Instant>,
    pub cancel: Option<&'a AtomicBool>,
}

impl SolveControl<'_> {
    pub fn check(&self) -> Result<(), NumericError> {
        if self.cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return Err(NumericError::Cancelled);
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(NumericError::DeadlineExceeded);
        }
        Ok(())
    }
}

/// A reusable factorization/preconditioner for one linear system.
pub trait LinearSolveFactor {
    fn dimension(&self) -> usize;

    /// Solve into `solution`.  Partial iterates must never be returned as `Ok`.
    fn solve(
        &mut self,
        rhs: &[f64],
        solution: &mut [f64],
        control: SolveControl<'_>,
    ) -> Result<(), NumericError>;
}

/// Backend capable of factorizing a KKT/saddle-point matrix.
pub trait KktBackend<M: CscMatrixView + ?Sized> {
    type Factor: LinearSolveFactor;

    fn factorize(
        &mut self,
        matrix: &M,
        control: SolveControl<'_>,
    ) -> Result<Self::Factor, NumericError>;
}
