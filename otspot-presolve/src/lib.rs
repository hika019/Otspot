//! Shared presolve orchestration.
//!
//! Transform mathematics stays in the LP/QP domain crates; this crate owns the
//! common fixpoint, pass-limit, deadline, and cancellation semantics.

use otspot_num::SolveControl;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStop {
    Stable,
    PassLimit,
    Interrupted,
}

/// Run transform passes until stable, interrupted, or the pass limit is hit.
///
/// The callback returns `true` when the pass changed state. Errors remain
/// domain-specific and are propagated unchanged.
pub fn run_fixpoint<E>(
    max_passes: usize,
    control: SolveControl<'_>,
    mut pass: impl FnMut(usize) -> Result<bool, E>,
) -> Result<PipelineStop, E> {
    for index in 0..max_passes {
        if control.check().is_err() {
            return Ok(PipelineStop::Interrupted);
        }
        if !pass(index)? {
            return Ok(PipelineStop::Stable);
        }
    }
    Ok(PipelineStop::PassLimit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_at_fixpoint() {
        let mut calls = 0;
        let stop = run_fixpoint(10, SolveControl::default(), |_| {
            calls += 1;
            Ok::<_, ()>(calls < 3)
        })
        .unwrap();
        assert_eq!(stop, PipelineStop::Stable);
        assert_eq!(calls, 3);
    }

    #[test]
    fn honors_pass_limit() {
        let stop = run_fixpoint(2, SolveControl::default(), |_| Ok::<_, ()>(true)).unwrap();
        assert_eq!(stop, PipelineStop::PassLimit);
    }
}
