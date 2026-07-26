//! Shared fixpoint control for presolve drivers.
//!
//! LP and QP presolve each own their transform mathematics in `otspot-core`
//! (the domain crate); this module owns only the control semantics they
//! share: run passes until stable, interrupted, or the pass limit is hit.

use crate::kkt::SolveControl;

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
#[must_use = "the returned PipelineStop tells the caller why the fixpoint loop ended (e.g. Interrupted must discard partial work)"]
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

/// Run a single sub-step of a multi-step pass, skipping it if `control`
/// already signals interruption.
///
/// `run_fixpoint` checks `control` once per pass. A pass built from several
/// independent sub-steps (e.g. presolve transforms run in sequence) may want
/// to stop before finishing all of them once the deadline/cancel token
/// fires, rather than waiting for the next pass boundary. `run_step` gives
/// each sub-step that finer-grained check without re-deriving the
/// deadline/cancel test at every call site.
///
/// Returns `Ok(true)` when `step` ran, `Ok(false)` when it was skipped
/// because `control` was already interrupted.
#[must_use = "Ok(false) means the step was skipped; ignoring it silently continues as if it ran"]
pub fn run_step<E>(
    control: SolveControl<'_>,
    step: impl FnOnce() -> Result<(), E>,
) -> Result<bool, E> {
    if control.check().is_err() {
        return Ok(false);
    }
    step()?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

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

    #[test]
    fn run_step_executes_when_not_interrupted() {
        let mut calls = 0;
        let ran = run_step(SolveControl::default(), || {
            calls += 1;
            Ok::<_, ()>(())
        })
        .unwrap();
        assert!(ran, "step must run when control is not interrupted");
        assert_eq!(calls, 1);
    }

    /// Sentinel: an already-cancelled control must skip the step entirely
    /// (the closure body must not execute), not merely report `false` after
    /// running it.
    #[test]
    fn run_step_skips_without_executing_when_interrupted() {
        let cancel = AtomicBool::new(true);
        let control = SolveControl {
            deadline: None,
            cancel: Some(&cancel),
        };
        let mut calls = 0;
        let ran = run_step(control, || {
            calls += 1;
            Ok::<_, ()>(())
        })
        .unwrap();
        assert!(!ran, "step must be skipped once control is interrupted");
        assert_eq!(calls, 0, "interrupted step must not execute its body");
    }

    #[test]
    fn run_step_propagates_step_error() {
        let result = run_step(SolveControl::default(), || Err::<(), _>("boom"));
        assert_eq!(result, Err("boom"));
    }
}
