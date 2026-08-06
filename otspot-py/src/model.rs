//! `Model` binding: mirrors `otspot_model::Model`'s algebraic modeling API.
//!
//! Setter methods that return `&mut Self` in Rust (for chaining) return
//! `None` in Python instead — PyO3 cannot cheaply hand back a borrowed
//! reference to `self` across the FFI boundary, and Python callers mutate
//! and move on rather than chain. All other names and semantics match Rust.

use otspot_model::{Model, QuadExpr};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::constraint::PyConstraint;
use crate::enums::{PyTolerance, PyVarKind};
use crate::errors::model_error_to_pyerr;
use crate::expr::{coerce, Operand};
use crate::result::PyModelResult;
use crate::variable::PyVariable;

/// Starting interval between `Python::check_signals` polls while `solve()`'s
/// underlying computation runs on its worker thread (see `solve` below).
///
/// A flat 10ms interval here previously put a 10ms latency *floor* under
/// every `solve()` call, regardless of how fast the solve itself was: a
/// 0.2ms LP took 10.2ms end to end (measured), a 60x slowdown, because the
/// first poll always had to wait out a full 10ms sleep before the loop's
/// `is_finished()` check could ever see a solve that had already returned.
/// Starting short and backing off (`SIGNAL_POLL_BACKOFF_FACTOR`, capped at
/// `SIGNAL_POLL_INTERVAL_MAX`) keeps that floor at roughly this constant's
/// order of magnitude for sub-millisecond solves, while a long solve still
/// settles into `SIGNAL_POLL_INTERVAL_MAX`-cadence polling within a few
/// doublings (~13ms of ramp-up), leaving `KeyboardInterrupt` latency for a
/// genuinely long solve essentially unchanged from the flat-interval design.
const SIGNAL_POLL_INTERVAL_INITIAL: Duration = Duration::from_micros(100);

/// Multiplier applied to the poll interval after each `check_signals` call
/// that finds nothing pending, until it reaches `SIGNAL_POLL_INTERVAL_MAX`.
const SIGNAL_POLL_BACKOFF_FACTOR: u32 = 2;

/// Ceiling on the poll interval once backoff has ramped up -- bounds
/// `KeyboardInterrupt` latency for a long solve the same way the old flat
/// interval did (this *is* the old flat interval's value).
const SIGNAL_POLL_INTERVAL_MAX: Duration = Duration::from_millis(10);

#[pyclass(module = "otspot", name = "Model")]
pub struct PyModel(pub(crate) Model);

fn objective_operand(obj: &Bound<'_, PyAny>) -> PyResult<QuadExpr> {
    match coerce(obj) {
        Some(Operand::F(f)) => Ok(QuadExpr::from(f)),
        Some(Operand::V(v)) => Ok(QuadExpr::from(v)),
        Some(Operand::E(e)) => Ok(QuadExpr::from(e)),
        Some(Operand::Q(q)) => Ok(q),
        None => Err(PyTypeError::new_err(
            "objective must be a Variable, Expression, QuadExpr, or number",
        )),
    }
}

#[pymethods]
impl PyModel {
    #[new]
    fn new(name: &str) -> Self {
        PyModel(Model::new(name))
    }

    /// Matches `Model::add_var`.
    fn add_var(&mut self, name: &str, lb: f64, ub: f64) -> PyVariable {
        PyVariable(self.0.add_var(name, lb, ub))
    }

    /// Matches `Model::add_int_var`.
    fn add_int_var(&mut self, name: &str, lb: f64, ub: f64) -> PyVariable {
        PyVariable(self.0.add_int_var(name, lb, ub))
    }

    /// Matches `Model::add_binary_var`.
    fn add_binary_var(&mut self, name: &str) -> PyVariable {
        PyVariable(self.0.add_binary_var(name))
    }

    /// Matches `Model::var_name`. Uses `Model::try_var_name` internally and
    /// raises `otspot.InvalidInputError` on misuse, rather than calling the
    /// panicking `var_name` directly: PyO3 converts an uncaught Rust panic
    /// into `PanicException`, which subclasses `BaseException` (not
    /// `Exception`), so `except Exception` would not catch it.
    fn var_name(&self, var: PyVariable, py: Python<'_>) -> PyResult<String> {
        self.0
            .try_var_name(var.0)
            .map(str::to_string)
            .map_err(|e| model_error_to_pyerr(py, e))
    }

    /// Matches `Model::var_kind`. Same panic-avoidance rationale as `var_name`.
    fn var_kind(&self, var: PyVariable, py: Python<'_>) -> PyResult<PyVarKind> {
        self.0
            .try_var_kind(var.0)
            .map(PyVarKind::from)
            .map_err(|e| model_error_to_pyerr(py, e))
    }

    /// Matches `Model::add_constraint`.
    fn add_constraint(&mut self, c: PyConstraint) {
        self.0.add_constraint(c.0);
    }

    /// Matches `Model::minimize`. Accepts `Variable`, `Expression`,
    /// `QuadExpr`, or a number, mirroring `impl Into<QuadExpr>`.
    fn minimize(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = objective_operand(obj)?;
        self.0.minimize(q);
        Ok(())
    }

    /// Matches `Model::maximize`.
    fn maximize(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = objective_operand(obj)?;
        self.0.maximize(q);
        Ok(())
    }

    /// Matches `Model::set_timeout`.
    fn set_timeout(&mut self, secs: f64) {
        self.0.set_timeout(secs);
    }

    /// Matches `Model::set_tolerance`.
    fn set_tolerance(&mut self, tol: PyTolerance) {
        self.0.set_tolerance(tol.into());
    }

    /// Matches `Model::set_presolve`.
    fn set_presolve(&mut self, flag: bool) {
        self.0.set_presolve(flag);
    }

    /// Matches `Model::set_threads`.
    fn set_threads(&mut self, n: usize) {
        self.0.set_threads(n);
    }

    /// Matches `Model::set_obj_offset`.
    fn set_obj_offset(&mut self, offset: f64) {
        self.0.set_obj_offset(offset);
    }

    /// Matches `Model::solve`. Raises a subclass of `otspot.OtspotError` on
    /// `Err` (see `errors.rs` for the `ModelError` variant -> exception map).
    ///
    /// The actual computation runs on a worker thread that never touches
    /// the GIL (`Model::solve` is pure `otspot-model`/`otspot-core`, no
    /// PyO3 calls anywhere in it) via `std::thread::scope`, while this
    /// (calling) thread polls `Python::check_signals` with an
    /// exponentially-backed-off interval (`SIGNAL_POLL_INTERVAL_INITIAL` up
    /// to `SIGNAL_POLL_INTERVAL_MAX`), releasing the GIL via
    /// `Python::detach` between polls so other Python threads keep making
    /// progress during a long solve (`test_solve_releases_the_gil`).
    ///
    /// The wait between polls uses `Condvar::wait_timeout`, not a plain
    /// `thread::sleep`: the worker calls `notify_one` the instant it
    /// finishes, waking this thread immediately instead of leaving it to
    /// sleep out the rest of whatever the current poll interval had backed
    /// off to (up to `SIGNAL_POLL_INTERVAL_MAX` for a solve that finishes
    /// mid-ramp-up -- a plain-sleep design would add that whole interval as
    /// pure dead latency on top of an already-finished solve). The
    /// completion flag is checked under the same lock before waiting, so a
    /// `notify_one` that lands before this thread starts waiting is not
    /// missed (the classic Mutex+Condvar race).
    ///
    /// This replaces an earlier design that ran the solve directly under
    /// `Python::detach` on the calling thread: releasing the GIL there let
    /// *other* Python threads run, but did nothing for `KeyboardInterrupt`
    /// on a `timeout_secs`-unbounded solve (the default) run from a
    /// script's own main thread -- `Python::check_signals`/the SIGINT
    /// handler only fire when the interpreter's main thread itself gets to
    /// run Python bytecode, and a `detach`-for-the-whole-solve design never
    /// gave it the chance to. Here, a caught signal (SIGINT ->
    /// `KeyboardInterrupt` via Python's default handler) sets a
    /// `Model::set_cancel_flag` cooperative-cancellation flag, which
    /// `otspot-core` honors at the same cadence as the wall-clock
    /// `timeout_secs` deadline (`SolverOptions::cancel_flag`, checked every
    /// simplex/IPM/B&B-node iteration) -- so the worker actually stops
    /// instead of running to completion unseen. The worker's own return
    /// value is discarded once a signal fires: `KeyboardInterrupt` is
    /// authoritative regardless of what status the now-cancelled solve
    /// happened to end on.
    fn solve(&mut self, py: Python<'_>) -> PyResult<PyModelResult> {
        let cancel = Arc::new(AtomicBool::new(false));
        self.0.set_cancel_flag(Arc::clone(&cancel));
        let model = &mut self.0;
        let done = Arc::new((Mutex::new(false), Condvar::new()));
        let done_worker = Arc::clone(&done);

        std::thread::scope(|scope| {
            let handle = scope.spawn(move || {
                let outcome = model.solve();
                let (done_lock, done_cvar) = &*done_worker;
                *done_lock.lock().unwrap() = true;
                done_cvar.notify_one();
                outcome
            });
            let mut poll_interval = SIGNAL_POLL_INTERVAL_INITIAL;
            let (done_lock, done_cvar) = &*done;
            loop {
                if handle.is_finished() {
                    let outcome = match handle.join() {
                        Ok(outcome) => outcome,
                        // Preserve PyO3's normal panic -> PanicException
                        // conversion for the pymethod call as a whole,
                        // rather than inventing a distinct "worker panicked"
                        // error shape.
                        Err(panic) => std::panic::resume_unwind(panic),
                    };
                    return outcome
                        .map(PyModelResult)
                        .map_err(|e| model_error_to_pyerr(py, e));
                }
                py.detach(|| {
                    let guard = done_lock.lock().unwrap();
                    if !*guard {
                        let _ = done_cvar.wait_timeout(guard, poll_interval).unwrap();
                    }
                });
                poll_interval = poll_interval
                    .saturating_mul(SIGNAL_POLL_BACKOFF_FACTOR)
                    .min(SIGNAL_POLL_INTERVAL_MAX);
                if let Err(sig_err) = py.check_signals() {
                    cancel.store(true, Ordering::Relaxed);
                    // Block until the worker actually observes the flag and
                    // returns -- `model` stays mutably borrowed by it until
                    // then, and `std::thread::scope` would block here on
                    // its own implicit join anyway; joining explicitly just
                    // makes that wait visible at the call site. Wrapped in
                    // `py.detach`: cancellation is cooperative, checked at
                    // the same per-iteration cadence as the wall-clock
                    // deadline (see the class doc comment above), so this
                    // join can itself take as long as the worker's next
                    // check-point is away -- without releasing the GIL here
                    // too, every other Python thread stays frozen for that
                    // whole stretch, breaking the same "GIL released for
                    // the duration of solve()" contract the poll loop above
                    // exists to uphold (Codex PR #31 review; lead-confirmed
                    // by direct read).
                    py.detach(|| {
                        let _ = handle.join();
                    });
                    return Err(sig_err);
                }
            }
        })
    }
}

pub(crate) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    Ok(())
}
