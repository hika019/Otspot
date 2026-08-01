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

use crate::constraint::PyConstraint;
use crate::enums::{PyTolerance, PyVarKind};
use crate::errors::model_error_to_pyerr;
use crate::expr::{coerce, Operand};
use crate::result::PyModelResult;
use crate::variable::PyVariable;

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
    /// Runs under `Python::detach` (GIL released): the underlying solve can
    /// run for the full `timeout_secs` budget (default: unbounded), and
    /// holding the GIL for that whole span would freeze every other Python
    /// thread in the process (including the one running `KeyboardInterrupt`
    /// delivery) for the duration.
    fn solve(&mut self, py: Python<'_>) -> PyResult<PyModelResult> {
        let model = &mut self.0;
        py.detach(|| model.solve())
            .map(PyModelResult)
            .map_err(|e| model_error_to_pyerr(py, e))
    }
}

pub(crate) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    Ok(())
}
