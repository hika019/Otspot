//! `Constraint` binding: mirrors `otspot_model::Constraint`.
//!
//! Opaque by design — Rust's `Constraint` fields are `pub(crate)` to
//! `otspot_model`, so the only public surface is construction (via
//! `Expression::leq/geq/eq_constraint`, see `expr.rs`) and passing the
//! result to `Model.add_constraint`.

use otspot_model::Constraint;
use pyo3::prelude::*;

#[pyclass(module = "otspot", name = "Constraint", from_py_object)]
#[derive(Clone)]
pub struct PyConstraint(pub(crate) Constraint);

#[pymethods]
impl PyConstraint {
    fn __repr__(&self) -> String {
        "Constraint(...)".to_string()
    }
}

pub(crate) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyConstraint>()?;
    Ok(())
}
