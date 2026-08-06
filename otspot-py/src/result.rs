//! `ModelResult` binding: mirrors `otspot_model::ModelResult`.

use otspot_model::ModelResult;
use pyo3::prelude::*;

use crate::enums::{PySolutionProof, PySolveStatus};
use crate::errors::model_error_to_pyerr;
use crate::variable::PyVariable;

#[pyclass(module = "otspot", name = "ModelResult")]
pub struct PyModelResult(pub(crate) ModelResult);

#[pymethods]
impl PyModelResult {
    #[getter]
    fn status(&self) -> PySolveStatus {
        self.0.status.clone().into()
    }

    #[getter]
    fn proof(&self) -> PySolutionProof {
        self.0.proof.into()
    }

    #[getter]
    fn objective_value(&self) -> f64 {
        self.0.objective_value
    }

    #[getter]
    fn dual_solution(&self) -> Option<Vec<f64>> {
        self.0.dual_solution.clone()
    }

    #[getter]
    fn reduced_costs(&self) -> Option<Vec<f64>> {
        self.0.reduced_costs.clone()
    }

    #[getter]
    fn slack(&self) -> Option<Vec<f64>> {
        self.0.slack.clone()
    }

    #[getter]
    fn bound_duals(&self) -> Vec<f64> {
        self.0.bound_duals.clone()
    }

    /// Matches `ModelResult::objective`.
    fn objective(&self) -> f64 {
        self.0.objective()
    }

    /// Matches `ModelResult::value`. Uses `ModelResult::try_value`
    /// internally and raises `otspot.InvalidInputError` on misuse (e.g. a
    /// `Variable` from a different `Model`) instead of calling the panicking
    /// `value` directly — see `Model.var_name`'s doc comment for why a raw
    /// Rust panic is not an acceptable Python-facing failure mode.
    fn value(&self, var: PyVariable, py: Python<'_>) -> PyResult<f64> {
        self.0
            .try_value(var.0)
            .map_err(|e| model_error_to_pyerr(py, e))
    }

    /// Matches `ModelResult::has_global_optimality_proof`.
    fn has_global_optimality_proof(&self) -> bool {
        self.0.has_global_optimality_proof()
    }

    /// Matches `Index<Variable> for ModelResult` (`result[x]` in Rust). Same
    /// `try_value`-based panic avoidance as `value`.
    fn __getitem__(&self, var: PyVariable, py: Python<'_>) -> PyResult<f64> {
        self.0
            .try_value(var.0)
            .map_err(|e| model_error_to_pyerr(py, e))
    }
}

pub(crate) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyModelResult>()?;
    Ok(())
}
