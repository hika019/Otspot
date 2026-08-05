//! Exception hierarchy mirroring `otspot_model::ModelError`'s variants.
//!
//! Rust returns `Result<_, ModelError>`; Python has no `Result` idiom, so each
//! `ModelError` variant maps to a distinct exception type instead of a single
//! generic error, preserving the ability to `except` on a specific failure
//! mode the way a Rust caller would `match` on the variant.
//!
//! `ModelError::NotSupported` has no Python exception: it is only produced by
//! the QCQP/SOCP bridge (`Model::add_qc_le`/`add_soc_le`), which this crate
//! does not bind (see api_manifest.json's `out_of_scope`) — the variant is
//! unreachable from any bound Python call, so no `NotSupportedError` type is
//! registered (a decorative, unraiseable exception is worse than none).

use otspot_model::ModelError;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use crate::enums::PySolveError;

create_exception!(otspot, OtspotError, PyException);
create_exception!(otspot, NoObjectiveError, OtspotError);
create_exception!(otspot, InvalidInputError, OtspotError);
create_exception!(otspot, SolveFailedError, OtspotError);
create_exception!(otspot, TimeoutError, OtspotError);
create_exception!(otspot, NonConvexError, OtspotError);
create_exception!(otspot, InternalError, OtspotError);

/// Converts a `ModelError` into the matching Python exception, preserving the
/// Display message. `ModelError::SolveError` additionally attaches an
/// `.error: otspot.SolveError` attribute holding the real inner variant (set
/// via `setattr` post-construction rather than a `#[pyclass(extends =
/// PyException)]` field: subclassing a native exception with extra fields
/// needs `Py_3_12` under the `abi3` feature — see PyO3's exception guide —
/// but this crate targets `abi3-py311`).
pub(crate) fn model_error_to_pyerr(py: Python<'_>, err: ModelError) -> PyErr {
    match &err {
        ModelError::NoObjective => NoObjectiveError::new_err(err.to_string()),
        ModelError::InvalidInput(_) => InvalidInputError::new_err(err.to_string()),
        ModelError::SolveError(e) => {
            let py_err = SolveFailedError::new_err(err.to_string());
            let solve_error: PySolveError = e.clone().into();
            if let Err(attach_err) = py_err.value(py).setattr("error", solve_error) {
                return attach_err;
            }
            py_err
        }
        ModelError::Timeout => TimeoutError::new_err(err.to_string()),
        ModelError::NonConvex(_) => NonConvexError::new_err(err.to_string()),
        ModelError::NotSupported(_) => {
            // Unreachable from any bound Python call (see module doc comment);
            // still handled explicitly rather than folded into the wildcard so
            // a future QCQP binding is forced to reconsider this arm.
            InternalError::new_err(format!(
                "unreachable: otspot-py does not bind the QCQP/SOCP entry points \
                 that produce NotSupported ({err})"
            ))
        }
        ModelError::Internal(_) => InternalError::new_err(err.to_string()),
        // #[non_exhaustive]: wildcard required for cross-crate matching.
        _ => InternalError::new_err(err.to_string()),
    }
}

pub(crate) fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    m.add("OtspotError", m.py().get_type::<OtspotError>())?;
    m.add("NoObjectiveError", m.py().get_type::<NoObjectiveError>())?;
    m.add("InvalidInputError", m.py().get_type::<InvalidInputError>())?;
    m.add("SolveFailedError", m.py().get_type::<SolveFailedError>())?;
    m.add("TimeoutError", m.py().get_type::<TimeoutError>())?;
    m.add("NonConvexError", m.py().get_type::<NonConvexError>())?;
    m.add("InternalError", m.py().get_type::<InternalError>())?;
    Ok(())
}
