//! Exception hierarchy mirroring `otspot_model::ModelError`'s variants.
//!
//! Rust returns `Result<_, ModelError>`; Python has no `Result` idiom, so each
//! `ModelError` variant maps to a distinct exception type instead of a single
//! generic error, preserving the ability to `except` on a specific failure
//! mode the way a Rust caller would `match` on the variant.

use otspot_model::{ModelError, SolveError};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(otspot, OtspotError, PyException);
create_exception!(otspot, NoObjectiveError, OtspotError);
create_exception!(otspot, InvalidInputError, OtspotError);
create_exception!(otspot, SolveFailedError, OtspotError);
create_exception!(otspot, TimeoutError, OtspotError);
create_exception!(otspot, NonConvexError, OtspotError);
create_exception!(otspot, NotSupportedError, OtspotError);
create_exception!(otspot, InternalError, OtspotError);

fn solve_error_name(e: &SolveError) -> &'static str {
    match e {
        SolveError::Infeasible => "Infeasible",
        SolveError::Unbounded => "Unbounded",
        SolveError::MaxIterations => "MaxIterations",
        SolveError::Stalled => "Stalled",
        SolveError::NumericalError => "NumericalError",
        // `SolveError` is `#[non_exhaustive]`; a wildcard is required for
        // cross-crate matching, matching otspot_model's own convention.
        _ => "Unknown",
    }
}

/// Converts a `ModelError` into the matching Python exception, preserving the
/// Display message and (for `SolveError`) the inner variant name.
pub(crate) fn model_error_to_pyerr(err: ModelError) -> PyErr {
    match &err {
        ModelError::NoObjective => NoObjectiveError::new_err(err.to_string()),
        ModelError::InvalidInput(_) => InvalidInputError::new_err(err.to_string()),
        ModelError::SolveError(e) => {
            SolveFailedError::new_err(format!("{} [{}]", err, solve_error_name(e)))
        }
        ModelError::Timeout => TimeoutError::new_err(err.to_string()),
        ModelError::NonConvex(_) => NonConvexError::new_err(err.to_string()),
        ModelError::NotSupported(_) => NotSupportedError::new_err(err.to_string()),
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
    m.add("NotSupportedError", m.py().get_type::<NotSupportedError>())?;
    m.add("InternalError", m.py().get_type::<InternalError>())?;
    Ok(())
}
