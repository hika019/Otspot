//! Python bindings for otspot (PyO3 / maturin).
//!
//! Wraps `otspot_model`'s algebraic modeling API (`Model`, `Variable`,
//! `Expression`, `QuadExpr`, `Constraint`) and its result/error/status types.
//! Every binding is a thin call-through to the real `otspot_model` /
//! `otspot_core` function or operator — see `api_manifest.json` for the
//! tracked Rust<->Python symbol mapping and `tests/api_manifest_rust.rs` /
//! the Python `tests/test_api_manifest.py` for the parity checks.

mod constraint;
mod enums;
mod errors;
mod expr;
mod model;
mod result;
mod variable;

use pyo3::prelude::*;

#[pymodule]
fn otspot(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    model::register(m)?;
    variable::register(m)?;
    expr::register(m)?;
    constraint::register(m)?;
    result::register(m)?;
    enums::register(m)?;
    errors::register(m)?;
    Ok(())
}
