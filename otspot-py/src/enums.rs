//! Enum types mirroring `otspot_model`/`otspot_core` variant-for-variant.
//!
//! Variant names are kept identical to Rust (`Continuous`, not `CONTINUOUS`)
//! so the parity manifest can assert a literal name match on both sides.
//!
//! `ConstraintSense` is not bound: `otspot_model::Constraint`'s fields
//! (including `sense`) are `pub(crate)` with no public getter anywhere in the
//! Rust API (unlike `Variable`'s kind, exposed via `Model::var_kind`), so a
//! Python `ConstraintSense` binding would be a decorative type nothing ever
//! produces or consumes. Re-add it if/when otspot-model grows a constraint
//! introspection API (see api_manifest.json's `out_of_scope`).

use otspot_core::options::Tolerance;
use otspot_core::problem::SolveStatus;
use otspot_model::{SolutionProof, SolveError, VarKind};
use pyo3::prelude::*;

#[pyclass(module = "otspot", name = "VarKind", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyVarKind {
    Continuous,
    Integer,
    Binary,
}

impl From<VarKind> for PyVarKind {
    fn from(k: VarKind) -> Self {
        match k {
            VarKind::Continuous => PyVarKind::Continuous,
            VarKind::Integer => PyVarKind::Integer,
            VarKind::Binary => PyVarKind::Binary,
        }
    }
}

#[pyclass(module = "otspot", name = "SolutionProof", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PySolutionProof {
    GlobalOptimal,
    LocalOptimal,
    FeasibleUnproven,
}

impl From<SolutionProof> for PySolutionProof {
    fn from(p: SolutionProof) -> Self {
        match p {
            SolutionProof::GlobalOptimal => PySolutionProof::GlobalOptimal,
            SolutionProof::LocalOptimal => PySolutionProof::LocalOptimal,
            SolutionProof::FeasibleUnproven => PySolutionProof::FeasibleUnproven,
            // `SolutionProof` is `#[non_exhaustive]`, so a wildcard is
            // mandatory to compile. Mapping an unknown future variant to an
            // arbitrary known one would silently misreport optimality
            // strength to callers branching on it — panic loudly instead
            // (`eq_int` variants carry no payload, so there is no honest
            // "Unknown" value to return; see `PySolveStatus` below for the
            // complex-enum case, which can carry one).
            _ => panic!(
                "otspot_core::problem::SolutionProof gained a variant unhandled by \
                 otspot-py/src/enums.rs; update this From impl and api_manifest.json"
            ),
        }
    }
}

#[pyclass(module = "otspot", name = "SolveError", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PySolveError {
    Infeasible,
    Unbounded,
    MaxIterations,
    Stalled,
    NumericalError,
}

impl From<SolveError> for PySolveError {
    fn from(e: SolveError) -> Self {
        match e {
            SolveError::Infeasible => PySolveError::Infeasible,
            SolveError::Unbounded => PySolveError::Unbounded,
            SolveError::MaxIterations => PySolveError::MaxIterations,
            SolveError::Stalled => PySolveError::Stalled,
            SolveError::NumericalError => PySolveError::NumericalError,
            // See `PySolutionProof::from`'s wildcard comment above.
            _ => panic!(
                "otspot_model::SolveError gained a variant unhandled by \
                 otspot-py/src/enums.rs; update this From impl and api_manifest.json"
            ),
        }
    }
}

/// `SolveStatus` has payload-carrying variants (`NonConvex`, `NotSupported`),
/// so it is bound as a PyO3 "complex enum" rather than a C-like `eq_int` enum:
/// each variant becomes a Python subclass, exactly mirroring the Rust shape.
///
/// `Unknown(String)` is not a real `SolveStatus` variant: it is the honest
/// fallback for the `#[non_exhaustive]` wildcard this `From` impl is forced
/// to have (unlike `PySolutionProof`/`PySolveError`, a complex enum *can*
/// carry the real `Display` text, so panicking here would throw away
/// information a panic-only fallback can't preserve).
#[pyclass(module = "otspot", name = "SolveStatus", from_py_object)]
#[derive(Clone)]
pub enum PySolveStatus {
    Optimal(),
    LocallyOptimal(),
    Infeasible(),
    Unbounded(),
    MaxIterations(),
    SuboptimalSolution(),
    Stalled(),
    FeasiblePoint(),
    Timeout(),
    NumericalError(),
    NonConvex(String),
    NonconvexLocal(),
    NonconvexGlobal(),
    NotSupported(String),
    Unknown(String),
}

impl From<SolveStatus> for PySolveStatus {
    fn from(s: SolveStatus) -> Self {
        match s {
            SolveStatus::Optimal => PySolveStatus::Optimal(),
            SolveStatus::LocallyOptimal => PySolveStatus::LocallyOptimal(),
            SolveStatus::Infeasible => PySolveStatus::Infeasible(),
            SolveStatus::Unbounded => PySolveStatus::Unbounded(),
            SolveStatus::MaxIterations => PySolveStatus::MaxIterations(),
            SolveStatus::SuboptimalSolution => PySolveStatus::SuboptimalSolution(),
            SolveStatus::Stalled => PySolveStatus::Stalled(),
            SolveStatus::FeasiblePoint => PySolveStatus::FeasiblePoint(),
            SolveStatus::Timeout => PySolveStatus::Timeout(),
            SolveStatus::NumericalError => PySolveStatus::NumericalError(),
            SolveStatus::NonConvex(msg) => PySolveStatus::NonConvex(msg),
            SolveStatus::NonconvexLocal => PySolveStatus::NonconvexLocal(),
            SolveStatus::NonconvexGlobal => PySolveStatus::NonconvexGlobal(),
            SolveStatus::NotSupported(msg) => PySolveStatus::NotSupported(msg),
            // #[non_exhaustive]: wildcard required for cross-crate matching.
            other => PySolveStatus::Unknown(other.to_string()),
        }
    }
}

/// `Tolerance::Custom(f64)` carries a payload; same complex-enum treatment as
/// `SolveStatus` above. This conversion only runs Python -> Rust (there is no
/// `ModelResult` field of type `Tolerance`), converting *from* this crate's
/// own exhaustively-defined `PyTolerance`, so no `#[non_exhaustive]` wildcard
/// is needed here (unlike the `From<Rust> for Py*` conversions above).
#[pyclass(module = "otspot", name = "Tolerance", from_py_object)]
#[derive(Clone)]
pub enum PyTolerance {
    High(),
    Medium(),
    Fast(),
    Custom(f64),
}

impl From<PyTolerance> for Tolerance {
    fn from(t: PyTolerance) -> Self {
        match t {
            PyTolerance::High() => Tolerance::High,
            PyTolerance::Medium() => Tolerance::Medium,
            PyTolerance::Fast() => Tolerance::Fast,
            PyTolerance::Custom(v) => Tolerance::Custom(v),
        }
    }
}

pub(crate) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyVarKind>()?;
    m.add_class::<PySolutionProof>()?;
    m.add_class::<PySolveError>()?;
    m.add_class::<PySolveStatus>()?;
    m.add_class::<PyTolerance>()?;
    Ok(())
}
