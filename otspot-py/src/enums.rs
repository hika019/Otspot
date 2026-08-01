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
//!
//! Every type here implements `__reduce__` so instances survive
//! `pickle`/`copy.deepcopy` (needed for e.g. `multiprocessing` error
//! propagation: `SolveFailedError.error` is a `PySolveError`, and pickling an
//! exception pickles its `__dict__`, including that attribute).

use otspot_core::options::Tolerance;
use otspot_core::problem::SolveStatus;
use otspot_model::{SolutionProof, SolveError, VarKind};
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::IntoPyObjectExt;

/// `__reduce__` payload for a plain `#[pyclass(eq, eq_int)]` unit variant:
/// `(getattr, (EnumClass, "VariantName"))`, so unpickling re-evaluates
/// `getattr(EnumClass, "VariantName")` — exactly the class-attribute access
/// every caller already uses to obtain one of these singletons.
fn reduce_via_getattr(py: Python<'_>, cls: Bound<'_, PyType>, name: &str) -> PyResult<Py<PyAny>> {
    let getattr = PyModule::import(py, "builtins")?.getattr("getattr")?;
    (getattr, (cls, name)).into_py_any(py)
}

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

#[pymethods]
impl PyVarKind {
    fn __reduce__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let name = match self {
            PyVarKind::Continuous => "Continuous",
            PyVarKind::Integer => "Integer",
            PyVarKind::Binary => "Binary",
        };
        reduce_via_getattr(py, py.get_type::<PyVarKind>(), name)
    }
}

#[pyclass(module = "otspot", name = "SolutionProof", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PySolutionProof {
    GlobalOptimal,
    LocalOptimal,
    FeasibleUnproven,
    /// Not a real `SolutionProof` variant: the honest fallback for the
    /// `#[non_exhaustive]` wildcard below. `eq_int` variants cannot carry a
    /// payload, so (unlike `PySolveStatus::Unknown`) this cannot preserve
    /// the original `Debug` text — but mapping to an arbitrary *known*
    /// variant instead would silently misreport optimality strength to
    /// callers branching on it, which is strictly worse.
    Unknown,
}

impl From<SolutionProof> for PySolutionProof {
    fn from(p: SolutionProof) -> Self {
        match p {
            SolutionProof::GlobalOptimal => PySolutionProof::GlobalOptimal,
            SolutionProof::LocalOptimal => PySolutionProof::LocalOptimal,
            SolutionProof::FeasibleUnproven => PySolutionProof::FeasibleUnproven,
            // `SolutionProof` is `#[non_exhaustive]`, so a wildcard is
            // mandatory to compile.
            _ => PySolutionProof::Unknown,
        }
    }
}

#[pymethods]
impl PySolutionProof {
    fn __reduce__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let name = match self {
            PySolutionProof::GlobalOptimal => "GlobalOptimal",
            PySolutionProof::LocalOptimal => "LocalOptimal",
            PySolutionProof::FeasibleUnproven => "FeasibleUnproven",
            PySolutionProof::Unknown => "Unknown",
        };
        reduce_via_getattr(py, py.get_type::<PySolutionProof>(), name)
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
    /// See `PySolutionProof::Unknown`'s doc comment.
    Unknown,
}

impl From<SolveError> for PySolveError {
    fn from(e: SolveError) -> Self {
        match e {
            SolveError::Infeasible => PySolveError::Infeasible,
            SolveError::Unbounded => PySolveError::Unbounded,
            SolveError::MaxIterations => PySolveError::MaxIterations,
            SolveError::Stalled => PySolveError::Stalled,
            SolveError::NumericalError => PySolveError::NumericalError,
            // #[non_exhaustive]: wildcard required for cross-crate matching.
            _ => PySolveError::Unknown,
        }
    }
}

#[pymethods]
impl PySolveError {
    fn __reduce__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let name = match self {
            PySolveError::Infeasible => "Infeasible",
            PySolveError::Unbounded => "Unbounded",
            PySolveError::MaxIterations => "MaxIterations",
            PySolveError::Stalled => "Stalled",
            PySolveError::NumericalError => "NumericalError",
            PySolveError::Unknown => "Unknown",
        };
        reduce_via_getattr(py, py.get_type::<PySolveError>(), name)
    }
}

/// `SolveStatus` has payload-carrying variants (`NonConvex`, `NotSupported`),
/// so it is bound as a PyO3 "complex enum" rather than a C-like `eq_int` enum:
/// each variant becomes a Python subclass, exactly mirroring the Rust shape.
///
/// `Unknown(String)` is not a real `SolveStatus` variant: it is the honest
/// fallback for the `#[non_exhaustive]` wildcard this `From` impl is forced
/// to have (unlike `PySolutionProof`/`PySolveError`, a complex enum *can*
/// carry the real `Display` text).
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

#[pymethods]
impl PySolveStatus {
    /// `(VariantClass, ())` for a unit variant, `(VariantClass, (payload,))`
    /// for a payload variant — `VariantClass(*args)` reconstructs the exact
    /// instance, mirroring how every variant is already constructed.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cls = py.get_type::<PySolveStatus>();
        let (name, payload): (&str, Option<&str>) = match self {
            PySolveStatus::Optimal() => ("Optimal", None),
            PySolveStatus::LocallyOptimal() => ("LocallyOptimal", None),
            PySolveStatus::Infeasible() => ("Infeasible", None),
            PySolveStatus::Unbounded() => ("Unbounded", None),
            PySolveStatus::MaxIterations() => ("MaxIterations", None),
            PySolveStatus::SuboptimalSolution() => ("SuboptimalSolution", None),
            PySolveStatus::Stalled() => ("Stalled", None),
            PySolveStatus::FeasiblePoint() => ("FeasiblePoint", None),
            PySolveStatus::Timeout() => ("Timeout", None),
            PySolveStatus::NumericalError() => ("NumericalError", None),
            PySolveStatus::NonConvex(msg) => ("NonConvex", Some(msg.as_str())),
            PySolveStatus::NonconvexLocal() => ("NonconvexLocal", None),
            PySolveStatus::NonconvexGlobal() => ("NonconvexGlobal", None),
            PySolveStatus::NotSupported(msg) => ("NotSupported", Some(msg.as_str())),
            PySolveStatus::Unknown(msg) => ("Unknown", Some(msg.as_str())),
        };
        let variant_cls = cls.getattr(name)?;
        match payload {
            Some(msg) => (variant_cls, (msg,)).into_py_any(py),
            None => (variant_cls, ()).into_py_any(py),
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

#[pymethods]
impl PyTolerance {
    fn __reduce__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cls = py.get_type::<PyTolerance>();
        let (name, payload): (&str, Option<f64>) = match self {
            PyTolerance::High() => ("High", None),
            PyTolerance::Medium() => ("Medium", None),
            PyTolerance::Fast() => ("Fast", None),
            PyTolerance::Custom(eps) => ("Custom", Some(*eps)),
        };
        let variant_cls = cls.getattr(name)?;
        match payload {
            Some(eps) => (variant_cls, (eps,)).into_py_any(py),
            None => (variant_cls, ()).into_py_any(py),
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
