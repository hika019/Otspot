//! Enum types mirroring `otspot_model`/`otspot_core` variant-for-variant.
//!
//! Variant names are kept identical to Rust (`Continuous`, not `CONTINUOUS`)
//! so the parity manifest can assert a literal name match on both sides.

use otspot_core::options::Tolerance;
use otspot_core::problem::SolveStatus;
use otspot_model::{ConstraintSense, SolutionProof, SolveError, VarKind};
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

#[pyclass(
    module = "otspot",
    name = "ConstraintSense",
    eq,
    eq_int,
    from_py_object
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyConstraintSense {
    Le,
    Ge,
    Eq,
}

impl From<ConstraintSense> for PyConstraintSense {
    fn from(s: ConstraintSense) -> Self {
        match s {
            ConstraintSense::Le => PyConstraintSense::Le,
            ConstraintSense::Ge => PyConstraintSense::Ge,
            ConstraintSense::Eq => PyConstraintSense::Eq,
            // `ConstraintSense` is `#[non_exhaustive]`; cross-crate wildcard required.
            _ => PyConstraintSense::Eq,
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
            // #[non_exhaustive]: wildcard required for cross-crate matching.
            _ => PySolutionProof::FeasibleUnproven,
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
            // #[non_exhaustive]: wildcard required for cross-crate matching.
            _ => PySolveError::NumericalError,
        }
    }
}

/// `SolveStatus` has two payload-carrying variants (`NonConvex`, `NotSupported`),
/// so it is bound as a PyO3 "complex enum" rather than a C-like `eq_int` enum:
/// each variant becomes a Python subclass, exactly mirroring the Rust shape.
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
            _ => PySolveStatus::NumericalError(),
        }
    }
}

/// `Tolerance::Custom(f64)` carries a payload; same complex-enum treatment as
/// `SolveStatus` above.
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
    m.add_class::<PyConstraintSense>()?;
    m.add_class::<PySolutionProof>()?;
    m.add_class::<PySolveError>()?;
    m.add_class::<PySolveStatus>()?;
    m.add_class::<PyTolerance>()?;
    Ok(())
}
