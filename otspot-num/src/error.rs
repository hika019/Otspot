//! Errors produced by numerical kernels.

/// Failure categories shared by sparse, factorization, and iterative backends.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum NumericError {
    DimensionMismatch {
        field: &'static str,
        expected: usize,
        got: usize,
    },
    IndexOutOfBounds {
        context: &'static str,
        index: usize,
        bound: usize,
    },
    NonFinite {
        field: &'static str,
        index: usize,
    },
    InvalidBounds {
        context: &'static str,
        index: usize,
        lower: f64,
        upper: f64,
    },
    InvalidSparseStructure {
        message: &'static str,
    },
    Singular {
        step: usize,
    },
    NonConvergence {
        iterations: usize,
        residual: f64,
    },
    DeadlineExceeded,
    Cancelled,
}

impl std::fmt::Display for NumericError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch {
                field,
                expected,
                got,
            } => write!(f, "{field}: expected dimension {expected}, got {got}"),
            Self::IndexOutOfBounds {
                context,
                index,
                bound,
            } => write!(f, "{context} index {index} out of bounds (size={bound})"),
            Self::NonFinite { field, index } => {
                write!(f, "non-finite value in {field} at index {index}")
            }
            Self::InvalidBounds {
                context,
                index,
                lower,
                upper,
            } => write!(
                f,
                "invalid {context} bounds at index {index}: lower={lower}, upper={upper}"
            ),
            Self::InvalidSparseStructure { message } => {
                write!(f, "invalid sparse structure: {message}")
            }
            Self::Singular { step } => write!(f, "singular matrix at step {step}"),
            Self::NonConvergence {
                iterations,
                residual,
            } => write!(
                f,
                "iterative solve did not converge after {iterations} iterations \
                 (residual={residual})"
            ),
            Self::DeadlineExceeded => write!(f, "deadline exceeded"),
            Self::Cancelled => write!(f, "operation cancelled"),
        }
    }
}

impl std::error::Error for NumericError {}

/// Legacy solver-wide error retained during the crate migration.
///
/// `otspot-core::SolverError` re-exports this exact type, so moving
/// `CscMatrix` into `otspot-num` does not change constructor signatures.
#[non_exhaustive]
#[derive(Debug)]
pub enum SolverError {
    DimensionMismatch {
        field: &'static str,
        expected: usize,
        got: usize,
    },
    IndexOutOfBounds {
        context: &'static str,
        index: usize,
        bound: usize,
    },
    SingularBasis {
        step: usize,
    },
    EmptyInput {
        context: &'static str,
    },
    DeadlineExceeded,
    NonFiniteCoefficient {
        field: &'static str,
        index: usize,
    },
    InvalidBounds {
        index: usize,
        lb: f64,
        ub: f64,
    },
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch {
                field,
                expected,
                got,
            } => write!(
                f,
                "Dimension mismatch: {field} expected {expected} but got {got}"
            ),
            Self::IndexOutOfBounds {
                context,
                index,
                bound,
            } => write!(f, "{context} index {index} out of bounds (size={bound})"),
            Self::SingularBasis { step } => {
                write!(f, "Singular matrix detected at step {step}")
            }
            Self::EmptyInput { context } => write!(f, "Empty input: {context}"),
            Self::DeadlineExceeded => write!(f, "Deadline exceeded during computation"),
            Self::NonFiniteCoefficient { field, index } => {
                write!(f, "Non-finite coefficient in {field}: index {index}")
            }
            Self::InvalidBounds { index, lb, ub } => write!(
                f,
                "Invalid bounds at index {index}: lb={lb} > ub={ub} or NaN"
            ),
        }
    }
}

impl std::error::Error for SolverError {}
