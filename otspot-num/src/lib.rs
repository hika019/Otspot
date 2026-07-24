//! Numerical foundation for Otspot.
//!
//! This crate deliberately has no dependency on solver problem types.  It owns
//! the contracts used by sparse matrices and linear/KKT backends, so LP, QP,
//! conic, and global algorithms can share one numerical implementation without
//! depending on one another.

// Numerical kernels use index loops over parallel arrays (values[k], row_ind[k])
// where iterator rewrites hurt readability; mirrors otspot-core's policy.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod error;
pub mod kkt;
pub mod linalg;
pub mod sparse;

pub use error::{NumericError, SolverError};
pub use kkt::{KktBackend, LinearSolveFactor, SolveControl};
pub use sparse::{validate_csc, CscMatrixView};

/// Structural-zero tolerance shared by sparse vector construction.
pub const ZERO_TOL: f64 = 1e-12;
/// Tiny-entry drop tolerance shared by sparse matrix construction.
pub const DROP_TOL: f64 = 1e-15;
