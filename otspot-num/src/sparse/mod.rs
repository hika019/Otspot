//! Sparse matrix and vector representations.

mod compress;
mod csc;
mod vec;
mod view;

pub use csc::CscMatrix;
pub use vec::SparseVec;
pub use view::{validate_csc, CscMatrixView};
