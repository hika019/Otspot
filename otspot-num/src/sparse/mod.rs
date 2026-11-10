//! Sparse matrix and vector representations.

mod compress;
mod csc;
mod spmv;
mod vec;
mod view;

pub use csc::transpose_construction_count;
pub use csc::CscMatrix;
#[cfg(feature = "transpose-count")]
pub use csc::{reset_transpose_constructions, transpose_constructions};
pub use spmv::mat_t_vec_mul_into;
pub use vec::SparseVec;
pub use view::{validate_csc, CscMatrixView};
