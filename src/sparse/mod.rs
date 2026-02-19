//! 疎行列・疎ベクトル演算モジュール
//!
//! CSC/CSR形式の疎行列、疎ベクトル、三角疎行列を提供する。

mod vec;
mod csc;
mod csr;
mod triangular;
mod compress;

// Public re-exports
pub use vec::SparseVec;
pub use csc::CscMatrix;
pub use csr::CsrMatrix;

// Crate-internal re-exports
pub(crate) use triangular::{SparseLowerCSC, SparseUpperCSR};
