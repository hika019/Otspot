//! 疎行列・疎ベクトル演算モジュール
//!
//! CSC形式の疎行列・疎ベクトルを提供する。

mod compress;
mod csc;
mod vec;

// Public re-exports
pub use csc::CscMatrix;
pub use vec::SparseVec;
