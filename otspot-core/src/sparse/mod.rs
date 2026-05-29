//! 疎行列・疎ベクトル演算モジュール
//!
//! CSC形式の疎行列・疎ベクトルを提供する。

mod vec;
mod csc;
mod compress;

// Public re-exports
pub use vec::SparseVec;
pub use csc::CscMatrix;
