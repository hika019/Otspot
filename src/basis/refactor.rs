//! Periodic refactorization support

use crate::sparse::CscMatrix;
use super::lu::LuFactorization;

/// Refactorize the basis matrix from scratch
/// Thin wrapper around LU factorization
pub(crate) fn refactor(a: &CscMatrix, basis: &[usize]) -> Result<LuFactorization, String> {
    LuFactorization::factorize(a, basis)
}
