//! Basis management for Revised Simplex method

pub(crate) mod lu;
pub(crate) mod eta;
pub(crate) mod refactor;

use crate::sparse::{CscMatrix, SparseVec};

/// Basis manager trait for Revised Simplex
/// Manages LU factorization of the basis matrix B
pub(crate) trait BasisManager: Send {
    /// FTRAN: solve B * x = rhs, result stored in rhs
    fn ftran(&self, rhs: &mut SparseVec);

    /// BTRAN: solve B^T * x = rhs, result stored in rhs
    fn btran(&self, rhs: &mut SparseVec);

    /// Update basis after pivot: entering_col replaces leaving_row
    fn update(&mut self, entering_col: usize, leaving_row: usize, pivot_col: &SparseVec);

    /// Check numerical stability and refactor if needed
    fn refactor_if_needed(&mut self, a: &CscMatrix, basis: &[usize]);
}

/// LU-based basis manager with eta-file updates
pub(crate) struct LuBasis {
    lu: lu::LuFactorization,
    eta_file: eta::EtaFile,
    basis_indices: Vec<usize>,
}

impl LuBasis {
    /// Create a new LuBasis by factorizing the initial basis
    pub fn new(a: &CscMatrix, basis: &[usize]) -> Result<Self, String> {
        let lu = lu::LuFactorization::factorize(a, basis)?;
        Ok(Self {
            lu,
            eta_file: eta::EtaFile::new(50),
            basis_indices: basis.to_vec(),
        })
    }
}

impl BasisManager for LuBasis {
    fn ftran(&self, rhs: &mut SparseVec) {
        let mut dense = rhs.to_dense();
        lu::solve_ftran(&self.lu, &mut dense);
        eta::apply_ftran(&self.eta_file.etas, &mut dense);
        *rhs = SparseVec::from_dense(&dense);
    }

    fn btran(&self, rhs: &mut SparseVec) {
        let mut dense = rhs.to_dense();
        eta::apply_btran(&self.eta_file.etas, &mut dense);
        lu::solve_btran(&self.lu, &mut dense);
        *rhs = SparseVec::from_dense(&dense);
    }

    fn update(&mut self, entering_col: usize, leaving_row: usize, pivot_col: &SparseVec) {
        let eta = eta::add_eta_sparse(pivot_col, leaving_row);
        self.eta_file.etas.push(eta);
        self.basis_indices[leaving_row] = entering_col;
    }

    fn refactor_if_needed(&mut self, a: &CscMatrix, basis: &[usize]) {
        if self.eta_file.needs_refactor() {
            self.lu = refactor::refactor(a, basis).expect("refactoring failed");
            self.eta_file.etas.clear();
            self.basis_indices = basis.to_vec();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_near(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert!(
                (a[i] - b[i]).abs() < tol,
                "Mismatch at {}: {} vs {} (diff={})",
                i, a[i], b[i], (a[i] - b[i]).abs()
            );
        }
    }

    fn dense_to_csc(dense: &[Vec<f64>], nrows: usize, ncols: usize) -> CscMatrix {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..nrows {
            for j in 0..ncols {
                if dense[i][j].abs() > 1e-15 {
                    rows.push(i);
                    cols.push(j);
                    vals.push(dense[i][j]);
                }
            }
        }
        CscMatrix::from_triplets(&rows, &cols, &vals, nrows, ncols).unwrap()
    }

    #[test]
    fn test_lu_basis_ftran_btran() {
        // B = [[2,1,0],[1,3,1],[0,1,2]]
        let dense = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let lb = LuBasis::new(&a, &basis).unwrap();

        // FTRAN test
        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs_sv = SparseVec::from_dense(&rhs_orig);
        lb.ftran(&mut rhs_sv);
        let x = rhs_sv.to_dense();

        // Verify B * x = rhs_orig
        let check = a.mat_vec_mul(&x).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);

        // BTRAN test
        let mut rhs_sv2 = SparseVec::from_dense(&rhs_orig);
        lb.btran(&mut rhs_sv2);
        let y = rhs_sv2.to_dense();

        // Verify B^T * y = rhs_orig
        let bt = a.transpose();
        let check2 = bt.mat_vec_mul(&y).unwrap();
        assert_vec_near(&check2, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_lu_basis_update() {
        // A = [[2,1,0,3],[1,3,1,1],[0,1,2,2]]
        // Initial basis = {0,1,2} → B = [[2,1,0],[1,3,1],[0,1,2]]
        let dense = vec![
            vec![2.0, 1.0, 0.0, 3.0],
            vec![1.0, 3.0, 1.0, 1.0],
            vec![0.0, 1.0, 2.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 4);
        let basis = vec![0, 1, 2];
        let mut lb = LuBasis::new(&a, &basis).unwrap();

        // Simulate: entering col 3, leaving row 1
        // Step 1: FTRAN the entering column to get pivot column in basis space
        let entering_col_dense = vec![3.0, 1.0, 2.0]; // column 3 of A
        let mut pivot_sv = SparseVec::from_dense(&entering_col_dense);
        lb.ftran(&mut pivot_sv);

        // Step 2: Update basis
        lb.update(3, 1, &pivot_sv);

        // New basis = {0, 3, 2} → B_new = [[2,3,0],[1,1,1],[0,2,2]]
        // Verify FTRAN on new basis
        let rhs_orig = vec![5.0, 2.0, 4.0];
        let mut rhs_sv = SparseVec::from_dense(&rhs_orig);
        lb.ftran(&mut rhs_sv);
        let x = rhs_sv.to_dense();

        // Verify B_new * x = rhs_orig
        let b_new_dense = vec![
            vec![2.0, 3.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 2.0, 2.0],
        ];
        let b_new = dense_to_csc(&b_new_dense, 3, 3);
        let check = b_new.mat_vec_mul(&x).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_lu_basis_refactor() {
        // Test that refactor_if_needed works correctly
        let dense = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let mut lb = LuBasis::new(&a, &basis).unwrap();

        // Set max_etas to 2 for easy testing
        lb.eta_file.max_etas = 2;

        // Add dummy etas to trigger refactor
        lb.eta_file.etas.push(eta::add_eta(&[1.0, 0.0, 0.0], 0));
        lb.eta_file.etas.push(eta::add_eta(&[0.0, 1.0, 0.0], 1));
        assert!(lb.eta_file.needs_refactor());

        // Refactor should reset etas
        lb.refactor_if_needed(&a, &basis);
        assert!(!lb.eta_file.needs_refactor());
        assert_eq!(lb.eta_file.etas.len(), 0);

        // FTRAN should still work correctly after refactor
        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs_sv = SparseVec::from_dense(&rhs_orig);
        lb.ftran(&mut rhs_sv);
        let x = rhs_sv.to_dense();

        let check = a.mat_vec_mul(&x).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }
}
