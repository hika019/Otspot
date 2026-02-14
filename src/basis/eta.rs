//! Eta-factoring for basis updates in Revised Simplex

/// Single eta matrix: E = I + (column - e_r) * e_r^T
/// where column[leaving_row] = 1/pivot and column[i] = -a_ir/pivot for i != leaving_row
#[derive(Debug, Clone)]
pub(crate) struct EtaMatrix {
    pub leaving_row: usize,
    pub column: Vec<f64>,
}

/// Collection of eta matrices accumulated since last refactorization
#[derive(Debug, Clone)]
pub(crate) struct EtaFile {
    pub etas: Vec<EtaMatrix>,
    pub max_etas: usize,
}

impl EtaFile {
    pub fn new(max_etas: usize) -> Self {
        Self {
            etas: Vec::new(),
            max_etas,
        }
    }

    pub fn needs_refactor(&self) -> bool {
        self.etas.len() >= self.max_etas
    }
}

/// Create an eta matrix from the FTRAN'd pivot column and the leaving row
/// pivot_col is the result of FTRAN on the entering column (i.e., B^{-1} * a_entering)
pub(crate) fn add_eta(pivot_col: &[f64], leaving_row: usize) -> EtaMatrix {
    let m = pivot_col.len();
    let pivot_element = pivot_col[leaving_row];
    let mut column = vec![0.0; m];

    for i in 0..m {
        if i == leaving_row {
            column[i] = 1.0 / pivot_element;
        } else {
            column[i] = -pivot_col[i] / pivot_element;
        }
    }

    EtaMatrix {
        leaving_row,
        column,
    }
}

/// Apply eta matrices in forward order for FTRAN
/// Each eta: apply E^{-1} * rhs.
///
/// The stored `column` IS the r-th column of E^{-1}:
///   column[r] = 1/d[r], column[i] = -d[i]/d[r] for i != r
///   where d is the original FTRAN'd pivot column.
///
/// E^{-1} = identity with column r replaced by `column`.
///
/// (E^{-1} * x)[r] = column[r] * x[r]
/// (E^{-1} * x)[i] = x[i] + column[i] * x[r]  for i != r
pub(crate) fn apply_ftran(etas: &[EtaMatrix], rhs: &mut Vec<f64>) {
    for eta in etas {
        let r = eta.leaving_row;
        let x_r = rhs[r];
        rhs[r] = eta.column[r] * x_r;
        for i in 0..rhs.len() {
            if i != r {
                rhs[i] += eta.column[i] * x_r;
            }
        }
    }
}

/// Apply eta matrices in reverse order for BTRAN
/// BTRAN applies E^{-T} in reverse order.
///
/// E^{-T} = transpose of E^{-1}.
/// Row r of E^{-T} = stored `column` (transposed from column r of E^{-1}).
/// Other rows are identity.
///
/// (E^{-T} * x)[i] = x[i]                          for i != r
/// (E^{-T} * x)[r] = sum_j column[j] * x[j]
pub(crate) fn apply_btran(etas: &[EtaMatrix], rhs: &mut Vec<f64>) {
    for eta in etas.iter().rev() {
        let r = eta.leaving_row;
        let mut dot = 0.0;
        for j in 0..rhs.len() {
            dot += eta.column[j] * rhs[j];
        }
        rhs[r] = dot;
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

    #[test]
    fn test_eta_single_update() {
        // Simulate: B = I (identity), then one basis update
        // Entering column (after FTRAN with identity) = [2.0, 1.0, 0.5]
        // Leaving row = 0
        let pivot_col = vec![2.0, 1.0, 0.5];
        let eta = add_eta(&pivot_col, 0);

        // eta.column should be:
        // col[0] = 1/2.0 = 0.5
        // col[1] = -1.0/2.0 = -0.5
        // col[2] = -0.5/2.0 = -0.25
        assert!((eta.column[0] - 0.5).abs() < 1e-10);
        assert!((eta.column[1] - (-0.5)).abs() < 1e-10);
        assert!((eta.column[2] - (-0.25)).abs() < 1e-10);

        // Apply FTRAN with this single eta to rhs = [1, 0, 0]
        // After update, B_new = B_old with col 0 replaced by [2, 1, 0.5]
        // B_new = [[2,0,0],[1,1,0],[0.5,0,1]]
        // B_new^{-1} * [1,0,0]:
        //   solve [[2,0,0],[1,1,0],[0.5,0,1]] * x = [1,0,0]
        //   x[0] = 0.5, x[1] = -0.5, x[2] = -0.25
        let mut rhs = vec![1.0, 0.0, 0.0];
        apply_ftran(&[eta], &mut rhs);
        assert_vec_near(&rhs, &[0.5, -0.5, -0.25], 1e-10);
    }

    #[test]
    fn test_eta_multiple_updates() {
        // Start with identity basis, apply 3 updates
        // Update 1: entering col (FTRAN'd) = [2, 1, 0], leaving row 0
        let eta1 = add_eta(&[2.0, 1.0, 0.0], 0);
        // Update 2: entering col (FTRAN'd through eta1) = [0.5, 3, 1], leaving row 1
        let eta2 = add_eta(&[0.5, 3.0, 1.0], 1);
        // Update 3: entering col (FTRAN'd through eta1,eta2) = [1, 0.5, 4], leaving row 2
        let eta3 = add_eta(&[1.0, 0.5, 4.0], 2);

        let etas = vec![eta1, eta2, eta3];

        // FTRAN: solve updated_B * x = rhs
        let mut rhs = vec![1.0, 2.0, 3.0];
        let rhs_orig = rhs.clone();
        apply_ftran(&etas, &mut rhs);

        // Verify: B_new * x should equal rhs_orig
        // B_new = B_0 * E_1 * E_2 * E_3 = E_1 * E_2 * E_3 (since B_0 = I)
        // So B_new * x = E_1 * (E_2 * (E_3 * x))
        // Apply in order: E_3 first (innermost), then E_2, then E_1
        let mut check = rhs.clone();

        // E3 * x: column 2 of E3 is [1, 0.5, 4], identity elsewhere
        let temp = check.clone();
        check[0] = temp[0] + 1.0 * temp[2];
        check[1] = temp[1] + 0.5 * temp[2];
        check[2] = 4.0 * temp[2];

        // E2 * check: column 1 replaced by [0.5, 3, 1]
        let temp = check.clone();
        check[0] = temp[0] + 0.5 * temp[1];
        check[1] = 3.0 * temp[1];
        check[2] = temp[2] + 1.0 * temp[1];

        // E1 * check: column 0 replaced by [2, 1, 0]
        let temp = check.clone();
        check[0] = 2.0 * temp[0];
        check[1] = temp[1] + 1.0 * temp[0];
        check[2] = temp[2] + 0.0 * temp[0];

        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_eta_btran() {
        // Single eta, verify BTRAN
        let eta = add_eta(&[2.0, 1.0, 0.5], 0);

        // E * x: column 0 is [2, 1, 0.5]
        // E = [[2,0,0],[1,1,0],[0.5,0,1]]
        // E^T = [[2,1,0.5],[0,1,0],[0,0,1]]
        // Solve E^T * y = [1, 2, 3]:
        //   2*y0 + y1 + 0.5*y2 = 1
        //   y1 = 2
        //   y2 = 3
        //   2*y0 = 1 - 2 - 1.5 = -2.5 => y0 = -1.25
        let mut rhs = vec![1.0, 2.0, 3.0];
        apply_btran(&[eta], &mut rhs);
        assert_vec_near(&rhs, &[-1.25, 2.0, 3.0], 1e-10);
    }

    #[test]
    fn test_eta_needs_refactor() {
        let mut ef = EtaFile::new(3);
        assert!(!ef.needs_refactor());
        ef.etas.push(add_eta(&[1.0], 0));
        assert!(!ef.needs_refactor());
        ef.etas.push(add_eta(&[1.0], 0));
        assert!(!ef.needs_refactor());
        ef.etas.push(add_eta(&[1.0], 0));
        assert!(ef.needs_refactor());
    }
}
