//! 修正シンプレックス法における基底更新のイータ因子分解
//!
//! 修正シンプレックス法では基底行列 B を直接保持する代わりに、
//! イータ行列 `E = I + (col - e_r) * e_r^T` の積で `B^{-1}` を逐次更新する。
//! このモジュールは疎なイータ行列の生成・蓄積・適用（FTRAN/BTRAN）を提供する。

use crate::sparse::SparseVec;
use crate::tolerances::*;

/// 単一のイータ行列: `E = I + (col - e_r) * e_r^T`
///
/// `B^{-1}` の1ステップ更新を疎表現で保持する。
/// 非零エントリのみを `indices` / `values` に格納し、空間効率を確保する。
#[derive(Debug, Clone)]
pub(crate) struct EtaMatrix {
    pub leaving_row: usize,
    pub indices: Vec<usize>,
    pub values: Vec<f64>,
}

/// 最後の再因子分解以降に蓄積されたイータ行列の集合
///
/// イータ行列が `max_etas` 個に達すると [`needs_refactor`] が `true` を返し、
/// 完全再因子分解のタイミングを通知する。
///
/// [`needs_refactor`]: EtaFile::needs_refactor
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

/// Builds an [`EtaMatrix`] from a dense pivot column (test-only).
#[cfg(test)]
pub(crate) fn add_eta(pivot_col: &[f64], leaving_row: usize) -> EtaMatrix {
    let pivot_element = pivot_col[leaving_row];
    let mut indices = Vec::new();
    let mut values = Vec::new();

    for (i, &pc) in pivot_col.iter().enumerate() {
        let val = if i == leaving_row {
            1.0 / pivot_element
        } else {
            -pc / pivot_element
        };
        if val.abs() > ZERO_TOL {
            indices.push(i);
            values.push(val);
        }
    }

    EtaMatrix {
        leaving_row,
        indices,
        values,
    }
}

/// Builds an [`EtaMatrix`] from a sparse pivot column. Avoids dense conversion; output is sorted by index.
pub(crate) fn add_eta_sparse(pivot_col: &SparseVec, leaving_row: usize) -> EtaMatrix {
    let pivot_element = match pivot_col.indices.binary_search(&leaving_row) {
        Ok(pos) => pivot_col.values[pos],
        Err(_) => 0.0,
    };
    let inv_pivot = 1.0 / pivot_element;
    let mut indices = Vec::new();
    let mut values = Vec::new();

    // The leaving_row entry: 1/pivot
    if inv_pivot.abs() > ZERO_TOL {
        indices.push(leaving_row);
        values.push(inv_pivot);
    }

    // Other non-zero entries: -pivot_col[i] / pivot_element
    for (k, &idx) in pivot_col.indices.iter().enumerate() {
        if idx == leaving_row {
            continue;
        }
        let val = -pivot_col.values[k] / pivot_element;
        if val.abs() > ZERO_TOL {
            indices.push(idx);
            values.push(val);
        }
    }

    // Sort by index for consistency
    let mut pairs: Vec<(usize, f64)> = indices.into_iter().zip(values).collect();
    pairs.sort_by_key(|&(idx, _)| idx);

    EtaMatrix {
        leaving_row,
        indices: pairs.iter().map(|&(idx, _)| idx).collect(),
        values: pairs.iter().map(|&(_, val)| val).collect(),
    }
}

/// FTRAN: applies accumulated eta matrices to `rhs` in-place (approximates B⁻¹ · rhs).
pub(crate) fn apply_ftran(etas: &[EtaMatrix], rhs: &mut [f64]) {
    for eta in etas {
        let r = eta.leaving_row;
        let x_r = rhs[r];
        if x_r.abs() < DROP_TOL {
            continue;
        }

        // Reset rhs[r], then accumulate from sparse entries
        rhs[r] = 0.0;
        for (k, &idx) in eta.indices.iter().enumerate() {
            rhs[idx] += eta.values[k] * x_r;
        }
    }
}

/// BTRAN: applies eta matrices in reverse order to `rhs` in-place (approximates rhs^T · B⁻¹).
pub(crate) fn apply_btran(etas: &[EtaMatrix], rhs: &mut [f64]) {
    for eta in etas.iter().rev() {
        let r = eta.leaving_row;
        let mut dot = 0.0;
        for (k, &idx) in eta.indices.iter().enumerate() {
            dot += eta.values[k] * rhs[idx];
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
                i,
                a[i],
                b[i],
                (a[i] - b[i]).abs()
            );
        }
    }

    #[test]
    fn test_eta_single_update() {
        let pivot_col = vec![2.0, 1.0, 0.5];
        let eta = add_eta(&pivot_col, 0);

        // Verify sparse column values
        let val_at = |idx: usize| -> f64 {
            for (k, &i) in eta.indices.iter().enumerate() {
                if i == idx {
                    return eta.values[k];
                }
            }
            0.0
        };
        assert!((val_at(0) - 0.5).abs() < 1e-10);
        assert!((val_at(1) - (-0.5)).abs() < 1e-10);
        assert!((val_at(2) - (-0.25)).abs() < 1e-10);

        let mut rhs = vec![1.0, 0.0, 0.0];
        apply_ftran(&[eta], &mut rhs);
        assert_vec_near(&rhs, &[0.5, -0.5, -0.25], 1e-10);
    }

    #[test]
    fn test_eta_multiple_updates() {
        let eta1 = add_eta(&[2.0, 1.0, 0.0], 0);
        let eta2 = add_eta(&[0.5, 3.0, 1.0], 1);
        let eta3 = add_eta(&[1.0, 0.5, 4.0], 2);

        let etas = vec![eta1, eta2, eta3];

        let mut rhs = vec![1.0, 2.0, 3.0];
        let rhs_orig = rhs.clone();
        apply_ftran(&etas, &mut rhs);

        let mut check = rhs.clone();

        let temp = check.clone();
        check[0] = temp[0] + 1.0 * temp[2];
        check[1] = temp[1] + 0.5 * temp[2];
        check[2] = 4.0 * temp[2];

        let temp = check.clone();
        check[0] = temp[0] + 0.5 * temp[1];
        check[1] = 3.0 * temp[1];
        check[2] = temp[2] + 1.0 * temp[1];

        let temp = check.clone();
        check[0] = 2.0 * temp[0];
        check[1] = temp[1] + 1.0 * temp[0];
        check[2] = temp[2] + 0.0 * temp[0];

        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_eta_btran() {
        let eta = add_eta(&[2.0, 1.0, 0.5], 0);

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

    #[test]
    fn test_eta_sparse_from_sparse_vec() {
        let sv = SparseVec::from_dense(&[2.0, 1.0, 0.5]);
        let eta = add_eta_sparse(&sv, 0);

        let val_at = |idx: usize| -> f64 {
            for (k, &i) in eta.indices.iter().enumerate() {
                if i == idx {
                    return eta.values[k];
                }
            }
            0.0
        };
        assert!((val_at(0) - 0.5).abs() < 1e-10);
        assert!((val_at(1) - (-0.5)).abs() < 1e-10);
        assert!((val_at(2) - (-0.25)).abs() < 1e-10);
    }
}
