//! 線形代数 / 行列ビルダー: bound 寄与の集約と A·Aᵀ の上三角 CSC 構築。
//! refine 系 helper が共通で必要とする小規模 utility のみ置く。

use crate::sparse::CscMatrix;

/// AAT 対角ε 正則化倍率 (rank-deficient 対策)。f64 eps より十分上、LDL dynamic reg より十分下。
pub(crate) const AAT_REG_FACTOR: f64 = 1e-12;

/// BTreeMap ノードあたり実測バイト数 (key 16 + value 8 + node overhead)。memory budget の係数。
const AAT_BUILD_BYTES_PER_ENTRY: u128 = 80;

/// bound dual `z` の stationarity への寄与: contrib_j = -z_lb,j + z_ub,j。
pub(crate) fn compute_bound_contrib(
    bounds: &[(f64, f64)],
    bound_duals: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut contrib = vec![0.0_f64; n];
    if bound_duals.is_empty() {
        return contrib;
    }
    let mut idx = 0usize;
    for (j, &(lb, _)) in bounds.iter().enumerate() {
        if lb.is_finite() && idx < bound_duals.len() {
            contrib[j] -= bound_duals[idx];
            idx += 1;
        }
    }
    for (j, &(_, ub)) in bounds.iter().enumerate() {
        if ub.is_finite() && idx < bound_duals.len() {
            contrib[j] += bound_duals[idx];
            idx += 1;
        }
    }
    contrib
}

/// A·A^T (m×m, 上三角 CSC) + 対角 ε 正則化 (rank-deficient でも factorize 可)。
/// nnz upper bound × 80 B が memory_budget 超なら None を返し caller は skip。
pub(crate) fn build_aat_upper_csc(a: &CscMatrix, n: usize, m: usize) -> Option<CscMatrix> {
    use std::collections::BTreeMap;
    let m_u = m as u128;
    let mut col_pair_sum: u128 = 0;
    for k in 0..n {
        let c_k = (a.col_ptr[k + 1] - a.col_ptr[k]) as u128;
        col_pair_sum = col_pair_sum.saturating_add(c_k.saturating_mul(c_k + 1) / 2);
    }
    let nnz_upper_bound = (m_u.saturating_mul(m_u + 1) / 2).min(col_pair_sum);
    let bytes_estimate = nnz_upper_bound.saturating_mul(AAT_BUILD_BYTES_PER_ENTRY);
    if bytes_estimate > crate::linalg::kkt_solver::memory_budget_bytes() as u128 {
        return None;
    }
    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for k in 0..n {
        let start = a.col_ptr[k];
        let end = a.col_ptr[k + 1];
        let cols_in_k: Vec<(usize, f64)> =
            (start..end).map(|p| (a.row_ind[p], a.values[p])).collect();
        for (idx_a, &(i, v_i)) in cols_in_k.iter().enumerate() {
            for &(j, v_j) in &cols_in_k[idx_a..] {
                let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
                *acc.entry((hi, lo)).or_insert(0.0) += v_i * v_j;
            }
        }
    }
    let max_diag = (0..m)
        .filter_map(|i| acc.get(&(i, i)).copied())
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let reg = AAT_REG_FACTOR * max_diag;
    for i in 0..m {
        *acc.entry((i, i)).or_insert(0.0) += reg;
    }
    let mut col_ptr = vec![0_usize; m + 1];
    let mut row_ind: Vec<usize> = Vec::with_capacity(acc.len());
    let mut values: Vec<f64> = Vec::with_capacity(acc.len());
    for ((col, row), val) in acc {
        row_ind.push(row);
        values.push(val);
        col_ptr[col + 1] = row_ind.len();
    }
    for i in 1..=m {
        if col_ptr[i] < col_ptr[i - 1] {
            col_ptr[i] = col_ptr[i - 1];
        }
    }
    Some(CscMatrix {
        col_ptr,
        row_ind,
        values,
        nrows: m,
        ncols: m,
    })
}
