//! A·Aᵀ (上三角 CSC) ビルダー。QP refine 系 helper 専用。
//! `crate::linalg` (otspot_num facade) とは無関係の QP 内部実装。

use otspot_num::sparse::CscMatrix;

/// AAT 対角ε 正則化倍率 (rank-deficient 対策)。f64 eps より十分上、LDL dynamic reg より十分下。
pub(crate) const AAT_REG_FACTOR: f64 = 1e-12;

/// BTreeMap ノードあたり実測バイト数 (key 16 + value 8 + node overhead)。memory budget の係数。
const AAT_BUILD_BYTES_PER_ENTRY: u128 = 80;

/// 列スケーリングを「実体なし」と判定する許容幅。IRLS 重み √w が 1 にこの差以内の
/// 列は掛け算をスキップする (旧 clone+in-place スケーリング手順と同じ条件・同じ
/// 演算回数を保つための値で、結果のビット列はスキップ与否に依存しない)。
const SCALE_IDENTITY_TOL: f64 = 1e-15;

/// A·A^T (m×m, 上三角 CSC) + 対角 ε 正則化 (rank-deficient でも factorize 可)。
/// nnz upper bound × 80 B が memory_budget 超なら None を返し caller は skip。
pub(crate) fn build_aat_upper_csc(a: &CscMatrix, n: usize, m: usize) -> Option<CscMatrix> {
    build_aat_upper_csc_scaled(a, n, m, None)
}

/// 列スケーリング付き AAT ビルダー。`col_scale`(長さ n) 指定時は列 k の成分を
/// `v * col_scale[k]` 倍してから A·Aᵀ を組み立てる。
///
/// 数値挙動は「`a` を clone → 同条件で values を同倍率で in-place スケーリング →
/// `build_aat_upper_csc`」とビット単位で同一: 各成分の掛け算は同じ丸みで行われ、
/// 内部 accumulate の順序・回数はスケール有無で不変。
/// IRLS 重み √w 適用の per-iteration clone + 別パッセージュールを省く目的。
/// `|s - 1| < SCALE_IDENTITY_TOL` の列は掛け算をスキップする (v*1.0==v のため
/// スキップ与否は無関係にビット同一だが、既存手順の演算回数を忠実に保つ)。
pub(crate) fn build_aat_upper_csc_scaled(
    a: &CscMatrix,
    n: usize,
    m: usize,
    col_scale: Option<&[f64]>,
) -> Option<CscMatrix> {
    use std::collections::BTreeMap;
    let m_u = m as u128;
    let mut col_pair_sum: u128 = 0;
    for k in 0..n {
        let c_k = (a.col_ptr()[k + 1] - a.col_ptr()[k]) as u128;
        col_pair_sum = col_pair_sum.saturating_add(c_k.saturating_mul(c_k + 1) / 2);
    }
    let nnz_upper_bound = (m_u.saturating_mul(m_u + 1) / 2).min(col_pair_sum);
    let bytes_estimate = nnz_upper_bound.saturating_mul(AAT_BUILD_BYTES_PER_ENTRY);
    if bytes_estimate > otspot_num::linalg::kkt_solver::memory_budget_bytes() as u128 {
        return None;
    }
    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for k in 0..n {
        let start = a.col_ptr()[k];
        let end = a.col_ptr()[k + 1];
        let s = col_scale.map(|cs| cs[k]);
        let scale_is_identity = s.is_none_or(|v| (v - 1.0).abs() < SCALE_IDENTITY_TOL);
        let cols_in_k: Vec<(usize, f64)> = (start..end)
            .map(|p| match s {
                Some(sv) if !scale_is_identity => (a.row_ind()[p], a.values()[p] * sv),
                _ => (a.row_ind()[p], a.values()[p]),
            })
            .collect();
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
    Some(CscMatrix::from_raw_parts(m, m, col_ptr, row_ind, values))
}

#[cfg(test)]
mod tests {
    use super::{build_aat_upper_csc, build_aat_upper_csc_scaled};
    use otspot_num::sparse::CscMatrix;

    fn aat_csc_bit_eq(x: &CscMatrix, y: &CscMatrix) -> bool {
        x.col_ptr() == y.col_ptr()
            && x.row_ind() == y.row_ind()
            && x.values().len() == y.values().len()
            && x.values()
                .iter()
                .zip(y.values())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    /// 独立オラクル: 「clone → 列ごと √w in-place 倍 → 無スケール構築」という
    /// 旧手順をテスト側で再現し、fused 版 (build_aat_upper_csc_scaled) との
    /// ビット一致を検証する。fused 版の掛け算位置・accumulate 順序を壊すと fail。
    #[test]
    fn scaled_aat_bit_identical_to_clone_scale_build() {
        // m=3 制約 × n=4 変数。明示 0.0 エントリ・負値・同一視tol内のスケールを含む。
        let a = CscMatrix::from_raw_parts(
            3,
            4,
            vec![0, 1, 3, 4, 6],
            vec![0, 0, 2, 1, 0, 2],
            vec![2.0, 0.5, -1.25, 0.0, -0.5, 4.25],
        );
        let n = 4;
        let m = 3;
        // 1 + 4ulp: 1.0 と区別できるが |s-1| < 1e-15 (スキップ条件の境界を確認)。
        let near_one = 1.0 + 4.0 * f64::EPSILON;
        let scales = [1.0_f64, 2.0, near_one, 0.25];

        // 旧手順オラクル (lsq.rs の元ロジックを忠実に再現)。
        let mut a_scaled = a.clone();
        for k in 0..n {
            let s = scales[k];
            if (s - 1.0).abs() < 1e-15 {
                continue;
            }
            let cs = a_scaled.col_ptr()[k];
            let ce = a_scaled.col_ptr()[k + 1];
            for idx in cs..ce {
                a_scaled.values_mut()[idx] *= s;
            }
        }
        let expected = build_aat_upper_csc(&a_scaled, n, m).expect("oracle aat");

        let fused = build_aat_upper_csc_scaled(&a, n, m, Some(&scales[..])).expect("fused aat");
        assert!(
            aat_csc_bit_eq(&expected, &fused),
            "scaled build must be bitwise identical to clone+scale+build"
        );

        // scale=None は無スケール版と完全一致。
        let unscaled = build_aat_upper_csc_scaled(&a, n, m, None).expect("unscaled fused");
        let plain = build_aat_upper_csc(&a, n, m).expect("plain aat");
        assert!(aat_csc_bit_eq(&plain, &unscaled));
    }
}
