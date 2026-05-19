//! Gershgorin 円定理ベースの λ_min(Q) 下界 helper。
//!
//! ## 公式
//! 対称行列 `Q` に対し `λ_min(Q) ≥ min_j (Q[j,j] − R_j)`、
//! ただし `R_j = Σ_{k≠j} |Q[j,k]|` (j 行非対角絶対値和)。
//!
//! ## 用途
//! - α-BB 凸化 (`qp::global::bound_alpha_bb::gershgorin_alpha`):
//!   `α = max(0, max_j(R_j − Q[j,j])) / 2`
//! - IPM 慣性修正 (`qp::ipm_core::kkt::compute_inertia_correction`):
//!   `δ_ic = max(0, max_j(R_j − Q[j,j]))`
//!
//! どちらも `max(0, max_j(R_j − Q[j,j])) = max(0, −λ_min) の Gershgorin 上界`
//! を共有するため本 helper に集約する。
//!
//! ## CSC 規約
//! 入力 `Q` は full-symmetric / 上三角どちらも許容。`row < col` の片半 entry を
//! 行と列双方の `R_j` に 1 度だけ反映するため、対称行列の R_j と一致する。
//! 対角は最後に書き込まれた値を採用 (CSC 慣例で 1 列 1 対角 entry を想定)。

use crate::sparse::CscMatrix;

/// `Q` の Gershgorin λ_min 下界から、PSD 化に必要な非負シフト
/// `max(0, max_j(R_j − Q[j,j])) = max(0, −λ_min_lower(Q))` を返す。
///
/// 戻り値 0 は「Gershgorin が `λ_min ≥ 0` を保証した = 何もシフト不要」を意味する
/// (Q が真に indefinite でも Gershgorin が保守的に non-negative を返す PSD 偽陽性は
/// 別途 `is_q_psd_by_cholesky` 等で扱う; 本 helper は素の Gershgorin に専念)。
pub(crate) fn psd_shift_from_gershgorin(q: &CscMatrix) -> f64 {
    let n = q.nrows;
    if n == 0 {
        return 0.0;
    }
    let mut diag = vec![0.0_f64; n];
    let mut row_offdiag_sum = vec![0.0_f64; n];
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            let val = q.values[k];
            if row == col {
                diag[col] = val;
            } else if row < col {
                let abs_val = val.abs();
                row_offdiag_sum[row] += abs_val;
                row_offdiag_sum[col] += abs_val;
            }
        }
    }
    let mut shift = 0.0_f64;
    for j in 0..n {
        let lower = diag[j] - row_offdiag_sum[j];
        if lower < 0.0 {
            shift = shift.max(-lower);
        }
    }
    shift
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upper_tri(n: usize, entries: &[(usize, usize, f64)]) -> CscMatrix {
        let rows: Vec<usize> = entries.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = entries.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<f64> = entries.iter().map(|&(_, _, v)| v).collect();
        CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
    }

    #[test]
    fn empty_matrix_returns_zero() {
        let q = CscMatrix::new(0, 0);
        assert_eq!(psd_shift_from_gershgorin(&q), 0.0);
    }

    #[test]
    fn diagonal_psd_returns_zero() {
        let q = upper_tri(2, &[(0, 0, 1.0), (1, 1, 2.0)]);
        assert_eq!(psd_shift_from_gershgorin(&q), 0.0);
    }

    #[test]
    fn diagonal_negative_returns_abs_min_diag() {
        // diag(-2, -3) → Gershgorin lower = (-2, -3), shift = 3
        let q = upper_tri(2, &[(0, 0, -2.0), (1, 1, -3.0)]);
        assert!((psd_shift_from_gershgorin(&q) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn pure_bilinear_zero_diag_off_one() {
        // Q=[[0,1],[1,0]]: row sums = (1,1), diag=(0,0), Gershgorin lower=(-1,-1), shift=1
        let q = upper_tri(2, &[(0, 1, 1.0)]);
        assert!((psd_shift_from_gershgorin(&q) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn full_symmetric_input_matches_upper() {
        // Full-symmetric: both (0,1) と (1,0) を入れても上三角と同 shift
        let q_full =
            CscMatrix::from_triplets(&[0, 1], &[1, 0], &[1.0, 1.0], 2, 2).unwrap();
        let q_upper = upper_tri(2, &[(0, 1, 1.0)]);
        assert_eq!(
            psd_shift_from_gershgorin(&q_full),
            psd_shift_from_gershgorin(&q_upper),
        );
    }

    #[test]
    fn mixed_zero_and_negative_diag() {
        // Q=[[0,1],[1,-1]]: diag=(0,-1), row sums=(1,1), Gershgorin=(-1,-2), shift=2
        let q = upper_tri(2, &[(0, 1, 1.0), (1, 1, -1.0)]);
        assert!((psd_shift_from_gershgorin(&q) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn extreme_offdiag_dominates_diag() {
        // Q[0,0]=2, Q[0,3]=3, rest 0 (n=4)
        // row sums = (3,0,0,3), diag=(2,0,0,0), Gershgorin=(-1,0,0,-3), shift=3
        let q = upper_tri(4, &[(0, 0, 2.0), (0, 3, 3.0)]);
        assert!((psd_shift_from_gershgorin(&q) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn psd_with_large_offdiag_returns_positive_shift_false_alarm() {
        // PD Q=[[1,1.1],[1.1,2]]: det=0.79 > 0 だが Gershgorin は λ_min ≥ -0.1 (誤検出)
        // helper は素 Gershgorin に専念し 0.1 を返す。caller 側で PSD 判定で吸収。
        let q = upper_tri(2, &[(0, 0, 1.0), (0, 1, 1.1), (1, 1, 2.0)]);
        assert!((psd_shift_from_gershgorin(&q) - 0.1).abs() < 1e-12);
    }
}
