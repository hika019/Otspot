//! CSC 格納から直接計算する行列ベクトル積 (明示的な転置を作らない)。
//!
//! `A.transpose().mat_vec_mul(y)` は評価ごとに転置 CSC (counting sort による
//! 新規確保 O(nrows + nnz)) を構築するが、反復内の `Aᵀy` 評価ではその転置は
//! 1 回しか使われない。[`mat_t_vec_mul_into`] は元の CSC の列内積で同じ値を
//! 直接計算するため転置構築が不要になる。

use super::view::CscMatrixView;
use crate::NumericError;

/// `out = Aᵀ y` を A の CSC 格納から直接計算する (列内積、呼び出し側確保の
/// `out` へ書き込み)。
///
/// 列 `j` ごとに `values[k] * y[row_ind[k]]` を列の格納順 (行添字の昇順、
/// CSC の不変条件) で 0.0 から加算して `out[j]` へ**上書き**する。
/// 列あたり独立の逐次加算なので、加算順は `a.transpose().mat_vec_mul(y)` の
/// 出力スロットごとの加算順 (転置 CSC が列順 scatter accumulation で元の行
/// インデックスの昇順に加算するのと同じ) と一致する。列内を行インデックス
/// 昇順で保持する CSC の不変条件により同一列の非ゼロの加算順は入れ替わら
/// ないため、結果は転置経由とビット単位で一致する。
///
/// 次元: `y.len() == a.nrows()`、`out.len() == a.ncols()`。不一致は
/// [`NumericError::DimensionMismatch`]。`a.nrows() == 0` のときは全列が
/// 空内積となり `out` は 0.0 で埋まる (転置経由と同一の挙動)。
pub fn mat_t_vec_mul_into<M: CscMatrixView + ?Sized>(
    a: &M,
    y: &[f64],
    out: &mut [f64],
) -> Result<(), NumericError> {
    if y.len() != a.nrows() {
        return Err(NumericError::DimensionMismatch {
            field: "mat_t_vec_mul_into/y",
            expected: a.nrows(),
            got: y.len(),
        });
    }
    if out.len() != a.ncols() {
        return Err(NumericError::DimensionMismatch {
            field: "mat_t_vec_mul_into/out",
            expected: a.ncols(),
            got: out.len(),
        });
    }
    let col_ptr = a.col_ptr();
    let row_ind = a.row_ind();
    let values = a.values();
    for j in 0..a.ncols() {
        let mut sum = 0.0;
        for k in col_ptr[j]..col_ptr[j + 1] {
            sum += values[k] * y[row_ind[k]];
        }
        out[j] = sum;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    /// 独立オラクル: 密行列 (行优先 Vec<Vec<f64>>) で `Aᵀy` を計算する。
    /// `from_triplets` の格納順に依存しない実装経路であり、本ファイルの
    /// カーネルとは共有コードを持たない。
    fn dense_aty(rows: &[usize], cols: &[usize], vals: &[f64], y: &[f64], n: usize) -> Vec<f64> {
        let m = y.len();
        let mut out = vec![0.0_f64; n];
        for i in 0..m {
            for j in 0..n {
                let mut aij = 0.0;
                for k in 0..rows.len() {
                    if rows[k] == i && cols[k] == j {
                        aij += vals[k];
                    }
                }
                out[j] += aij * y[i];
            }
        }
        out
    }

    fn assert_bits_eq(got: &[f64], expected: &[f64]) {
        assert_eq!(
            got.len(),
            expected.len(),
            "length mismatch: {got:?} vs {expected:?}"
        );
        for (i, (g, e)) in got.iter().zip(expected).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "index {i}: got {g:?} ({}) expected {e:?} ({})",
                g.to_bits(),
                e.to_bits()
            );
        }
    }

    /// 長方形 (2x3 / 3x2)・符号混在の手計算ケース。
    /// A = [[2, 0, -1], [0, 3, 4]] (2 行 3 列)、y = (5, -2):
    ///   Aᵀy = (2*5, 3*(-2), -1*5 + 4*(-2)) = (10, -6, -13)
    #[test]
    fn aty_rectangular_mixed_signs_hand_computed() {
        let a = CscMatrix::from_triplets(
            &[0, 1, 1, 0, 1],
            &[0, 0, 1, 2, 2],
            &[2.0, 0.0, 3.0, -1.0, 4.0],
            2,
            3,
        )
        .unwrap();
        let mut out = vec![-99.0; 3];
        mat_t_vec_mul_into(&a, &[5.0, -2.0], &mut out).unwrap();
        assert_bits_eq(&out, &[10.0, -6.0, -13.0]);

        // 転置形状 (3x2) でも同じ経路: Aₜ = [[2,0],[0,3],[0,4]]、yₜ=(1,-3,2)
        //   Aₜᵀyₜ = (2*1, 3*(-3) + 4*2) = (2, -1)
        let at = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 1], &[2.0, 3.0, 4.0], 3, 2).unwrap();
        let mut out2 = vec![-99.0; 2];
        mat_t_vec_mul_into(&at, &[1.0, -3.0, 2.0], &mut out2).unwrap();
        assert_bits_eq(&out2, &[2.0, -1.0]);
    }

    /// 空行・空列・ゼロ次元。
    #[test]
    fn aty_empty_rows_empty_cols_zero_dims() {
        // 空行 (row 1 に非ゼロなし) を含む 2x2。列0=(3 at row0)、列1=(-4 at row0)
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[3.0, -4.0], 2, 2).unwrap();
        let mut out = vec![-1.0; 2];
        mat_t_vec_mul_into(&a, &[2.0, 7.0], &mut out).unwrap();
        assert_bits_eq(&out, &[6.0, -8.0]);

        // 空行列 (nnz=0, 2x3): 出力は長さ ncols のゼロ。
        let zero = CscMatrix::from_triplets(&[], &[], &[], 2, 3).unwrap();
        let mut out = vec![-1.0; 3];
        mat_t_vec_mul_into(&zero, &[5.0, 6.0], &mut out).unwrap();
        assert_bits_eq(&out, &[0.0, 0.0, 0.0]);

        // ゼロ次元: 0x0 と 0x4。
        let mut out0 = Vec::new();
        mat_t_vec_mul_into(&CscMatrix::new(0, 0), &[], &mut out0).unwrap();
        assert!(out0.is_empty());
        let mut out4 = vec![7.0; 4];
        mat_t_vec_mul_into(&CscMatrix::new(0, 4), &[], &mut out4).unwrap();
        assert_bits_eq(&out4, &[0.0, 0.0, 0.0, 0.0]);

        // ncols=0 (Aᵀ の側が空): nrows>0 でも out は空。
        let no_cols = CscMatrix::new(3, 0);
        let mut out: Vec<f64> = Vec::new();
        mat_t_vec_mul_into(&no_cols, &[1.0, 2.0, 3.0], &mut out).unwrap();
        assert!(out.is_empty());
    }

    /// 極端なスケール混在でも転置経由とビット一致 (加算順序が同じであること)。
    /// 1e-160 級は `from_triplets` の DROP_TOL で落ちるため、明示保存ゼロと
    /// 同様に `from_raw_parts` で構造を直接与える。
    #[test]
    fn aty_extreme_scales_bitwise_matches_transpose_path() {
        // 1 列に 1e-160 〜 1e300 を同居させ、スロットごとの加算順を問う形状。
        let a = CscMatrix::from_raw_parts(
            3,
            3,
            vec![0, 3, 6, 8],
            vec![0, 1, 2, 0, 1, 2, 0, 2],
            vec![
                1e-160, -3.5e160, 7.0, 1e300, -1e299, 0.5, -2.0e-161, 4.0e-13,
            ],
        );
        for y in [
            vec![1.0, -1.0, 0.5],
            vec![1e-160, -4.0e150, 1e160],
            vec![0.0, 0.0, 0.0],
            vec![-0.0, 1e-300, 1e300],
        ] {
            let mut out = vec![f64::NAN; 3];
            mat_t_vec_mul_into(&a, &y, &mut out).unwrap();
            let expected = a.transpose().mat_vec_mul(&y).unwrap();
            assert_bits_eq(&out, &expected);
        }
    }

    /// 独立密オラクルとの一致 (符号混在・長方形)。`from_triplets` が畳み込み
    /// しない位置に重複のない COO を使うので密オラクルの足し合わせとも一致。
    #[test]
    fn aty_matches_independent_dense_oracle() {
        let rows = [0usize, 1, 2, 1, 0, 2];
        let cols = [0usize, 0, 1, 2, 3, 3];
        let vals = [1.25_f64, -0.75, 3.0, 2.5, -1.5, 0.125];
        let y = [0.5_f64, -2.0, 10.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 4).unwrap();
        let mut out = vec![f64::NAN; 4];
        mat_t_vec_mul_into(&a, &y, &mut out).unwrap();
        let expected = dense_aty(&rows, &cols, &vals, &y, 4);
        for (j, (g, e)) in out.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() <= 8.0 * f64::EPSILON * e.abs().max(1.0),
                "col {j}: got {g}, dense oracle {e}"
            );
        }
        // 手計算: 列0: 1.25*y0 + (-0.75)*y1 = 2.125 / 列1: 3*y2 = 30 /
        //         列2: 2.5*y1 = -5 / 列3: (-1.5)*y0 + 0.125*y2 = 0.5
        assert!(
            (out[0] - 2.125).abs() < 1e-12
                && (out[1] - 30.0).abs() < 1e-12
                && (out[2] + 5.0).abs() < 1e-12
                && (out[3] - 0.5).abs() < 1e-12,
            "hand-computed check failed: {out:?}"
        );
    }

    /// 転置経由 `transpose().mat_vec_mul` とスキャン形状を跨いでビット一致。
    /// 行ごとにまたがる加算順 (CSC 列内は行昇順) が両者で同じであることを
    /// 固定する sentinel: 列内積の順序を並び替えると fail する。
    #[test]
    fn aty_bitwise_matches_transpose_mat_vec_mul_scans() {
        let a = CscMatrix::from_triplets(
            &[0, 3, 0, 1, 2, 0, 2, 3, 1, 3, 0, 1, 2, 3],
            &[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4],
            &[
                1.0, -2.0, 3.0, -4.0, 5.0, 6.5, -7.25, 8.0, -9.0, 10.5, 11.0, -12.0, 13.0, -14.5,
            ],
            4,
            5,
        )
        .unwrap();
        let y = [1.5_f64, -0.25, 3.0, -7.0];
        let mut out = vec![f64::NAN; 5];
        mat_t_vec_mul_into(&a, &y, &mut out).unwrap();
        assert_bits_eq(&out, &a.transpose().mat_vec_mul(&y).unwrap());
    }

    /// 明示保存ゼロ (from_raw_parts で保持) でも転置経由とビット一致。
    /// from_triplets は DROP_TOL 未満を落とすが、転置は保存ゼロを保持する。
    #[test]
    fn aty_bitwise_matches_transpose_with_stored_zeros() {
        let a = CscMatrix::from_raw_parts(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![0.0, 5.0, -3.0, 0.0],
        );
        let y = [2.0_f64, -9.0];
        let mut out = vec![f64::NAN; 2];
        mat_t_vec_mul_into(&a, &y, &mut out).unwrap();
        assert_bits_eq(&out, &a.transpose().mat_vec_mul(&y).unwrap());
    }

    /// 次元不一致: y と out の双方で DimensionMismatch を返す。
    #[test]
    fn aty_rejects_dimension_mismatch() {
        let a = CscMatrix::identity(3);
        let mut out = vec![0.0; 3];
        let err = mat_t_vec_mul_into(&a, &[1.0, 2.0], &mut out).unwrap_err();
        assert!(matches!(
            err,
            NumericError::DimensionMismatch {
                expected: 3,
                got: 2,
                ..
            }
        ));
        let err = mat_t_vec_mul_into(&a, &[1.0, 2.0, 3.0], &mut [0.0, 0.0]).unwrap_err();
        assert!(matches!(
            err,
            NumericError::DimensionMismatch {
                expected: 3,
                got: 2,
                ..
            }
        ));
    }

    /// 失敗時に `out` を汚染しない (次元チェックが書き込みに先行する)。
    #[test]
    fn aty_leaves_out_untouched_on_error() {
        let a = CscMatrix::identity(3);
        let mut out = [7.0_f64; 2];
        assert!(mat_t_vec_mul_into(&a, &[1.0, 2.0], &mut out).is_err());
        assert_eq!(out, [7.0, 7.0]);
    }

    /// 一般 `CscMatrixView` 実装 (CscMatrix 以外) でも動作する。
    #[test]
    fn aty_accepts_any_csc_matrix_view() {
        struct View {
            ptr: Vec<usize>,
            rows: Vec<usize>,
            vals: Vec<f64>,
        }
        impl CscMatrixView for View {
            fn nrows(&self) -> usize {
                2
            }
            fn ncols(&self) -> usize {
                3
            }
            fn col_ptr(&self) -> &[usize] {
                &self.ptr
            }
            fn row_ind(&self) -> &[usize] {
                &self.rows
            }
            fn values(&self) -> &[f64] {
                &self.vals
            }
        }
        let v = View {
            ptr: vec![0, 1, 1, 3],
            rows: vec![1, 0, 1],
            vals: vec![4.0, -2.0, 0.5],
        };
        let mut out = vec![f64::NAN; 3];
        mat_t_vec_mul_into(&v, &[3.0, -6.0], &mut out).unwrap();
        // 列0: 4*y[1]=-24 / 列1: 空=0 / 列2: -2*y[0]+0.5*y[1]=-9
        assert_bits_eq(&out, &[-24.0, 0.0, -9.0]);
    }
}
