//! CSC 格納から直接計算する行列ベクトル積 (明示的な転置を作らない)。
//!
//! `A.transpose().mat_vec_mul(y)` は評価ごとに転置 CSC (counting sort による
//! 新規確保 O(nrows + nnz)) を構築するが、反復内の `Aᵀy` 評価ではその転置は
//! 1 回しか使われない。[`mat_t_vec_mul_into`] は元の CSC の列内積で同じ値を
//! 直接計算するため転置構築が不要になる。

use super::view::CscMatrixView;
use crate::NumericError;
use rayon::prelude::*;
use std::sync::Arc;

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
    check_aty_dims(a, y, out)?;
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

/// `mat_t_vec_mul_into` / `mat_t_vec_mul_into_parallel` 共通の次元検証。
/// 逐次・並列で失敗挙動 (エラー型・field 名・`out` 非汚染) を divergence
/// させないため、判定は 1 箇所に置く。
fn check_aty_dims<M: CscMatrixView + ?Sized>(
    a: &M,
    y: &[f64],
    out: &[f64],
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
    Ok(())
}

/// `Aᵀy` 列分担並列化の nnz 下限 (ゲート)。
///
/// 根拠: 本工作ブランチのプローブ実測 (`aty_crossover` 計装、生データ
/// `/tmp/opencode/w2_postsolve/crossover_t2.tsv` / `crossover_t4.tsv`)。
/// 計測形状は production の Cauchy trial ループそのもの — solver pool
/// install の内側から同一プールへ install する反復呼び出し (best-of-2、
/// uniform 列負荷と 40% ホット列 1 本 skewed の両形状)。threads=2 の実測:
/// nnz≒1k で ×0.99〜1.14 (交差点)、2k で ×1.31〜1.42、4k で ×1.40〜1.63、
/// 14k 以上で ×1.55〜1.92。threads=4 では 4k で ×1.99〜2.37、630k で
/// ×2.19〜2.98 (ゲート 4k 時点の最小でも ×1.40 を確認)。
///
/// ゲートを交差点 (≒2k) より上側 4k に置くのは余裕代: 実ループでは matvec
/// 間に O(n+m) の逐次部が挟まり rayon ワーカーがコールド化して起こしコスト
/// が乗るため、hot ループ計測より交差点が右へ沈む。4k 未満ではゲート不発で
/// 逐次関数を直接呼ぶ (rayon 経由しない)。
///
/// 更新手順: 判定カーネルを触る変更では probe を再走し、ゲート nnz での
/// ratio が ×1.0 を下回ったら数値を上げて doc の実測値を改める。
pub const ATY_PAR_MIN_NNZ: usize = 4_000;

/// [`mat_t_vec_mul_into_parallel`] が列分担に使う計画 (列境界 + 実行プール)。
///
/// 境界は CSC の `col_ptr` (列 nnz の累積和そのもの) から 1 回だけ計算する。
/// `refine_dual_projected_gradient` のような反復呼び出しでは、Cauchy step ごと
/// に呼ばれる matvec とは別に、呼び出し開始時に 1 回だけ作って使い回す
/// (作成 O(ncols)、matvec 1 回 O(nnz))。
///
/// 生成条件 (いずれかを満たさなければ `None` = 呼び出し側は逐次
/// [`mat_t_vec_mul_into`] を使う — threads=1 では rayon を経由しないことを
/// 型ではなく生成側で保証する):
/// * `threads >= 2`
/// * `nnz >= `[`ATY_PAR_MIN_NNZ`]
/// * `ncols >= 2` (列単位分割が本体なので、単列では分割できない)
/// * solver プールの生成に成功する (`threads` 予算以上のプールを作らない —
///   `parallelism` モジュール doc の過剰割り当て禁止方針と同じ)
pub struct AtyParallelPlan {
    /// 分担セグメントの列境界 (長さ parts+1、単調非減少、先頭 0 / 末尾 ncols)。
    col_bounds: Vec<usize>,
    pool: Arc<rayon::ThreadPool>,
}

impl AtyParallelPlan {
    /// `a` の CSC 構造から `threads` 幅の列分担計画を作る (条件不平なら `None`)。
    pub fn new<M: CscMatrixView + ?Sized>(a: &M, threads: usize) -> Option<Self> {
        if threads <= 1 || a.values().len() < ATY_PAR_MIN_NNZ {
            return None;
        }
        let ncols = a.ncols();
        if ncols < 2 {
            return None;
        }
        let pool = crate::linalg::parallelism::solver_thread_pool(threads)?;
        Some(Self {
            col_bounds: split_columns_balanced(a.col_ptr(), threads.min(ncols)),
            pool,
        })
    }

    /// 列分担の境界 (検証・計装用。実カーネルは内部で読む)。
    pub fn column_bounds(&self) -> &[usize] {
        &self.col_bounds
    }

    /// `f` をこの計画の実行プール内で走らせる (プールの内部は公開しない)。
    ///
    /// 呼び出し側が Cauchy step のような反復本体ごとプール内で実行したい場合
    /// 用: 本体をプール外スレッドで走らせると step ごとに外側から
    /// [`mat_t_vec_mul_into_parallel`] の `install` が入り、ワーカーの
    /// 睡眠→起床コストが毎ステップ乗って逐次より遅くなる (実測 +13〜25%)。
    /// 本体を worker 上で実行すると内側 `install` は current-join になり、
    /// 呼び出し側ワーカー自身が計算に参加するため起床が実質 1 ワーカー分に
    /// 落ちる (交差点プローブが測った形状)。
    pub fn install_in_pool<F, T>(&self, f: F) -> T
    where
        F: Send + FnOnce() -> T,
        T: Send,
    {
        self.pool.install(f)
    }
}

/// 列 nnz が均荷重になるよう列番号空間を `parts` 個の連続区間に切る。
///
/// 重みは `col_ptr` の差分 (= 列 nnz)。境界は累積 nnz 目標に対する単走査で
/// 決めるため O(ncols + parts)。空区間 (列数 < parts など) を許す。
fn split_columns_balanced(col_ptr: &[usize], parts: usize) -> Vec<usize> {
    debug_assert!(col_ptr.len() >= 2, "CSC col_ptr has ncols+1 entries");
    debug_assert!(parts >= 2, "callers gate ncols >= 2 and pass parts >= 2");
    let items = col_ptr.len() - 1;
    let total = col_ptr[items];
    let mut bounds = Vec::with_capacity(parts + 1);
    bounds.push(0);
    let mut k = 0usize;
    for p in 1..parts {
        let target = ((total as u128 * p as u128) / parts as u128) as usize;
        while k < items && col_ptr[k + 1] < target {
            k += 1;
        }
        bounds.push(k);
    }
    bounds.push(items);
    bounds
}

/// [`mat_t_vec_mul_into`] の列分担並列版。**逐次版とビット単位で一致する**。
///
/// 各出力列 `j` の加算は `[col_ptr[j], col_ptr[j+1])` を CSC の格納順 (行
/// インデックス昇順) で行う — これは逐次版と 1 命令も変わらない同一ループで、
/// 列セグメントの割り当てが「どの列をどのスレッドが足すか」を変えるだけで
/// 列ごとの加算列には一切触らない。よってスレッド数・セグメント境界の
/// 作り方に依らず出力は逐次とビット一致となる (P4 spike 実測 408 結果行で
/// 検証済みの性質を production に移したもの)。
///
/// プラン生成条件 (threads / nnz 閾値 / 列数) は [`AtyParallelPlan::new`] が
/// 保証する。`plan` は同じ `a` に対して作ったものを使うこと (境界が列番号に
/// 紐づくため)。次元検証は逐次版と共通で、失敗時に `out` は汚染しない。
///
/// 実行は `plan` のプールへ `install` する。solve 経路 (with_solver_pool の
/// 中) から呼べば同一プールへの install なのでワーカー移行なし、プール外の
/// caller からも予算以上の並列度にはならない。
pub fn mat_t_vec_mul_into_parallel<M: CscMatrixView + ?Sized>(
    a: &M,
    y: &[f64],
    out: &mut [f64],
    plan: &AtyParallelPlan,
) -> Result<(), NumericError> {
    debug_assert_eq!(
        a.values().len(),
        a.col_ptr()[a.ncols()],
        "plan was built for a matrix shape the caller must not swap mid-loop"
    );
    check_aty_dims(a, y, out)?;
    let col_ptr = a.col_ptr();
    let row_ind = a.row_ind();
    let values = a.values();
    plan.pool.install(|| {
        let mut segs: Vec<(usize, &mut [f64])> = Vec::with_capacity(plan.col_bounds.len() - 1);
        let mut rest = out;
        for w in plan.col_bounds.windows(2) {
            let (head, tail) = rest.split_at_mut(w[1] - w[0]);
            segs.push((w[0], head));
            rest = tail;
        }
        segs.into_par_iter().for_each(|(j0, seg)| {
            for (idx, slot) in seg.iter_mut().enumerate() {
                let j = j0 + idx;
                let mut sum = 0.0;
                for k in col_ptr[j]..col_ptr[j + 1] {
                    sum += values[k] * y[row_ind[k]];
                }
                *slot = sum;
            }
        });
    });
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

    // ---------------------------------------------------------------------
    // 列分担並列版 (AtyParallelPlan / mat_t_vec_mul_into_parallel)
    // ---------------------------------------------------------------------

    /// テスト専用 LCG (決定的・外部依存なし)。rand 非依存で再現性を確保する。
    struct Lcg(u64);
    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn next_f64(&mut self) -> f64 {
            // [-1, 1) 決定的マッピング (上位 53 ビット使用)。
            let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            u * 2.0 - 1.0
        }
    }

    /// `rows_in_col` にソート済み行添字列を与えて CSC を直接構築する
    /// (from_triplets のソート・畳み込みを迂回し、格納順をテスト側で固定する)。
    fn csc_from_cols(
        nrows: usize,
        rows_in_col: &[Vec<usize>],
        mut value: impl FnMut(usize, usize) -> f64,
    ) -> CscMatrix {
        let mut col_ptr = Vec::with_capacity(rows_in_col.len() + 1);
        let mut row_ind = Vec::new();
        let mut values = Vec::new();
        col_ptr.push(0);
        for (j, rows) in rows_in_col.iter().enumerate() {
            for &r in rows {
                row_ind.push(r);
                values.push(value(j, r));
            }
            col_ptr.push(row_ind.len());
        }
        CscMatrix::from_raw_parts(nrows, rows_in_col.len(), col_ptr, row_ind, values)
    }

    /// 列平均 `avg` の uniform ランダム CSC (空列ありうる)。
    fn random_csc(rng: &mut Lcg, nrows: usize, ncols: usize, avg: usize) -> CscMatrix {
        let rows_in_col: Vec<Vec<usize>> = (0..ncols)
            .map(|_| {
                let k = rng.next_u64() % (avg as u64 + 1);
                let mut rows: Vec<usize> = (0..k)
                    .map(|_| (rng.next_u64() % nrows as u64) as usize)
                    .collect();
                rows.sort_unstable();
                rows.dedup();
                rows
            })
            .collect();
        csc_from_cols(nrows, &rows_in_col, |_, _| rng.next_f64())
    }

    /// parallel vs serial を全スロットビット一致で検証する共通アサーション。
    fn assert_parallel_bitwise_matches_serial(a: &CscMatrix, y: &[f64], threads: usize) {
        let plan = AtyParallelPlan::new(a, threads).expect("plan for gated matrix");
        assert!(
            plan.column_bounds().len() > 2 || a.ncols() < threads,
            "plan must actually split columns (bounds {:?})",
            plan.column_bounds()
        );
        let mut serial = vec![f64::NAN; a.ncols()];
        let mut par = vec![f64::NAN; a.ncols()];
        mat_t_vec_mul_into(a, y, &mut serial).unwrap();
        mat_t_vec_mul_into_parallel(a, y, &mut par, &plan).unwrap();
        assert_bits_eq(&par, &serial);
    }

    /// ゲート sentinel: threads・nnz・ncols の各条件で plan が生成される境界を
    /// 固定する。閾値比較を `>=` から `>` に崩す (または nnz 判定を消す) と
    /// 境界ケースで fail する。
    #[test]
    fn aty_parallel_plan_gate_sentinel() {
        let nnz = ATY_PAR_MIN_NNZ;
        // nnz == 閾値ちょうど → Some (threads=2)。ncols はゲート外側 (>=2)。
        let rows_in_col = vec![vec![0usize, 1]; nnz / 2];
        let at = csc_from_cols(2, &rows_in_col, |_, _| 1.0);
        assert_eq!(at.values().len(), nnz);
        assert!(
            AtyParallelPlan::new(&at, 2).is_some(),
            "nnz == gate must parallelize"
        );
        // nnz == 閾値 - 1 → None (逐次フォールバック)。
        let mut rows_short = rows_in_col.clone();
        rows_short.pop();
        rows_short.push(vec![0]);
        let short = csc_from_cols(2, &rows_short, |_, _| 1.0);
        assert_eq!(short.values().len(), nnz - 1);
        assert!(
            AtyParallelPlan::new(&short, 2).is_none(),
            "nnz < gate must stay serial"
        );
        // threads <= 1 は rayon に一切触らせない。
        assert!(AtyParallelPlan::new(&at, 0).is_none());
        assert!(
            AtyParallelPlan::new(&at, 1).is_none(),
            "threads=1 must be the serial path"
        );
        // 単列 (ncols < 2) は分割不能 → None (nnz が閾値を超えていても)。
        let single1 =
            CscMatrix::from_raw_parts(nnz, 1, vec![0, nnz], (0..nnz).collect(), vec![1.0; nnz]);
        assert!(
            AtyParallelPlan::new(&single1, 4).is_none(),
            "1 column cannot be column-split"
        );
    }

    /// 主流域 (uniform ランダム) + threads 2/4 で parallel は serial とビット一致。
    #[test]
    fn aty_parallel_bitwise_matches_serial_random() {
        let mut rng = Lcg(0x5eed_2026_1111);
        // nnz ≒ 平均 8 nnz × 6000 列 ≒ 45k → ゲートを明確に超える現実的疎構造。
        let a = random_csc(&mut rng, 4111, 6000, 16);
        assert!(a.values().len() >= ATY_PAR_MIN_NNZ);
        let y: Vec<f64> = (0..4111)
            .map(|i| rng.next_f64() * (i as f64 + 1.0))
            .collect();
        for threads in [2usize, 4] {
            assert_parallel_bitwise_matches_serial(&a, &y, threads);
        }
    }

    /// 歪んだ nnz 分布 (1 超ホット列 + 空列帯 + 微小列) と、非有限入力の混在。
    /// 列内積の順序保持だけでビット一致が言えるため、スレッド間で共有される
    /// 加算がある設計なら必ず崩れる形状を与える。
    #[test]
    fn aty_parallel_bitwise_matches_serial_skewed_and_nonfinite() {
        let mut rng = Lcg(0x00c0_ffee_1234);
        let nrows = 5000usize;
        let ncols = 4000usize;
        let hot_rows: Vec<usize> = (0..nrows).collect();
        let rows_in_col: Vec<Vec<usize>> = (0..ncols)
            .map(|j| match j {
                0 => hot_rows.clone(),
                1 => vec![],
                2 => vec![7],
                _ => {
                    let k = 2 + rng.next_u64() % 8;
                    let mut rows: Vec<usize> = (0..k)
                        .map(|_| (rng.next_u64() % nrows as u64) as usize)
                        .collect();
                    rows.sort_unstable();
                    rows.dedup();
                    rows
                }
            })
            .collect();
        let a = csc_from_cols(nrows, &rows_in_col, |j, r| {
            // 値にも非有限 (NaN/Inf) を決定的位置で混ぜる。
            match (j, r) {
                (0, 10) => f64::NAN,
                (0, 20) => f64::INFINITY,
                (2, 7) => f64::NEG_INFINITY,
                _ => rng.next_f64() * 1e3,
            }
        });
        assert!(a.values().len() >= ATY_PAR_MIN_NNZ);
        let mut y: Vec<f64> = (0..nrows).map(|_| rng.next_f64()).collect();
        y[10] = 1e300;
        y[20] = f64::INFINITY;
        y[99] = f64::NAN;
        for threads in [2usize, 4] {
            assert_parallel_bitwise_matches_serial(&a, &y, threads);
        }
    }

    /// 順序敏感性 sentinel: 加算順を入れ替えると値が変わる列を明示的に持つ。
    /// par 側が列内で並べ替える (または 2 分木還元にする) と fail する。
    #[test]
    fn aty_parallel_order_sensitive_column() {
        let nnz = ATY_PAR_MIN_NNZ;
        let mut rows_in_col: Vec<Vec<usize>> = vec![vec![0usize, 1]; nnz / 2 - 2];
        // 列 0 (行 0..4 に 1e300, 1.0, -1e300, 1e-20、y=1): 格納順では
        // ((1e300+1)-1e300)+1e-20 = 1e-20。逆順なら 0、2 分木還元
        // ((1e300+1)+(-1e300+1e-20)) でも 0。値アサートが逐次の加算順を
        // 固定し、par == serial アサートが列内の順序組み替えを検出する。
        rows_in_col.insert(0, vec![0, 1, 2, 3]);
        // 列 1: 1e-160 と 4e-13 の極端スケール同居 (加算吸収の形状確認)。
        rows_in_col.insert(1, vec![0, 1]);
        let mut values = Vec::new();
        let mut row_ind = Vec::new();
        let mut col_ptr = vec![0];
        for (j, rows) in rows_in_col.iter().enumerate() {
            for (i, &r) in rows.iter().enumerate() {
                row_ind.push(r);
                values.push(match (j, i) {
                    (0, 0) => 1e300,
                    (0, 1) => 1.0,
                    (0, 2) => -1e300,
                    (0, 3) => 1e-20,
                    (1, 0) => 1e-160,
                    (1, 1) => 4.0e-13,
                    _ => 0.25,
                });
            }
            col_ptr.push(row_ind.len());
        }
        let a = CscMatrix::from_raw_parts(4, rows_in_col.len(), col_ptr, row_ind, values);
        let y = [1.0_f64; 4];
        let mut out = vec![f64::NAN; a.ncols()];
        mat_t_vec_mul_into(&a, &y, &mut out).unwrap();
        assert_eq!(out[0].to_bits(), 1e-20_f64.to_bits(), "storage-order sum");
        assert_eq!(
            out[1].to_bits(),
            4.0e-13_f64.to_bits(),
            "absorbed tiny term"
        );
        let plan = AtyParallelPlan::new(&a, 2).expect("gated");
        let mut par = vec![f64::NAN; a.ncols()];
        mat_t_vec_mul_into_parallel(&a, &y, &mut par, &plan).unwrap();
        assert_bits_eq(&par, &out);
    }

    /// 独立オラクル (転置経由) と極端スケール・歪み形状でビット一致。
    /// 逐次版の既存 sentinel が結ぶ (逐次 == 転置) と本ファイルの
    /// (並列 == 逐次) から、並列 == 転置を独立路径経由で直接固定する。
    #[test]
    fn aty_parallel_bitwise_matches_transpose_oracle() {
        let mut rng = Lcg(0xdead_beef);
        let a = random_csc(&mut rng, 2500, 3000, 32);
        assert!(a.values().len() >= ATY_PAR_MIN_NNZ);
        let y: Vec<f64> = (0..2500).map(|_| rng.next_f64() * 1e50).collect();
        let expected = a.transpose().mat_vec_mul(&y).unwrap();
        let mut par = vec![f64::NAN; a.ncols()];
        let plan = AtyParallelPlan::new(&a, 2).expect("gated");
        mat_t_vec_mul_into_parallel(&a, &y, &mut par, &plan).unwrap();
        assert_bits_eq(&par, &expected);
    }

    /// 次元不一致の失敗挙動は逐次版と完全共通 (エラー一致・out 非汚染)。
    #[test]
    fn aty_parallel_rejects_dim_mismatch_like_serial() {
        let nnz = ATY_PAR_MIN_NNZ;
        let rows_in_col = vec![vec![0usize, 1]; nnz / 2];
        let a = csc_from_cols(2, &rows_in_col, |_, _| 1.0);
        let plan = AtyParallelPlan::new(&a, 2).expect("gated");
        let mut out = vec![7.0; a.ncols()];
        let err = mat_t_vec_mul_into_parallel(&a, &[1.0], &mut out, &plan).unwrap_err();
        let serr = mat_t_vec_mul_into(&a, &[1.0], &mut vec![0.0; a.ncols()]).unwrap_err();
        assert_eq!(format!("{err:?}"), format!("{serr:?}"));
        assert!(
            out.iter().all(|&v| v == 7.0),
            "out must stay untouched on dim error"
        );
    }

    /// 分割器の被覆性: どの threads でも境界は 0..=ncols を単調に被覆し、
    /// セグメント列数の総和 == ncols (空セグメントは許すが重複・欠落は禁止)。
    #[test]
    fn split_columns_balanced_covers_columns_exactly() {
        let mut rng = Lcg(777);
        for ncols in [2usize, 5, 64, 1000] {
            let a = random_csc(&mut rng, 300, ncols, 40);
            let cp = a.col_ptr().to_vec();
            for parts in [2usize, 3, 7, 64, 4096] {
                let parts_eff = parts.min(ncols);
                if parts_eff < 2 {
                    continue;
                }
                let b = split_columns_balanced(&cp, parts_eff);
                assert_eq!(b.len(), parts_eff + 1);
                assert_eq!(b[0], 0);
                assert_eq!(*b.last().unwrap(), ncols);
                assert!(b.windows(2).all(|w| w[0] <= w[1]), "monotone bounds");
            }
        }
    }
}
