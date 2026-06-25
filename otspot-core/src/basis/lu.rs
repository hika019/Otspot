//! 疎LU分解モジュール (faer simplicial backend)
//!
//! 線形計画法の単体法 (改訂単体法) で用いる基底行列 B に対して、faer の
//! 疎LU分解 (`faer::sparse::linalg::lu`) を利用する。faer LU は COLAMD column
//! ordering と partial pivoting で fill-in を抑え、Markowitz より高速。
//!
//! 設計選択: symbolic factorize に `SupernodalThreshold::FORCE_SIMPLICIAL` を渡し
//! simplicial 経路を恒久的に強制する。faer 0.24.0 の AUTO ヒューリスティックは
//! 構造的特異基底で usize underflow を起こす (`factorize_timed` 内コメント参照) ため、
//! supernodal 自動選択は採用しない。ETA 機構 (`src/basis/eta.rs`) は LU の上に被せる
//! 更新層で、本 module の変更とは独立に動作する。

use crate::error::SolverError;
use crate::sparse::CscMatrix;

/// (col_ptr, row_ind, values) triple for a reconstructed basis CSC matrix.
type BasisCscParts = (Vec<usize>, Vec<usize>, Vec<f64>);
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::sparse::linalg::lu::{
    factorize_symbolic_lu, LuRef, LuSymbolicParams, NumericLu, SymbolicLu,
};
use faer::sparse::linalg::{LuError, SupernodalThreshold};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par};
use std::time::Instant;

/// LU分解の結果を保持する構造体。
///
/// faer の (`SymbolicLu`, `NumericLu`) ペアを保持し、必要時に `LuRef` を
/// 構築して solve に使う。基底次元 `n` を併せて保持。
#[derive(Debug, Clone)]
pub(crate) struct LuFactorization {
    pub(crate) symbolic: SymbolicLu<usize>,
    pub(crate) numeric: NumericLu<usize, f64>,
    pub(crate) n: usize,
}

impl LuFactorization {
    /// deadline 付き LU 分解。faer 自体は deadline 非対応のため、前後 2 段で
    /// チェックする (AMD wrapper と同じパターン)。
    pub(crate) fn factorize_timed(
        a: &CscMatrix,
        basis: &[usize],
        deadline: Option<Instant>,
    ) -> Result<Self, SolverError> {
        let m = basis.len();
        if m == 0 {
            return Err(SolverError::EmptyInput { context: "basis" });
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(SolverError::DeadlineExceeded);
        }

        let (col_ptr, row_ind, values) = build_basis_csc(a, basis, m)?;

        if !values.iter().all(|v| v.is_finite()) {
            return Err(SolverError::SingularBasis { step: 0 });
        }

        let a_sym = unsafe {
            SymbolicSparseColMatRef::<usize>::new_unchecked(m, m, &col_ptr, None, &row_ind)
        };
        // simplicial を明示的に強制する。`LuSymbolicParams::default()` の
        // `supernodal_flop_ratio_threshold` は AUTO (1.0) で、faer は supernodal/
        // simplicial 自動判定の col-count ヒューリスティックを走らせる。この経路
        // (faer 0.24.0 lu.rs:2245 `h_col_counts[parent] += h_col_counts[j] - 1`)
        // は構造的特異な基底 (空行を持つ m×m など) で usize 減算 underflow を起こし、
        // debug では panic、release では wrap して supernodal 判定を汚染する。
        // FORCE_SIMPLICIAL は当該ブロックをスキップして両方を根絶する。構造的特異な
        // 基底は numeric 段で SymbolicSingular として検出され SingularBasis に写る。
        // なお AUTO とは FP rounding が異なるため退化問題では simplex の pivot 系列が
        // 変わり得る (当該特異基底に到達しない経路を辿る場合もある) が、LP corpus
        // (feasible 109 / infeasible 29) で status・obj 不変を実測済み。
        let symbolic_params = LuSymbolicParams {
            supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SIMPLICIAL,
            ..Default::default()
        };
        let symbolic = factorize_symbolic_lu(a_sym, symbolic_params)
            .map_err(|_| SolverError::SingularBasis { step: 0 })?;

        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(SolverError::DeadlineExceeded);
        }

        let a_num = SparseColMatRef::<'_, usize, f64>::new(a_sym, &values);

        let mut numeric = NumericLu::<usize, f64>::new();
        let req = symbolic.factorize_numeric_lu_scratch::<f64>(Par::Seq, Default::default());
        let mut mem = MemBuffer::new(req);
        let stack = MemStack::new(&mut mem);

        symbolic
            .factorize_numeric_lu(&mut numeric, a_num, Par::Seq, stack, Default::default())
            .map_err(|e| match e {
                LuError::SymbolicSingular { index } => SolverError::SingularBasis { step: index },
                _ => SolverError::SingularBasis { step: 0 },
            })?;

        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(SolverError::DeadlineExceeded);
        }

        // 数値 singularity post-check: faer の partial pivoting は 0 pivot を
        // 許容して進む (LuError::SymbolicSingular は pivot 不在ケースのみ raise)
        // ため、Mehrotra IPM の Markowitz pivoting で検出されていた数値 singular
        // (test_lu_singular_detection の [1,2,3;4,5,6;1,2,3] 等) が後段に漏れる。
        // 単位 vector を 1 回 solve → 結果が非有限なら singular とみなす。
        // O(nnz) コストで factorize 全体のコストに対して 1% 未満。
        let mut probe = vec![0.0_f64; m];
        probe[0] = 1.0;
        let probe_req = symbolic.solve_in_place_scratch::<f64>(1, Par::Seq);
        let mut probe_mem = MemBuffer::new(probe_req);
        let probe_stack = MemStack::new(&mut probe_mem);
        let probe_lu = LuRef::new_unchecked(&symbolic, &numeric);
        let probe_mat = MatMut::from_column_major_slice_mut(&mut probe, m, 1);
        probe_lu.solve_in_place_with_conj(Conj::No, probe_mat, Par::Seq, probe_stack);
        if !probe.iter().all(|v| v.is_finite()) {
            return Err(SolverError::SingularBasis { step: 0 });
        }

        Ok(LuFactorization {
            symbolic,
            numeric,
            n: m,
        })
    }
}

/// 基底列を CSC 形式 (m×m) で再構築する。faer は列内 row 昇順を期待するため
/// (`SymbolicSparseColMatRef::new_unchecked` の前提)、列ごとにソートする。
fn build_basis_csc(a: &CscMatrix, basis: &[usize], m: usize) -> Result<BasisCscParts, SolverError> {
    let mut col_ptr = vec![0usize; m + 1];
    let mut row_ind: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    let mut tmp: Vec<(usize, f64)> = Vec::new();

    for (j, &col_idx) in basis.iter().enumerate() {
        if col_idx >= a.ncols {
            return Err(SolverError::IndexOutOfBounds {
                context: "basis_column",
                index: col_idx,
                bound: a.ncols,
            });
        }
        let start = a.col_ptr[col_idx];
        let end = a.col_ptr[col_idx + 1];
        tmp.clear();
        for k in start..end {
            let row = a.row_ind[k];
            if row < m {
                tmp.push((row, a.values[k]));
            }
        }
        tmp.sort_by_key(|&(r, _)| r);
        for &(r, v) in &tmp {
            row_ind.push(r);
            values.push(v);
        }
        col_ptr[j + 1] = row_ind.len();
    }
    Ok((col_ptr, row_ind, values))
}

/// FTRAN: `B × x = rhs` を LU 因子で解く。in-place で rhs を書き換える。
pub(crate) fn solve_ftran(lu: &LuFactorization, rhs: &mut [f64]) {
    let lu_ref = LuRef::new_unchecked(&lu.symbolic, &lu.numeric);
    let req = lu.symbolic.solve_in_place_scratch::<f64>(1, Par::Seq);
    let mut mem = MemBuffer::new(req);
    let stack = MemStack::new(&mut mem);
    let rhs_mat = MatMut::from_column_major_slice_mut(rhs, lu.n, 1);
    lu_ref.solve_in_place_with_conj(Conj::No, rhs_mat, Par::Seq, stack);
}

/// BTRAN: `B^T × x = rhs` を LU 因子で解く。in-place で rhs を書き換える。
pub(crate) fn solve_btran(lu: &LuFactorization, rhs: &mut [f64]) {
    let lu_ref = LuRef::new_unchecked(&lu.symbolic, &lu.numeric);
    let req = lu
        .symbolic
        .solve_transpose_in_place_scratch::<f64>(1, Par::Seq);
    let mut mem = MemBuffer::new(req);
    let stack = MemStack::new(&mut mem);
    let rhs_mat = MatMut::from_column_major_slice_mut(rhs, lu.n, 1);
    lu_ref.solve_transpose_in_place_with_conj(Conj::No, rhs_mat, Par::Seq, stack);
}

pub(crate) fn make_solve_scratch(lu: &LuFactorization) -> MemBuffer {
    let req_f = lu.symbolic.solve_in_place_scratch::<f64>(1, Par::Seq);
    let req_b = lu.symbolic.solve_transpose_in_place_scratch::<f64>(1, Par::Seq);
    MemBuffer::new(StackReq::any_of(&[req_f, req_b]))
}

pub(crate) fn solve_ftran_cached(lu: &LuFactorization, rhs: &mut [f64], scratch: &mut MemBuffer) {
    let req = lu.symbolic.solve_in_place_scratch::<f64>(1, Par::Seq);
    if scratch.len() < req.size_bytes() {
        *scratch = MemBuffer::new(req);
    }
    let lu_ref = LuRef::new_unchecked(&lu.symbolic, &lu.numeric);
    let stack = MemStack::new(scratch);
    let rhs_mat = MatMut::from_column_major_slice_mut(rhs, lu.n, 1);
    lu_ref.solve_in_place_with_conj(Conj::No, rhs_mat, Par::Seq, stack);
}

pub(crate) fn solve_btran_cached(lu: &LuFactorization, rhs: &mut [f64], scratch: &mut MemBuffer) {
    let req = lu.symbolic.solve_transpose_in_place_scratch::<f64>(1, Par::Seq);
    if scratch.len() < req.size_bytes() {
        *scratch = MemBuffer::new(req);
    }
    let lu_ref = LuRef::new_unchecked(&lu.symbolic, &lu.numeric);
    let stack = MemStack::new(scratch);
    let rhs_mat = MatMut::from_column_major_slice_mut(rhs, lu.n, 1);
    lu_ref.solve_transpose_in_place_with_conj(Conj::No, rhs_mat, Par::Seq, stack);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::test_utils::*;

    #[test]
    fn test_lu_identity() {
        let a = CscMatrix::identity(3);
        let basis = vec![0, 1, 2];
        let lu = LuFactorization::factorize_timed(&a, &basis, None).unwrap();

        for i in 0..3 {
            let mut rhs = vec![0.0; 3];
            rhs[i] = 1.0;
            let expected = rhs.clone();
            solve_ftran(&lu, &mut rhs);
            assert_vec_near(&rhs, &expected, 1e-10);
        }
    }

    #[test]
    fn test_lu_3x3() {
        let dense = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let lu = LuFactorization::factorize_timed(&a, &basis, None).unwrap();

        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs = rhs_orig.clone();
        solve_ftran(&lu, &mut rhs);

        let check = a.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_lu_4x4_sparse() {
        let dense = vec![
            vec![4.0, 0.0, 1.0, 0.0],
            vec![0.0, 3.0, 0.0, 2.0],
            vec![1.0, 0.0, 5.0, 0.0],
            vec![0.0, 1.0, 0.0, 6.0],
        ];
        let a = dense_to_csc(&dense, 4, 4);
        let basis = vec![0, 1, 2, 3];
        let lu = LuFactorization::factorize_timed(&a, &basis, None).unwrap();

        let rhs_orig = vec![5.0, 5.0, 6.0, 7.0];
        let mut rhs = rhs_orig.clone();
        solve_ftran(&lu, &mut rhs);

        let check = a.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_lu_btran() {
        let dense = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let lu = LuFactorization::factorize_timed(&a, &basis, None).unwrap();

        let rhs_orig = vec![3.0, 5.0, 3.0];
        let mut rhs = rhs_orig.clone();
        solve_btran(&lu, &mut rhs);

        let bt = a.transpose();
        let check = bt.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-10);
    }

    #[test]
    fn test_lu_singular_detection() {
        let dense = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![1.0, 2.0, 3.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let result = LuFactorization::factorize_timed(&a, &basis, None);
        assert!(result.is_err(), "Should detect singular matrix");
    }

    #[test]
    fn test_lu_markowitz() {
        let dense = vec![
            vec![0.001, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let lu = LuFactorization::factorize_timed(&a, &basis, None).unwrap();

        let rhs_orig = vec![1.001, 2.0, 2.0];
        let mut rhs = rhs_orig.clone();
        solve_ftran(&lu, &mut rhs);

        let check = a.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-8);
    }

    #[test]
    fn test_lu_ftran_btran_consistency() {
        let dense = vec![
            vec![3.0, 1.0, 0.0],
            vec![1.0, 4.0, 2.0],
            vec![0.0, 2.0, 5.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let basis = vec![0, 1, 2];
        let lu = LuFactorization::factorize_timed(&a, &basis, None).unwrap();

        let b = vec![1.0, 2.0, 3.0];
        let c = vec![4.0, 5.0, 6.0];

        let mut x = b.clone();
        solve_ftran(&lu, &mut x);

        let mut y = c.clone();
        solve_btran(&lu, &mut y);

        let xtc: f64 = x.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
        let bty: f64 = b.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (xtc - bty).abs() < 1e-10,
            "Adjoint property failed: x^T*c={} vs b^T*y={}",
            xtc,
            bty
        );
    }

    #[test]
    fn test_lu_20x20_sparse() {
        let n = 20;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(10.0 + (i as f64) * 0.5);
        }

        let off_diag: Vec<(usize, usize, f64)> = vec![
            (0, 3, 1.5),
            (0, 7, -0.8),
            (1, 5, 2.1),
            (1, 12, -1.3),
            (2, 8, 0.9),
            (2, 15, -0.4),
            (3, 0, -1.2),
            (3, 11, 0.7),
            (4, 9, 1.8),
            (4, 16, -0.6),
            (5, 1, -0.5),
            (5, 13, 1.1),
            (6, 2, 0.3),
            (6, 14, -1.9),
            (7, 0, 0.8),
            (7, 17, -0.7),
            (8, 2, -1.4),
            (8, 18, 0.6),
            (9, 4, 1.0),
            (9, 19, -0.3),
            (10, 6, -0.9),
            (10, 3, 1.7),
            (11, 3, -0.2),
            (11, 8, 0.5),
            (12, 1, 0.4),
            (12, 7, -1.6),
            (13, 5, -0.8),
            (13, 10, 1.3),
            (14, 6, 0.7),
            (14, 11, -1.1),
            (15, 2, 0.6),
            (15, 9, -0.5),
            (16, 4, -1.0),
            (16, 13, 0.9),
            (17, 7, 1.2),
            (17, 12, -0.4),
            (18, 8, -0.3),
            (18, 15, 1.5),
            (19, 9, 0.8),
            (19, 14, -0.6),
        ];
        for (r, c, v) in &off_diag {
            rows.push(*r);
            cols.push(*c);
            vals.push(*v);
        }

        let a = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        let basis: Vec<usize> = (0..n).collect();
        let lu = LuFactorization::factorize_timed(&a, &basis, None).unwrap();

        for k in 0..3 {
            let rhs_orig: Vec<f64> = (0..n).map(|i| ((i + k * 7) % 11) as f64 - 5.0).collect();
            let mut rhs = rhs_orig.clone();
            solve_ftran(&lu, &mut rhs);
            let check = a.mat_vec_mul(&rhs).unwrap();
            assert_vec_near(&check, &rhs_orig, 1e-8);
        }

        let bt = a.transpose();
        let rhs_orig: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3 - 3.0).collect();
        let mut rhs = rhs_orig.clone();
        solve_btran(&lu, &mut rhs);
        let check = bt.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-8);
    }

    #[test]
    fn test_lu_fill_in() {
        // Arrow matrix: dense first row/col causes fill-in
        let n = 8;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for j in 0..n {
            rows.push(0);
            cols.push(j);
            vals.push(1.0 + j as f64 * 0.3);
        }
        for i in 1..n {
            rows.push(i);
            cols.push(0);
            vals.push(0.5 + i as f64 * 0.2);
        }
        for i in 1..n {
            rows.push(i);
            cols.push(i);
            vals.push(10.0 + i as f64);
        }

        let a = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        let basis: Vec<usize> = (0..n).collect();
        let lu = LuFactorization::factorize_timed(&a, &basis, None).unwrap();

        // Verify correctness of FTRAN (内部表現は faer 内部、L/U 直接検査は不可)
        let rhs_orig: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut rhs = rhs_orig.clone();
        solve_ftran(&lu, &mut rhs);
        let check = a.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-8);

        let bt = a.transpose();
        let mut rhs2 = rhs_orig.clone();
        solve_btran(&lu, &mut rhs2);
        let check2 = bt.mat_vec_mul(&rhs2).unwrap();
        assert_vec_near(&check2, &rhs_orig, 1e-8);
    }

    #[test]
    fn test_lu_1x1() {
        let a = CscMatrix::from_triplets(&[0usize], &[0usize], &[5.0f64], 1, 1).unwrap();
        let basis = vec![0usize];
        let lu = LuFactorization::factorize_timed(&a, &basis, None).unwrap();

        assert_eq!(lu.n, 1);

        // solve_ftran: B * x = [1.0] → x = [0.2]
        let mut rhs = vec![1.0f64];
        solve_ftran(&lu, &mut rhs);
        assert!(
            (rhs[0] - 0.2).abs() < 1e-10,
            "ftran: expected 0.2, got {}",
            rhs[0]
        );

        let mut rhs2 = vec![1.0f64];
        solve_btran(&lu, &mut rhs2);
        assert!(
            (rhs2[0] - 0.2).abs() < 1e-10,
            "btran: expected 0.2, got {}",
            rhs2[0]
        );
    }

    /// 構造的特異 (空行を持つ) 基底で symbolic factorize が panic しないことを
    /// 守る load-bearing sentinel (#42)。`LuSymbolicParams::default()` (AUTO) では
    /// faer 0.24.0 の supernodal/simplicial 自動判定が usize underflow を起こし
    /// debug で panic する。`FORCE_SIMPLICIAL` 強制でこれを根絶し、構造的特異な
    /// 基底は SingularBasis として返ることを検査する。
    ///
    /// 入力 3 件は AUTO 下で実際に panic することを確認済みの 10×10 構造
    /// (いずれも行 9 が空 → 構造的特異)。本 fix を戻す (AUTO に戻す) と debug で
    /// panic し本 test は FAIL する。pilot (m=4722) の実発火基底と同じ性質を
    /// 小規模に再現したもの。
    #[test]
    fn test_lu_structurally_singular_no_panic() {
        // (col_ptr, row_ind): 各列の行は昇順。行 9 はどの列にも現れない (空行)。
        let cases: [(&[usize], &[usize]); 3] = [
            (
                &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                &[0, 8, 0, 1, 2, 3, 1, 3, 4, 8, 0, 5, 3, 6, 3, 7, 3, 8, 4, 6],
            ),
            (
                &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                &[0, 3, 1, 5, 2, 6, 1, 3, 4, 6, 5, 6, 6, 8, 4, 7, 4, 8, 2, 3],
            ),
            (
                &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                &[0, 8, 0, 1, 2, 7, 3, 8, 4, 8, 5, 8, 1, 6, 2, 7, 1, 8, 4, 6],
            ),
        ];
        let m = 10;
        for (idx, (col_ptr, row_ind)) in cases.iter().enumerate() {
            // 行 9 が空であること (= 構造的特異) を明示的に確認。
            assert!(
                !row_ind.contains(&(m - 1)),
                "case {idx}: row {} must be empty for structural singularity",
                m - 1
            );
            let a = CscMatrix {
                col_ptr: col_ptr.to_vec(),
                row_ind: row_ind.to_vec(),
                values: vec![1.0; row_ind.len()],
                nrows: m,
                ncols: m,
            };
            let basis: Vec<usize> = (0..m).collect();
            // 旧 (AUTO) 実装ではこの呼び出しが debug で panic していた。
            let result = LuFactorization::factorize_timed(&a, &basis, None);
            assert!(
                matches!(result, Err(SolverError::SingularBasis { .. })),
                "case {idx}: structurally singular basis must return SingularBasis, got {:?}",
                result.map(|_| "Ok")
            );
        }
    }

    #[test]
    fn test_lu_ill_conditioned() {
        let dense = vec![
            vec![1.0, 0.0, 0.0, 0.0, 1e-4],
            vec![0.0, 1.0, 0.0, 1e-4, 0.0],
            vec![0.0, 0.0, 1e-4, 0.0, 0.0],
            vec![0.0, 1e-4, 0.0, 1.0, 0.0],
            vec![1e-4, 0.0, 0.0, 0.0, 1.0],
        ];
        let a = dense_to_csc(&dense, 5, 5);
        let basis = vec![0, 1, 2, 3, 4];
        let lu = LuFactorization::factorize_timed(&a, &basis, None).unwrap();

        let rhs_orig = vec![1.0001, 1.0001, 0.0001, 1.0001, 1.0001];
        let mut rhs = rhs_orig.clone();
        solve_ftran(&lu, &mut rhs);
        let check = a.mat_vec_mul(&rhs).unwrap();
        assert_vec_near(&check, &rhs_orig, 1e-6);

        let bt = a.transpose();
        let mut rhs2 = rhs_orig.clone();
        solve_btran(&lu, &mut rhs2);
        let check2 = bt.mat_vec_mul(&rhs2).unwrap();
        assert_vec_near(&check2, &rhs_orig, 1e-6);
    }
}
