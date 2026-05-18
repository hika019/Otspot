//! Simplex crash basis (#15 速度改善 F1).
//!
//! 大規模 LP (dfl001, ken-13/18, pds-20 等) で cold start の人工変数
//! Phase I が反復数の主因。構造列で行を被覆して `needs_artificial` を
//! 減らすことで Phase I の最適化対象が縮減される。
//!
//! アルゴリズム: Bixby-Maros 系 lower-triangular sparse factor (LTSF) 風
//! greedy crash + 符号一致 pivot 選択 (Bixby 1992 §4 "feasibility-aware crash")。
//!
//! 1. 既に slack で被覆できる行はそのまま (`needs_artificial[i] == false`)。
//! 2. 残った行 (= artificial 候補) について、構造列を以下の優先で巡回:
//!    - 列の NNZ が小 (少数行のみに影響、triangular 構造に乗せやすい)
//!    - tie-break: 列インデックス昇順
//! 3. 各構造列について、未被覆の artificial 行のうち以下を満たす行から
//!    最大 |pivot| を持つ行を選んで割り当て:
//!    - `|a[i,j]| >= CRASH_PIVOT_REL * max_in_col` (markowitz 安定性)
//!    - `b[i] == 0` または `sign(a[i,j]) == sign(b[i])` (x_B[i] ≈ b[i]/a[i,j] ≥ 0)
//! 4. 行は一度被覆されたら他の列に渡らない (triangular 不変)。
//!
//! 符号一致条件は Phase I の x_B ≥ 0 不変式を最大限尊重するための feasibility-aware
//! 選択。これにより crash 後の x_B = B^{-1}*b の負成分が大幅に減り、primal.rs 側の
//! partial revert ループが収束しやすくなる。

use crate::sparse::CscMatrix;

/// 列内最大 |pivot| に対する相対閾値 (これ未満は不安定 pivot として却下)。
/// 0.1 は LP solver の一般的な markowitz threshold (Suhl & Suhl 1990)。
const CRASH_PIVOT_REL: f64 = 0.1;

/// 絶対 pivot 下限。Ruiz scaling 前のため大きめ。
const CRASH_PIVOT_ABS: f64 = 1e-8;

/// 行被覆判定: crash 後の `num_artificial` (= 元 num_artificial − 置換成功数)
/// と更新済 basis/needs_artificial を返す。
///
/// 入力:
/// - `a`: standard form 行列 (CSC, m × n_total, ruiz scaling 前)
/// - `m`: 行数
/// - `n_shifted`: 構造列範囲 [0, n_shifted) (n_shifted 以上は slack)
/// - `initial_basis_in`: build_standard_form の `initial_basis` (artificial 行は
///   slack 列をプレースホルダで持つ)
/// - `needs_artificial_in`: build_standard_form の `needs_artificial`
///
/// 出力:
/// - `(basis_out, needs_artificial_out, num_artificial_out)`
pub(crate) fn compute_crash_basis(
    a: &CscMatrix,
    b: &[f64],
    m: usize,
    n_shifted: usize,
    initial_basis_in: &[usize],
    needs_artificial_in: &[bool],
) -> (Vec<usize>, Vec<bool>, usize) {
    debug_assert_eq!(initial_basis_in.len(), m);
    debug_assert_eq!(needs_artificial_in.len(), m);
    debug_assert_eq!(b.len(), m);

    let mut basis = initial_basis_in.to_vec();
    let mut needs_artificial = needs_artificial_in.to_vec();
    let mut row_covered: Vec<bool> = needs_artificial.iter().map(|&v| !v).collect();
    let mut col_used: Vec<bool> = vec![false; a.ncols];

    // slack 列は既に被覆行に対して使われているので除外。
    for (i, &covered) in row_covered.iter().enumerate() {
        if covered {
            col_used[basis[i]] = true;
        }
    }

    let num_artificial_initial = needs_artificial.iter().filter(|&&v| v).count();
    if num_artificial_initial == 0 {
        return (basis, needs_artificial, 0);
    }

    // 構造列の NNZ (artificial 候補行に限定) を集計し優先順位付け。
    let mut col_priority: Vec<(usize, usize)> = Vec::with_capacity(n_shifted);
    for j in 0..n_shifted {
        if col_used[j] {
            continue;
        }
        let mut nnz_in_artif = 0usize;
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let row = a.row_ind[k];
            if !row_covered[row] {
                nnz_in_artif += 1;
            }
        }
        if nnz_in_artif > 0 {
            col_priority.push((nnz_in_artif, j));
        }
    }
    col_priority.sort_unstable();

    for (_nnz, j) in col_priority {
        let cs = a.col_ptr[j];
        let ce = a.col_ptr[j + 1];

        // 列内最大 |entry| をスキャン (markowitz 安定性 threshold の分母)。
        let mut col_max_abs = 0.0_f64;
        for k in cs..ce {
            let v = a.values[k].abs();
            if v > col_max_abs {
                col_max_abs = v;
            }
        }
        if col_max_abs < CRASH_PIVOT_ABS {
            continue;
        }
        let pivot_min = (CRASH_PIVOT_REL * col_max_abs).max(CRASH_PIVOT_ABS);

        // 未被覆 artificial 行のうち |pivot| 最大、かつ符号一致 (b[i]=0 含む) の行。
        // 符号一致条件: x_B[i] ≈ b[i] / a[i,j] が ≥ 0 になる sign 関係。
        let mut best_row: Option<usize> = None;
        let mut best_abs = 0.0_f64;
        for k in cs..ce {
            let row = a.row_ind[k];
            if row_covered[row] {
                continue;
            }
            let val = a.values[k];
            let abs = val.abs();
            if abs < pivot_min {
                continue;
            }
            // sign(val) * sign(b[row]) >= 0 (b[row]=0 は許容)
            let bi = b[row];
            if bi != 0.0 && val.signum() != bi.signum() {
                continue;
            }
            if abs > best_abs {
                best_abs = abs;
                best_row = Some(row);
            }
        }
        if let Some(row) = best_row {
            basis[row] = j;
            needs_artificial[row] = false;
            row_covered[row] = true;
            col_used[j] = true;
        }
    }

    let num_artificial_out = needs_artificial.iter().filter(|&&v| v).count();
    (basis, needs_artificial, num_artificial_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    /// 単純対角ケース: artif 行が n 個、対角構造列で全行被覆できる。
    #[test]
    fn diagonal_crash_eliminates_all_artificials() {
        // 3 行 × 3 列 (構造列のみ、対角)
        let a = CscMatrix::from_triplets(
            &[0, 1, 2],
            &[0, 1, 2],
            &[1.0, 1.0, 1.0],
            3, 3,
        ).unwrap();
        let b = vec![1.0, 2.0, 3.0];
        let initial_basis = vec![0usize, 0, 0];
        let needs_artif = vec![true, true, true];
        let (basis, needs_out, num_art) = compute_crash_basis(
            &a, &b, 3, 3, &initial_basis, &needs_artif,
        );
        assert_eq!(num_art, 0, "全行被覆可能");
        assert_eq!(basis, vec![0, 1, 2]);
        assert_eq!(needs_out, vec![false; 3]);
    }

    /// pivot 不安定列は使わない: 列内の最大が tiny、relative pivot 失格。
    #[test]
    fn small_pivot_column_rejected() {
        // 1 行 × 1 列、値 1e-12 < CRASH_PIVOT_ABS
        let a = CscMatrix::from_triplets(
            &[0], &[0], &[1e-12], 1, 1,
        ).unwrap();
        let b = vec![1.0];
        let initial_basis = vec![0usize];
        let needs_artif = vec![true];
        let (_, needs_out, num_art) = compute_crash_basis(
            &a, &b, 1, 1, &initial_basis, &needs_artif,
        );
        assert_eq!(num_art, 1, "tiny pivot は被覆しない");
        assert_eq!(needs_out, vec![true]);
    }

    /// 既に slack で被覆済の行はそのまま、artificial 行のみ crash 対象。
    #[test]
    fn covered_rows_kept_as_is() {
        // 2 行 × 3 列。col 0,1 は構造、col 2 は slack (列 idx >= n_shifted=2)。
        // 行 0 は slack basis (col 2) で被覆済 (needs_artificial=false)。
        // 行 1 は artificial 行 (col 0 / col 1 で被覆可能)。
        let a = CscMatrix::from_triplets(
            &[0, 1, 1, 0], &[0, 0, 1, 2], &[1.0, 2.0, 0.5, 1.0],
            2, 3,
        ).unwrap();
        let b = vec![1.0, 1.0];
        let initial_basis = vec![2usize, 0];
        let needs_artif = vec![false, true];
        let (basis, needs_out, num_art) = compute_crash_basis(
            &a, &b, 2, 2, &initial_basis, &needs_artif,
        );
        assert_eq!(num_art, 0);
        assert_eq!(basis[0], 2, "行 0 の slack basis 維持");
        assert!(basis[1] == 0 || basis[1] == 1, "行 1 は構造列で被覆");
        assert_eq!(needs_out, vec![false, false]);
    }

    /// 部分被覆: artif 行が 2 つ、構造列が 1 つしか被覆できないケース。
    #[test]
    fn partial_coverage() {
        // 2 行 × 1 列。col 0 は行 0 のみに nonzero。行 1 は被覆不能。
        let a = CscMatrix::from_triplets(
            &[0], &[0], &[1.0], 2, 1,
        ).unwrap();
        let b = vec![1.0, 1.0];
        let initial_basis = vec![0usize, 0];
        let needs_artif = vec![true, true];
        let (basis, needs_out, num_art) = compute_crash_basis(
            &a, &b, 2, 1, &initial_basis, &needs_artif,
        );
        assert_eq!(num_art, 1);
        assert_eq!(basis[0], 0);
        assert!(needs_out[1], "行 1 は artificial 必要");
    }

    /// 符号不一致は被覆を見送る: x_B < 0 を避けるための feasibility-aware 選択。
    #[test]
    fn sign_mismatch_rejected() {
        // 1 行 × 1 列、a[0,0] = 1.0, b[0] = -1.0
        // → x_B = b/a = -1 < 0 になるため crash しない。
        let a = CscMatrix::from_triplets(
            &[0], &[0], &[1.0], 1, 1,
        ).unwrap();
        let b = vec![-1.0];
        let initial_basis = vec![0usize];
        let needs_artif = vec![true];
        let (_, _, num_art) = compute_crash_basis(
            &a, &b, 1, 1, &initial_basis, &needs_artif,
        );
        assert_eq!(num_art, 1, "符号不一致行は被覆しない");
    }

    /// 符号一致なら被覆する。
    #[test]
    fn sign_match_accepted() {
        let a = CscMatrix::from_triplets(
            &[0], &[0], &[-1.0], 1, 1,
        ).unwrap();
        let b = vec![-1.0];
        let initial_basis = vec![0usize];
        let needs_artif = vec![true];
        let (basis, _, num_art) = compute_crash_basis(
            &a, &b, 1, 1, &initial_basis, &needs_artif,
        );
        assert_eq!(num_art, 0);
        assert_eq!(basis[0], 0);
    }
}
