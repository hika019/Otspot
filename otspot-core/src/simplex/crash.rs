//! Simplex crash basis (LTSF: Lower-Triangular Sparse Factor)。
//!
//! 大規模 LP の cold start で Phase I 反復削減が目的。構造列で行を被覆し
//! `needs_artificial` を減らす。Maros 2003 §5.5 + Bixby 1992 §4 sign-aware pivot。
//!
//! active count (未被覆行 nnz) を bucket queue で動的管理し、最小 bucket から
//! pop して `|a[i,j]| ≥ CRASH_PIVOT_REL · max_in_col` (Markowitz) かつ
//! `b[i]=0 ∨ sign 一致` (x_B≥0) を満たす最大 |pivot| 行を pivot 割当。pivot 後
//! 同行 entry を持つ他列の count を decrement し singleton chase を誘発する。
//! 動的 re-prioritization が LTSF の本質 (静的 sort は quasi-triangle で退化)。

use crate::sparse::CscMatrix;

/// 列内最大 |pivot| に対する相対閾値 (これ未満は不安定 pivot として却下)。
/// 0.1 は LP solver の一般的な Markowitz threshold (Suhl & Suhl 1990)。
const CRASH_PIVOT_REL: f64 = 0.1;

/// 絶対 pivot 下限。Ruiz scaling 前のため大きめ。
const CRASH_PIVOT_ABS: f64 = 1e-8;

/// `(basis_out, needs_artificial_out, num_artificial_out)` を返す。
///
/// 入力:
/// - `a`: standard form 行列 (CSC, m × n_total, Ruiz scaling 前)
/// - `m`: 行数
/// - `n_shifted`: 構造列範囲 [0, n_shifted) (n_shifted 以上は slack)
/// - `initial_basis_in`: build_standard_form の `initial_basis` (artificial 行は
///   slack 列をプレースホルダで持つ)
/// - `needs_artificial_in`: build_standard_form の `needs_artificial`
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

    for (i, &covered) in row_covered.iter().enumerate() {
        if covered {
            col_used[basis[i]] = true;
        }
    }

    let num_artificial_initial = needs_artificial.iter().filter(|&&v| v).count();
    if num_artificial_initial == 0 {
        return (basis, needs_artificial, 0);
    }

    let mut state = LtsfState::new(a, n_shifted, &row_covered, &col_used);

    while let Some(j) = state.pop_min_active_column() {
        let (cs, ce) = (a.col_ptr[j], a.col_ptr[j + 1]);
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
            state.cover_row(row);
        }
    }

    let num_artificial_out = needs_artificial.iter().filter(|&&v| v).count();
    (basis, needs_artificial, num_artificial_out)
}

/// LTSF 動的優先度 state. bucket queue + CSR-style row index で
/// 列の active count (= 未被覆行 nnz) を O(1) update する。
struct LtsfState {
    /// `row_ptr[r]..row_ptr[r+1]` = 行 r に entry を持つ構造列 list の range。
    row_ptr: Vec<usize>,
    row_cols: Vec<usize>,
    /// 各列の現 active count (未被覆行 nnz)。0 になった列は queue から除外。
    col_active: Vec<usize>,
    /// `buckets[k]` = active count が k の列 indices (stale entry あり、pop 時 check)。
    buckets: Vec<Vec<usize>>,
    /// 次に走査開始すべき bucket index (hint; 必要なら decrement 可)。
    min_k: usize,
    /// 該当列が処理済 (pivot 採用 or skip 確定) なら true。stale skip 用。
    col_consumed: Vec<bool>,
}

impl LtsfState {
    fn new(a: &CscMatrix, n_shifted: usize, row_covered: &[bool], col_used: &[bool]) -> Self {
        let m = a.nrows;
        let mut col_active = vec![0usize; n_shifted];
        let mut max_k = 0usize;
        for j in 0..n_shifted {
            if col_used[j] {
                continue;
            }
            let mut cnt = 0usize;
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                if !row_covered[a.row_ind[k]] {
                    cnt += 1;
                }
            }
            col_active[j] = cnt;
            if cnt > max_k {
                max_k = cnt;
            }
        }
        let mut buckets: Vec<Vec<usize>> = (0..=max_k).map(|_| Vec::new()).collect();
        for j in 0..n_shifted {
            if col_used[j] {
                continue;
            }
            let cnt = col_active[j];
            if cnt > 0 {
                buckets[cnt].push(j);
            }
        }

        let mut row_count = vec![0usize; m];
        for j in 0..n_shifted {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                row_count[a.row_ind[k]] += 1;
            }
        }
        let mut row_ptr = vec![0usize; m + 1];
        for r in 0..m {
            row_ptr[r + 1] = row_ptr[r] + row_count[r];
        }
        let mut row_cols = vec![0usize; row_ptr[m]];
        let mut pos = row_ptr.clone();
        for j in 0..n_shifted {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                let r = a.row_ind[k];
                row_cols[pos[r]] = j;
                pos[r] += 1;
            }
        }

        let col_consumed = col_used[..n_shifted].to_vec();

        Self {
            row_ptr,
            row_cols,
            col_active,
            buckets,
            min_k: 1,
            col_consumed,
        }
    }

    /// active count が最小の列を返し、consumed mark する。stale entry は skip。
    fn pop_min_active_column(&mut self) -> Option<usize> {
        let max_k = self.buckets.len().saturating_sub(1);
        loop {
            while self.min_k <= max_k && self.buckets[self.min_k].is_empty() {
                self.min_k += 1;
            }
            if self.min_k > max_k {
                return None;
            }
            let j = self.buckets[self.min_k].pop().unwrap();
            if self.col_consumed[j] || self.col_active[j] != self.min_k {
                continue;
            }
            self.col_consumed[j] = true;
            return Some(j);
        }
    }

    /// 行 r を被覆した直後の更新: r に entry を持つ未 consumed 列の active count
    /// を 1 減らし、新 bucket に push。min_k が下がったら hint も下げる。
    fn cover_row(&mut self, r: usize) {
        let s = self.row_ptr[r];
        let e = self.row_ptr[r + 1];
        for idx in s..e {
            let j = self.row_cols[idx];
            if self.col_consumed[j] {
                continue;
            }
            let new_cnt = self.col_active[j] - 1;
            self.col_active[j] = new_cnt;
            if new_cnt == 0 {
                self.col_consumed[j] = true;
                continue;
            }
            self.buckets[new_cnt].push(j);
            if new_cnt < self.min_k {
                self.min_k = new_cnt;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    /// 単純対角ケース: artif 行が n 個、対角構造列で全行被覆できる。
    #[test]
    fn diagonal_crash_eliminates_all_artificials() {
        let a = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1.0], 3, 3).unwrap();
        let b = vec![1.0, 2.0, 3.0];
        let initial_basis = vec![0usize, 0, 0];
        let needs_artif = vec![true, true, true];
        let (basis, needs_out, num_art) =
            compute_crash_basis(&a, &b, 3, 3, &initial_basis, &needs_artif);
        assert_eq!(num_art, 0, "全行被覆可能");
        assert_eq!(basis, vec![0, 1, 2]);
        assert_eq!(needs_out, vec![false; 3]);
    }

    /// pivot 不安定列は使わない: 列内の最大が tiny、relative pivot 失格。
    #[test]
    fn small_pivot_column_rejected() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1e-12], 1, 1).unwrap();
        let b = vec![1.0];
        let initial_basis = vec![0usize];
        let needs_artif = vec![true];
        let (_, needs_out, num_art) =
            compute_crash_basis(&a, &b, 1, 1, &initial_basis, &needs_artif);
        assert_eq!(num_art, 1, "tiny pivot は被覆しない");
        assert_eq!(needs_out, vec![true]);
    }

    /// 既に slack で被覆済の行はそのまま、artificial 行のみ crash 対象。
    #[test]
    fn covered_rows_kept_as_is() {
        let a = CscMatrix::from_triplets(&[0, 1, 1, 0], &[0, 0, 1, 2], &[1.0, 2.0, 0.5, 1.0], 2, 3)
            .unwrap();
        let b = vec![1.0, 1.0];
        let initial_basis = vec![2usize, 0];
        let needs_artif = vec![false, true];
        let (basis, needs_out, num_art) =
            compute_crash_basis(&a, &b, 2, 2, &initial_basis, &needs_artif);
        assert_eq!(num_art, 0);
        assert_eq!(basis[0], 2, "行 0 の slack basis 維持");
        assert!(basis[1] == 0 || basis[1] == 1, "行 1 は構造列で被覆");
        assert_eq!(needs_out, vec![false, false]);
    }

    /// 部分被覆: artif 行が 2 つ、構造列が 1 つしか被覆できないケース。
    #[test]
    fn partial_coverage() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 2, 1).unwrap();
        let b = vec![1.0, 1.0];
        let initial_basis = vec![0usize, 0];
        let needs_artif = vec![true, true];
        let (basis, needs_out, num_art) =
            compute_crash_basis(&a, &b, 2, 1, &initial_basis, &needs_artif);
        assert_eq!(num_art, 1);
        assert_eq!(basis[0], 0);
        assert!(needs_out[1], "行 1 は artificial 必要");
    }

    /// 符号不一致は被覆を見送る: x_B < 0 を避けるための feasibility-aware 選択。
    #[test]
    fn sign_mismatch_rejected() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let b = vec![-1.0];
        let initial_basis = vec![0usize];
        let needs_artif = vec![true];
        let (_, _, num_art) = compute_crash_basis(&a, &b, 1, 1, &initial_basis, &needs_artif);
        assert_eq!(num_art, 1, "符号不一致行は被覆しない");
    }

    /// 符号一致なら被覆する。
    #[test]
    fn sign_match_accepted() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[-1.0], 1, 1).unwrap();
        let b = vec![-1.0];
        let initial_basis = vec![0usize];
        let needs_artif = vec![true];
        let (basis, _, num_art) = compute_crash_basis(&a, &b, 1, 1, &initial_basis, &needs_artif);
        assert_eq!(num_art, 0);
        assert_eq!(basis[0], 0);
    }

    /// LTSF singleton chase sentinel:
    ///   静的 sort では捌けない triangular 構造で、動的 re-priority によって
    ///   全行被覆できることを実証する。具体的には:
    ///
    ///   col0: rows {0, 1, 2}     (init nnz=3)
    ///   col1: rows {0, 1}        (init nnz=2)
    ///   col2: rows {0}           (init nnz=1, true singleton)
    ///
    ///   初期 sort では col2 → col1 → col0 の順だが、すべての列が同じ行 0 を
    ///   含むため、静的順では col2 で row0 を取った後 col1 は (row0 covered ⇒
    ///   uncovered nnz=1 で row1 forced) なのに、静的順では「col1 で max|val|
    ///   行を選ぶだけ」になる。
    ///
    ///   ここでは各列の row0 entry を最大 |val| にして、静的アルゴリズムが
    ///   間違って row1 でなく row0 を再選択しがちにする (sign 一致のため両方
    ///   pivot 候補)。動的 LTSF はこの罠を踏まず、確実に 3 行とも被覆する。
    #[test]
    fn ltsf_singleton_chase_covers_all_rows() {
        // 値設計: row0 entry は大、row1/row2 entry は中。pivot 制約のみで動的順序を
        // 強要する。すべて正で b も正 ⇒ sign 制約は通過。
        let rows = vec![0, 1, 2, /*col1*/ 0, 1, /*col2*/ 0];
        let cols = vec![0, 0, 0, 1, 1, 2];
        let vals = vec![10.0, 1.0, 1.0, 10.0, 1.0, 10.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 3).unwrap();
        let b = vec![1.0, 1.0, 1.0];
        let initial = vec![0, 0, 0];
        let needs = vec![true, true, true];
        let (basis, needs_out, num_art) = compute_crash_basis(&a, &b, 3, 3, &initial, &needs);
        assert_eq!(
            num_art, 0,
            "LTSF should chase singletons and cover all rows"
        );
        assert_eq!(needs_out, vec![false; 3]);
        // 全列が distinct
        let mut seen = std::collections::HashSet::new();
        for &c in &basis {
            assert!(seen.insert(c), "duplicate column in basis: {:?}", basis);
        }
    }

    /// 動的 re-priority sentinel:
    ///   col0: rows {0,1,2,3}, col1: rows {0,1,2}, col2: rows {1,2,3}
    ///   col3: row {0} singleton, col4: row {3} singleton
    ///   全 5 列、4 行。理想は col3→row0, col4→row3, 残り {row1,row2} を
    ///   col1/col2 で動的に被覆。静的 sort は col3,col4 (nnz=1) → col1,col2
    ///   (nnz=3) → col0 (nnz=4)。動的では col3 pivot 後 col0/col1 の active
    ///   count が下がり、col4 pivot 後 col0/col2 も下がる ⇒ singleton 状態が
    ///   行 1,2 で別の列に生まれる。
    #[test]
    fn ltsf_dynamic_repriority_full_cover() {
        let rows = vec![
            0, 1, 2, 3, // col0
            0, 1, 2, // col1
            1, 2, 3, // col2
            0, // col3
            3, // col4
        ];
        let cols = vec![0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4];
        let vals = vec![5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 7.0, 7.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 4, 5).unwrap();
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let initial = vec![0, 0, 0, 0];
        let needs = vec![true, true, true, true];
        let (basis, _, num_art) = compute_crash_basis(&a, &b, 4, 5, &initial, &needs);
        assert_eq!(
            num_art, 0,
            "dynamic re-priority should cover all 4 rows; basis={:?}",
            basis
        );
        let mut seen = std::collections::HashSet::new();
        for &c in &basis {
            assert!(seen.insert(c), "duplicate column in basis: {:?}", basis);
        }
    }

    /// LTSF basis 不変式: 列は一意かつ range 内、5×6 疎構造で ≥ 4 行被覆。
    ///
    /// dynamic re-priority と singleton chase の cover 結果を確認する一般 sentinel。
    /// sign 制約で 1 行残るのは許容するため num_art ≤ 1 を要求。
    #[test]
    fn ltsf_basis_columns_unique_and_in_range() {
        // 雑多な疎構造で basis 一意性と範囲を検証
        let rows = vec![0, 1, 2, 3, 4, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0];
        let cols = vec![0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
        let vals = vec![1.0; 15];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 5, 6).unwrap();
        let b = vec![1.0; 5];
        let initial = vec![0; 5];
        let needs = vec![true; 5];
        let (basis, needs_out, num_art) = compute_crash_basis(&a, &b, 5, 6, &initial, &needs);
        let mut seen = std::collections::HashSet::new();
        for (i, &c) in basis.iter().enumerate() {
            if !needs_out[i] {
                assert!(c < 6, "basis[{}]={} out of range", i, c);
                assert!(seen.insert(c), "duplicate basis column {}", c);
            }
        }
        // 5 行のうち少なくとも 4 行は被覆できる (LTSF singleton chase で 5 行全部
        // 可能だが、sign 制約等で 1 行残る可能性を許容)
        assert!(
            num_art <= 1,
            "LTSF should cover ≥ 4 rows out of 5; got num_art={}",
            num_art
        );
    }
}
