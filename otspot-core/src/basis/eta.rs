//! 修正シンプレックス法における基底更新のイータ因子分解
//!
//! 修正シンプレックス法では基底行列 B を直接保持する代わりに、
//! イータ行列 `E = I + (col - e_r) * e_r^T` の積で `B^{-1}` を逐次更新する。
//! このモジュールは疎なイータ行列の生成・蓄積・適用（FTRAN/BTRAN）を提供する。

use crate::tolerances::*;
use otspot_num::sparse::SparseVec;

/// nnz バジェットの倍率: リファクタ閾値 = `ETA_NNZ_BUDGET_MULTIPLIER × nnz(B)`。
///
/// 個数判定 (`max_etas`) だけでは eta 1 個の nnz (= fill-in を含んだ B⁻¹a_q) に
/// 上限がなく、最悪ケースの eta 常駐が `max_etas × m × 16B` (インデックス 8B +
/// 値 8B / nnz) に発散しうる。累積 nnz に nnz(B) の定数倍という上限制限を置く。
/// 乗数 8 の根拠: 実測したリファクタ直前 Σnnz_eta / nnz(B) 比は健全な疎問題で
/// ≤ 3.1 (bandm 3.03 / 25fv47 2.78 / 80bau3b 1.42 / n3700 0.60)、fill-heavy な
/// 問題だけ 13.9 (fome13) と一桁大きい。8 は後者で発火し前者では発火しない
/// 中間値 (2026-11-24 eta_diag、lp_problems 全件 + fome13 実測)。
pub(crate) const ETA_NNZ_BUDGET_MULTIPLIER: usize = 8;

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
/// リファクタ判定は2つの上限の OR:
/// - 個数: イータ行列が `max_etas` 個に到達
/// - 累積 nnz バジェット: 蓄積 nnz が `nnz_budget` (= 倍率 × nnz(B)) に到達
///
/// 後者は fill-in の大きい基底での eta 常駐を nnz(B) の定数倍に抑えるための
/// 上限制限。[`needs_refactor`] が `true` を返すと完全再因子分解のタイミング。
///
/// [`needs_refactor`]: EtaFile::needs_refactor
#[derive(Debug, Clone)]
pub(crate) struct EtaFile {
    pub etas: Vec<EtaMatrix>,
    pub max_etas: usize,
    /// 前回リファクタ以降の累積 eta nnz。[`push_eta`] 経由での追加分のみ計上。
    pub total_nnz: usize,
    /// 累積 nnz の上限。`usize::MAX` で無効 (個数判定のみ)。
    pub nnz_budget: usize,
}

impl EtaFile {
    /// 個数判定専用のファイル (バジェット無効)。production の基底管理は
    /// 必ず nnz バジェット付き [`with_budget`] を使うため、テスト専用。
    #[cfg(test)]
    pub fn new(max_etas: usize) -> Self {
        Self::with_budget(max_etas, usize::MAX)
    }

    pub fn with_budget(max_etas: usize, nnz_budget: usize) -> Self {
        Self {
            etas: Vec::new(),
            max_etas,
            total_nnz: 0,
            nnz_budget,
        }
    }

    /// イータを追加し、その nnz をバジェットカウンタに計上する。
    /// バジェット判定を使う経路は必ずこの経由で追加すること。
    pub fn push_eta(&mut self, eta: EtaMatrix) {
        self.total_nnz += eta.indices.len();
        self.etas.push(eta);
    }

    /// 蓄積イータと nnz カウンタをリセットする (リファクタ成功時)。
    pub fn clear(&mut self) {
        self.etas.clear();
        self.total_nnz = 0;
    }

    pub fn needs_refactor(&self) -> bool {
        self.etas.len() >= self.max_etas || self.total_nnz >= self.nnz_budget
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
///
/// `pivot_element` must be the value of `pivot_col` at `leaving_row`, already located and
/// validated (finite, non-zero) by the caller — this function does not re-search for it, so a
/// mismatched `pivot_element` silently corrupts the resulting eta matrix.
pub(crate) fn add_eta_sparse(
    pivot_col: &SparseVec,
    leaving_row: usize,
    pivot_element: f64,
) -> EtaMatrix {
    debug_assert!(pivot_element.is_finite() && pivot_element != 0.0);
    let inv_pivot = 1.0 / pivot_element;
    let mut indices = Vec::new();
    let mut values = Vec::new();

    // The leaving_row entry: 1/pivot
    if inv_pivot.abs() > ZERO_TOL {
        indices.push(leaving_row);
        values.push(inv_pivot);
    }

    // Other non-zero entries: -pivot_col[i] / pivot_element
    for (k, &idx) in pivot_col.indices().iter().enumerate() {
        if idx == leaving_row {
            continue;
        }
        let val = -pivot_col.values()[k] / pivot_element;
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

    /// SENTINEL (eta nnz バジェット案): 個数上限 (`max_etas`) に遠く届かない本数でも、
    /// 累積 nnz がバジェットに到達すればリファクタが発火する。個数判定だけの
    /// 旧実装 (revert 対象) ではこの発火は起きず、本テストは fail する。
    #[test]
    fn nnz_budget_triggers_refactor_below_count_cap() {
        // nnz 6 の eta を 1 本ずつ push_eta 経由で追加する。
        // pivot_col は index 0..6 すべて |v| > ZERO_TOL → eta の nnz はちょうど 6。
        let make_eta = |r: usize| add_eta(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], r % 6);
        let mut ef = EtaFile::with_budget(100, 50);
        for i in 0..8 {
            ef.push_eta(make_eta(i));
            // 累積 nnz = 6×(i+1) ≤ 48 < 50 かつ本数 ≤ 8 < 100 → 発火しない。
            assert!(!ef.needs_refactor(), "must not fire at {} etas", i + 1);
        }
        assert_eq!(ef.total_nnz, 48);
        ef.push_eta(make_eta(8)); // 累積 54 ≥ 50
        assert_eq!(ef.total_nnz, 54);
        assert!(
            ef.needs_refactor(),
            "accumulated nnz 54 >= budget 50 must trigger refactor at count 9 << max_etas 100"
        );
        ef.clear();
        assert_eq!(ef.total_nnz, 0);
        assert!(!ef.needs_refactor());
    }

    /// 個数上限は稀疏な eta 列に対する後方互換の上限として残り続ける
    /// (バジェットが未達でも `max_etas` 到達で発火)。
    #[test]
    fn count_cap_still_bounds_sparse_eta_runs() {
        let mut ef = EtaFile::with_budget(3, 1_000_000);
        ef.push_eta(add_eta(&[1.0], 0));
        ef.push_eta(add_eta(&[1.0], 0));
        assert!(!ef.needs_refactor());
        ef.push_eta(add_eta(&[1.0], 0));
        assert!(ef.needs_refactor());
        assert_eq!(ef.total_nnz, 3, "push_eta must account every eta's nnz");
    }

    #[test]
    fn test_eta_sparse_from_sparse_vec() {
        let sv = SparseVec::from_dense(&[2.0, 1.0, 0.5]);
        let eta = add_eta_sparse(&sv, 0, 2.0);

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

    /// `add_eta_sparse` must use the caller-supplied `pivot_element` verbatim and must not
    /// re-derive it from `pivot_col` internally. This is the contract that lets the sole caller
    /// (`basis::LuBasis::update`) reuse its own already-validated pivot lookup instead of
    /// duplicating a binary search. Passing a `pivot_element` that intentionally diverges from
    /// `pivot_col`'s stored value at `leaving_row` proves there is no hidden re-derivation: if one
    /// were reintroduced, this test's expected values (computed from the injected element) would
    /// no longer match.
    #[test]
    fn test_eta_sparse_uses_supplied_pivot_element_verbatim() {
        let sv = SparseVec::from_dense(&[2.0, 1.0, 0.5]);
        // Deliberately not 2.0 (the value actually stored at leaving_row=0 in `sv`).
        let injected_pivot_element = 4.0;
        let eta = add_eta_sparse(&sv, 0, injected_pivot_element);

        let val_at = |idx: usize| -> f64 {
            for (k, &i) in eta.indices.iter().enumerate() {
                if i == idx {
                    return eta.values[k];
                }
            }
            0.0
        };
        // 1/pivot_element and -pivot_col[i]/pivot_element, using the injected 4.0 (not the 2.0
        // that a re-derivation from `sv` would have produced).
        assert!((val_at(0) - 0.25).abs() < 1e-10);
        assert!((val_at(1) - (-0.25)).abs() < 1e-10);
        assert!((val_at(2) - (-0.125)).abs() < 1e-10);
    }
}
