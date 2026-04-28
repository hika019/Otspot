//! IPM KKT 行列構築ユーティリティ
//!
//! - 拡張制約行列構築 (`build_extended_constraints`)
//! - augmented KKT system 構築 (`build_augmented_system`)
//! - 疎行列-ベクトル演算ヘルパー

use crate::problem::ConstraintType;
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;

// ---------------------------------------------------------------------------
// 疎行列-ベクトル演算
// ---------------------------------------------------------------------------

/// out = A * x（上書き）
#[inline]
#[allow(clippy::needless_range_loop)]
pub(crate) fn spmv(a: &CscMatrix, x: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for col in 0..a.ncols {
        let xv = x[col];
        for k in a.col_ptr[col]..a.col_ptr[col + 1] {
            out[a.row_ind[k]] += a.values[k] * xv;
        }
    }
}

/// out = A^T * v（上書き）
#[inline]
#[allow(clippy::needless_range_loop)]
pub(crate) fn spmtv(a: &CscMatrix, v: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|o| *o = 0.0);
    for col in 0..a.ncols {
        let mut s = 0.0;
        for k in a.col_ptr[col]..a.col_ptr[col + 1] {
            s += a.values[k] * v[a.row_ind[k]];
        }
        out[col] = s;
    }
}

/// out = Q * x（全要素格納の対称 Q に対応）
#[inline]
#[allow(clippy::needless_range_loop)]
pub(crate) fn spmv_q(q: &CscMatrix, x: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for col in 0..q.ncols {
        let xv = x[col];
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            out[q.row_ind[k]] += q.values[k] * xv;
        }
    }
}

/// ||v||_∞
#[inline]
pub(crate) fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

// ---------------------------------------------------------------------------
// 拡張制約行列構築
// ---------------------------------------------------------------------------

/// 制約行列を拡張する（Eq/Le/Geネイティブ対応）
///
/// 戻り値: (A_ext, b_ext, m_ext, m_orig, n_lb, is_eq_ext)
/// - is_eq_ext: 各拡張行が等式制約かどうか（true=等式、スラック不要）
/// - Ge行は符号反転してLe扱いで格納（is_eq_ext=false）
///   順序: [original constraints | lower bound rows | upper bound rows]
pub(crate) fn build_extended_constraints(
    problem: &QpProblem,
) -> (CscMatrix, Vec<f64>, usize, usize, usize, Vec<bool>) {
    let n = problem.num_vars;
    let m = problem.num_constraints;

    let n_lb: usize = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    let n_ub: usize = problem
        .bounds
        .iter()
        .filter(|&&(_, ub)| ub.is_finite())
        .count();
    let m_ext = m + n_lb + n_ub;

    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let mut b_ext = Vec::with_capacity(m_ext);
    let mut is_eq_ext = Vec::with_capacity(m_ext);

    // 元の制約（Eq/Le/Geをネイティブ処理）
    for col in 0..n {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let row = problem.a.row_ind[k];
            let sign = if problem.constraint_types[row] == ConstraintType::Ge {
                -1.0
            } else {
                1.0
            };
            rows.push(row);
            cols.push(col);
            vals.push(problem.a.values[k] * sign);
        }
    }
    for i in 0..m {
        match problem.constraint_types[i] {
            ConstraintType::Eq => {
                b_ext.push(problem.b[i]);
                is_eq_ext.push(true);
            }
            ConstraintType::Le => {
                b_ext.push(problem.b[i]);
                is_eq_ext.push(false);
            }
            ConstraintType::Ge => {
                // Ge: -a·x <= -b として格納
                b_ext.push(-problem.b[i]);
                is_eq_ext.push(false);
            }
        }
    }

    // 下界制約: x_j >= lb_j → -x_j <= -lb_j
    let mut lb_row = m;
    for (j, &(lb, _)) in problem.bounds.iter().enumerate() {
        if lb.is_finite() {
            rows.push(lb_row);
            cols.push(j);
            vals.push(-1.0);
            b_ext.push(-lb);
            is_eq_ext.push(false);
            lb_row += 1;
        }
    }

    // 上界制約: x_j <= ub_j
    let mut ub_row = m + n_lb;
    for (j, &(_, ub)) in problem.bounds.iter().enumerate() {
        if ub.is_finite() {
            rows.push(ub_row);
            cols.push(j);
            vals.push(1.0);
            b_ext.push(ub);
            is_eq_ext.push(false);
            ub_row += 1;
        }
    }

    let a_ext = if m_ext == 0 || rows.is_empty() {
        CscMatrix::new(0, n)
    } else {
        CscMatrix::from_triplets(&rows, &cols, &vals, m_ext, n).unwrap()
    };

    (a_ext, b_ext, m_ext, m, n_lb, is_eq_ext)
}

/// 拡張dualベクトルから元制約空間(m_orig長)のdualを復元する
///
/// 境界行を除去し、Ge行の符号を反転して元の制約空間に戻す
pub(crate) fn collapse_extended_dual(
    dual: &[f64],
    m_orig: usize,
    constraint_types: &[ConstraintType],
) -> Vec<f64> {
    let mut result = Vec::with_capacity(m_orig);
    for i in 0..m_orig {
        let d = dual[i];
        if constraint_types[i] == ConstraintType::Ge {
            result.push(-d);
        } else {
            result.push(d);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// augmented KKT system 構築
// ---------------------------------------------------------------------------

/// augmented KKT system の上三角 CSC を構築する
///
/// ```text
/// [ Q + δ_p·I     A_ext^T        ] [dx]   [rx]
/// [ A_ext      -(Σ + δ_d·I)      ] [dy] = [ry]
/// ```
///
/// サイズ: (n + m_ext) × (n + m_ext)。上三角のみ CSC として返す。
///
#[allow(clippy::needless_range_loop)]
pub(crate) fn build_augmented_system(
    q: &CscMatrix,
    a_ext: &CscMatrix,
    sigma_vec: &[f64],
    delta_p: f64,
    delta_d: f64,
) -> CscMatrix {
    let n = q.nrows;
    let m_ext = a_ext.nrows;
    let total = n + m_ext;

    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    // Part 1: Q + δ_p·I (上三角のみ)
    let mut diag_added = vec![false; n];
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            if row <= col {
                let v = q.values[k] + if row == col { delta_p } else { 0.0 };
                rows.push(row);
                cols.push(col);
                vals.push(v);
                if row == col {
                    diag_added[col] = true;
                }
            }
        }
    }
    for i in 0..n {
        if !diag_added[i] {
            rows.push(i);
            cols.push(i);
            vals.push(delta_p);
        }
    }

    // Part 2: A_ext^T ブロック（右上、row < col 保証）
    for j in 0..n {
        for idx in a_ext.col_ptr[j]..a_ext.col_ptr[j + 1] {
            let k = a_ext.row_ind[idx];
            let v = a_ext.values[idx];
            rows.push(j);
            cols.push(n + k);
            vals.push(v);
        }
    }

    // Part 3: -(Σ + δ_d)·I 対角ブロック（インデックス n..n+m_ext）
    for k in 0..m_ext {
        rows.push(n + k);
        cols.push(n + k);
        vals.push(-(sigma_vec[k] + delta_d));
    }

    if rows.is_empty() {
        CscMatrix::new(total, total)
    } else {
        CscMatrix::from_triplets(&rows, &cols, &vals, total, total).unwrap()
    }
}

// ---------------------------------------------------------------------------
// KKT 差分更新キャッシュ
// ---------------------------------------------------------------------------

/// KKT 行列の差分更新キャッシュ。
///
/// augmented system のスパースパターンは IPM 反復間で完全固定。
/// 初回構築後は values のみを更新することで `from_triplets` の O(nnz log nnz) を回避する。
pub(crate) struct KktCache {
    /// 初回 `build_augmented_system` で構築した CSC 行列（パターン固定）
    pub mat: CscMatrix,
    /// Part 1 対角要素の mat.values インデックス（n 個）
    pub part1_diag_idx: Vec<usize>,
    /// Q 対角の元値（`q_diag_base[i] + delta_p` で書き込む）
    pub q_diag_base: Vec<f64>,
    /// Part 3 対角の mat.values インデックス（m_ext 個）
    pub part3_diag_idx: Vec<usize>,
    /// 更新対象の列インデックス（全 n 列）
    pub part1_updated_idx: Vec<usize>,
}

/// augmented KKT 行列の Part 1（Q + δ_p·I）対角要素の values インデックスを収集する。
///
/// 上三角 CSC では列 i の対角要素（row = i）が列内最大行インデックスとなるため
/// `col_ptr[i+1] - 1` の位置にある。
pub(crate) fn collect_part1_diag_indices(aug_mat: &CscMatrix, n: usize) -> Vec<usize> {
    (0..n).map(|i| aug_mat.col_ptr[i + 1] - 1).collect()
}

/// augmented KKT 行列の Part 3（-(Σ + δ_d·I)）対角要素の values インデックスを収集する。
///
/// 列 n+k の対角要素（row = n+k）は列内最大行インデックスなので `col_ptr[n+k+1] - 1`。
pub(crate) fn collect_part3_diag_indices(
    aug_mat: &CscMatrix,
    n: usize,
    m_ext: usize,
) -> Vec<usize> {
    (0..m_ext).map(|k| aug_mat.col_ptr[n + k + 1] - 1).collect()
}

/// Q の対角要素値を収集する（delta_p 更新のベースライン）。
///
/// Q に対角要素がない列は 0.0 とする。
pub(crate) fn collect_q_diag_base(q: &CscMatrix, n: usize) -> Vec<f64> {
    let mut base = vec![0.0f64; n];
    for (col, val) in base.iter_mut().enumerate().take(n) {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            if q.row_ind[k] == col {
                *val = q.values[k];
                break;
            }
        }
    }
    base
}

/// `KktCache` の mat.values を差分更新する（O(n + m_ext)）。
///
/// `build_augmented_system` の代替として IPM ループ内で呼ぶ。
/// スパースパターンは不変なので values のみ更新する。
pub(crate) fn update_augmented_values(
    cache: &mut KktCache,
    sigma_vec: &[f64],
    delta_p: f64,
    delta_d: f64,
) {
    // Part 1: Q_ii + delta_p を更新（n 要素）
    for &i in &cache.part1_updated_idx {
        let idx = cache.part1_diag_idx[i];
        cache.mat.values[idx] = cache.q_diag_base[i] + delta_p;
    }
    // Part 3: -(sigma_k + delta_d) を更新（m_ext 要素）
    for (k, &idx) in cache.part3_diag_idx.iter().enumerate() {
        cache.mat.values[idx] = -(sigma_vec[k] + delta_d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 上三角 CscMatrix をエントリリストから構築するヘルパー
    fn upper_tri_csc(n: usize, entries: &[(usize, usize, f64)]) -> CscMatrix {
        let rows: Vec<usize> = entries.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = entries.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<f64> = entries.iter().map(|&(_, _, v)| v).collect();
        CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
    }

    /// collect_part1_diag_indices: n=2 の小規模問題でインデックス正確性を確認
    #[test]
    fn test_collect_part1_diag_indices() {
        // Q: 2x2 対称行列 [[2.0, 0.5], [0.5, 3.0]]（上三角格納）
        let q = upper_tri_csc(2, &[(0, 0, 2.0), (0, 1, 0.5), (1, 1, 3.0)]);
        // A_ext: 2x2 単位行列
        let a_rows = vec![0usize, 1];
        let a_cols = vec![0usize, 1];
        let a_vals = vec![1.0f64, 1.0];
        let a_ext = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, 2, 2).unwrap();

        let sigma_vec = [0.5f64, 0.8];
        let aug = build_augmented_system(&q, &a_ext, &sigma_vec, 0.1, 0.05);
        let n = 2;

        let diag_idx = collect_part1_diag_indices(&aug, n);
        // 各列の対角要素が col_ptr[i+1]-1 の位置にあることを確認
        for i in 0..n {
            let idx = diag_idx[i];
            assert_eq!(
                aug.row_ind[idx], i,
                "列 {i} の対角インデックス aug.row_ind[{idx}]={} が対角でない",
                aug.row_ind[idx]
            );
        }
        // 値も確認: Q[0,0]+delta_p = 2.1, Q[1,1]+delta_p = 3.1
        assert!((aug.values[diag_idx[0]] - 2.1).abs() < 1e-14);
        assert!((aug.values[diag_idx[1]] - 3.1).abs() < 1e-14);
    }

    /// collect_part3_diag_indices: m_ext=2 の小規模問題で対角インデックス正確性を確認
    #[test]
    fn test_collect_part3_diag_indices() {
        let q = upper_tri_csc(2, &[(0, 0, 2.0), (0, 1, 0.5), (1, 1, 3.0)]);
        let a_rows = vec![0usize, 1];
        let a_cols = vec![0usize, 1];
        let a_vals = vec![1.0f64, 1.0];
        let a_ext = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, 2, 2).unwrap();

        let sigma_vec = [0.5f64, 0.8];
        let aug = build_augmented_system(&q, &a_ext, &sigma_vec, 0.1, 0.05);
        let n = 2;
        let m_ext = 2;

        let diag_idx = collect_part3_diag_indices(&aug, n, m_ext);
        for k in 0..m_ext {
            let idx = diag_idx[k];
            assert_eq!(
                aug.row_ind[idx],
                n + k,
                "Part3 列 {k} の対角インデックス aug.row_ind[{idx}]={} が対角でない",
                aug.row_ind[idx]
            );
        }
        // 値: -(sigma[0]+delta_d) = -0.55, -(sigma[1]+delta_d) = -0.85
        assert!((aug.values[diag_idx[0]] - (-0.55)).abs() < 1e-14);
        assert!((aug.values[diag_idx[1]] - (-0.85)).abs() < 1e-14);
    }

    /// spmv: 2x2行列ベクトル積の正確性確認
    #[test]
    fn test_spmv_basic() {
        // A = [[1,2],[3,4]] (CSC形式), x = [1, 1] → Ax = [3, 7]
        let rows = vec![0usize, 1, 0, 1];
        let cols = vec![0usize, 0, 1, 1];
        let vals = vec![1.0f64, 3.0, 2.0, 4.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 2, 2).unwrap();
        let x = [1.0f64, 1.0];
        let mut out = [0.0f64, 0.0];
        spmv(&a, &x, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-14, "out[0]={}", out[0]);
        assert!((out[1] - 7.0).abs() < 1e-14, "out[1]={}", out[1]);
    }

    /// norm_inf: 各種入力での無限ノルム確認
    #[test]
    fn test_norm_inf_basic() {
        assert!((norm_inf(&[3.0, -5.0, 2.0]) - 5.0).abs() < 1e-15);
        assert!((norm_inf(&[0.0, 0.0]) - 0.0).abs() < 1e-15);
        assert!((norm_inf(&[-7.0]) - 7.0).abs() < 1e-15);
    }

    /// build_extended_constraints: 境界あり問題での行列サイズ確認
    #[test]
    fn test_build_extended_constraints_dimensions() {
        use crate::qp::problem::QpProblem;
        // QP: 2変数, Le制約1本: x0 + x1 <= 3
        // bounds: x0 in [0, inf), x1 in [0, inf) → n_lb=2, n_ub=0
        // m_ext = 1 + 2 + 0 = 3
        let a = CscMatrix::from_triplets(&[0usize, 0], &[0usize, 1], &[1.0f64, 1.0], 1, 2).unwrap();
        let q = CscMatrix::new(2, 2);
        let prob = QpProblem::new_all_le(
            q,
            vec![0.0; 2],
            a,
            vec![3.0],
            vec![(0.0, f64::INFINITY); 2],
        ).unwrap();
        let (a_ext, b_ext, m_ext, m_orig, n_lb, is_eq_ext) = build_extended_constraints(&prob);
        assert_eq!(m_orig, 1, "m_orig should be 1");
        assert_eq!(n_lb, 2, "n_lb should be 2 (both lower bounds finite)");
        assert_eq!(m_ext, 3, "m_ext = m + n_lb + n_ub = 1+2+0 = 3");
        assert_eq!(a_ext.nrows, 3, "a_ext.nrows should be m_ext=3");
        assert_eq!(a_ext.ncols, 2, "a_ext.ncols should be n=2");
        assert_eq!(b_ext.len(), 3, "b_ext.len() should be m_ext=3");
        assert_eq!(is_eq_ext, vec![false, false, false], "all Le constraints");
    }

    /// update_augmented_values: 更新後の values が build_augmented_system 結果と一致
    #[test]
    fn test_update_augmented_values_matches_build() {
        let q = upper_tri_csc(2, &[(0, 0, 2.0), (0, 1, 0.5), (1, 1, 3.0)]);
        let a_rows = vec![0usize, 1];
        let a_cols = vec![0usize, 1];
        let a_vals = vec![1.0f64, 1.0];
        let a_ext = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, 2, 2).unwrap();

        let n = 2;
        let m_ext = 2;
        // 初回構築（delta_p=0.1, delta_d=0.05, sigma=[0.5, 0.8]）
        let sigma1 = [0.5f64, 0.8];
        let aug_init = build_augmented_system(&q, &a_ext, &sigma1, 0.1, 0.05);

        let part1_idx = collect_part1_diag_indices(&aug_init, n);
        let part3_idx = collect_part3_diag_indices(&aug_init, n, m_ext);
        let q_base = collect_q_diag_base(&q, n);
        let mut cache = KktCache {
            mat: aug_init,
            part1_diag_idx: part1_idx,
            q_diag_base: q_base,
            part3_diag_idx: part3_idx,
            part1_updated_idx: (0..n).collect(),
        };

        // 2回目以降の更新パラメータ
        let sigma2 = [1.2f64, 0.3];
        let dp2 = 0.02f64;
        let dd2 = 0.01f64;
        update_augmented_values(&mut cache, &sigma2, dp2, dd2);

        // 参照: 同じパラメータで build_augmented_system を直接実行
        let aug_ref = build_augmented_system(&q, &a_ext, &sigma2, dp2, dd2);

        // values が完全一致することを確認
        assert_eq!(cache.mat.values.len(), aug_ref.values.len());
        for (i, (&got, &expected)) in cache.mat.values.iter().zip(aug_ref.values.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-14,
                "values[{i}]: got={got} expected={expected}"
            );
        }
    }
}
