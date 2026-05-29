//! IPM KKT 行列構築・疎行列-ベクトル演算。

use crate::problem::ConstraintType;
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;

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

#[inline]
pub(crate) fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

/// 戻り値: (A_ext, b_ext, m_ext, m_orig, n_lb, is_eq_ext)。
/// 順序は [original | lb | ub]。Ge は符号反転で Le 扱い。
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
                b_ext.push(-problem.b[i]);
                is_eq_ext.push(false);
            }
        }
    }

    // x_j ≥ lb_j → -x_j ≤ -lb_j
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

/// 境界行を除去し Ge 符号を反転して m_orig 長 dual に戻す。
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

/// augmented KKT の構造キャッシュ。`materialize` で σ/δ_p/δ_d だけ書き換えて反復間共有する。
pub(crate) struct AugmentedKktCache {
    col_ptr: Vec<usize>,
    row_ind: Vec<usize>,
    /// Q off-diag + A^T + Q diag + constraint diag placeholder。反復毎に対角を書き換える。
    static_values: Vec<f64>,
    diag_var_slot: Vec<usize>,
    /// 制約 k ∈ 0..m_ext の対角 slot (values[slot] = -(σ_k + δ_d) で上書き)
    diag_con_slot: Vec<usize>,
    n: usize,
    m_ext: usize,
}

impl AugmentedKktCache {
    /// 反復毎に呼ぶ: σ/δ_p/δ_d を template に反映した CscMatrix を返す。
    /// O(nnz + n + m_ext) — sort/compress を行わないため `from_triplets` より大幅に速い。
    pub(crate) fn materialize(&self, sigma_vec: &[f64], delta_p: f64, delta_d: f64) -> CscMatrix {
        debug_assert_eq!(sigma_vec.len(), self.m_ext);
        let mut values = self.static_values.clone();
        for j in 0..self.n {
            values[self.diag_var_slot[j]] += delta_p;
        }
        for k in 0..self.m_ext {
            values[self.diag_con_slot[k]] = -(sigma_vec[k] + delta_d);
        }
        let total = self.n + self.m_ext;
        CscMatrix {
            col_ptr: self.col_ptr.clone(),
            row_ind: self.row_ind.clone(),
            values,
            nrows: total,
            ncols: total,
        }
    }

    /// AMD 置換を適用した permuted aug_mat キャッシュを生成する。
    /// 1 回計算しておき、`PermutedAugmentedKkt::materialize` を反復毎に呼ぶ。
    /// permute_sym_upper の sort/compress を回避する (BOYD2 で 15-20ms/call 削減)。
    pub(crate) fn permute(&self, perm: &[usize]) -> PermutedAugmentedKkt {
        let total = self.n + self.m_ext;
        debug_assert_eq!(perm.len(), total);

        let mut inv_perm = vec![0usize; total];
        for (k, &i) in perm.iter().enumerate() {
            inv_perm[i] = k;
        }

        let nnz = self.row_ind.len();
        let mut perm_entries: Vec<(usize, usize, usize)> = Vec::with_capacity(nnz); // (new_row, new_col, orig_slot)
        for col in 0..total {
            let cs = self.col_ptr[col];
            let ce = self.col_ptr[col + 1];
            let new_col_for_orig_col = inv_perm[col];
            for s in cs..ce {
                let row = self.row_ind[s];
                let new_row = inv_perm[row];
                let new_col = new_col_for_orig_col;
                let (r, c) = if new_row <= new_col { (new_row, new_col) } else { (new_col, new_row) };
                perm_entries.push((r, c, s));
            }
        }
        perm_entries.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        let mut new_col_ptr = Vec::with_capacity(total + 1);
        let mut new_row_ind = Vec::with_capacity(nnz);
        let mut new_static_values = Vec::with_capacity(nnz);
        let mut orig_slot_for_new_slot = Vec::with_capacity(nnz);
        let mut new_diag_var_slot = vec![usize::MAX; self.n];
        let mut new_diag_con_slot = vec![usize::MAX; self.m_ext];

        new_col_ptr.push(0);
        let mut cur_col = 0usize;
        for &(r, c, orig_s) in &perm_entries {
            while cur_col < c {
                new_col_ptr.push(new_row_ind.len());
                cur_col += 1;
            }
            let new_slot = new_row_ind.len();
            new_row_ind.push(r);
            new_static_values.push(self.static_values[orig_s]);
            orig_slot_for_new_slot.push(orig_s);
            if r == c {
                let orig_idx = perm[c];
                if orig_idx < self.n {
                    new_diag_var_slot[orig_idx] = new_slot;
                } else {
                    new_diag_con_slot[orig_idx - self.n] = new_slot;
                }
            }
        }
        while cur_col < total {
            new_col_ptr.push(new_row_ind.len());
            cur_col += 1;
        }

        debug_assert!(new_diag_var_slot.iter().all(|&s| s != usize::MAX));
        debug_assert!(new_diag_con_slot.iter().all(|&s| s != usize::MAX));

        PermutedAugmentedKkt {
            col_ptr: new_col_ptr,
            row_ind: new_row_ind,
            static_values: new_static_values,
            diag_var_slot: new_diag_var_slot,
            diag_con_slot: new_diag_con_slot,
            n: self.n,
            m_ext: self.m_ext,
        }
    }
}

/// AMD 置換済みキャッシュ。`permute_sym_upper` を事前計算しておき再因子化コストを削減する。
pub(crate) struct PermutedAugmentedKkt {
    col_ptr: Vec<usize>,
    row_ind: Vec<usize>,
    static_values: Vec<f64>,
    diag_var_slot: Vec<usize>,
    diag_con_slot: Vec<usize>,
    n: usize,
    m_ext: usize,
}

impl PermutedAugmentedKkt {
    pub(crate) fn materialize(&self, sigma_vec: &[f64], delta_p: f64, delta_d: f64) -> CscMatrix {
        debug_assert_eq!(sigma_vec.len(), self.m_ext);
        let mut values = self.static_values.clone();
        for j in 0..self.n {
            values[self.diag_var_slot[j]] += delta_p;
        }
        for k in 0..self.m_ext {
            values[self.diag_con_slot[k]] = -(sigma_vec[k] + delta_d);
        }
        let total = self.n + self.m_ext;
        CscMatrix {
            col_ptr: self.col_ptr.clone(),
            row_ind: self.row_ind.clone(),
            values,
            nrows: total,
            ncols: total,
        }
    }
}

/// 不定 Q に対する慣性修正量 δ_ic (Q が上三角 CSC 前提)。
/// まず LLT で PSD 判定 (Gershgorin の保守誤判定を回避)、indefinite なら共通 helper
/// `linalg::gershgorin::psd_shift_from_gershgorin` で δ_ic = max(0, max_j(R_j − Q[j,j])) を返す。
pub(crate) fn compute_inertia_correction(q: &CscMatrix) -> f64 {
    if q.nrows == 0 || q.values.iter().all(|&v| v == 0.0) {
        return 0.0;
    }
    if crate::linalg::ldl::is_q_psd_by_cholesky(q) {
        return 0.0;
    }
    crate::linalg::gershgorin::psd_shift_from_gershgorin(q)
}

/// `AugmentedKktCache` を構築。Q は上三角全要素格納前提。
#[allow(clippy::needless_range_loop)]
pub(crate) fn build_augmented_cache(q: &CscMatrix, a_ext: &CscMatrix) -> AugmentedKktCache {
    let n = q.nrows;
    let m_ext = a_ext.nrows;
    let total = n + m_ext;

    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); total];
    let mut diag_var_present = vec![false; n];

    // Part 1: Q 上三角 (row <= col、Q が完全格納でも上三角だけ取る)
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            if row <= col {
                col_entries[col].push((row, q.values[k]));
                if row == col {
                    diag_var_present[col] = true;
                }
            }
        }
    }
    // Q diag が無い列には placeholder 0 を追加 (materialize で δ_p が乗る)
    for j in 0..n {
        if !diag_var_present[j] {
            col_entries[j].push((j, 0.0));
        }
    }

    // Part 2: A_ext^T (右上ブロック、row=j ∈ 0..n, col=n+k)
    for j in 0..n {
        for idx in a_ext.col_ptr[j]..a_ext.col_ptr[j + 1] {
            let k = a_ext.row_ind[idx];
            col_entries[n + k].push((j, a_ext.values[idx]));
        }
    }

    // Part 3: -(Σ + δ_d)·I 対角 (placeholder 0、materialize で上書き)
    for k in 0..m_ext {
        col_entries[n + k].push((n + k, 0.0));
    }

    // 列ごとに row 昇順で並べ替え (CSC 慣例)
    for entries in col_entries.iter_mut() {
        entries.sort_by_key(|&(r, _)| r);
    }

    let nnz: usize = col_entries.iter().map(|v| v.len()).sum();
    let mut col_ptr = Vec::with_capacity(total + 1);
    let mut row_ind = Vec::with_capacity(nnz);
    let mut static_values = Vec::with_capacity(nnz);
    let mut diag_var_slot = vec![usize::MAX; n];
    let mut diag_con_slot = vec![usize::MAX; m_ext];

    col_ptr.push(0);
    for col in 0..total {
        for &(row, val) in col_entries[col].iter() {
            let slot = row_ind.len();
            row_ind.push(row);
            static_values.push(val);
            if row == col {
                if col < n {
                    diag_var_slot[col] = slot;
                } else {
                    diag_con_slot[col - n] = slot;
                }
            }
        }
        col_ptr.push(row_ind.len());
    }

    debug_assert!(diag_var_slot.iter().all(|&s| s != usize::MAX));
    debug_assert!(diag_con_slot.iter().all(|&s| s != usize::MAX));

    AugmentedKktCache {
        col_ptr,
        row_ind,
        static_values,
        diag_var_slot,
        diag_con_slot,
        n,
        m_ext,
    }
}

/// augmented KKT 上三角 CSC: [[Q+δ_p I, Aᵀ],[A, −(Σ+δ_d I)]]。
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

    for j in 0..n {
        for idx in a_ext.col_ptr[j]..a_ext.col_ptr[j + 1] {
            let k = a_ext.row_ind[idx];
            let v = a_ext.values[idx];
            rows.push(j);
            cols.push(n + k);
            vals.push(v);
        }
    }

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

/// Schur complement S = (Q+ρI) + AᵀD⁻¹A の上三角 CSC (n×n SPD)。dy を消去した S·dx = r_d + AᵀD⁻¹r_p の形。
/// 戻り値: (S, d_inv) where d_inv[i] = 1/(σ[i]+δ_d)。
pub(crate) fn build_schur_system(
    q: &CscMatrix,
    a_ext: &CscMatrix,
    sigma_vec: &[f64],
    rho_p: f64,
    delta_d: f64,
) -> (CscMatrix, Vec<f64>) {
    use std::collections::BTreeMap;

    let n = q.nrows;
    let m_ext = a_ext.nrows;

    let d_inv: Vec<f64> = sigma_vec
        .iter()
        .map(|&s| 1.0 / (s + delta_d))
        .collect();

    let a_t = a_ext.transpose();

    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();

    for k in 0..m_ext {
        let start = a_t.col_ptr[k];
        let end = a_t.col_ptr[k + 1];
        let row_entries: Vec<(usize, f64)> = (start..end)
            .map(|p| (a_t.row_ind[p], a_t.values[p]))
            .collect();
        let dk = d_inv[k];
        for (idx_a, &(i, v_i)) in row_entries.iter().enumerate() {
            for &(j, v_j) in &row_entries[idx_a..] {
                let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
                *acc.entry((hi, lo)).or_insert(0.0) += dk * v_i * v_j;
            }
        }
    }

    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            if row <= col {
                *acc.entry((col, row)).or_insert(0.0) += q.values[k];
            }
        }
    }

    for i in 0..n {
        *acc.entry((i, i)).or_insert(0.0) += rho_p;
    }

    let mut col_ptr = vec![0_usize; n + 1];
    let mut row_ind: Vec<usize> = Vec::with_capacity(acc.len());
    let mut values: Vec<f64> = Vec::with_capacity(acc.len());
    for ((col, row), val) in acc {
        row_ind.push(row);
        values.push(val);
        col_ptr[col + 1] = row_ind.len();
    }
    for i in 1..=n {
        if col_ptr[i] < col_ptr[i - 1] {
            col_ptr[i] = col_ptr[i - 1];
        }
    }

    let s = CscMatrix {
        col_ptr,
        row_ind,
        values,
        nrows: n,
        ncols: n,
    };
    (s, d_inv)
}

/// 反復間不変なスパースパターンを利用した values 差分更新キャッシュ。
#[cfg(test)]
pub(crate) struct KktCache {
    pub mat: CscMatrix,
    pub part1_diag_idx: Vec<usize>,
    pub q_diag_base: Vec<f64>,
    pub part3_diag_idx: Vec<usize>,
    pub part1_updated_idx: Vec<usize>,
}

#[cfg(test)]
pub(crate) fn collect_part1_diag_indices(aug_mat: &CscMatrix, n: usize) -> Vec<usize> {
    (0..n).map(|i| aug_mat.col_ptr[i + 1] - 1).collect()
}

#[cfg(test)]
pub(crate) fn collect_part3_diag_indices(
    aug_mat: &CscMatrix,
    n: usize,
    m_ext: usize,
) -> Vec<usize> {
    (0..m_ext).map(|k| aug_mat.col_ptr[n + k + 1] - 1).collect()
}

#[cfg(test)]
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

#[cfg(test)]
pub(crate) fn update_augmented_values(
    cache: &mut KktCache,
    sigma_vec: &[f64],
    delta_p: f64,
    delta_d: f64,
) {
    for &i in &cache.part1_updated_idx {
        let idx = cache.part1_diag_idx[i];
        cache.mat.values[idx] = cache.q_diag_base[i] + delta_p;
    }
    for (k, &idx) in cache.part3_diag_idx.iter().enumerate() {
        cache.mat.values[idx] = -(sigma_vec[k] + delta_d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upper_tri_csc(n: usize, entries: &[(usize, usize, f64)]) -> CscMatrix {
        let rows: Vec<usize> = entries.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = entries.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<f64> = entries.iter().map(|&(_, _, v)| v).collect();
        CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
    }

    /// PSD 行列 (det>0) では Gershgorin の誤検出を避け δ=0 を返すこと。
    #[test]
    fn test_compute_inertia_correction_psd_no_correction() {
        let q = upper_tri_csc(2, &[(0, 0, 1.0), (0, 1, 1.1), (1, 1, 2.0)]);
        let delta = compute_inertia_correction(&q);
        assert_eq!(delta, 0.0, "got {}", delta);
    }

    #[test]
    fn test_compute_inertia_correction_indefinite() {
        let q = upper_tri_csc(2, &[(0, 0, 1.0), (1, 1, -2.0)]);
        let delta = compute_inertia_correction(&q);
        assert!(delta > 0.0, "got {}", delta);
        assert!((delta - 2.0).abs() < 1e-10, "expected 2.0, got {}", delta);
    }

    #[test]
    fn test_compute_inertia_correction_zero_matrix() {
        let q = upper_tri_csc(3, &[]);
        assert_eq!(compute_inertia_correction(&q), 0.0);
    }

    /// zero-diag indefinite Q を Gershgorin lb で正しく
    /// indefinite 判定し、δ_ic > 0 を返すこと。複数 data pattern。
    /// no-op proof: `is_q_psd_by_cholesky` の shift を 0 に戻すと FAIL。
    #[test]
    #[allow(clippy::type_complexity)]
    fn sentinel_inertia_zero_diag_indefinite_multi_pattern() {
        // (label, n, entries, expected_min_delta)
        let cases: &[(&str, usize, &[(usize, usize, f64)], f64)] = &[
            // bilinear pure: Q=[[0,1],[1,0]], λ=±1, Gershgorin row sum=1 → δ ≥ 1
            ("pure_bilinear_2x2", 2, &[(0, 1, 1.0)], 1.0),
            // mixed zero + negative diag: Q=[[0,1],[1,-1]], indefinite
            // diag=(0,-1), row sums = (1,1) → Gershgorin = (-1,-2) → δ ≥ 2
            ("zero_plus_negative_diag", 2, &[(0, 1, 1.0), (1, 1, -1.0)], 2.0),
            // 3x3 zero-diag bilinear: Q=[[0,1,0],[1,0,1],[0,1,0]]
            // row sums = (1,2,1), diag=(0,0,0) → max(R-Q)=2 → δ ≥ 2
            ("zero_diag_3x3_chain", 3, &[(0, 1, 1.0), (1, 2, 1.0)], 2.0),
            // 4x4 partial zero-diag: Q[0,0]=2, Q[0,3]=3, rest zero
            // row sums = (3,0,0,3), diag=(2,0,0,0) → Gershgorin = (-1,0,0,-3) → δ ≥ 3
            ("zero_diag_4x4_extreme_offdiag", 4, &[(0, 0, 2.0), (0, 3, 3.0)], 3.0),
        ];
        for &(label, n, entries, expected_min) in cases {
            let q = upper_tri_csc(n, entries);
            let delta = compute_inertia_correction(&q);
            assert!(
                delta >= expected_min - 1e-12,
                "[{label}] expected δ_ic ≥ {expected_min}, got {delta} (indefinite Q misclassified as PSD)"
            );
        }
    }

    /// 真の PSD/PD は perturbation 後も δ_ic=0 を維持。
    /// shift が大きすぎて全 PSD に過剰補正を出さないことを確認。
    #[test]
    #[allow(clippy::type_complexity)]
    fn sentinel_inertia_psd_no_over_correction() {
        // (label, n, entries)
        let psd_cases: &[(&str, usize, &[(usize, usize, f64)])] = &[
            ("pd_2x2_diag", 2, &[(0, 0, 1.0), (1, 1, 1.0)]),
            ("pd_2x2_offdiag", 2, &[(0, 0, 4.0), (0, 1, 1.0), (1, 1, 3.0)]),
            ("psd_singular_rank1", 2, &[(0, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0)]),
            ("psd_with_zero_eig_3x3", 3, &[(0, 0, 1.0), (1, 1, 1.0)]),
            ("pd_large_offdiag_gershgorin_false_alarm", 2,
             &[(0, 0, 1.0), (0, 1, 1.1), (1, 1, 2.0)]),
        ];
        for &(label, n, entries) in psd_cases {
            let q = upper_tri_csc(n, entries);
            let delta = compute_inertia_correction(&q);
            assert_eq!(
                delta, 0.0,
                "[{label}] PSD matrix must not get correction, got δ_ic={delta}"
            );
        }
    }


    #[test]
    fn test_collect_part1_diag_indices() {
        let q = upper_tri_csc(2, &[(0, 0, 2.0), (0, 1, 0.5), (1, 1, 3.0)]);
        let a_rows = vec![0usize, 1];
        let a_cols = vec![0usize, 1];
        let a_vals = vec![1.0f64, 1.0];
        let a_ext = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, 2, 2).unwrap();

        let sigma_vec = [0.5f64, 0.8];
        let aug = build_augmented_system(&q, &a_ext, &sigma_vec, 0.1, 0.05);
        let n = 2;

        let diag_idx = collect_part1_diag_indices(&aug, n);
        for i in 0..n {
            let idx = diag_idx[i];
            assert_eq!(aug.row_ind[idx], i);
        }
        assert!((aug.values[diag_idx[0]] - 2.1).abs() < 1e-14);
        assert!((aug.values[diag_idx[1]] - 3.1).abs() < 1e-14);
    }

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
            assert_eq!(aug.row_ind[idx], n + k);
        }
        assert!((aug.values[diag_idx[0]] - (-0.55)).abs() < 1e-14);
        assert!((aug.values[diag_idx[1]] - (-0.85)).abs() < 1e-14);
    }

    #[test]
    fn test_spmv_basic() {
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

    #[test]
    fn test_norm_inf_basic() {
        assert!((norm_inf(&[3.0, -5.0, 2.0]) - 5.0).abs() < 1e-15);
        assert!((norm_inf(&[0.0, 0.0]) - 0.0).abs() < 1e-15);
        assert!((norm_inf(&[-7.0]) - 7.0).abs() < 1e-15);
    }

    #[test]
    fn test_build_extended_constraints_dimensions() {
        use crate::qp::problem::QpProblem;
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
        assert_eq!(m_orig, 1);
        assert_eq!(n_lb, 2);
        assert_eq!(m_ext, 3);
        assert_eq!(a_ext.nrows, 3);
        assert_eq!(a_ext.ncols, 2);
        assert_eq!(b_ext.len(), 3);
        assert_eq!(is_eq_ext, vec![false, false, false]);
    }

    #[test]
    fn test_update_augmented_values_matches_build() {
        let q = upper_tri_csc(2, &[(0, 0, 2.0), (0, 1, 0.5), (1, 1, 3.0)]);
        let a_rows = vec![0usize, 1];
        let a_cols = vec![0usize, 1];
        let a_vals = vec![1.0f64, 1.0];
        let a_ext = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, 2, 2).unwrap();

        let n = 2;
        let m_ext = 2;
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

        let sigma2 = [1.2f64, 0.3];
        let dp2 = 0.02f64;
        let dd2 = 0.01f64;
        update_augmented_values(&mut cache, &sigma2, dp2, dd2);

        let aug_ref = build_augmented_system(&q, &a_ext, &sigma2, dp2, dd2);

        assert_eq!(cache.mat.values.len(), aug_ref.values.len());
        for (i, (&got, &expected)) in cache.mat.values.iter().zip(aug_ref.values.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-14,
                "values[{i}]: got={got} expected={expected}"
            );
        }
    }

    /// 境界行係数: lb 行 = -1·x ≤ -lb、ub 行 = +1·x ≤ ub。
    #[test]
    fn test_build_extended_constraints_bound_rows() {
        use crate::qp::problem::QpProblem;
        let q = CscMatrix::new(2, 2);
        let a = CscMatrix::new(0, 2);
        let prob = QpProblem::new_all_le(
            q, vec![0.0; 2], a, vec![],
            vec![(0.0, 5.0), (f64::NEG_INFINITY, 10.0)],
        ).unwrap();
        let (a_ext, b_ext, m_ext, m_orig, n_lb, _) = build_extended_constraints(&prob);
        assert_eq!(m_orig, 0);
        assert_eq!(n_lb, 1);
        assert_eq!(m_ext, 3);
        assert_eq!(b_ext, vec![-0.0, 5.0, 10.0]);

        let mut row_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; m_ext];
        for col in 0..a_ext.ncols {
            for k in a_ext.col_ptr[col]..a_ext.col_ptr[col + 1] {
                row_entries[a_ext.row_ind[k]].push((col, a_ext.values[k]));
            }
        }
        assert_eq!(row_entries[0], vec![(0, -1.0)]);
        assert_eq!(row_entries[1], vec![(0, 1.0)]);
        assert_eq!(row_entries[2], vec![(1, 1.0)]);
    }

    /// Ge は拡張時に -1 倍格納されるため復元で符号反転。
    #[test]
    fn test_collapse_extended_dual_ge_sign_flip() {
        use crate::problem::ConstraintType as CT;
        let dual_ext = vec![1.0_f64, 2.0, 3.0, 4.0];
        let cts = vec![CT::Le, CT::Ge, CT::Eq];
        let collapsed = collapse_extended_dual(&dual_ext, 3, &cts);
        assert_eq!(collapsed, vec![1.0, -2.0, 3.0]);
    }
}
