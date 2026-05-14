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

/// augmented KKT system の構造キャッシュ。`build_augmented_cache` で 1 回構築し、
/// `materialize` で σ/δ_p/δ_d だけ反映した CscMatrix を生成する。
///
/// 動機: Q と A の sparsity pattern は IPM 反復間で不変。col_ptr/row_ind と
/// 「変化しない値 (Q off-diag, Q diag, A^T)」を template として持ち、毎反復
/// 「変化する値 (δ_p、Σ+δ_d)」だけ書き換える。`from_triplets` の sort/compress を回避し
/// BOYD2 で per-call 71ms → ~5ms 程度を狙う。
pub(crate) struct AugmentedKktCache {
    col_ptr: Vec<usize>,
    row_ind: Vec<usize>,
    /// Q off-diag + A^T + Q diag (Q[j,j] そのまま) + constraint diag (placeholder 0)
    /// 反復毎に clone() し、対角だけ書き換える。
    static_values: Vec<f64>,
    /// 変数 j ∈ 0..n の対角 slot (values[slot] に δ_p を加算する用)
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

        // inv_perm[i] = 新しいインデックス k such that perm[k] = i
        let mut inv_perm = vec![0usize; total];
        for (k, &i) in perm.iter().enumerate() {
            inv_perm[i] = k;
        }

        // 各 unpermuted slot s に対して: row=row_ind[s], col=実際の col (col_ptr で逆引き)
        // permuted (new_row, new_col) を計算し、上三角に正規化。
        // permuted の col_ptr/row_ind を組み立てつつ、permuted_slot[s] = 新しい slot を記録。
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
        // 列優先・行昇順でソート
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
            // permuted 上で対角の場合は new_slot を記録
            if r == c {
                // どの var/con の対角かは perm[c] で逆引き
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

/// AMD 置換適用済みの augmented KKT 構造キャッシュ。
/// `factorize_quasidefinite_pre_permuted` に渡すと permute_sym_upper を skip できる。
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
    /// 置換済み CSC を materialize する (σ/δ_p/δ_d を反映)。
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

/// 不定 Q 行列に対して慣性修正量 δ_ic を計算する。
///
/// 手順:
/// 1. LLT 因子化（正則化なし）で Q が PSD かどうかを判定する。
///    PSD であれば δ_ic = 0 を返す（Gershgorin の保守的誤判定を回避）。
///
/// 2. LLT 失敗（負のピボット）のとき: Gershgorin の円定理から
///    λ_min(Q) の下界を導出し、Q + δ_ic·I を PSD にする最小量を返す。
///    Gershgorin: λ_min(Q) >= min_j(Q[j,j] - R_j)
///    δ_ic = max(0, max_j(R_j - Q[j,j]))
///
/// Q が上三角 CSC で格納されている前提。
///
/// 根拠:
/// - PSD 判定を LLT で行うのは「Gershgorin が PSD 行列でも偽陽性を返す」問題を
///   解消するため。Gershgorin は十分条件であり、PSD 行列でも R_j > Q[j,j] と
///   なる（大きい対角外要素を持つ）場合は δ_ic > 0 を誤って返す。
/// - LLT 成功 → Q が真に PSD → KKT の (1,1) ブロック Q + rho·I は rho > 0 で
///   自動的に正定値 → 慣性修正不要。
/// - LLT 失敗 → Q に負の固有値あり → Gershgorin bound で最小量を推定。
pub(crate) fn compute_inertia_correction(q: &CscMatrix) -> f64 {
    let n = q.nrows;
    if n == 0 {
        return 0.0;
    }
    // Q が全ゼロなら LP ケース → 慣性修正不要
    if q.values.iter().all(|&v| v == 0.0) {
        return 0.0;
    }

    // Step 1: LLT 因子化で PSD を判定する。
    // PSD → δ_ic = 0（Gershgorin の誤検出を回避）。
    if crate::linalg::ldl::is_q_psd_by_cholesky(q) {
        return 0.0;
    }

    // Step 2: Q が indefinite → Gershgorin 円定理でδを推定する。
    // Q は上三角 CSC: Q[row, col] (row <= col) が格納されている。
    // 対角外要素 Q[row, col] (row < col) は:
    //   - 行 row の R_{row} に寄与（上三角側）
    //   - 行 col の R_{col} に寄与（下三角の対称側）
    let mut diag = vec![0.0_f64; n];
    let mut row_offdiag_sum = vec![0.0_f64; n];

    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            let val = q.values[k];
            if row == col {
                diag[col] = val;
            } else if row < col {
                // 上三角要素: 対称性により row の行と col の行の両方の offdiag sum に加える
                let abs_val = val.abs();
                row_offdiag_sum[row] += abs_val;
                row_offdiag_sum[col] += abs_val;
            }
        }
    }

    // δ_ic = max(0, max_j(R_j - Q[j,j]))
    // Gershgorin: λ_min >= min_j(diag[j] - row_offdiag_sum[j])
    // → δ_ic を加えると λ_min(Q + δ_ic·I) >= 0
    let mut delta_ic = 0.0_f64;
    for j in 0..n {
        let gershgorin_lower = diag[j] - row_offdiag_sum[j];
        if gershgorin_lower < 0.0 {
            delta_ic = delta_ic.max(-gershgorin_lower);
        }
    }
    delta_ic
}

/// `AugmentedKktCache` を構築する。Q/A の sparsity pattern を 1 回走査して
/// col_ptr/row_ind/diag slot を確定する。`build_augmented_system` と同じ非ゼロ集合を生成する。
///
/// 制約: Q は上三角全要素格納 (build_augmented_system の前提と同じ)。
#[allow(clippy::needless_range_loop)]
pub(crate) fn build_augmented_cache(q: &CscMatrix, a_ext: &CscMatrix) -> AugmentedKktCache {
    let n = q.nrows;
    let m_ext = a_ext.nrows;
    let total = n + m_ext;

    // 列ごとの (row, value) リストを集める。Part 1 (Q 上三角)、Part 2 (A^T)、Part 3 (Σ 対角) の
    // 順に追加し、最後に列ごとに row 昇順で並べ替える。
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

    // CSC 構造に組み直し、同時に diag slot を記録
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

/// Schur complement system S = (Q + ρI) + A^T D^{-1} A を構築する (上三角 CSC、n×n SPD)。
///
/// augmented system:
///   [Q+ρI  A^T] [dx]   [r_d]
///   [A    -D  ] [dy] = [r_p]   where D = diag(sigma_vec) + δI
///
/// dy を消去 (dy = D^{-1}(A·dx − r_p)) して dx の n×n SPD system を得る:
///   S·dx = r_d + A^T D^{-1} r_p
///
/// 利点:
/// - サイズ削減 (n+m_ext → n)
/// - SPD なので Cholesky で安定
/// - cond(S) は cond(K) と異なる (LISWET 系で改善期待、要実測)
///
/// 戻り値: (S, d_inv) where d_inv[i] = 1/(sigma_vec[i] + δ_d)
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

    // d_inv[k] = 1 / (sigma_vec[k] + delta_d)
    // 等式行は sigma=0 → d_inv = 1/delta_d (大きいが有限)。
    // 不等式 active (sigma=0): d_inv = 1/delta_d 大、A^T D^{-1} A の寄与大。
    // 不等式 inactive (sigma=∞): d_inv ≈ 0、寄与小。
    let d_inv: Vec<f64> = sigma_vec
        .iter()
        .map(|&s| 1.0 / (s + delta_d))
        .collect();

    // A^T D^{-1} A の上三角を sparse 蓄積。
    // a_t = A^T (n × m_ext) なので、a_t の col k = a_ext の row k (CSR 風アクセス)。
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
                // 上三角 (col=hi, row=lo) で row ≤ col
                *acc.entry((hi, lo)).or_insert(0.0) += dk * v_i * v_j;
            }
        }
    }

    // Q (上三角) を加算
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            if row <= col {
                *acc.entry((col, row)).or_insert(0.0) += q.values[k];
            }
        }
    }

    // ρI を対角に加算
    for i in 0..n {
        *acc.entry((i, i)).or_insert(0.0) += rho_p;
    }

    // CSC 構築 (BTreeMap は (col, row) の lexicographic 順なので既に col-major)
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

    /// compute_inertia_correction: PSD 行列は 0 を返す (Gershgorin の誤検出を回避)
    ///
    /// Q = [[1, 1.1],[1.1, 2]] — PD (det = 2 - 1.21 = 0.79 > 0)
    /// Gershgorin では R_0 = 1.1 > Q[0,0] = 1 → 下界 = -0.1 < 0 → δ=0.1 を誤って返す
    /// LLT ベースの新実装では PSD を正しく検出し δ=0 を返すべき。
    #[test]
    fn test_compute_inertia_correction_psd_no_correction() {
        let q = upper_tri_csc(2, &[(0, 0, 1.0), (0, 1, 1.1), (1, 1, 2.0)]);
        let delta = compute_inertia_correction(&q);
        assert_eq!(delta, 0.0,
            "PSD matrix (det>0) should have inertia_correction=0, got {}", delta);
    }

    /// compute_inertia_correction: 不定行列は正の δ を返す
    #[test]
    fn test_compute_inertia_correction_indefinite() {
        // Q = [[1, 0],[0, -2]] — indefinite (固有値 1, -2)
        let q = upper_tri_csc(2, &[(0, 0, 1.0), (1, 1, -2.0)]);
        let delta = compute_inertia_correction(&q);
        assert!(delta > 0.0,
            "Indefinite matrix should have inertia_correction > 0, got {}", delta);
        // Gershgorin: λ_min >= min(1-0, -2-0) = -2 → δ = 2
        assert!((delta - 2.0).abs() < 1e-10,
            "Expected delta=2.0 for Q=diag(1,-2), got {}", delta);
    }

    /// compute_inertia_correction: ゼロ行列は 0 を返す (LP ケース)
    #[test]
    fn test_compute_inertia_correction_zero_matrix() {
        let q = upper_tri_csc(3, &[]);
        assert_eq!(compute_inertia_correction(&q), 0.0,
            "Zero matrix (LP) should have inertia_correction=0");
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

    /// build_extended_constraints: 境界行 (-1 for lb, +1 for ub) と b_ext (lb 行 = -lb,
    /// ub 行 = ub) が仕様通り構築されることを確認。bound_dual 経路で y[bound_row] が
    /// stationarity に正しく寄与するための前提。
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
        assert_eq!(n_lb, 1);  // col 0 のみ lb 有限
        assert_eq!(m_ext, 3); // 0 orig + 1 lb + 2 ub
        assert_eq!(b_ext, vec![-0.0, 5.0, 10.0]); // [lb 群; ub 群]

        let mut row_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; m_ext];
        for col in 0..a_ext.ncols {
            for k in a_ext.col_ptr[col]..a_ext.col_ptr[col + 1] {
                row_entries[a_ext.row_ind[k]].push((col, a_ext.values[k]));
            }
        }
        assert_eq!(row_entries[0], vec![(0, -1.0)], "lb row of col 0 has -1");
        assert_eq!(row_entries[1], vec![(0, 1.0)],  "ub row of col 0 has +1");
        assert_eq!(row_entries[2], vec![(1, 1.0)],  "ub row of col 1 has +1");
    }

    /// collapse_extended_dual: Ge 制約は拡張時に -1 倍されて格納されるため、
    /// 元空間 dual 復元時に y[ge_row] の符号を反転する。Le/Eq はそのまま。
    #[test]
    fn test_collapse_extended_dual_ge_sign_flip() {
        use crate::problem::ConstraintType as CT;
        let dual_ext = vec![1.0_f64, 2.0, 3.0, 4.0];
        let cts = vec![CT::Le, CT::Ge, CT::Eq];
        let collapsed = collapse_extended_dual(&dual_ext, 3, &cts);
        assert_eq!(collapsed, vec![1.0, -2.0, 3.0]);
    }
}
