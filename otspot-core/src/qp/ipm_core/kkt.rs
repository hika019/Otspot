//! IPM KKT 行列構築・疎行列-ベクトル演算。

use crate::problem::ConstraintType;
use crate::qp::problem::QpProblem;
use otspot_num::sparse::CscMatrix;

#[inline]
#[allow(clippy::needless_range_loop)]
pub(crate) fn spmv(a: &CscMatrix, x: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for col in 0..a.ncols() {
        let xv = x[col];
        for k in a.col_ptr()[col]..a.col_ptr()[col + 1] {
            out[a.row_ind()[k]] += a.values()[k] * xv;
        }
    }
}

#[inline]
#[allow(clippy::needless_range_loop)]
pub(crate) fn spmtv(a: &CscMatrix, v: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|o| *o = 0.0);
    for col in 0..a.ncols() {
        let mut s = 0.0;
        for k in a.col_ptr()[col]..a.col_ptr()[col + 1] {
            s += a.values()[k] * v[a.row_ind()[k]];
        }
        out[col] = s;
    }
}

#[inline]
pub(crate) fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

/// 元制約行 (m_orig 行) の Le 化係数行を構築する。
///
/// Ge は符号反転で Le 扱い。`build_extended_constraints` の元行部分と
/// box 化 KKT の実制約ブロック (`a_base`) はこの関数の単一生成物を共有する
/// (行番号・符号規約の二重実装を防止)。境界行は含まない。
/// ゼロ係数行のみ (rows 空) の場合も nrows = m_orig を保つ (sigma 長と
/// KKT の対角ブロック行数の対応を壊さない)。
pub(crate) fn build_base_constraints(problem: &QpProblem) -> (CscMatrix, Vec<f64>, Vec<bool>) {
    let n = problem.num_vars;
    let m = problem.num_constraints;
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let mut b_base = Vec::with_capacity(m);
    let mut is_eq_base = Vec::with_capacity(m);

    for col in 0..n {
        for k in problem.a.col_ptr()[col]..problem.a.col_ptr()[col + 1] {
            let row = problem.a.row_ind()[k];
            let sign = if problem.constraint_types[row] == ConstraintType::Ge {
                -1.0
            } else {
                1.0
            };
            rows.push(row);
            cols.push(col);
            vals.push(problem.a.values()[k] * sign);
        }
    }
    for i in 0..m {
        match problem.constraint_types[i] {
            ConstraintType::Eq => {
                b_base.push(problem.b[i]);
                is_eq_base.push(true);
            }
            ConstraintType::Le => {
                b_base.push(problem.b[i]);
                is_eq_base.push(false);
            }
            ConstraintType::Ge => {
                b_base.push(-problem.b[i]);
                is_eq_base.push(false);
            }
        }
    }

    let a_base = if rows.is_empty() {
        CscMatrix::new(m, n)
    } else {
        CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap()
    };
    (a_base, b_base, is_eq_base)
}

/// 戻り値: (A_ext, b_ext, m_ext, m_orig, n_lb, is_eq_ext)。
/// 順序は [original | lb | ub]。Ge は符号反転で Le 扱い。
pub(crate) fn build_extended_constraints(
    problem: &QpProblem,
) -> (CscMatrix, Vec<f64>, usize, usize, usize, Vec<bool>) {
    let n = problem.num_vars;
    let (mut rows, mut cols, mut vals, mut b_ext, is_eq_ext_raw) = {
        let (a_base, b_base, is_eq_base) = build_base_constraints(problem);
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for col in 0..n {
            for k in a_base.col_ptr()[col]..a_base.col_ptr()[col + 1] {
                rows.push(a_base.row_ind()[k]);
                cols.push(col);
                vals.push(a_base.values()[k]);
            }
        }
        (rows, cols, vals, b_base, is_eq_base)
    };
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
    let mut is_eq_ext = is_eq_ext_raw;
    is_eq_ext.reserve(n_lb + n_ub);

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

/// 依存境界スラックの下限床。
///
/// `update_variables` が不等式スラックに課す既存の正床 (1e-12) と同源:
/// x が境界上に着いたとき従属スラック s_t = b_t − a_t x が 0 になると
/// fraction-to-boundary の α が瞬時に 0 へ張り付く (x が内側へ戻れない)。
/// 床は数値ガードであり、双対 y は s と独立に保持されるためbarrier力学は
/// σ_t = s_t/y_t 側の床 (δ_d 下限) で制御される。
pub(crate) const BOUND_SLACK_FLOOR: f64 = 1e-12;

/// 境界行 Schur 対角寄与 1/(σ_t + δ_d) の分母床。
///
/// box 化境界行は設計上 δ_d = 0 で呼ばれるため、この床は常時有効な経路を
/// 持つ: x が境界に着いたとき σ_t = s_t/y_t → 0 となり 1/0 = inf が KKT 対角へ
/// 流れ込むのを防ぐ。上限 1/ε_machine は `probe_ldl_health` の amplification
/// 上限 (cond が f64 域内の検査) と同源。
pub(crate) const BOUND_DUAL_DENOM_FLOOR: f64 = f64::EPSILON;

/// 境界行 t の Schur 対角寄与 d_t = 1/(σ_t + δ_d) (床適用)。
///
/// `AugmentedKktCache::materialize` とループ側の dy_bound 復元が同一関数を
/// 共有し、KKT 対角の D_b と後退代入の除算の一致を保証する。
///
/// box 化境界行には必ず δ_d = 0 を渡すこと。dual damping δ は extended 系の
/// 独立スラック行 (実制約行・lb==ub 退化対の保持行) だけの正則化で、box 化行
/// に δ を混ぜると σ_t ≪ δ の区間で対角寄与が 1/δ の定数に潰え、境界制約が
/// KKT 系から消える。その系は box を無視する的方向 (発散方向) を作り、
/// fraction-to-boundary が α≈0 で x を壁に押さえつけ、δ 減衰の燃料である
/// μ 低下が止まる自己閉鎖 deadlock になる (実測: 幅 3e-3 の box・Q=I で
/// δ=8 固定のまま 52 反復 Stalled、δ 非適用で Optimal/5 反復)。
#[inline]
pub(crate) fn bound_row_dual_scale(sigma_t: f64, delta_d: f64) -> f64 {
    1.0 / (sigma_t + delta_d).max(BOUND_DUAL_DENOM_FLOOR)
}

/// box 化 KKT がeliminateする境界行メタデータ。
///
/// 行順は `build_extended_constraints` の追加行と同一 (lb 行を var 昇順、
/// その後に ub 行を var 昇順)。よって `y[m_orig..]` の extended レイアウト・
/// `bound_duals` の出力規約は box 化前後で不変 (受理ゲートの証書コードは無変更)。
/// 境界行 t の係数は a_t = sign[t]·e_{var[t]} (lb 行 −1 / ub 行 +1)、
/// 右辺 b_t = rhs[t] (lb 行 −lb / ub 行 ub) で、従属スラックは
/// s_t(x) = b_t − a_t·x、その方向微分は ds_t = −a_t·dx。
pub(crate) struct BoundBox {
    /// 参照変数 j。
    pub(crate) var: Vec<usize>,
    /// 係数 a_t の符号 (lb 行 −1.0 / ub 行 +1.0)。
    pub(crate) sign: Vec<f64>,
    /// 行右辺 b_t。
    pub(crate) rhs: Vec<f64>,
    /// true = 正確 Schur で box 化 (従属スラック)。false = 系に残る保持行
    /// (`build_bound_box` の doc: lb_j == ub_j の退化対のみ)。
    pub(crate) eliminated: Vec<bool>,
}

impl BoundBox {
    pub(crate) fn count(&self) -> usize {
        self.var.len()
    }

    /// box 化されず系に残る境界行インデックス t (昇順)。
    pub(crate) fn kept_rows(&self) -> Vec<usize> {
        (0..self.count()).filter(|&t| !self.eliminated[t]).collect()
    }

    /// box 化される境界行インデックス t (昇順)。
    pub(crate) fn eliminated_rows(&self) -> Vec<usize> {
        (0..self.count()).filter(|&t| self.eliminated[t]).collect()
    }

    /// box 化 KKT 系行 k → extended レイアウト行インデックス の対応表。
    /// 系行順は [実制約行 0..m_orig | 保持境界行 (t 昇順)]。
    pub(crate) fn system_ext_rows(&self, m_orig: usize) -> Vec<usize> {
        (0..m_orig)
            .chain(self.kept_rows().into_iter().map(|t| m_orig + t))
            .collect()
    }

    /// 従属スラック s_t(x) = b_t − a_t·x_j (床 `BOUND_SLACK_FLOOR` 適用)。
    #[inline]
    pub(crate) fn slack(&self, t: usize, x: &[f64]) -> f64 {
        (self.rhs[t] - self.sign[t] * x[self.var[t]]).max(BOUND_SLACK_FLOOR)
    }

    /// s_full の尾部 [m_orig..m_ext] へ従属スラックを書き込む。
    /// 保持行 (lb==ub 退化対) は独立スラックなのでスロットを触らない —
    /// そこは update_variables が ds から積分する (BASE  extended 系の力学)。
    /// tail の長さは count() と一致していること (debug_assert)。
    pub(crate) fn fill_slacks(&self, x: &[f64], tail: &mut [f64]) {
        debug_assert_eq!(tail.len(), self.count());
        for (t, slot) in tail.iter_mut().enumerate() {
            if self.eliminated[t] {
                *slot = self.slack(t, x);
            }
        }
    }

    /// init 専用の全スロット書き込み: box 化境界行は従属導出、保持境界行は
    /// 整合する独立初期値 s_t = b_t − a_t·x (床付き)。ループ中の再導出は
    /// `fill_slacks` (保持行スキップ) を使い、保持行の s は ds 積分に委ねる。
    pub(crate) fn init_slacks(&self, x: &[f64], tail: &mut [f64]) {
        debug_assert_eq!(tail.len(), self.count());
        for (t, slot) in tail.iter_mut().enumerate() {
            *slot = self.slack(t, x);
        }
    }
}

/// 境界行が Schur 消去可能か (box 化設計の妥当条件)。
///
/// lb_j == ub_j の退化対では s_lb + s_ub ≡ 0 が厳密成立し、両スラックが
/// 床 BOUND_SLACK_FLOOR に張り付いたままになる。従属形 ds_t = −a_t·dx は
/// 固定変数の dx の微小 noise で fraction-to-boundary の α が全局的に潰れ
/// 反復が停留する (実測: 固定変数を含む BD-T6 形状で 51 反復 Stalled、
/// 同問題を保持行化すると Optimal)。base 系がこの対を独立スラック行として
/// 運んでいたことを踏まえ、退化対は box 化せず系行として保持する。
#[inline]
fn bound_pair_is_degenerate(lb: f64, ub: f64) -> bool {
    lb.is_finite() && ub.is_finite() && lb == ub
}

/// 有限境界から box 化メタデータを構築する (行順は上記 doc 参照)。
///
/// lb_j == ub_j の退化対 (両側有限・一致) は `eliminated = false` とする:
/// 従属スラックが存在し得ないため box 化せず、系行として保持する
/// (`bound_pair_is_degenerate` の doc)。
pub(crate) fn build_bound_box(problem: &QpProblem) -> BoundBox {
    let mut var = Vec::new();
    let mut sign = Vec::new();
    let mut rhs = Vec::new();
    for (j, &(lb, _)) in problem.bounds.iter().enumerate() {
        if lb.is_finite() {
            var.push(j);
            sign.push(-1.0);
            rhs.push(-lb);
        }
    }
    for (j, &(_, ub)) in problem.bounds.iter().enumerate() {
        if ub.is_finite() {
            var.push(j);
            sign.push(1.0);
            rhs.push(ub);
        }
    }
    let eliminated = (0..var.len())
        .map(|t| {
            let (lb, ub) = problem.bounds[var[t]];
            !bound_pair_is_degenerate(lb, ub)
        })
        .collect();
    BoundBox {
        var,
        sign,
        rhs,
        eliminated,
    }
}

/// box 化 KKT の系行列 A_sys = [実制約行 | 保持境界行 (t 昇順)]。
///
/// 行番号・符号規約は `build_base_constraints` の単一生成物を共有する
/// (stack 先頭を a_base そのままに保つことで二重実装を防止)。キャッシュ構築と
/// Schur 系構築と求解側が同じ行列を参照するため、この関数 1 つで生成する。
pub(crate) fn build_boxed_system_matrix(a_base: &CscMatrix, bound: &BoundBox) -> CscMatrix {
    let n = a_base.ncols();
    let m_base = a_base.nrows();
    let kept = bound.kept_rows();
    if kept.is_empty() {
        return a_base.clone();
    }
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for col in 0..n {
        for k in a_base.col_ptr()[col]..a_base.col_ptr()[col + 1] {
            rows.push(a_base.row_ind()[k]);
            cols.push(col);
            vals.push(a_base.values()[k]);
        }
    }
    for (i, &t) in kept.iter().enumerate() {
        rows.push(m_base + i);
        cols.push(bound.var[t]);
        vals.push(bound.sign[t]);
    }
    CscMatrix::from_triplets(&rows, &cols, &vals, m_base + kept.len(), n).unwrap()
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
///
/// box 化 (`bound_var = Some`) では境界行は行として存在せず、その Schur 寄与
/// `1/(σ_t + δ_d)` を対角 slot `diag_var_slot[var_t]` へ畳み込む。このとき
/// `sigma_vec` は extended レイアウト ([実行 | 境界行]) の長さ n_bound + m_rows
/// を持つ (境界行 σ は従属スラックからループ側が供給)。
pub(crate) struct AugmentedKktCache {
    col_ptr: Vec<usize>,
    row_ind: Vec<usize>,
    /// Q off-diag + A^T + Q diag + constraint diag placeholder。反復毎に対角を書き換える。
    static_values: Vec<f64>,
    diag_var_slot: Vec<usize>,
    /// 制約 k ∈ 0..m_rows の対角 slot (values[slot] = -(σ_k + δ_d) で上書き)
    diag_con_slot: Vec<usize>,
    n: usize,
    m_rows: usize,
    /// box 化時のみ Some: box 化される境界行 t の参照変数
    /// (行順は `BoundBox::eliminated_rows()` の昇順)。保持行 (lb==ub 退化対) は
    /// 系行側に立つためここには現れない。
    bound_var: Option<Vec<usize>>,
    /// 系行 k → `materialize` が受け取る extended 風 sigma_vec 内の index。
    /// box 化時は `[実制約行 | 保持境界行]` 系行順から extended レイアウト
    /// ([実制約 | 全境界行]) の σ 位置への写像 (デフォルトは恒等)。
    sys_sigma_idx: Vec<usize>,
    /// box 化境界 t (eliminated 昇順) → extended sigma_vec 内の σ 位置。
    bound_sigma_idx: Vec<usize>,
}

impl AugmentedKktCache {
    /// `materialize` が受け取る sigma_vec の期待長 (extended レイアウト長)。
    pub(crate) fn sigma_len(&self) -> usize {
        self.m_rows + self.bound_var.as_ref().map_or(0, |v| v.len())
    }

    /// box 化時の box 化境界行参照変数スライス (保持行は系行側)。
    pub(crate) fn bound_var(&self) -> Option<&[usize]> {
        self.bound_var.as_deref()
    }

    /// 反復毎に呼ぶ: σ/δ_p/δ_d を template に反映した CscMatrix を返す。
    /// O(nnz + n + m_ext) — sort/compress を行わないため `from_triplets` より大幅に速い。
    pub(crate) fn materialize(&self, sigma_vec: &[f64], delta_p: f64, delta_d: f64) -> CscMatrix {
        debug_assert_eq!(sigma_vec.len(), self.sigma_len());
        let mut values = self.static_values.clone();
        for j in 0..self.n {
            values[self.diag_var_slot[j]] += delta_p;
        }
        for k in 0..self.m_rows {
            values[self.diag_con_slot[k]] = -(sigma_vec[self.sys_sigma_idx[k]] + delta_d);
        }
        if let Some(bound_var) = &self.bound_var {
            // 正確 Schur: 境界行の 1×1 対角ブロック -(σ_t+δ_d) を消去した対角寄与
            // を x ブロック対角へ加算する (kkt.rs 上部 doc の box-ification 設計)。
            for (t, &j) in bound_var.iter().enumerate() {
                values[self.diag_var_slot[j]] +=
                    bound_row_dual_scale(sigma_vec[self.bound_sigma_idx[t]], 0.0);
            }
        }
        let total = self.n + self.m_rows;
        CscMatrix::from_raw_parts(
            total,
            total,
            self.col_ptr.clone(),
            self.row_ind.clone(),
            values,
        )
    }

    /// AMD 置換を適用した permuted aug_mat キャッシュを生成する。
    /// 1 回計算しておき、`PermutedAugmentedKkt::materialize` を反復毎に呼ぶ。
    /// permute_sym_upper の sort/compress を回避する (BOYD2 で 15-20ms/call 削減)。
    pub(crate) fn permute(&self, perm: &[usize]) -> PermutedAugmentedKkt {
        let total = self.n + self.m_rows;
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
                let (r, c) = if new_row <= new_col {
                    (new_row, new_col)
                } else {
                    (new_col, new_row)
                };
                perm_entries.push((r, c, s));
            }
        }
        perm_entries.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        let mut new_col_ptr = Vec::with_capacity(total + 1);
        let mut new_row_ind = Vec::with_capacity(nnz);
        let mut new_static_values = Vec::with_capacity(nnz);
        let mut orig_slot_for_new_slot = Vec::with_capacity(nnz);
        let mut new_diag_var_slot = vec![usize::MAX; self.n];
        let mut new_diag_con_slot = vec![usize::MAX; self.m_rows];

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
            m_rows: self.m_rows,
            bound_var: self.bound_var.clone(),
            sys_sigma_idx: self.sys_sigma_idx.clone(),
            bound_sigma_idx: self.bound_sigma_idx.clone(),
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
    m_rows: usize,
    /// box 化時のみ Some (親 `AugmentedKktCache::bound_var` を引き継ぐ)。
    bound_var: Option<Vec<usize>>,
    /// σ index 写像 (親キャッシュから不変、置換は対角 slot 側で吸収)。
    sys_sigma_idx: Vec<usize>,
    bound_sigma_idx: Vec<usize>,
}

impl PermutedAugmentedKkt {
    pub(crate) fn materialize(&self, sigma_vec: &[f64], delta_p: f64, delta_d: f64) -> CscMatrix {
        debug_assert_eq!(
            sigma_vec.len(),
            self.m_rows + self.bound_var.as_ref().map_or(0, |v| v.len())
        );
        let mut values = self.static_values.clone();
        for j in 0..self.n {
            values[self.diag_var_slot[j]] += delta_p;
        }
        for k in 0..self.m_rows {
            values[self.diag_con_slot[k]] = -(sigma_vec[self.sys_sigma_idx[k]] + delta_d);
        }
        if let Some(bound_var) = &self.bound_var {
            for (t, &j) in bound_var.iter().enumerate() {
                values[self.diag_var_slot[j]] +=
                    bound_row_dual_scale(sigma_vec[self.bound_sigma_idx[t]], 0.0);
            }
        }
        let total = self.n + self.m_rows;
        CscMatrix::from_raw_parts(
            total,
            total,
            self.col_ptr.clone(),
            self.row_ind.clone(),
            values,
        )
    }
}

/// 不定 Q に対する慣性修正量 δ_ic (Q が上三角 CSC 前提)。
/// まず LLT で PSD 判定 (Gershgorin の保守誤判定を回避)、indefinite なら共通 helper
/// `linalg::gershgorin::psd_shift_from_gershgorin` で δ_ic = max(0, max_j(R_j − Q[j,j])) を返す。
pub(crate) fn compute_inertia_correction(q: &CscMatrix) -> f64 {
    if q.nrows() == 0 || q.values().iter().all(|&v| v == 0.0) {
        return 0.0;
    }
    if otspot_num::linalg::ldl::is_q_psd_by_cholesky(q) {
        return 0.0;
    }
    otspot_num::linalg::gershgorin::psd_shift_from_gershgorin(q)
}

/// box 化 augmented KKT のキャッシュ構築。
///
/// `a_base` は実制約行のみ (境界行なし)。境界行 t の対角寄与は
/// `materialize` ごとに `1/(σ_t + δ_d)` として x 対角 slot へ畳み込まれるため、
/// 行列次元は n + m_base のまま (extended の n + m_ext との差が box 化利得)。
/// `BoundBox` が空の場合は従来 (未 box 化) キャッシュと代数同値になる。
pub(crate) fn build_boxed_augmented_cache(
    q: &CscMatrix,
    a_base: &CscMatrix,
    bound: &BoundBox,
) -> AugmentedKktCache {
    // 保持境界行 (lb==ub 退化対) は系行として A_sys に stack される。
    let a_sys = build_boxed_system_matrix(a_base, bound);
    let m_orig = a_base.nrows();
    let sys_sigma_idx = bound.system_ext_rows(m_orig);
    let eliminated = bound.eliminated_rows();
    let bound_var: Vec<usize> = eliminated.iter().map(|&t| bound.var[t]).collect();
    let bound_sigma_idx: Vec<usize> = eliminated.iter().map(|&t| m_orig + t).collect();
    let mut cache = build_augmented_cache_inner(q, &a_sys, Some(bound_var));
    cache.sys_sigma_idx = sys_sigma_idx;
    cache.bound_sigma_idx = bound_sigma_idx;
    cache
}

#[allow(clippy::needless_range_loop)]
fn build_augmented_cache_inner(
    q: &CscMatrix,
    a_ext: &CscMatrix,
    bound_var: Option<Vec<usize>>,
) -> AugmentedKktCache {
    let n = q.nrows();
    let m_ext = a_ext.nrows();
    let total = n + m_ext;

    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); total];
    let mut diag_var_present = vec![false; n];

    // Part 1: Q 上三角 (row <= col、Q が完全格納でも上三角だけ取る)
    for col in 0..n {
        for k in q.col_ptr()[col]..q.col_ptr()[col + 1] {
            let row = q.row_ind()[k];
            if row <= col {
                col_entries[col].push((row, q.values()[k]));
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
        for idx in a_ext.col_ptr()[j]..a_ext.col_ptr()[j + 1] {
            let k = a_ext.row_ind()[idx];
            col_entries[n + k].push((j, a_ext.values()[idx]));
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
        m_rows: m_ext,
        bound_var,
        // 非 box / 保持行なしでは恒等写像 (box 化経路は呼び出し側が上書き)。
        sys_sigma_idx: (0..m_ext).collect(),
        bound_sigma_idx: Vec::new(),
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
    let n = q.nrows();
    let m_ext = a_ext.nrows();
    let total = n + m_ext;

    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    let mut diag_added = vec![false; n];
    for col in 0..n {
        for k in q.col_ptr()[col]..q.col_ptr()[col + 1] {
            let row = q.row_ind()[k];
            if row <= col {
                let v = q.values()[k] + if row == col { delta_p } else { 0.0 };
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
        for idx in a_ext.col_ptr()[j]..a_ext.col_ptr()[j + 1] {
            let k = a_ext.row_ind()[idx];
            let v = a_ext.values()[idx];
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

    let n = q.nrows();
    let m_ext = a_ext.nrows();

    let d_inv: Vec<f64> = sigma_vec.iter().map(|&s| 1.0 / (s + delta_d)).collect();

    let a_t = a_ext.transpose();

    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();

    for k in 0..m_ext {
        let start = a_t.col_ptr()[k];
        let end = a_t.col_ptr()[k + 1];
        let row_entries: Vec<(usize, f64)> = (start..end)
            .map(|p| (a_t.row_ind()[p], a_t.values()[p]))
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
        for k in q.col_ptr()[col]..q.col_ptr()[col + 1] {
            let row = q.row_ind()[k];
            if row <= col {
                *acc.entry((col, row)).or_insert(0.0) += q.values()[k];
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

    let s = CscMatrix::from_raw_parts(n, n, col_ptr, row_ind, values);
    (s, d_inv)
}

/// box 化 Schur 系 S = (Q + ρI + D_b) + A_sysᵀ D⁻¹ A_sys の上三角 CSC (n×n)。
///
/// box 化境界行 t は行として存在せず、対角寄与 d_t = 1/σ_t (δ_d 非適用 —
/// `bound_row_dual_scale` の doc) を D_b[j] = Σ_{t: var t} d_t として Q 対角へ
/// 加算する (augmented 版 `build_boxed_augmented_cache` の materialize と同一の
/// 正確 Schur 変換)。保持境界行 (lb==ub 退化対) は `build_boxed_system_matrix`
/// で A_sys の系行に stack され、実制約行と同じ独立スラック力学で解かれる。
/// `sigma_full` は extended レイアウト ([実制約 | 全境界行]) を受け取り、
/// 系行順への gather はこの関数が行う (`AugmentedKktCache::materialize` と同一規約)。
/// 戻り値:
/// - S: n×n 係数行列
/// - `d_inv_sys[k] = 1/(σ_k + δ_d)` (系行 k の dy 後退代入用、行順は A_sys)
/// - `d_b[t] = bound_row_dual_scale(σ_t, δ_d)` (box 化境界行 t の dy 復元・右辺変換用、
///   extended 境界 index t で添え字、保持行 slot は 0。`bound_row_dual_scale` 共有により
///   augmented バックエンドと同一値)
pub(crate) fn build_boxed_schur_system(
    q: &CscMatrix,
    a_base: &CscMatrix,
    sigma_full: &[f64],
    bound: &BoundBox,
    rho_p: f64,
    delta_d: f64,
) -> (CscMatrix, Vec<f64>, Vec<f64>) {
    use std::collections::BTreeMap;

    let n = q.nrows();
    debug_assert_eq!(sigma_full.len(), a_base.nrows() + bound.count());

    let a_sys = build_boxed_system_matrix(a_base, bound);
    let m_sys = a_sys.nrows();
    let sys_ext_rows = bound.system_ext_rows(a_base.nrows());
    let d_inv_sys: Vec<f64> = sys_ext_rows
        .iter()
        .map(|&ext| 1.0 / (sigma_full[ext] + delta_d))
        .collect();
    let eliminated = bound.eliminated_rows();
    let mut d_b = vec![0.0_f64; bound.count()];
    for &t in &eliminated {
        d_b[t] = bound_row_dual_scale(sigma_full[a_base.nrows() + t], 0.0);
    }

    let a_t = a_sys.transpose();

    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();

    for k in 0..m_sys {
        let start = a_t.col_ptr()[k];
        let end = a_t.col_ptr()[k + 1];
        let row_entries: Vec<(usize, f64)> = (start..end)
            .map(|p| (a_t.row_ind()[p], a_t.values()[p]))
            .collect();
        let dk = d_inv_sys[k];
        for (idx_a, &(i, v_i)) in row_entries.iter().enumerate() {
            for &(j, v_j) in &row_entries[idx_a..] {
                let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
                *acc.entry((hi, lo)).or_insert(0.0) += dk * v_i * v_j;
            }
        }
    }

    for col in 0..n {
        for k in q.col_ptr()[col]..q.col_ptr()[col + 1] {
            let row = q.row_ind()[k];
            if row <= col {
                *acc.entry((col, row)).or_insert(0.0) += q.values()[k];
            }
        }
    }

    for i in 0..n {
        *acc.entry((i, i)).or_insert(0.0) += rho_p;
    }

    // box 化境界行の正確 Schur 対角寄与 (係数 a_t = ±1 より a_t² d_t = d_t)。
    for &t in &eliminated {
        let j = bound.var[t];
        *acc.entry((j, j)).or_insert(0.0) += d_b[t];
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

    let s = CscMatrix::from_raw_parts(n, n, col_ptr, row_ind, values);
    (s, d_inv_sys, d_b)
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
    (0..n).map(|i| aug_mat.col_ptr()[i + 1] - 1).collect()
}

#[cfg(test)]
pub(crate) fn collect_part3_diag_indices(
    aug_mat: &CscMatrix,
    n: usize,
    m_ext: usize,
) -> Vec<usize> {
    (0..m_ext)
        .map(|k| aug_mat.col_ptr()[n + k + 1] - 1)
        .collect()
}

#[cfg(test)]
pub(crate) fn collect_q_diag_base(q: &CscMatrix, n: usize) -> Vec<f64> {
    let mut base = vec![0.0f64; n];
    for (col, val) in base.iter_mut().enumerate().take(n) {
        for k in q.col_ptr()[col]..q.col_ptr()[col + 1] {
            if q.row_ind()[k] == col {
                *val = q.values()[k];
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
        cache.mat.values_mut()[idx] = cache.q_diag_base[i] + delta_p;
    }
    for (k, &idx) in cache.part3_diag_idx.iter().enumerate() {
        cache.mat.values_mut()[idx] = -(sigma_vec[k] + delta_d);
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
            (
                "zero_plus_negative_diag",
                2,
                &[(0, 1, 1.0), (1, 1, -1.0)],
                2.0,
            ),
            // 3x3 zero-diag bilinear: Q=[[0,1,0],[1,0,1],[0,1,0]]
            // row sums = (1,2,1), diag=(0,0,0) → max(R-Q)=2 → δ ≥ 2
            ("zero_diag_3x3_chain", 3, &[(0, 1, 1.0), (1, 2, 1.0)], 2.0),
            // 4x4 partial zero-diag: Q[0,0]=2, Q[0,3]=3, rest zero
            // row sums = (3,0,0,3), diag=(2,0,0,0) → Gershgorin = (-1,0,0,-3) → δ ≥ 3
            (
                "zero_diag_4x4_extreme_offdiag",
                4,
                &[(0, 0, 2.0), (0, 3, 3.0)],
                3.0,
            ),
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
            (
                "pd_2x2_offdiag",
                2,
                &[(0, 0, 4.0), (0, 1, 1.0), (1, 1, 3.0)],
            ),
            (
                "psd_singular_rank1",
                2,
                &[(0, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0)],
            ),
            ("psd_with_zero_eig_3x3", 3, &[(0, 0, 1.0), (1, 1, 1.0)]),
            (
                "pd_large_offdiag_gershgorin_false_alarm",
                2,
                &[(0, 0, 1.0), (0, 1, 1.1), (1, 1, 2.0)],
            ),
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
            assert_eq!(aug.row_ind()[idx], i);
        }
        assert!((aug.values()[diag_idx[0]] - 2.1).abs() < 1e-14);
        assert!((aug.values()[diag_idx[1]] - 3.1).abs() < 1e-14);
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
            assert_eq!(aug.row_ind()[idx], n + k);
        }
        assert!((aug.values()[diag_idx[0]] - (-0.55)).abs() < 1e-14);
        assert!((aug.values()[diag_idx[1]] - (-0.85)).abs() < 1e-14);
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
        let prob =
            QpProblem::new_all_le(q, vec![0.0; 2], a, vec![3.0], vec![(0.0, f64::INFINITY); 2])
                .unwrap();
        let (a_ext, b_ext, m_ext, m_orig, n_lb, is_eq_ext) = build_extended_constraints(&prob);
        assert_eq!(m_orig, 1);
        assert_eq!(n_lb, 2);
        assert_eq!(m_ext, 3);
        assert_eq!(a_ext.nrows(), 3);
        assert_eq!(a_ext.ncols(), 2);
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

        assert_eq!(cache.mat.values().len(), aug_ref.values().len());
        for (i, (&got, &expected)) in cache
            .mat
            .values()
            .iter()
            .zip(aug_ref.values().iter())
            .enumerate()
        {
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
            q,
            vec![0.0; 2],
            a,
            vec![],
            vec![(0.0, 5.0), (f64::NEG_INFINITY, 10.0)],
        )
        .unwrap();
        let (a_ext, b_ext, m_ext, m_orig, n_lb, _) = build_extended_constraints(&prob);
        assert_eq!(m_orig, 0);
        assert_eq!(n_lb, 1);
        assert_eq!(m_ext, 3);
        assert_eq!(b_ext, vec![-0.0, 5.0, 10.0]);

        let mut row_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; m_ext];
        for col in 0..a_ext.ncols() {
            for k in a_ext.col_ptr()[col]..a_ext.col_ptr()[col + 1] {
                row_entries[a_ext.row_ind()[k]].push((col, a_ext.values()[k]));
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

    /// CSC (対称格納) を dense 両三角に展開する (テスト専用オラクル補助)。
    fn dense_sym(m: &CscMatrix) -> Vec<Vec<f64>> {
        let nn = m.nrows();
        let mut d = vec![vec![0.0_f64; nn]; nn];
        for col in 0..m.ncols().min(nn) {
            for k in m.col_ptr()[col]..m.col_ptr()[col + 1] {
                let r = m.row_ind()[k];
                let v = m.values()[k];
                d[r][col] = v;
                d[col][r] = v;
            }
        }
        d
    }

    /// 部分ピボット Gauss 消去 (小規模専用、faer/otspot 経路に依存しない独立オラクル)。
    fn dense_gauss(a: Vec<Vec<f64>>, b: &[f64]) -> Vec<f64> {
        let nn = b.len();
        let mut m = a;
        let mut r = b.to_vec();
        for c in 0..nn {
            let piv = (c..nn).max_by(|&i, &j| m[i][c].abs().partial_cmp(&m[j][c].abs()).unwrap());
            let piv = piv.expect("pivot exists");
            if m[piv][c].abs() < 1e-300 {
                panic!("singular oracle system at col {c}");
            }
            m.swap(c, piv);
            r.swap(c, piv);
            for i in (c + 1)..nn {
                let f = m[i][c] / m[c][c];
                if f == 0.0 {
                    continue;
                }
                for j in c..nn {
                    let s = m[c][j];
                    m[i][j] -= f * s;
                }
                let rc = r[c];
                r[i] -= f * rc;
            }
        }
        let mut sol = vec![0.0_f64; nn];
        for i in (0..nn).rev() {
            let mut v = r[i];
            for j in (i + 1)..nn {
                let xj = sol[j];
                v -= m[i][j] * xj;
            }
            sol[i] = v / m[i][i];
        }
        sol
    }

    fn close_all(got: &[f64], want: &[f64], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 1e-8 * 1.0_f64.max(w.abs()),
                "{label}[{i}]: got {g}, want {w}"
            );
        }
    }

    /// KKT 次元 + 正確 Schur の sentinel (revert すると fail):
    /// 境界行を box 化した augmented KKT は次元 n + m_real を保ち、その解
    /// (dx, dy_real) と後退代入復元 dy_bound = d_t(a_t·dx − rhs_t) は、境界行を
    /// 明示行として持つ extended 系 (独立オラクル: dense LU) と代数厳密一致する。
    /// box 化を元に戻すと (a) 次元 assert、(b) 右辺散布・復元レイアウトの対応が
    /// 崩れ、必ずどちらかが fail する。
    #[test]
    fn sentinel_boxed_kkt_exact_schur_vs_extended_oracle() {
        use crate::problem::ConstraintType;
        use crate::qp::problem::QpProblem;
        // n=3、実 Le 行 1 本、境界: x0 lb / x1 ub / x2 lb+ub → n_bound=4。
        let q = upper_tri_csc(3, &[(0, 0, 2.0), (1, 1, 1.0), (2, 2, 3.0), (0, 1, 0.4)]);
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 3).unwrap();
        let bounds = vec![
            (-1.0_f64, f64::INFINITY),
            (f64::NEG_INFINITY, 4.0),
            (-2.0, 2.0),
        ];
        let problem = QpProblem::new(
            q.clone(),
            vec![0.0; 3],
            a,
            vec![2.0],
            bounds,
            vec![ConstraintType::Le],
        )
        .unwrap();

        let (a_base, _b_base, _is_eq_base) = build_base_constraints(&problem);
        let bound = build_bound_box(&problem);
        let m_real = 1usize;
        let n_bound = bound.count();
        assert_eq!(n_bound, 4, "lb(2) + ub(2) rows");

        let n = 3usize;
        let (a_ext, _b_ext, m_ext, _m_orig, _n_lb, _is_eq_ext) =
            build_extended_constraints(&problem);
        assert_eq!(m_ext, m_real + n_bound);

        // sigma レイアウトは [実 | 境界] (境界は lb 昇順 → ub 昇順、extended と同一順)。
        let sigma_full = vec![0.7_f64, 0.3, 0.9, 0.15, 0.55];
        let rho = 0.11_f64;
        let delta = 0.17_f64;

        // 右辺は境界行も非ゼロにして Schur の右辺散布・後退代入を同時に検証する。
        let r_d = vec![0.3_f64, -1.1, 0.7];
        let rp_real = vec![0.6_f64];
        let rp_bound = vec![0.25_f64, -0.4, 1.2, -0.75];
        let mut rhs_ext: Vec<f64> = r_d.clone();
        rhs_ext.extend_from_slice(&rp_real);
        rhs_ext.extend_from_slice(&rp_bound);

        // --- オラクル: extended 系 (境界行を明示行として保持) を dense LU で解く ---
        let aug_ext = build_augmented_system(&q, &a_ext, &sigma_full, rho, delta);
        assert_eq!(aug_ext.nrows(), n + m_ext);
        // オラクルの δ 適用範囲を box 化の設計不変条件に揃える: dual damping は
        // 実制約行のみ。box 化境界行は従属スラックの正確 barrier 対角 1/σ_t で
        // 消去される (δ を入れると σ≪δ で対角寄与が消え、fraction-to-boundary
        // と組み合わさって μ 停止の deadlock になる — iter.rs x0 doc 参照)。
        // build_augmented_system は全行へ一律 δ を施すため、境界行 slot だけ
        // 対角を +δ してオラクルから除外する。
        let mut dense_ext = dense_sym(&aug_ext);
        for k in m_real..m_ext {
            dense_ext[n + k][n + k] += delta;
        }
        let sol_ext = dense_gauss(dense_ext, &rhs_ext);
        let dx_oracle = &sol_ext[..n];
        let dy_real_oracle = &sol_ext[n..n + m_real];
        let dy_bound_oracle = &sol_ext[n + m_real..];

        // --- box 化 augmented: 次元 sentinel + 解の一致 ---
        let cache = build_boxed_augmented_cache(&q, &a_base, &bound);
        assert_eq!(cache.sigma_len(), sigma_full.len());
        let mat_box = cache.materialize(&sigma_full, rho, delta);
        assert_eq!(
            mat_box.nrows(),
            n + m_real,
            "KKT dimension must be n+m_real (box-ified), not n+m_ext"
        );

        let d_b: Vec<f64> = sigma_full[m_real..]
            .iter()
            .map(|&s| bound_row_dual_scale(s, 0.0))
            .collect();
        let mut rhs_x_box = r_d.clone();
        for t in 0..n_bound {
            let j = bound.var[t];
            rhs_x_box[j] += bound.sign[t] * d_b[t] * rp_bound[t];
        }
        let mut rhs_box = rhs_x_box.clone();
        rhs_box.extend_from_slice(&rp_real);
        let sol_box = dense_gauss(dense_sym(&mat_box), &rhs_box);
        close_all(&sol_box[..n], dx_oracle, "dx (aug box vs extended oracle)");
        close_all(
            &sol_box[n..],
            dy_real_oracle,
            "dy_real (aug box vs extended oracle)",
        );
        for t in 0..n_bound {
            let j = bound.var[t];
            let dy_t = d_b[t] * (bound.sign[t] * sol_box[j] - rp_bound[t]);
            let want = dy_bound_oracle[t];
            assert!(
                (dy_t - want).abs() <= 1e-8 * 1.0_f64.max(want.abs()),
                "dy_bound[{t}]: back-substitution {dy_t} != oracle {want}"
            );
        }

        // --- 対角 D_b の畳み込みを手計算で直接検証 (変数 2: lb+ub 両方) ---
        let d_lb_2 = bound_row_dual_scale(sigma_full[m_real + 1], 0.0);
        let d_ub_2 = bound_row_dual_scale(sigma_full[m_real + 3], 0.0);
        let dense_box = dense_sym(&mat_box);
        let want_22 = q_diag(&q, 2) + rho + d_lb_2 + d_ub_2;
        assert!(
            (dense_box[2][2] - want_22).abs() <= 1e-12 * want_22.abs(),
            "D_b fold into x-diag 2: got {}, want {}",
            dense_box[2][2],
            want_22
        );

        // --- box 化 Schur backend: dx/dy_real が extended オラクルと一致 ---
        let (s_box, d_inv_real, d_b_schur) =
            build_boxed_schur_system(&q, &a_base, &sigma_full, &bound, rho, delta);
        assert_eq!(s_box.nrows(), n, "Schur S must be n×n");
        close_all(&d_b_schur, &d_b, "d_b (schur vs aug shared scale)");
        let mut rhs_schur = rhs_x_box.clone();
        // A_realᵀ D⁻¹ r_p: A 行 1 本 (係数 1.0, 列 0/1) なので scatter 明示。
        let transport = rp_real[0] * d_inv_real[0];
        rhs_schur[0] += transport;
        rhs_schur[1] += transport;
        let dx_schur = dense_gauss(dense_sym(&s_box), &rhs_schur);
        close_all(&dx_schur, dx_oracle, "dx (schur box vs extended oracle)");
        let a_dx: f64 = dx_schur[0] + dx_schur[1];
        let dy_real_schur = a_dx - rp_real[0];
        let dy_real_schur = dy_real_schur * d_inv_real[0];
        assert!(
            (dy_real_schur - dy_real_oracle[0]).abs() <= 1e-8,
            "dy_real schur: {dy_real_schur} != oracle {}",
            dy_real_oracle[0]
        );
    }

    /// SENTINEL: lb==ub 退化対は必ず系行へ保持 (eliminated=false)。
    ///
    /// 退化対では s_lb + s_ub ≡ 0 が厳密成立し、box 化 (従属スラック) すると
    /// 両スラックが床 BOUND_SLACK_FLOOR に常時張り付く。固定変数の dx 微少
    /// noise だけで fraction-to-boundary の α が全局的に潰れ反復が停留する
    /// (実測: 固定変数を含む BD-T6 形状 presolve-off で 51 反復 Stalled)。
    /// base 系がこの対を独立スラックの明示行として運んでいたことから、
    /// 退化対は系行として保持する (`bound_pair_is_degenerate` の doc)。
    ///
    /// Revert-fail (確認済み): `build_bound_box` の guard を外して全行 box 化
    /// (eliminated 常時 true) にすると本テストは kept_rows assert で FAIL する。
    #[test]
    fn sentinel_degenerate_bound_pair_kept_as_system_row() {
        use crate::qp::problem::QpProblem;
        let q = upper_tri_csc(2, &[(0, 0, 2.0), (1, 1, 2.0)]);
        let problem = QpProblem::new(
            q,
            vec![0.0, 0.0],
            CscMatrix::new(0, 2),
            vec![],
            vec![(-1.0_f64, 3.0), (2.0, 2.0)],
            vec![],
        )
        .unwrap();
        let bound = build_bound_box(&problem);
        assert_eq!(bound.count(), 4, "lb 2 + ub 2 境界行");
        // 行順は lb 昇順 → ub 昇順: t0=lb(var0), t1=lb(var1), t2=ub(var0), t3=ub(var1)。
        assert_eq!(
            bound.kept_rows(),
            vec![1, 3],
            "固定変数 (var1) の退化対は系行保持でなければならない"
        );
        assert_eq!(bound.eliminated_rows(), vec![0, 2]);
        let (a_base, _b_base, _is_eq) = build_base_constraints(&problem);
        let sys = build_boxed_system_matrix(&a_base, &bound);
        assert_eq!(sys.nrows(), 2, "系行 = 実制約 0 + 保持境界行 2");
        assert_eq!(bound.system_ext_rows(0), vec![1, 3]);
    }

    fn q_diag(q: &CscMatrix, j: usize) -> f64 {
        for k in q.col_ptr()[j]..q.col_ptr()[j + 1] {
            if q.row_ind()[k] == j {
                return q.values()[k];
            }
        }
        0.0
    }
}
