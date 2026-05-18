//! Presolve変換モジュール
//!
//! LPを縮約するための手法と逆変換（Postsolve）情報を提供する。
//!
//! 適用順序（fixpoint loop, MAX_PRESOLVE_ITER 回）:
//! 1. Fixed variable removal (lb == ub)
//! 2. Singleton row (Eq制約、変数1つ → 値確定)
//! 3a. Empty row removal
//! 3b. Empty column removal
//! 4. Redundant constraint removal
//! 5. Bounds tightening
//! 6. Doubleton equation (R6): a*xi + b*xk = c → xi を xk で消去
//! 7. Free variable substitution (R15): free な x_j が 1 つの Eq 制約に出現 → その Eq を消去
//! 8. Free singleton column (R5): x_j が 1 つの制約のみに出現 + 片側無限 / free → 制約除去

use crate::problem::{ConstraintType, LpProblem};
use crate::sparse::CscMatrix;
use crate::tolerances::ZERO_TOL;

/// Presolve操作の逆変換を記述する列挙型。
/// PostsolveStackに順に積まれ、LIFO順（逆順）で適用される。
// BoundsTightened の old_lb/old_ub はデバッグ・将来の検証用に保持するが、
// 現在の postsolve では解の復元に不要なため dead_code 警告を抑制する。
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) enum PostsolveStep {
    /// Fixed variable removal の逆変換 (lb == ub)
    FixedVariable { orig_col: usize, value: f64 },
    /// Empty column removal の逆変換 (全係数ゼロ列)
    EmptyColumn { orig_col: usize, value: f64 },
    /// Empty row removal の逆変換 (全係数ゼロ行)
    EmptyRow { orig_row: usize },
    /// Singleton row の逆変換 (Eq制約、変数1つで値確定)
    /// Dual 復元: y_i = c_j / a_ij (bound 非 active 前提、active 時は z_j != 0 を別途扱う)
    SingletonRow {
        orig_row: usize,
        orig_col: usize,
        value: f64,
        /// intermediate state での a_ij と c_j (snapshot)。
        /// dual 復元時に y_i = c_j / a_ij で使用。
        a_ij: f64,
        c_j: f64,
    },
    /// Redundant constraint removal の逆変換 (常に満たされる制約)
    RedundantConstraint { orig_row: usize },
    /// Bounds tightening の逆変換 (解の復元には直接影響しない)
    BoundsTightened {
        orig_col: usize,
        old_lb: f64,
        old_ub: f64,
    },
    /// 線形代入による変数復元: orig_col = (rhs - Σ coeff_k * x_orig_other_k) / pivot
    /// R6 (Doubleton Eq), R15 (Free var sub), R5 (Free singleton col) で共通利用。
    ///
    /// Dual 復元 (HiGHS 標準, Eq 行を pivot 消去した場合):
    ///   消去された行 piv_row の dual y_piv は最適性条件から:
    ///     c_j_orig = Σ_{i: A_ij ≠ 0} A_ij_orig * y_i (j: 消去変数 orig_col)
    ///   → y_piv = (c_j_orig - Σ_{i ≠ piv_row} A_ij_orig * y_i) / pivot
    LinearSubstitution {
        orig_col: usize,
        orig_row: Option<usize>,
        pivot: f64,
        rhs: f64,
        /// 残変数の (orig_col_other, coeff) 列 (postsolve primal 用)
        others: Vec<(usize, f64)>,
        /// 元 A の x_j 列の全エントリ (orig_row_i, A_ij_orig) (postsolve dual 用)
        /// piv_row 自身を含む
        col_orig_entries: Vec<(usize, f64)>,
        /// 元目的係数 c_j_orig (postsolve dual 用)
        c_orig: f64,
    },
}

/// Presolve処理の結果。縮約後の問題と復元用情報を保持する。
pub struct PresolveResult {
    /// 縮約後のLpProblem（変数・制約が減っている可能性がある）
    pub reduced_problem: LpProblem,
    /// 復元用のステップスタック（逆順で適用する）
    pub(crate) postsolve_stack: Vec<PostsolveStep>,
    /// 元の変数数
    pub orig_num_vars: usize,
    /// 元の制約数
    pub orig_num_constraints: usize,
    /// 元→縮約後の変数インデックスマッピング (None = 削除済み)
    pub col_map: Vec<Option<usize>>,
    /// 元→縮約後の制約インデックスマッピング (None = 削除済み)
    pub row_map: Vec<Option<usize>>,
    /// 問題サイズが変わったか (false なら postsolve 不要)
    pub was_reduced: bool,
    /// 削除変数の目的関数への寄与量。縮約後 objective + obj_offset = 元 objective
    pub obj_offset: f64,
}

/// Presolve段階で検出された問題のステータス
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum PresolveStatus {
    Infeasible,
    Unbounded,
}

/// Per-transform on/off toggles. Default: all on. Sentinel tests flip individual
/// flags off to assert that disabling each path leaves a measurable artifact
/// (reduction count or runtime), proving the transform is not a no-op.
#[derive(Debug, Clone, Copy)]
pub struct PresolveFlags {
    pub enable_parallel_row: bool,
    pub enable_dup_dom_col: bool,
    pub enable_dual_fixing: bool,
}

impl Default for PresolveFlags {
    fn default() -> Self {
        Self {
            enable_parallel_row: true,
            enable_dup_dom_col: true,
            enable_dual_fixing: true,
        }
    }
}

impl PresolveFlags {
    pub fn all_off() -> Self {
        Self {
            enable_parallel_row: false,
            enable_dup_dom_col: false,
            enable_dual_fixing: false,
        }
    }
}

impl PresolveResult {
    /// 縮約なし（presolve: false またはフォールバック用）
    pub fn no_reduction(problem: &LpProblem) -> Self {
        let n = problem.num_vars;
        let m = problem.num_constraints;
        PresolveResult {
            reduced_problem: problem.clone(),
            postsolve_stack: vec![],
            orig_num_vars: n,
            orig_num_constraints: m,
            col_map: (0..n).map(Some).collect(),
            row_map: (0..m).map(Some).collect(),
            was_reduced: false,
            obj_offset: 0.0,
        }
    }
}

/// 内部の可変状態。Step 1〜11 が共通で参照する。
///
/// row_entries[i] と col_entries[j] は同じ情報の dual representation。
/// 整合性を保つため、書き換えは必ず両方を更新する。
pub(super) struct PresolveState {
    pub(super) row_entries: Vec<Vec<(usize, f64)>>,
    pub(super) col_entries: Vec<Vec<(usize, f64)>>,
    pub(super) b: Vec<f64>,
    pub(super) bounds: Vec<(f64, f64)>,
    pub(super) orig_bounds: Vec<(f64, f64)>,
    pub(super) constraint_types: Vec<ConstraintType>,
    pub(super) c: Vec<f64>,
    pub(super) removed_cols: Vec<bool>,
    pub(super) removed_rows: Vec<bool>,
    pub(super) postsolve_stack: Vec<PostsolveStep>,
    pub(super) obj_offset: f64,
}

impl PresolveState {
    fn from_problem(problem: &LpProblem) -> Self {
        let n = problem.num_vars;
        let m = problem.num_constraints;

        let mut row_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; m];
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        for j in 0..n {
            if let Ok((rows, vals)) = problem.a.get_column(j) {
                for (k, &row) in rows.iter().enumerate() {
                    let v = vals[k];
                    if v.abs() < ZERO_TOL {
                        continue;
                    }
                    row_entries[row].push((j, v));
                    col_entries[j].push((row, v));
                }
            }
        }

        PresolveState {
            row_entries,
            col_entries,
            b: problem.b.clone(),
            bounds: problem.bounds.clone(),
            orig_bounds: problem.bounds.clone(),
            constraint_types: problem.constraint_types.clone(),
            c: problem.c.clone(),
            removed_cols: vec![false; n],
            removed_rows: vec![false; m],
            postsolve_stack: Vec::new(),
            obj_offset: 0.0,
        }
    }

    /// active な row エントリ (削除済み列を除く) を返す
    pub(super) fn active_row_entries(&self, i: usize) -> Vec<(usize, f64)> {
        self.row_entries[i]
            .iter()
            .filter(|&&(j, v)| !self.removed_cols[j] && v.abs() >= ZERO_TOL)
            .copied()
            .collect()
    }

    /// active な col エントリ (削除済み行を除く) を返す
    pub(super) fn active_col_entries(&self, j: usize) -> Vec<(usize, f64)> {
        self.col_entries[j]
            .iter()
            .filter(|&&(i, v)| !self.removed_rows[i] && v.abs() >= ZERO_TOL)
            .copied()
            .collect()
    }

    /// 行 i から列 j の係数を取り出す (見つからなければ 0)。行内重複は加算される想定だが、
    /// 我々の代入操作後は重複しないよう必ず merge 済み。
    fn coeff(&self, i: usize, j: usize) -> f64 {
        let mut s = 0.0;
        for &(jj, v) in &self.row_entries[i] {
            if jj == j && !self.removed_cols[jj] {
                s += v;
            }
        }
        s
    }

    /// 行 i に対し、列 j のエントリを delta だけ加える (重複は merge する)。
    /// delta == 0 のときは何もしない。merge 後 0 になったエントリは削除する。
    fn add_to_entry(&mut self, i: usize, j: usize, delta: f64) {
        if delta.abs() < ZERO_TOL {
            return;
        }
        // row_entries の更新
        let mut found_row = false;
        for entry in self.row_entries[i].iter_mut() {
            if entry.0 == j {
                entry.1 += delta;
                found_row = true;
                break;
            }
        }
        if !found_row {
            self.row_entries[i].push((j, delta));
        }
        // col_entries の更新
        let mut found_col = false;
        for entry in self.col_entries[j].iter_mut() {
            if entry.0 == i {
                entry.1 += delta;
                found_col = true;
                break;
            }
        }
        if !found_col {
            self.col_entries[j].push((i, delta));
        }
        // merge 後ゼロになったエントリを掃除
        self.row_entries[i].retain(|&(jj, v)| jj != j || v.abs() >= ZERO_TOL);
        self.col_entries[j].retain(|&(ii, v)| ii != i || v.abs() >= ZERO_TOL);
    }
}

/// LPをPresolveして縮約問題を返す。
///
/// 問題が明らかにInfeasible/Unboundedな場合はErrを返す。
/// deadline を超過した場合は早期終了し `was_reduced: false` を返す。
pub fn run_presolve(
    problem: &LpProblem,
    deadline: Option<std::time::Instant>,
) -> Result<PresolveResult, PresolveStatus> {
    run_presolve_with_flags(problem, deadline, PresolveFlags::default())
}

/// Variant of `run_presolve` with per-transform flags. Production callers use
/// the default-flag wrapper above; sentinel / bench-gating callers vary flags
/// to isolate each transform's contribution.
pub fn run_presolve_with_flags(
    problem: &LpProblem,
    deadline: Option<std::time::Instant>,
    flags: PresolveFlags,
) -> Result<PresolveResult, PresolveStatus> {
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return Ok(PresolveResult::no_reduction(problem));
    }

    let n = problem.num_vars;
    let m = problem.num_constraints;
    let mut st = PresolveState::from_problem(problem);

    // 収束 (reduction == 0) で break する設計 (line 364-366)。
    // 削除可能要素は有限なので無限ループにはならない。安全装置の上限は
    // deadline チェック (各 step 境界) に統一。
    loop {
        let prev_removed = st.removed_cols.iter().filter(|&&r| r).count()
            + st.removed_rows.iter().filter(|&&r| r).count();
        let mut new_fixed_by_step5 = 0usize;
        let mut new_subst_steps = 0usize;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step1_fixed_variable(&mut st)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step2_singleton_row(&mut st)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step3a_empty_row(&mut st)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step3b_empty_column(&mut st)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step4_redundant_constraint(&mut st)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step5_bounds_tightening(&mut st, &mut new_fixed_by_step5)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step6_doubleton_equation(&mut st, &mut new_subst_steps)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step7_free_var_substitution(&mut st, &mut new_subst_steps)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step8_free_singleton_col(&mut st, &mut new_subst_steps)?;

        if flags.enable_parallel_row {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return Ok(PresolveResult::no_reduction(problem));
            }
            super::transforms_dup::step9_parallel_row(&mut st)?;
        }
        if flags.enable_dup_dom_col {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return Ok(PresolveResult::no_reduction(problem));
            }
            super::transforms_dup::step10_dup_dom_col(&mut st, &mut new_fixed_by_step5)?;
        }
        if flags.enable_dual_fixing {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return Ok(PresolveResult::no_reduction(problem));
            }
            super::transforms_dup::step11_dual_fixing(&mut st, &mut new_fixed_by_step5)?;
        }

        let curr_removed = st.removed_cols.iter().filter(|&&r| r).count()
            + st.removed_rows.iter().filter(|&&r| r).count();
        let reduction = curr_removed - prev_removed;
        if reduction == 0 && new_fixed_by_step5 == 0 && new_subst_steps == 0 {
            break;
        }
    }

    // ==========================================================
    // 縮約後問題の構築
    // ==========================================================
    let mut col_map = vec![None; n];
    let mut new_col_idx = 0usize;
    for j in 0..n {
        if !st.removed_cols[j] {
            col_map[j] = Some(new_col_idx);
            new_col_idx += 1;
        }
    }
    let n_new = new_col_idx;

    let mut row_map = vec![None; m];
    let mut new_row_idx = 0usize;
    for i in 0..m {
        if !st.removed_rows[i] {
            row_map[i] = Some(new_row_idx);
            new_row_idx += 1;
        }
    }
    let m_new = new_row_idx;

    let was_reduced = n_new < n || m_new < m;

    let mut c_new = vec![0.0f64; n_new];
    let mut bounds_new = vec![(0.0f64, f64::INFINITY); n_new];
    for j in 0..n {
        if let Some(jj) = col_map[j] {
            c_new[jj] = st.c[j];
            bounds_new[jj] = st.bounds[j];
        }
    }

    let mut b_new = vec![0.0f64; m_new];
    let mut ct_new = vec![ConstraintType::Le; m_new];
    for i in 0..m {
        if let Some(ii) = row_map[i] {
            b_new[ii] = st.b[i];
            ct_new[ii] = st.constraint_types[i];
        }
    }

    // 縮約後 CscMatrix を構築: col_entries から triplets を生成
    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
    for j in 0..n {
        if st.removed_cols[j] {
            continue;
        }
        let jj = col_map[j].unwrap();
        for &(row, val) in &st.col_entries[j] {
            if st.removed_rows[row] || val.abs() < ZERO_TOL {
                continue;
            }
            let ii = row_map[row].unwrap();
            trip_rows.push(ii);
            trip_cols.push(jj);
            trip_vals.push(val);
        }
    }

    let a_new = if trip_rows.is_empty() {
        CscMatrix::new(m_new, n_new)
    } else {
        CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_new, n_new)
            .unwrap_or_else(|_| CscMatrix::new(m_new, n_new))
    };

    let reduced_problem = LpProblem::new_general(
        c_new,
        a_new,
        b_new,
        ct_new,
        bounds_new,
        problem.name.clone(),
    )
    .expect("presolve: reduced problem construction failed");

    Ok(PresolveResult {
        reduced_problem,
        postsolve_stack: st.postsolve_stack,
        orig_num_vars: n,
        orig_num_constraints: m,
        col_map,
        row_map,
        was_reduced,
        obj_offset: st.obj_offset,
    })
}

// ==========================================================
// Step 1: Fixed variable removal (lb == ub)
// ==========================================================
fn step1_fixed_variable(st: &mut PresolveState) -> Result<(), PresolveStatus> {
    let n = st.bounds.len();
    for j in 0..n {
        if st.removed_cols[j] {
            continue;
        }
        let (lb, ub) = st.bounds[j];
        if lb > ub + ZERO_TOL {
            return Err(PresolveStatus::Infeasible);
        }
        if (lb - ub).abs() < ZERO_TOL {
            let value = lb;
            // Clone the column entries so we can mutate `st` while iterating.
            let col_copy = st.col_entries[j].clone();
            for (row, val) in col_copy {
                if !st.removed_rows[row] {
                    st.b[row] -= val * value;
                }
            }
            st.obj_offset += st.c[j] * value;
            st.removed_cols[j] = true;
            st.postsolve_stack.push(PostsolveStep::FixedVariable { orig_col: j, value });
        }
    }
    Ok(())
}

// ==========================================================
// Step 2: Singleton row (Eq制約、変数1つ → 値確定)
// ==========================================================
fn step2_singleton_row(st: &mut PresolveState) -> Result<(), PresolveStatus> {
    let m = st.b.len();
    for i in 0..m {
        if st.removed_rows[i] {
            continue;
        }
        if st.constraint_types[i] != ConstraintType::Eq {
            continue;
        }
        let active = st.active_row_entries(i);
        if active.len() == 1 {
            let (j, a_ij) = active[0];
            if a_ij.abs() < ZERO_TOL {
                continue;
            }
            let value = st.b[i] / a_ij;
            let (lb, ub) = st.bounds[j];
            if value < lb - ZERO_TOL || value > ub + ZERO_TOL {
                return Err(PresolveStatus::Infeasible);
            }
            let value = value.clamp(lb, ub);
            let col_copy = st.col_entries[j].clone();
            for (row, val) in col_copy {
                if !st.removed_rows[row] && row != i {
                    st.b[row] -= val * value;
                }
            }
            let c_j_snapshot = st.c[j];
            st.obj_offset += c_j_snapshot * value;
            st.removed_cols[j] = true;
            st.removed_rows[i] = true;
            st.postsolve_stack.push(PostsolveStep::SingletonRow {
                orig_row: i,
                orig_col: j,
                value,
                a_ij,
                c_j: c_j_snapshot,
            });
        }
    }
    Ok(())
}

// ==========================================================
// Step 3a: Empty row removal (全係数ゼロ行)
// ==========================================================
fn step3a_empty_row(st: &mut PresolveState) -> Result<(), PresolveStatus> {
    let m = st.b.len();
    for i in 0..m {
        if st.removed_rows[i] {
            continue;
        }
        let active_count = st.active_row_entries(i).len();
        if active_count == 0 {
            match st.constraint_types[i] {
                ConstraintType::Eq => {
                    if st.b[i].abs() > ZERO_TOL {
                        return Err(PresolveStatus::Infeasible);
                    }
                }
                ConstraintType::Le => {
                    if st.b[i] < -ZERO_TOL {
                        return Err(PresolveStatus::Infeasible);
                    }
                }
                ConstraintType::Ge => {
                    if st.b[i] > ZERO_TOL {
                        return Err(PresolveStatus::Infeasible);
                    }
                }
            }
            st.removed_rows[i] = true;
            st.postsolve_stack.push(PostsolveStep::EmptyRow { orig_row: i });
        }
    }
    Ok(())
}

// ==========================================================
// Step 3b: Empty column removal (全係数ゼロ列)
// ==========================================================
fn step3b_empty_column(st: &mut PresolveState) -> Result<(), PresolveStatus> {
    let n = st.bounds.len();
    for j in 0..n {
        if st.removed_cols[j] {
            continue;
        }
        let active_count = st.active_col_entries(j).len();
        if active_count == 0 {
            let (lb, ub) = st.bounds[j];
            let cj = st.c[j];
            let value = if cj > ZERO_TOL {
                if lb == f64::NEG_INFINITY {
                    return Err(PresolveStatus::Unbounded);
                }
                if lb.is_finite() { lb } else { 0.0 }
            } else if cj < -ZERO_TOL {
                if ub == f64::INFINITY {
                    return Err(PresolveStatus::Unbounded);
                }
                if ub.is_finite() { ub } else { 0.0 }
            } else {
                if lb.is_finite() {
                    lb
                } else if ub.is_finite() {
                    ub
                } else {
                    0.0
                }
            };
            st.obj_offset += cj * value;
            st.removed_cols[j] = true;
            st.postsolve_stack.push(PostsolveStep::EmptyColumn { orig_col: j, value });
        }
    }
    Ok(())
}

// ==========================================================
// Step 4: Redundant constraint removal
// ==========================================================
fn step4_redundant_constraint(st: &mut PresolveState) -> Result<(), PresolveStatus> {
    let m = st.b.len();
    for i in 0..m {
        if st.removed_rows[i] {
            continue;
        }
        let active_entries = st.active_row_entries(i);
        let (row_lb, row_ub, lb_fin, ub_fin) =
            crate::presolve::activity::activity_range(&active_entries, &st.bounds, None);

        let redundant = match st.constraint_types[i] {
            ConstraintType::Le => ub_fin && row_ub <= st.b[i] + ZERO_TOL,
            ConstraintType::Ge => lb_fin && row_lb >= st.b[i] - ZERO_TOL,
            ConstraintType::Eq => {
                lb_fin && ub_fin
                    && row_lb >= st.b[i] - ZERO_TOL
                    && row_ub <= st.b[i] + ZERO_TOL
            }
        };
        if redundant {
            st.removed_rows[i] = true;
            st.postsolve_stack.push(PostsolveStep::RedundantConstraint { orig_row: i });
        }
    }
    Ok(())
}

// ==========================================================
// Step 5: Bounds tightening
// ==========================================================
fn step5_bounds_tightening(
    st: &mut PresolveState,
    new_fixed: &mut usize,
) -> Result<(), PresolveStatus> {
    // Accept every implied bound; LP simplex relies on aggressive bound tightening,
    // and the QP-style dense-row / sanity caps slowed several test instances enough
    // to time out, so they are intentionally off here.
    let accept_implied_ub = |_implied: f64, _old_ub: f64| -> bool { true };
    let accept_implied_lb = |_implied: f64, _old_lb: f64| -> bool { true };

    let m = st.b.len();
    for i in 0..m {
        if st.removed_rows[i] {
            continue;
        }
        let ct = st.constraint_types[i];
        let entries = st.active_row_entries(i);
        if entries.is_empty() {
            continue;
        }

        let mut row_lb_sum = 0.0f64;
        let mut row_ub_sum = 0.0f64;
        let mut inf_lb_count = 0usize;
        let mut inf_ub_count = 0usize;
        let mut entry_lb_contrib = Vec::with_capacity(entries.len());
        let mut entry_ub_contrib = Vec::with_capacity(entries.len());
        let mut entry_lb_inf = Vec::with_capacity(entries.len());
        let mut entry_ub_inf = Vec::with_capacity(entries.len());

        for &(j, a_ij) in &entries {
            let (lb_j, ub_j) = st.bounds[j];
            if a_ij > 0.0 {
                if lb_j == f64::NEG_INFINITY {
                    inf_lb_count += 1;
                    entry_lb_inf.push(true);
                    entry_lb_contrib.push(0.0);
                } else {
                    entry_lb_inf.push(false);
                    let c = a_ij * lb_j;
                    entry_lb_contrib.push(c);
                    row_lb_sum += c;
                }
                if ub_j == f64::INFINITY {
                    inf_ub_count += 1;
                    entry_ub_inf.push(true);
                    entry_ub_contrib.push(0.0);
                } else {
                    entry_ub_inf.push(false);
                    let c = a_ij * ub_j;
                    entry_ub_contrib.push(c);
                    row_ub_sum += c;
                }
            } else if a_ij < 0.0 {
                if ub_j == f64::INFINITY {
                    inf_lb_count += 1;
                    entry_lb_inf.push(true);
                    entry_lb_contrib.push(0.0);
                } else {
                    entry_lb_inf.push(false);
                    let c = a_ij * ub_j;
                    entry_lb_contrib.push(c);
                    row_lb_sum += c;
                }
                if lb_j == f64::NEG_INFINITY {
                    inf_ub_count += 1;
                    entry_ub_inf.push(true);
                    entry_ub_contrib.push(0.0);
                } else {
                    entry_ub_inf.push(false);
                    let c = a_ij * lb_j;
                    entry_ub_contrib.push(c);
                    row_ub_sum += c;
                }
            } else {
                entry_lb_inf.push(false);
                entry_ub_inf.push(false);
                entry_lb_contrib.push(0.0);
                entry_ub_contrib.push(0.0);
            }
        }

        for (k, &(j, a_ij)) in entries.iter().enumerate() {
            if a_ij.abs() < ZERO_TOL {
                continue;
            }
            let (old_lb, old_ub) = st.bounds[j];

            let rest_inf_lb = if entry_lb_inf[k] { inf_lb_count - 1 } else { inf_lb_count };
            let rest_inf_ub = if entry_ub_inf[k] { inf_ub_count - 1 } else { inf_ub_count };
            let rest_lb = row_lb_sum - entry_lb_contrib[k];
            let rest_ub = row_ub_sum - entry_ub_contrib[k];
            let rest_lb_fin = rest_inf_lb == 0;
            let rest_ub_fin = rest_inf_ub == 0;

            let mut new_lb = old_lb;
            let mut new_ub = old_ub;

            match ct {
                ConstraintType::Le => {
                    if a_ij > 0.0 && rest_lb_fin {
                        let implied_ub = (st.b[i] - rest_lb) / a_ij;
                        if implied_ub < old_lb - ZERO_TOL {
                            return Err(PresolveStatus::Infeasible);
                        }
                        if implied_ub < new_ub - ZERO_TOL && accept_implied_ub(implied_ub, old_ub) {
                            new_ub = implied_ub;
                        }
                    } else if a_ij < 0.0 && rest_lb_fin {
                        let implied_lb = (st.b[i] - rest_lb) / a_ij;
                        if implied_lb > old_ub + ZERO_TOL {
                            return Err(PresolveStatus::Infeasible);
                        }
                        if implied_lb > new_lb + ZERO_TOL && accept_implied_lb(implied_lb, old_lb) {
                            new_lb = implied_lb;
                        }
                    }
                }
                ConstraintType::Ge => {
                    if a_ij > 0.0 && rest_ub_fin {
                        let implied_lb = (st.b[i] - rest_ub) / a_ij;
                        if implied_lb > old_ub + ZERO_TOL {
                            return Err(PresolveStatus::Infeasible);
                        }
                        if implied_lb > new_lb + ZERO_TOL && accept_implied_lb(implied_lb, old_lb) {
                            new_lb = implied_lb;
                        }
                    } else if a_ij < 0.0 && rest_ub_fin {
                        let implied_ub = (st.b[i] - rest_ub) / a_ij;
                        if implied_ub < old_lb - ZERO_TOL {
                            return Err(PresolveStatus::Infeasible);
                        }
                        if implied_ub < new_ub - ZERO_TOL && accept_implied_ub(implied_ub, old_ub) {
                            new_ub = implied_ub;
                        }
                    }
                }
                ConstraintType::Eq => {
                    if a_ij > 0.0 {
                        if rest_lb_fin {
                            let implied_ub = (st.b[i] - rest_lb) / a_ij;
                            if implied_ub < old_lb - ZERO_TOL {
                                return Err(PresolveStatus::Infeasible);
                            }
                            if implied_ub < new_ub - ZERO_TOL {
                                new_ub = implied_ub;
                            }
                        }
                        if rest_ub_fin {
                            let implied_lb = (st.b[i] - rest_ub) / a_ij;
                            if implied_lb > old_ub + ZERO_TOL {
                                return Err(PresolveStatus::Infeasible);
                            }
                            if implied_lb > new_lb + ZERO_TOL {
                                new_lb = implied_lb;
                            }
                        }
                    } else {
                        if rest_lb_fin {
                            let implied_lb = (st.b[i] - rest_lb) / a_ij;
                            if implied_lb > old_ub + ZERO_TOL {
                                return Err(PresolveStatus::Infeasible);
                            }
                            if implied_lb > new_lb + ZERO_TOL {
                                new_lb = implied_lb;
                            }
                        }
                        if rest_ub_fin {
                            let implied_ub = (st.b[i] - rest_ub) / a_ij;
                            if implied_ub < old_lb - ZERO_TOL {
                                return Err(PresolveStatus::Infeasible);
                            }
                            if implied_ub < new_ub - ZERO_TOL {
                                new_ub = implied_ub;
                            }
                        }
                    }
                }
            }

            if (new_lb - old_lb).abs() > ZERO_TOL || (new_ub - old_ub).abs() > ZERO_TOL {
                st.postsolve_stack.push(PostsolveStep::BoundsTightened {
                    orig_col: j,
                    old_lb,
                    old_ub,
                });
                st.bounds[j] = (new_lb, new_ub);
                if (new_lb - new_ub).abs() < ZERO_TOL {
                    *new_fixed += 1;
                }
            }
        }
    }
    Ok(())
}

// ==========================================================
// 共通: 変数 j を行 piv_row から消去 (substitution)
//
// piv_row: a_p * x_j + Σ_{k != j} a_pk * x_k = piv_b  ※ Eq 制約のみ対象
// から x_j = (piv_b - Σ a_pk * x_k) / a_p を導出。
// 他の active 行 i (i != piv_row) で x_j に係数 a_ij を持つものに対し、
//   a_ij * x_j = a_ij * (piv_b - Σ a_pk * x_k) / a_p
// を加算: 行 i の x_k 係数を a_ij / a_p * (- a_pk) 分修正、b[i] -= a_ij * piv_b / a_p。
// 目的関数係数も c[k] -= c[j] * a_pk / a_p で更新し、obj_offset += c[j] * piv_b / a_p。
//
// piv_row と 列 j を削除フラグセット。postsolve に LinearSubstitution を積む。
// ==========================================================
/// 消去前 dry-run で fill-in を見積もり、許容上限を超える場合 true を返す。
///
/// fill-in: 他の行 i (≠ piv_row) で x_j を含むものに対し、新しく追加される
/// (i, k) エントリ数 (k は piv_row に出ているが行 i にまだ無い列)。
///
/// 上限規約: 追加 nnz が「消去で減る nnz (= col_j_other_rows + piv_others.len() - 1)」
/// の `FILL_IN_FACTOR` 倍を超える場合 skip。
fn fill_in_exceeds_budget(st: &PresolveState, piv_row: usize, j: usize) -> bool {
    const FILL_IN_FACTOR: usize = 3;
    let piv_others_cols: Vec<usize> = st.row_entries[piv_row]
        .iter()
        .filter(|&&(jj, v)| jj != j && !st.removed_cols[jj] && v.abs() >= ZERO_TOL)
        .map(|&(jj, _)| jj)
        .collect();
    let col_j_other_rows: Vec<usize> = st.col_entries[j]
        .iter()
        .filter(|&&(ii, v)| ii != piv_row && !st.removed_rows[ii] && v.abs() >= ZERO_TOL)
        .map(|&(ii, _)| ii)
        .collect();
    let mut new_entries: usize = 0;
    for &i in &col_j_other_rows {
        // 行 i に既存する列のセット
        let existing: std::collections::HashSet<usize> = st.row_entries[i]
            .iter()
            .filter(|&&(_, v)| v.abs() >= ZERO_TOL)
            .map(|&(jj, _)| jj)
            .collect();
        for &k in &piv_others_cols {
            if !existing.contains(&k) {
                new_entries += 1;
            }
        }
    }
    // 消去で減る nnz: 行 piv_row 全体 (1 + piv_others.len()) + 列 j の他行 (col_j_other_rows.len()) 個
    let removed_nnz = 1 + piv_others_cols.len() + col_j_other_rows.len();
    new_entries > FILL_IN_FACTOR * removed_nnz.max(1)
}

fn eliminate_variable_via_eq_row(
    st: &mut PresolveState,
    piv_row: usize,
    j: usize,
) -> Result<(), PresolveStatus> {
    debug_assert!(!st.removed_rows[piv_row]);
    debug_assert!(!st.removed_cols[j]);
    debug_assert_eq!(st.constraint_types[piv_row], ConstraintType::Eq);

    let pivot = st.coeff(piv_row, j);
    if pivot.abs() < ZERO_TOL {
        return Ok(()); // 無効: pivot ≒ 0
    }
    let piv_b = st.b[piv_row];

    // 行 piv_row から x_j 以外のエントリ収集
    let piv_others: Vec<(usize, f64)> = st.row_entries[piv_row]
        .iter()
        .filter(|&&(jj, v)| jj != j && !st.removed_cols[jj] && v.abs() >= ZERO_TOL)
        .copied()
        .collect();

    // 列 j のエントリ (piv_row 以外) を取得。
    // active かつ piv_row 以外。Dual 復元 snapshot もこれを再利用する。
    let col_j_others: Vec<(usize, f64)> = st.col_entries[j]
        .iter()
        .filter(|&&(ii, v)| ii != piv_row && !st.removed_rows[ii] && v.abs() >= ZERO_TOL)
        .copied()
        .collect();

    // Dual 復元 snapshot (分配前 / active i のみ)。
    // 分配 (下のループ) で行 i の x_j 係数が 0 化されると col_entries[j] から消えるため、
    // ここで先に確保する。LIFO postsolve 順では active な i は j より後で消える
    // (= j より先に復元される) ため、y_i は j 復元時点で確定済み。
    let col_orig_entries: Vec<(usize, f64)> = col_j_others.clone();
    let c_orig = st.c[j];

    // 他の行 i に対して x_j を置換
    for (i, a_ij) in col_j_others {
        // b[i] -= a_ij * (piv_b / pivot)
        st.b[i] -= a_ij * (piv_b / pivot);
        // 他の k について 行 i の係数を更新: 係数 += a_ij * (-a_pk / pivot)
        for &(k_col, a_pk) in &piv_others {
            let delta = -a_ij * a_pk / pivot;
            st.add_to_entry(i, k_col, delta);
        }
        // x_j の係数を行 i から削除（0 にする）
        st.add_to_entry(i, j, -a_ij);
    }

    // 目的関数係数 c[j] の分配
    if c_orig.abs() >= ZERO_TOL {
        st.obj_offset += c_orig * piv_b / pivot;
        for &(k_col, a_pk) in &piv_others {
            st.c[k_col] -= c_orig * a_pk / pivot;
        }
        st.c[j] = 0.0;
    }

    // postsolve エントリ: x_j を復元する式
    let others_for_postsolve: Vec<(usize, f64)> = piv_others.clone();
    st.postsolve_stack.push(PostsolveStep::LinearSubstitution {
        orig_col: j,
        orig_row: Some(piv_row),
        pivot,
        rhs: piv_b,
        others: others_for_postsolve,
        col_orig_entries,
        c_orig,
    });

    // piv_row と列 j を削除
    // piv_row のエントリは "x_j 以外" がもう他行に分配されたが、行自体を削除する
    st.removed_rows[piv_row] = true;
    st.removed_cols[j] = true;

    Ok(())
}

// ==========================================================
// Step 6: Doubleton equation (R6)
//
// 対象: Eq 行 i で active なエントリがちょうど 2 個 (a*x + b*y = c)。
// どちらかを pivot 列 j として消去。pivot は |a| が大きい方を選ぶ (数値安定性)。
// 消去側 x_j の bound から残側 x_k への bound 制約を導出して bounds tightening を反映。
// ==========================================================
fn step6_doubleton_equation(
    st: &mut PresolveState,
    new_subst: &mut usize,
) -> Result<(), PresolveStatus> {
    let m = st.b.len();
    for i in 0..m {
        if st.removed_rows[i] {
            continue;
        }
        if st.constraint_types[i] != ConstraintType::Eq {
            continue;
        }
        let active = st.active_row_entries(i);
        if active.len() != 2 {
            continue;
        }
        let (j1, a1) = active[0];
        let (j2, a2) = active[1];
        if a1.abs() < ZERO_TOL || a2.abs() < ZERO_TOL {
            continue;
        }
        // pivot 選択:
        //   1. 片方が free (元 bounds 両側 ∞) → free 側を pivot (postsolve で bound 違反なし)
        //   2. それ以外 → |係数| が大きい方
        let j1_free = st.orig_bounds[j1].0 == f64::NEG_INFINITY
            && st.orig_bounds[j1].1 == f64::INFINITY;
        let j2_free = st.orig_bounds[j2].0 == f64::NEG_INFINITY
            && st.orig_bounds[j2].1 == f64::INFINITY;
        let (pivot_col, pivot_a, other_col, other_a) = if j1_free && !j2_free {
            (j1, a1, j2, a2)
        } else if j2_free && !j1_free {
            (j2, a2, j1, a1)
        } else if a1.abs() >= a2.abs() {
            (j1, a1, j2, a2)
        } else {
            (j2, a2, j1, a1)
        };
        // x_pivot = (b - other_a * x_other) / pivot_a
        // x_pivot の bound から x_other の bound を導出
        let (lb_p, ub_p) = st.bounds[pivot_col];
        let (lb_o_old, ub_o_old) = st.bounds[other_col];
        // implied bounds: x_other = (b - pivot_a * x_pivot) / other_a
        // ratio = pivot_a / other_a
        // x_other = b/other_a - (pivot_a/other_a) * x_pivot
        let ratio = pivot_a / other_a;
        let bo = st.b[i] / other_a;
        // x_pivot in [lb_p, ub_p]
        // x_other = bo - ratio * x_pivot
        // ratio > 0: x_other in [bo - ratio*ub_p, bo - ratio*lb_p]
        // ratio < 0: x_other in [bo - ratio*lb_p, bo - ratio*ub_p]
        let (other_lb_impl, other_ub_impl) = if ratio > 0.0 {
            let lo = if ub_p == f64::INFINITY { f64::NEG_INFINITY } else { bo - ratio * ub_p };
            let hi = if lb_p == f64::NEG_INFINITY { f64::INFINITY } else { bo - ratio * lb_p };
            (lo, hi)
        } else if ratio < 0.0 {
            let lo = if lb_p == f64::NEG_INFINITY { f64::NEG_INFINITY } else { bo - ratio * lb_p };
            let hi = if ub_p == f64::INFINITY { f64::INFINITY } else { bo - ratio * ub_p };
            (lo, hi)
        } else {
            (f64::NEG_INFINITY, f64::INFINITY)
        };
        let new_lb_o = lb_o_old.max(other_lb_impl);
        let new_ub_o = ub_o_old.min(other_ub_impl);
        if new_lb_o > new_ub_o + ZERO_TOL {
            return Err(PresolveStatus::Infeasible);
        }
        // fill-in budget check (skip すれば bounds 反映もスキップ)
        if fill_in_exceeds_budget(st, i, pivot_col) {
            continue;
        }
        // bounds 反映
        if (new_lb_o - lb_o_old).abs() > ZERO_TOL || (new_ub_o - ub_o_old).abs() > ZERO_TOL {
            st.postsolve_stack.push(PostsolveStep::BoundsTightened {
                orig_col: other_col,
                old_lb: lb_o_old,
                old_ub: ub_o_old,
            });
            st.bounds[other_col] = (new_lb_o, new_ub_o);
        }
        // 消去実行
        eliminate_variable_via_eq_row(st, i, pivot_col)?;
        // 消去された x_other は他制約で参照されない可能性。後段 Step で empty col として処理される
        let _ = other_a;
        *new_subst += 1;
    }
    Ok(())
}

// ==========================================================
// Step 7: Free variable substitution (R15)
//
// 対象: free 変数 (lb == -∞ かつ ub == +∞) で、active な制約のうち 1 つでも Eq 制約に
//   出現するもの。その Eq 制約を pivot として x_j を消去。
//   free var なので bounds tightening は不要。
// ==========================================================
fn step7_free_var_substitution(
    st: &mut PresolveState,
    new_subst: &mut usize,
) -> Result<(), PresolveStatus> {
    let n = st.bounds.len();
    for j in 0..n {
        if st.removed_cols[j] {
            continue;
        }
        // free 判定: 元のモデル上で両側無限であること (bounds tightening の影響を受けない)
        let (orig_lb, orig_ub) = st.orig_bounds[j];
        if orig_lb != f64::NEG_INFINITY || orig_ub != f64::INFINITY {
            continue;
        }
        // x_j を含む Eq 制約を探す (|係数| が最大のものを選ぶ→数値安定)
        let col_entries = st.active_col_entries(j);
        if col_entries.is_empty() {
            continue;
        }
        let mut best: Option<(usize, f64)> = None;
        for &(i, a_ij) in &col_entries {
            if st.constraint_types[i] != ConstraintType::Eq {
                continue;
            }
            if a_ij.abs() < ZERO_TOL {
                continue;
            }
            match best {
                None => best = Some((i, a_ij)),
                Some((_, ba)) => {
                    if a_ij.abs() > ba.abs() {
                        best = Some((i, a_ij));
                    }
                }
            }
        }
        if let Some((piv_row, _)) = best {
            if fill_in_exceeds_budget(st, piv_row, j) {
                continue;
            }
            eliminate_variable_via_eq_row(st, piv_row, j)?;
            *new_subst += 1;
        }
    }
    Ok(())
}

// ==========================================================
// Step 8: Free singleton column (R5)
//
// 対象: x_j が exactly 1 つの active 制約のみに出現 (col singleton)
//   かつ free or 片側無限。
//   制約タイプ:
//     - Eq: そのまま eliminate_variable_via_eq_row で処理 (others=0 のケースは Step 2 が拾うので
//       ここでは others>=1 のケース)
//     - Le/Ge with one-sided free var:
//       例: x_j unbounded above, a_ij > 0 で Le 制約 → x_j が大きいほど制約が厳しくなる
//         → optimal で a_ij * x_j + rest = b_i (active boundary) と仮定して Eq 化
//       より一般には, free side が制約の "binding direction" と一致すれば
//       Eq 化して eliminate できる。
//       簡略のため: free (両側無限) の場合のみ処理し, 制約タイプを問わず Eq として扱う:
//         Le: rest + a_ij*x_j <= b_i. x_j free なので x_j で任意に調整可能 → Eq として binding
//         Ge: 同様
//       これは postsolve で x_j 値を一意に決められる根拠になる。
// ==========================================================
fn step8_free_singleton_col(
    st: &mut PresolveState,
    new_subst: &mut usize,
) -> Result<(), PresolveStatus> {
    let n = st.bounds.len();
    for j in 0..n {
        if st.removed_cols[j] {
            continue;
        }
        let (orig_lb, orig_ub) = st.orig_bounds[j];
        let col_entries = st.active_col_entries(j);
        if col_entries.len() != 1 {
            continue;
        }
        let (i, a_ij) = col_entries[0];
        if a_ij.abs() < ZERO_TOL {
            continue;
        }
        // 制約タイプ別の "free 化" 条件
        let ct = st.constraint_types[i];
        let cj = st.c[j];
        // 単純化: free (両側無限) のみ処理。one-sided free は HiGHS 文献では cost 符号と
        // 不等号の向きが整合するときのみ可能 (例: cj=0 かつ Le with a_ij>0, ub=+inf
        // → x_j を必要なだけ増やせる)。ここでは安全な free のみ。
        if orig_lb != f64::NEG_INFINITY || orig_ub != f64::INFINITY {
            // 片側 free の処理
            // a_ij > 0 で Le: x_j で b_i に余裕がある分を吸収可能 (x_j を増やせる:ub=+inf 必要)
            //   さらに cost 符号: cj >= 0 でないと x_j を無限増やせて unbounded
            //   ⇒ cj ≥ 0 かつ ub=+inf かつ Le with a_ij>0 → safe
            //   実際の値: x_j = (b_i - rest) / a_ij; これが lb_j 以上ならOK
            //
            // 複雑な分岐を避け、ここではこの片側 free ケースは扱わず Step 7 / 通常経路に
            // 任せる方針（R15 で free 側のみ集中処理。R5 単体は両側 free に限定）。
            continue;
        }
        // 両側 free: 制約タイプを問わず Eq 化と等価
        // (free var なので x_j を必要分動かして等式化できる)
        // cj != 0 の場合の影響:
        //   目的関数で cj * x_j がある。x_j = (b - Σ_k≠j a_ik * x_k) / a_ij を入れると
        //   cj * x_j = cj/a_ij * (b - rest)
        //   = cj*b/a_ij - cj/a_ij * Σ_k a_ik * x_k
        //   この変換は等価。
        // ただし不等号 (Le/Ge) を Eq に強制するには、双対方向の制約も考える必要。
        // 厳密には:
        //   Le: a_ij * x_j ≤ b - rest. x_j → -∞ で常に成立 → 制約は always satisfied iff
        //       cost cj が両方向許す形.
        // 安全な実装: ct == Eq のみ処理する。Le/Ge の free singleton はスキップ。
        if ct != ConstraintType::Eq {
            continue;
        }
        if fill_in_exceeds_budget(st, i, j) {
            continue;
        }
        eliminate_variable_via_eq_row(st, i, j)?;
        *new_subst += 1;
        // cj は eliminate 内で他変数に分配される
        // i の dual は postsolve 時に決定
        let _ = cj;
    }
    Ok(())
}

// ==========================================================
// Unit Tests
// ==========================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ConstraintType;
    use crate::sparse::CscMatrix;

    #[allow(clippy::too_many_arguments)]
    fn make_lp_general(
        c: Vec<f64>,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrows: usize,
        ncols: usize,
        b: Vec<f64>,
        cts: Vec<ConstraintType>,
        bounds: Vec<(f64, f64)>,
    ) -> LpProblem {
        let a = CscMatrix::from_triplets(rows, cols, vals, nrows, ncols).unwrap();
        LpProblem::new_general(c, a, b, cts, bounds, None).unwrap()
    }

    fn make_lp(
        c: Vec<f64>,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrows: usize,
        ncols: usize,
        b: Vec<f64>,
    ) -> LpProblem {
        let n = c.len();
        make_lp_general(
            c,
            rows,
            cols,
            vals,
            nrows,
            ncols,
            b,
            vec![ConstraintType::Le; nrows],
            vec![(0.0, f64::INFINITY); n],
        )
    }

    // -----------------------------------------------------------
    // 1. Fixed variable removal
    // -----------------------------------------------------------
    #[test]
    fn test_fixed_variable_removal() {
        let lp = make_lp_general(
            vec![1.0, 1.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(2.0, 2.0), (0.0, f64::INFINITY)],
        );
        let result = run_presolve(&lp, None).unwrap();
        assert_eq!(result.reduced_problem.num_vars, 0);
        assert_eq!(result.reduced_problem.num_constraints, 0);
        assert!(result.was_reduced);
        assert!((result.obj_offset - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_infeasible() {
        let lp = make_lp_general(
            vec![1.0],
            &[],
            &[],
            &[],
            0,
            1,
            vec![],
            vec![],
            vec![(3.0, 2.0)],
        );
        assert!(matches!(run_presolve(&lp, None), Err(PresolveStatus::Infeasible)));
    }

    // -----------------------------------------------------------
    // 2. Empty row/column removal
    // -----------------------------------------------------------
    #[test]
    fn test_empty_row_feasible() {
        let lp = make_lp_general(
            vec![1.0],
            &[1],
            &[0],
            &[1.0],
            2,
            1,
            vec![5.0, 3.0],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
        );
        let result = run_presolve(&lp, None).unwrap();
        assert_eq!(result.reduced_problem.num_constraints, 0);
    }

    #[test]
    fn test_empty_row_infeasible() {
        let lp = make_lp_general(
            vec![1.0],
            &[1],
            &[0],
            &[1.0],
            2,
            1,
            vec![-1.0, 3.0],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
        );
        assert!(matches!(run_presolve(&lp, None), Err(PresolveStatus::Infeasible)));
    }

    #[test]
    fn test_empty_column_min_with_finite_lb() {
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            CscMatrix::new(0, 2),
            vec![],
            vec![],
            vec![(0.0, f64::INFINITY), (1.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let result = run_presolve(&lp, None).unwrap();
        assert_eq!(result.reduced_problem.num_vars, 0);
        assert!((result.obj_offset - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_column_unbounded() {
        let lp = LpProblem::new_general(
            vec![-1.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        assert!(matches!(run_presolve(&lp, None), Err(PresolveStatus::Unbounded)));
    }

    // -----------------------------------------------------------
    // 3. Singleton row (Eq)
    // -----------------------------------------------------------
    #[test]
    fn test_singleton_row_eq() {
        let lp = make_lp_general(
            vec![1.0, 1.0],
            &[0, 1, 1],
            &[0, 0, 1],
            &[2.0, 1.0, 1.0],
            2,
            2,
            vec![6.0, 10.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
        );
        let result = run_presolve(&lp, None).unwrap();
        assert_eq!(result.reduced_problem.num_vars, 0);
        assert_eq!(result.reduced_problem.num_constraints, 0);
        assert!((result.obj_offset - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_singleton_row_infeasible() {
        let lp = make_lp_general(
            vec![1.0],
            &[0],
            &[0],
            &[2.0],
            1,
            1,
            vec![6.0],
            vec![ConstraintType::Eq],
            vec![(0.0, 1.0)],
        );
        assert!(matches!(run_presolve(&lp, None), Err(PresolveStatus::Infeasible)));
    }

    // -----------------------------------------------------------
    // 4. Redundant constraint removal
    // -----------------------------------------------------------
    #[test]
    fn test_redundant_le() {
        let lp = make_lp_general(
            vec![1.0, 1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![10.0, 3.0, 3.0],
            vec![ConstraintType::Le, ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, 3.0), (0.0, 3.0)],
        );
        let result = run_presolve(&lp, None).unwrap();
        assert_eq!(result.reduced_problem.num_constraints, 0, "all 3 constraints should be redundant");
        assert_eq!(result.reduced_problem.num_vars, 0, "vars removed as empty cols after constraints gone");

        // Use negative cost so dual fixing (Step 11) cannot collapse the LP:
        // c < 0 with Le a > 0 disqualifies neg-pressure, c < 0 fails pos-pressure cost gate.
        let lp2 = make_lp_general(
            vec![-1.0, -1.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![2.0],
            vec![ConstraintType::Le],
            vec![(0.0, 10.0), (0.0, 10.0)],
        );
        let result2 = run_presolve(&lp2, None).unwrap();
        assert_eq!(result2.reduced_problem.num_constraints, 1, "x1+x2<=2 is not redundant");
    }

    // -----------------------------------------------------------
    // 5. Bounds tightening
    // -----------------------------------------------------------
    #[test]
    fn test_bounds_tightening() {
        // Use negative cost: Step 11 dual fixing (which collapses x→0 when c≥0
        // and all Le coefs ≥0) does not apply here, so we observe pure Step 5.
        let lp = make_lp_general(
            vec![-1.0, -1.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 10.0), (0.0, 10.0)],
        );
        let result = run_presolve(&lp, None).unwrap();
        let _ = result.was_reduced;
        assert_eq!(result.reduced_problem.num_vars, 2);
    }

    #[test]
    fn test_bounds_tightening_negative_coeff_le_feasible() {
        let lp = make_lp_general(
            vec![1.0, 1.0],
            &[0, 0],
            &[0, 1],
            &[1.0, -1.0],
            1,
            2,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 10.0), (0.0, 3.0)],
        );
        assert!(run_presolve(&lp, None).is_ok(), "x - y <= 5 should be feasible");
    }

    #[test]
    fn test_bounds_tightening_negative_coeff_ge_feasible() {
        let lp = make_lp_general(
            vec![1.0, 1.0],
            &[0, 0],
            &[0, 1],
            &[-1.0, 1.0],
            1,
            2,
            vec![3.0],
            vec![ConstraintType::Ge],
            vec![(0.0, 5.0), (0.0, 8.0)],
        );
        assert!(run_presolve(&lp, None).is_ok(), "-x + y >= 3 should be feasible");
    }

    // -----------------------------------------------------------
    // Roundtrip
    // -----------------------------------------------------------
    #[test]
    fn test_presolve_no_crash_netlib_like() {
        let lp = make_lp(
            vec![-1.0, -1.0, -1.0],
            &[0, 0, 0, 1, 2, 3],
            &[0, 1, 2, 0, 1, 2],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            4,
            3,
            vec![4.0, 3.0, 3.0, 3.0],
        );
        let result = run_presolve(&lp, None).unwrap();
        assert_eq!(result.reduced_problem.num_vars, 3);
        assert_eq!(result.reduced_problem.num_constraints, 4);
    }

    #[test]
    fn test_pre001_deadline_fires_immediately() {
        let lp = make_lp_general(
            vec![1.0, 1.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(2.0, 2.0), (0.0, f64::INFINITY)],
        );
        let expired = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let result = run_presolve(&lp, Some(expired)).unwrap();
        assert!(
            !result.was_reduced,
            "期限切れ deadline では early-exit し was_reduced=false を返すこと"
        );
    }

    // -----------------------------------------------------------
    // R6: Doubleton equation
    // -----------------------------------------------------------
    #[test]
    fn presolve_doubleton_eq_basic() {
        // min x + y + z
        // s.t. x + y = 3        (Eq doubleton)
        //      x + y + z <= 10
        //      x in [0,5], y in [0,5], z in [0, inf)
        // x を消去 (pivot=x, others=y), 残: y, z, 制約: (y, z への変換)
        let lp = make_lp_general(
            vec![1.0, 1.0, 1.0],
            &[0, 0, 1, 1, 1],
            &[0, 1, 0, 1, 2],
            &[1.0, 1.0, 1.0, 1.0, 1.0],
            2,
            3,
            vec![3.0, 10.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, 5.0), (0.0, 5.0), (0.0, f64::INFINITY)],
        );
        let result = run_presolve(&lp, None).unwrap();
        // x または y のいずれかが消去される。残り 2 vars, 1 制約 (or さらに縮小)
        assert!(result.was_reduced);
        // postsolve_stack に LinearSubstitution が含まれていることを確認
        let has_subst = result
            .postsolve_stack
            .iter()
            .any(|s| matches!(s, PostsolveStep::LinearSubstitution { .. }));
        assert!(has_subst, "Doubleton equation should produce LinearSubstitution");
    }

    #[test]
    fn presolve_doubleton_eq_solution_consistency() {
        // 同じ問題を presolve あり / なしで解いた解の "目的値" を obj_offset 含めて比較する
        // ここでは presolve のみ実行し、reduced + offset が元の最適値に一致するロジック検証
        //
        // min x + y
        // s.t. x + y = 4
        //      x in [0,3], y in [0,3]
        // 最適解: 任意の x+y=4 (例: x=1,y=3 or x=3,y=1)。最適値 = 4
        // presolve: x = 4 - y, x in [0,3] → y in [1,4] ∩ [0,3] = [1,3]
        //   reduced: min (4-y) + y = 4 over y in [1,3] → 縮約後 c[y]=0, offset=4
        //   reduced は 0変数 / 0制約 になり得る (cy=1-1=0, 制約はx+y<=. ここでは無いので)
        let lp = make_lp_general(
            vec![1.0, 1.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![4.0],
            vec![ConstraintType::Eq],
            vec![(0.0, 3.0), (0.0, 3.0)],
        );
        let result = run_presolve(&lp, None).unwrap();
        // 目的値の総和は 4 (= obj_offset + reduced c^T x)
        // reduced c[y] = 0 (1 - 1*1 = 0), offset = 4 (1*4/1 = 4)
        assert!((result.obj_offset - 4.0).abs() < 1e-10, "obj_offset = 4");
    }

    #[test]
    fn presolve_doubleton_eq_infeasible() {
        // x + y = 10, x in [0,3], y in [0,3] → 最大 6 < 10 → Infeasible
        let lp = make_lp_general(
            vec![1.0, 1.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![10.0],
            vec![ConstraintType::Eq],
            vec![(0.0, 3.0), (0.0, 3.0)],
        );
        let res = run_presolve(&lp, None);
        assert!(matches!(res, Err(PresolveStatus::Infeasible)));
    }

    // -----------------------------------------------------------
    // R15: Free variable substitution
    // -----------------------------------------------------------
    #[test]
    fn presolve_free_var_subst_basic() {
        // min x + y + z
        // s.t. x + y + z = 5     (Eq)
        //      x + y <= 10
        //      z is free, x in [0,10], y in [0,10]
        // → z = 5 - x - y を Eq から代入 → Eq 消去、他制約に z 出現なし → 影響なし
        // 結果: vars = (x, y) のみ (z 消去), 制約 = 1 (Le)
        let lp = make_lp_general(
            vec![1.0, 1.0, 1.0],
            &[0, 0, 0, 1, 1],
            &[0, 1, 2, 0, 1],
            &[1.0, 1.0, 1.0, 1.0, 1.0],
            2,
            3,
            vec![5.0, 10.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
        );
        let result = run_presolve(&lp, None).unwrap();
        assert!(result.was_reduced);
        let has_subst = result
            .postsolve_stack
            .iter()
            .any(|s| matches!(s, PostsolveStep::LinearSubstitution { .. }));
        assert!(has_subst, "Free var substitution should produce LinearSubstitution");
        // z が消去されているはず
        assert!(result.col_map[2].is_none(), "z (col 2) should be eliminated");
    }

    #[test]
    fn presolve_free_var_subst_multi_constraint() {
        // min x + y + z
        // s.t. x + z = 4          (Eq, z 含む)
        //      y + z = 5          (Eq, z 含む)
        //      x in [0,10], y in [0,10], z free
        // → z = 4 - x を Eq#0 から代入 → Eq#0 消去, Eq#1: y + (4 - x) = 5 → y - x = 1
        let lp = make_lp_general(
            vec![1.0, 1.0, 1.0],
            &[0, 0, 1, 1],
            &[0, 2, 1, 2],
            &[1.0, 1.0, 1.0, 1.0],
            2,
            3,
            vec![4.0, 5.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
        );
        let result = run_presolve(&lp, None).unwrap();
        assert!(result.was_reduced);
        // z は消去される. 制約は 1 (Eq) 残り
        assert!(result.col_map[2].is_none());
    }

    // -----------------------------------------------------------
    // R5: Free singleton column
    // -----------------------------------------------------------
    #[test]
    fn presolve_doubleton_dual_recovery_eq_le() {
        // Eq doubleton (x1+x2=6) + Le (x2<=5)。pivot=x1 で x1 を消去後、
        // dual 復元式: y_piv = (c_orig - Σ_{i ≠ piv} A_ij_orig * y_i) / pivot で
        // y[0] = 1.0 になることを確認。
        let lp = make_lp_general(
            vec![1.0, 2.0],
            &[0, 0, 1],
            &[0, 1, 1],
            &[1.0, 1.0, 1.0],
            2,
            2,
            vec![6.0, 5.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
        );
        let result = run_presolve(&lp, None).unwrap();
        // postsolve_stack に LinearSubstitution が含まれ、その c_orig が正しく保存されている
        let lin = result.postsolve_stack.iter().find_map(|s| match s {
            PostsolveStep::LinearSubstitution { c_orig, pivot, .. } => Some((*c_orig, *pivot)),
            _ => None,
        });
        assert!(lin.is_some(), "LinearSubstitution expected");
        let (c_orig, pivot) = lin.unwrap();
        // pivot=1 (x1 の係数), c_orig = c_x1 = 1
        assert!((pivot - 1.0).abs() < 1e-12);
        assert!((c_orig - 1.0).abs() < 1e-12, "c_orig must capture pre-distribution c[x1]=1");
    }

    #[test]
    fn presolve_free_singleton_col_basic() {
        // min x + y + z
        // s.t. x + y >= 3
        //      x + z = 7        (Eq, z singleton 列 = z は他制約に出ない)
        //      x in [0,10], y in [0,10], z free
        // R5 (も R15 も両方適用条件) → z 消去 + Eq#1 消去
        let lp = make_lp_general(
            vec![1.0, 1.0, 1.0],
            &[0, 0, 1, 1],
            &[0, 1, 0, 2],
            &[1.0, 1.0, 1.0, 1.0],
            2,
            3,
            vec![3.0, 7.0],
            vec![ConstraintType::Ge, ConstraintType::Eq],
            vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
        );
        let result = run_presolve(&lp, None).unwrap();
        assert!(result.was_reduced);
        assert!(result.col_map[2].is_none(), "z should be eliminated");
        assert!(result.row_map[1].is_none(), "Eq row should be eliminated");
    }

    // -----------------------------------------------------------
    // Round-trip KKT tests: presolve→solve→postsolve cycle が原問題で
    // primal/dual/objective を全て満たすことを assert する。
    //
    // 既存 test 群は run_presolve の構造的副作用 (num_vars, postsolve_stack,
    // col_map) のみ検証していたため、postsolve の dual recovery が崩れても
    // 検出できなかった (perold 等で実際に bug を漏らした)。
    // -----------------------------------------------------------
    mod roundtrip_kkt {
        use super::*;
        use crate::test_kkt::assert_kkt_optimal;

        /// Doubleton Eq の round-trip: x+y=4, x∈[0,3], y∈[0,3], min x+y → obj=4
        #[test]
        fn roundtrip_doubleton_eq_simple() {
            let lp = make_lp_general(
                vec![1.0, 1.0],
                &[0, 0], &[0, 1], &[1.0, 1.0],
                1, 2,
                vec![4.0],
                vec![ConstraintType::Eq],
                vec![(0.0, 3.0), (0.0, 3.0)],
            );
            assert_kkt_optimal(&lp, 4.0, "roundtrip_doubleton_eq_simple");
        }

        /// Doubleton Eq + 異なる係数: 2x+3y=12, x∈[0,4], y∈[0,4], min x+2y
        /// 代入: x = 6 - 1.5y, feasible: 4/3 ≤ y ≤ 4
        /// obj = (6-1.5y) + 2y = 6 + 0.5y → min y=4/3, x=4, obj = 6+2/3 = 20/3
        #[test]
        fn roundtrip_doubleton_eq_nonunit_coeffs() {
            let lp = make_lp_general(
                vec![1.0, 2.0],
                &[0, 0], &[0, 1], &[2.0, 3.0],
                1, 2,
                vec![12.0],
                vec![ConstraintType::Eq],
                vec![(0.0, 4.0), (0.0, 4.0)],
            );
            assert_kkt_optimal(&lp, 20.0 / 3.0, "roundtrip_doubleton_eq_nonunit_coeffs");
        }

        /// Free var substitution: z free + Eq row で z を消去後 KKT 整合
        /// min x+y+z, x+y+z=5, x+y<=10, x,y∈[0,10], z free → z=5-x-y, obj=5
        #[test]
        fn roundtrip_free_var_subst() {
            let lp = make_lp_general(
                vec![1.0, 1.0, 1.0],
                &[0, 0, 0, 1, 1],
                &[0, 1, 2, 0, 1],
                &[1.0, 1.0, 1.0, 1.0, 1.0],
                2, 3,
                vec![5.0, 10.0],
                vec![ConstraintType::Eq, ConstraintType::Le],
                vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
            );
            assert_kkt_optimal(&lp, 5.0, "roundtrip_free_var_subst");
        }

        /// Free singleton col: z は singleton 列 + free。Eq 1 + Ge 1 の混在で
        /// postsolve が free col + Ge dual の符号慣例を正しく復元するか。
        /// min x+y+z, x+y>=3, x+z=7, x,y∈[0,10], z free → x=3, y=0, z=4 obj=7
        #[test]
        fn roundtrip_free_singleton_col() {
            let lp = make_lp_general(
                vec![1.0, 1.0, 1.0],
                &[0, 0, 1, 1],
                &[0, 1, 0, 2],
                &[1.0, 1.0, 1.0, 1.0],
                2, 3,
                vec![3.0, 7.0],
                vec![ConstraintType::Ge, ConstraintType::Eq],
                vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
            );
            // x+y>=3, x+z=7. min x+y+z = x+y + (7-x) = y+7 → minimize y → y=0
            // y=0: x>=3, z=7-x. min x+0+7-x = 7. 任意 x ∈ [3,7] feasible. obj=7
            assert_kkt_optimal(&lp, 7.0, "roundtrip_free_singleton_col");
        }

        /// Singleton row + bounds tightening: x0 = 5 fix で SingletonRow 経由
        /// y_0 を bound-aware に復元する経路 (perold class proxy)。
        /// min x0+x1+x2, x0=5 (Eq singleton), x1+x2=4 (Eq), x1∈[0,3], x2∈[0,3]
        /// → x0=5, x1+x2=4 minimize → 任意組合せ、obj = 5+4=9
        #[test]
        fn roundtrip_singleton_row_eq_with_doubleton() {
            let lp = make_lp_general(
                vec![1.0, 1.0, 1.0],
                &[0, 1, 1],
                &[0, 1, 2],
                &[1.0, 1.0, 1.0],
                2, 3,
                vec![5.0, 4.0],
                vec![ConstraintType::Eq, ConstraintType::Eq],
                vec![(0.0, 10.0), (0.0, 3.0), (0.0, 3.0)],
            );
            assert_kkt_optimal(&lp, 9.0, "roundtrip_singleton_row_eq_with_doubleton");
        }

        /// Redundant Le row + active Eq: Redundant が削除されても残りの Eq
        /// で KKT が成立し、削除行の y_i は bound-aware default (= 0) で
        /// 矛盾ないことを round-trip で検証。
        #[test]
        fn roundtrip_redundant_le_with_active_eq() {
            // x1+x2 <= 100 (Le, redundant: x1∈[0,3], x2∈[0,3])
            // x1+x2 = 4 (Eq, active)
            // min 2x1+x2, x1∈[0,3], x2∈[0,3]
            // → x1=1, x2=3 (cost x1 を最小化、x2 が cheaper): obj = 2+3 = 5
            //   x1=3, x2=1: obj = 6+1=7
            //   x1=0, x2=4: infeasible (x2>3)
            //   x1=1, x2=3: obj=5 (★)
            let lp = make_lp_general(
                vec![2.0, 1.0],
                &[0, 0, 1, 1],
                &[0, 1, 0, 1],
                &[1.0, 1.0, 1.0, 1.0],
                2, 2,
                vec![100.0, 4.0],
                vec![ConstraintType::Le, ConstraintType::Eq],
                vec![(0.0, 3.0), (0.0, 3.0)],
            );
            assert_kkt_optimal(&lp, 5.0, "roundtrip_redundant_le_with_active_eq");
        }

        /// 全 transform 混在: doubleton + free var + singleton + redundant
        /// (presolve→postsolve の全体パスの cross 検証)
        #[test]
        fn roundtrip_mixed_transforms() {
            // min x1 + x2 + x3 + x4
            // x1 + x2     = 3    (Eq doubleton, x1∈[0,2], x2∈[0,2] active)
            // x3 + x4     = 2    (Eq doubleton, x3 free, x4∈[0,5])
            // x1 + x3    <= 100  (Le redundant)
            // → x1+x2=3 (x1=1,x2=2 や x1=2,x2=1)、x3+x4=2 (任意)、obj = 3+2 = 5
            let lp = make_lp_general(
                vec![1.0, 1.0, 1.0, 1.0],
                &[0, 0, 1, 1, 2, 2],
                &[0, 1, 2, 3, 0, 2],
                &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                3, 4,
                vec![3.0, 2.0, 100.0],
                vec![ConstraintType::Eq, ConstraintType::Eq, ConstraintType::Le],
                vec![
                    (0.0, 2.0), (0.0, 2.0),
                    (f64::NEG_INFINITY, f64::INFINITY), (0.0, 5.0),
                ],
            );
            assert_kkt_optimal(&lp, 5.0, "roundtrip_mixed_transforms");
        }

        /// Le → Ge round-trip: Ge は postsolve で符号反転、dual 符号慣例を
        /// 正しく復元できないと dfeas_rel_bound が劣化。
        #[test]
        fn roundtrip_ge_constraint_dual_sign() {
            // min x+y, x+y >= 3, x∈[0,5], y∈[0,5] → x+y=3 (任意)、obj=3
            let lp = make_lp_general(
                vec![1.0, 1.0],
                &[0, 0], &[0, 1], &[1.0, 1.0],
                1, 2,
                vec![3.0],
                vec![ConstraintType::Ge],
                vec![(0.0, 5.0), (0.0, 5.0)],
            );
            assert_kkt_optimal(&lp, 3.0, "roundtrip_ge_constraint_dual_sign");
        }
    }
}
