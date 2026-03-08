//! Presolve変換モジュール
//!
//! LPを縮約するための5種の手法と逆変換（Postsolve）情報を提供する。
//!
//! 適用順序:
//! 1. Fixed variable removal (lb == ub)
//! 2. Singleton row (Eq制約、変数1つ)
//! 3. Empty row/column removal
//! 4. Redundant constraint removal
//! 5. Bounds tightening

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
    SingletonRow {
        orig_row: usize,
        orig_col: usize,
        value: f64,
    },
    /// Redundant constraint removal の逆変換 (常に満たされる制約)
    RedundantConstraint { orig_row: usize },
    /// Bounds tightening の逆変換 (解の復元には直接影響しない)
    BoundsTightened {
        orig_col: usize,
        old_lb: f64,
        old_ub: f64,
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

/// 行の活動範囲 [row_lb, row_ub] を計算する。
///
/// entries: 行のエントリ (col, val) のスライス（削除済み列を除外済み想定）
/// bounds: 現在の変数bounds
/// exclude_col: 除外する列（bounds tightening でその変数を除く際に使用）
///
/// 戻り値: (lb, ub, lb_is_finite, ub_is_finite)
fn activity_range(
    entries: &[(usize, f64)],
    bounds: &[(f64, f64)],
    exclude_col: Option<usize>,
) -> (f64, f64, bool, bool) {
    let mut row_lb = 0.0f64;
    let mut row_ub = 0.0f64;
    let mut lb_finite = true;
    let mut ub_finite = true;

    for &(j, a_ij) in entries {
        if Some(j) == exclude_col {
            continue;
        }
        let (lb_j, ub_j) = bounds[j];
        if a_ij > 0.0 {
            if lb_j == f64::NEG_INFINITY {
                lb_finite = false;
            } else if lb_finite {
                row_lb += a_ij * lb_j;
            }
            if ub_j == f64::INFINITY {
                ub_finite = false;
            } else if ub_finite {
                row_ub += a_ij * ub_j;
            }
        } else if a_ij < 0.0 {
            if ub_j == f64::INFINITY {
                lb_finite = false;
            } else if lb_finite {
                row_lb += a_ij * ub_j;
            }
            if lb_j == f64::NEG_INFINITY {
                ub_finite = false;
            } else if ub_finite {
                row_ub += a_ij * lb_j;
            }
        }
    }
    (row_lb, row_ub, lb_finite, ub_finite)
}

/// LPをPresolveして縮約問題を返す。
///
/// 問題が明らかにInfeasible/Unboundedな場合はErrを返す。
pub fn run_presolve(problem: &LpProblem) -> Result<PresolveResult, PresolveStatus> {
    let n = problem.num_vars;
    let m = problem.num_constraints;

    // 行情報の構築: row_entries[i] = Vec<(col, val)>
    // CscMatrixは列アクセスのみ効率的なので、O(nnz)で行情報を前処理する
    let mut row_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; m];
    for j in 0..n {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                row_entries[row].push((j, vals[k]));
            }
        }
    }

    let mut removed_cols = vec![false; n];
    let mut removed_rows = vec![false; m];
    let mut b = problem.b.clone();
    let mut bounds = problem.bounds.clone();
    let mut postsolve_stack: Vec<PostsolveStep> = vec![];
    let mut obj_offset = 0.0f64;

    // ==========================================================
    // Step 1: Fixed variable removal (lb == ub)
    // ==========================================================
    for j in 0..n {
        if removed_cols[j] {
            continue;
        }
        let (lb, ub) = bounds[j];
        if lb > ub + ZERO_TOL {
            return Err(PresolveStatus::Infeasible);
        }
        if (lb - ub).abs() < ZERO_TOL {
            let value = lb;
            // b[i] -= A[i,j] * value for all active rows
            if let Ok((rows, vals)) = problem.a.get_column(j) {
                for (k, &row) in rows.iter().enumerate() {
                    if !removed_rows[row] {
                        b[row] -= vals[k] * value;
                    }
                }
            }
            obj_offset += problem.c[j] * value;
            removed_cols[j] = true;
            postsolve_stack.push(PostsolveStep::FixedVariable { orig_col: j, value });
        }
    }

    // ==========================================================
    // Step 2: Singleton row (Eq制約、変数1つ → 値確定)
    // ==========================================================
    for i in 0..m {
        if removed_rows[i] {
            continue;
        }
        if problem.constraint_types[i] != ConstraintType::Eq {
            continue;
        }
        let active: Vec<(usize, f64)> = row_entries[i]
            .iter()
            .filter(|&&(j, _)| !removed_cols[j])
            .copied()
            .collect();
        if active.len() == 1 {
            let (j, a_ij) = active[0];
            if a_ij.abs() < ZERO_TOL {
                continue;
            }
            let value = b[i] / a_ij;
            let (lb, ub) = bounds[j];
            if value < lb - ZERO_TOL || value > ub + ZERO_TOL {
                return Err(PresolveStatus::Infeasible);
            }
            let value = value.clamp(lb, ub);
            // 変数jを含む他の行のbを更新
            if let Ok((rows, vals)) = problem.a.get_column(j) {
                for (k, &row) in rows.iter().enumerate() {
                    if !removed_rows[row] && row != i {
                        b[row] -= vals[k] * value;
                    }
                }
            }
            obj_offset += problem.c[j] * value;
            removed_cols[j] = true;
            removed_rows[i] = true;
            postsolve_stack.push(PostsolveStep::SingletonRow {
                orig_row: i,
                orig_col: j,
                value,
            });
        }
    }

    // ==========================================================
    // Step 3a: Empty row removal (全係数ゼロ行)
    // ==========================================================
    for i in 0..m {
        if removed_rows[i] {
            continue;
        }
        let active_count = row_entries[i]
            .iter()
            .filter(|&&(j, _)| !removed_cols[j])
            .count();
        if active_count == 0 {
            // 制約が 0 <= b[i] / 0 >= b[i] / 0 == b[i] に退化
            match problem.constraint_types[i] {
                ConstraintType::Eq => {
                    if b[i].abs() > ZERO_TOL {
                        return Err(PresolveStatus::Infeasible);
                    }
                }
                ConstraintType::Le => {
                    if b[i] < -ZERO_TOL {
                        return Err(PresolveStatus::Infeasible);
                    }
                }
                ConstraintType::Ge => {
                    if b[i] > ZERO_TOL {
                        return Err(PresolveStatus::Infeasible);
                    }
                }
            }
            removed_rows[i] = true;
            postsolve_stack.push(PostsolveStep::EmptyRow { orig_row: i });
        }
    }

    // ==========================================================
    // Step 3b: Empty column removal (全係数ゼロ列)
    // ==========================================================
    for j in 0..n {
        if removed_cols[j] {
            continue;
        }
        let active_count = if let Ok((rows, _)) = problem.a.get_column(j) {
            rows.iter().filter(|&&r| !removed_rows[r]).count()
        } else {
            0
        };
        if active_count == 0 {
            let (lb, ub) = bounds[j];
            let cj = problem.c[j];
            let value = if cj > ZERO_TOL {
                // minimize: 小さい方が最適 → lb
                if lb == f64::NEG_INFINITY {
                    return Err(PresolveStatus::Unbounded);
                }
                if lb.is_finite() { lb } else { 0.0 }
            } else if cj < -ZERO_TOL {
                // minimize c*x where c<0: 大きい方が最適 → ub
                if ub == f64::INFINITY {
                    return Err(PresolveStatus::Unbounded);
                }
                if ub.is_finite() { ub } else { 0.0 }
            } else {
                // c=0: 任意の実行可能値
                if lb.is_finite() {
                    lb
                } else if ub.is_finite() {
                    ub
                } else {
                    0.0
                }
            };
            obj_offset += cj * value;
            removed_cols[j] = true;
            postsolve_stack.push(PostsolveStep::EmptyColumn { orig_col: j, value });
        }
    }

    // ==========================================================
    // Step 4: Redundant constraint removal
    // ==========================================================
    for i in 0..m {
        if removed_rows[i] {
            continue;
        }
        let active_entries: Vec<(usize, f64)> = row_entries[i]
            .iter()
            .filter(|&&(j, _)| !removed_cols[j])
            .copied()
            .collect();
        let (row_lb, row_ub, lb_fin, ub_fin) =
            activity_range(&active_entries, &bounds, None);

        let redundant = match problem.constraint_types[i] {
            ConstraintType::Le => ub_fin && row_ub <= b[i] + ZERO_TOL,
            ConstraintType::Ge => lb_fin && row_lb >= b[i] - ZERO_TOL,
            ConstraintType::Eq => {
                lb_fin && ub_fin
                    && row_lb >= b[i] - ZERO_TOL
                    && row_ub <= b[i] + ZERO_TOL
            }
        };
        if redundant {
            removed_rows[i] = true;
            postsolve_stack.push(PostsolveStep::RedundantConstraint { orig_row: i });
        }
    }

    // ==========================================================
    // Step 5: Bounds tightening (1パス, O(m×n) 増分更新)
    //
    // 符号による導出:
    //   Le: a_j*x_j + rest <= b
    //     a_j > 0: x_j <= (b - rest_lb) / a_j  [rest >= rest_lb なので b-rest <= b-rest_lb]
    //     a_j < 0: x_j >= (b - rest_lb) / a_j  [存在条件: rest_lb <= b - a_j*x_j]
    //   Ge: a_j*x_j + rest >= b
    //     a_j > 0: x_j >= (b - rest_ub) / a_j  [rest <= rest_ub なので b-rest >= b-rest_ub]
    //     a_j < 0: x_j <= (b - rest_ub) / a_j  [存在条件: rest_ub >= b - a_j*x_j]
    //
    // 最適化: 行全体の activity_range を O(n) で1回計算し、
    //   変数j除外時は O(1) 増分減算で rest を得る。
    //   旧実装は各変数でO(n)全走査 → O(m×n²)。BOYD1で127s超の原因。
    // ==========================================================
    for i in 0..m {
        if removed_rows[i] {
            continue;
        }
        let ct = problem.constraint_types[i];
        let entries: Vec<(usize, f64)> = row_entries[i]
            .iter()
            .filter(|&&(j, _)| !removed_cols[j])
            .copied()
            .collect();

        if entries.is_empty() {
            continue;
        }

        // --- Phase A: 行全体の activity_range を O(n) で事前計算 ---
        // PARAM: 各エントリのlb/ub寄与を記録し、除外時にO(1)で引けるようにする
        let mut row_lb_sum = 0.0f64;  // 有限なlb寄与の合計
        let mut row_ub_sum = 0.0f64;  // 有限なub寄与の合計
        let mut inf_lb_count = 0usize; // lb無限大にする寄与の数
        let mut inf_ub_count = 0usize; // ub無限大にする寄与の数
        // 各エントリが lb/ub に与える寄与（有限分のみ）と無限大フラグ
        let mut entry_lb_contrib = Vec::with_capacity(entries.len());
        let mut entry_ub_contrib = Vec::with_capacity(entries.len());
        let mut entry_lb_inf = Vec::with_capacity(entries.len());
        let mut entry_ub_inf = Vec::with_capacity(entries.len());

        for &(j, a_ij) in &entries {
            let (lb_j, ub_j) = bounds[j];
            if a_ij > 0.0 {
                // lb への寄与: a_ij * lb_j (lb_j が -∞ なら無限大)
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
                // ub への寄与: a_ij * ub_j (ub_j が +∞ なら無限大)
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
                // lb への寄与: a_ij * ub_j (負係数 × 上界 → 下界方向)
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
                // ub への寄与: a_ij * lb_j (負係数 × 下界 → 上界方向)
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
                // a_ij == 0: 寄与なし
                entry_lb_inf.push(false);
                entry_ub_inf.push(false);
                entry_lb_contrib.push(0.0);
                entry_ub_contrib.push(0.0);
            }
        }

        // --- Phase B: 各変数について O(1) でrestを計算 ---
        for (k, &(j, a_ij)) in entries.iter().enumerate() {
            if a_ij.abs() < ZERO_TOL {
                continue;
            }
            let (old_lb, old_ub) = bounds[j];

            // rest = 行全体 - j の寄与 (O(1))
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
                    // Σ a_ij*xj <= b[i]
                    if a_ij > 0.0 && rest_lb_fin {
                        // x_j <= (b - rest_lb) / a_j  (rest >= rest_lb なので b-rest <= b-rest_lb)
                        let implied_ub = (b[i] - rest_lb) / a_ij;
                        if implied_ub < old_lb - ZERO_TOL {
                            return Err(PresolveStatus::Infeasible);
                        }
                        if implied_ub < new_ub - ZERO_TOL {
                            new_ub = implied_ub;
                        }
                    } else if a_ij < 0.0 && rest_lb_fin {
                        // x_j >= (b - rest_lb) / a_j  (存在条件: rest_lb <= b - a_j*x_j)
                        let implied_lb = (b[i] - rest_lb) / a_ij;
                        if implied_lb > old_ub + ZERO_TOL {
                            return Err(PresolveStatus::Infeasible);
                        }
                        if implied_lb > new_lb + ZERO_TOL {
                            new_lb = implied_lb;
                        }
                    }
                }
                ConstraintType::Ge => {
                    // Σ a_ij*xj >= b[i]
                    if a_ij > 0.0 && rest_ub_fin {
                        // x_j >= (b - rest_ub) / a_j  (rest <= rest_ub なので b-rest >= b-rest_ub)
                        let implied_lb = (b[i] - rest_ub) / a_ij;
                        if implied_lb > old_ub + ZERO_TOL {
                            return Err(PresolveStatus::Infeasible);
                        }
                        if implied_lb > new_lb + ZERO_TOL {
                            new_lb = implied_lb;
                        }
                    } else if a_ij < 0.0 && rest_ub_fin {
                        // x_j <= (b - rest_ub) / a_j  (存在条件: rest_ub >= b - a_j*x_j)
                        let implied_ub = (b[i] - rest_ub) / a_ij;
                        if implied_ub < old_lb - ZERO_TOL {
                            return Err(PresolveStatus::Infeasible);
                        }
                        if implied_ub < new_ub - ZERO_TOL {
                            new_ub = implied_ub;
                        }
                    }
                }
                ConstraintType::Eq => {
                    // Eq: Le + Ge の両方を適用
                    if a_ij > 0.0 {
                        // Le方向: x_j <= (b - rest_lb) / a_j
                        if rest_lb_fin {
                            let implied_ub = (b[i] - rest_lb) / a_ij;
                            if implied_ub < old_lb - ZERO_TOL {
                                return Err(PresolveStatus::Infeasible);
                            }
                            if implied_ub < new_ub - ZERO_TOL {
                                new_ub = implied_ub;
                            }
                        }
                        // Ge方向: x_j >= (b - rest_ub) / a_j
                        if rest_ub_fin {
                            let implied_lb = (b[i] - rest_ub) / a_ij;
                            if implied_lb > old_ub + ZERO_TOL {
                                return Err(PresolveStatus::Infeasible);
                            }
                            if implied_lb > new_lb + ZERO_TOL {
                                new_lb = implied_lb;
                            }
                        }
                    } else {
                        // a_ij < 0: 不等号の向きが逆転
                        // Le方向: x_j >= (b - rest_lb) / a_j  (存在条件)
                        if rest_lb_fin {
                            let implied_lb = (b[i] - rest_lb) / a_ij;
                            if implied_lb > old_ub + ZERO_TOL {
                                return Err(PresolveStatus::Infeasible);
                            }
                            if implied_lb > new_lb + ZERO_TOL {
                                new_lb = implied_lb;
                            }
                        }
                        // Ge方向: x_j <= (b - rest_ub) / a_j  (存在条件)
                        if rest_ub_fin {
                            let implied_ub = (b[i] - rest_ub) / a_ij;
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

            // bounds が実際に変化した場合のみ記録
            if (new_lb - old_lb).abs() > ZERO_TOL || (new_ub - old_ub).abs() > ZERO_TOL {
                postsolve_stack.push(PostsolveStep::BoundsTightened {
                    orig_col: j,
                    old_lb,
                    old_ub,
                });
                bounds[j] = (new_lb, new_ub);
            }
        }
    }

    // ==========================================================
    // 縮約後問題の構築
    // ==========================================================
    let mut col_map = vec![None; n];
    let mut new_col_idx = 0usize;
    for j in 0..n {
        if !removed_cols[j] {
            col_map[j] = Some(new_col_idx);
            new_col_idx += 1;
        }
    }
    let n_new = new_col_idx;

    let mut row_map = vec![None; m];
    let mut new_row_idx = 0usize;
    for i in 0..m {
        if !removed_rows[i] {
            row_map[i] = Some(new_row_idx);
            new_row_idx += 1;
        }
    }
    let m_new = new_row_idx;

    let was_reduced = n_new < n || m_new < m;

    // 新しい目的関数・bounds・b・制約種別
    let mut c_new = vec![0.0f64; n_new];
    let mut bounds_new = vec![(0.0f64, f64::INFINITY); n_new];
    for j in 0..n {
        if let Some(jj) = col_map[j] {
            c_new[jj] = problem.c[j];
            bounds_new[jj] = bounds[j]; // tightened bounds を反映
        }
    }

    let mut b_new = vec![0.0f64; m_new];
    let mut ct_new = vec![ConstraintType::Le; m_new];
    for i in 0..m {
        if let Some(ii) = row_map[i] {
            b_new[ii] = b[i]; // presolve で更新済みの b
            ct_new[ii] = problem.constraint_types[i];
        }
    }

    // 縮約後 CscMatrix を構築
    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
    for j in 0..n {
        if removed_cols[j] {
            continue;
        }
        let jj = col_map[j].unwrap();
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                if removed_rows[row] {
                    continue;
                }
                let ii = row_map[row].unwrap();
                trip_rows.push(ii);
                trip_cols.push(jj);
                trip_vals.push(vals[k]);
            }
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
        postsolve_stack,
        orig_num_vars: n,
        orig_num_constraints: m,
        col_map,
        row_map,
        was_reduced,
        obj_offset,
    })
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
        // min x1 + x2
        // s.t. x1 + x2 <= 5
        //      x1 in [2,2] (fixed), x2 in [0,inf)
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
        let result = run_presolve(&lp).unwrap();
        // x1 is fixed at 2, so it's removed
        assert_eq!(result.reduced_problem.num_vars, 1, "x1 should be removed");
        assert_eq!(result.reduced_problem.num_constraints, 1);
        // b should be updated: 5.0 - 1.0*2 = 3.0
        assert!((result.reduced_problem.b[0] - 3.0).abs() < 1e-10);
        assert!(result.was_reduced);
        assert!((result.obj_offset - 2.0).abs() < 1e-10); // c[0]*2 = 1*2
    }

    #[test]
    fn test_fixed_infeasible() {
        // x in [3, 2] -> infeasible (lb > ub)
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
        assert!(matches!(run_presolve(&lp), Err(PresolveStatus::Infeasible)));
    }

    // -----------------------------------------------------------
    // 2. Empty row/column removal
    // -----------------------------------------------------------
    #[test]
    fn test_empty_row_feasible() {
        // min x
        // s.t. 0 <= 5 (Le, always satisfied)
        //      x <= 3
        // Row 0 is empty (no variables), b=5 >= 0 → feasible, delete row
        let lp = make_lp_general(
            vec![1.0],
            &[1],         // only row 1 has an entry
            &[0],
            &[1.0],
            2,
            1,
            vec![5.0, 3.0],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
        );
        let result = run_presolve(&lp).unwrap();
        assert_eq!(result.reduced_problem.num_constraints, 1);
    }

    #[test]
    fn test_empty_row_infeasible() {
        // 0 <= -1 → infeasible
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
        assert!(matches!(run_presolve(&lp), Err(PresolveStatus::Infeasible)));
    }

    #[test]
    fn test_empty_column_min_with_finite_lb() {
        // min x1 + x2
        // x1 in [0, inf), x2 in [1, inf)
        // No constraints (so x2 is an empty column with c>0)
        // → x2 should be set to lb=1
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            CscMatrix::new(0, 2),
            vec![],
            vec![],
            vec![(0.0, f64::INFINITY), (1.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let result = run_presolve(&lp).unwrap();
        // Both columns are empty (no constraints) → removed
        assert_eq!(result.reduced_problem.num_vars, 0);
        assert!((result.obj_offset - 1.0).abs() < 1e-10); // x1=0, x2=1 → offset=1
    }

    #[test]
    fn test_empty_column_unbounded() {
        // min -x (c=-1) with no constraints and ub=INF → unbounded
        let lp = LpProblem::new_general(
            vec![-1.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        assert!(matches!(run_presolve(&lp), Err(PresolveStatus::Unbounded)));
    }

    // -----------------------------------------------------------
    // 3. Singleton row (Eq)
    // -----------------------------------------------------------
    #[test]
    fn test_singleton_row_eq() {
        // min x1 + x2
        // s.t. 2*x1 = 6 (singleton Eq → x1 = 3)
        //      x1 + x2 <= 10
        //      x1 in [0,inf), x2 in [0,inf)
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
        let result = run_presolve(&lp).unwrap();
        // x1 removed (fixed at 3), row 0 removed
        assert_eq!(result.reduced_problem.num_vars, 1);
        // b[1] updated: 10 - 1*3 = 7
        assert!((result.reduced_problem.b[0] - 7.0).abs() < 1e-10);
        assert!((result.obj_offset - 3.0).abs() < 1e-10); // c[0]*3 = 1*3
    }

    #[test]
    fn test_singleton_row_infeasible() {
        // 2*x = 6, but x in [0, 1] → value=3 > ub=1 → infeasible
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
        assert!(matches!(run_presolve(&lp), Err(PresolveStatus::Infeasible)));
    }

    // -----------------------------------------------------------
    // 4. Redundant constraint removal
    // -----------------------------------------------------------
    #[test]
    fn test_redundant_le() {
        // x1+x2 <= 10, x1 in [0,3], x2 in [0,3]
        // - max(x1+x2) = 6 <= 10 → redundant
        // - x1 <= 3 (ub=3) → redundant
        // - x2 <= 3 (ub=3) → redundant
        // 全3制約が冗長なので削除される
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
        let result = run_presolve(&lp).unwrap();
        // 全制約が冗長 → 縮約後制約数 = 0
        assert_eq!(result.reduced_problem.num_constraints, 0, "all 3 constraints should be redundant");
        assert_eq!(result.reduced_problem.num_vars, 2, "vars retained");

        // 独立した非冗長制約のテスト: x1+x2<=2 with x1,x2 in [0,10]
        // row_ub = 10+10=20 > 2 → NOT redundant
        let lp2 = make_lp_general(
            vec![1.0, 1.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![2.0],
            vec![ConstraintType::Le],
            vec![(0.0, 10.0), (0.0, 10.0)],
        );
        let result2 = run_presolve(&lp2).unwrap();
        assert_eq!(result2.reduced_problem.num_constraints, 1, "x1+x2<=2 is not redundant");
    }

    // -----------------------------------------------------------
    // 5. Bounds tightening
    // -----------------------------------------------------------
    #[test]
    fn test_bounds_tightening() {
        // x + y <= 5, x in [0,10], y in [0,10]
        // a_x=1 > 0, rest_lb (y's min) = 0 → x <= (5-0)/1 = 5 (tighter than 10)
        let lp = make_lp_general(
            vec![1.0, 1.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 10.0), (0.0, 10.0)],
        );
        let result = run_presolve(&lp).unwrap();
        // Constraint is redundant? x+y <= 5 with x,y in [0,10]:
        // row_ub = 1*10 + 1*10 = 20 > 5 → not redundant
        // But bounds should be tightened: x_ub = min(10, 5) = 5, y_ub = min(10, 5) = 5
        // After tightening, row_ub = 1*5 + 1*5 = 10 > 5 → still not redundant
        let _ = result.was_reduced; // just check no crash
        // The reduced problem should have both vars still
        assert_eq!(result.reduced_problem.num_vars, 2);
    }

    /// 負係数Le制約のbounds tighteningで偽陽性Infeasibleが出ないことを確認
    #[test]
    fn test_bounds_tightening_negative_coeff_le_feasible() {
        // x - y <= 5, x in [0, 10], y in [0, 3]
        // 実行可能: x=0, y=0 → 0 <= 5. 偽陽性Infeasibleを出してはいけない
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
        assert!(run_presolve(&lp).is_ok(), "x - y <= 5 should be feasible");
    }

    /// 負係数Ge制約のbounds tighteningで偽陽性Infeasibleが出ないことを確認
    #[test]
    fn test_bounds_tightening_negative_coeff_ge_feasible() {
        // -x + y >= 3, x in [0, 5], y in [0, 8]
        // 実行可能: x=0, y=3 → 0+3=3 >= 3. 偽陽性Infeasibleを出してはいけない
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
        assert!(run_presolve(&lp).is_ok(), "-x + y >= 3 should be feasible");
    }

    // -----------------------------------------------------------
    // Presolve roundtrip: verify no-crash + obj_offset consistency
    // -----------------------------------------------------------
    #[test]
    fn test_presolve_no_crash_netlib_like() {
        // 3var LP: min -x1 - x2 - x3
        // s.t. x1 + x2 + x3 <= 4, x1 <= 3, x2 <= 3, x3 <= 3
        let lp = make_lp(
            vec![-1.0, -1.0, -1.0],
            &[0, 0, 0, 1, 2, 3],
            &[0, 1, 2, 0, 1, 2],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            4,
            3,
            vec![4.0, 3.0, 3.0, 3.0],
        );
        let result = run_presolve(&lp).unwrap();
        // No reductions expected (no fixed vars, no empty rows/cols, not redundant)
        assert_eq!(result.reduced_problem.num_vars, 3);
        assert_eq!(result.reduced_problem.num_constraints, 4);
    }
}
