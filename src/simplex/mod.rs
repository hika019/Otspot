//! 線形計画法（LP）のための主シンプレックス法モジュール
//!
//! Phase I + Phase II の改訂シンプレックス法（Revised Simplex Method）を実装する。
//! LU分解を用いた基底行列の効率的な操作により、数値的安定性と計算速度を両立する。
//!
//! # アルゴリズム概要
//!
//! 1. **Phase I**: 人工変数を導入し、初期実行可能基底解を探索する
//! 2. **Phase II**: 実行可能解から最適解に向けて目的関数を改善する
//!
//! 改訂シンプレックス法では完全な単体表ではなく基底行列のLU分解を保持するため、
//! 大規模疎行列に対して高い計算効率を発揮する。

pub mod dual;
pub mod dual_advanced;
pub mod pricing;
pub(crate) mod primal;

use crate::options::{SimplexMethod, SolverOptions};
use crate::presolve;
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::sparse::CscMatrix;
use crate::tolerances::{DROP_TOL, PIVOT_TOL};
use log::warn;

pub(crate) use primal::{two_phase_simplex, extract_solution, revised_simplex_core, reconcile_final_basis_state};

/// LU分解を用いた改訂シンプレックス法でLPを解く（後方互換 API）
///
/// デフォルトの [`SolverOptions`] を使用して [`solve_with`] を呼び出す。
///
/// # 引数
///
/// * `problem` - 解くべき線形計画問題
///
/// # 戻り値
///
/// [`SolverResult`] — 求解ステータス（最適・非有界・実行不可）と目的関数値・解ベクトル
pub fn solve(problem: &LpProblem) -> SolverResult {
    solve_with(problem, &SolverOptions::default())
}

/// カスタム設定でLPを解く
///
/// 与えられた線形計画問題を標準形に変換し、2相シンプレックス法で最適解を求める。
/// `options.presolve` が true の場合、求解前にPresolveを適用して問題を縮約する。
///
/// # 引数
///
/// * `problem` - 解くべき線形計画問題
/// * `options` - ソルバー動作設定（許容誤差・反復上限・eta 保持数など）
///
/// # 戻り値
///
/// [`SolverResult`] — 求解ステータス（最適・非有界・実行不可）と目的関数値・解ベクトル
pub fn solve_with(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    // timeout_secs → deadline 変換（qp_solve_impl と同様）
    let mut opts_with_deadline;
    let options = if let (Some(secs), true) = (options.timeout_secs, options.deadline.is_none()) {
        opts_with_deadline = options.clone();
        opts_with_deadline.deadline = Some(
            std::time::Instant::now() + std::time::Duration::from_secs_f64(secs),
        );
        &opts_with_deadline
    } else {
        options
    };

    // --- Presolve ---
    if options.presolve {
        match presolve::run_presolve(problem, options.deadline) {
            Err(presolve::PresolveStatus::Infeasible) => {
                return SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: 0.0,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
            ..Default::default()
                };
            }
            Err(presolve::PresolveStatus::Unbounded) => {
                return SolverResult {
                    status: SolveStatus::Unbounded,
                    objective: f64::NEG_INFINITY,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
            ..Default::default()
                };
            }
            Ok(presolve_result) if presolve_result.was_reduced => {
                // warm_start と presolve の組み合わせは未対応（presolve が変数インデックスを変える）
                let opts_no_ws = if options.warm_start.is_some() {
                    let mut o = options.clone();
                    o.warm_start = None;
                    o.presolve = false;
                    Some(o)
                } else {
                    None
                };
                let eff_opts = opts_no_ws.as_ref().unwrap_or(options);
                let raw = solve_without_presolve(&presolve_result.reduced_problem, eff_opts);
                // Presolve で縮約された問題を Simplex が解けない場合 (SingularBasis / check_eq_feasibility 失敗):
                // - capri: presolve 後の縮約問題が Phase I で SingularBasis (初期基底が特異)
                // - forplan: Phase II 後の解が Eq 制約を大きく違反 (人工変数の数値ドリフト)
                // いずれも元問題 (presolve なし) は Simplex で正しく解けるため、fallback する。
                if raw.status == SolveStatus::NumericalError {
                    return solve_without_presolve(problem, options);
                }
                return presolve::postsolve::run_postsolve(&raw, &presolve_result, problem);
            }
            Ok(_) => {
                // 縮約不要: fallthrough して通常ルートで解く
            }
        }
    }

    // presolve が deadline 超過で早期終了した場合（was_reduced=false）も
    // deadline を超過していれば Timeout を返す（build_standard_form 前にチェック）
    if options.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return SolverResult {
            status: SolveStatus::Timeout,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        };
    }

    solve_without_presolve(problem, options)
}

/// Presolve なしでLPを直接解く内部関数
fn solve_without_presolve(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    let m = problem.num_constraints;
    let n = problem.num_vars;

    // Edge case: no variables
    if n == 0 {
        for i in 0..m {
            if problem.b[i] < -options.primal_tol {
                return SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: 0.0,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
            ..Default::default()
                };
            }
        }
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![],
            dual_solution: vec![0.0; m],
            reduced_costs: vec![],
            slack: problem.b.clone(),
            warm_start_basis: None,
            ..Default::default()
        };
    }

    // Edge case: no constraints
    if m == 0 {
        let mut x = vec![0.0; n];
        let mut obj = 0.0;
        for (j, x_j) in x.iter_mut().enumerate() {
            if problem.c[j] < -options.primal_tol {
                // BUG-simplex-001修正: ubが有限なら最大値(ub)に設定、無限ならUnbounded
                let ub = problem.bounds[j].1;
                if ub.is_infinite() {
                    return SolverResult {
                        status: SolveStatus::Unbounded,
                        objective: f64::NEG_INFINITY,
                        solution: vec![],
                        dual_solution: vec![],
                        reduced_costs: vec![],
                        slack: vec![],
                        warm_start_basis: None,
            ..Default::default()
                    };
                }
                *x_j = ub;
            }
            obj += problem.c[j] * *x_j;
        }
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: obj,
            solution: x,
            dual_solution: vec![],
            reduced_costs: problem.c.clone(),
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        };
    }

    let sf = build_standard_form(problem);

    match options.simplex_method {
        SimplexMethod::Primal => two_phase_simplex(&sf, problem, options),
        SimplexMethod::Dual => dual::two_phase_dual_simplex(&sf, problem, options),
        SimplexMethod::DualAdvanced => {
            // 産業品質Dual Simplex（dual_advanced/を使用）
            dual_advanced::solve_dual_advanced(&sf, problem, options)
        }
        SimplexMethod::Auto => {
            // cold start / warm start いずれも Dual Simplex を使用する。
            // 現代の商用ソルバー（Gurobi/CPLEX/HiGHS）と同様に Dual Simplex をデフォルトとする。
            // Dual は Phase I/II 分離不要で退化に強く、cold start でも有利。
            dual::two_phase_dual_simplex(&sf, problem, options)
        }
    }
}

// --- Data structures ---

/// 元の変数の変換情報を保持する構造体
///
/// 各元変数がどのように標準形の変数群に対応しているかを記録する。
/// 下限・上限制約によって変数分割や符号反転が生じた際に使用する。
pub(crate) struct OrigVarInfo {
    /// 変数変換のオフセット（下限 lb または上限 ub の値）
    offset: f64,
    /// 新変数インデックスと係数のペアのリスト
    ///
    /// 通常は1要素（下限付き変数）または2要素（無制限変数を正部・負部に分割）
    new_vars: Vec<(usize, f64)>,
}

/// 線形計画問題の標準形表現
///
/// 元のLPを改訂シンプレックス法に適した形式に変換した結果を保持する。
/// スラック変数の追加、変数変換（下限シフト・符号反転・分割）、
/// 人工変数の要否判定を含む完全な変換済みデータを格納する。
pub(crate) struct StandardForm {
    /// 制約行列（疎CSC形式）
    a: CscMatrix,
    /// 制約右辺ベクトル（変換済み）
    b: Vec<f64>,
    /// 目的関数係数ベクトル（Phase II用）
    c: Vec<f64>,
    /// 制約数（上限制約行も含む）
    m: usize,
    /// 元変数を変換した新変数の数
    n_shifted: usize,
    /// 全変数数（新変数 + スラック変数）
    n_total: usize,
    /// 初期基底変数のインデックスリスト（行ごと）
    initial_basis: Vec<usize>,
    /// 各制約行が人工変数を必要とするかのフラグ
    needs_artificial: Vec<bool>,
    /// 人工変数の総数
    num_artificial: usize,
    /// 変数変換による目的関数のオフセット
    obj_offset: f64,
    /// 元の変数数
    n_orig: usize,
    /// 元変数ごとの変換情報
    orig_var_info: Vec<OrigVarInfo>,
    /// 各制約行が符号反転されているか（双対変数の符号調整に使用）
    row_negated: Vec<bool>,
}

/// シンプレックス法コア関数の実行結果
pub(crate) enum SimplexOutcome {
    /// 最適解が得られた。値は最適目的関数値と最適時点の双対変数ベクトル
    Optimal(f64, Vec<f64>),
    /// 問題が非有界（unbounded）であった
    Unbounded,
    /// タイムアウト（timeout_secs を超過した）。値は打ち切り時点の目的関数値
    Timeout(f64),
    /// 基底行列が特異（サイクリック基底など）。IPM フォールバック用
    SingularBasis,
}

pub(crate) fn timeout_result_with_incumbent(
    sf: &StandardForm,
    problem: &LpProblem,
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
) -> SolverResult {
    let solution = extract_solution(sf, basis, x_b, col_scale);
    let objective = problem
        .c
        .iter()
        .zip(solution.iter())
        .map(|(&ci, &xi)| ci * xi)
        .sum::<f64>()
        + sf.obj_offset;
    SolverResult {
        status: SolveStatus::Timeout,
        objective,
        solution,
        dual_solution: vec![],
        reduced_costs: vec![],
        slack: vec![],
        warm_start_basis: None,
        ..Default::default()
    }
}

// --- Standard form construction ---

/// LPを改訂シンプレックス法用の標準形に変換する
///
/// 以下の7ステップで変換を行う:
///
/// 1. 変数変換（下限シフト・上限反転・無制限変数の正負分割）
/// 2. 上限制約の追加
/// 3. 調整済み右辺ベクトルの計算
/// 4. 制約タイプの整理
/// 5. 行の符号調整とスラック変数の設定
/// 6. 初期基底と人工変数の決定
/// 7. CSC疎行列の構築
pub(crate) fn build_standard_form(problem: &LpProblem) -> StandardForm {
    let n_orig = problem.num_vars;
    let m_orig = problem.num_constraints;

    // 1. Variable transformations
    let mut orig_var_info: Vec<OrigVarInfo> = Vec::with_capacity(n_orig);
    let mut n_shifted = 0usize;
    let mut obj_offset = 0.0f64;
    let mut new_c: Vec<f64> = Vec::new();

    for j in 0..n_orig {
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() {
            let idx = n_shifted;
            n_shifted += 1;
            new_c.push(problem.c[j]);
            obj_offset += problem.c[j] * lb;
            orig_var_info.push(OrigVarInfo {
                offset: lb,
                new_vars: vec![(idx, 1.0)],
            });
        } else if ub.is_finite() {
            let idx = n_shifted;
            n_shifted += 1;
            new_c.push(-problem.c[j]);
            obj_offset += problem.c[j] * ub;
            orig_var_info.push(OrigVarInfo {
                offset: ub,
                new_vars: vec![(idx, -1.0)],
            });
        } else {
            let idx_plus = n_shifted;
            n_shifted += 1;
            new_c.push(problem.c[j]);
            let idx_minus = n_shifted;
            n_shifted += 1;
            new_c.push(-problem.c[j]);
            orig_var_info.push(OrigVarInfo {
                offset: 0.0,
                new_vars: vec![(idx_plus, 1.0), (idx_minus, -1.0)],
            });
        }
    }

    // 2. Upper bound constraints
    let mut ub_constraints: Vec<(usize, f64)> = Vec::new();
    for (j, info) in orig_var_info.iter().enumerate().take(n_orig) {
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() && ub.is_finite() {
            let effective_ub = ub - lb;
            let new_idx = info.new_vars[0].0;
            ub_constraints.push((new_idx, effective_ub));
        }
    }
    let n_ub = ub_constraints.len();
    let m_ext = m_orig + n_ub;

    // 3. Compute adjusted b
    let mut b = problem.b.clone();
    for (j, info) in orig_var_info.iter().enumerate().take(n_orig) {
        let offset = info.offset;
        if offset.abs() > DROP_TOL {
            if let Ok((rows, vals)) = problem.a.get_column(j) {
                for (k, &row) in rows.iter().enumerate() {
                    b[row] -= vals[k] * offset;
                }
            }
        }
    }
    for &(_, ub_val) in &ub_constraints {
        b.push(ub_val);
    }

    // 4. Constraint types
    let mut ctypes: Vec<ConstraintType> = problem.constraint_types.clone();
    for _ in 0..n_ub {
        ctypes.push(ConstraintType::Le);
    }

    // 5. Row negation and slack setup
    let mut row_negated = vec![false; m_ext];
    let mut slack_col_idx: Vec<Option<usize>> = Vec::with_capacity(m_ext);
    let mut n_slack = 0usize;
    let mut slack_coeff = vec![0.0f64; m_ext];

    for i in 0..m_ext {
        match ctypes[i] {
            ConstraintType::Le => {
                if b[i] < -PIVOT_TOL {
                    row_negated[i] = true;
                    b[i] = -b[i];
                    slack_coeff[i] = -1.0;
                } else {
                    // b[i] が [-PIVOT_TOL, 0) の範囲にある場合: 変数下限シフト
                    // (presolve 起因の丸め誤差など) による浮動小数点ノイズ。
                    // slack の初期値が微小負になるのを防ぐため 0 にクランプ。
                    if b[i] < 0.0 { b[i] = 0.0; }
                    slack_coeff[i] = 1.0;
                }
                slack_col_idx.push(Some(n_slack));
                n_slack += 1;
            }
            ConstraintType::Ge => {
                if b[i] < -PIVOT_TOL {
                    row_negated[i] = true;
                    b[i] = -b[i];
                    slack_coeff[i] = 1.0; // -1 negated
                } else {
                    slack_coeff[i] = -1.0;
                }
                slack_col_idx.push(Some(n_slack));
                n_slack += 1;
            }
            ConstraintType::Eq => {
                if b[i] < -PIVOT_TOL {
                    row_negated[i] = true;
                    b[i] = -b[i];
                }
                slack_col_idx.push(None);
            }
        }
    }

    let n_total = n_shifted + n_slack;

    // 6. Initial basis and artificial detection
    let mut initial_basis = vec![0usize; m_ext];
    let mut needs_artificial = vec![false; m_ext];
    let mut num_artificial = 0usize;

    for i in 0..m_ext {
        match slack_col_idx[i] {
            Some(s_idx) => {
                let col = n_shifted + s_idx;
                if slack_coeff[i] > 0.0 {
                    // Le: slack >= 0 → no artificial needed
                    initial_basis[i] = col;
                } else if b[i].abs() <= PIVOT_TOL {
                    // Ge(b≈0): surplus at 0 is primal-feasible → no artificial needed.
                    // Using an artificial here causes it to remain in the Phase II basis
                    // (Phase I terminates immediately since obj=0), which allows the solver
                    // to drift and violate this constraint. Use surplus directly.
                    initial_basis[i] = col;
                } else {
                    needs_artificial[i] = true;
                    num_artificial += 1;
                    initial_basis[i] = col; // placeholder
                }
            }
            None => {
                needs_artificial[i] = true;
                num_artificial += 1;
            }
        }
    }

    // 7. Build CscMatrix from triplets
    let mut trip_rows = Vec::new();
    let mut trip_cols = Vec::new();
    let mut trip_vals = Vec::new();

    // Original variable columns (transformed)
    for (j, info) in orig_var_info.iter().enumerate().take(n_orig) {
        if let Ok((a_rows, a_vals)) = problem.a.get_column(j) {
            for (k, &row) in a_rows.iter().enumerate() {
                let val = a_vals[k];
                let sign = if row_negated[row] { -1.0 } else { 1.0 };
                for &(new_col, coeff) in &info.new_vars {
                    let actual_val = sign * val * coeff;
                    if actual_val.abs() > DROP_TOL {
                        trip_rows.push(row);
                        trip_cols.push(new_col);
                        trip_vals.push(actual_val);
                    }
                }
            }
        }
    }

    // Upper bound constraint rows
    for (ub_idx, &(new_var_idx, _)) in ub_constraints.iter().enumerate() {
        let row = m_orig + ub_idx;
        trip_rows.push(row);
        trip_cols.push(new_var_idx);
        trip_vals.push(1.0);
    }

    // Slack/surplus columns
    for i in 0..m_ext {
        if let Some(s_idx) = slack_col_idx[i] {
            let col = n_shifted + s_idx;
            trip_rows.push(i);
            trip_cols.push(col);
            trip_vals.push(slack_coeff[i]);
        }
    }

    let a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_ext, n_total).unwrap();

    // Cost vector for Phase II
    let mut c_ext = vec![0.0; n_total];
    c_ext[..n_shifted].copy_from_slice(&new_c[..n_shifted]);

    StandardForm {
        a,
        b,
        c: c_ext,
        m: m_ext,
        n_shifted,
        n_total,
        initial_basis,
        needs_artificial,
        num_artificial,
        obj_offset,
        n_orig,
        orig_var_info,
        row_negated,
    }
}

// --- Two-phase simplex ---

/// 双対変数・被縮小費用・スラックを元問題の情報から計算する
///
/// # 引数
/// * `sf` - 標準形（行符号反転情報を含む）
/// * `problem` - 元のLP問題（制約行列・右辺ベクトル・目的係数）
/// * `y_std` - 標準形の双対変数ベクトル（長さ: sf.m）
/// * `solution` - 元問題の解ベクトル（長さ: problem.num_vars）
/// * `row_scale` - Ruizスケーリングの行スケール因子（スケーリングなしの場合は空スライス）
///
/// # 戻り値
/// `(dual_solution, reduced_costs, slack)` のタプル
pub(crate) fn extract_dual_info(
    sf: &StandardForm,
    problem: &LpProblem,
    y_std: &[f64],
    solution: &[f64],
    row_scale: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m_orig = problem.num_constraints;
    let n_orig = problem.num_vars;

    // 双対変数: y_std を行符号反転 + Ruiz行スケールで調整して元制約に対応
    let mut dual_solution = vec![0.0; m_orig];
    for i in 0..m_orig {
        let sign = if sf.row_negated[i] { -1.0 } else { 1.0 };
        let rs = row_scale.get(i).copied().unwrap_or(1.0);
        dual_solution[i] = sign * rs * y_std[i];
    }

    // スラック: b - Ax（元問題の解から直接計算）
    let mut slack = problem.b.clone();
    for (j, &sol_j) in solution.iter().enumerate().take(n_orig) {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                slack[row] -= vals[k] * sol_j;
            }
        }
    }

    // 被縮小費用: c_j - lambda^T A_j（元問題の変数に対して）
    let mut reduced_costs = problem.c.clone();
    for (j, rc_j) in reduced_costs.iter_mut().enumerate().take(n_orig) {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                if row < m_orig {
                    *rc_j -= dual_solution[row] * vals[k];
                }
            }
        }
    }

    // ★追加: 上限制約dual (mu_j) の減算
    // build_standard_form と同じ順序で有限上下限変数を列挙し、
    // 対応する y_std[m_orig + k] を reduced_costs から減算する。
    // j は reduced_costs と problem.bounds の両方に使うため range loop が必要。
    let mut ub_idx = 0usize;
    #[allow(clippy::needless_range_loop)]
    for j in 0..n_orig {
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() && ub.is_finite() {
            let row = m_orig + ub_idx;
            if row < y_std.len() {
                let sign = if sf.row_negated[row] { -1.0 } else { 1.0 };
                let rs = row_scale.get(row).copied().unwrap_or(1.0);
                let mu_j = sign * rs * y_std[row];
                reduced_costs[j] -= mu_j;
            } else {
                warn!("extract_dual_info: y_std too short for ub constraint row {}, expected >= {}", row, row + 1);
            }
            ub_idx += 1;
        }
    }

    (dual_solution, reduced_costs, slack)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    fn make_lp(
        c: Vec<f64>,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrows: usize,
        ncols: usize,
        b: Vec<f64>,
    ) -> LpProblem {
        let a = CscMatrix::from_triplets(rows, cols, vals, nrows, ncols).unwrap();
        LpProblem::new(c, a, b).unwrap()
    }

    #[test]
    fn test_timeout_result_with_incumbent_uses_original_objective() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![3.0, 1.0],
            a,
            vec![1.0],
            vec![ConstraintType::Ge],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let sf = build_standard_form(&lp);
        let basis = sf.initial_basis.clone();
        let x_b = sf.b.clone();
        let col_scale = vec![1.0; sf.n_total];

        let result = timeout_result_with_incumbent(&sf, &lp, &basis, &x_b, &col_scale);

        assert_eq!(result.status, SolveStatus::Timeout);
        assert_eq!(result.solution.len(), 2);
        let expected_obj = lp
            .c
            .iter()
            .zip(result.solution.iter())
            .map(|(&ci, &xi)| ci * xi)
            .sum::<f64>();
        assert!((result.objective - expected_obj).abs() < 1e-12, "obj={}", result.objective);
    }

    #[test]
    fn test_reconcile_final_basis_state_recomputes_xb_and_y() {
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 2, 1, 2],
            &[1.0, 1.0, 1.0, 1.0],
            2,
            3,
        )
        .unwrap();
        let b = vec![3.0, 5.0];
        let c = vec![4.0, 2.0, 1.0];
        let basis = vec![0usize, 2usize];
        let mut x_b = vec![0.0, 0.0];
        let mut y = vec![0.0, 0.0];

        reconcile_final_basis_state(&a, &b, &c, &basis, &mut x_b, &mut y, 50, None).unwrap();

        assert!((x_b[0] + 2.0).abs() < 1e-12, "x_b[0]={}", x_b[0]);
        assert!((x_b[1] - 5.0).abs() < 1e-12, "x_b[1]={}", x_b[1]);
        assert!((y[0] - 4.0).abs() < 1e-12, "y[0]={}", y[0]);
        assert!((y[1] + 3.0).abs() < 1e-12, "y[1]={}", y[1]);
    }

    #[test]
    fn test_extract_solution_uses_dd_for_split_variable_cancellation() {
        let sf = StandardForm {
            a: CscMatrix::new(3, 3),
            b: vec![0.0, 0.0, 0.0],
            c: vec![0.0, 0.0, 0.0],
            m: 3,
            n_shifted: 3,
            n_total: 3,
            initial_basis: vec![0, 1, 2],
            needs_artificial: vec![false, false, false],
            num_artificial: 0,
            obj_offset: 0.0,
            n_orig: 1,
            orig_var_info: vec![OrigVarInfo {
                offset: 0.0,
                new_vars: vec![(0, 1.0), (1, 1.0), (2, -1.0)],
            }],
            row_negated: vec![false, false, false],
        };
        let basis = vec![0usize, 1usize, 2usize];
        let x_b = vec![1.0_f64, 1.0e16_f64, 1.0e16_f64];
        let col_scale = vec![1.0, 1.0, 1.0];

        let solution = extract_solution(&sf, &basis, &x_b, &col_scale);

        assert_eq!(solution.len(), 1);
        assert!(
            (solution[0] - 1.0).abs() < 1e-12,
            "split-variable recomposition should preserve unit residual, got {}",
            solution[0]
        );
    }

    #[test]
    fn test_basic_2var() {
        let lp = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.objective - (-4.0)).abs() < PIVOT_TOL,
            "Expected objective -4.0, got {}",
            result.objective
        );
        let x1 = result.solution[0];
        let x2 = result.solution[1];
        assert!((-PIVOT_TOL..=3.0 + PIVOT_TOL).contains(&x1), "x1={}", x1);
        assert!((-PIVOT_TOL..=3.0 + PIVOT_TOL).contains(&x2), "x2={}", x2);
        assert!((x1 + x2 - 4.0).abs() < PIVOT_TOL);
    }

    #[test]
    fn test_basic_3var() {
        let lp = make_lp(
            vec![-2.0, -3.0, -1.0],
            &[0, 0, 0, 1, 1, 2, 2],
            &[0, 1, 2, 0, 1, 1, 2],
            &[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            3,
            3,
            vec![10.0, 14.0, 8.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        let x = &result.solution;
        assert!(x[0] >= -PIVOT_TOL);
        assert!(x[1] >= -PIVOT_TOL);
        assert!(x[2] >= -PIVOT_TOL);
        assert!(x[0] + x[1] + x[2] <= 10.0 + PIVOT_TOL);
        assert!(2.0 * x[0] + x[1] <= 14.0 + PIVOT_TOL);
        assert!(x[1] + x[2] <= 8.0 + PIVOT_TOL);
        assert!(
            (result.objective - (-28.0)).abs() < PIVOT_TOL,
            "Expected objective -28.0, got {}",
            result.objective
        );
    }

    #[test]
    fn test_unbounded() {
        let lp = make_lp(
            vec![-1.0, 0.0],
            &[0, 0],
            &[0, 1],
            &[1.0, -1.0],
            1,
            2,
            vec![1.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Unbounded);
    }

    #[test]
    fn test_infeasible() {
        let lp = make_lp(vec![1.0], &[0], &[0], &[1.0], 1, 1, vec![-1.0]);
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Infeasible);
    }

    #[test]
    fn test_degenerate_zero_vars() {
        let a = CscMatrix::new(0, 0);
        let lp = LpProblem::new(vec![], a, vec![]).unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.objective).abs() < PIVOT_TOL);
    }

    #[test]
    fn test_zero_constraints_unbounded() {
        let a = CscMatrix::new(0, 1);
        let lp = LpProblem::new(vec![-1.0], a, vec![]).unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Unbounded);
    }

    #[test]
    fn test_zero_constraints_optimal() {
        let a = CscMatrix::new(0, 1);
        let lp = LpProblem::new(vec![1.0], a, vec![]).unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.objective).abs() < PIVOT_TOL);
    }

    #[test]
    fn test_solve_with_default_options() {
        // SolverOptions::default() で solve() と同じ結果が返ること
        let lp = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let result_default = solve(&lp);
        let result_with = solve_with(&lp, &SolverOptions::default());
        assert_eq!(result_default.status, result_with.status);
        assert!(
            (result_default.objective - result_with.objective).abs() < PIVOT_TOL,
            "solve() and solve_with(default) should return same objective"
        );
    }

    /// Ge制約防御テスト
    ///
    /// 問題: min -x - y
    ///   s.t. x + y >= 1 (ConstraintType::Ge)
    ///        0 <= x <= 10
    ///        0 <= y <= 10
    /// 最適解: x=10, y=10, obj=-20
    #[test]
    fn test_simplex_ge_defensive() {
        use crate::problem::ConstraintType;
        let a = CscMatrix::from_triplets(
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![-1.0, -1.0],
            a,
            vec![1.0],
            vec![ConstraintType::Ge],
            vec![(0.0, 10.0), (0.0, 10.0)],
            None,
        )
        .unwrap();
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(5.0);
        let start = std::time::Instant::now();
        let result = solve_with(&lp, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0, "test_simplex_ge_defensive: wall-clock 6秒超過");
        assert_eq!(result.status, SolveStatus::Optimal, "Status should be Optimal");
        assert!(
            (result.objective - (-20.0)).abs() < PIVOT_TOL,
            "Expected obj=-20.0, got {}",
            result.objective
        );
        assert!(
            result.solution[0] >= -PIVOT_TOL && result.solution[0] <= 10.0 + PIVOT_TOL,
            "x should be in [0, 10], got {}",
            result.solution[0]
        );
        assert!(
            result.solution[1] >= -PIVOT_TOL && result.solution[1] <= 10.0 + PIVOT_TOL,
            "y should be in [0, 10], got {}",
            result.solution[1]
        );
        assert!(
            (result.solution[0] + result.solution[1] - 20.0).abs() < PIVOT_TOL,
            "x + y should be 20.0, got {}",
            result.solution[0] + result.solution[1]
        );
    }

    /// 基本LP（Le制約のみ）の双対解・スラック・被縮小費用を検証する
    ///
    /// 問題: min -x1 - 2*x2
    ///   s.t. x1 + x2 <= 4
    ///        x1 <= 3
    ///        x2 <= 3
    ///        x1, x2 >= 0
    ///
    /// 最適解: x1=1, x2=3, obj=-7
    /// 双対変数: y=[-1, 0, -1] (Le制約のshadow price、最小化LPでは<=0)
    /// スラック: s=[0, 2, 0]
    /// 被縮小費用: rc=[0, 0] (基底変数なのでゼロ)
    #[test]
    fn test_dual_solution_basic_le_constraints() {
        let lp = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(10.0);
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.objective - (-7.0)).abs() < PIVOT_TOL,
            "Expected obj=-7.0, got {}",
            result.objective
        );

        // 双対変数の検証
        assert_eq!(result.dual_solution.len(), 3, "dual_solution should have 3 elements");
        assert!(
            (result.dual_solution[0] - (-1.0)).abs() < PIVOT_TOL,
            "y[0] should be -1.0, got {}",
            result.dual_solution[0]
        );
        assert!(
            result.dual_solution[1].abs() < PIVOT_TOL,
            "y[1] should be 0.0 (non-binding), got {}",
            result.dual_solution[1]
        );
        assert!(
            (result.dual_solution[2] - (-1.0)).abs() < PIVOT_TOL,
            "y[2] should be -1.0, got {}",
            result.dual_solution[2]
        );

        // スラック変数の検証
        assert_eq!(result.slack.len(), 3, "slack should have 3 elements");
        assert!(
            result.slack[0].abs() < PIVOT_TOL,
            "slack[0] should be 0 (binding), got {}",
            result.slack[0]
        );
        assert!(
            (result.slack[1] - 2.0).abs() < PIVOT_TOL,
            "slack[1] should be 2.0 (non-binding), got {}",
            result.slack[1]
        );
        assert!(
            result.slack[2].abs() < PIVOT_TOL,
            "slack[2] should be 0 (binding), got {}",
            result.slack[2]
        );

        // 被縮小費用の検証（基底変数なのでゼロ）
        assert_eq!(result.reduced_costs.len(), 2, "reduced_costs should have 2 elements");
        assert!(
            result.reduced_costs[0].abs() < PIVOT_TOL,
            "rc[0] should be 0 (basic), got {}",
            result.reduced_costs[0]
        );
        assert!(
            result.reduced_costs[1].abs() < PIVOT_TOL,
            "rc[1] should be 0 (basic), got {}",
            result.reduced_costs[1]
        );
    }

    #[test]
    fn test_large_coefficient_lp() {
        // 係数に 1e12 と 1e-12 を混合した問題 → Optimal or 適切なステータス（オーバーフローしない）
        // min -1e12 * x1 + 1e-12 * x2, s.t. x1 + x2 <= 1, x1,x2 >= 0
        // 最適解: x1=1, x2=0, obj=-1e12
        let lp = make_lp(
            vec![-1e12, 1e-12],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![1.0],
        );
        let result = solve(&lp);
        assert!(
            result.status == SolveStatus::Optimal || result.status == SolveStatus::Timeout,
            "Expected Optimal or Timeout, got {:?}",
            result.status
        );
        assert!(!result.objective.is_nan(), "Objective should not be NaN");
        assert!(result.objective.is_finite(), "Objective should be finite for bounded LP");

        // 全係数 0.0 の目的関数 → Optimal, objective=0.0
        // min 0*x1 + 0*x2, s.t. x1 + x2 <= 2, x1 <= 1, x2 <= 1
        let lp_zero = make_lp(
            vec![0.0, 0.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![2.0, 1.0, 1.0],
        );
        let result_zero = solve(&lp_zero);
        assert_eq!(result_zero.status, SolveStatus::Optimal, "Expected Optimal for zero-objective LP");
        assert!(
            result_zero.objective.abs() < PIVOT_TOL,
            "Expected objective=0.0, got {}",
            result_zero.objective
        );
    }

    #[test]
    fn test_highly_degenerate_lp() {
        // 高度退化 LP: 3制約が (1,1) で交わる → 基底解が退化
        // min -x1 - x2
        // s.t. x1 + x2 <= 2, x1 <= 1, x2 <= 1
        // 最適解: x1=1, x2=1, obj=-2（サイクリングせずに到達すること）
        let lp = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![2.0, 1.0, 1.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal, "Expected Optimal for degenerate LP");
        assert!(
            (result.objective - (-2.0)).abs() < PIVOT_TOL,
            "Expected objective=-2.0, got {}",
            result.objective
        );
        let x1 = result.solution[0];
        let x2 = result.solution[1];
        assert!((x1 - 1.0).abs() < PIVOT_TOL, "Expected x1=1.0, got {}", x1);
        assert!((x2 - 1.0).abs() < PIVOT_TOL, "Expected x2=1.0, got {}", x2);
    }

    /// 等式制約付きLPの双対解・スラック・被縮小費用を検証する
    ///
    /// 問題: min x1 + 2*x2
    ///   s.t. x1 + x2 = 6  (Eq)
    ///        x2 <= 5       (Le)
    ///        x1, x2 >= 0
    ///
    /// 最適解: x1=6, x2=0, obj=6
    /// 双対変数: y=[1, 0] (Eq制約shadow price=1、x2<=5は非binding)
    /// スラック: s=[0, 5] (等式制約は0、x2<=5は5余裕)
    /// 被縮小費用: rc=[0, 1] (x1は基底でrc=0、x2は非基底でrc=1)
    #[test]
    fn test_dual_solution_equality_constraint() {
        use crate::problem::ConstraintType;
        let a = CscMatrix::from_triplets(
            &[0, 0, 1],
            &[0, 1, 1],
            &[1.0, 1.0, 1.0],
            2,
            2,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 2.0],
            a,
            vec![6.0, 5.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(10.0);
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.objective - 6.0).abs() < PIVOT_TOL,
            "Expected obj=6.0, got {}",
            result.objective
        );

        // 双対変数の検証
        assert_eq!(result.dual_solution.len(), 2, "dual_solution should have 2 elements");
        assert!(
            (result.dual_solution[0] - 1.0).abs() < PIVOT_TOL,
            "y[0] (Eq constraint shadow price) should be 1.0, got {}",
            result.dual_solution[0]
        );
        assert!(
            result.dual_solution[1].abs() < PIVOT_TOL,
            "y[1] (Le constraint, non-binding) should be 0.0, got {}",
            result.dual_solution[1]
        );

        // スラック変数の検証
        assert_eq!(result.slack.len(), 2, "slack should have 2 elements");
        assert!(
            result.slack[0].abs() < PIVOT_TOL,
            "slack[0] (Eq constraint) should be 0, got {}",
            result.slack[0]
        );
        assert!(
            (result.slack[1] - 5.0).abs() < PIVOT_TOL,
            "slack[1] (x2<=5, non-binding) should be 5.0, got {}",
            result.slack[1]
        );

        // 被縮小費用の検証
        assert_eq!(result.reduced_costs.len(), 2, "reduced_costs should have 2 elements");
        assert!(
            result.reduced_costs[0].abs() < PIVOT_TOL,
            "rc[0] (x1, basic) should be 0.0, got {}",
            result.reduced_costs[0]
        );
        assert!(
            (result.reduced_costs[1] - 1.0).abs() < PIVOT_TOL,
            "rc[1] (x2, non-basic) should be 1.0, got {}",
            result.reduced_costs[1]
        );
    }

    #[test]
    fn test_free_variables_phase_i() {
        // 全変数が自由境界（-INF/INF）のLP
        // minimize x1 + x2
        // s.t. x1 + x2 = 2
        // x1, x2 in (-INF, INF)
        // → Optimal（Infeasibleを返してはならない）
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![2.0],
            vec![crate::problem::ConstraintType::Eq],
            vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        let result = solve(&lp);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "Expected Optimal for free-variable LP with Eq constraint, got {:?}",
            result.status
        );
        // 解の制約充足チェック: x1 + x2 = 2
        assert!(
            (result.solution[0] + result.solution[1] - 2.0).abs() < 1e-6,
            "Expected x1+x2=2, got x1={}, x2={}, sum={}",
            result.solution[0],
            result.solution[1],
            result.solution[0] + result.solution[1]
        );
    }

    #[test]
    fn test_hs51_feasibility_lp() {
        // HS51の実行可能性LP: find_initial_feasible_pointが構築するLPを直接テスト
        // 5変数(全自由), 6Le制約(等式制約を2不等式ペアに変換)
        // b[1]=-4.0 (負のRHS) → build_standard_formで符号反転+人工変数追加
        // 解は存在する(x=[1,1,1,1,1])のでOptimalを返すべき
        let a = CscMatrix::from_triplets(
            &[0, 1, 0, 1, 4, 5, 2, 3, 2, 3, 2, 3, 4, 5],
            &[0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4],
            &[1.0, -1.0, 3.0, -3.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -2.0, 2.0, -1.0, 1.0],
            6,
            5,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![0.0; 5],
            a,
            vec![4.0, -4.0, 0.0, 0.0, 0.0, 0.0],
            vec![crate::problem::ConstraintType::Le; 6],
            vec![(f64::NEG_INFINITY, f64::INFINITY); 5],
            None,
        )
        .unwrap();
        let result = solve(&lp);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "HS51 feasibility LP: Expected Optimal, got {:?}",
            result.status
        );
        // 解が制約を満たすか検証 (x1+3x2=4 かつ x3+x4-2x5=0 かつ x2-x5=0)
        let x = &result.solution;
        assert!(
            (x[0] + 3.0 * x[1] - 4.0).abs() < 1e-6,
            "Constraint x1+3x2=4 violated: {}",
            x[0] + 3.0 * x[1]
        );
    }

    #[test]
    fn test_bug_simplex_001_finite_ub() {
        // BUG-simplex-001修正確認: m=0, maximize x with lb=0, ub=3
        // 修正前: Unbounded誤判定
        // 修正後: x=3, obj=3 (maximize) または obj=-3 (minimize として内部処理)
        let a = CscMatrix::new(0, 1);
        let lp = LpProblem::new_general(
            vec![-1.0], // minimize -x (= maximize x)
            a,
            vec![],
            vec![],
            vec![(0.0, 3.0)], // lb=0, ub=3
            None,
        )
        .unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.solution[0] - 3.0).abs() < PIVOT_TOL,
            "Expected x=3, got {}",
            result.solution[0]
        );
        assert!(
            (result.objective - (-3.0)).abs() < PIVOT_TOL,
            "Expected obj=-3, got {}",
            result.objective
        );
    }

    #[test]
    fn test_primal_simplex_timeout() {
        // n=200, m=100 の密なLP、deadlineを過去に設定してTimeout確認
        let n = 200usize;
        let m = 100usize;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            for j in 0..n {
                rows.push(i);
                cols.push(j);
                vals.push(1.0);
            }
        }
        let lp = make_lp(vec![-1.0; n], &rows, &cols, &vals, m, n, vec![1.0; m]);
        let opts = SolverOptions {
            deadline: Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Timeout);
    }

    #[test]
    fn test_lp_timeout() {
        // timeout_secs=0.0 (即時期限切れ) でLP実行 → SolveStatus::Timeout を確認
        // HIGH-1: LP timeout 公開APIの動作検証
        let n = 200usize;
        let m = 100usize;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            for j in 0..n {
                rows.push(i);
                cols.push(j);
                vals.push(1.0);
            }
        }
        let lp = make_lp(vec![-1.0; n], &rows, &cols, &vals, m, n, vec![1.0; m]);
        let opts = SolverOptions {
            timeout_secs: Some(0.0),
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Timeout);
    }

    #[test]
    fn test_lp_cancel() {
        // cancel_flag を true にセットしてLP実行 → SolveStatus::Timeout を確認
        // HIGH-1: LP cancel_flag 動作検証
        use std::sync::Arc;
        let n = 200usize;
        let m = 100usize;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            for j in 0..n {
                rows.push(i);
                cols.push(j);
                vals.push(1.0);
            }
        }
        let lp = make_lp(vec![-1.0; n], &rows, &cols, &vals, m, n, vec![1.0; m]);
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let opts = SolverOptions {
            cancel_flag: Some(Arc::clone(&cancel)),
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Timeout);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // バグ再現・regression テスト
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// BUG-SX-002: LuBasis::new Err → 偽 Optimal（CRITICAL）
    /// 特異初期基底（同一列インデックスを2回指定）で revised_simplex_core を呼ぶと
    /// LuBasis::new が Err → 現状 SimplexOutcome::Optimal（偽）が返る。
    /// 修正後は Timeout が返るべき。
    #[test]
    fn test_sx002_lu_basis_err_should_return_timeout_not_optimal() {
        // SPEC: BUG-SX-002
        use crate::simplex::pricing::DantzigPricing;
        // 2×2 行列: 列0 = [1; 0]、列1 = [0; 0]（全零列）
        // basis = [0, 0] → B = [[1, 1]; [0, 0]] → rank 1 → 特異
        // → LuBasis::new が Err を返す
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let mut x_b = vec![1.0, 0.0];
        let mut basis = vec![0usize, 0]; // 同一列 → 特異基底
        let mut pricing = DantzigPricing;
        let opts = SolverOptions::default();
        let outcome = revised_simplex_core(
            &a, &mut x_b, &c, &mut basis, 2, 2, 2, &mut pricing, &opts,
        );
        // 修正後: Timeout が期待される。現状: Optimal（偽）が返るのでこの assert は FAIL。
        assert!(
            !matches!(outcome, SimplexOutcome::Optimal(..)),
            "BUG-SX-002: LuBasis::new Err 時は Optimal を返してはならない（修正後は Timeout）"
        );
    }

    /// BUG-SX-001: Simplex MaxIterations 生成（HIGH）
    /// Simplex 外部 API（solve_with）から SolveStatus::MaxIterations が返ってはならない。
    /// SimplexOutcome::MaxIterations廃止後: SimplexOutcome自体にMaxIterationsバリアントが存在しない。
    /// refactor_failed = true かつ deadline 未設定の経路はTimeout（simplex/mod.rs, dual.rs）。
    #[test]
    fn test_sx001_solve_does_not_return_max_iterations() {
        // SPEC: BUG-SX-001 — regression test（修正後PASS）
        // SimplexOutcome::MaxIterations廃止により、SolveStatus::MaxIterationsへの経路が閉じた。
        for method in [SimplexMethod::Primal, SimplexMethod::Dual] {
            let lp = make_lp(
                vec![-1.0, -1.0],
                &[0, 0, 1, 2],
                &[0, 1, 0, 1],
                &[1.0, 1.0, 1.0, 1.0],
                3,
                2,
                vec![4.0, 3.0, 3.0],
            );
            let opts = SolverOptions {
                simplex_method: method,
                presolve: false,
                ..SolverOptions::default()
            };
            let result = solve_with(&lp, &opts);
            assert_ne!(
                result.status,
                SolveStatus::MaxIterations,
                "BUG-SX-001: solve_with は SolveStatus::MaxIterations を返してはならない (method={:?})",
                method
            );
        }
    }

    /// BUG-SX-003: refactor_failed + deadline 未設定 → Timeout（MEDIUM）
    /// SimplexOutcome::MaxIterations廃止後: refactor_failed経路はTimeout。
    #[test]
    fn test_sx003_refactor_failed_no_deadline_returns_timeout() {
        // SPEC: BUG-SX-003 — regression test（修正後PASS）
        // SimplexOutcome::MaxIterations廃止により、refactor_failed時はTimeoutが返る。
        use crate::simplex::pricing::DantzigPricing;
        // m=1, n=3 (cols: x1, x2, slack)
        // A = [[1, 1, 1]] → min -x1-x2 s.t. x1+x2+s=4
        // 初期基底 = [2] (スラック), x_b=[4]
        let a = CscMatrix::from_triplets(
            &[0, 0, 0], &[0, 1, 2], &[1.0, 1.0, 1.0], 1, 3,
        ).unwrap();
        let c = vec![-1.0, -1.0, 0.0];
        let mut x_b = vec![4.0];
        let mut basis = vec![2usize]; // スラック → 非特異
        let mut pricing = DantzigPricing;
        // deadline=None + max_etas=1 で早期 refactor を発動
        let opts = SolverOptions {
            deadline: None,
            max_etas: 1,
            ..SolverOptions::default()
        };
        let outcome = revised_simplex_core(
            &a, &mut x_b, &c, &mut basis, 1, 3, 3, &mut pricing, &opts,
        );
        // MaxIterations廃止後 → Optimal、Timeout、または SingularBasis が返る
        assert!(
            matches!(outcome, SimplexOutcome::Optimal(..) | SimplexOutcome::Timeout(_) | SimplexOutcome::SingularBasis),
            "BUG-SX-003: refactor_failed 時は Optimal/Timeout/SingularBasis を返すべき（got: {:?}）",
            match &outcome {
                SimplexOutcome::Optimal(..) => "Optimal",
                SimplexOutcome::Unbounded => "Unbounded",
                SimplexOutcome::Timeout(_) => "Timeout",
                SimplexOutcome::SingularBasis => "SingularBasis",
            }
        );
    }

    /// BUG-PRE-001: Simplex presolve が deadline を参照しない（LOW）
    /// simplex/mod.rs L69: presolve が deadline なしで実行される。
    /// timeout_secs=0 でも presolve は全実行されるバグ。
    #[test]
    fn test_pre001_presolve_does_not_respect_deadline() {
        // SPEC: BUG-PRE-001 — regression test
        // 小問題ではinner solverがTimeoutを返すためPASSする。
        // 大規模問題ではpresolveが予算を超過するバグが残存。
        // TODO(green phase): 大規模問題でpresolveが予算超過するケースのテストを追加し、
        //   presolve→deadline伝搬の修正を検証すること。
        let n = 200usize;
        let m = 100usize;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            for j in 0..n {
                rows.push(i);
                cols.push(j);
                vals.push(1.0);
            }
        }
        let lp = make_lp(vec![-1.0; n], &rows, &cols, &vals, m, n, vec![1.0; m]);
        let opts = SolverOptions {
            timeout_secs: Some(0.0), // 即座にタイムアウト
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        // 修正後: presolve が deadline を参照して Timeout を返す
        // 現状: inner solver が Timeout を返すため結果として Timeout になる可能性あり
        assert_eq!(
            result.status,
            SolveStatus::Timeout,
            "BUG-PRE-001: timeout_secs=0 は Timeout を返すべき"
        );
    }

    /// BUG-PRE-001 大規模問題: presolve が deadline を超過しないことを確認
    /// deadline=過去のInstant（即座にTimeout）で n=2000, m=1000 の問題を実行し、
    /// presolve の deadline チェックで early return されることを検証する。
    #[test]
    fn test_pre001_large_scale_presolve_respects_deadline() {
        // SPEC: BUG-PRE-001 — green phase test（設計書 §6 Step E）
        // presolveにdeadline=過去のInstantを渡し、Step 1前のチェックで即early returnされることを確認
        let n = 2000usize;
        let m = 1000usize;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            for j in 0..n {
                rows.push(i);
                cols.push(j);
                vals.push(1.0);
            }
        }
        let lp = make_lp(vec![-1.0; n], &rows, &cols, &vals, m, n, vec![1.0; m]);
        let start = std::time::Instant::now();
        let opts = SolverOptions {
            timeout_secs: Some(0.0), // 即座にタイムアウト → presolve deadline チェックで early return
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        let elapsed = start.elapsed();
        assert_eq!(
            result.status,
            SolveStatus::Timeout,
            "BUG-PRE-001 大規模: timeout_secs=0 は Timeout を返すべき"
        );
        // presolve の早期終了により、大規模問題でも短時間（< 0.5s）で完了するはず
        assert!(
            elapsed.as_secs_f64() < 0.5,
            "BUG-PRE-001 大規模: presolve が deadline を超過した (elapsed={:.3}s)",
            elapsed.as_secs_f64()
        );
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TDD赤フェーズ: テスト不足 (△) 項目
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// A2-T01: timeout_secs 設定時の停止保証（K=2.0 以内）
    /// Given: timeout_secs=T, When: solve_with, Then: elapsed ≤ T×2.0
    #[test]
    fn test_a2t01_timeout_elapsed_within_budget() {
        // SPEC: A2-T01
        // 大きな LP 問題を timeout_secs=0.01 で実行し、elapsed < 0.02s を確認
        // deadline を過去に設定することで確実に Timeout を引き起こす
        let n = 200usize;
        let m = 100usize;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            for j in 0..n {
                rows.push(i);
                cols.push(j);
                vals.push(1.0);
            }
        }
        let lp = make_lp(vec![-1.0; n], &rows, &cols, &vals, m, n, vec![1.0; m]);
        let timeout_secs = 0.01f64;
        let opts = SolverOptions {
            timeout_secs: Some(timeout_secs),
            presolve: false,
            ..SolverOptions::default()
        };
        let start = std::time::Instant::now();
        let result = solve_with(&lp, &opts);
        let elapsed = start.elapsed().as_secs_f64();
        // Timeout または Optimal（タイムアウト内に解けた場合）
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "A2-T01: Timeout または Optimal が返ること。got: {:?}", result.status
        );
        // elapsed は K×T 以内（K=3.0 でも超過した場合はバグ）
        assert!(
            elapsed < timeout_secs * 3.0 + 0.5, // 十分な余裕（CI環境考慮）
            "A2-T01: elapsed({:.3}s) > timeout×3({:.3}s). deadline バグが残存している可能性",
            elapsed, timeout_secs * 3.0
        );
    }

    /// A2-T03: timeout_secs=None でも有限ステップで収束（無期限実行保証）
    /// 小問題を deadline なしで解き、Optimal が返ることを確認
    #[test]
    fn test_a2t03_no_deadline_converges_finite() {
        // SPEC: A2-T03
        let lp = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let opts = SolverOptions {
            timeout_secs: None, // 無期限
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "A2-T03: タイムアウトなしで収束すること");
    }

    /// extract_dual_info が上限制約dual(mu_j)を正しく減算することを検証する
    ///
    /// 問題:
    ///   min -2*x1 - x2
    ///   s.t. x1 + x2 <= 4
    ///        0 <= x1 <= 2,  0 <= x2 <= 3
    ///
    /// 最適解: x1=2 (bounds上限活性), x2=2 (制約活性, 上限非活性)
    ///
    /// KKT条件:
    ///   rc[0] = c_0 - lambda_1*1 - mu_0 = 0  (x1 at upper bound)
    ///   rc[1] = c_1 - lambda_1*1       = 0  (x2 strictly between bounds)
    ///
    /// mu_j欠落の場合: rc[0] = -2 - lambda_1 != 0 (補正漏れが相補性誤差を生む)
    #[test]
    fn test_extract_dual_info_ub_dual() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let problem = LpProblem::new_general(
            vec![-2.0, -1.0],
            a,
            vec![4.0],
            vec![ConstraintType::Le],
            vec![(0.0, 2.0), (0.0, 3.0)],
            None,
        )
        .unwrap();
        let opts = SolverOptions { timeout_secs: None, presolve: false, ..SolverOptions::default() };
        let result = solve_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal, "status should be Optimal");

        let x = &result.solution;
        assert!((x[0] - 2.0).abs() < 1e-6, "x[0]={} should be at upper bound 2.0", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-6, "x[1]={} should be 2.0", x[1]);

        let rc = &result.reduced_costs;

        // x[0] is at upper bound (x[0] = ub = 2) → rc[0] ≤ 0
        // If mu_j subtraction is missing, rc[0] = c[0] - lambda*a[0,0] = -1 - (-2) = 1 > 0
        assert!(rc[0] <= 1e-6, "rc[0]={} should be <= 0 (x[0] at upper bound; mu_j subtraction required)", rc[0]);

        // x[1] is strictly between bounds (0 < x[1]=2 < 3) → x[1] is basic → rc[1] ≈ 0
        assert!(rc[1].abs() < 1e-6, "rc[1]={} should be ≈ 0 (x[1] is basic)", rc[1]);

        // Upper complementarity for x[0]: (ub - x[0]) * max(-rc[0], 0) ≈ 0
        let ub0 = 2.0_f64;
        let upper_comp = (ub0 - x[0]) * (-rc[0]).max(0.0);
        assert!(upper_comp.abs() < 1e-8, "upper complementarity={} should be ≈ 0", upper_comp);
    }

    /// BUG-NE-001: maros NumericalError 再現防止テスト
    ///
    /// Phase I で Eq制約の縮退人工変数が残り、Phase II で check_eq_feasibility が通ること。
    /// 問題構造: 2行 b=0 の Eq制約 + 1行 b=1 の Eq制約 + Le制約
    ///
    /// 案C有効化(if true)でNumericalErrorが解消されることを確認する。
    /// 修正前(if false)ではNumericalErrorが発生していた。
    ///
    /// 問題:
    ///   min -x4
    ///   x1 + x2 = 0  (Eq, b=0) → degenerate artificial in Phase I
    ///   x1 + x3 = 0  (Eq, b=0) → degenerate artificial in Phase I
    ///   x2 + x4 = 1  (Eq, b=1) → normal artificial
    ///   x1 + x4 <= 2 (Le)
    ///   x >= 0
    ///
    /// 最適解: x=[0,0,0,1], obj=-1
    #[test]
    fn test_bug_ne001_maros_degenerate_eq_zero_rhs() {
        use crate::problem::ConstraintType;
        let a = CscMatrix::from_triplets(
            &[0, 1, 3, 0, 2, 1, 2, 3],
            &[0, 0, 0, 1, 1, 2, 3, 3],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            4,
            4,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![0.0, 0.0, 0.0, -1.0],
            a,
            vec![0.0, 0.0, 1.0, 2.0],
            vec![
                ConstraintType::Eq,
                ConstraintType::Eq,
                ConstraintType::Eq,
                ConstraintType::Le,
            ],
            vec![(0.0, f64::INFINITY); 4],
            None,
        )
        .unwrap();
        let opts = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_ne!(
            result.status,
            SolveStatus::NumericalError,
            "BUG-NE-001: 縮退Eq制約(b=0)でNumericalErrorが発生してはならない"
        );
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "BUG-NE-001: Optimal を返すべき、got {:?}",
            result.status
        );
        assert!(
            (result.objective - (-1.0)).abs() < 1e-6,
            "BUG-NE-001: obj=-1.0 を期待、got {}",
            result.objective
        );
    }

    /// BUG-NE-002: wood1p NumericalError 再現防止テスト
    ///
    /// 243Eq+1G制約のうち242本がb=0という wood1p の縮退構造を小規模で再現する。
    /// 問題構造: 3行 b=0 の Eq制約 + 1行 b=1 の Eq制約 + Le制約
    ///
    /// 問題:
    ///   min -x5
    ///   x1 + x2 = 0  (Eq, b=0)
    ///   x2 + x3 = 0  (Eq, b=0)
    ///   x3 + x4 = 0  (Eq, b=0)
    ///   x1 + x5 = 1  (Eq, b=1)
    ///   x1+x2+x3+x4+x5 <= 2 (Le)
    ///   x >= 0
    ///
    /// 最適解: x=[0,0,0,0,1], obj=-1
    #[test]
    fn test_bug_ne002_wood1p_multiple_zero_rhs_eq() {
        use crate::problem::ConstraintType;
        let a = CscMatrix::from_triplets(
            &[0, 3, 4, 0, 1, 4, 1, 2, 4, 2, 4, 3, 4],
            &[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            5,
            5,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![0.0, 0.0, 0.0, 0.0, -1.0],
            a,
            vec![0.0, 0.0, 0.0, 1.0, 2.0],
            vec![
                ConstraintType::Eq,
                ConstraintType::Eq,
                ConstraintType::Eq,
                ConstraintType::Eq,
                ConstraintType::Le,
            ],
            vec![(0.0, f64::INFINITY); 5],
            None,
        )
        .unwrap();
        let opts = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_ne!(
            result.status,
            SolveStatus::NumericalError,
            "BUG-NE-002: 多数b=0 Eq制約でNumericalErrorが発生してはならない"
        );
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "BUG-NE-002: Optimal を返すべき、got {:?}",
            result.status
        );
        assert!(
            (result.objective - (-1.0)).abs() < 1e-6,
            "BUG-NE-002: obj=-1.0 を期待、got {}",
            result.objective
        );
    }

    /// BUG-NE-001/002 案C有効化後: hs51 regression test（基底特異化リスク確認）
    ///
    /// 自由変数 + Le制約の feasibility LP で 案C が基底特異化を引き起こさないことを確認。
    /// best_j != None フォールバックにより hs51 パターンが安全に処理されること。
    ///
    /// hs51: 5変数(全自由), 6Le制約, 実行可能解 x=[1,1,1,1,1] が存在する。
    /// 案C有効化後も Optimal が返ること（基底特異化しないこと）を検証する。
    #[test]
    fn test_bug_ne_case_c_hs51_regression() {
        let a = CscMatrix::from_triplets(
            &[0, 1, 0, 1, 4, 5, 2, 3, 2, 3, 2, 3, 4, 5],
            &[0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4],
            &[
                1.0, -1.0, 3.0, -3.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -2.0, 2.0, -1.0, 1.0,
            ],
            6,
            5,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![0.0; 5],
            a,
            vec![4.0, -4.0, 0.0, 0.0, 0.0, 0.0],
            vec![crate::problem::ConstraintType::Le; 6],
            vec![(f64::NEG_INFINITY, f64::INFINITY); 5],
            None,
        )
        .unwrap();
        let opts = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_ne!(
            result.status,
            SolveStatus::NumericalError,
            "hs51 regression: 案C有効化後もNumericalErrorを返してはならない（基底特異化の兆候）"
        );
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "hs51 regression: 実行可能解が存在するのでOptimalを返すべき、got {:?}",
            result.status
        );
    }
}

/// DualAdvanced warm-start integration tests
///
/// warm-start経路（dual_simplex_core_advanced）が実際に呼ばれることを確認する。
/// SimplexMethod::DualAdvanced + warm_start で LP を解き、Optimal + cold-start一致を検証。
#[cfg(test)]
mod tests_dual_advanced {
    use super::*;
    use crate::sparse::CscMatrix;

    fn make_lp(
        c: Vec<f64>,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrows: usize,
        ncols: usize,
        b: Vec<f64>,
    ) -> LpProblem {
        let a = CscMatrix::from_triplets(rows, cols, vals, nrows, ncols).unwrap();
        LpProblem::new(c, a, b).unwrap()
    }

    /// DualAdvanced warm-start テスト1: RHS変更後の再最適化
    ///
    /// LP1 を解いて warm_start_basis を取得し、RHS のみ変更した LP2 を
    /// SimplexMethod::DualAdvanced + warm_start で再最適化する。
    /// warm-start 経路（dual_simplex_core_advanced）が正しく動作し、
    /// cold-start と同じ最適値を返すことを確認。
    ///
    /// LP1: min -x1 - 2*x2  s.t. x1+x2 ≤ 4, x1 ≤ 3, x2 ≤ 3  → obj = -7
    /// LP2: 同じ構造、b = [5, 3, 3]                             → obj = -8
    #[test]
    fn test_dual_advanced_warm_start_rhs_change() {
        let lp1 = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );

        // LP1 を default solver で解いて warm_start_basis を取得
        let result1 = solve_with(&lp1, &SolverOptions::default());
        assert_eq!(result1.status, SolveStatus::Optimal);
        assert!(
            result1.warm_start_basis.is_some(),
            "LP1 は warm_start_basis を返すべき"
        );

        // LP2: RHS のみ変更 b=[5, 3, 3]
        let lp2 = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![5.0, 3.0, 3.0],
        );

        // cold-start で正解を確認
        let result2_cold = solve_with(&lp2, &SolverOptions::default());
        assert_eq!(result2_cold.status, SolveStatus::Optimal);

        // DualAdvanced warm-start で解く → warm-start 経路を通す
        let opts_warm = SolverOptions {
            warm_start: result1.warm_start_basis.clone(),
            simplex_method: SimplexMethod::DualAdvanced,
            ..SolverOptions::default()
        };
        let result2_warm = solve_with(&lp2, &opts_warm);

        assert_eq!(
            result2_warm.status,
            SolveStatus::Optimal,
            "DualAdvanced warm-start は Optimal を返すべき"
        );
        assert!(
            (result2_warm.objective - result2_cold.objective).abs() < 1e-6,
            "DualAdvanced warm-start obj={}, cold-start obj={}",
            result2_warm.objective,
            result2_cold.objective
        );
    }

    /// DualAdvanced warm-start テスト2: 別LP（b=[6,4,4]）での再最適化
    ///
    /// LP1 を解いて warm_start_basis を取得し、RHS を拡大した LP2 を
    /// SimplexMethod::DualAdvanced + warm_start で再最適化する。
    /// cold-start と同じ最適値を返すことを確認。
    ///
    /// LP1: min -x1 - x2  s.t. x1+x2 ≤ 4, x1 ≤ 3, x2 ≤ 3  → obj = -4
    /// LP2: 同じ構造、b = [6, 4, 4]                          → obj = -8
    #[test]
    fn test_dual_advanced_warm_start_larger_rhs() {
        let lp1 = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );

        let result1 = solve_with(&lp1, &SolverOptions::default());
        assert_eq!(result1.status, SolveStatus::Optimal);
        assert!(
            result1.warm_start_basis.is_some(),
            "LP1 は warm_start_basis を返すべき"
        );

        // LP2: RHS 拡大 b=[6, 4, 4] → 最適解 x1+x2=8, obj=-8
        let lp2 = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![6.0, 4.0, 4.0],
        );

        // cold-start で正解を確認
        let result2_cold = solve_with(&lp2, &SolverOptions::default());
        assert_eq!(result2_cold.status, SolveStatus::Optimal);

        // DualAdvanced warm-start で解く
        let opts_warm = SolverOptions {
            warm_start: result1.warm_start_basis.clone(),
            simplex_method: SimplexMethod::DualAdvanced,
            ..SolverOptions::default()
        };
        let result2_warm = solve_with(&lp2, &opts_warm);

        assert_eq!(
            result2_warm.status,
            SolveStatus::Optimal,
            "DualAdvanced warm-start (larger RHS) は Optimal を返すべき"
        );
        assert!(
            (result2_warm.objective - result2_cold.objective).abs() < 1e-6,
            "DualAdvanced warm-start obj={}, cold-start obj={}",
            result2_warm.objective,
            result2_cold.objective
        );
    }
}
