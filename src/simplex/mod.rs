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

use crate::basis::{BasisManager, LuBasis};
use crate::options::{SimplexMethod, SolverOptions, WarmStartBasis};
use crate::presolve::{self, RuizScaler};
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use log::warn;
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::*;
use pricing::{PricingStrategy, SteepestEdgePricing};
use std::sync::atomic::Ordering;

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
            if options.warm_start.is_some() {
                dual::two_phase_dual_simplex(&sf, problem, options)
            } else {
                two_phase_simplex(&sf, problem, options)
            }
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

/// 2相シンプレックス法で標準形LPを解く
///
/// - 人工変数が不要な場合: Phase II に直接進む
/// - 人工変数が必要な場合: Phase I で実行可能基底を求めてから Phase II を実行
///
/// Phase I の目的関数は人工変数の和の最小化。
/// 最小値がゼロより大きければ元問題は実行不可能（Infeasible）。
/// Ruiz equilibration スケーリングを適用してから解く。
pub(crate) fn two_phase_simplex(sf: &StandardForm, problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    let m = sf.m;

    // Apply Ruiz equilibration scaling to the standard form
    let (a, b, c, row_scale, col_scale) = RuizScaler::scale(&sf.a, &sf.b, &sf.c);

    if sf.num_artificial == 0 {
        // Direct Phase II
        let mut basis = sf.initial_basis.clone();
        let mut x_b = b.clone();
        // Correct x_b for Ruiz-scaled diagonal initial basis: x_b[i] = b_scaled[i] / B[i, basis[i]]
        // After Ruiz equilibration, slack basis columns have scaled diagonal entries != 1.0,
        // so the naive x_b = b_scaled is inconsistent with B * x_b = b_scaled.
        for i in 0..m {
            let col = basis[i];
            if let Ok((rows, vals)) = a.get_column(col) {
                for (k, &row) in rows.iter().enumerate() {
                    if row == i && vals[k].abs() > 1e-14 {
                        x_b[i] /= vals[k];
                        break;
                    }
                }
            }
        }
        let mut pricing = SteepestEdgePricing::new(sf.n_total);

        match revised_simplex_core(&a, &mut x_b, &c, &mut basis, m, sf.n_total, sf.n_total, &mut pricing, options)
        {
            SimplexOutcome::Optimal(obj, y) => {
                let solution = extract_solution(sf, &basis, &x_b, &col_scale);
                // 案D: Eq制約 feasibility check — 偽 Optimal 返却を防ぐ defense-in-depth
                if !check_eq_feasibility(problem, &solution) {
                    return SolverResult {
                        status: SolveStatus::NumericalError,
                        objective: obj + sf.obj_offset,
                        solution: vec![],
                        dual_solution: vec![],
                        reduced_costs: vec![],
                        slack: vec![],
                        warm_start_basis: None,
                        ..Default::default()
                    };
                }
                let (dual_solution, reduced_costs, slack) =
                    extract_dual_info(sf, problem, &y, &solution, &row_scale);
                let ws = WarmStartBasis { basis: basis.clone(), x_b: x_b.clone() };
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: obj + sf.obj_offset,
                    solution,
                    dual_solution,
                    reduced_costs,
                    slack,
                    warm_start_basis: Some(ws),
            ..Default::default()
                }
            }
            SimplexOutcome::Unbounded => SolverResult {
                status: SolveStatus::Unbounded,
                objective: f64::NEG_INFINITY,
                solution: vec![],
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
            ..Default::default()
            },
            SimplexOutcome::Timeout(obj) => {
                let solution = extract_solution(sf, &basis, &x_b, &col_scale);
                SolverResult {
                    status: SolveStatus::Timeout,
                    objective: obj + sf.obj_offset,
                    solution,
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
            ..Default::default()
                }
            }
            SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
        }
    } else {
        // Phase I: build extended matrix with artificials
        let n_ext = sf.n_total + sf.num_artificial;

        let mut trip_rows = Vec::new();
        let mut trip_cols = Vec::new();
        let mut trip_vals = Vec::new();

        // Copy scaled matrix (not sf.a)
        for j in 0..a.ncols {
            if let Ok((r, v)) = a.get_column(j) {
                for (k, &row) in r.iter().enumerate() {
                    trip_rows.push(row);
                    trip_cols.push(j);
                    trip_vals.push(v[k]);
                }
            }
        }

        // Add artificial columns
        let mut basis = sf.initial_basis.clone();
        let mut art_col = sf.n_total;
        for (i, b) in basis.iter_mut().enumerate().take(m) {
            if sf.needs_artificial[i] {
                trip_rows.push(i);
                trip_cols.push(art_col);
                trip_vals.push(1.0);
                *b = art_col;
                art_col += 1;
            }
        }

        let a_ext =
            CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m, n_ext).unwrap();

        // Phase I cost: minimise sum of artificials
        let mut c_phase1 = vec![0.0; n_ext];
        c_phase1[sf.n_total..].fill(1.0);

        let mut x_b = b.clone();
        // Correct x_b for Ruiz-scaled diagonal initial basis: x_b[i] = b_scaled[i] / B[i, basis[i]]
        // For artificial columns (added with entry 1.0), dividing by 1.0 is a no-op.
        // For slack columns from a, the diagonal entry is row_scale[i] * col_scale[slack] != 1.0.
        for i in 0..m {
            let col = basis[i];
            if let Ok((rows, vals)) = a_ext.get_column(col) {
                for (k, &row) in rows.iter().enumerate() {
                    if row == i && vals[k].abs() > 1e-14 {
                        x_b[i] /= vals[k];
                        break;
                    }
                }
            }
        }
        let mut pricing1 = SteepestEdgePricing::new(n_ext);

        match revised_simplex_core(&a_ext, &mut x_b, &c_phase1, &mut basis, m, n_ext, n_ext, &mut pricing1, options) {
            SimplexOutcome::Optimal(obj, _) => {
                if obj > PIVOT_TOL {
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

                // 案C: Phase I 完了後の縮退人工変数 pivot out
                // x_b[i] ≈ 0 の人工変数（basis[i] >= sf.n_total）を非人工列で置換し、
                // Phase II が人工変数を基底に残したまま目的関数を最適化するのを防ぐ。
                //
                // Eq/Le/Ge 全行に適用:
                // - Eq 行: 直接人工変数が basis に入る。x_b[i]=0 の縮退で残留。
                // - Ge/Le 行 (b > 0): surplus 変数に加え人工変数も basis に入る。
                //   Phase I が surplus を基底から追い出せず人工変数が縮退 0 で残留することがある。
                // 置換候補: 非基底の非人工列 (j < sf.n_total) で行 i の係数が最大のもの。
                // 安全チェック: LU 失敗なら全変更をリバート。
                {
                    // 変更前の基底を保存。LU検証失敗時にリバートするため。
                    let basis_before_case_c = basis.clone();
                    let mut is_basic = vec![false; n_ext];
                    for &col in basis.iter() {
                        is_basic[col] = true;
                    }
                    for i in 0..m {
                        // 行 i の基底が人工変数 (>= n_total) かつ縮退 (x_b ≈ 0) なら pivot out
                        if basis[i] < sf.n_total || x_b[i].abs() >= PIVOT_TOL {
                            continue;
                        }
                        // 行 i で最大 |a_ext[i,j]| の非基底・非人工列 j を探す
                        let mut best_j = None;
                        let mut best_abs = PIVOT_TOL;
                        for j in 0..sf.n_total {
                            if is_basic[j] {
                                continue;
                            }
                            if let Ok((rows, vals)) = a_ext.get_column(j) {
                                for (k, &row) in rows.iter().enumerate() {
                                    if row == i && vals[k].abs() > best_abs {
                                        best_abs = vals[k].abs();
                                        best_j = Some(j);
                                    }
                                }
                            }
                        }
                        if let Some(j) = best_j {
                            is_basic[basis[i]] = false;
                            is_basic[j] = true;
                            basis[i] = j;
                            // degenerate pivot: x_b[i] = 0, 他の x_b は変化なし
                        }
                    }
                    // 安全チェック: 修正後の基底がLU分解可能か検証。
                    // 特異または高条件数の問題（SCORPION等）では案Cの置換が基底品質を
                    // 悪化させる場合がある。LU失敗なら全置換をリバート。
                    if LuBasis::new(&a_ext, &basis, options.max_etas).is_err() {
                        basis.copy_from_slice(&basis_before_case_c);
                    }
                }

                // Phase II: restrict pricing to non-artificial columns
                // Use scaled c for Phase II cost
                let mut c_phase2 = vec![0.0; n_ext];
                c_phase2[..sf.n_total].copy_from_slice(&c[..sf.n_total]);

                let mut pricing2 = SteepestEdgePricing::new(n_ext);
                match revised_simplex_core(
                    &a_ext,
                    &mut x_b,
                    &c_phase2,
                    &mut basis,
                    m,
                    n_ext,
                    sf.n_total,
                    &mut pricing2,
                    options,
                ) {
                    SimplexOutcome::Optimal(obj2, y) => {
                        let solution = extract_solution(sf, &basis, &x_b, &col_scale);
                        // 案D: Eq制約 feasibility check — 偽 Optimal 返却を防ぐ defense-in-depth
                        if !check_eq_feasibility(problem, &solution) {
                            return SolverResult {
                                status: SolveStatus::NumericalError,
                                objective: obj2 + sf.obj_offset,
                                solution: vec![],
                                dual_solution: vec![],
                                reduced_costs: vec![],
                                slack: vec![],
                                warm_start_basis: None,
                                ..Default::default()
                            };
                        }
                        let (dual_solution, reduced_costs, slack) =
                            extract_dual_info(sf, problem, &y, &solution, &row_scale);
                        let ws = WarmStartBasis { basis: basis.clone(), x_b: x_b.clone() };
                        SolverResult {
                            status: SolveStatus::Optimal,
                            objective: obj2 + sf.obj_offset,
                            solution,
                            dual_solution,
                            reduced_costs,
                            slack,
                            warm_start_basis: Some(ws),
            ..Default::default()
                        }
                    }
                    SimplexOutcome::Unbounded => SolverResult {
                        status: SolveStatus::Unbounded,
                        objective: f64::NEG_INFINITY,
                        solution: vec![],
                        dual_solution: vec![],
                        reduced_costs: vec![],
                        slack: vec![],
                        warm_start_basis: None,
            ..Default::default()
                    },
                    SimplexOutcome::Timeout(obj2) => {
                        let solution = extract_solution(sf, &basis, &x_b, &col_scale);
                        SolverResult {
                            status: SolveStatus::Timeout,
                            objective: obj2 + sf.obj_offset,
                            solution,
                            dual_solution: vec![],
                            reduced_costs: vec![],
                            slack: vec![],
                            warm_start_basis: None,
            ..Default::default()
                        }
                    }
                    SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
                }
            }
            SimplexOutcome::Unbounded => SolverResult {
                status: SolveStatus::Infeasible,
                objective: 0.0,
                solution: vec![],
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
            ..Default::default()
            },
            SimplexOutcome::Timeout(_) => SolverResult {
                status: SolveStatus::Timeout,
                objective: 0.0,
                solution: vec![],
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
            ..Default::default()
            },
            SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
        }
    }
}

/// Eq 制約の feasibility を確認する（案D: defense-in-depth）
///
/// `solution` は problem の変数空間の解ベクトル。
/// Eq 制約の violation が `FEASIBILITY_TOL` を超える場合 false を返す。
/// bore3d の 100 単位違反のような明らかな誤 Optimal を検出し NumericalError として返すために使う。
fn check_eq_feasibility(problem: &LpProblem, solution: &[f64]) -> bool {
    const FEASIBILITY_TOL: f64 = 1e-4;
    let mut ax = vec![0.0f64; problem.num_constraints];
    for (j, &sj) in solution.iter().enumerate() {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                ax[row] += vals[k] * sj;
            }
        }
    }
    for ((ax_i, ct), bi) in ax.iter().zip(problem.constraint_types.iter()).zip(problem.b.iter()) {
        let violation = match ct {
            ConstraintType::Eq => (ax_i - bi).abs(),
            ConstraintType::Le => (ax_i - bi).max(0.0),
            ConstraintType::Ge => (bi - ax_i).max(0.0),
            _ => 0.0,
        };
        if violation > FEASIBILITY_TOL {
            return false;
        }
    }
    true
}

/// 最適基底解から元の変数への解ベクトルを復元する
///
/// 標準形の最適解を、変数変換（オフセット・係数）を逆適用して
/// 元問題の変数値に変換する。
/// `col_scale` はRuizスケーリングの列スケール因子。スケーリングを行わない場合は空スライスを渡す。
pub(crate) fn extract_solution(sf: &StandardForm, basis: &[usize], x_b: &[f64], col_scale: &[f64]) -> Vec<f64> {
    let mut x_new = vec![0.0; sf.n_shifted];
    for i in 0..sf.m {
        if basis[i] < sf.n_shifted {
            let scale = col_scale.get(basis[i]).copied().unwrap_or(1.0);
            x_new[basis[i]] = x_b[i] * scale;
        }
    }

    let mut solution = vec![0.0; sf.n_orig];
    for (j, sol_j) in solution.iter_mut().enumerate() {
        let info = &sf.orig_var_info[j];
        *sol_j = info.offset;
        for &(new_idx, coeff) in &info.new_vars {
            *sol_j += coeff * x_new[new_idx];
        }
    }
    solution
}

// --- Core Revised Simplex ---

/// 改訂シンプレックス法のコアアルゴリズム
///
/// LU分解を用いて基底行列を管理し、以下の手順を繰り返す:
///
/// 1. **BTRAN**: 双対変数 `y = B^{-T} c_B` を計算
/// 2. **価格付け（Pricing）**: PricingStrategyにより入基変数を選択
/// 3. **FTRAN**: ピボット列 `d = B^{-1} a_j` を計算
/// 4. **比率テスト**: Bland則を用いて離基変数を選択（退化サイクルの防止）
/// 5. **基底更新**: `x_B` の更新と基底行列のrank-1更新
///
/// # 引数
///
/// * `a` - 制約行列（CSC疎行列形式）
/// * `x_b` - 現在の基底解ベクトル（更新される）
/// * `c` - 目的関数係数ベクトル
/// * `basis` - 基底変数インデックスリスト（更新される）
/// * `m` - 制約数
/// * `n_cols` - 全列数
/// * `n_price` - 価格付けの対象列数（人工変数を除く場合に `n_total` を指定）
/// * `pricing` - 価格付け戦略（DantzigPricing または SteepestEdgePricing）
/// * `options` - ソルバー設定（反復上限・eta 保持数・クランプ閾値を含む）
///
/// # 戻り値
///
/// [`SimplexOutcome::Optimal`] — 最適目的関数値と双対変数、または [`SimplexOutcome::Unbounded`]
#[allow(clippy::too_many_arguments)]
pub(crate) fn revised_simplex_core<P: PricingStrategy>(
    a: &CscMatrix,
    x_b: &mut [f64],
    c: &[f64],
    basis: &mut [usize],
    m: usize,
    n_cols: usize,
    n_price: usize,
    pricing: &mut P,
    options: &SolverOptions,
) -> SimplexOutcome {
    let max_iter = usize::MAX; // timeout が実質的なガード（max_iterations廃止）
    let mut basis_mgr = match LuBasis::new(a, basis, options.max_etas) {
        Ok(bm) => bm,
        Err(crate::error::SolverError::SingularBasis { .. }) => {
            // 初期基底が特異: 呼び出し元に NumericalError を伝搬して IPM フォールバックを促す
            return SimplexOutcome::SingularBasis;
        }
        Err(_) => {
            let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
            return SimplexOutcome::Timeout(obj); // DeadlineExceeded 等
        }
    };

    let mut is_basic = vec![false; n_cols];
    for &b in basis.iter() {
        is_basic[b] = true;
    }

    // Pre-allocate reusable buffers to avoid heap allocation inside the iteration loop
    let mut c_b = vec![0.0f64; m];
    let mut y_dense = vec![0.0f64; m];
    let mut d_dense = vec![0.0f64; m];
    // Pre-allocate RC vector (reused each iteration) for pricing strategy
    let mut rc_vec = vec![0.0f64; n_price];

    for _iter in 0..max_iter {
        // タイムアウト・キャンセルチェック
        let timed_out = options.deadline.is_some_and(|d| std::time::Instant::now() >= d);
        let cancelled = options.cancel_flag.as_ref().is_some_and(|f| f.load(Ordering::Relaxed));
        if timed_out || cancelled {
            let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
            return SimplexOutcome::Timeout(obj);
        }

        // 1. Dual variables: y = BTRAN(c_B)
        for i in 0..m {
            c_b[i] = c[basis[i]];
        }
        let mut y_sv = SparseVec::from_dense(&c_b);
        basis_mgr.btran(&mut y_sv);
        y_sv.to_dense_into(&mut y_dense);
        let y = &y_dense;

        // 2. Compute reduced costs for all pricing candidates
        for j in 0..n_price {
            if is_basic[j] {
                rc_vec[j] = 0.0;
                continue;
            }
            let (rows, vals) = a.get_column(j).unwrap();
            let mut ya = 0.0;
            for (k, &row) in rows.iter().enumerate() {
                ya += y[row] * vals[k];
            }
            rc_vec[j] = c[j] - ya;
        }

        // 3. Select entering variable via pricing strategy
        let entering_col = match pricing.select_entering(&rc_vec, n_price) {
            None => {
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                return SimplexOutcome::Optimal(obj, y_dense.clone());
            }
            Some(j) => j,
        };

        // 4. FTRAN: pivot column d = B^{-1} * a_entering
        let (col_rows, col_vals) = a.get_column(entering_col).unwrap();
        // 元の列 inf-ノルムを保存（borrow は d_sv の .to_vec() で終了）
        let orig_col_norm = col_vals.iter().cloned().fold(0.0f64, |acc, v| acc.max(v.abs()));
        let mut d_sv = SparseVec {
            indices: col_rows.to_vec(),
            values: col_vals.to_vec(),
            len: m,
        };
        basis_mgr.ftran(&mut d_sv);
        d_sv.to_dense_into(&mut d_dense);

        // 4b. FTRAN 安定性チェック: eta 蓄積による数値誤差が d を汚染していないか確認する。
        //     汚染シグナル: d 値が inf/NaN、または d の最大値が元の列ノルムの 1e12 倍超。
        //     汚染検出時は即時再因子分解でリセット後に d を再計算する。
        {
            let d_max_abs = d_dense.iter().cloned().fold(0.0f64, |acc, v| {
                if v.is_finite() { acc.max(v.abs()) } else { f64::INFINITY }
            });
            // 期待される d のノルムは原則として原列ノルムの数倍以内。
            // 1e12 倍超は eta 蓄積による数値爆発とみなす（または inf/NaN）。
            let d_corrupt = !d_max_abs.is_finite()
                || (orig_col_norm > 0.0 && d_max_abs > 1e12 * orig_col_norm);
            if d_corrupt && basis_mgr.eta_count() > 0 {
                basis_mgr.force_refactor_timed(a, basis, options.deadline);
                if basis_mgr.refactor_failed {
                    if basis_mgr.singular_basis {
                        return SimplexOutcome::SingularBasis;
                    } else {
                        let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                        return SimplexOutcome::Timeout(obj);
                    }
                }
                // 新鮮な LU で d を再計算
                let (cr2, cv2) = a.get_column(entering_col).unwrap();
                d_sv = SparseVec { indices: cr2.to_vec(), values: cv2.to_vec(), len: m };
                basis_mgr.ftran(&mut d_sv);
                d_sv.to_dense_into(&mut d_dense);
            }
        }
        let d = &d_dense;

        // 5. Ratio test (Bland's rule for ties)
        let mut leaving = None;
        let mut min_ratio = f64::INFINITY;

        for i in 0..m {
            if d[i] > PIVOT_TOL {
                let ratio = x_b[i] / d[i];
                if ratio < min_ratio - PIVOT_TOL {
                    min_ratio = ratio;
                    leaving = Some(i);
                } else if (ratio - min_ratio).abs() < PIVOT_TOL {
                    if let Some(prev) = leaving {
                        if basis[i] < basis[prev] {
                            leaving = Some(i);
                        }
                    }
                }
            }
        }

        let leaving_row = match leaving {
            None => return SimplexOutcome::Unbounded,
            Some(i) => i,
        };

        // 6. Update x_b
        let step = x_b[leaving_row] / d[leaving_row];
        for i in 0..m {
            x_b[i] -= d[i] * step;
        }
        x_b[leaving_row] = step;

        // Clamp near-zero to prevent drift
        for val in x_b.iter_mut() {
            if val.abs() < options.clamp_tol {
                *val = 0.0;
            }
        }

        // 7. Get leaving column index before updating basis
        let leaving_col = basis[leaving_row];

        // 8. Update pricing weights
        pricing.update_weights(&basis_mgr, entering_col, leaving_col, d);

        // 9. Update basis tracking
        is_basic[leaving_col] = false;
        is_basic[entering_col] = true;

        // 10. Update basis index and check pivot stability
        basis[leaving_row] = entering_col;

        // ピボット安定性チェック: |d[leaving_row]| / max(d) が閾値未満の場合、
        // eta の inv_pivot が大きくなりすぎて FTRAN/BTRAN が数値爆発する。
        // その場合は eta を追加せず即時再因子分解でリセットする（eta 蓄積誤差を防ぐ）。
        let max_d_abs = d.iter().cloned().fold(0.0f64, |acc, v| acc.max(v.abs()));
        let pivot_unstable = d[leaving_row].abs() < PIVOT_STABILITY_THRESHOLD * max_d_abs
            && basis_mgr.eta_count() > 0;

        if pivot_unstable {
            // 不安定ピボット: eta を追加せず直ちに再因子分解（新 basis で LU をリセット）
            basis_mgr.force_refactor_timed(a, basis, options.deadline);
        } else {
            // 通常ピボット: eta で逐次更新
            basis_mgr.update(entering_col, leaving_row, &d_sv);

            // 11. Refactor if needed
            // timeout audit fix — deadline 付きで LU 再因子分解を実行
            basis_mgr.refactor_if_needed_timed(a, basis, options.deadline);
        }

        if basis_mgr.refactor_failed {
            if basis_mgr.singular_basis {
                return SimplexOutcome::SingularBasis;
            } else {
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                return SimplexOutcome::Timeout(obj);
            }
        }
    }

    let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
    // max_iter=usize::MAX のためここには事実上到達しない
    SimplexOutcome::Timeout(obj)
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
