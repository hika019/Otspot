//! Active Set法 QPソルバー実装
//!
//! Phase I（初期実行可能点探索）と Phase II（Active Setメインループ）を実装する。
//! NC1修正済み KktSolver を使用する。

use crate::options::SolverOptions;
use crate::problem::{ConstraintType, LpProblem, SolveStatus};
use crate::qp::active_set::WorkingSet;
use crate::qp::kkt::{self, KktSolver};
use crate::qp::problem::{QpProblem, QpResult, QpWarmStart};
use crate::sparse::CscMatrix;
use crate::tolerances::*;
use crate::{qp::kkt::extract_active_rows, simplex};

/// QP求解の実装コア（Active Set法）
pub(crate) fn qp_solve_impl(
    problem: &QpProblem,
    warm_start: Option<&QpWarmStart>,
    options: &SolverOptions,
) -> QpResult {
    let n = problem.num_vars;

    // Q=0 の退化ケース（LP問題）: LP solverに委譲
    if problem.is_zero_q() {
        return solve_as_lp(problem, options);
    }

    // Phase I: 初期実行可能点の取得
    let initial_x = if let Some(ws) = warm_start {
        if let Some(ref x0) = ws.initial_point {
            if x0.len() == n {
                x0.clone()
            } else {
                match find_initial_feasible_point(problem, options) {
                    Some(x) => x,
                    None => return QpResult::infeasible(),
                }
            }
        } else {
            match find_initial_feasible_point(problem, options) {
                Some(x) => x,
                None => return QpResult::infeasible(),
            }
        }
    } else {
        match find_initial_feasible_point(problem, options) {
            Some(x) => x,
            None => return QpResult::infeasible(),
        }
    };

    // Phase II: Active Set メインループ
    // 初期working setは空から始める（等式制約の2不等式エンコード時の線形従属を防ぐため）
    // warm-startの場合は提供されたactive_setを使用するが、線形独立性が保証された集合を前提とする
    let initial_active = if let Some(ws) = warm_start {
        WorkingSet::from_indices(ws.initial_active_set.clone())
    } else {
        WorkingSet::from_indices(vec![])
    };

    active_set_loop(problem, initial_x, initial_active, options)
}

/// LP ソルバーに委譲してQP結果に変換（Q=0 ケース）
fn solve_as_lp(problem: &QpProblem, options: &SolverOptions) -> QpResult {
    let n = problem.num_vars;
    let m = problem.num_constraints;

    let ct = vec![ConstraintType::Le; m];
    let lp = match LpProblem::new_general(
        problem.c.clone(),
        problem.a.clone(),
        problem.b.clone(),
        ct,
        problem.bounds.clone(),
        None,
    ) {
        Ok(lp) => lp,
        Err(_) => return QpResult::infeasible(),
    };

    let result = simplex::solve_with(&lp, options);
    match result.status {
        SolveStatus::Optimal => {
            let x = result.solution.clone();
            let obj = problem.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();
            // active_set: 活性制約インデックス
            let active: Vec<usize> = (0..m)
                .filter(|&i| {
                    let ax_i: f64 = (0..n)
                        .map(|j| get_a_element(&problem.a, i, j) * x[j])
                        .sum();
                    (ax_i - problem.b[i]).abs() < PIVOT_TOL
                })
                .collect();
            QpResult {
                status: SolveStatus::Optimal,
                objective: obj,
                solution: x,
                dual_solution: result.dual_solution,
                active_set: active,
                iterations: 0,
            }
        }
        SolveStatus::Infeasible => QpResult::infeasible(),
        SolveStatus::Unbounded => QpResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            solution: vec![],
            dual_solution: vec![],
            active_set: vec![],
            iterations: 0,
        },
        SolveStatus::MaxIterations => QpResult::max_iterations(vec![], f64::INFINITY, vec![], 0),
    }
}

/// Phase I: LP を使って初期実行可能点を求める
fn find_initial_feasible_point(
    problem: &QpProblem,
    options: &SolverOptions,
) -> Option<Vec<f64>> {
    let m = problem.num_constraints;
    let n = problem.num_vars;

    // 制約なしの場合: 初期点は bounds の lower bound（または 0）
    if m == 0 {
        let x: Vec<f64> = problem
            .bounds
            .iter()
            .map(|&(lb, _ub)| if lb.is_finite() { lb } else { 0.0 })
            .collect();
        return Some(x);
    }

    // LP: min 0 s.t. Ax <= b, bounds （実行可能性判定）
    let c_zero = vec![0.0f64; n];
    let ct = vec![ConstraintType::Le; m];
    let lp = LpProblem::new_general(
        c_zero,
        problem.a.clone(),
        problem.b.clone(),
        ct,
        problem.bounds.clone(),
        None,
    )
    .ok()?;

    let result = simplex::solve_with(&lp, options);
    match result.status {
        SolveStatus::Optimal => Some(result.solution),
        _ => None,
    }
}


/// Active Set メインループ
fn active_set_loop(
    problem: &QpProblem,
    mut x: Vec<f64>,
    mut working_set: WorkingSet,
    options: &SolverOptions,
) -> QpResult {
    let n = problem.num_vars;
    let m = problem.num_constraints;
    let max_iter = options.max_iterations.unwrap_or(100 * (n + m) + 1000);

    for iter in 0..max_iter {
        // 勾配 grad = Qx + c を計算
        let grad = kkt::compute_gradient(&problem.q, &x, &problem.c);

        // KKTシステムを構築して解く
        let a_active = match extract_active_rows(&problem.a, working_set.indices()) {
            Ok(a) => a,
            Err(_) => {
                let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
                return QpResult::max_iterations(x, obj, working_set.indices().to_vec(), iter);
            }
        };

        let (d, lambda) = if working_set.is_empty() {
            // 活性制約なし: 制約なし最適化方向
            match solve_unconstrained_direction(&problem.q, &grad) {
                Ok(d) => (d, vec![]),
                Err(_) => {
                    // Q が特異: 停留点として扱う
                    (vec![0.0; n], vec![])
                }
            }
        } else {
            let kkt_solver = match KktSolver::new(&problem.q, &a_active) {
                Ok(s) => s,
                Err(_) => {
                    let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
                    return QpResult::max_iterations(x, obj, working_set.indices().to_vec(), iter);
                }
            };
            match kkt_solver.solve(&grad) {
                Ok(result) => result,
                Err(_) => {
                    let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
                    return QpResult::max_iterations(x, obj, working_set.indices().to_vec(), iter);
                }
            }
        };

        let d_norm: f64 = d.iter().map(|&di| di * di).sum::<f64>().sqrt();

        if d_norm < PIVOT_TOL {
            // d ≈ 0: KKT条件確認
            if lambda.is_empty() {
                // 制約なし最適: 勾配が小さければ最適
                if grad.iter().map(|&g| g * g).sum::<f64>().sqrt() < PIVOT_TOL {
                    let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
                    return QpResult {
                        status: SolveStatus::Optimal,
                        objective: obj,
                        solution: x,
                        dual_solution: lambda,
                        active_set: working_set.indices().to_vec(),
                        iterations: iter + 1,
                    };
                }
            }

            // 最小のラグランジュ乗数を確認
            let min_lambda_val = lambda.iter().cloned().fold(f64::INFINITY, f64::min);
            if min_lambda_val >= -PIVOT_TOL {
                // KKT条件満足: 最適解
                let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
                return QpResult {
                    status: SolveStatus::Optimal,
                    objective: obj,
                    solution: x,
                    dual_solution: lambda.clone(),
                    active_set: working_set.indices().to_vec(),
                    iterations: iter + 1,
                };
            }

            // 最小λを持つ制約を除去（Bland則: 複数ある場合は最小インデックスを選択）
            let min_lambda_idx = lambda
                .iter()
                .enumerate()
                .filter(|(_, &lv)| lv < -PIVOT_TOL)
                .min_by(|a, b| {
                    // Bland則: 活性集合内の制約インデックスが小さい方を選ぶ
                    let idx_a = working_set.get(a.0).unwrap_or(usize::MAX);
                    let idx_b = working_set.get(b.0).unwrap_or(usize::MAX);
                    idx_a.cmp(&idx_b)
                })
                .map(|(i, _)| i);

            if let Some(k) = min_lambda_idx {
                if let Some(constraint_idx) = working_set.get(k) {
                    working_set.remove(constraint_idx);
                }
            }
        } else {
            // d ≠ 0: ステップ幅計算
            let alpha = compute_step_size(problem, &x, &d, &working_set, m);

            // x を更新
            for i in 0..n {
                x[i] += alpha.step * d[i];
            }

            // α < 1: ブロッキング制約を活性集合に追加
            if alpha.step < 1.0 - ZERO_TOL {
                if let Some(blocking) = alpha.blocking_constraint {
                    working_set.add(blocking);
                }
            }
        }
    }

    // 反復上限超過
    let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
    QpResult::max_iterations(x, obj, working_set.indices().to_vec(), max_iter)
}

/// 制約なしの探索方向: Q * d = -grad を解く（対角Q高速パス）
fn solve_unconstrained_direction(
    q: &CscMatrix,
    grad: &[f64],
) -> Result<Vec<f64>, ()> {
    let n = grad.len();
    let mut d = vec![0.0f64; n];

    // 対角行列の場合: d[i] = -grad[i] / q[i][i]
    let mut is_diag = true;
    for col in 0..n {
        let start = q.col_ptr[col];
        let end = q.col_ptr[col + 1];
        for k in start..end {
            if q.row_ind[k] != col {
                is_diag = false;
                break;
            }
        }
        if !is_diag {
            break;
        }
    }

    if is_diag {
        for i in 0..n {
            let q_ii = get_diagonal(q, i);
            if q_ii.abs() < 1e-12 {
                return Err(()); // 特異
            }
            d[i] = -grad[i] / q_ii;
        }
        return Ok(d);
    }

    // 一般PSDの場合: LU分解で解く
    // 一時的にQをKKT行列として使用（活性制約なし）
    let a_empty = CscMatrix::new(0, n);
    match KktSolver::new(q, &a_empty) {
        Ok(solver) => match solver.solve(grad) {
            Ok((d_result, _)) => Ok(d_result),
            Err(_) => Err(()),
        },
        Err(_) => Err(()),
    }
}

/// ステップ幅計算の結果
struct StepResult {
    step: f64,
    blocking_constraint: Option<usize>,
}

/// ステップ幅 α* を計算する（ライン探索）
///
/// 非活性制約が活性化しないよう最大ステップ幅を計算する。
fn compute_step_size(
    problem: &QpProblem,
    x: &[f64],
    d: &[f64],
    working_set: &WorkingSet,
    m: usize,
) -> StepResult {
    let n = x.len();
    let mut alpha_crit = 1.0f64;
    let mut blocking: Option<usize> = None;

    for i in 0..m {
        // 活性制約はスキップ
        if working_set.contains(i) {
            continue;
        }

        // a_i^T d を計算
        let ai_d = dot_row_a(&problem.a, i, d);
        if ai_d <= ZERO_TOL {
            continue; // この制約はブロックしない
        }

        // a_i^T x を計算
        let ai_x = dot_row_a(&problem.a, i, x);
        let slack = problem.b[i] - ai_x;

        // α ≤ slack / (a_i^T d)
        let alpha_i = slack / ai_d;
        if alpha_i < alpha_crit {
            alpha_crit = alpha_i;
            blocking = Some(i); // Bland則: 最小インデックスを採用
        } else if (alpha_i - alpha_crit).abs() < ZERO_TOL {
            // タイブレーク: 最小インデックスを採用（Bland則）
            if let Some(prev) = blocking {
                if i < prev {
                    blocking = Some(i);
                }
            }
        }
    }

    // 変数境界によるステップ制限
    for j in 0..n {
        let (lb, ub) = problem.bounds[j];
        if d[j] > ZERO_TOL && ub.is_finite() {
            let slack = ub - x[j];
            let alpha_j = slack / d[j];
            if alpha_j < alpha_crit {
                alpha_crit = alpha_j;
                // 変数境界はblockingに含めない（制約インデックスがないため）
                blocking = None;
            }
        } else if d[j] < -ZERO_TOL && lb.is_finite() {
            let slack = x[j] - lb;
            let alpha_j = slack / (-d[j]);
            if alpha_j < alpha_crit {
                alpha_crit = alpha_j;
                blocking = None;
            }
        }
    }

    StepResult {
        step: alpha_crit.max(0.0),
        blocking_constraint: blocking,
    }
}

/// 行列 A の第 row 行と x のドット積を計算する
fn dot_row_a(a: &CscMatrix, row: usize, x: &[f64]) -> f64 {
    let mut result = 0.0f64;
    for (col, &xj) in x.iter().enumerate().take(a.ncols) {
        let start = a.col_ptr[col];
        let end = a.col_ptr[col + 1];
        for k in start..end {
            if a.row_ind[k] == row {
                result += a.values[k] * xj;
                break;
            }
        }
    }
    result
}

/// 行列 A の (row, col) 要素を返す
fn get_a_element(a: &CscMatrix, row: usize, col: usize) -> f64 {
    let start = a.col_ptr[col];
    let end = a.col_ptr[col + 1];
    for k in start..end {
        if a.row_ind[k] == row {
            return a.values[k];
        }
    }
    0.0
}

/// 対角要素 Q[i,i] を返す
fn get_diagonal(q: &CscMatrix, i: usize) -> f64 {
    let start = q.col_ptr[i];
    let end = q.col_ptr[i + 1];
    for k in start..end {
        if q.row_ind[k] == i {
            return q.values[k];
        }
    }
    0.0
}
