//! LP dispatch モジュール
//!
//! Q=0 退化ケース（LP 問題）を Simplex に委譲する関数群。
//! `qp/mod.rs` から分離。ロジック変更なし。

use crate::options::SolverOptions;
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::backend::SimplexBackend;
use crate::backend::LpBackend;

use super::QpProblem;
use super::ipm_solver;

/// Q=0 退化ケース（LP 問題）を LP ソルバーに委譲して QP 結果に変換する
pub(crate) fn solve_as_lp_pub(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    solve_as_lp(problem, options)
}

fn solve_as_lp(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    // Simplex と IPM fallback が同一の deadline を共有するよう、最初に deadline を確定する。
    // これにより IPM fallback が fresh な timeout_secs を取得して二重タイムアウトになるのを防ぐ。
    let opts_with_deadline;
    let options: &SolverOptions = if options.deadline.is_none() {
        if let Some(secs) = options.timeout_secs {
            opts_with_deadline = {
                let mut o = options.clone();
                o.deadline = Some(std::time::Instant::now() + std::time::Duration::from_secs_f64(secs));
                o.timeout_secs = None;
                o
            };
            &opts_with_deadline
        } else {
            options
        }
    } else {
        options
    };

    // Eq/Ge/Le制約型をそのままSimplexに渡す（設計書§2.4）。
    // Simplexは ConstraintType::Eq を Phase I 人工変数で正しく処理する。
    // to_all_le()は全パスで廃止済み。IPMもSimplexもConstraintTypeをネイティブ処理。
    let lp = match LpProblem::new_general(
        problem.c.clone(),
        problem.a.clone(),
        problem.b.clone(),
        problem.constraint_types.clone(),
        problem.bounds.clone(),
        None,
    ) {
        Ok(lp) => lp,
        Err(_) => return SolverResult::infeasible(),
    };

    let simplex_result = SimplexBackend.solve(&lp, options);

    // 特異基底（サイクリック構造のネットワーク流 LP など）では Simplex が NumericalError を返す。
    // Simplex は基底行列を必要とするが IPM は不要なので、IPM にフォールバックする。
    if simplex_result.status == SolveStatus::NumericalError {
        return ipm_solver::solve_qp_v2(problem, options);
    }

    // Simplex が Optimal を返しても reduced_costs に負値が残る場合がある。
    // これは LU 基底の数値精度劣化 (ill-conditioning) や退化縮退が原因で、
    // primal feasible だが dual infeasible な解となる。
    // bench の compute_dfeas_orig と同じ基準 (成分相対化) で dfr を検査し、
    // dfr > eps ならば Simplex の解は双対非実行可能として IPM にフォールバックする。
    //
    // LP 最適性条件 (双対): rc_j ≥ 0 for LB 非基底, rc_j ≤ 0 for UB 非基底, rc_j = 0 for 基底。
    // UB 非基底 (x_j ≈ ub_j) は rc_j < 0 が正常。誤って負値を dual infeasible と判定しないよう
    // 変数の状態に応じて符号チェックを切り替える。
    if simplex_result.status == SolveStatus::Optimal {
        let rc = &simplex_result.reduced_costs;
        let sol = &simplex_result.solution;
        let n = lp.num_vars;
        if !rc.is_empty() && rc.len() == n {
            let mut dfr: f64 = 0.0;
            for j in 0..n {
                // FX 変数 (lb ≈ ub) は除外
                let (lb_j, ub_j) = lp.bounds[j];
                if lb_j.is_finite() && ub_j.is_finite() && (lb_j - ub_j).abs() < 1e-12 {
                    continue;
                }
                // EmptyCol (A の列が空) は除外
                if lp.a.col_ptr.len() > j + 1 && lp.a.col_ptr[j + 1] - lp.a.col_ptr[j] == 0 {
                    continue;
                }
                let rc_j = rc[j];
                // UB 非基底変数 (x_j ≈ ub_j) か判定。これらは rc_j ≤ 0 が最適性条件。
                let x_j = sol.get(j).copied().unwrap_or(0.0);
                let at_ub = ub_j.is_finite()
                    && (x_j - ub_j).abs() <= 1e-8 * (1.0 + ub_j.abs());
                let viol = if at_ub {
                    f64::max(0.0, rc_j)   // UB 非基底: rc_j > 0 が違反
                } else {
                    f64::max(0.0, -rc_j)  // LB 非基底 / 自由: rc_j < 0 が違反
                };
                if viol > 0.0 {
                    let scale_j = 1.0 + rc_j.abs() + lp.c[j].abs();
                    dfr = dfr.max(viol / scale_j);
                }
            }
            if dfr > options.ipm_eps() {
                return ipm_solver::solve_qp_v2(problem, options);
            }
        }

        if let Some((pfeas_rel, bfeas_rel)) = simplex_primal_quality(problem, &simplex_result.solution) {
            // simplex の check_eq_feasibility が FEASIBILITY_TOL=1e-4 以下を保証するので、
            // pfeas_rel は常に 1e-4 未満。制約残差の閾値は FEASIBILITY_TOL を使い、
            // 境界違反 (bfeas_rel) のみ ipm_eps で厳密チェックする。
            const PFEAS_SIMPLEX_TOL: f64 = 1e-4;
            if pfeas_rel > PFEAS_SIMPLEX_TOL || bfeas_rel > options.ipm_eps() {
                return ipm_solver::solve_qp_v2(problem, options);
            }
        } else {
            return ipm_solver::solve_qp_v2(problem, options);
        }
    }

    let result = simplex_result;
    match result.status {
        SolveStatus::Optimal => {
            let x = result.solution.clone();
            // c^T x + obj_offset (QPS の N-row RHS による定数項)。
            let obj: f64 = problem.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum::<f64>()
                + problem.obj_offset;
            // 双対解はSimplex出力をそのまま使用（展開なし）
            let dual = result.dual_solution.clone();
            SolverResult {
                status: SolveStatus::Optimal,
                objective: obj,
                solution: x,
                dual_solution: dual,
                reduced_costs: result.reduced_costs.clone(),
                slack: result.slack.clone(),
                warm_start_basis: result.warm_start_basis.clone(),
                bound_duals: vec![],
                iterations: result.iterations,
                solver_used: None,
                final_residuals: None,
                pfeas: None,
                dfeas: None,
                gap: None,
                duality_gap_rel: None,
            }
        }
        SolveStatus::Infeasible => SolverResult::infeasible(),
        SolveStatus::Unbounded => SolverResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            bound_duals: vec![],
            iterations: 0,
            solver_used: None,
            final_residuals: None,
            pfeas: None,
            dfeas: None,
            gap: None,
            duality_gap_rel: None,
        },
        SolveStatus::MaxIterations => {
            // DEAD PATH: SimplexOutcome::MaxIterations廃止により到達不能。
            // SolveStatus enum variant自体は未削除。
            unreachable!("MaxIterations is dead code - not reachable via simplex path")
        }
        SolveStatus::SuboptimalSolution => {
            // DEAD PATH: SuboptimalSolution is not reachable via current simplex implementation
            SolverResult::numerical_error()
        }
        SolveStatus::Timeout => SolverResult {
            status: SolveStatus::Timeout,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            bound_duals: vec![],
            iterations: 0,
            solver_used: None,
            final_residuals: None,
            pfeas: None,
            dfeas: None,
            gap: None,
            duality_gap_rel: None,
        },
        SolveStatus::NumericalError => SolverResult {
            status: SolveStatus::NumericalError,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            bound_duals: vec![],
            iterations: 0,
            solver_used: None,
            final_residuals: None,
            pfeas: None,
            dfeas: None,
            gap: None,
            duality_gap_rel: None,
        },
        SolveStatus::NonConvex(_) => SolverResult {
            status: result.status,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            bound_duals: vec![],
            iterations: 0,
            solver_used: None,
            final_residuals: None,
            pfeas: None,
            dfeas: None,
            gap: None,
            duality_gap_rel: None,
        },
        // LocallyOptimal は LP path では発生しない (Q=0 なら Simplex を使うため)。
        // exhaustive match のためのフォールバック。
        SolveStatus::LocallyOptimal => SolverResult::numerical_error(),
    }
}

fn simplex_primal_quality(problem: &QpProblem, solution: &[f64]) -> Option<(f64, f64)> {
    if solution.is_empty() || solution.len() != problem.num_vars {
        return None;
    }

    let pfeas_rel = if problem.num_constraints == 0 {
        0.0
    } else {
        let ax = problem.a.mat_vec_mul(solution).ok()?;
        let mut max_rel = 0.0_f64;
        for (i, (&ax_i, &b_i)) in ax.iter().zip(problem.b.iter()).enumerate() {
            let violation = match problem.constraint_types.get(i) {
                Some(ConstraintType::Eq) => (ax_i - b_i).abs(),
                Some(ConstraintType::Ge) => (b_i - ax_i).max(0.0),
                _ => (ax_i - b_i).max(0.0),
            };
            let scale_i = 1.0 + ax_i.abs() + b_i.abs();
            max_rel = max_rel.max(violation / scale_i);
        }
        max_rel
    };

    let mut max_v = 0.0_f64;
    let mut max_x = 0.0_f64;
    let mut max_bnd = 0.0_f64;
    for (&xi, &(lb, ub)) in solution.iter().zip(problem.bounds.iter()) {
        let lb_viol = if lb.is_finite() { (lb - xi).max(0.0) } else { 0.0 };
        let ub_viol = if ub.is_finite() { (xi - ub).max(0.0) } else { 0.0 };
        max_v = max_v.max(lb_viol.max(ub_viol));
        max_x = max_x.max(xi.abs());
        if lb.is_finite() { max_bnd = max_bnd.max(lb.abs()); }
        if ub.is_finite() { max_bnd = max_bnd.max(ub.abs()); }
    }
    let bfeas_rel = max_v / (1.0 + max_x.max(max_bnd));

    if pfeas_rel.is_finite() && bfeas_rel.is_finite() {
        Some((pfeas_rel, bfeas_rel))
    } else {
        None
    }
}
