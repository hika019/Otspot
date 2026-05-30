//! bound dual (z) の postsolve 系操作。
//!
//! - reduced → orig 空間への展開 (`remap_bound_duals_to_orig`)
//! - singleton 列の停留性から導出される y 区間への射影 (`project_duals_from_singleton_columns`)
//! - 明確 slack ある不等式行の dual を 0 にする (`zero_inactive_inequality_duals`)

use crate::problem::SolverResult;
use crate::qp::postsolve::dual_recovery::{
    compute_dual_recovery_row_activity, compute_dual_recovery_row_bounds,
    dual_recovery_row_slack_tol,
};
use crate::qp::problem::QpProblem;
use crate::tolerances::SLACK_TOL_REL;

/// reduced bound_duals を元問題空間に展開。除去変数の bound_dual は 0.0 で埋める。
pub(crate) fn remap_bound_duals_to_orig(
    presolve_result: &crate::presolve::QpPresolveResult,
    orig_bounds: &[(f64, f64)],
    reduced_bound_duals: &[f64],
) -> Vec<f64> {
    let n_lb_orig = orig_bounds.iter().filter(|(lb, _)| lb.is_finite()).count();
    let n_ub_orig = orig_bounds.iter().filter(|(_, ub)| ub.is_finite()).count();
    if n_lb_orig + n_ub_orig == 0 {
        return Vec::new();
    }
    let reduced_bounds = &presolve_result.reduced.bounds;
    let n_lb_reduced = reduced_bounds
        .iter()
        .filter(|(lb, _)| lb.is_finite())
        .count();
    let n_reduced = reduced_bounds.len();

    let mut lb_bd_idx: Vec<Option<usize>> = vec![None; n_reduced];
    let mut ub_bd_idx: Vec<Option<usize>> = vec![None; n_reduced];
    {
        let mut li = 0usize;
        for (jj, &(lb, _)) in reduced_bounds.iter().enumerate() {
            if lb.is_finite() {
                lb_bd_idx[jj] = Some(li);
                li += 1;
            }
        }
        let mut ui = 0usize;
        for (jj, &(_, ub)) in reduced_bounds.iter().enumerate() {
            if ub.is_finite() {
                ub_bd_idx[jj] = Some(n_lb_reduced + ui);
                ui += 1;
            }
        }
    }

    let mut new_bd = vec![0.0_f64; n_lb_orig + n_ub_orig];
    if !reduced_bound_duals.is_empty() {
        let mut orig_li = 0usize;
        for (j, &(lb, _)) in orig_bounds.iter().enumerate() {
            if lb.is_finite() {
                if let Some(jj) = presolve_result.col_map[j] {
                    if let Some(bd_idx) = lb_bd_idx[jj] {
                        if bd_idx < reduced_bound_duals.len() {
                            new_bd[orig_li] = reduced_bound_duals[bd_idx];
                        }
                    }
                }
                orig_li += 1;
            }
        }
        let mut orig_ui = 0usize;
        for (j, &(_, ub)) in orig_bounds.iter().enumerate() {
            if ub.is_finite() {
                if let Some(jj) = presolve_result.col_map[j] {
                    if let Some(bd_idx) = ub_bd_idx[jj] {
                        if bd_idx < reduced_bound_duals.len() {
                            new_bd[n_lb_orig + orig_ui] = reduced_bound_duals[bd_idx];
                        }
                    }
                }
                orig_ui += 1;
            }
        }
    }
    new_bd
}

/// singleton column の停留性から row dual の feasible interval を作り、現在 y を射影する。
/// unconstrained LSQ refine では one-sided bound 列で「非負 z で補正不能な y」が出るのを補正。
pub(crate) fn project_duals_from_singleton_columns(
    problem: &QpProblem,
    result: &mut SolverResult,
) {
    let Some((lower, upper)) = compute_dual_recovery_row_bounds(problem, &result.solution) else {
        return;
    };
    if result.dual_solution.len() != problem.num_constraints {
        return;
    }
    for row in 0..problem.num_constraints {
        let lo = lower[row];
        let hi = upper[row];
        if lo > hi {
            continue;
        }
        let y = &mut result.dual_solution[row];
        if *y < lo {
            *y = lo;
        } else if *y > hi {
            *y = hi;
        }
    }
}

/// 明確に slack ある不等式行の dual を相補性から 0 にする。LSQ/IR は stationarity のみ見るため
/// slack 行に dual が残る場合がある。
pub(crate) fn zero_inactive_inequality_duals(problem: &QpProblem, result: &mut SolverResult) {
    if result.solution.len() != problem.num_vars
        || result.dual_solution.len() != problem.num_constraints
    {
        return;
    }
    let Some((ax, row_abs_activity)) =
        compute_dual_recovery_row_activity(problem, &result.solution)
    else {
        return;
    };
    for i in 0..problem.num_constraints {
        let slack = match problem.constraint_types[i] {
            crate::problem::ConstraintType::Le => problem.b[i] - ax[i],
            crate::problem::ConstraintType::Ge => ax[i] - problem.b[i],
            crate::problem::ConstraintType::Eq => continue,
        };
        let tol = dual_recovery_row_slack_tol(problem, i, ax[i], row_abs_activity[i], SLACK_TOL_REL);
        if slack > tol {
            result.dual_solution[i] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presolve::QpPresolveResult;
    use crate::problem::{ConstraintType, SolveStatus};
    use crate::sparse::CscMatrix;

    fn make_presolve_result(
        orig_n: usize,
        col_map: Vec<Option<usize>>,
        reduced_bounds: Vec<(f64, f64)>,
    ) -> QpPresolveResult {
        let n_red = reduced_bounds.len();
        let reduced = QpProblem::new_all_le(
            CscMatrix::new(n_red, n_red),
            vec![0.0; n_red],
            CscMatrix::new(0, n_red),
            vec![],
            reduced_bounds,
        )
        .unwrap();
        let orig = QpProblem::new_all_le(
            CscMatrix::new(orig_n, orig_n),
            vec![0.0; orig_n],
            CscMatrix::new(0, orig_n),
            vec![],
            vec![(f64::NEG_INFINITY, f64::INFINITY); orig_n],
        )
        .unwrap();
        let mut pr = QpPresolveResult::no_reduction(&orig);
        pr.reduced = reduced;
        pr.col_map = col_map;
        pr
    }

    /// remap layout = [lb 有限の z_lb; ub 有限の z_ub] が orig/reduced 双方を満たすこと。
    /// 複数パターン: lb-only + fix / ub-only + free / 両側 box。
    /// no-op 化検証: 全 0 を返す版に書換 → 全 case FAIL を手動確認済 → revert。
    #[test]
    fn remap_bound_duals_to_orig_layouts_table_driven() {
        struct Case {
            name: &'static str,
            orig_bounds: Vec<(f64, f64)>,
            col_map: Vec<Option<usize>>,
            reduced_bounds: Vec<(f64, f64)>,
            reduced_duals: Vec<f64>,
            expect: Vec<f64>,
        }
        let cases = vec![
            Case {
                name: "lb-only with fixed middle var",
                orig_bounds: vec![
                    (0.0, f64::INFINITY),
                    (0.0, f64::INFINITY),
                    (0.0, f64::INFINITY),
                ],
                col_map: vec![Some(0), None, Some(1)],
                reduced_bounds: vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
                reduced_duals: vec![1.5, 2.5],
                expect: vec![1.5, 0.0, 2.5],
            },
            Case {
                name: "ub-only with free middle var",
                orig_bounds: vec![
                    (f64::NEG_INFINITY, 4.0),
                    (f64::NEG_INFINITY, f64::INFINITY),
                    (f64::NEG_INFINITY, 7.0),
                ],
                col_map: vec![Some(0), Some(1), Some(2)],
                reduced_bounds: vec![
                    (f64::NEG_INFINITY, 4.0),
                    (f64::NEG_INFINITY, f64::INFINITY),
                    (f64::NEG_INFINITY, 7.0),
                ],
                reduced_duals: vec![1.0, 2.0],
                expect: vec![1.0, 2.0],
            },
            Case {
                name: "two-sided box: lb block then ub block",
                orig_bounds: vec![(0.0, 5.0), (0.0, 6.0)],
                col_map: vec![Some(0), Some(1)],
                reduced_bounds: vec![(0.0, 5.0), (0.0, 6.0)],
                reduced_duals: vec![0.5, 0.75, 1.25, 1.75],
                expect: vec![0.5, 0.75, 1.25, 1.75],
            },
        ];
        for case in &cases {
            let pr = make_presolve_result(
                case.orig_bounds.len(),
                case.col_map.clone(),
                case.reduced_bounds.clone(),
            );
            let got = remap_bound_duals_to_orig(&pr, &case.orig_bounds, &case.reduced_duals);
            assert_eq!(
                got.len(),
                case.expect.len(),
                "{}: length mismatch ({:?})",
                case.name,
                got
            );
            for (i, (g, e)) in got.iter().zip(case.expect.iter()).enumerate() {
                assert!(
                    (g - e).abs() < 1e-12,
                    "{}: slot {} expected {} got {} (full={:?})",
                    case.name,
                    i,
                    e,
                    g,
                    got
                );
            }
        }
    }

    /// project_duals: Le over-shoot を 0 に clamp / Eq pass-through / lb-only から y を引上げ。
    /// no-op 化検証: 空 body 版で各 case FAIL を手動確認済 ("Le clamp failed: y=5") → revert。
    #[test]
    fn project_duals_from_singleton_columns_table_driven() {
        // Le over-shoot
        {
            let q = CscMatrix::new(2, 2);
            let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, 1.0], 1, 2).unwrap();
            let problem = QpProblem::new_all_le(
                q,
                vec![0.0, 0.0],
                a,
                vec![0.0],
                vec![(0.0, f64::INFINITY); 2],
            )
            .unwrap();
            let mut result = SolverResult {
                status: SolveStatus::Optimal,
                solution: vec![0.0, 0.0],
                dual_solution: vec![5.0],
                bound_duals: vec![0.0, 0.0],
                ..SolverResult::default()
            };
            project_duals_from_singleton_columns(&problem, &mut result);
            assert!(
                result.dual_solution[0].abs() < 1e-12,
                "Le clamp failed: y={}",
                result.dual_solution[0]
            );
        }
        // Eq pass-through
        {
            let q = CscMatrix::new(1, 1);
            let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
            let mut problem = QpProblem::new_all_le(
                q,
                vec![0.0],
                a,
                vec![3.0],
                vec![(f64::NEG_INFINITY, f64::INFINITY)],
            )
            .unwrap();
            problem.constraint_types[0] = ConstraintType::Eq;
            let mut result = SolverResult {
                status: SolveStatus::Optimal,
                solution: vec![3.0],
                dual_solution: vec![-7.5],
                bound_duals: vec![],
                ..SolverResult::default()
            };
            project_duals_from_singleton_columns(&problem, &mut result);
            assert!(
                (result.dual_solution[0] + 7.5).abs() < 1e-12,
                "Eq pass-through failed: y={}",
                result.dual_solution[0]
            );
        }
        // lb-only singleton lifts y up to 2.0
        {
            let q = CscMatrix::new(1, 1);
            let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
            let problem = QpProblem::new_all_le(
                q,
                vec![-2.0],
                a,
                vec![0.0],
                vec![(0.0, f64::INFINITY)],
            )
            .unwrap();
            let mut result = SolverResult {
                status: SolveStatus::Optimal,
                solution: vec![0.0],
                dual_solution: vec![0.0],
                bound_duals: vec![0.0],
                ..SolverResult::default()
            };
            project_duals_from_singleton_columns(&problem, &mut result);
            assert!(
                (result.dual_solution[0] - 2.0).abs() < 1e-12,
                "lb-only lift failed: y={}",
                result.dual_solution[0]
            );
        }
    }

    /// zero_inactive: Le slack huge → cleared / Le active → kept / Eq → never cleared.
    /// no-op 化検証: 空 body 版で huge-slack case FAIL を手動確認済
    /// ("Le slack huge → cleared: expected cleared, got y=3.5") → revert。
    #[test]
    fn zero_inactive_inequality_duals_table_driven() {
        struct Case {
            name: &'static str,
            ct: ConstraintType,
            b: f64,
            x: f64,
            initial_y: f64,
            expect_cleared: bool,
        }
        let cases = vec![
            Case {
                name: "Le slack huge → cleared",
                ct: ConstraintType::Le,
                b: 100.0,
                x: 1.0,
                initial_y: 3.5,
                expect_cleared: true,
            },
            Case {
                name: "Le active (slack≈0) → kept",
                ct: ConstraintType::Le,
                b: 1.0,
                x: 1.0,
                initial_y: 3.5,
                expect_cleared: false,
            },
            Case {
                name: "Eq → never cleared",
                ct: ConstraintType::Eq,
                b: 100.0,
                x: 1.0,
                initial_y: -2.5,
                expect_cleared: false,
            },
        ];
        for case in &cases {
            let q = CscMatrix::new(1, 1);
            let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
            let mut problem = QpProblem::new_all_le(
                q,
                vec![0.0],
                a,
                vec![case.b],
                vec![(f64::NEG_INFINITY, f64::INFINITY)],
            )
            .unwrap();
            problem.constraint_types[0] = case.ct;
            let mut result = SolverResult {
                status: SolveStatus::Optimal,
                solution: vec![case.x],
                dual_solution: vec![case.initial_y],
                bound_duals: vec![],
                ..SolverResult::default()
            };
            zero_inactive_inequality_duals(&problem, &mut result);
            if case.expect_cleared {
                assert!(
                    result.dual_solution[0].abs() < 1e-12,
                    "{}: expected cleared, got y={}",
                    case.name,
                    result.dual_solution[0]
                );
            } else {
                assert!(
                    (result.dual_solution[0] - case.initial_y).abs() < 1e-12,
                    "{}: expected y={} preserved, got {}",
                    case.name,
                    case.initial_y,
                    result.dual_solution[0]
                );
            }
        }
    }
}
