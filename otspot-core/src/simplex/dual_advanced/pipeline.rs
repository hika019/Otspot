//! Phase pipeline and cold-start driver for dual advanced solver.

use super::bounded_core::{
    bounded_primal_phase1, bounded_primal_phase2_aug, extract_dual_info_bounded,
    extract_solution_bounded, phase2_primal_bounded, BoundedDualState, BoundedOutcome,
};
use super::{
    bounded_obj_from_state, fallback_profile_enabled, make_leaving_strategy,
    maybe_perturb_initial_xb, reconcile_bounded_terminal_state, BoundedTerminalReconcile,
    PHASE1_BOUND_VIOLATION_FALLBACKS, UB_VIOLATION_FALLBACKS,
};
use super::{extract_dual_info, extract_solution, SimplexOutcome, StandardForm};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::sparse::CscMatrix;
use std::sync::atomic::Ordering;

use super::BoundedStandardForm;

#[allow(clippy::too_many_arguments)]
pub(super) fn run_phase1_then_phase2<F>(
    bsf: &BoundedStandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
    state_factory: F,
    b: &[f64],
    c: &[f64],
    row_scale: &[f64],
    col_scale: &[f64],
) -> Option<SolverResult>
where
    F: FnOnce() -> (
        CscMatrix,
        Vec<Option<usize>>,
        Vec<f64>,
        Vec<usize>,
        Vec<bool>,
        Vec<f64>,
    ),
{
    fn mark_eq_ub_path(mut r: SolverResult) -> SolverResult {
        r.stats.bounded_eq_ub_path = true;
        r
    }

    let (a_aug, art_col_of_row, mut ubs_aug, basis, is_basic, mut x_b) = state_factory();
    let n_aug = a_aug.ncols;
    maybe_perturb_initial_xb(&mut x_b);
    let mut state = BoundedDualState {
        basis,
        at_upper: vec![false; n_aug],
        x_b,
        reduced_costs: vec![0.0; n_aug],
        is_basic,
        iterations: 0,
        price_start: 0,
    };

    // Phase I: minimise sum of artificials. Structural cost = 0.
    let mut c_p1 = vec![0.0f64; n_aug];
    for col in art_col_of_row.iter().flatten() {
        c_p1[*col] = 1.0;
    }
    let mut iters: usize = 0;
    let p1_out = bounded_primal_phase1(
        &a_aug,
        &c_p1,
        &ubs_aug,
        bsf.n_total,
        &mut state,
        options,
        &mut iters,
    );

    match p1_out {
        SimplexOutcome::SingularBasis => {
            return Some(mark_eq_ub_path(SolverResult::numerical_error()));
        }
        SimplexOutcome::Unbounded => {
            return None;
        }
        SimplexOutcome::Timeout(_) => {
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            return Some(mark_eq_ub_path(SolverResult {
                status: SolveStatus::Timeout,
                objective: bsf.obj_offset,
                solution,
                iterations: iters,
                ..Default::default()
            }));
        }
        SimplexOutcome::Optimal(_, _) => {
            let art_sum = match reconcile_bounded_terminal_state(
                &a_aug, b, &c_p1, &ubs_aug, &mut state, options,
            ) {
                BoundedTerminalReconcile::Optimal(obj) => obj,
                BoundedTerminalReconcile::Timeout(_) => {
                    let solution = extract_solution_bounded(bsf, &state, col_scale);
                    return Some(mark_eq_ub_path(SolverResult {
                        status: SolveStatus::Timeout,
                        objective: bsf.obj_offset,
                        solution,
                        iterations: iters,
                        ..Default::default()
                    }));
                }
                BoundedTerminalReconcile::BoundViolation => {
                    if fallback_profile_enabled() {
                        PHASE1_BOUND_VIOLATION_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                    }
                    return None;
                }
                BoundedTerminalReconcile::MatrixAccessError
                | BoundedTerminalReconcile::SingularBasis => {
                    return Some(mark_eq_ub_path(SolverResult::numerical_error()));
                }
            };
            if art_sum > options.primal_tol {
                let mut r = SolverResult::infeasible();
                r.iterations = iters;
                return Some(mark_eq_ub_path(r));
            }
        }
    }

    // Pin artificials to ub = 0 for Phase II.
    for col in art_col_of_row.iter().flatten() {
        ubs_aug[*col] = 0.0;
    }

    // Phase II: minimise true objective on augmented matrix.
    let mut c_p2 = vec![0.0f64; n_aug];
    c_p2[..bsf.n_total].copy_from_slice(c);
    let p2_out = bounded_primal_phase2_aug(
        &a_aug,
        &c_p2,
        &ubs_aug,
        bsf.n_total,
        &mut state,
        options,
        &mut iters,
    );

    match p2_out {
        SimplexOutcome::Optimal(_, y) => {
            let pre_reconcile_x_b = state.x_b.clone();
            let obj = match reconcile_bounded_terminal_state(
                &a_aug, b, &c_p2, &ubs_aug, &mut state, options,
            ) {
                BoundedTerminalReconcile::Optimal(obj) => obj,
                BoundedTerminalReconcile::Timeout(obj) => {
                    let solution = extract_solution_bounded(bsf, &state, col_scale);
                    return Some(mark_eq_ub_path(SolverResult {
                        status: SolveStatus::Timeout,
                        objective: obj + bsf.obj_offset,
                        solution,
                        iterations: iters,
                        ..Default::default()
                    }));
                }
                BoundedTerminalReconcile::BoundViolation => {
                    state.x_b = pre_reconcile_x_b;
                    let obj = bounded_obj_from_state(&c_p2, &ubs_aug, &state);
                    let solution = extract_solution_bounded(bsf, &state, col_scale);
                    return Some(mark_eq_ub_path(SolverResult {
                        status: SolveStatus::Timeout,
                        objective: obj + bsf.obj_offset,
                        solution,
                        iterations: iters,
                        ..Default::default()
                    }));
                }
                BoundedTerminalReconcile::MatrixAccessError
                | BoundedTerminalReconcile::SingularBasis => {
                    return Some(mark_eq_ub_path(SolverResult::numerical_error()));
                }
            };
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info_bounded(bsf, problem, &y, &solution, row_scale);
            let ws = if state.basis.iter().all(|&j| j < bsf.n_total) {
                Some(WarmStartBasis {
                    basis: state.basis.clone(),
                    x_b: state.x_b.clone(),
                })
            } else {
                None
            };
            Some(mark_eq_ub_path(SolverResult {
                status: SolveStatus::Optimal,
                objective: obj + bsf.obj_offset,
                solution,
                dual_solution,
                reduced_costs,
                slack,
                warm_start_basis: ws,
                iterations: iters,
                ..Default::default()
            }))
        }
        SimplexOutcome::Unbounded => Some(mark_eq_ub_path(SolverResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            iterations: iters,
            ..Default::default()
        })),
        SimplexOutcome::Timeout(obj) => {
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            Some(mark_eq_ub_path(SolverResult {
                status: SolveStatus::Timeout,
                objective: obj + bsf.obj_offset,
                solution,
                iterations: iters,
                ..Default::default()
            }))
        }
        SimplexOutcome::SingularBasis => Some(mark_eq_ub_path(SolverResult::numerical_error())),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn finish_bounded(
    dual_out: BoundedOutcome,
    dual_state: BoundedDualState,
    bsf: &BoundedStandardForm,
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    row_scale: &[f64],
    col_scale: &[f64],
    ubs: &[f64],
    problem: &LpProblem,
    options: &SolverOptions,
    total_iters: &mut usize,
) -> Option<SolverResult> {
    match dual_out {
        BoundedOutcome::UbViolationOutOfScope { .. } => {
            if fallback_profile_enabled() {
                UB_VIOLATION_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            }
            None
        }
        BoundedOutcome::Unbounded => Some(SolverResult {
            status: SolveStatus::Infeasible,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        }),
        BoundedOutcome::Timeout(obj) => {
            let solution = extract_solution_bounded(bsf, &dual_state, col_scale);
            Some(SolverResult {
                status: SolveStatus::Timeout,
                objective: obj + bsf.obj_offset,
                solution,
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
                iterations: *total_iters,
                ..Default::default()
            })
        }
        BoundedOutcome::SingularBasis => Some(SolverResult::numerical_error()),
        BoundedOutcome::Optimal(_, _) => {
            let (p2_out, mut p2_state) =
                phase2_primal_bounded(bsf, dual_state, a, c, options, total_iters, ubs);
            let p2_out = match p2_out {
                SimplexOutcome::Optimal(_, y) => {
                    let pre_reconcile_x_b = p2_state.x_b.clone();
                    match reconcile_bounded_terminal_state(a, b, c, ubs, &mut p2_state, options) {
                        BoundedTerminalReconcile::Optimal(obj) => SimplexOutcome::Optimal(obj, y),
                        BoundedTerminalReconcile::Timeout(obj) => SimplexOutcome::Timeout(obj),
                        BoundedTerminalReconcile::BoundViolation => {
                            p2_state.x_b = pre_reconcile_x_b;
                            let obj = bounded_obj_from_state(c, ubs, &p2_state);
                            SimplexOutcome::Timeout(obj)
                        }
                        BoundedTerminalReconcile::MatrixAccessError
                        | BoundedTerminalReconcile::SingularBasis => SimplexOutcome::SingularBasis,
                    }
                }
                other => other,
            };
            Some(finish_bounded_phase2(
                p2_out,
                p2_state,
                bsf,
                col_scale,
                row_scale,
                problem,
                *total_iters,
            ))
        }
    }
}

fn finish_bounded_phase2(
    out: SimplexOutcome,
    state: BoundedDualState,
    bsf: &BoundedStandardForm,
    col_scale: &[f64],
    row_scale: &[f64],
    problem: &LpProblem,
    total_iters: usize,
) -> SolverResult {
    match out {
        SimplexOutcome::Optimal(obj, y) => {
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info_bounded(bsf, problem, &y, &solution, row_scale);
            let ws = WarmStartBasis {
                basis: state.basis,
                x_b: state.x_b,
            };
            SolverResult {
                status: SolveStatus::Optimal,
                objective: obj + bsf.obj_offset,
                solution,
                dual_solution,
                reduced_costs,
                slack,
                warm_start_basis: Some(ws),
                iterations: total_iters,
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
            let solution = extract_solution_bounded(bsf, &state, col_scale);
            SolverResult {
                status: SolveStatus::Timeout,
                objective: obj + bsf.obj_offset,
                solution,
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
                iterations: total_iters,
                ..Default::default()
            }
        }
        SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
    }
}

/// Le-only cold startでHarris Dual Simplexを使用する
#[allow(clippy::too_many_arguments)]
pub(super) fn cold_start_advanced(
    sf: &StandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    row_scale: &[f64],
    col_scale: &[f64],
) -> SolverResult {
    let m = sf.m;

    let mut basis = sf.initial_basis.clone();
    let mut x_b = b.to_vec();

    // コスト摂動: c̃_j = max(c_j, 0) → スラック基底（y=0）で r̃_j = c̃_j ≥ 0 → 双対実行可能
    let c_perturbed: Vec<f64> = c.iter().map(|&v| v.max(0.0)).collect();

    let mut leaving = make_leaving_strategy(options.dual_pricing, m);

    let mut total_iters: usize = 0;
    let phase1_outcome = super::core::dual_simplex_core_advanced(
        a,
        &mut x_b,
        &c_perturbed,
        &mut basis,
        m,
        sf.n_total,
        sf.n_total,
        false,
        options,
        leaving.as_mut(),
        &mut total_iters,
    );

    match phase1_outcome {
        SimplexOutcome::Unbounded => {
            return SolverResult {
                status: SolveStatus::Infeasible,
                objective: f64::INFINITY,
                solution: vec![],
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
                ..Default::default()
            };
        }
        SimplexOutcome::Timeout(_) => {
            return super::super::timeout_result_with_incumbent(
                sf,
                problem,
                &basis,
                &x_b,
                col_scale,
                total_iters,
            );
        }
        SimplexOutcome::SingularBasis => {
            return SolverResult::numerical_error();
        }
        SimplexOutcome::Optimal(_, _) => {}
    }

    // Phase 2: 元のコストで主実行可能点からPrimal Simplexで最適化
    use super::super::pricing::SteepestEdgePricing;
    let mut pricing = SteepestEdgePricing::new(sf.n_total);
    let phase2_outcome = super::super::revised_simplex_core(
        a,
        &mut x_b,
        c,
        b,
        &mut basis,
        m,
        sf.n_total,
        sf.n_total,
        &mut pricing,
        options,
        &mut total_iters,
        false,
        None,
    );

    let mut result = match phase2_outcome {
        SimplexOutcome::Optimal(obj, y) => {
            let solution = extract_solution(sf, &basis, &x_b, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info(sf, problem, &y, &solution, row_scale);
            let ws = WarmStartBasis {
                basis: basis.to_vec(),
                x_b: x_b.to_vec(),
            };
            SolverResult {
                status: SolveStatus::Optimal,
                objective: obj + sf.obj_offset,
                solution,
                dual_solution,
                reduced_costs,
                slack,
                warm_start_basis: Some(ws),
                iterations: total_iters,
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
            let solution = extract_solution(sf, &basis, &x_b, col_scale);
            SolverResult {
                status: SolveStatus::Timeout,
                objective: obj + sf.obj_offset,
                solution,
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
                iterations: total_iters,
                ..Default::default()
            }
        }
        SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
    };
    result.iterations = total_iters;
    result
}
