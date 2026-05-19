//! POST_STAGE_TRACE 系統 (env=1 で有効) の診断ログ。
//! 全関数は side-effect-only (eprintln) で本筋ロジックから分離。

use crate::presolve::qp_transforms::QpPostsolveStep;
use crate::presolve::QpPresolveResult;
use crate::problem::SolverResult;
use crate::qp::ipm_solver::kkt::{kkt_residual_rel, primal_residual_rel};
use crate::qp::ipm_solver::outcome::ProblemView;
use crate::qp::problem::QpProblem;

pub(super) fn trace_enabled() -> bool {
    std::env::var("POST_STAGE_TRACE").ok().as_deref() == Some("1")
}

/// IPM 内部終了直後 (まだ scaled + reduced 空間)。
pub(super) fn log_ipm_exit_reduced(reduced: &QpProblem, result: &SolverResult) {
    let view = build_view(reduced);
    let pres = primal_residual_rel(&view, &result.solution);
    let kkt = kkt_residual_rel(&view, &result.solution, &result.dual_solution, &result.bound_duals);
    let ax = reduced
        .a
        .mat_vec_mul(&result.solution)
        .unwrap_or_else(|_| vec![0.0; reduced.num_constraints]);
    let mut pres_abs = 0.0_f64;
    let mut max_ax = 0.0_f64;
    let mut max_b = 0.0_f64;
    use crate::problem::ConstraintType as CT;
    for (i, (&ax_i, &b_i)) in ax.iter().zip(reduced.b.iter()).enumerate() {
        let v = match reduced.constraint_types[i] {
            CT::Eq => (ax_i - b_i).abs(),
            CT::Ge => (b_i - ax_i).max(0.0),
            _ => (ax_i - b_i).max(0.0),
        };
        pres_abs = pres_abs.max(v);
        max_ax = max_ax.max(ax_i.abs());
        max_b = max_b.max(b_i.abs());
    }
    let denom = 1.0 + max_ax.max(max_b);
    eprintln!("POST_STAGE [IPM exit (scaled+reduced)] pres_rel={:.3e} pres_abs={:.3e} denom={:.3e} kkt_rel={:.3e} n={} m={}",
        pres, pres_abs, denom, kkt, reduced.num_vars, reduced.num_constraints);
}

/// Ruiz unscale 前後の |x|_inf, |y|_inf 比。
pub(super) fn log_ruiz_scale_ratio(
    scaler: &crate::linalg::ruiz::RuizScaler,
    pre_x: &[f64],
    pre_y: &[f64],
    post_x: &[f64],
    post_y: &[f64],
) {
    let x_pre_inf = pre_x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let y_pre_inf = pre_y.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let x_post_inf = post_x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let y_post_inf = post_y.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    eprintln!("POST_STAGE [Ruiz scale ratio] x_inf {:.3e}->{:.3e} (×{:.3e}) y_inf {:.3e}->{:.3e} (×{:.3e}) c_scale={:.3e}",
        x_pre_inf, x_post_inf, x_post_inf / x_pre_inf.max(1e-300),
        y_pre_inf, y_post_inf, y_post_inf / y_pre_inf.max(1e-300),
        scaler.c);
}

/// Unscaled (まだ reduced 空間)。
pub(super) fn log_unscaled_reduced(reduced: &QpProblem, result: &SolverResult) {
    let view = build_view(reduced);
    let pres = primal_residual_rel(&view, &result.solution);
    let kkt = kkt_residual_rel(&view, &result.solution, &result.dual_solution, &result.bound_duals);
    eprintln!(
        "POST_STAGE [unscaled (still reduced)] pres_rel={:.3e} kkt_rel={:.3e}",
        pres, kkt
    );
}

pub(super) fn log_presolve_transforms(
    presolve_result: &QpPresolveResult,
    reduced: &QpProblem,
    orig_problem: &QpProblem,
) {
    let mut n_fixed = 0;
    let mut n_singleton = 0;
    let mut n_empty = 0;
    let mut n_largescale = 0;
    let mut row_scales_for_diag: Option<Vec<f64>> = None;
    for step in presolve_result.postsolve_stack.steps.iter() {
        match step {
            QpPostsolveStep::FixedVar { .. } => n_fixed += 1,
            QpPostsolveStep::SingletonRow { .. } => n_singleton += 1,
            QpPostsolveStep::EmptyCol { .. } => n_empty += 1,
            QpPostsolveStep::LargeCoeffRowScale { row_scales } => {
                n_largescale += 1;
                row_scales_for_diag = Some(row_scales.clone());
            }
        }
    }
    eprintln!("POST_STAGE [presolve transforms] FixedVar={} SingletonRow={} EmptyCol={} LargeCoeffRowScale={} reduced_vars={} orig_vars={}",
        n_fixed, n_singleton, n_empty, n_largescale,
        reduced.num_vars, orig_problem.num_vars);
    if let Some(scales) = &row_scales_for_diag {
        let n_scaled = scales.iter().filter(|&&s| (s - 1.0).abs() > 1e-12).count();
        let smin = scales.iter().fold(f64::INFINITY, |a, &v| a.min(v));
        let smax = scales.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v));
        let mut indexed: Vec<(usize, f64)> = scales
            .iter()
            .enumerate()
            .filter(|(_, &s)| (s - 1.0).abs() > 1e-12)
            .map(|(i, &s)| (i, s))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let top5: Vec<String> = indexed
            .iter()
            .take(5)
            .map(|(i, s)| format!("row[{}]=σ:{:.3e}(amp:×{:.2e})", i, s, 1.0 / s))
            .collect();
        eprintln!("POST_STAGE [LargeCoeffRowScale] n_scaled={} σ_min={:.3e} σ_max={:.3e} smallest_5: {}",
            n_scaled, smin, smax, top5.join(", "));
    }
}

pub(super) fn log_postsolve_remap_bd(orig_problem: &QpProblem, final_sol: &SolverResult) {
    let view = build_view(orig_problem);
    let pres = primal_residual_rel(&view, &final_sol.solution);
    let kkt = kkt_residual_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    eprintln!(
        "POST_STAGE [postsolve+remap_bd (orig space)] pres_rel={:.3e} kkt_rel={:.3e}",
        pres, kkt
    );
}

/// 元空間 violation 分布 + top-1 row 詳細。
pub(super) fn log_violation_distribution(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    reduced: &QpProblem,
    final_sol: &SolverResult,
) {
    use crate::problem::ConstraintType;
    let view = build_view(orig_problem);
    let pres = primal_residual_rel(&view, &final_sol.solution);
    let kkt = kkt_residual_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    let ax = orig_problem
        .a
        .mat_vec_mul(&final_sol.solution)
        .unwrap_or_else(|_| vec![0.0; orig_problem.num_constraints]);
    let mut pres_abs = 0.0_f64;
    let mut max_ax = 0.0_f64;
    let mut max_b = 0.0_f64;
    for (i, (&ax_i, &b_i)) in ax.iter().zip(orig_problem.b.iter()).enumerate() {
        let v = match orig_problem.constraint_types[i] {
            ConstraintType::Eq => (ax_i - b_i).abs(),
            ConstraintType::Ge => (b_i - ax_i).max(0.0),
            _ => (ax_i - b_i).max(0.0),
        };
        pres_abs = pres_abs.max(v);
        max_ax = max_ax.max(ax_i.abs());
        max_b = max_b.max(b_i.abs());
    }
    let denom = 1.0 + max_ax.max(max_b);
    eprintln!("POST_STAGE [postsolve+remap (orig space, pre bounds-clip)] pres_rel={:.3e} pres_abs={:.3e} denom={:.3e} kkt_rel={:.3e} n={} m={}",
        pres, pres_abs, denom, kkt, orig_problem.num_vars, orig_problem.num_constraints);

    let mut viol: Vec<(usize, f64)> = (0..orig_problem.num_constraints)
        .map(|i| {
            let raw = ax[i] - orig_problem.b[i];
            let v = match orig_problem.constraint_types[i] {
                ConstraintType::Eq => raw.abs(),
                ConstraintType::Ge => {
                    if raw < 0.0 { -raw } else { 0.0 }
                }
                ConstraintType::Le => {
                    if raw > 0.0 { raw } else { 0.0 }
                }
            };
            (i, v)
        })
        .collect();
    viol.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top10: Vec<String> = viol
        .iter()
        .take(10)
        .map(|(i, v)| format!("row[{}]={:.2e}", i, v))
        .collect();
    let total_viol: f64 = viol.iter().map(|(_, v)| v).sum();
    let top1_share = if total_viol > 0.0 { viol[0].1 / total_viol * 100.0 } else { 0.0 };
    let top10_share: f64 =
        viol.iter().take(10).map(|(_, v)| v).sum::<f64>() / total_viol.max(1e-300) * 100.0;
    eprintln!(
        "POST_STAGE [violation distribution] top1_share={:.1}% top10_share={:.1}% top10: {}",
        top1_share, top10_share, top10.join(", ")
    );
    if !viol.is_empty() && viol[0].1 > 0.0 {
        log_top1_row_detail(orig_problem, presolve_result, reduced, final_sol, viol[0].0);
    }
}

fn log_top1_row_detail(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    reduced: &QpProblem,
    final_sol: &SolverResult,
    top_row: usize,
) {
    let mut row_terms: Vec<(usize, f64, f64, bool)> = Vec::new();
    for col in 0..orig_problem.num_vars {
        let cs = orig_problem.a.col_ptr[col];
        let ce = orig_problem.a.col_ptr[col + 1];
        for k in cs..ce {
            if orig_problem.a.row_ind[k] == top_row {
                let a_val = orig_problem.a.values[k];
                let x_val = final_sol.solution[col];
                let is_reduced = presolve_result
                    .col_map
                    .get(col)
                    .copied()
                    .flatten()
                    .is_some();
                row_terms.push((col, a_val, x_val, is_reduced));
            }
        }
    }
    row_terms.sort_by(|a, b| {
        (b.1 * b.2)
            .abs()
            .partial_cmp(&(a.1 * a.2).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_str: Vec<String> = row_terms
        .iter()
        .take(8)
        .map(|(c, a, x, red)| {
            format!(
                "col[{}]{}·{:.2e}·{:.2e}={:.2e}",
                c, if *red { "(IPM)" } else { "(FIXED)" },
                a, x, a * x
            )
        })
        .collect();
    let sum: f64 = row_terms.iter().map(|(_, a, x, _)| a * x).sum();
    let n_fixed_in_row = row_terms.iter().filter(|(_, _, _, r)| !r).count();
    let n_ipm_in_row = row_terms.iter().filter(|(_, _, _, r)| *r).count();
    eprintln!("POST_STAGE [top-1 viol row {}] b={:.3e} A·x_sum={:.3e} viol={:.3e} (fixed_vars={} ipm_vars={}) top8: {}",
        top_row, orig_problem.b[top_row], sum, sum - orig_problem.b[top_row],
        n_fixed_in_row, n_ipm_in_row, top_str.join(", "));
    let red_row = presolve_result.row_map.get(top_row).copied().flatten();
    let row_max_orig: f64 = max_abs_row(&orig_problem.a, top_row, orig_problem.num_vars);
    let red_row_max: Option<f64> = red_row.map(|rr| max_abs_row(&reduced.a, rr, reduced.num_vars));
    let scale_factor = match (row_max_orig, red_row_max) {
        (o, Some(r)) if o > 0.0 => Some(r / o),
        _ => None,
    };
    eprintln!("POST_STAGE [top-1 viol row {} mapping] orig→reduced={:?} orig_row_max={:.3e} reduced_row_max={:?} effective_σ={:?}",
        top_row, red_row, row_max_orig, red_row_max, scale_factor);
}

fn max_abs_row(a: &crate::sparse::CscMatrix, target_row: usize, ncols: usize) -> f64 {
    let mut mx = 0.0_f64;
    for col in 0..ncols {
        let cs = a.col_ptr[col];
        let ce = a.col_ptr[col + 1];
        for k in cs..ce {
            if a.row_ind[k] == target_row {
                mx = mx.max(a.values[k].abs());
            }
        }
    }
    mx
}

pub(super) fn log_bounds_clip(
    orig_problem: &QpProblem,
    final_sol: &SolverResult,
    clip_count_pre: usize,
    total_bound_clip: f64,
) {
    let view = build_view(orig_problem);
    let pres = primal_residual_rel(&view, &final_sol.solution);
    let kkt = kkt_residual_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    eprintln!("POST_STAGE [bounds clip applied] count={} max_amt={:.3e} pres_rel={:.3e} kkt_rel={:.3e}",
        clip_count_pre, total_bound_clip, pres, kkt);
}

pub(super) fn log_pre_post_processing(orig_problem: &QpProblem, final_sol: &SolverResult) {
    let view = build_view(orig_problem);
    let pres0 = primal_residual_rel(&view, &final_sol.solution);
    let kkt0 = kkt_residual_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    eprintln!(
        "POST_STAGE [pre post-processing] pres_rel={:.3e} kkt_rel={:.3e}",
        pres0, kkt0
    );
}

fn build_view(p: &QpProblem) -> ProblemView<'_> {
    ProblemView {
        q: &p.q,
        a: &p.a,
        c: &p.c,
        b: &p.b,
        bounds: &p.bounds,
        constraint_types: &p.constraint_types,
    }
}
