//! IPM 数値カーネル + 後処理 (Ruiz unscale, postsolve, bound clip, 元空間 KKT)。
//! IpmOutcome は元空間の解と残差のみ持ち、satisfies_eps(user_eps) は常に元空間判定。

use super::kkt::{bound_violation, complementarity_residual_rel, kkt_residual_rel, primal_residual_rel};
use super::outcome::{IpmOutcome, ProblemView};
use crate::options::SolverOptions;
use crate::presolve::qp_transforms::QpPostsolveStep;
use crate::presolve::{
    bound_contrib_at_var, postsolve_qp_with_dual_recovery, recover_y_for_singleton_row_with_bound,
    QpPresolveResult,
};
use crate::problem::SolveStatus;
use crate::qp::problem::QpProblem;

pub type InnerSolver = fn(&QpProblem, &SolverOptions) -> crate::problem::SolverResult;

/// presolve で col/row が reduced 空間に縮んだ場合、warm_start_qp.x / .y を col_map_inv /
/// row_map で reduced 空間に翻訳する。dropped 列・行 (col_map[j]/row_map[i] = None) の warm
/// 値は reduced 問題に存在しないため棄却。dim 不一致は警告付き drop。
///
/// 加えて presolve 内 Ruiz scaler (qp_transforms.rs 末尾) が reduced 問題を scaled 空間に
/// 書き換えている場合、warm の (x, y) を同じ scaled 空間に変換する:
///   x_s = D^{-1} x_orig         (RuizScaler::scale_problem の `bounds_s = bounds / d` と整合)
///   y_s = c * y_orig / e        (KKT より: y_orig = e * y_s / c → y_s = c * y_orig / e)
/// この変換が無いと `presolve_did_ruiz` 経路 (attempt.rs) で IPM 側 use_ruiz=false 固定の
/// ため IPM 入口 Ruiz scaling (ipm_core/scaling.rs) も bypass され、orig 空間の warm が
/// scaled reduced 問題に入り誤位置 init になる。
fn translate_warm_start_for_presolve(
    opts: &mut SolverOptions,
    presolve_result: &crate::presolve::QpPresolveResult,
    reduced: &QpProblem,
) {
    let needs_reduce = presolve_result.was_reduced;
    let needs_ruiz = presolve_result.ruiz_scaler.is_some();
    if !needs_reduce && !needs_ruiz {
        return;
    }
    let Some(ws) = opts.warm_start_qp.as_mut() else { return };

    let n_orig = presolve_result.orig_num_vars;
    let m_orig = presolve_result.orig_num_constraints;
    if ws.x.len() != n_orig || ws.y.len() != m_orig {
        eprintln!(
            "[warm_start_qp dropped] presolve dim mismatch: ws.x={}/{} ws.y={}/{}",
            ws.x.len(), n_orig, ws.y.len(), m_orig
        );
        opts.warm_start_qp = None;
        return;
    }

    let n_red = reduced.num_vars;
    let m_red = reduced.num_constraints;

    let mut x_red = vec![0.0_f64; n_red];
    if needs_reduce {
        for (k, &j_orig) in presolve_result.col_map_inv.iter().enumerate() {
            if k < n_red && j_orig < n_orig {
                x_red[k] = ws.x[j_orig];
            }
        }
    } else if ws.x.len() == n_red {
        x_red.copy_from_slice(&ws.x);
    }

    let mut y_red = vec![0.0_f64; m_red];
    if needs_reduce {
        for (i_orig, mapped) in presolve_result.row_map.iter().enumerate() {
            if let Some(i_red) = mapped {
                if *i_red < m_red {
                    y_red[*i_red] = ws.y[i_orig];
                }
            }
        }
    } else if ws.y.len() == m_red {
        y_red.copy_from_slice(&ws.y);
    }

    if let Some(scaler) = &presolve_result.ruiz_scaler {
        if scaler.d.len() != n_red || scaler.e.len() != m_red
            || !scaler.c.is_finite() || scaler.c <= 0.0
        {
            eprintln!(
                "[warm_start_qp dropped] ruiz scaler dim/c invalid: d={}/{} e={}/{} c={}",
                scaler.d.len(), n_red, scaler.e.len(), m_red, scaler.c
            );
            opts.warm_start_qp = None;
            return;
        }
        for k in 0..n_red {
            let dk = scaler.d[k];
            if !dk.is_finite() || dk == 0.0 {
                eprintln!("[warm_start_qp dropped] ruiz d[{}]={} non-finite/zero", k, dk);
                opts.warm_start_qp = None;
                return;
            }
            x_red[k] /= dk;
        }
        for i in 0..m_red {
            let ei = scaler.e[i];
            if !ei.is_finite() || ei == 0.0 {
                eprintln!("[warm_start_qp dropped] ruiz e[{}]={} non-finite/zero", i, ei);
                opts.warm_start_qp = None;
                return;
            }
            y_red[i] = scaler.c * y_red[i] / ei;
        }
    }

    if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
        eprintln!(
            "[warm_start_qp translated] presolve reduction n:{}→{} m:{}→{} ruiz={}",
            n_orig, n_red, m_orig, m_red, needs_ruiz
        );
    }

    ws.x = x_red;
    ws.y = y_red;
}

/// 1 回の IPPMM 呼出 + 後処理。元空間の解と残差を返す。
pub fn run_ipm(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    opts: &SolverOptions,
) -> IpmOutcome {
    run_ipm_with(
        orig_problem,
        presolve_result,
        opts,
        crate::qp::ipm_core::solve_qp_ippmm,
    )
}

fn run_ipm_with(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    opts: &SolverOptions,
    inner_solver: InnerSolver,
) -> IpmOutcome {
    let reduced = &presolve_result.reduced;

    // presolve スケーリング (LargeCoeffRowScale × Ruiz E / c·D) で問題が σ 倍に縮むと
    // unscale 時に残差が 1/σ 倍に増幅される。primal 側 e_min × LargeCoeffRowScale と
    // dual 側 c·d_min の小さい方を sigma_total とし、IPM eps を user_eps×σ に厳しくする。
    let mut primal_row_scale_min = 1.0_f64;
    for step in presolve_result.postsolve_stack.steps.iter() {
        if let QpPostsolveStep::LargeCoeffRowScale { row_scales } = step {
            let local_min = row_scales
                .iter()
                .filter(|&&v| v > 0.0 && v.is_finite())
                .fold(f64::INFINITY, |a, &v| a.min(v));
            if local_min.is_finite() {
                primal_row_scale_min *= local_min;
            }
        }
    }
    let mut dual_col_scale_min = f64::INFINITY;
    if let Some(scaler) = &presolve_result.ruiz_scaler {
        let e_min = scaler
            .e
            .iter()
            .filter(|&&v| v > 0.0 && v.is_finite())
            .fold(f64::INFINITY, |a, &v| a.min(v));
        if e_min.is_finite() {
            primal_row_scale_min *= e_min;
        }
        let d_min = scaler
            .d
            .iter()
            .filter(|&&v| v > 0.0 && v.is_finite())
            .fold(f64::INFINITY, |a, &v| a.min(v));
        if d_min.is_finite() && scaler.c.is_finite() && scaler.c > 0.0 {
            dual_col_scale_min = scaler.c * d_min;
        }
    }
    let sigma_total = primal_row_scale_min.min(dual_col_scale_min);
    let mut opts_for_ipm: SolverOptions = if sigma_total < 1.0 && sigma_total > 0.0 {
        let mut tightened = opts.clone();
        let eps_orig = opts.ipm_eps();
        let eps_scaled = (eps_orig * sigma_total).max(f64::MIN_POSITIVE);
        tightened.tolerance = None;
        tightened.ipm.eps = eps_scaled;
        if std::env::var("POST_STAGE_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "POST_STAGE [IPM eps tighten] σ_total={:.3e} eps_orig={:.3e} → eps_scaled={:.3e}",
                sigma_total, eps_orig, eps_scaled
            );
        }
        tightened
    } else {
        opts.clone()
    };

    // warm_start_qp を presolve reduced 空間に翻訳。dropped 列/行の warm 値は無視
    // (reduced 問題はその dof を持たないため安全)。dim 不一致なら drop + 警告。
    translate_warm_start_for_presolve(&mut opts_for_ipm, presolve_result, reduced);

    let mut result = inner_solver(reduced, &opts_for_ipm);

    // 確定的 Infeasible/Unbounded/NonConvex は outcome に保持して Timeout 隠蔽を避ける。
    if matches!(
        result.status,
        SolveStatus::Infeasible | SolveStatus::Unbounded | SolveStatus::NonConvex(_)
    ) {
        return IpmOutcome::infeasibility(result.status);
    }

    // 不定 Q + 慣性修正 IPM 収束時は LocallyOptimal フラグを保持。
    // 後処理は Optimal と同パスで行うため一旦 Optimal に昇格。
    let is_locally_optimal = result.status == SolveStatus::LocallyOptimal;
    if is_locally_optimal {
        result.status = SolveStatus::Optimal;
    }

    let invalid = result.solution.is_empty()
        || result.solution.iter().any(|v| !v.is_finite())
        || matches!(result.status, SolveStatus::NumericalError);
    if invalid {
        return IpmOutcome {
            solution: Vec::new(),
            dual_solution: Vec::new(),
            bound_duals: Vec::new(),
            objective: f64::INFINITY,
            iterations: result.iterations,
            kkt_residual_rel: f64::INFINITY,
            primal_residual_rel: f64::INFINITY,
            bound_violation: f64::INFINITY,
            complementarity_residual_rel: f64::INFINITY,
            duality_gap_rel: f64::INFINITY,
            numerical_failure: true,
            infeasibility_status: None,
            is_locally_optimal: false,
        };
    }

    // dual の LSQ refine は元空間に戻してから行う。scaled 空間で LSQ を回すと L2 ノルム
    // 最小化が scaled 残差分布に過剰適合し、unscale 後に元空間残差が悪化することがある。

    let post_trace = std::env::var("POST_STAGE_TRACE").ok().as_deref() == Some("1");
    if post_trace {
        let view_red = ProblemView {
            q: &reduced.q,
            a: &reduced.a,
            c: &reduced.c,
            b: &reduced.b,
            bounds: &reduced.bounds,
            constraint_types: &reduced.constraint_types,
        };
        let pres_red = primal_residual_rel(&view_red, &result.solution);
        let kkt_red = kkt_residual_rel(
            &view_red,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        let ax_red = reduced
            .a
            .mat_vec_mul(&result.solution)
            .unwrap_or_else(|_| vec![0.0; reduced.num_constraints]);
        let mut pres_abs_red = 0.0_f64;
        let mut max_ax_red = 0.0_f64;
        let mut max_b_red = 0.0_f64;
        use crate::problem::ConstraintType as CT;
        for (i, (&ax_i, &b_i)) in ax_red.iter().zip(reduced.b.iter()).enumerate() {
            let v = match reduced.constraint_types[i] {
                CT::Eq => (ax_i - b_i).abs(),
                CT::Ge => (b_i - ax_i).max(0.0),
                _ => (ax_i - b_i).max(0.0),
            };
            pres_abs_red = pres_abs_red.max(v);
            max_ax_red = max_ax_red.max(ax_i.abs());
            max_b_red = max_b_red.max(b_i.abs());
        }
        let denom_red = 1.0 + max_ax_red.max(max_b_red);
        eprintln!("POST_STAGE [IPM exit (scaled+reduced)] pres_rel={:.3e} pres_abs={:.3e} denom={:.3e} kkt_rel={:.3e} n={} m={}",
            pres_red, pres_abs_red, denom_red, kkt_red, reduced.num_vars, reduced.num_constraints);
    }

    if let Some(scaler) = &presolve_result.ruiz_scaler {
        if post_trace {
            let x_pre_inf = result.solution.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            let y_pre_inf = result
                .dual_solution
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()));
            let (x_unscaled, y_unscaled) =
                scaler.unscale_solution(&result.solution, &result.dual_solution);
            let x_post_inf = x_unscaled.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            let y_post_inf = y_unscaled.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            eprintln!("POST_STAGE [Ruiz scale ratio] x_inf {:.3e}->{:.3e} (×{:.3e}) y_inf {:.3e}->{:.3e} (×{:.3e}) c_scale={:.3e}",
                x_pre_inf, x_post_inf, x_post_inf / x_pre_inf.max(1e-300),
                y_pre_inf, y_post_inf, y_post_inf / y_pre_inf.max(1e-300),
                scaler.c);
        }
        let (x, y) = scaler.unscale_solution(&result.solution, &result.dual_solution);
        result.solution = x;
        result.dual_solution = y;
        result.bound_duals = scaler.unscale_bound_duals(&result.bound_duals, &reduced.bounds);
        if scaler.c.abs() > 1e-300 {
            result.objective /= scaler.c;
        }
    }
    if post_trace {
        let view_red = ProblemView {
            q: &reduced.q,
            a: &reduced.a,
            c: &reduced.c,
            b: &reduced.b,
            bounds: &reduced.bounds,
            constraint_types: &reduced.constraint_types,
        };
        let pres_red = primal_residual_rel(&view_red, &result.solution);
        let kkt_red = kkt_residual_rel(
            &view_red,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        eprintln!(
            "POST_STAGE [unscaled (still reduced)] pres_rel={:.3e} kkt_rel={:.3e}",
            pres_red, kkt_red
        );
    }

    if post_trace {
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

    // postsolve: reduced 空間 → 元問題空間。eliminated 行 / 固定変数の dual 復元込み。
    let mut final_sol = postsolve_qp_with_dual_recovery(presolve_result, &result, orig_problem);

    if presolve_result.was_reduced {
        final_sol.bound_duals = crate::qp::remap_bound_duals_to_orig(
            presolve_result,
            &orig_problem.bounds,
            &final_sol.bound_duals,
        );
    }

    if post_trace {
        let view = ProblemView {
            q: &orig_problem.q,
            a: &orig_problem.a,
            c: &orig_problem.c,
            b: &orig_problem.b,
            bounds: &orig_problem.bounds,
            constraint_types: &orig_problem.constraint_types,
        };
        let pres_post = primal_residual_rel(&view, &final_sol.solution);
        let kkt_post = kkt_residual_rel(
            &view,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        eprintln!(
            "POST_STAGE [postsolve+remap_bd (orig space)] pres_rel={:.3e} kkt_rel={:.3e}",
            pres_post, kkt_post
        );
    }

    if post_trace {
        let view = ProblemView {
            q: &orig_problem.q,
            a: &orig_problem.a,
            c: &orig_problem.c,
            b: &orig_problem.b,
            bounds: &orig_problem.bounds,
            constraint_types: &orig_problem.constraint_types,
        };
        let pres = primal_residual_rel(&view, &final_sol.solution);
        let kkt = kkt_residual_rel(
            &view,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        use crate::problem::ConstraintType;
        let ax_orig = orig_problem
            .a
            .mat_vec_mul(&final_sol.solution)
            .unwrap_or_else(|_| vec![0.0; orig_problem.num_constraints]);
        let mut pres_abs_orig = 0.0_f64;
        let mut max_ax_orig = 0.0_f64;
        let mut max_b_orig = 0.0_f64;
        for (i, (&ax_i, &b_i)) in ax_orig.iter().zip(orig_problem.b.iter()).enumerate() {
            let v = match orig_problem.constraint_types[i] {
                ConstraintType::Eq => (ax_i - b_i).abs(),
                ConstraintType::Ge => (b_i - ax_i).max(0.0),
                _ => (ax_i - b_i).max(0.0),
            };
            pres_abs_orig = pres_abs_orig.max(v);
            max_ax_orig = max_ax_orig.max(ax_i.abs());
            max_b_orig = max_b_orig.max(b_i.abs());
        }
        let denom_orig = 1.0 + max_ax_orig.max(max_b_orig);
        eprintln!("POST_STAGE [postsolve+remap (orig space, pre bounds-clip)] pres_rel={:.3e} pres_abs={:.3e} denom={:.3e} kkt_rel={:.3e} n={} m={}",
            pres, pres_abs_orig, denom_orig, kkt, orig_problem.num_vars, orig_problem.num_constraints);
        let ax = orig_problem
            .a
            .mat_vec_mul(&final_sol.solution)
            .unwrap_or_else(|_| vec![0.0; orig_problem.num_constraints]);
        let mut viol: Vec<(usize, f64)> = (0..orig_problem.num_constraints)
            .map(|i| {
                let raw = ax[i] - orig_problem.b[i];
                let v = match orig_problem.constraint_types[i] {
                    ConstraintType::Eq => raw.abs(),
                    ConstraintType::Ge => {
                        if raw < 0.0 {
                            -raw
                        } else {
                            0.0
                        }
                    }
                    ConstraintType::Le => {
                        if raw > 0.0 {
                            raw
                        } else {
                            0.0
                        }
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
        let top1_share = if total_viol > 0.0 {
            viol[0].1 / total_viol * 100.0
        } else {
            0.0
        };
        let top10_share: f64 =
            viol.iter().take(10).map(|(_, v)| v).sum::<f64>() / total_viol.max(1e-300) * 100.0;
        eprintln!(
            "POST_STAGE [violation distribution] top1_share={:.1}% top10_share={:.1}% top10: {}",
            top1_share,
            top10_share,
            top10.join(", ")
        );
        if !viol.is_empty() && viol[0].1 > 0.0 {
            let top_row = viol[0].0;
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
                        c,
                        if *red { "(IPM)" } else { "(FIXED)" },
                        a,
                        x,
                        a * x
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
            let row_max_orig: f64 = {
                let mut mx = 0.0_f64;
                for col in 0..orig_problem.num_vars {
                    let cs = orig_problem.a.col_ptr[col];
                    let ce = orig_problem.a.col_ptr[col + 1];
                    for k in cs..ce {
                        if orig_problem.a.row_ind[k] == top_row {
                            mx = mx.max(orig_problem.a.values[k].abs());
                        }
                    }
                }
                mx
            };
            let red_row_max: Option<f64> = red_row.map(|rr| {
                let mut mx = 0.0_f64;
                for col in 0..reduced.num_vars {
                    let cs = reduced.a.col_ptr[col];
                    let ce = reduced.a.col_ptr[col + 1];
                    for k in cs..ce {
                        if reduced.a.row_ind[k] == rr {
                            mx = mx.max(reduced.a.values[k].abs());
                        }
                    }
                }
                mx
            });
            let scale_factor = match (row_max_orig, red_row_max) {
                (o, Some(r)) if o > 0.0 => Some(r / o),
                _ => None,
            };
            eprintln!("POST_STAGE [top-1 viol row {} mapping] orig→reduced={:?} orig_row_max={:.3e} reduced_row_max={:?} effective_σ={:?}",
                top_row, red_row, row_max_orig, red_row_max, scale_factor);
        }
    }

    // bounds clip (Ruiz unscale 増幅由来の微小違反補正)
    let mut total_bound_clip = 0.0_f64;
    let mut clip_count_pre = 0_usize;
    for (xi, &(lb, ub)) in final_sol
        .solution
        .iter_mut()
        .zip(orig_problem.bounds.iter())
    {
        let pre = *xi;
        if lb.is_finite() {
            *xi = xi.max(lb);
        }
        if ub.is_finite() {
            *xi = xi.min(ub);
        }
        let amt = (pre - *xi).abs();
        if amt > 0.0 {
            clip_count_pre += 1;
            total_bound_clip = total_bound_clip.max(amt);
        }
    }
    if post_trace {
        let view = ProblemView {
            q: &orig_problem.q,
            a: &orig_problem.a,
            c: &orig_problem.c,
            b: &orig_problem.b,
            bounds: &orig_problem.bounds,
            constraint_types: &orig_problem.constraint_types,
        };
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

    // 元空間 dual 一括復元: postsolve_qp_with_dual_recovery は col_first 停留性のみで
    // y[row] を復元するため、関与 fixed col の停留性が z で吸収されない。ここで
    // refine_dual_lsq を回し x/z 固定で y を LSQ-optimal に更新する。
    if presolve_result.was_reduced
        && !presolve_result.postsolve_stack.steps.is_empty()
        && final_sol.solution.len() == orig_problem.num_vars
        && final_sol.dual_solution.len() == orig_problem.num_constraints
    {
        let view0 = ProblemView {
            q: &orig_problem.q,
            a: &orig_problem.a,
            c: &orig_problem.c,
            b: &orig_problem.b,
            bounds: &orig_problem.bounds,
            constraint_types: &orig_problem.constraint_types,
        };
        const POST_LSQ_PROGRESS_EPS: f64 = 1e-12;
        let mut prev = kkt_residual_rel(
            &view0,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        let mut best_sol = final_sol.clone();
        let mut pass = 0usize;
        loop {
            if opts
                .deadline
                .is_some_and(|d| std::time::Instant::now() >= d)
            {
                final_sol = best_sol;
                break;
            }
            crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
            crate::qp::refine_dual_lsq(orig_problem, &mut final_sol, opts.deadline);
            crate::qp::zero_inactive_inequality_duals(orig_problem, &mut final_sol);
            crate::qp::project_duals_from_singleton_columns(orig_problem, &mut final_sol);
            crate::qp::refine_dual_projected_gradient(orig_problem, &mut final_sol, opts.deadline);
            crate::qp::refine_dual_worst_active_block(orig_problem, &mut final_sol, opts.deadline);
            crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
            let cur = kkt_residual_rel(
                &view0,
                &final_sol.solution,
                &final_sol.dual_solution,
                &final_sol.bound_duals,
            );
            if post_trace {
                eprintln!(
                    "POST_STAGE [postsolve dual_lsq pass {}] kkt_rel={:.3e}",
                    pass, cur
                );
            }
            if cur + POST_LSQ_PROGRESS_EPS >= prev {
                final_sol = best_sol;
                break;
            }
            prev = cur;
            best_sol = final_sol.clone();
            pass += 1;
        }
    }

    // Stage 0: postsolve y/z 交互反復。一括 LSQ で残った fixed-row dofs を col_first
    // 停留性で締める。
    if result.iterations > 0
        && presolve_result.was_reduced
        && !presolve_result.postsolve_stack.steps.is_empty()
        && final_sol.solution.len() == orig_problem.num_vars
        && final_sol.dual_solution.len() == orig_problem.num_constraints
    {
        const STAGE0_PROGRESS_EPS: f64 = 1e-12;
        let view0 = ProblemView {
            q: &orig_problem.q,
            a: &orig_problem.a,
            c: &orig_problem.c,
            b: &orig_problem.b,
            bounds: &orig_problem.bounds,
            constraint_types: &orig_problem.constraint_types,
        };
        let mut prev_kkt = kkt_residual_rel(
            &view0,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        let mut best_sol = final_sol.clone();
        let mut pass = 0usize;
        loop {
            if opts
                .deadline
                .is_some_and(|d| std::time::Instant::now() >= d)
            {
                final_sol = best_sol;
                break;
            }
            crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
            // y[row] を逆順で SingletonRow/RedundantRowFix から復元 (後退代入)
            for step in presolve_result.postsolve_stack.steps.iter().rev() {
                let (row, col) = match step {
                    QpPostsolveStep::SingletonRow { row, col, .. } => (*row, *col),
                    _ => continue,
                };
                let bc = bound_contrib_at_var(&orig_problem.bounds, &final_sol.bound_duals, col);
                recover_y_for_singleton_row_with_bound(row, col, orig_problem, &mut final_sol, bc);
            }
            crate::qp::zero_inactive_inequality_duals(orig_problem, &mut final_sol);
            crate::qp::project_duals_from_singleton_columns(orig_problem, &mut final_sol);
            crate::qp::refine_dual_projected_gradient(orig_problem, &mut final_sol, opts.deadline);
            crate::qp::refine_dual_worst_active_block(orig_problem, &mut final_sol, opts.deadline);
            crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
            let cur_kkt = kkt_residual_rel(
                &view0,
                &final_sol.solution,
                &final_sol.dual_solution,
                &final_sol.bound_duals,
            );
            if post_trace {
                eprintln!(
                    "POST_STAGE [postsolve recovery pass {}] kkt_rel={:.3e}",
                    pass, cur_kkt
                );
            }
            if cur_kkt + STAGE0_PROGRESS_EPS >= prev_kkt {
                final_sol = best_sol;
                break;
            }
            prev_kkt = cur_kkt;
            best_sol = final_sol.clone();
            pass += 1;
        }
    }

    let view = ProblemView {
        q: &orig_problem.q,
        a: &orig_problem.a,
        c: &orig_problem.c,
        b: &orig_problem.b,
        bounds: &orig_problem.bounds,
        constraint_types: &orig_problem.constraint_types,
    };

    // 元空間 post-processing 3 段階: (1) primal projection, (2) y/z 交互 refit,
    // (3) saddle-point Krylov IR。
    // IPM が 1 度も iterate しなかった場合 (cancel/timeout=0) は冷状態 x=0 から
    // 後処理が独自解を作り cancel/Timeout セマンティクスを破壊するため skip。
    let ipm_made_progress = result.iterations > 0;

    // factorize 時間予算ガード。LDL 因子化が分単位かかる規模では skip。
    const PRIMAL_PROJECTION_SIZE_LIMIT: usize = 50_000;
    let problem_size = orig_problem.num_vars + orig_problem.num_constraints;
    let allow_primal_projection = problem_size <= PRIMAL_PROJECTION_SIZE_LIMIT;

    if post_trace {
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

    // 既に IPM で eps 達成済の Optimal は post-processing skip (大規模で LSQ が秒単位)。
    // Suboptimal/Timeout は component-wise dfr が残るため skip しない。
    let user_eps_for_skip = opts.ipm_eps();
    let kkt_already_pass = if !final_sol.solution.is_empty()
        && orig_problem.num_constraints > 0
        && result.status == SolveStatus::Optimal
    {
        let kkt0 = kkt_residual_rel(
            &view,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        let pres0 = primal_residual_rel(&view, &final_sol.solution);
        kkt0 < user_eps_for_skip && pres0 < user_eps_for_skip
    } else {
        false
    };

    let kkt = if !final_sol.solution.is_empty()
        && orig_problem.num_constraints > 0
        && ipm_made_progress
        && !kkt_already_pass
    {
        // (1) primal projection: 違反制約に対して x を最小ノルム射影。
        //     pres 悪化時は revert。near-rank-deficient AAT で LSQ y refit が膨張する
        //     系統では primal のみ動かし dual は IPM 値を維持。
        if allow_primal_projection {
            let pre_x = final_sol.solution.clone();
            let pre_pres = primal_residual_rel(&view, &final_sol.solution);
            crate::qp::refine_primal_lsq(orig_problem, &mut final_sol, opts.deadline);
            let post_pres = primal_residual_rel(&view, &final_sol.solution);
            if post_pres > pre_pres {
                final_sol.solution = pre_x;
            } else {
                // x 改善時は z を新 x に合わせて refit。Q·δx が KKT 停留性に効くため
                // kkt は一時的に増えうるが、後段の y/z refit と KKT IR で回復する。
                crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
            }
            if std::env::var("PRIMAL_LSQ_TRACE").ok().as_deref() == Some("1") {
                let post_pres2 = primal_residual_rel(&view, &final_sol.solution);
                let post_kkt2 = kkt_residual_rel(
                    &view,
                    &final_sol.solution,
                    &final_sol.dual_solution,
                    &final_sol.bound_duals,
                );
                eprintln!("PRIMAL_LSQ: pre_pres={:.3e} post_pres={:.3e} final_pres={:.3e} final_kkt={:.3e} guard={}",
                    pre_pres, post_pres, post_pres2, post_kkt2,
                    if post_pres > pre_pres { "REVERT" } else { "ACCEPT" });
            }
        }

        // (2) y/z 交互 refit。y (refine_dual_lsq) と z (refit_bound_duals_kkt) は
        // 双方向依存。ill-conditioned 系では数回反復で kkt_rel が桁単位で減る。
        // 各 step は KKT-guard 付きで悪化時 revert。
        const REFIT_PROGRESS_EPS: f64 = 1e-12;
        let mut current_kkt = kkt_residual_rel(
            &view,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        loop {
            if opts
                .deadline
                .is_some_and(|d| std::time::Instant::now() >= d)
            {
                break;
            }
            let prev_kkt = current_kkt;

            let pre_dual_step = final_sol.clone();
            crate::qp::refine_dual_lsq(orig_problem, &mut final_sol, opts.deadline);
            crate::qp::zero_inactive_inequality_duals(orig_problem, &mut final_sol);
            crate::qp::project_duals_from_singleton_columns(orig_problem, &mut final_sol);
            crate::qp::refine_dual_projected_gradient(orig_problem, &mut final_sol, opts.deadline);
            crate::qp::refine_dual_worst_active_block(orig_problem, &mut final_sol, opts.deadline);
            let post_kkt = kkt_residual_rel(
                &view,
                &final_sol.solution,
                &final_sol.dual_solution,
                &final_sol.bound_duals,
            );
            if post_kkt <= current_kkt {
                current_kkt = post_kkt;
            } else {
                final_sol = pre_dual_step;
            }

            let pre_z = final_sol.bound_duals.clone();
            crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
            let post_kkt = kkt_residual_rel(
                &view,
                &final_sol.solution,
                &final_sol.dual_solution,
                &final_sol.bound_duals,
            );
            if post_kkt <= current_kkt {
                current_kkt = post_kkt;
            } else {
                final_sol.bound_duals = pre_z;
            }

            if current_kkt + REFIT_PROGRESS_EPS >= prev_kkt {
                break;
            }
        }

        // 標準 LSQ が componentwise eps を満たさない場合 IRLS で L∞ 風 y を試行。
        // 改善時は z refit + 再 IRLS の固定点反復。
        let user_eps = opts.ipm_eps();
        const IRLS_INNER_MAX_ITERS: usize = 30;
        loop {
            if current_kkt <= user_eps {
                break;
            }
            if opts
                .deadline
                .is_some_and(|d| std::time::Instant::now() >= d)
            {
                break;
            }
            let prev_kkt = current_kkt;

            let pre_dual_step = final_sol.clone();
            crate::qp::refine_dual_lsq_irls(
                orig_problem,
                &mut final_sol,
                user_eps,
                IRLS_INNER_MAX_ITERS,
                opts.deadline,
            );
            crate::qp::zero_inactive_inequality_duals(orig_problem, &mut final_sol);
            crate::qp::project_duals_from_singleton_columns(orig_problem, &mut final_sol);
            crate::qp::refine_dual_projected_gradient(orig_problem, &mut final_sol, opts.deadline);
            crate::qp::refine_dual_worst_active_block(orig_problem, &mut final_sol, opts.deadline);
            let post_kkt_irls = kkt_residual_rel(
                &view,
                &final_sol.solution,
                &final_sol.dual_solution,
                &final_sol.bound_duals,
            );
            if post_kkt_irls < current_kkt {
                current_kkt = post_kkt_irls;
                let pre_z = final_sol.bound_duals.clone();
                crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
                let post_kkt_z = kkt_residual_rel(
                    &view,
                    &final_sol.solution,
                    &final_sol.dual_solution,
                    &final_sol.bound_duals,
                );
                if post_kkt_z <= current_kkt {
                    current_kkt = post_kkt_z;
                } else {
                    final_sol.bound_duals = pre_z;
                }
            } else {
                final_sol = pre_dual_step;
                break;
            }

            if current_kkt + REFIT_PROGRESS_EPS >= prev_kkt {
                break;
            }
        }

        current_kkt
    } else {
        kkt_residual_rel(
            &view,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        )
    };

    // (3) saddle-point Krylov refinement: K [dx; dy] = -[r_d; r_p] を古典的 IR で解く。
    // IPM 内部 Newton step の LDL 誤差 accumulate 由来の primal/dual feasibility を
    // refine する。残差は full-f64 (env REFINE_KKT_DD=1 で DD) で再計算。
    if !final_sol.solution.is_empty() && orig_problem.num_constraints > 0 && ipm_made_progress {
        const KRYLOV_MAX_ITERS: usize = 400;
        let user_eps = opts.ipm_eps();
        let target_pf = user_eps;
        if post_trace {
            let pres_pre = primal_residual_rel(&view, &final_sol.solution);
            let kkt_pre = kkt_residual_rel(
                &view,
                &final_sol.solution,
                &final_sol.dual_solution,
                &final_sol.bound_duals,
            );
            eprintln!(
                "POST_STAGE [pre saddle-point IR] pres_rel={:.3e} kkt_rel={:.3e}",
                pres_pre, kkt_pre
            );
        }
        let _refined = crate::qp::refine_kkt_iterative(
            orig_problem,
            &mut final_sol,
            KRYLOV_MAX_ITERS,
            target_pf,
            opts.deadline,
        );
        if post_trace {
            let pres_post = primal_residual_rel(&view, &final_sol.solution);
            let kkt_post = kkt_residual_rel(
                &view,
                &final_sol.solution,
                &final_sol.dual_solution,
                &final_sol.bound_duals,
            );
            eprintln!(
                "POST_STAGE [post saddle-point IR] refined_iters={} pres_rel={:.3e} kkt_rel={:.3e}",
                _refined, pres_post, kkt_post
            );
        }

        // (3b) KKT IR 後に pres > eps なら primal projection を 1 回追加。
        // 採用条件: pres 改善 AND kkt <= user_eps を厳守 (df 退行防止)。
        if allow_primal_projection
            && !opts
                .deadline
                .is_some_and(|d| std::time::Instant::now() >= d)
        {
            let pres_post_ir = primal_residual_rel(&view, &final_sol.solution);
            let kkt_post_ir = kkt_residual_rel(
                &view,
                &final_sol.solution,
                &final_sol.dual_solution,
                &final_sol.bound_duals,
            );
            if pres_post_ir > user_eps && kkt_post_ir <= user_eps {
                let pre_sol2 = final_sol.clone();
                crate::qp::refine_primal_lsq(orig_problem, &mut final_sol, opts.deadline);
                let post_pres2 = primal_residual_rel(&view, &final_sol.solution);
                if post_pres2 < pres_post_ir {
                    crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
                    let kkt_after2 = kkt_residual_rel(
                        &view,
                        &final_sol.solution,
                        &final_sol.dual_solution,
                        &final_sol.bound_duals,
                    );
                    if kkt_after2 > user_eps {
                        final_sol = pre_sol2;
                    } else if post_trace {
                        eprintln!("POST_STAGE [2nd primal proj] pre_pres={:.3e} post_pres={:.3e} kkt_after={:.3e} ACCEPT",
                            pres_post_ir, post_pres2, kkt_after2);
                    }
                } else {
                    final_sol = pre_sol2;
                }
            }
        }
    }

    let kkt_final = kkt_residual_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    let kkt_out = kkt_final;
    let _ = kkt;

    let pres = primal_residual_rel(&view, &final_sol.solution);
    let bv = bound_violation(orig_problem.bounds.as_slice(), &final_sol.solution);
    let comp = complementarity_residual_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    let dual_gap = compute_duality_gap_rel(orig_problem, &final_sol);

    // Invariant: 報告 objective は返却 x で計算。post-processing 後の整合性を保証。
    let objective_recomputed = {
        let qx = orig_problem
            .q
            .mat_vec_mul(&final_sol.solution)
            .unwrap_or_else(|_| vec![0.0; orig_problem.num_vars]);
        let xqx: f64 = qx
            .iter()
            .zip(final_sol.solution.iter())
            .map(|(&q, &x)| q * x)
            .sum();
        let cx: f64 = orig_problem
            .c
            .iter()
            .zip(final_sol.solution.iter())
            .map(|(&c, &x)| c * x)
            .sum();
        0.5 * xqx + cx + orig_problem.obj_offset
    };

    IpmOutcome {
        solution: final_sol.solution,
        dual_solution: final_sol.dual_solution,
        bound_duals: final_sol.bound_duals,
        objective: objective_recomputed,
        iterations: result.iterations,
        kkt_residual_rel: kkt_out,
        primal_residual_rel: pres,
        bound_violation: bv,
        complementarity_residual_rel: comp,
        duality_gap_rel: dual_gap,
        numerical_failure: false,
        infeasibility_status: None,
        is_locally_optimal,
    }
}

/// 元空間 双対ギャップ相対値: |primal_obj − dual_obj| / max(|p|, |d|, 1)。
/// QP 弱双対性 dual_obj = -1/2 x'Qx - b'y + lb'z_lb - ub'z_ub。rank-deficient Q の
/// 偽 Optimal (KKT 小だが gap 大) を弾く。FX 変数の bound 寄与は lb·停留性で解析的に置換。
fn compute_duality_gap_rel(
    problem: &crate::qp::QpProblem,
    result: &crate::problem::SolverResult,
) -> f64 {
    let n = problem.num_vars;
    if result.solution.len() != n {
        return f64::INFINITY;
    }
    let x = &result.solution;
    let qx = match problem.q.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };
    let aty: Vec<f64> = if problem.a.nrows > 0 && !result.dual_solution.is_empty() {
        match problem.a.transpose().mat_vec_mul(&result.dual_solution) {
            Ok(v) => v,
            Err(_) => return f64::INFINITY,
        }
    } else {
        vec![0.0_f64; n]
    };
    let xqx: f64 = qx.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
    let cx: f64 = problem.c.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
    let primal_obj = 0.5 * xqx + cx + problem.obj_offset;

    let mut by: f64 = 0.0;
    for (&bi, &yi) in problem.b.iter().zip(result.dual_solution.iter()) {
        by += bi * yi;
    }

    // FX (lb=ub) は postsolve で z_lb,z_ub が 0 埋めされるため、
    // val * (z_lb - z_ub) = val * (qx + c + aty) で解析的に置換。
    let mut bnd_term: f64 = 0.0;
    let mut lb_idx = 0_usize;
    let mut ub_idx = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();
        let is_fx = lb_finite && ub_finite && (lb - ub).abs() < crate::qp::FX_TOL;
        if is_fx {
            let stat_no_bnd = qx[j] + problem.c[j] + aty[j];
            bnd_term += lb * stat_no_bnd;
            if lb_finite {
                lb_idx += 1;
            }
            if ub_finite {
                ub_idx += 1;
            }
        } else {
            if lb_finite && lb_idx < result.bound_duals.len() {
                bnd_term += lb * result.bound_duals[lb_idx];
                lb_idx += 1;
            }
            if ub_finite && ub_idx < result.bound_duals.len() {
                bnd_term -= ub * result.bound_duals[ub_idx];
                ub_idx += 1;
            }
        }
    }
    let dual_obj = -0.5 * xqx - by + bnd_term + problem.obj_offset;
    let gap_abs = (primal_obj - dual_obj).abs();
    let denom = primal_obj.abs().max(dual_obj.abs()).max(1.0);
    if denom > 0.0 && gap_abs.is_finite() {
        gap_abs / denom
    } else {
        f64::INFINITY
    }
}
