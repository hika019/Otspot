//! IPM 数値カーネル + 後処理 (Ruiz unscale, postsolve, bound clip, 元空間 KKT) の一貫処理。
//!
//! 設計原則:
//! - 入力は元 QpProblem と presolve 結果。reduced(scaled) は内部で扱う。
//! - 出力 IpmOutcome は **元空間** の解と残差のみを持つ。
//! - これにより `satisfies_eps(user_eps)` が常に元空間判定として機能する。
//!
//! 採用アルゴリズムは設計概要 (`docs/solver_overview_design.md`) に従い IPM/IPPMM のみ。
//! Active Set 法等は採用しない。post-processing は `refine_dual_lsq` (qp/mod.rs の
//! 既存関数、A^T y = -(Qx + c + bound_contrib) の最小二乗解) のみ使用する。

use crate::options::SolverOptions;
use crate::presolve::{
    postsolve_qp_with_dual_recovery, QpPresolveResult,
    recover_y_for_singleton_row_with_bound, bound_contrib_at_var,
};
use crate::presolve::qp_transforms::QpPostsolveStep;
use crate::problem::SolveStatus;
use crate::qp::problem::QpProblem;
use super::outcome::{IpmOutcome, ProblemView};
use super::kkt::{kkt_residual_rel, primal_residual_rel, bound_violation};

/// inner_solver の関数型 (現在は IP-PMM のみ)
pub type InnerSolver = fn(&QpProblem, &SolverOptions) -> crate::problem::SolverResult;

/// 1 回の IPPMM 呼出 + 後処理。元空間の解と残差を返す。
pub fn run_ipm(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    opts: &SolverOptions,
) -> IpmOutcome {
    run_ipm_with(orig_problem, presolve_result, opts, crate::qp::ipm::solve_qp_ippmm)
}

/// 内部 solver を引数に取る一般化 wrapper。
fn run_ipm_with(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    opts: &SolverOptions,
    inner_solver: InnerSolver,
) -> IpmOutcome {
    let reduced = &presolve_result.reduced;

    // Tier 2: presolve スケーリングを考慮した IPM eps の事前調整。
    //
    // 動機: presolve の LargeCoeffRowScale (Phase1: σ=1/√rmax) + constraint_precond
    //       (Phase2: σ=1/√rmax) で合成 σ_total = 1/orig_row_max 倍にスケーリングされる。
    //       IPM が scaled 空間で eps を達成しても、unscale で 1/σ_total 倍に絶対残差が
    //       増幅され、orig 空間で eps を満たさない (QPILOTNO で 6.77 桁悪化を実証)。
    //
    // 修正: IPM に渡す eps を eps × σ_total に厳しくして、unscale 後に orig eps を
    //       満たす状態を IPM 完了時点で作る。
    //
    // 適用条件: σ_total < SIGMA_TIGHTEN_THRESHOLD のときのみ (大きな増幅が予想される問題)。
    //   保守的な閾値: 0.01 (= 100 倍以上の増幅が予想される問題のみ調整)。
    //   全問題で適用すると IPM の収束時間が無闇に増える可能性あり。
    //
    // 限界: σ_total が極端に小さいと eps_scaled が f64 の達成精度限界 (cond×u)
    //       を下回り IPM が達成できない。その場合は IPM が SuboptimalSolution で
    //       正直に申告 (現状動作と同じ)、嘘の Optimal は出さない。
    const SIGMA_TIGHTEN_THRESHOLD: f64 = 0.01;
    let mut sigma_total = 1.0_f64;
    for step in presolve_result.postsolve_stack.steps.iter() {
        if let QpPostsolveStep::LargeCoeffRowScale { row_scales } = step {
            let local_min = row_scales.iter()
                .filter(|&&v| v > 0.0 && v.is_finite())
                .fold(f64::INFINITY, |a, &v| a.min(v));
            if local_min.is_finite() {
                sigma_total *= local_min;
            }
        }
    }
    // Ruiz scaler の row scale e[i] も含める。
    //
    // 動機: A_scaled = E × A × D, b_scaled = E × b の関係から、
    //       r_p_scaled[i] = e[i] × r_p_orig[i]
    //   →  ||r_p_orig||_inf ≤ ||r_p_scaled||_inf / min(e[i])
    //   IPM は scaled 空間で eps を要求するが、unscale 後 orig 空間で 1/min(e[i]) 倍に
    //   増幅される。LargeCoeffRowScale だけでなく Ruiz の e も Tier 2 で考慮すべき。
    //
    // 実証 (QPLIB_10034): IPM scaled pres_rel=3.6e-7 (eps=1e-6 達成) が orig 空間で
    //   3.083e-4 (308 倍悪化)。原因は Ruiz の e min。Tier 2 拡張で eps_scaled を更に
    //   厳しくして救う試行。
    if let Some(scaler) = &presolve_result.ruiz_scaler {
        let e_min = scaler.e.iter()
            .filter(|&&v| v > 0.0 && v.is_finite())
            .fold(f64::INFINITY, |a, &v| a.min(v));
        if e_min.is_finite() {
            sigma_total *= e_min;
        }
    }
    let opts_for_ipm: SolverOptions = if sigma_total < SIGMA_TIGHTEN_THRESHOLD {
        let mut tightened = opts.clone();
        let eps_orig = opts.ipm_eps();
        let eps_scaled = (eps_orig * sigma_total).max(f64::MIN_POSITIVE);
        // tolerance を None にして ipm.eps を直接効かせる
        tightened.tolerance = None;
        tightened.ipm.eps = eps_scaled;
        if std::env::var("POST_STAGE_TRACE").ok().as_deref() == Some("1") {
            eprintln!("POST_STAGE [IPM eps tighten] σ_total={:.3e} eps_orig={:.3e} → eps_scaled={:.3e}",
                sigma_total, eps_orig, eps_scaled);
        }
        tightened
    } else {
        opts.clone()
    };
    let mut result = inner_solver(reduced, &opts_for_ipm);

    // 確定的 Infeasible / Unbounded / NonConvex は IpmOutcome に保持して finalize_outcome に
    // 伝える。ここで握りつぶすと外部 status は Timeout に丸められて status 隠蔽になる。
    if matches!(
        result.status,
        SolveStatus::Infeasible | SolveStatus::Unbounded | SolveStatus::NonConvex(_)
    ) {
        return IpmOutcome::infeasibility(result.status);
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
            duality_gap_rel: f64::INFINITY,
            numerical_failure: true,
            infeasibility_status: None,
        };
    }

    // dual の post-process refinement (LSQ): scaled 空間で動かす方が IPM 出力との整合性が高い。
    if reduced.num_constraints > 0 {
        crate::qp::refine_dual_lsq(reduced, &mut result, opts.deadline);
    }

    // [DIAG] POST_STAGE_TRACE: 後処理 chain で primal/kkt 残差がどこで膨らむか観測
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
        let kkt_red = kkt_residual_rel(&view_red, &result.solution, &result.dual_solution, &result.bound_duals);
        // 絶対 pres と normalize denominator を計算
        let ax_red = reduced.a.mat_vec_mul(&result.solution).unwrap_or_else(|_| vec![0.0; reduced.num_constraints]);
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
        eprintln!("POST_STAGE [scaled+reduced after IPM+refine_dual_lsq] pres_rel={:.3e} pres_abs={:.3e} denom={:.3e} kkt_rel={:.3e} n={} m={}",
            pres_red, pres_abs_red, denom_red, kkt_red, reduced.num_vars, reduced.num_constraints);
    }

    // Ruiz unscale: presolve が scaling 適用済みの場合のみ。
    if let Some(scaler) = &presolve_result.ruiz_scaler {
        if post_trace {
            // Ruiz scaler の値域を観測 (scale factor が大きいと unscale で残差が増幅)。
            // RuizScaler 構造: c (objective scale), r (row scale), s (col scale) を保持。
            // 本診断では scaler 全要素を直接読まずアクセサ経由が必要だが、簡易に
            // unscale 前後の解ノルム比を出すことで実効的な scale 増幅率を測る。
            let x_pre_inf = result.solution.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            let y_pre_inf = result.dual_solution.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            let (x_unscaled, y_unscaled) = scaler.unscale_solution(&result.solution, &result.dual_solution);
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
        result.bound_duals = scaler.unscale_bound_duals(
            &result.bound_duals,
            &reduced.bounds,
        );
        if scaler.c.abs() > 1e-300 {
            result.objective /= scaler.c;
        }
    }
    if post_trace {
        // unscale 後 (まだ reduced space)、postsolve 前
        let view_red = ProblemView {
            q: &reduced.q,
            a: &reduced.a,
            c: &reduced.c,
            b: &reduced.b,
            bounds: &reduced.bounds,
            constraint_types: &reduced.constraint_types,
        };
        let pres_red = primal_residual_rel(&view_red, &result.solution);
        let kkt_red = kkt_residual_rel(&view_red, &result.solution, &result.dual_solution, &result.bound_duals);
        eprintln!("POST_STAGE [unscaled (still reduced)] pres_rel={:.3e} kkt_rel={:.3e}",
            pres_red, kkt_red);
    }

    if post_trace {
        // presolve transform 内訳を実測 (postsolve で primal が悪化する機序仮説の検証)。
        let mut n_fixed = 0; let mut n_singleton = 0; let mut n_empty = 0;
        let mut n_redundant = 0; let mut n_largescale = 0;
        let mut row_scales_for_diag: Option<Vec<f64>> = None;
        for step in presolve_result.postsolve_stack.steps.iter() {
            match step {
                QpPostsolveStep::FixedVar { .. } => n_fixed += 1,
                QpPostsolveStep::SingletonRow { .. } => n_singleton += 1,
                QpPostsolveStep::EmptyCol { .. } => n_empty += 1,
                QpPostsolveStep::RedundantRowFix { .. } => n_redundant += 1,
                QpPostsolveStep::LargeCoeffRowScale { row_scales } => {
                    n_largescale += 1;
                    row_scales_for_diag = Some(row_scales.clone());
                }
            }
        }
        eprintln!("POST_STAGE [presolve transforms] FixedVar={} SingletonRow={} EmptyCol={} RedundantRowFix={} LargeCoeffRowScale={} reduced_vars={} orig_vars={}",
            n_fixed, n_singleton, n_empty, n_redundant, n_largescale,
            reduced.num_vars, orig_problem.num_vars);
        // LargeCoeffRowScale の row_scales 統計と極端値を出力
        if let Some(scales) = &row_scales_for_diag {
            let n_scaled = scales.iter().filter(|&&s| (s - 1.0).abs() > 1e-12).count();
            let smin = scales.iter().fold(f64::INFINITY, |a, &v| a.min(v));
            let smax = scales.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v));
            // 最も小さい (= 最も増幅される) 5 row を抽出
            let mut indexed: Vec<(usize, f64)> = scales.iter().enumerate()
                .filter(|(_, &s)| (s - 1.0).abs() > 1e-12)
                .map(|(i, &s)| (i, s)).collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let top5: Vec<String> = indexed.iter().take(5)
                .map(|(i, s)| format!("row[{}]=σ:{:.3e}(amp:×{:.2e})", i, s, 1.0 / s))
                .collect();
            eprintln!("POST_STAGE [LargeCoeffRowScale] n_scaled={} σ_min={:.3e} σ_max={:.3e} smallest_5: {}",
                n_scaled, smin, smax, top5.join(", "));
        }
    }

    // postsolve: reduced 空間 → 元問題空間。dual recovery 付き (T1.4 系列の真因対処)。
    let mut final_sol = postsolve_qp_with_dual_recovery(presolve_result, &result, orig_problem);

    // bound_duals を元問題空間に remap
    if presolve_result.was_reduced {
        final_sol.bound_duals = crate::qp::remap_bound_duals_to_orig(
            presolve_result,
            &orig_problem.bounds,
            &final_sol.bound_duals,
        );
    }

    if post_trace {
        let view = ProblemView {
            q: &orig_problem.q, a: &orig_problem.a, c: &orig_problem.c, b: &orig_problem.b,
            bounds: &orig_problem.bounds, constraint_types: &orig_problem.constraint_types,
        };
        let pres = primal_residual_rel(&view, &final_sol.solution);
        let kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        // 絶対 pres と denom (orig 空間)
        use crate::problem::ConstraintType;
        let ax_orig = orig_problem.a.mat_vec_mul(&final_sol.solution).unwrap_or_else(|_| vec![0.0; orig_problem.num_constraints]);
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
        let ax = orig_problem.a.mat_vec_mul(&final_sol.solution)
            .unwrap_or_else(|_| vec![0.0; orig_problem.num_constraints]);
        let mut viol: Vec<(usize, f64)> = (0..orig_problem.num_constraints).map(|i| {
            let raw = ax[i] - orig_problem.b[i];
            let v = match orig_problem.constraint_types[i] {
                ConstraintType::Eq => raw.abs(),
                ConstraintType::Ge => if raw < 0.0 { -raw } else { 0.0 },
                ConstraintType::Le => if raw > 0.0 { raw } else { 0.0 },
            };
            (i, v)
        }).collect();
        viol.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top10: Vec<String> = viol.iter().take(10)
            .map(|(i, v)| format!("row[{}]={:.2e}", i, v)).collect();
        let total_viol: f64 = viol.iter().map(|(_, v)| v).sum();
        let top1_share = if total_viol > 0.0 { viol[0].1 / total_viol * 100.0 } else { 0.0 };
        let top10_share: f64 = viol.iter().take(10).map(|(_, v)| v).sum::<f64>()
            / total_viol.max(1e-300) * 100.0;
        eprintln!("POST_STAGE [violation distribution] top1_share={:.1}% top10_share={:.1}% top10: {}",
            top1_share, top10_share, top10.join(", "));
        // top-1 違反 row の内訳: A[top_row,:] の各項を計算、x が presolve fix か IPM か区別
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
                        // col_map で reduced 空間にあるか (= IPM が解いた変数) を判定
                        let is_reduced = presolve_result.col_map.get(col).copied().flatten().is_some();
                        row_terms.push((col, a_val, x_val, is_reduced));
                    }
                }
            }
            row_terms.sort_by(|a, b| (b.1 * b.2).abs().partial_cmp(&(a.1 * a.2).abs())
                .unwrap_or(std::cmp::Ordering::Equal));
            let top_str: Vec<String> = row_terms.iter().take(8)
                .map(|(c, a, x, red)| format!("col[{}]{}·{:.2e}·{:.2e}={:.2e}",
                    c, if *red { "(IPM)" } else { "(FIXED)" }, a, x, a * x))
                .collect();
            let sum: f64 = row_terms.iter().map(|(_, a, x, _)| a * x).sum();
            let n_fixed_in_row = row_terms.iter().filter(|(_, _, _, r)| !r).count();
            let n_ipm_in_row = row_terms.iter().filter(|(_, _, _, r)| *r).count();
            eprintln!("POST_STAGE [top-1 viol row {}] b={:.3e} A·x_sum={:.3e} viol={:.3e} (fixed_vars={} ipm_vars={}) top8: {}",
                top_row, orig_problem.b[top_row], sum, sum - orig_problem.b[top_row],
                n_fixed_in_row, n_ipm_in_row, top_str.join(", "));
            // top-1 viol row が LargeCoeffRowScale 対象か確認 (row_map で reduced index 取得)
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
            // reduced.A の同一行で scaling 後の max を見る (presolve scaled A の影響)
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
    for (xi, &(lb, ub)) in final_sol.solution.iter_mut().zip(orig_problem.bounds.iter()) {
        let pre = *xi;
        if lb.is_finite() { *xi = xi.max(lb); }
        if ub.is_finite() { *xi = xi.min(ub); }
        let amt = (pre - *xi).abs();
        if amt > 0.0 {
            clip_count_pre += 1;
            total_bound_clip = total_bound_clip.max(amt);
        }
    }
    if post_trace {
        let view = ProblemView {
            q: &orig_problem.q, a: &orig_problem.a, c: &orig_problem.c, b: &orig_problem.b,
            bounds: &orig_problem.bounds, constraint_types: &orig_problem.constraint_types,
        };
        let pres = primal_residual_rel(&view, &final_sol.solution);
        let kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        eprintln!("POST_STAGE [bounds clip applied] count={} max_amt={:.3e} pres_rel={:.3e} kkt_rel={:.3e}",
            clip_count_pre, total_bound_clip, pres, kkt);
    }


    // Stage 0: postsolve y/z 交互反復 (bound_duals が orig レイアウト確定後)。
    //
    // postsolve_qp_with_dual_recovery 内の forward pass は bound_contrib=0 仮定で
    // y[row] を復元する。boundary に張り付いた fixed/empty 列が存在する場合、
    // KKT 式の bound_contrib が非ゼロのため y[row] が wrong stays。
    //
    // 本反復で bound_duals (orig 空間) を refit_bound_duals_kkt で計算 → bound_contrib
    // を取得 → recover_y_for_singleton_row_with_bound で y を更新、を交互に行うことで
    // 連立を不動点で解く。実問題で 3 pass で収束する経験。
    if result.iterations > 0
        && presolve_result.was_reduced
        && !presolve_result.postsolve_stack.steps.is_empty()
        && final_sol.solution.len() == orig_problem.num_vars
        && final_sol.dual_solution.len() == orig_problem.num_constraints
    {
        /// 連鎖依存解消の固定反復回数。各 pass で y / z を交互更新する。
        const POSTSOLVE_RECOVERY_PASSES: usize = 5;
        for _pass in 0..POSTSOLVE_RECOVERY_PASSES {
            // (i) z (bound_duals) を current y に基づいて refit
            crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
            // (ii) y[row] を SingletonRow / RedundantRowFix step で更新
            //      bound_contrib を bound_duals から取得して KKT 完全式で解く
            for step in presolve_result.postsolve_stack.steps.iter() {
                let (row, col) = match step {
                    QpPostsolveStep::SingletonRow { row, col, .. }
                    | QpPostsolveStep::RedundantRowFix { row, col, .. } => (*row, *col),
                    _ => continue,
                };
                let bc = bound_contrib_at_var(&orig_problem.bounds, &final_sol.bound_duals, col);
                recover_y_for_singleton_row_with_bound(
                    row, col, orig_problem, &mut final_sol, bc,
                );
            }
        }
    }

    // 元空間で KKT 残差を計算 (元空間判定ベース)
    let view = ProblemView {
        q: &orig_problem.q,
        a: &orig_problem.a,
        c: &orig_problem.c,
        b: &orig_problem.b,
        bounds: &orig_problem.bounds,
        constraint_types: &orig_problem.constraint_types,
    };

    // 元空間 post-processing:
    //   stage A: x の primal projection (LISWET 系 T1.3)
    //   stage B: y を refine_dual_lsq で再計算
    //   stage C: z (bound_duals) を refit_bound_duals_kkt で再計算 (T1.2 QRECIPE)
    //
    // primal を動かすと dual との整合が崩れるため、stage A は stage A〜C の combined
    // (KKT max-rel + primal max-rel) で guard する。stage B/C は kkt_residual_rel で
    // 個別 guard。
    //
    // presolve が変数 fix / 行除去すると postsolve は dual_solution・bound_duals に 0 を
    // 埋め込み KKT が破壊される。Catastrophic 8 件 (QADLITTL/QBORE3D/QCAPRI/QETAMACR/
    // QFFFFF80/QPCBOEI1/QSEBA/QSHELL) は本後処理では完全復元できず、proper dual
    // postsolve (各 presolve 変換ごとに dual を記録) が必要 — 別 PR で対応 (#11)。
    // IPM が一度も iterate しなかった場合 (cancel_flag 即停止 / timeout=0 等) は
    // post-processing をスキップ。post-processing が冷状態 x=[0,..0] から analytic に
    // 最適解を計算してしまい、cancel_flag セマンティクス (Timeout 期待) を破壊するのを防ぐ。
    let ipm_made_progress = result.iterations > 0;

    // 大規模問題で refine_primal_lsq の AAT factorize が時間予算を圧迫するのを防ぐ。
    // BOYD2 (n+m≈280k) では LDL 因子化に分単位かかり、bench の external timeout を
    // 超える。実問題では LISWET (n+m≈20k) が現実的な上限。
    const PRIMAL_PROJECTION_SIZE_LIMIT: usize = 50_000;
    let problem_size = orig_problem.num_vars + orig_problem.num_constraints;
    let allow_primal_projection = problem_size <= PRIMAL_PROJECTION_SIZE_LIMIT;

    if post_trace {
        let pres0 = primal_residual_rel(&view, &final_sol.solution);
        let kkt0 = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        eprintln!("POST_STAGE [pre Stage A/B/C/D] pres_rel={:.3e} kkt_rel={:.3e}", pres0, kkt0);
    }

    let kkt = if !final_sol.solution.is_empty() && orig_problem.num_constraints > 0 && ipm_made_progress {
        // (A) x の primal projection — combined guard 付き (サイズ制限あり)。
        // LISWET 系で primal projection 後の dual LSQ refit は ill-conditioned A
        // (near-rank-deficient AAT) で破綻する (|y_new| が IPM y の 5e4 倍に膨張)
        // ため、ここでは primal x のみを射影し dual は IPM 値を保持する。
        // 多くの問題で primal projection は KKT 改善せず combined guard で revert
        // されるが、QRECIPE / 一部 borderline では効く。LISWET 系の precision floor
        // 突破は IPM 内部の深い数値改修 (Mehrotra centering / step size scheduling)
        // が必要 — 別タスク。
        if allow_primal_projection {
            let pre_x = final_sol.solution.clone();
            let pre_y_for_a = final_sol.dual_solution.clone();
            let pre_z_for_a = final_sol.bound_duals.clone();
            let pre_pres_a = primal_residual_rel(&view, &final_sol.solution);
            let pre_kkt_a = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            let pre_combined_a = pre_pres_a.max(pre_kkt_a);
            crate::qp::refine_primal_lsq(orig_problem, &mut final_sol, opts.deadline);
            crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
            let post_pres_a = primal_residual_rel(&view, &final_sol.solution);
            let post_kkt_a = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            let post_combined_a = post_pres_a.max(post_kkt_a);
            if post_combined_a > pre_combined_a {
                final_sol.solution = pre_x;
                final_sol.dual_solution = pre_y_for_a;
                final_sol.bound_duals = pre_z_for_a;
            }
        }

        // (B) y / z refit — fixed point に達するまで反復。
        //
        // 各反復:
        //   1. refine_dual_lsq: bound_duals を固定して y を LSQ で最適化
        //   2. refit_bound_duals_kkt: y を固定して z (bound_duals) を再計算
        // 反復間で y と z が双方向に依存するため、1 回では十分に reach されず、
        // QPILOTNO のような ill-conditioned 問題で kkt_rel が 4.6e-5 で停滞していた
        // (1 回 stage 後 3.4e-6、追加 stage で更に降りる余地あり)。
        //
        // 収束判定: kkt_rel の改善率が REFIT_CONVERGE_RATIO 未満になれば停止。
        // 各 stage は KKT-guard 付きで悪化なら revert するため、安全に反復可能。
        const REFIT_MAX_ITERS: usize = 8;
        const REFIT_CONVERGE_RATIO: f64 = 0.99;
        let mut current_kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        for _refit_iter in 0..REFIT_MAX_ITERS {
            if opts.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                break;
            }
            let prev_kkt = current_kkt;

            let pre_y = final_sol.dual_solution.clone();
            crate::qp::refine_dual_lsq(orig_problem, &mut final_sol, opts.deadline);
            let post_kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            if post_kkt <= current_kkt {
                current_kkt = post_kkt;
            } else {
                final_sol.dual_solution = pre_y;
            }

            let pre_z = final_sol.bound_duals.clone();
            crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
            let post_kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            if post_kkt <= current_kkt {
                current_kkt = post_kkt;
            } else {
                final_sol.bound_duals = pre_z;
            }

            // 改善が止まれば早期 break (固定点)
            if current_kkt >= prev_kkt * REFIT_CONVERGE_RATIO {
                break;
            }
        }
        current_kkt
    } else {
        kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals)
    };

    // (D) Krylov refinement on saddle-point KKT (Tier 1 (B) Krylov)
    //
    // 動機: LISWET 系の precision floor は LDL backward error eps×cond×‖dx‖ に由来。
    //   IPM 内部の Newton step ごとの LDL 誤差が accumulate して pf=1.5e-6 で停滞。
    //   IPM 収束後の (x, y, z) を初期推定として saddle-point K に対し古典的 IR を実行する。
    //   各 iter で full-f64 残差を計算するため、cond の影響を受けず eps·‖A‖ レベルまで refine 可能。
    //
    // 実行条件: ipm_made_progress AND サイズ制限内 AND constraints あり (refine_kkt_iterative 内 guard)。
    // 退行防止: 関数内で pf 厳密減少 AND df 許容内の guard、退行時 break。
    if !final_sol.solution.is_empty() && orig_problem.num_constraints > 0 && ipm_made_progress {
        // max iter: LISWET 系の cond が高く per-iter 収束率 ~0.96 なので 30 iter で
        // pf を 0.3 倍級まで引き下げる予算 (LISWET9 で 1.5e-6 → ~9e-7 想定)。
        // 1 iter あたり LDL solve 1 回なので n+m ≤ 50k なら数秒で完了。
        const KRYLOV_MAX_ITERS: usize = 30;
        let user_eps = opts.ipm_eps();
        // target_pf: ユーザー eps と整合 (PASS 判定相当)。pf < target で early exit。
        let target_pf = user_eps;
        if post_trace {
            let pres_pre_d = primal_residual_rel(&view, &final_sol.solution);
            let kkt_pre_d = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            eprintln!("POST_STAGE [pre Stage D Krylov] pres_rel={:.3e} kkt_rel={:.3e}", pres_pre_d, kkt_pre_d);
        }
        let _refined = crate::qp::refine_kkt_iterative(
            orig_problem, &mut final_sol, KRYLOV_MAX_ITERS, target_pf, opts.deadline,
        );
        if post_trace {
            let pres_post_d = primal_residual_rel(&view, &final_sol.solution);
            let kkt_post_d = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            eprintln!("POST_STAGE [post Stage D Krylov] refined_iters={} pres_rel={:.3e} kkt_rel={:.3e}",
                _refined, pres_post_d, kkt_post_d);
        }
    }

    let pres = primal_residual_rel(&view, &final_sol.solution);
    let bv = bound_violation(orig_problem.bounds.as_slice(), &final_sol.solution);
    let dual_gap = compute_duality_gap_rel(orig_problem, &final_sol);

    IpmOutcome {
        solution: final_sol.solution,
        dual_solution: final_sol.dual_solution,
        bound_duals: final_sol.bound_duals,
        objective: final_sol.objective,
        iterations: result.iterations,
        kkt_residual_rel: kkt,
        primal_residual_rel: pres,
        bound_violation: bv,
        duality_gap_rel: dual_gap,
        numerical_failure: false,
        infeasibility_status: None,
    }
}

/// 元空間 双対ギャップ相対値: |primal_obj - dual_obj| / max(|p|, |d|, 1)
///
/// QP の弱双対性: dual_obj = -1/2 x^T Q x - b^T y + lb^T z_lb - ub^T z_ub
///   (KKT 停留性 Qx + c + A^T y - z_lb + z_ub = 0 を Lagrangian に代入して導出)
/// 真の Optimal では gap → 0。rank-deficient Q で KKT 残差が小さくても gap が
/// 大きい偽 Optimal (UBH1: gap=9.49 で obj 54% 誤差) を弾くゲート。
///
/// FX (lb=ub) 変数は postsolve で bound_duals が 0 埋めされる慣例 + KKT 評価から
/// 除外される設計のため、result.bound_duals[j] には FX 変数の正しい dual が入って
/// いない。ここでは FX 変数の bound 寄与を「lb_j * 停留性」で解析的に置き換え、
/// 偽の gap 検出を防ぐ (BD-T2: FX 変数 z=3 で gap=1.0 → 0 に修正される)。
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

    // bnd_term = lb^T z_lb - ub^T z_ub
    // FX (lb=ub=val) は z_lb_j, z_ub_j が postsolve で 0 埋め (refit でも更新されない)
    // のため、解析的に val * net_z_at_j (= val * -(qx+c+aty)) で置換する。
    let mut bnd_term: f64 = 0.0;
    let mut lb_idx = 0_usize;
    let mut ub_idx = problem.bounds.iter().filter(|&&(lb, _)| lb.is_finite()).count();
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();
        let is_fx = lb_finite && ub_finite && (lb - ub).abs() < crate::qp::FX_TOL;
        if is_fx {
            // FX: lb_j * z_lb_j - ub_j * z_ub_j = val * (z_lb - z_ub)。
            // bound_contrib[j] = -z_lb + z_ub = -(qx + c + aty) (停留性) なので
            //   val * (z_lb - z_ub) = -val * bound_contrib = val * (qx + c + aty)
            let stat_no_bnd = qx[j] + problem.c[j] + aty[j];
            bnd_term += lb * stat_no_bnd;
            // bound_duals layout 上 idx は進める (FX 用 slot は使わない)
            if lb_finite { lb_idx += 1; }
            if ub_finite { ub_idx += 1; }
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
    if denom > 0.0 && gap_abs.is_finite() { gap_abs / denom } else { f64::INFINITY }
}
