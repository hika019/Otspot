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

    // presolve スケーリングを考慮した IPM eps の事前調整。
    //
    // presolve の LargeCoeffRowScale + Ruiz E (行) / Ruiz D・c (列) で問題が σ 倍に
    // 縮められると、unscale 時に残差が 1/σ 倍に増幅される。
    //   r_p_scaled[i] = e[i] × r_p_orig[i]              → ||r_p_orig|| ≤ ||r_p_scaled|| / e_min
    //   r_d_scaled[j] = (c × d[j]) × r_d_orig[j]        → ||r_d_orig|| ≤ ||r_d_scaled|| / (c × d_min)
    // primal 側 e_min と LargeCoeffRowScale の積、dual 側 c × d_min の小さい方を
    // sigma_total として、IPM に渡す eps を `user_eps × sigma_total` に厳しくする。
    //
    // 限界: σ_total が極端に小さいと eps_scaled が cond×u 限界を下回り IPM が達成できない。
    // その場合は IPM が SuboptimalSolution で正直に申告し、Optimal を偽装しない。

    // 行ごとの primal scale の積 (LargeCoeffRowScale × Ruiz E の合成) の最小値で primal
    // 増幅率を推定。dual 増幅率は Ruiz の (c × d_min)。両者の小さい方を採用。
    let mut primal_row_scale_min = 1.0_f64;
    for step in presolve_result.postsolve_stack.steps.iter() {
        if let QpPostsolveStep::LargeCoeffRowScale { row_scales } = step {
            let local_min = row_scales.iter()
                .filter(|&&v| v > 0.0 && v.is_finite())
                .fold(f64::INFINITY, |a, &v| a.min(v));
            if local_min.is_finite() {
                primal_row_scale_min *= local_min;
            }
        }
    }
    let mut dual_col_scale_min = f64::INFINITY;
    if let Some(scaler) = &presolve_result.ruiz_scaler {
        let e_min = scaler.e.iter()
            .filter(|&&v| v > 0.0 && v.is_finite())
            .fold(f64::INFINITY, |a, &v| a.min(v));
        if e_min.is_finite() {
            primal_row_scale_min *= e_min;
        }
        let d_min = scaler.d.iter()
            .filter(|&&v| v > 0.0 && v.is_finite())
            .fold(f64::INFINITY, |a, &v| a.min(v));
        if d_min.is_finite() && scaler.c.is_finite() && scaler.c > 0.0 {
            dual_col_scale_min = scaler.c * d_min;
        }
    }
    let sigma_total = primal_row_scale_min.min(dual_col_scale_min);
    // sigma_total < 1.0 (= 任意のスケール縮小あり) なら常に eps tighten を適用する。
    // 旧実装は < 0.01 のみ tighten していたが、sigma=0.05 級の中程度スケーリングでも
    // unscale で 20x 増幅されて元空間 eps を超える事象がある (中間 KKT が緩むのを防ぐ)。
    let opts_for_ipm: SolverOptions = if sigma_total < 1.0 && sigma_total > 0.0 {
        let mut tightened = opts.clone();
        let eps_orig = opts.ipm_eps();
        let eps_scaled = (eps_orig * sigma_total).max(f64::MIN_POSITIVE);
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

    // dual の post-process refinement (LSQ) は **元空間に戻してから** Stage B/C で行う。
    //
    // scaled 空間 (Ruiz 適用後) で refine_dual_lsq を回すと、ill-conditioned 問題で
    // 「scaled 空間では LSQ-y が IPM-y より小さい残差を持つが、unscale 後の元空間では
    //  LSQ-y のほうが残差が大きくなる」現象が起きる (QPILOTNO で実測)。原因は LSQ が
    // L2 ノルムを最小化するため、スケール固有の residual 分布最適化に過度に当たる。
    // IPM-y は元空間で正しい構造を持つので、scaled での "improvement" を捨てて、
    // unscale 後に元空間で 1 度だけ LSQ refine + Stage B/C 反復で詰めるほうが安全。
    //
    // 大規模問題で AAT factorize が分単位かかる遅延も同時に回避する
    // (Stage B/C で必要に応じて 1 回 factorize するだけで済む)。

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
        eprintln!("POST_STAGE [IPM exit (scaled+reduced)] pres_rel={:.3e} pres_abs={:.3e} denom={:.3e} kkt_rel={:.3e} n={} m={}",
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

    // postsolve: reduced 空間 → 元問題空間。eliminated 行 / 固定変数の dual 復元込み。
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
        // 純粋 postsolve (dual recovery + remap) 直後の元空間残差。
        // 以後の bounds clip / 後処理が postsolve 段階の精度をどう動かすかの基準値。
        let view = ProblemView {
            q: &orig_problem.q, a: &orig_problem.a, c: &orig_problem.c, b: &orig_problem.b,
            bounds: &orig_problem.bounds, constraint_types: &orig_problem.constraint_types,
        };
        let pres_post = primal_residual_rel(&view, &final_sol.solution);
        let kkt_post = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        eprintln!("POST_STAGE [postsolve+remap_bd (orig space)] pres_rel={:.3e} kkt_rel={:.3e}",
            pres_post, kkt_post);
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


    // 元空間 dual の一括復元: postsolve_qp_with_dual_recovery は SingletonRow /
    // RedundantRowFix の y[row] を col_first の停留性のみで復元するが、その row が
    // 関与する他の固定 col の停留性は z で別途吸収しないと壊れたままになる。
    // ここで refine_dual_lsq を 1 度走らせ、x と現在の z を固定して y を LSQ-optimal に
    // 更新する。AAT factorize が小〜中規模 (n+m ≤ LSQ_DUAL_SIZE_LIMIT) でだけ走る。
    if presolve_result.was_reduced
        && !presolve_result.postsolve_stack.steps.is_empty()
        && final_sol.solution.len() == orig_problem.num_vars
        && final_sol.dual_solution.len() == orig_problem.num_constraints
    {
        let view0 = ProblemView {
            q: &orig_problem.q, a: &orig_problem.a, c: &orig_problem.c, b: &orig_problem.b,
            bounds: &orig_problem.bounds, constraint_types: &orig_problem.constraint_types,
        };
        const POST_LSQ_MAX_PASSES: usize = 5;
        const POST_LSQ_CONVERGE_RATIO: f64 = 0.5;
        let mut prev = kkt_residual_rel(&view0, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        for pass in 0..POST_LSQ_MAX_PASSES {
            crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
            crate::qp::refine_dual_lsq(orig_problem, &mut final_sol, opts.deadline);
            crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
            let cur = kkt_residual_rel(&view0, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            if post_trace {
                eprintln!("POST_STAGE [postsolve dual_lsq pass {}] kkt_rel={:.3e}", pass, cur);
            }
            if cur >= prev * POST_LSQ_CONVERGE_RATIO {
                break;
            }
            prev = cur;
        }
    }

    // Stage 0: postsolve y/z 交互反復 (bound_duals が orig レイアウト確定後)。
    // 上の一括 LSQ で残った fixed-row dofs (SingletonRow/RedundantRowFix) を col_first
    // の停留性で正確に締める。
    if result.iterations > 0
        && presolve_result.was_reduced
        && !presolve_result.postsolve_stack.steps.is_empty()
        && final_sol.solution.len() == orig_problem.num_vars
        && final_sol.dual_solution.len() == orig_problem.num_constraints
    {
        /// 連鎖依存解消用の最大反復回数。各 pass で z (refit) → y (recover_y_with_bound)
        /// を交互更新する。改善が STAGE0_CONVERGE_RATIO 未満で停滞したら早期終了。
        const STAGE0_MAX_PASSES: usize = 16;
        const STAGE0_CONVERGE_RATIO: f64 = 0.99;
        let view0 = ProblemView {
            q: &orig_problem.q, a: &orig_problem.a, c: &orig_problem.c, b: &orig_problem.b,
            bounds: &orig_problem.bounds, constraint_types: &orig_problem.constraint_types,
        };
        let mut prev_kkt = kkt_residual_rel(&view0, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        for pass in 0..STAGE0_MAX_PASSES {
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
            let cur_kkt = kkt_residual_rel(&view0, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            if post_trace {
                eprintln!("POST_STAGE [postsolve recovery pass {}] kkt_rel={:.3e}", pass, cur_kkt);
            }
            if cur_kkt >= prev_kkt * STAGE0_CONVERGE_RATIO {
                break;
            }
            prev_kkt = cur_kkt;
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

    // 元空間 post-processing は 3 段階で行う:
    //   1. primal projection: x を violating 制約方向に最小ノルム射影 (refine_primal_lsq)
    //   2. dual / bound-dual refit: y と z を交互に LSQ で再最適化
    //   3. saddle-point Krylov refinement: K [x; y] = -[r_d; r_p] を IR で解く
    //
    // primal projection は y との整合を崩しうるため、primal/KKT 双方の max で guard。
    // dual/bound 個別 refit は kkt_residual_rel で個別 guard。
    //
    // IPM が一度も iterate しなかった場合 (cancel_flag 即停止 / timeout=0 等) は
    // 後処理をスキップ: 冷状態 x=[0,..,0] から後処理が独自の analytic な解を作って
    // しまい、外部 cancel/Timeout セマンティクスを破壊するのを防ぐ。
    let ipm_made_progress = result.iterations > 0;

    // refine_primal_lsq / refine_kkt_iterative の AAT/K factorize が時間予算を
    // 圧迫しないようサイズ上限を設ける。LDL 因子化が分単位かかる規模では skip。
    const PRIMAL_PROJECTION_SIZE_LIMIT: usize = 50_000;
    let problem_size = orig_problem.num_vars + orig_problem.num_constraints;
    let allow_primal_projection = problem_size <= PRIMAL_PROJECTION_SIZE_LIMIT;

    if post_trace {
        let pres0 = primal_residual_rel(&view, &final_sol.solution);
        let kkt0 = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        eprintln!("POST_STAGE [pre post-processing] pres_rel={:.3e} kkt_rel={:.3e}", pres0, kkt0);
    }

    // 既に IPM で eps を満たしている場合、primal projection と dual/bound refit は無駄
    // (改善余地なし)。大規模問題では LSQ が秒単位かかるため skip 必須。
    let user_eps_for_skip = opts.ipm_eps();
    let kkt_already_pass = if !final_sol.solution.is_empty() && orig_problem.num_constraints > 0 {
        let kkt0 = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        let pres0 = primal_residual_rel(&view, &final_sol.solution);
        kkt0 < user_eps_for_skip && pres0 < user_eps_for_skip
    } else {
        false
    };

    let kkt = if !final_sol.solution.is_empty() && orig_problem.num_constraints > 0 && ipm_made_progress && !kkt_already_pass {
        // (1) primal projection: 違反制約に対して x を最小ノルム射影。
        //     primal/KKT 合算の guard で悪化時 revert。near-rank-deficient AAT で
        //     LSQ y refit が膨張する系統では primal のみ動かし dual は IPM 値を維持する。
        if allow_primal_projection {
            let pre_x = final_sol.solution.clone();
            let pre_y = final_sol.dual_solution.clone();
            let pre_z = final_sol.bound_duals.clone();
            let pre_pres = primal_residual_rel(&view, &final_sol.solution);
            let pre_kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            let pre_combined = pre_pres.max(pre_kkt);
            crate::qp::refine_primal_lsq(orig_problem, &mut final_sol, opts.deadline);
            crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
            let post_pres = primal_residual_rel(&view, &final_sol.solution);
            let post_kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            if post_pres.max(post_kkt) > pre_combined {
                final_sol.solution = pre_x;
                final_sol.dual_solution = pre_y;
                final_sol.bound_duals = pre_z;
            }
        }

        // (2) y / z 交互 refit — fixed point に達するまで反復。
        //
        // 各反復:
        //   - refine_dual_lsq: bound_duals 固定で y を LSQ 最適化
        //   - refit_bound_duals_kkt: y 固定で z (bound_duals) を KKT 停留性から再計算
        // 双方向依存するため 1 回では届かず、ill-conditioned 系では数回反復で
        // kkt_rel が桁単位で減ることがある。
        //
        // 収束判定: 改善率が REFIT_CONVERGE_RATIO 未満で停止。各 step は KKT-guard 付き
        // で悪化時 revert するため安全に反復できる。
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

        // 標準 LSQ refine が componentwise eps を満たさないなら IRLS で L∞ 風の y を試す。
        // 改善した場合は z refit + 再度 IRLS のループを回し fixed point に達するまで反復。
        let user_eps = opts.ipm_eps();
        const IRLS_OUTER_MAX_PASSES: usize = 5;
        const IRLS_INNER_MAX_ITERS: usize = 30;
        for _outer_pass in 0..IRLS_OUTER_MAX_PASSES {
            if current_kkt <= user_eps { break; }
            if opts.deadline.is_some_and(|d| std::time::Instant::now() >= d) { break; }
            let prev_kkt = current_kkt;

            let pre_y = final_sol.dual_solution.clone();
            crate::qp::refine_dual_lsq_irls(
                orig_problem, &mut final_sol, user_eps, IRLS_INNER_MAX_ITERS, opts.deadline,
            );
            let post_kkt_irls = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            if post_kkt_irls < current_kkt {
                current_kkt = post_kkt_irls;
                // y が変わった → z 再 refit で停留性を取り直し
                let pre_z = final_sol.bound_duals.clone();
                crate::qp::refit_bound_duals_kkt(orig_problem, &mut final_sol);
                let post_kkt_z = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
                if post_kkt_z <= current_kkt {
                    current_kkt = post_kkt_z;
                } else {
                    final_sol.bound_duals = pre_z;
                }
            } else {
                final_sol.dual_solution = pre_y;
                break; // IRLS が改善できなければ outer loop も終了
            }

            // outer pass の収束判定 (10% 以上改善あれば継続)
            if current_kkt >= prev_kkt * 0.9 { break; }
        }

        current_kkt
    } else {
        kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals)
    };

    // (3) saddle-point Krylov refinement: K [dx; dy] = -[r_d; r_p] を古典的 IR で解く。
    //
    // IPM 内部の Newton step ごとの LDL 誤差が accumulate して pf が IPM 段階で
    // 下げきれない問題で、IPM 収束後の (x, y, z) を初期推定として saddle-point K
    // に対する IR を回す。残差を full-f64 (or DD env=REFINE_KKT_DD=1) で再計算し、
    // cond の影響を受けず eps·‖A‖ レベルまで refine する。
    //
    // 実行条件: ipm_made_progress (cold-start を避ける) AND constraints あり。
    // refine_kkt_iterative 内で size 制限・退行 guard・deadline を見る。
    if !final_sol.solution.is_empty() && orig_problem.num_constraints > 0 && ipm_made_progress {
        // per-iter 収束率は cond に依存するが経験的に ~0.96 級で 30 iter で 1 桁。
        // 1 iter あたり LDL solve 1 回なので n+m ≤ 50k で数秒程度。
        const KRYLOV_MAX_ITERS: usize = 30;
        let user_eps = opts.ipm_eps();
        let target_pf = user_eps;
        if post_trace {
            let pres_pre = primal_residual_rel(&view, &final_sol.solution);
            let kkt_pre = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            eprintln!("POST_STAGE [pre saddle-point IR] pres_rel={:.3e} kkt_rel={:.3e}", pres_pre, kkt_pre);
        }
        let _refined = crate::qp::refine_kkt_iterative(
            orig_problem, &mut final_sol, KRYLOV_MAX_ITERS, target_pf, opts.deadline,
        );
        if post_trace {
            let pres_post = primal_residual_rel(&view, &final_sol.solution);
            let kkt_post = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            eprintln!("POST_STAGE [post saddle-point IR] refined_iters={} pres_rel={:.3e} kkt_rel={:.3e}",
                _refined, pres_post, kkt_post);
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
/// 偽の gap 検出を防ぐ。
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
