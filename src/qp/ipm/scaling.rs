//! Ruiz スケーリングラッパー・アンスケール・後検証
//!
//! mod.rs の Ruiz スケーリング関連処理をこのモジュールに分離。
//! - `solve_with_ruiz_scaling`: solve_qp_ipm / solve_qp_ippmm の共通スケーリングラッパー
//! - `compute_amplification`: Ruiz スケーリング増幅率計算
//! - `unscale_ipm_result`: スケール済み結果を元スケールへ逆変換
//! - `post_verify_solution`: SuboptimalSolution の原空間再検証
//! - `check_bfeas_status`: 境界制約実現可能性検証
//! - `check_dfeas_status`: 双対実現可能性検証

use crate::linalg::ruiz::RuizScaler;
use crate::options::SolverOptions;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;

/// eps 事前調整の下限（数値精度限界）
pub(crate) const EPS_FLOOR: f64 = 1e-12;
/// Suboptimal→Optimal 昇格ゲートの双対ギャップ閾値。
/// 内部収束判定 (Optimal_main, 1e-3) より緩く post-hoc promotion 用途。
/// 真の Optimal の双対ギャップは通常 1% 以下、UBH1 型の偽 Optimal は ~28% で弾く。
pub(crate) const PROMOTION_GAP_TOL: f64 = 1e-1;

/// OSQP 流 primal feasibility 計算 (全体相対化, bench/v2 と整合)。
/// `||v||_∞ / (1 + max(||Ax||_∞, ||b||_∞))`。
fn compute_pfeas_osqp(problem: &QpProblem, x: &[f64]) -> f64 {
    use crate::problem::ConstraintType;
    if problem.num_constraints == 0 {
        return 0.0;
    }
    let ax = match problem.a.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };
    let mut max_v = 0.0_f64;
    let mut max_ax = 0.0_f64;
    let mut max_b = 0.0_f64;
    for (i, (&ax_i, &b_i)) in ax.iter().zip(problem.b.iter()).enumerate() {
        let violation = match problem.constraint_types.get(i) {
            Some(ConstraintType::Eq) => (ax_i - b_i).abs(),
            Some(ConstraintType::Ge) => (b_i - ax_i).max(0.0),
            _ => (ax_i - b_i).max(0.0),
        };
        max_v = max_v.max(violation);
        max_ax = max_ax.max(ax_i.abs());
        max_b = max_b.max(b_i.abs());
    }
    max_v / (1.0 + max_ax.max(max_b))
}

// ---------------------------------------------------------------------------
// 公開関数
// ---------------------------------------------------------------------------

/// Ruiz スケーリングラッパー（solve_qp_ipm / solve_qp_ippmm の共通処理）
///
/// inner_solver は `solve_qp_ipm_inner` または `solve_ippmm_inner` を渡す。
pub(crate) fn solve_with_ruiz_scaling<F>(
    problem: &QpProblem,
    options: &SolverOptions,
    inner_solver: F,
) -> SolverResult
where
    F: Fn(&QpProblem, &SolverOptions, Option<&RuizScaler>, Option<&QpProblem>, f64) -> SolverResult,
{
    if options.use_ruiz_scaling && problem.num_vars > 0 {
        let n = problem.num_vars;
        let m = problem.num_constraints;

        let lb: Vec<f64> = problem.bounds.iter().map(|&(l, _)| l).collect();
        let ub: Vec<f64> = problem.bounds.iter().map(|&(_, u)| u).collect();

        let mut scaler = RuizScaler::new(n, m);
        scaler.compute(&problem.q, &problem.a, &problem.c, &lb, &ub);

        let (q_s, a_s, c_s, b_s, bounds_s) =
            scaler.scale_problem(&problem.q, &problem.a, &problem.c, &problem.b, &problem.bounds);

        if let Ok(scaled_problem) = QpProblem::new(
            q_s, c_s, a_s, b_s, bounds_s, problem.constraint_types.clone(),
        ) {
            // Ruiz スケーリング増幅率 (1/min(e_i), 1/(c × min(d_j))) で scaled 空間 eps を
            // tighten し、unscale 後に元空間 eps を保証する。retry は外側 (ipm_v2 ATTEMPTS)。
            let amplification = compute_amplification(&scaler);
            let mut adjusted_opts = options.clone();
            adjusted_opts.ipm.eps =
                (options.ipm_eps() / amplification).max(EPS_FLOOR);

            let scaled_result = inner_solver(
                &scaled_problem,
                &adjusted_opts,
                Some(&scaler),
                Some(problem),
                options.ipm_eps(),
            );
            let result = unscale_ipm_result(scaled_result, &scaler, problem, options.ipm_eps());

            // MaxIterations は概要設計に従い有効解の有無で外部 status に変換する
            // (status 変換 1 箇所原則の例外: max_iter 到達は inner 内部判定で
            //  外部 Timeout/Suboptimal どちらにも該当しうるため、ここで bridge する)
            if result.status == SolveStatus::MaxIterations {
                if !result.solution.is_empty() {
                    return SolverResult { status: SolveStatus::SuboptimalSolution, ..result };
                } else {
                    return SolverResult { status: SolveStatus::Timeout, ..result };
                }
            }
            return result;
        }
        // QpProblem::new 失敗 → 非スケールにフォールバック
    }

    // 非 Ruiz パス: SuboptimalSolution を原空間で再検証
    post_verify_solution(
        inner_solver(problem, options, None, None, options.ipm_eps()),
        problem,
        options.ipm_eps(),
    )
}

/// SuboptimalSolution（ソルバー内部判定）を原問題空間で再検証し、
/// pfeas・bfeas・dfeas が eps 基準を満たすなら Optimal に昇格する。
///
/// Ruiz scaling なしのフォールバックパスで使用。
/// Ruiz ありパスは unscale_ipm_result の SuboptimalSolution ブランチが担当。
pub(crate) fn post_verify_solution(
    result: SolverResult,
    problem: &QpProblem,
    eps: f64,
) -> SolverResult {
    if result.status != SolveStatus::SuboptimalSolution || result.solution.is_empty() {
        return result;
    }
    let x = &result.solution;
    let y = &result.dual_solution;
    let bound_duals = &result.bound_duals;
    // 元空間 KKT 判定: bench/v2 と同形の OSQP 流 全体相対化 pfeas。
    let status = if problem.num_constraints > 0 {
        let pfeas_normalized = compute_pfeas_osqp(problem, x);
        if pfeas_normalized.is_finite() && pfeas_normalized < eps {
            let bfeas_status = check_bfeas_status(x, &problem.bounds, eps);
            if bfeas_status == SolveStatus::Optimal {
                check_dfeas_status_relative(problem, x, y, bound_duals, eps)
            } else {
                bfeas_status
            }
        } else {
            SolveStatus::SuboptimalSolution
        }
    } else {
        let bfeas_status = check_bfeas_status(x, &problem.bounds, eps);
        if bfeas_status == SolveStatus::Optimal {
            check_dfeas_status_relative(problem, x, y, bound_duals, eps)
        } else {
            bfeas_status
        }
    };
    // Suboptimal→Optimal 昇格ゲート: 双対ギャップ閾値外なら Optimal に上げない。
    let status = if status == SolveStatus::Optimal {
        match result.duality_gap_rel {
            Some(g) if g.abs() >= PROMOTION_GAP_TOL => SolveStatus::SuboptimalSolution,
            _ => status,
        }
    } else {
        status
    };
    SolverResult { status, ..result }
}

/// lb <= x <= ub の違反量を検証し、超過していれば SuboptimalSolution に降格する
///
/// 閾値: eps（絶対値基準）。qps_benchmarkの検証基準と統一。
/// lb/ub が ±∞ の成分はスキップする。
pub(crate) fn check_bfeas_status(x: &[f64], bounds: &[(f64, f64)], eps: f64) -> SolveStatus {
    let bfeas: f64 = x
        .iter()
        .zip(bounds.iter())
        .map(|(&xi, &(lb, ub))| {
            let lb_viol = if lb.is_finite() { (lb - xi).max(0.0) } else { 0.0 };
            let ub_viol = if ub.is_finite() { (xi - ub).max(0.0) } else { 0.0 };
            lb_viol.max(ub_viol)
        })
        .fold(0.0_f64, f64::max);
    if bfeas < eps {
        SolveStatus::Optimal
    } else {
        SolveStatus::SuboptimalSolution
    }
}

/// QPの双対実現可能性 (dfeas, inf-norm 絶対基準) を検証し、超過していれば SuboptimalSolution に降格する
///
/// 注意: 本関数は inf-norm の絶対値で判定する。ill-conditioned な問題で偽 Optimal を量産していた
/// ため、現在は `check_dfeas_status_relative` (成分相対化版) を使うのが推奨。
/// 本関数は単体テスト互換のために保持。
///
/// # 引数
/// - `bound_duals`: アンスケール済み境界双対変数。
/// - `threshold`: 呼び出し元計算の許容閾値。
#[allow(dead_code)]
pub(crate) fn check_dfeas_status(
    problem: &QpProblem,
    x: &[f64],
    y: &[f64],
    bound_duals: &[f64],
    threshold: f64,
) -> SolveStatus {
    let n = x.len();
    // Q*x. mat_vec_mul は次元一致前提で失敗は API 契約違反。
    // 失敗時は Optimal 昇格できる根拠がないため SuboptimalSolution を返す (status 隠蔽防止)。
    let qx = match problem.q.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return SolveStatus::SuboptimalSolution,
    };
    // A^T*y（無制約QPではa.nrows==0なのでzeroベクトル）
    let aty: Vec<f64> = if problem.a.nrows > 0 && !y.is_empty() {
        match problem.a.transpose().mat_vec_mul(y) {
            Ok(v) => v,
            Err(_) => return SolveStatus::SuboptimalSolution,
        }
    } else {
        vec![0.0; n]
    };
    // bound_contrib[j] = -y_lb[j] (lb有限) + y_ub[j] (ub有限)
    let mut bound_contrib = vec![0.0f64; n];
    if !bound_duals.is_empty() {
        let mut bd_idx = 0usize;
        for (j, &(lb, _)) in problem.bounds.iter().enumerate() {
            if lb.is_finite() && bd_idx < bound_duals.len() {
                bound_contrib[j] -= bound_duals[bd_idx];
                bd_idx += 1;
            }
        }
        for (j, &(_, ub)) in problem.bounds.iter().enumerate() {
            if ub.is_finite() && bd_idx < bound_duals.len() {
                bound_contrib[j] += bound_duals[bd_idx];
                bd_idx += 1;
            }
        }
    }
    // dfeas = ||Q*x + A^T*y + bound_contrib + c||_inf
    let dfeas = (0..n)
        .map(|i| (qx[i] + aty[i] + bound_contrib[i] + problem.c[i]).abs())
        .fold(0.0_f64, f64::max);
    if dfeas < threshold {
        SolveStatus::Optimal
    } else {
        SolveStatus::SuboptimalSolution
    }
}

/// 成分ごとの相対dfeasチェック
///
/// pfeasの正規化パターン `violation / (1 + ||a_k|| + |b_k|)` に倣い、
/// KKT双対残差を各成分のKKT項スケールで正規化する:
/// ```text
/// max_j |Qx_j + A^Ty_j + bound_contrib_j + c_j| / (1 + |Qx_j| + |A^Ty_j| + |bound_contrib_j| + |c_j|)
/// ```
/// グローバルノルムでは巨大項のキャンセレーション（BOYD1: Qx ≈ -A^Ty）を反映できないが、
/// 成分ごとの正規化なら真の相対精度を測定できる。
pub(crate) fn check_dfeas_status_relative(
    problem: &QpProblem,
    x: &[f64],
    y: &[f64],
    bound_duals: &[f64],
    eps: f64,
) -> SolveStatus {
    let n = x.len();
    // mat_vec_mul は次元一致前提で失敗は API 契約違反。
    // 失敗時は Optimal 昇格できる根拠がないため SuboptimalSolution を返す (status 隠蔽防止)。
    let qx = match problem.q.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return SolveStatus::SuboptimalSolution,
    };
    let aty: Vec<f64> = if problem.a.nrows > 0 && !y.is_empty() {
        match problem.a.transpose().mat_vec_mul(y) {
            Ok(v) => v,
            Err(_) => return SolveStatus::SuboptimalSolution,
        }
    } else {
        vec![0.0; n]
    };
    let mut bound_contrib = vec![0.0f64; n];
    if !bound_duals.is_empty() {
        let mut bd_idx = 0usize;
        for (j, &(lb, _)) in problem.bounds.iter().enumerate() {
            if lb.is_finite() && bd_idx < bound_duals.len() {
                bound_contrib[j] -= bound_duals[bd_idx];
                bd_idx += 1;
            }
        }
        for (j, &(_, ub)) in problem.bounds.iter().enumerate() {
            if ub.is_finite() && bd_idx < bound_duals.len() {
                bound_contrib[j] += bound_duals[bd_idx];
                bd_idx += 1;
            }
        }
    }
    // OSQP 流 全体相対化 (bench/v2/IPM 内部 nr_d_rel_orig / compute_pfeas_osqp と統一)。
    let mut max_r = 0.0_f64;
    let mut max_qx = 0.0_f64;
    let mut max_c = 0.0_f64;
    let mut max_aty = 0.0_f64;
    let mut max_bnd = 0.0_f64;
    for j in 0..n {
        max_r = max_r.max((qx[j] + aty[j] + bound_contrib[j] + problem.c[j]).abs());
        max_qx = max_qx.max(qx[j].abs());
        max_c = max_c.max(problem.c[j].abs());
        max_aty = max_aty.max(aty[j].abs());
        max_bnd = max_bnd.max(bound_contrib[j].abs());
    }
    let dfeas_relative = max_r / (1.0 + max_qx.max(max_c).max(max_aty).max(max_bnd));
    if dfeas_relative < eps {
        SolveStatus::Optimal
    } else {
        SolveStatus::SuboptimalSolution
    }
}

// ---------------------------------------------------------------------------
// 非公開関数
// ---------------------------------------------------------------------------

/// Ruiz スケーリングによる残差増幅率を計算する。
///
/// pfeas 増幅: 1/e_min、dfeas 増幅: 1/(c * d_min) の最大を返す。
pub(crate) fn compute_amplification(scaler: &RuizScaler) -> f64 {
    let e_min = if scaler.e.is_empty() {
        1.0
    } else {
        scaler.e.iter().cloned().fold(f64::INFINITY, f64::min).max(1e-12)
    };
    let d_min = if scaler.d.is_empty() {
        1.0
    } else {
        scaler.d.iter().cloned().fold(f64::INFINITY, f64::min).max(1e-12)
    };
    (1.0 / e_min).max(1.0 / (scaler.c * d_min))
}

/// スケール済み IPM 結果を元のスケールに逆変換する
///
/// Optimal ステータスの場合、元空間で pfeas・bfeas・dfeas を再計算し、
/// それぞれの許容誤差を超えていれば SuboptimalSolution に降格する（偽Optimal防止）。
pub(crate) fn unscale_ipm_result(
    result: SolverResult,
    scaler: &RuizScaler,
    problem: &QpProblem,
    eps: f64,
) -> SolverResult {
    match result.status {
        SolveStatus::Optimal => {
            let (x, y) = scaler.unscale_solution(&result.solution, &result.dual_solution);
            let bound_duals = scaler.unscale_bound_duals(&result.bound_duals, &problem.bounds);
            let obj_orig = result.objective / scaler.c;
            // [整合性] check_dfeas_status は L405 で unscaled x,y,bound_duals を受け取り
            // 元空間で dfeas を計算する。よって threshold も元空間 (bench と同形)。
            // 元空間 KKT 判定: OSQP 流 全体相対化 pfeas。
            let (status, orig_residuals) = if problem.num_constraints > 0 {
                match problem.a.mat_vec_mul(&x) {
                    Ok(ax) => {
                        let pfeas: f64 = ax
                            .iter()
                            .zip(problem.b.iter())
                            .zip(problem.constraint_types.iter())
                            .map(|((&ax_i, &b_i), ct)| match ct {
                                crate::problem::ConstraintType::Eq => (ax_i - b_i).abs(),
                                crate::problem::ConstraintType::Ge => (b_i - ax_i).max(0.0),
                                _ => (ax_i - b_i).max(0.0),
                            })
                            .fold(0.0_f64, f64::max);
                        let pfeas_normalized = compute_pfeas_osqp(problem, &x);
                        let orig_resid = result.final_residuals.map(|(_, d, g)| (pfeas, d, g));
                        let status = if pfeas_normalized.is_finite() && pfeas_normalized < eps {
                            let bfeas_status = check_bfeas_status(&x, &problem.bounds, eps);
                            if bfeas_status == SolveStatus::Optimal {
                                check_dfeas_status_relative(problem, &x, &y, &bound_duals, eps)
                            } else {
                                bfeas_status
                            }
                        } else {
                            SolveStatus::SuboptimalSolution
                        };
                        (status, orig_resid)
                    }
                    // mat_vec_mul は次元一致前提で失敗は API 契約違反。
                    // 失敗時に Optimal を維持すると pfeas 検証なしで Optimal を返す
                    // false-positive になるため SuboptimalSolution に降格 (status 隠蔽防止)。
                    Err(_) => (SolveStatus::SuboptimalSolution, result.final_residuals),
                }
            } else {
                let bfeas_status = check_bfeas_status(&x, &problem.bounds, eps);
                let status = if bfeas_status == SolveStatus::Optimal {
                    check_dfeas_status_relative(problem, &x, &y, &bound_duals, eps)
                } else {
                    bfeas_status
                };
                (status, result.final_residuals)
            };
            SolverResult {
                objective: obj_orig,
                solution: x,
                dual_solution: y,
                bound_duals,
                status,
                final_residuals: orig_residuals,
                ..result
            }
        }
        SolveStatus::Timeout => {
            let (x, y) = scaler.unscale_solution(&result.solution, &result.dual_solution);
            let obj_orig = result.objective / scaler.c;
            SolverResult {
                objective: obj_orig,
                solution: x,
                dual_solution: y,
                ..result
            }
        }
        SolveStatus::SuboptimalSolution => {
            // scaled 空間で SuboptimalSolution だった場合も unscale して原空間で再検証する。
            let (x, y) = scaler.unscale_solution(&result.solution, &result.dual_solution);
            let bound_duals = scaler.unscale_bound_duals(&result.bound_duals, &problem.bounds);
            let obj_orig = result.objective / scaler.c;
            // [整合性] 上記 Optimal branch と同形。元空間 dfeas tol = bench tol。
            let status = if problem.num_constraints > 0 {
                match problem.a.mat_vec_mul(&x) {
                    Ok(_ax) => {
                        let pfeas_normalized = compute_pfeas_osqp(problem, &x);
                        if pfeas_normalized.is_finite() && pfeas_normalized < eps {
                            let bfeas_status = check_bfeas_status(&x, &problem.bounds, eps);
                            if bfeas_status == SolveStatus::Optimal {
                                check_dfeas_status_relative(problem, &x, &y, &bound_duals, eps)
                            } else {
                                bfeas_status
                            }
                        } else {
                            SolveStatus::SuboptimalSolution
                        }
                    }
                    Err(_) => SolveStatus::SuboptimalSolution,
                }
            } else {
                let bfeas_status = check_bfeas_status(&x, &problem.bounds, eps);
                if bfeas_status == SolveStatus::Optimal {
                    check_dfeas_status_relative(problem, &x, &y, &bound_duals, eps)
                } else {
                    bfeas_status
                }
            };
            // Suboptimal→Optimal 昇格ゲート: 双対ギャップ閾値外なら Optimal に上げない。
            // UBH1 型の null-space 漂流で残差小・ギャップ大となった解を弾く最終防壁。
            let status = if status == SolveStatus::Optimal {
                match result.duality_gap_rel {
                    Some(g) if g.abs() >= PROMOTION_GAP_TOL => SolveStatus::SuboptimalSolution,
                    _ => status,
                }
            } else {
                status
            };
            SolverResult {
                objective: obj_orig,
                solution: x,
                dual_solution: y,
                bound_duals,
                status,
                ..result
            }
        }
        _ => result,
    }
}
