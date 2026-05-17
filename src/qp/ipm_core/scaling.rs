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

/// IPM target eps の f64 表現可能な絶対下限。`user_eps / amplification` がこの
/// 値を下回ると target が完全に表現不能 (denormal/zero) になるため抑える。
pub(crate) const EPS_FLOOR: f64 = f64::EPSILON;

/// Suboptimal → Optimal 昇格時の双対ギャップ閾値 (真の Optimal は通常 < 1%、
/// 偽 Optimal は >> 10% で弾く)。
pub(crate) const PROMOTION_GAP_TOL: f64 = 1e-1;

/// OSQP 流 primal feasibility 計算 (全体相対化, bench/v2 と整合)。
/// `||v||_∞ / (1 + max(||Ax||_∞, ||b||_∞))`。A·x は DD で積算
/// (cancellation で違反を見逃さないため)。
fn compute_pfeas_osqp(problem: &QpProblem, x: &[f64]) -> f64 {
    use crate::problem::ConstraintType;
    use twofloat::TwoFloat;
    if problem.num_constraints == 0 {
        return 0.0;
    }
    let m = problem.a.nrows;
    let zero_dd = TwoFloat::from(0.0);
    let mut ax_dd: Vec<TwoFloat> = vec![zero_dd; m];
    for col in 0..problem.a.ncols {
        let xv = x[col];
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            ax_dd[problem.a.row_ind[k]] = ax_dd[problem.a.row_ind[k]] + TwoFloat::new_mul(problem.a.values[k], xv);
        }
    }
    let mut max_v = 0.0_f64;
    let mut max_ax = 0.0_f64;
    let mut max_b = 0.0_f64;
    for (i, (ax_dd_i, &b_i)) in ax_dd.iter().zip(problem.b.iter()).enumerate() {
        let raw_dd = *ax_dd_i - TwoFloat::from(b_i);
        let raw = f64::from(raw_dd);
        let violation = match problem.constraint_types.get(i) {
            Some(ConstraintType::Eq) => raw.abs(),
            Some(ConstraintType::Ge) => (-raw).max(0.0),
            _ => raw.max(0.0),
        };
        max_v = max_v.max(violation);
        max_ax = max_ax.max(f64::from(*ax_dd_i).abs());
        max_b = max_b.max(b_i.abs());
    }
    max_v / (1.0 + max_ax.max(max_b))
}

// ---------------------------------------------------------------------------
// 公開関数
// ---------------------------------------------------------------------------

/// Ruiz スケーリングラッパー（solve_qp_ipm / solve_qp_ippmm の共通処理）
///
/// inner_solver は `solve_ippmm_inner` を渡す。
pub(crate) fn solve_with_ruiz_scaling<F>(
    problem: &QpProblem,
    options: &SolverOptions,
    inner_solver: F,
) -> SolverResult
where
    F: Fn(&QpProblem, &SolverOptions, f64) -> SolverResult,
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

        if let Ok(mut scaled_problem) = QpProblem::new(
            q_s, c_s, a_s, b_s, bounds_s, problem.constraint_types.clone(),
        ) {
            scaled_problem.obj_offset = problem.obj_offset;
            // scaled 空間 eps を amp 倍だけ tighten し、unscale 後に元空間 eps を保証。
            let amplification = compute_amplification(&scaler);
            let mut adjusted_opts = options.clone();
            adjusted_opts.ipm.eps =
                (options.ipm_eps() / amplification).max(EPS_FLOOR);

            let scaled_result = inner_solver(
                &scaled_problem,
                &adjusted_opts,
                options.ipm_eps(),
            );
            let result = unscale_ipm_result(scaled_result, &scaler, problem, options.ipm_eps());

            // MaxIterations は外部 Timeout/Suboptimal に bridge する。
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
        inner_solver(problem, options, options.ipm_eps()),
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

/// 成分ごとの相対 KKT 残差 + complementarity チェック。
///
/// stationarity (`|Qx + A^Ty + bound_contrib + c|` 成分相対化) に加えて
/// complementarity (`y_i · slack_i`, `z_j · (x_j - bnd_j)`) を成分相対化で評価し、
/// 両方 eps 以下の場合のみ Optimal を返す。
///
/// stationarity だけ見るとLISWET9/YAOのように feasible だが optimal でない点が
/// Optimal と判定される (inactive 制約の y が大、slack が小、積が中程度)。
/// 同形の正規化は `ipm_solver::kkt::complementarity_residual_rel` と整合。
pub(crate) fn check_dfeas_status_relative(
    problem: &QpProblem,
    x: &[f64],
    y: &[f64],
    bound_duals: &[f64],
    eps: f64,
) -> SolveStatus {
    use twofloat::TwoFloat;
    let n = x.len();
    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = x[col];
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            qx_dd[problem.q.row_ind[k]] = qx_dd[problem.q.row_ind[k]] + TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let mut aty_dd: Vec<TwoFloat> = vec![zero_dd; n];
    if problem.a.nrows > 0 && !y.is_empty() {
        for col in 0..n {
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                aty_dd[col] = aty_dd[col] + TwoFloat::new_mul(problem.a.values[k], y[problem.a.row_ind[k]]);
            }
        }
    }
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
    // 成分ごとの相対化（bench の dfeas_rel_componentwise と整合）。
    // 全体最大値スケールでは 1 成分のみ大きく外れた残差をマスクするため、
    // 各成分 j を独立に正規化し max を取る。
    let mut dfeas_relative = 0.0_f64;
    for j in 0..n {
        let r_dd = qx_dd[j] + aty_dd[j] + TwoFloat::from(bound_contrib[j]) + TwoFloat::from(problem.c[j]);
        let r = f64::from(r_dd).abs();
        let scale_j = 1.0
            + f64::from(qx_dd[j]).abs()
            + f64::from(aty_dd[j]).abs()
            + bound_contrib[j].abs()
            + problem.c[j].abs();
        dfeas_relative = dfeas_relative.max(r / scale_j);
    }
    if dfeas_relative >= eps {
        return SolveStatus::SuboptimalSolution;
    }
    // complementarity (4 つ目の KKT 条件)。stationarity だけ通っても
    // y·slack や z·(x-bnd) が崩れた解は optimal でない。
    let comp = complementarity_relative(problem, x, y, bound_duals);
    if comp < eps {
        SolveStatus::Optimal
    } else {
        SolveStatus::SuboptimalSolution
    }
}

/// 元空間 complementarity 残差 (問題全体スケール正規化)。
/// `ipm_solver::kkt::complementarity_residual_rel` と同形の双対対スケール正規化を使う。
fn complementarity_relative(
    problem: &QpProblem,
    x: &[f64],
    y: &[f64],
    bound_duals: &[f64],
) -> f64 {
    use crate::problem::ConstraintType;
    use twofloat::TwoFloat;
    let zero_dd = TwoFloat::from(0.0);

    let m = problem.a.nrows;
    let ax_dd: Vec<TwoFloat> = if m > 0 {
        let mut ax = vec![zero_dd; m];
        for col in 0..problem.a.ncols {
            let xv = x[col];
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                ax[problem.a.row_ind[k]] =
                    ax[problem.a.row_ind[k]] + TwoFloat::new_mul(problem.a.values[k], xv);
            }
        }
        ax
    } else {
        Vec::new()
    };

    let yb: f64 = y.iter().zip(problem.b.iter()).map(|(&yi, &bi)| yi * bi).sum();
    let yax: f64 = y
        .iter()
        .zip(ax_dd.iter())
        .map(|(&yi, &ax_dd_i)| yi * f64::from(ax_dd_i))
        .sum();
    let zx: f64 = {
        let mut s = 0.0_f64;
        let mut idx = 0_usize;
        for (j, &(lb, _)) in problem.bounds.iter().enumerate() {
            if lb.is_finite() && idx < bound_duals.len() {
                s += bound_duals[idx] * x[j];
                idx += 1;
            }
        }
        for (j, &(_, ub)) in problem.bounds.iter().enumerate() {
            if ub.is_finite() && idx < bound_duals.len() {
                s += bound_duals[idx] * x[j];
                idx += 1;
            }
        }
        s
    };
    let cx: f64 = problem.c.iter().zip(x.iter()).map(|(&c, &xi)| c * xi).sum();
    let xqx: f64 = {
        let qx = problem.q.mat_vec_mul(x).unwrap_or_else(|_| vec![0.0; x.len()]);
        qx.iter().zip(x.iter()).map(|(&q, &xi)| q * xi).sum()
    };
    let scale = 1.0
        + yb.abs()
        + yax.abs()
        + zx.abs()
        + cx.abs()
        + (0.5 * xqx).abs();

    let mut max_abs = 0.0_f64;

    if m > 0 && !y.is_empty() {
        for (i, ct) in problem.constraint_types.iter().enumerate() {
            let slack_dd = match ct {
                ConstraintType::Eq => continue,
                ConstraintType::Le => TwoFloat::from(problem.b[i]) - ax_dd[i],
                ConstraintType::Ge => ax_dd[i] - TwoFloat::from(problem.b[i]),
            };
            let prod = (f64::from(slack_dd) * y[i]).abs();
            if prod > max_abs {
                max_abs = prod;
            }
        }
    }

    if !bound_duals.is_empty() {
        let mut idx = 0_usize;
        for (j, &(lb, _)) in problem.bounds.iter().enumerate() {
            if lb.is_finite() && idx < bound_duals.len() {
                let prod = (bound_duals[idx] * (x[j] - lb)).abs();
                if prod > max_abs {
                    max_abs = prod;
                }
                idx += 1;
            }
        }
        for (j, &(_, ub)) in problem.bounds.iter().enumerate() {
            if ub.is_finite() && idx < bound_duals.len() {
                let prod = (bound_duals[idx] * (ub - x[j])).abs();
                if prod > max_abs {
                    max_abs = prod;
                }
                idx += 1;
            }
        }
    }
    max_abs / scale
}

// ---------------------------------------------------------------------------
// 非公開関数
// ---------------------------------------------------------------------------

/// Ruiz スケーリング後の unscale 残差増幅率。
/// pfeas 増幅 = `1/min(e)`、dfeas 増幅 = `1/(c × min(d))` の最大値。
/// `f64::MIN_POSITIVE` は div0 防護 (Ruiz の `scale_floor_for_eps` で通常下限が保証される)。
pub(crate) fn compute_amplification(scaler: &RuizScaler) -> f64 {
    let e_min = if scaler.e.is_empty() {
        1.0
    } else {
        scaler.e.iter().cloned().fold(f64::INFINITY, f64::min).max(f64::MIN_POSITIVE)
    };
    let d_min = if scaler.d.is_empty() {
        1.0
    } else {
        scaler.d.iter().cloned().fold(f64::INFINITY, f64::min).max(f64::MIN_POSITIVE)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::ruiz::RuizScaler;

    /// compute_amplification は primal 側 (1/e_min) と dual 側 (1/(c*d_min)) の
    /// 大きい方を返す。dual 側を取りこぼすと QPILOTNO のような ill-conditioned 問題で
    /// IPM 完了後の元空間 dfeas を eps 以下に保証できなくなる。
    #[test]
    fn compute_amplification_includes_dual_side() {
        // primal amp = 1/e_min = 100、dual amp = 1/(c*d_min) = 10000 のケース。
        // 期待: max(100, 10000) = 10000。
        let mut scaler = RuizScaler::new(2, 2);
        scaler.e = vec![0.01, 1.0]; // e_min = 0.01 → primal amp = 100
        scaler.d = vec![0.001, 1.0]; // d_min = 0.001
        scaler.c = 0.1; // c * d_min = 1e-4 → dual amp = 1e4
        let amp = compute_amplification(&scaler);
        assert!((amp - 10000.0).abs() < 1.0,
            "dual amp 1/(c*d_min)=1e4 が支配するはず, got {:.3e}", amp);
    }

    /// primal 側が支配する場合の確認 (dual は十分小さい amp)。
    #[test]
    fn compute_amplification_primal_dominant() {
        let mut scaler = RuizScaler::new(2, 2);
        scaler.e = vec![1e-5, 1.0];  // primal amp = 1e5
        scaler.d = vec![0.5, 1.0];   // d_min = 0.5
        scaler.c = 1.0;              // c * d_min = 0.5 → dual amp = 2
        let amp = compute_amplification(&scaler);
        assert!((amp - 1e5).abs() < 10.0,
            "primal amp 1/e_min=1e5 が支配するはず, got {:.3e}", amp);
    }
}
