//! 内点法（IP-PMM: Interior Point Proximal Method of Multipliers）QPソルバー
//!
//! Mehrotra predictor-corrector + IP-PMM 正則化による QP 求解。
//!
//! # ファイル構成
//! - `mod.rs`:  公開 API・定数・Ruiz スケーリングラッパー・テスト
//! - `kkt.rs`:  KKT 行列構築・疎行列演算ヘルパー
//! - `step.rs`: メインループ（`solve_qp_ipm_inner`）・fraction-to-boundary・ユーティリティ
//! - `init.rs`: 初期点計算（Mehrotra heuristic）

pub(crate) mod init;
pub(crate) mod kkt;
pub(crate) mod step;
pub(crate) mod ippmm;

use crate::linalg::ruiz::RuizScaler;
use crate::linalg::timeout::TimeoutCtx;
use crate::options::SolverOptions;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use self::kkt::norm_inf;

// ---------------------------------------------------------------------------
// IPM 固定パラメータ
// ---------------------------------------------------------------------------

/// fraction-to-boundary τ
pub(crate) const TAU: f64 = 0.995;
/// IP-PMM 正則化最小値
#[allow(dead_code)]
pub(crate) const DELTA_MIN: f64 = 1e-8;
/// eps事前調整の下限（数値精度限界）
const EPS_FLOOR: f64 = 1e-12;
/// post-verification失敗時の再ソルブ上限回数（1回目=通常, 2〜N回目=10倍ずつ厳格化）
const POST_VERIFY_MAX_RESOLV: usize = 3;

// ---------------------------------------------------------------------------
// Gondzio multiple centrality correctors パラメータ
// ---------------------------------------------------------------------------

/// Gondzio: target step size factor β // PARAM: β=1.0でα=1.0(最大ステップ)を目指す | 要検証=β<1.0の効果
pub(crate) const BETA_GONDZIO: f64 = 1.0;
/// Gondzio: complementarity lower bound factor // PARAM: 根拠=Gondzio(1996) | 要検証=小規模問題
pub(crate) const GAMMA_L: f64 = 0.1;
/// Gondzio: complementarity upper bound factor // PARAM: 根拠=Gondzio(1996) | 要検証=小規模問題
pub(crate) const GAMMA_U: f64 = 10.0;
/// Gondzio: step size 改善の最小閾値 // PARAM: これ未満の改善は誤差程度 | 要検証=タイトな問題
pub(crate) const ALPHA_IMPROVE_THRESHOLD: f64 = 1e-3;

// ---------------------------------------------------------------------------
// 公開 API
// ---------------------------------------------------------------------------

/// IPM (Mehrotra predictor-corrector + IP-PMM) で QP を解く
///
/// Ruiz equilibration スケーリングを適用してから内部ソルバーを呼ぶ。
/// options.use_ruiz_scaling=false のときはスケーリングをスキップ。
pub fn solve_qp_ipm(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    if options.use_ruiz_scaling && problem.num_vars > 0 {
        let n = problem.num_vars;
        let m = problem.num_constraints;

        let lb: Vec<f64> = problem.bounds.iter().map(|&(l, _)| l).collect();
        let ub: Vec<f64> = problem.bounds.iter().map(|&(_, u)| u).collect();

        let mut scaler = RuizScaler::new(n, m);
        scaler.compute(&problem.q, &problem.a, &problem.c, &lb, &ub);

        let (q_s, a_s, c_s, b_s, bounds_s) =
            scaler.scale_problem(&problem.q, &problem.a, &problem.c, &problem.b, &problem.bounds);

        if let Ok(scaled_problem) = QpProblem::new(q_s, c_s, a_s, b_s, bounds_s) {
            let amplification = compute_amplification(&scaler);
            let mut last_result: Option<SolverResult> = None;
            // T9修正: POST_VERIFYループ前にdeadlineを1回確定し、ループ内では固定値を使う。
            // APIユーザーがtimeout_secs=Some(t), deadline=Noneで呼び出した場合、
            // ループごとにtimeout_secsから新しいdeadlineを計算すると最大3×timeout_secsの超過が起きる。
            // TimeoutCtx::from_optionsを使って既存パターンに統一（再発明を避ける）。
            let effective_deadline = TimeoutCtx::from_options(options).deadline;
            // post-verification再ソルブループ:
            // unscale後のeps検証が不合格（SuboptimalSolution）なら
            // scaled空間のepsを10倍ずつ厳格化して最大POST_VERIFY_MAX_RESOLV回まで再試行。
            // DTOC3/QPCBOEI2のようなRuiz scaling増幅問題に有効。
            for attempt in 0..POST_VERIFY_MAX_RESOLV {
                let tighten = 10f64.powi(attempt as i32); // 1.0, 10.0, 100.0
                let adjusted_eps =
                    (options.ipm_eps() / (amplification * tighten)).max(EPS_FLOOR);
                let mut adjusted_opts = options.clone();
                adjusted_opts.ipm.eps = adjusted_eps;
                adjusted_opts.deadline = effective_deadline;  // T9: 固定deadline（二重タイムアウト防止）
                adjusted_opts.timeout_secs = None;            // T9: 二重計算防止
                let scaled_result = step::solve_qp_ipm_inner(&scaled_problem, &adjusted_opts, Some(&scaler), Some(problem), options.ipm_eps());
                let result = unscale_ipm_result(scaled_result, &scaler, problem, options.ipm_eps());
                // 再ソルブ条件: 原空間で SuboptimalSolution かつ残り試行回数がある場合。
                // unscale_ipm_result が SuboptimalSolution/Optimalを原空間で判定するため
                // scaled側のステータスを別途確認する必要はない。
                if result.status == SolveStatus::SuboptimalSolution
                    && attempt + 1 < POST_VERIFY_MAX_RESOLV
                {
                    last_result = Some(result);
                    continue;
                }
                // Fix-D相当: MaxIterations → Timeout 変換
                if result.status == SolveStatus::MaxIterations {
                    return SolverResult { status: SolveStatus::Timeout, ..result };
                }
                // Timeout / Infeasible / Unbounded はそのまま返す
                if result.status == SolveStatus::Timeout
                    || matches!(result.status, SolveStatus::Infeasible | SolveStatus::Unbounded)
                {
                    return result;
                }
                // Fix-D相当: SuboptimalSolution → Timeout（Optimal|Timeout 2択を保証）
                let final_status = if result.status == SolveStatus::SuboptimalSolution {
                    SolveStatus::Timeout
                } else {
                    result.status
                };
                return SolverResult { status: final_status, ..result };
            }
            return last_result.expect("POST_VERIFY_MAX_RESOLV >= 1");
        }
        // QpProblem::new 失敗 → 非スケールにフォールバック
    }

    // BUG-A1修正: 非RuizパスのFix-D漏れ（SuboptimalSolution→Timeout変換）
    let raw = post_verify_solution(step::solve_qp_ipm_inner(problem, options, None, None, options.ipm_eps()), problem, options.ipm_eps());
    // MaxIterations: Simplex（solve_as_lp）はMaxIterationsを返す場合がある。
    // IPMパスからは返らないが、as_lpパス（SimplexSolver）を経由した場合に対応するため意図的に残存。
    if raw.status == SolveStatus::MaxIterations || raw.status == SolveStatus::SuboptimalSolution {
        SolverResult { status: SolveStatus::Timeout, ..raw }
    } else {
        raw
    }
}

/// IP-PMM（Interior Point-Proximal Method of Multipliers）で QP を解く
///
/// 完全独立実装（step.rs / kkt.rs 不使用）。Ruiz スケーリングラッパー付き。
pub(crate) fn solve_qp_ippmm(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    if options.use_ruiz_scaling && problem.num_vars > 0 {
        let n = problem.num_vars;
        let m = problem.num_constraints;

        let lb: Vec<f64> = problem.bounds.iter().map(|&(l, _)| l).collect();
        let ub: Vec<f64> = problem.bounds.iter().map(|&(_, u)| u).collect();

        let mut scaler = RuizScaler::new(n, m);
        scaler.compute(&problem.q, &problem.a, &problem.c, &lb, &ub);

        let (q_s, a_s, c_s, b_s, bounds_s) =
            scaler.scale_problem(&problem.q, &problem.a, &problem.c, &problem.b, &problem.bounds);

        if let Ok(scaled_problem) = QpProblem::new(q_s, c_s, a_s, b_s, bounds_s) {
            let amplification = compute_amplification(&scaler);
            let mut last_result: Option<SolverResult> = None;
            // T9修正: POST_VERIFYループ前にdeadlineを1回確定し、ループ内では固定値を使う。
            let effective_deadline = TimeoutCtx::from_options(options).deadline;
            for attempt in 0..POST_VERIFY_MAX_RESOLV {
                let tighten = 10f64.powi(attempt as i32);
                let adjusted_eps =
                    (options.ipm_eps() / (amplification * tighten)).max(EPS_FLOOR);
                let mut adjusted_opts = options.clone();
                adjusted_opts.ipm.eps = adjusted_eps;
                adjusted_opts.deadline = effective_deadline;  // T9: 固定deadline（二重タイムアウト防止）
                adjusted_opts.timeout_secs = None;            // T9: 二重計算防止
                let scaled_result = ippmm::solve_ippmm_inner(&scaled_problem, &adjusted_opts, Some(&scaler), Some(problem), options.ipm_eps());
                let result = unscale_ipm_result(scaled_result, &scaler, problem, options.ipm_eps());
                if result.status == SolveStatus::SuboptimalSolution
                    && attempt + 1 < POST_VERIFY_MAX_RESOLV
                {
                    last_result = Some(result);
                    continue;
                }
                if matches!(result.status, SolveStatus::Timeout | SolveStatus::MaxIterations) {
                    return result;
                }
                let final_status = if result.status == SolveStatus::SuboptimalSolution {
                    SolveStatus::Timeout
                } else {
                    result.status
                };
                return SolverResult { status: final_status, ..result };
            }
            return last_result.expect("POST_VERIFY_MAX_RESOLV >= 1");
        }
    }

    let raw = post_verify_solution(
        ippmm::solve_ippmm_inner(problem, options, None, None, options.ipm_eps()),
        problem,
        options.ipm_eps(),
    );
    if raw.status == SolveStatus::SuboptimalSolution {
        SolverResult { status: SolveStatus::Timeout, ..raw }
    } else {
        raw
    }
}

/// SuboptimalSolution（ソルバー内部判定）を原問題空間で再検証し、
/// pfeas・bfeas・dfeas が eps 基準を満たすなら Optimal に昇格する。
///
/// Ruiz scaling なしのフォールバックパスで使用。
/// Ruiz ありパスは unscale_ipm_result の SuboptimalSolution ブランチが担当。
fn post_verify_solution(result: SolverResult, problem: &QpProblem, eps: f64) -> SolverResult {
    if result.status != SolveStatus::SuboptimalSolution || result.solution.is_empty() {
        return result;
    }
    let x = &result.solution;
    let y = &result.dual_solution;
    let bound_duals = &result.bound_duals;
    let d_min = 1.0_f64; // スケーリングなし: d_min=1
    let norm_c_s = norm_inf(&problem.c).max(1.0);
    let dfeas_threshold = 10.0 * eps * (1.0 + norm_c_s) / d_min;
    let status = if problem.num_constraints > 0 {
        match problem.a.mat_vec_mul(x) {
            Ok(ax) => {
                let pfeas: f64 = ax
                    .iter()
                    .zip(problem.b.iter())
                    .map(|(&ax_i, &b_i)| (ax_i - b_i).max(0.0))
                    .fold(0.0_f64, f64::max);
                let norm_b = norm_inf(&problem.b).max(1.0);
                if pfeas < eps * (1.0 + norm_b) {
                    let bfeas_status = check_bfeas_status(x, &problem.bounds, eps);
                    if bfeas_status == SolveStatus::Optimal {
                        check_dfeas_status(problem, x, y, bound_duals, dfeas_threshold)
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
        let bfeas_status = check_bfeas_status(x, &problem.bounds, eps);
        if bfeas_status == SolveStatus::Optimal {
            check_dfeas_status(problem, x, y, bound_duals, dfeas_threshold)
        } else {
            bfeas_status
        }
    };
    SolverResult { status, ..result }
}

/// Ruizスケーリングによる残差増幅率を計算する。
///
/// pfeas増幅: 1/e_min、dfeas増幅: 1/(c * d_min) の最大を返す。
/// IPM内部のepsをtighterに設定するために使用する。
fn compute_amplification(scaler: &RuizScaler) -> f64 {
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

/// lb <= x <= ub の違反量を検証し、超過していれば SuboptimalSolution に降格する
///
/// 閾値: eps（絶対値基準）。qps_benchmarkの検証基準と統一。
/// lb/ub が ±∞ の成分はスキップする。
fn check_bfeas_status(x: &[f64], bounds: &[(f64, f64)], eps: f64) -> SolveStatus {
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
        // bfeas違反: 境界制約を満たさない偽Optimal
        SolveStatus::SuboptimalSolution
    }
}

/// QPの双対実現可能性 (dfeas) を検証し、超過していれば SuboptimalSolution に降格する
///
/// dfeas = ||Q*x + A^T*y - y_lb + y_ub + c||_inf
/// 無制約QP（A=0, bounds=all infinite）では A^T*y = 0, bound_contrib = 0 として計算する。
///
/// # 引数
/// - `bound_duals`: アンスケール済み境界双対変数。lb有限変数の下界dual（昇順）、次にub有限変数の上界dual（昇順）の順
/// - `threshold`: 呼び出し元で計算した許容閾値。Ruizスケーリングの増幅係数を考慮した値を渡すこと。
fn check_dfeas_status(problem: &QpProblem, x: &[f64], y: &[f64], bound_duals: &[f64], threshold: f64) -> SolveStatus {
    let n = x.len();
    // Q*x
    let qx = match problem.q.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return SolveStatus::Optimal, // 計算失敗時はstatusを保持（安全側）
    };
    // A^T*y（無制約QPではa.nrows==0なのでzeroベクトル）
    let aty: Vec<f64> = if problem.a.nrows > 0 && !y.is_empty() {
        match problem.a.transpose().mat_vec_mul(y) {
            Ok(v) => v,
            Err(_) => return SolveStatus::Optimal, // 計算失敗時はstatusを保持（安全側）
        }
    } else {
        vec![0.0; n]
    };
    // bound_contrib[j] = -y_lb[j] (lb有限) + y_ub[j] (ub有限)
    // KKT: Q*x + c + A^T*y - y_lb + y_ub = 0
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
        // dfeas違反: 双対実現可能性を満たさない偽Optimal
        SolveStatus::SuboptimalSolution
    }
}

/// スケール済み IPM 結果を元のスケールに逆変換する
///
/// Optimal ステータスの場合、元空間で pfeas・bfeas・dfeas を再計算し、
/// それぞれの許容誤差を超えていれば SuboptimalSolution に降格する（偽Optimal防止）。
fn unscale_ipm_result(
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
            // post-unscaling検証: 元空間で primal feasibility (pfeas) と
            // bounds feasibility (bfeas) を確認。
            // scaled空間でeps以下でも、unscale後に残差が増幅される問題（偽Optimal）を検出する
            // dfeas閾値: Ruizスケーリングの増幅係数を考慮して計算する。
            // dfeas_orig ≤ eps * (1+norm_c_s) / (scaler.c * d_min) の理論上限に
            // 安全係数10を掛けて浮動小数点誤差とIPM停止タイミングのずれを吸収する。
            // norm_c_s = scaled空間でのcノルム = scaler.c * max_j |scaler.d[j] * c[j]|
            let d_min = if scaler.d.is_empty() {
                1.0
            } else {
                scaler.d.iter().cloned().fold(f64::INFINITY, f64::min).max(1e-12)
            };
            let norm_c_s = scaler.d.iter().enumerate()
                .map(|(j, &dj)| (scaler.c * dj * problem.c[j]).abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let dfeas_threshold = 10.0 * eps * (1.0 + norm_c_s) / (scaler.c * d_min);
            let (status, orig_residuals) = if problem.num_constraints > 0 {
                match problem.a.mat_vec_mul(&x) {
                    Ok(ax) => {
                        let pfeas: f64 = ax
                            .iter()
                            .zip(problem.b.iter())
                            .map(|(&ax_i, &b_i)| (ax_i - b_i).max(0.0))
                            .fold(0.0_f64, f64::max);
                        let norm_b = norm_inf(&problem.b).max(1.0);
                        // 元空間pfeasでfinal_residualsを更新（dfeas/gapはscaled値を流用）
                        let orig_resid = result.final_residuals.map(|(_, d, g)| (pfeas, d, g));
                        let status = if pfeas < eps * (1.0 + norm_b) {
                            // pfeas OK: bfeas → dfeas の順で検証
                            let bfeas_status = check_bfeas_status(&x, &problem.bounds, eps);
                            if bfeas_status == SolveStatus::Optimal {
                                check_dfeas_status(problem, &x, &y, &bound_duals, dfeas_threshold)
                            } else {
                                bfeas_status
                            }
                        } else {
                            // 偽Optimal検出: scaled空間での収束判定を元空間で再検証した結果、不合格
                            SolveStatus::SuboptimalSolution
                        };
                        (status, orig_resid)
                    }
                    Err(_) => (SolveStatus::Optimal, result.final_residuals), // mat_vec_mul失敗時はstatusを保持（安全側）
                }
            } else {
                // 制約なし問題: pfeas検証不要だがbfeas → dfeas は検証
                let bfeas_status = check_bfeas_status(&x, &problem.bounds, eps);
                let status = if bfeas_status == SolveStatus::Optimal {
                    check_dfeas_status(problem, &x, &y, &bound_duals, dfeas_threshold)
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
            // scaled空間でSuboptimalSolutionだった場合も unscale して原空間で再検証する。
            // 原空間の pfeas・bfeas が eps 基準を満たせば Optimal に昇格する（DTOC3対策）。
            let (x, y) = scaler.unscale_solution(&result.solution, &result.dual_solution);
            let bound_duals = scaler.unscale_bound_duals(&result.bound_duals, &problem.bounds);
            let obj_orig = result.objective / scaler.c;
            let d_min = if scaler.d.is_empty() {
                1.0
            } else {
                scaler.d.iter().cloned().fold(f64::INFINITY, f64::min).max(1e-12)
            };
            let norm_c_s = scaler.d.iter().enumerate()
                .map(|(j, &dj)| (scaler.c * dj * problem.c[j]).abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let dfeas_threshold = 10.0 * eps * (1.0 + norm_c_s) / (scaler.c * d_min);
            let status = if problem.num_constraints > 0 {
                match problem.a.mat_vec_mul(&x) {
                    Ok(ax) => {
                        let pfeas: f64 = ax
                            .iter()
                            .zip(problem.b.iter())
                            .map(|(&ax_i, &b_i)| (ax_i - b_i).max(0.0))
                            .fold(0.0_f64, f64::max);
                        let norm_b = norm_inf(&problem.b).max(1.0);
                        if pfeas < eps * (1.0 + norm_b) {
                            let bfeas_status = check_bfeas_status(&x, &problem.bounds, eps);
                            if bfeas_status == SolveStatus::Optimal {
                                check_dfeas_status(problem, &x, &y, &bound_duals, dfeas_threshold)
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
                    check_dfeas_status(problem, &x, &y, &bound_duals, dfeas_threshold)
                } else {
                    bfeas_status
                }
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

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SolverOptions;
    use crate::sparse::CscMatrix;

    const EPS: f64 = 1e-5;

    fn close(a: f64, b: f64, name: &str) {
        assert!(
            (a - b).abs() < EPS,
            "{}: expected {:.8}, got {:.8} (diff={:.2e})",
            name,
            b,
            a,
            (a - b).abs()
        );
    }

    fn default_opts() -> SolverOptions {
        SolverOptions::default()
    }

    /// IPM-T1: 2変数基本 QP
    /// min x^2 + y^2  (Q=2I, c=0)  s.t. x + y >= 1
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_ipm_basic_2d() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // x + y >= 1 → -(x+y) <= -1
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp_ipm(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T1: status");
        close(result.solution[0], 0.5, "IPM-T1: x[0]");
        close(result.solution[1], 0.5, "IPM-T1: x[1]");
        close(result.objective, 0.5, "IPM-T1: objective");
    }

    /// IPM-T2: 制約なし QP (解析解)
    /// min (x-3)^2 + (y-4)^2 = 1/2*2*(x^2+y^2) - 6x - 8y + const
    /// Q=2I, c=[-6,-8], 制約なし
    /// 期待: x*=3, y*=4, obj=-25
    #[test]
    fn test_ipm_unconstrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-6.0, -8.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp_ipm(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T2: status");
        close(result.solution[0], 3.0, "IPM-T2: x[0]");
        close(result.solution[1], 4.0, "IPM-T2: x[1]");
        close(result.objective, -25.0, "IPM-T2: objective");
    }

    /// IPM-T3: 等式制約付き QP
    /// min x^2 + y^2  s.t. x + y = 1
    /// 等式を 2 不等式で表現: x+y<=1, -(x+y)<=-1
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_ipm_equality_constraint() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, -1.0, -1.0],
            2,
            2,
        )
        .unwrap();
        let b = vec![1.0, -1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp_ipm(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T3: status");
        close(result.solution[0], 0.5, "IPM-T3: x[0]");
        close(result.solution[1], 0.5, "IPM-T3: x[1]");
        close(result.objective, 0.5, "IPM-T3: objective");
    }

    /// IPM-T4: Box 制約付き QP
    /// min (x-2)^2 + (y-2)^2  s.t. 0 <= x <= 1, 0 <= y <= 1
    /// Q=2I, c=[-4,-4], bounds=[0,1]^2
    /// 期待: x*=y*=1, obj=-6
    #[test]
    fn test_ipm_box_constrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-4.0, -4.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp_ipm(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T4: status");
        close(result.solution[0], 1.0, "IPM-T4: x[0]");
        close(result.solution[1], 1.0, "IPM-T4: x[1]");
        close(result.objective, -6.0, "IPM-T4: objective");
    }

    /// IPM-T5: ポートフォリオ最適化（3変数等式+非負制約）
    /// min 1/2 w^T Σ w  s.t. sum(w)=1, w >= 0
    /// Σ = diag(2,2,2), 対称解: w* = [1/3, 1/3, 1/3], obj = 1/3
    #[test]
    fn test_ipm_portfolio() {
        let q = CscMatrix::from_triplets(
            &[0, 1, 2],
            &[0, 1, 2],
            &[2.0, 2.0, 2.0],
            3,
            3,
        )
        .unwrap();
        let c = vec![0.0, 0.0, 0.0];
        // 等式 sum=1 (2不等式) + 非負制約 w>=0 (3不等式)
        let a = CscMatrix::from_triplets(
            &[0, 0, 0, 1, 1, 1, 2, 3, 4],
            &[0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            5,
            3,
        )
        .unwrap();
        let b = vec![1.0, -1.0, 0.0, 0.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp_ipm(&problem, &default_opts());
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T5: status");
        close(result.solution[0], 1.0 / 3.0, "IPM-T5: w[0]");
        close(result.solution[1], 1.0 / 3.0, "IPM-T5: w[1]");
        close(result.solution[2], 1.0 / 3.0, "IPM-T5: w[2]");
        close(result.objective, 1.0 / 3.0, "IPM-T5: objective");
    }

    /// IPM-T6: タイムアウト動作確認（極小 timeout で Timeout が返ること）
    #[test]
    fn test_ipm_timeout() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let mut opts = SolverOptions { timeout_secs: Some(0.0001), ..Default::default() };
        opts.use_ruiz_scaling = false;
        let result = solve_qp_ipm(&problem, &opts);
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "IPM-T6: expected Timeout or Optimal, got {:?}",
            result.status
        );
    }


    /// IPM-T8: post-unscaling 偽Optimal検出テスト
    ///
    /// scaled空間での収束を満たしているが元空間で許容誤差を超過する SolverResult を
    /// unscale_ipm_result に渡し、Optimal が返らないことを確認する。
    ///
    /// 設定:
    ///   問題: min x^2  s.t. x <= 0.5  (n=1, m=1, A=[[1.0]], b=[0.5])
    ///   RuizScaler: d=[2.0], e=[1.0], c=1.0
    ///   mock解: x_scaled=[1.0]  → x_orig = d[0]*x_s[0] = 2.0
    ///   pfeas = A*x_orig - b = 2.0 - 0.5 = 1.5  >> eps*(1+0.5) ≈ 1.5e-6
    #[test]
    fn test_ipm_post_unscaling_false_optimal_detection() {
        // 問題構築: min x^2  s.t. x <= 0.5
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c_vec = vec![0.0];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let b = vec![0.5];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new(q, c_vec, a, b, bounds).unwrap();

        // RuizScaler を手動設定: d=[2.0] により x_orig = 2.0*x_scaled
        let mut scaler = RuizScaler::new(1, 1);
        scaler.d = vec![2.0];
        scaler.e = vec![1.0];
        scaler.c = 1.0;

        // scaled解 x_scaled=[1.0] → x_orig=2.0  は  x<=0.5  を大幅違反
        let mock_result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0],      // scaled x_s
            dual_solution: vec![0.0], // scaled y_s
            objective: 1.0,
            ..SolverResult::default()
        };

        let eps = 1e-6_f64;
        let result = unscale_ipm_result(mock_result, &scaler, &problem, eps);

        // pfeas = 2.0 - 0.5 = 1.5 >> eps*(1+0.5) → 偽Optimal → SuboptimalSolution に降格
        assert_ne!(
            result.status,
            SolveStatus::Optimal,
            "IPM-T8: 偽Optimal（元空間pfeas超過）でOptimalを返してはならない。got {:?}",
            result.status
        );
        assert_eq!(
            result.status,
            SolveStatus::SuboptimalSolution,
            "IPM-T8: 降格先はSuboptimalSolutionであること"
        );
        // unscale後の解が正しいことも確認
        close(result.solution[0], 2.0, "IPM-T8: x[0] unscaled");
    }

    /// IPM-T9: post-unscaling 正常Optimal確認テスト
    ///
    /// 元空間でも許容誤差内に収まる場合、Optimal がそのまま維持されることを確認。
    ///
    /// 設定:
    ///   問題: min x^2  s.t. x <= 1.0  (n=1, m=1, A=[[1.0]], b=[1.0])
    ///   RuizScaler: d=[1.0], e=[1.0], c=1.0 (恒等変換)
    ///   mock解: x_scaled=[0.0]  → x_orig=0.0  pfeas=max(0, 0-1)=0 < eps
    #[test]
    fn test_ipm_post_unscaling_genuine_optimal_preserved() {
        // 問題構築: min x^2  s.t. x <= 1.0
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c_vec = vec![0.0];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new(q, c_vec, a, b, bounds).unwrap();

        // 恒等スケーラー
        let scaler = RuizScaler::new(1, 1);

        // x_orig=0.0 は x<=1.0 を満たす (pfeas=0)
        let mock_result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![0.0],
            objective: 0.0,
            ..SolverResult::default()
        };

        let eps = 1e-6_f64;
        let result = unscale_ipm_result(mock_result, &scaler, &problem, eps);

        // pfeas = 0 < eps*(1+1) → 真のOptimal → そのままOptimalを返すべき
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "IPM-T9: 真のOptimal（元空間pfeas許容内）でOptimalが維持されること"
        );
    }

    /// IPM-T10: bfeas違反の検出テスト
    ///
    /// lb/ubを持つ問題に対し、境界を大幅に超過するmock解を渡した場合に
    /// unscale_ipm_result が Optimal を返さないことを確認する。
    ///
    /// 設定:
    ///   問題: min x^2  bounds: 0.0 <= x <= 0.5
    ///   RuizScaler: 恒等変換 (d=[1.0], e=[], c=1.0)
    ///   mock解: x=[1.0]  → ub違反: 1.0 - 0.5 = 0.5 >> eps
    #[test]
    fn test_ipm_bfeas_violation_detected() {
        // bounds: 0.0 <= x <= 0.5 (constraint as bounds)
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c_vec = vec![0.0];
        let a = CscMatrix::new(0, 1); // 制約なし（boundsのみ）
        let b = vec![];
        let bounds = vec![(0.0_f64, 0.5_f64)];
        let problem = QpProblem::new(q, c_vec, a, b, bounds).unwrap();

        // 恒等スケーラー（Ruiz変換なし相当）
        let scaler = RuizScaler::new(1, 0);

        // mock解: x=1.0 は ub=0.5 を大幅違反
        let mock_result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0],
            dual_solution: vec![],
            objective: 1.0,
            ..SolverResult::default()
        };

        let eps = 1e-6_f64;
        let result = unscale_ipm_result(mock_result, &scaler, &problem, eps);

        // bfeas = 0.5 >> eps*(1+0.5) ≈ 1.5e-6 → 偽Optimal → SuboptimalSolution に降格
        assert_ne!(
            result.status,
            SolveStatus::Optimal,
            "IPM-T10: bfeas違反でOptimalを返してはならない。got {:?}",
            result.status
        );
        assert_eq!(
            result.status,
            SolveStatus::SuboptimalSolution,
            "IPM-T10: 降格先はSuboptimalSolutionであること"
        );
    }

    /// IPM-T11: bfeas OK かつ dfeas OK → Optimal 維持テスト
    ///
    /// lb/ubを持つ問題に対し、KKT最適解のmock解を渡した場合に
    /// unscale_ipm_result が Optimal を維持することを確認する。
    ///
    /// 設定:
    ///   問題: min x^2  bounds: -1.0 <= x <= 1.0
    ///   mock解: x=[0.0] (KKT最適解: ∇f(0)=0, bounds非活性 → bound_duals=0)
    ///   dfeas = |Q*x + c + bound_contrib| = |2*0 + 0 + 0| = 0 < threshold
    ///
    /// NOTE: このテストは bound_duals修正(cmd_503)後にdfeasチェックが有効になった。
    /// KKT最適解x=0を使用（旧x=0.5はdfeas=1.0でSUBに降格するため修正）。
    #[test]
    fn test_ipm_bfeas_within_bounds_preserved() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c_vec = vec![0.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(-1.0_f64, 1.0_f64)];
        let problem = QpProblem::new(q, c_vec, a, b, bounds).unwrap();

        let scaler = RuizScaler::new(1, 0);

        // mock解: x=0.0 (KKT最適解。dfeas = |2*0 + 0| = 0 < threshold)
        let mock_result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![],
            objective: 0.0,
            ..SolverResult::default()
        };

        let eps = 1e-6_f64;
        let result = unscale_ipm_result(mock_result, &scaler, &problem, eps);

        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "IPM-T11: KKT最適解はOptimalを維持すること"
        );
    }

    /// IPM-T12: post-unscaling dfeas違反の検出テスト
    ///
    /// dfeas = ||Q*x + A^T*y + c||_inf が閾値を大幅超過する mock 解を渡したとき
    /// unscale_ipm_result が Optimal を返さないことを確認する。
    ///
    /// 設定:
    ///   問題: min x^2 + x  (Q=[[2.0]], c=[1.0], 制約なし)
    ///   真の最適解: x* = -0.5 (dfeas=0)
    ///   mock解: x=[10.0], y=[]
    ///   dfeas = |2*10 + 1| = 21 >> eps*(1+1) ≈ 2e-6
    #[test]
    fn test_ipm_dfeas_violation_detected() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c_vec = vec![1.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new(q, c_vec, a, b, bounds).unwrap();

        // 恒等スケーラー
        let scaler = RuizScaler::new(1, 0);

        // mock解: x=10.0 → dfeas = |2*10 + 1| = 21 >> eps*(1+1)
        let mock_result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![10.0],
            dual_solution: vec![],
            objective: 110.0,
            ..SolverResult::default()
        };

        let eps = 1e-6_f64;
        let result = unscale_ipm_result(mock_result, &scaler, &problem, eps);

        // dfeas 超過 → 偽Optimal → SuboptimalSolution に降格
        assert_ne!(
            result.status,
            SolveStatus::Optimal,
            "IPM-T12: dfeas違反でOptimalを返してはならない。got {:?}",
            result.status
        );
        assert_eq!(
            result.status,
            SolveStatus::SuboptimalSolution,
            "IPM-T12: 降格先はSuboptimalSolutionであること"
        );
    }

    /// IPM-T13: post-unscaling dfeas正常（Optimal維持）テスト
    ///
    /// dfeas が閾値内に収まる真の最適解を mock で渡したとき
    /// unscale_ipm_result が Optimal を維持することを確認する。
    ///
    /// 設定:
    ///   問題: min x^2 + x  (Q=[[2.0]], c=[1.0], 制約なし)
    ///   mock解: x=-0.5 (真の最適解)
    ///   dfeas = |2*(-0.5) + 1| = 0 < eps*(1+1)
    #[test]
    fn test_ipm_dfeas_within_tolerance_preserved() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c_vec = vec![1.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new(q, c_vec, a, b, bounds).unwrap();

        // 恒等スケーラー
        let scaler = RuizScaler::new(1, 0);

        // mock解: x=-0.5 → dfeas = |2*(-0.5) + 1| = 0.0 < eps*(1+1)
        let mock_result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![-0.5],
            dual_solution: vec![],
            objective: -0.25,
            ..SolverResult::default()
        };

        let eps = 1e-6_f64;
        let result = unscale_ipm_result(mock_result, &scaler, &problem, eps);

        // dfeas = 0 < eps*(1+1) → 真のOptimal → Optimalを維持すること
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "IPM-T13: dfeas許容内の解はOptimalを維持すること"
        );
    }

    /// IPM-T14: compute_amplification 計算検証
    ///
    /// e_min=0.1, d_min=0.01, c=1.0 → amplification=max(1/0.1, 1/(1.0*0.01))=max(10.0, 100.0)=100.0
    #[test]
    fn test_compute_amplification_calculation() {
        let mut scaler = RuizScaler::new(2, 2);
        scaler.d = vec![0.01, 1.0]; // d_min=0.01
        scaler.e = vec![0.1, 1.0]; // e_min=0.1
        scaler.c = 1.0;
        let amp = compute_amplification(&scaler);
        assert!(
            (amp - 100.0).abs() < 1e-10,
            "IPM-T14: expected amplification=100.0, got {:.6}",
            amp
        );
    }

    /// IPM-T15: 強スケーリング環境での adjusted_eps 確認
    ///
    /// amplification > 1.0 → adjusted_eps = user_eps / amplification < user_eps であること
    #[test]
    fn test_adjusted_eps_less_than_user_eps() {
        let mut scaler = RuizScaler::new(2, 1);
        scaler.d = vec![0.1, 0.5]; // d_min=0.1
        scaler.e = vec![0.5];      // e_min=0.5
        scaler.c = 1.0;
        let amp = compute_amplification(&scaler);
        assert!(amp > 1.0, "IPM-T15: amplification > 1.0 を期待, got {:.6}", amp);
        let user_eps = 1e-6_f64;
        let adjusted_eps = (user_eps / amp).max(EPS_FLOOR);
        assert!(
            adjusted_eps < user_eps,
            "IPM-T15: adjusted_eps ({:.2e}) < user_eps ({:.2e}) であること",
            adjusted_eps,
            user_eps
        );
    }

    /// IPM-T16: lb-only 活性ケースの bound_duals 値検証（TC-01, TC-02対応）
    ///
    /// min x s.t. x >= 1.0  (lb活性: x*=1, y_lb=1.0 > 0)
    /// Q=[[1]], c=[0] → 実際には Q=[[2]]（「1/2あり」規約: min 1/2*2*x^2 = min x^2, min x には c=[-1] 相当）
    ///
    /// 問題: min x s.t. x >= 1.0
    /// Q=[[0]] (zero, 線形問題), c=[1.0]（min x）, bounds=[(1.0, +∞)]
    /// ただし Q=0 はIPMに不安定。代わりに:
    /// min 1/2*ε*x^2 + x s.t. x >= 1 (ε→0+, lb活性)
    /// KKT: ε*x + 1 - y_lb = 0, x=1 → y_lb = ε + 1 ≈ 1.0
    /// → bound_duals[0] ≈ 1.0 (lb dual)
    ///
    /// NOTE: bound_duals修正(cmd_503)後。y[m_orig..m_ext]が正しく計算される。
    #[test]
    fn test_ipm_bound_duals_lb_only_active() {
        // min 1/2*0.001*x^2 + x s.t. x >= 1.0
        // Q = [[0.001]] (ε = 0.001), c = [1.0], bounds = [(1.0, +∞)]
        // KKT at x=1: 0.001*1 + 1 - y_lb = 0 → y_lb = 1.001
        let q = CscMatrix::from_triplets(&[0], &[0], &[0.001], 1, 1).unwrap();
        let c_vec = vec![1.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(1.0_f64, f64::INFINITY)];
        let problem = QpProblem::new(q, c_vec, a, b, bounds).unwrap();

        let opts = SolverOptions { use_ruiz_scaling: false, ..Default::default() };
        let result = solve_qp_ipm(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T16: status should be Optimal");
        assert!(
            (result.solution[0] - 1.0).abs() < 1e-4,
            "IPM-T16: x*≈1.0, got {:.6}", result.solution[0]
        );
        // bound_duals[0] = y_lb ≈ 1.001 (lb dual at x=1)
        assert!(!result.bound_duals.is_empty(), "IPM-T16: bound_duals should be non-empty");
        assert!(
            (result.bound_duals[0] - 1.001).abs() < 0.1,
            "IPM-T16: bound_duals[0] (lb dual) ≈ 1.001, got {:.6}", result.bound_duals[0]
        );
    }

    /// IPM-T17: ub-only 活性ケースの bound_duals 値検証（TC-01, TC-02対応）
    ///
    /// min -x s.t. x <= 0.5  (ub活性: x*=0.5, y_ub=1.0 > 0)
    /// Q=[[ε]] (小さい正則化), c=[-1.0], bounds=[(-∞, 0.5)]
    /// KKT: ε*x - 1 + y_ub = 0, x=0.5 → y_ub = 1 - ε*0.5 ≈ 1.0
    ///
    /// NOTE: bound_duals修正(cmd_503)後。lb有限変数なし→ub分のみ格納。
    #[test]
    fn test_ipm_bound_duals_ub_only_active() {
        // min 1/2*0.001*x^2 - x s.t. x <= 0.5
        // KKT at x=0.5: 0.001*0.5 - 1 + y_ub = 0 → y_ub = 1 - 0.0005 = 0.9995
        let q = CscMatrix::from_triplets(&[0], &[0], &[0.001], 1, 1).unwrap();
        let c_vec = vec![-1.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, 0.5_f64)];
        let problem = QpProblem::new(q, c_vec, a, b, bounds).unwrap();

        let opts = SolverOptions { use_ruiz_scaling: false, ..Default::default() };
        let result = solve_qp_ipm(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T17: status should be Optimal");
        assert!(
            (result.solution[0] - 0.5).abs() < 1e-4,
            "IPM-T17: x*≈0.5, got {:.6}", result.solution[0]
        );
        // bound_duals[0] = y_ub ≈ 0.9995 (ub dual. lb-infiniteのためindex=0がub dual)
        assert!(!result.bound_duals.is_empty(), "IPM-T17: bound_duals should be non-empty");
        assert!(
            (result.bound_duals[0] - 1.0).abs() < 0.1,
            "IPM-T17: bound_duals[0] (ub dual) ≈ 1.0, got {:.6}", result.bound_duals[0]
        );
    }

    /// IPM-T18: 両端有限 lb活性ケース — T12のbound_dualsアサート追加（TC-01対応）
    ///
    /// T4と同じ問題: min (x-2)^2 + (y-2)^2 s.t. 0<=x<=1, 0<=y<=1
    /// x*=y*=1 (ub活性), ub duals > 0, lb duals = 0 (lb非活性)
    ///
    /// NOTE: bound_duals修正(cmd_503)後。
    /// bound_duals格納順: lb_x, lb_y (index 0,1), ub_x, ub_y (index 2,3)
    /// x=y=1 (ub活性) → ub_duals > 0, lb_duals ≈ 0
    #[test]
    fn test_ipm_bound_duals_box_constrained() {
        // min (x-2)^2+(y-2)^2 = 1/2*4*(x^2+y^2) - 4x - 4y + const
        // Q=[[2,0],[0,2]] (「1/2あり」: 1/2*2*x^2=x^2, grad=2x), c=[-4,-4]
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c_vec = vec![-4.0, -4.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64), (0.0_f64, 1.0_f64)];
        let problem = QpProblem::new(q, c_vec, a, b, bounds).unwrap();

        let opts = SolverOptions { use_ruiz_scaling: false, ..Default::default() };
        let result = solve_qp_ipm(&problem, &opts);

        // NOTE: bound_duals修正(cmd_503)後もstatus=Optimalを維持することを確認。
        // もしSUBに変化する場合はKKT解析で原因を確認すること。
        assert_eq!(result.status, SolveStatus::Optimal, "IPM-T18: status");
        assert!((result.solution[0] - 1.0).abs() < 0.01, "IPM-T18: x*≈1.0");
        assert!((result.solution[1] - 1.0).abs() < 0.01, "IPM-T18: y*≈1.0");

        // bound_duals格納順: [lb_x, lb_y, ub_x, ub_y] (全4変数×両端有限)
        // x=y=1はub活性 → ub_duals(index 2,3) > 0
        // x=y=1はlb非活性 → lb_duals(index 0,1) ≈ 0 (interior point法では正の値になるが小さい)
        assert_eq!(result.bound_duals.len(), 4, "IPM-T18: 4 bound duals expected");
        // KKT: 2*x - 4 - y_lb_x + y_ub_x = 0 → y_ub_x = 4 - 2*1 + y_lb_x ≈ 2
        assert!(
            result.bound_duals[2] > 0.5,
            "IPM-T18: ub_dual_x should be positive (active), got {:.6}", result.bound_duals[2]
        );
        assert!(
            result.bound_duals[3] > 0.5,
            "IPM-T18: ub_dual_y should be positive (active), got {:.6}", result.bound_duals[3]
        );
    }

    /// T9 sanityチェック: Ruiz有効時のPOST_VERIFYループがtimeoutを大幅超過しないこと。
    /// 注意: このテストはT9修正のsanityチェック。
    /// 実問題での効果確認はベンチ（Step5 Maros/QPLIB）で行う。
    #[test]
    fn test_ipm_post_verify_timeout_stays_within_budget() {
        // 適度なサイズの問題でRuiz scaling有効 + 短めのtimeout
        // use_ruiz_scaling=true → POST_VERIFYループを通るパス
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[2.0, 2.0, 2.0], 3, 3).unwrap();
        let c = vec![0.0; 3];
        let a = CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[-1.0, -1.0, -1.0], 1, 3).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let timeout_secs = 0.01;
        let mut opts = SolverOptions { timeout_secs: Some(timeout_secs), ..Default::default() };
        opts.use_ruiz_scaling = true;  // POST_VERIFYループを通るパス（T9修正対象）

        let start = std::time::Instant::now();
        let result = solve_qp_ipm(&problem, &opts);
        let elapsed = start.elapsed().as_secs_f64();

        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "T9 sanity: expected Timeout or Optimal, got {:?}", result.status
        );
        // T9修正後: deadline固定によりPOST_VERIFYループの超過は最大1イテレーション分のみ
        // 2.0×で十分厳格（バグ残存なら3×超過し検出できる）
        assert!(
            elapsed < timeout_secs * 2.0,
            "T9 sanity: elapsed({:.3}s) > timeout×2({:.3}s). T9バグが残存している可能性",
            elapsed, timeout_secs * 2.0
        );
    }

    /// IPM-T7: Ruiz スケーリング有無で同一解が得られることを確認
    /// T1 と同じ問題 (min x^2+y^2, s.t. x+y>=1) で比較
    #[test]
    fn test_ipm_ruiz_scaling_consistency() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result_ruiz = solve_qp_ipm(&problem, &SolverOptions::default());

        let opts_no_ruiz = SolverOptions { use_ruiz_scaling: false, ..Default::default() };
        let result_no_ruiz = solve_qp_ipm(&problem, &opts_no_ruiz);

        assert_eq!(result_ruiz.status, SolveStatus::Optimal, "IPM-T7: ruiz status");
        assert_eq!(result_no_ruiz.status, SolveStatus::Optimal, "IPM-T7: no-ruiz status");
        close(result_ruiz.solution[0], result_no_ruiz.solution[0], "IPM-T7: x[0]");
        close(result_ruiz.solution[1], result_no_ruiz.solution[1], "IPM-T7: x[1]");
        close(result_ruiz.objective, result_no_ruiz.objective, "IPM-T7: objective");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // cmd_589 TDD赤フェーズ: テスト不足 (△) 項目
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// A2-T02: timeout_secs=0 で即停止（IPM版）
    /// Given: timeout_secs=0, When: solve_qp_ipm, Then: 即 Timeout
    #[test]
    fn test_a2t02_ipm_timeout_zero_returns_immediately() {
        // SPEC: A2-T02
        // timeout_secs=0 → deadline = now() → 最初の should_stop() で Timeout
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions { timeout_secs: Some(0.0), ..SolverOptions::default() };
        let start = std::time::Instant::now();
        let result = solve_qp_ipm(&problem, &opts);
        let elapsed = start.elapsed().as_secs_f64();
        // timeout_secs=0 → Timeout または Optimal（初期化中に偶然解けた場合）
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "A2-T02: timeout_secs=0 は Timeout または Optimal を返すこと。got: {:?}", result.status
        );
        assert!(
            elapsed < 0.5,
            "A2-T02: timeout_secs=0 は即座に返るべき。elapsed={:.3}s", elapsed
        );
    }

    /// A5-S01: Ruiz scaling 前後で解が等価（別問題でも確認）
    /// 既存 test_ipm_ruiz_scaling_consistency とは異なる制約条件の問題で確認
    #[test]
    fn test_a5s01_scaling_solution_equivalence_constrained() {
        // SPEC: A5-S01
        // 制約あり問題（T2: 等式制約 x+y=1）で scaling 有無で解が一致することを確認
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // x+y = 1 を Ax <= b 形式で: x+y<=1 かつ -(x+y)<=-1
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, -1.0, -1.0], 2, 2,
        ).unwrap();
        let b = vec![1.0, -1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();
        let result_ruiz = solve_qp_ipm(&problem, &SolverOptions::default());
        let opts_no_ruiz = SolverOptions { use_ruiz_scaling: false, ..SolverOptions::default() };
        let result_no_ruiz = solve_qp_ipm(&problem, &opts_no_ruiz);
        assert_eq!(result_ruiz.status, SolveStatus::Optimal, "A5-S01: ruiz=true → Optimal");
        assert_eq!(result_no_ruiz.status, SolveStatus::Optimal, "A5-S01: ruiz=false → Optimal");
        assert!(
            (result_ruiz.solution[0] - result_no_ruiz.solution[0]).abs() < 1e-4,
            "A5-S01: scaling 有無で x[0] が一致すること"
        );
        assert!(
            (result_ruiz.solution[1] - result_no_ruiz.solution[1]).abs() < 1e-4,
            "A5-S01: scaling 有無で x[1] が一致すること"
        );
    }

    /// A5-S02: POST_VERIFY で SuboptimalSolution を外部 API から返さない
    /// IPM/IPPMM パスでは SuboptimalSolution → Timeout 変換済み（cmd_582 以降）
    #[test]
    fn test_a5s02_post_verify_no_false_optimal() {
        // SPEC: A5-S02
        // Ruiz scaling で POST_VERIFY が SuboptimalSolution を検出した場合、
        // Timeout に変換されることを確認。
        // 小さな問題では通常 Optimal が返るため、「SuboptimalSolution が返らない」を確認。
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions { use_ruiz_scaling: true, ..SolverOptions::default() };
        let result = solve_qp_ipm(&problem, &opts);
        // SuboptimalSolution はバグステータス。返ってはならない。
        assert_ne!(
            result.status,
            SolveStatus::SuboptimalSolution,
            "A5-S02: SuboptimalSolution は外部 API から返ってはならない"
        );
        assert_eq!(result.status, SolveStatus::Optimal, "A5-S02: 正常ケースは Optimal");
    }

}
