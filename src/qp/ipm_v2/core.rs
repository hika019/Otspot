//! IPM 数値カーネル + 後処理 (Ruiz unscale, postsolve, bound clip, 元空間 KKT) の一貫処理。
//!
//! 設計原則:
//! - 入力は元 QpProblem と presolve 結果。reduced(scaled) は内部で扱う。
//! - 出力 IpmOutcome は **元空間** の解と残差のみを持つ。
//! - これにより `satisfies_eps(user_eps)` が常に元空間判定として機能する。

use crate::options::SolverOptions;
use crate::presolve::{postsolve_qp, QpPresolveResult};
use crate::problem::SolveStatus;
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;
use super::outcome::{IpmOutcome, ProblemView};
use super::kkt::{kkt_residual_rel, primal_residual_rel, bound_violation};

/// FX (固定) 変数判定の許容差 (kkt.rs と同じ値)
const FX_TOL: f64 = 1e-12;
/// KKT Newton refinement での NNLS-style iteration 上限
const KKT_NNLS_MAX_ITER: usize = 5;
/// x bound snap の許容差: x_j が bound から この値未満の距離なら bound に強制する (現状未使用)。
/// IPM は内点法で bound に完全到達しないため、postsolve 後に微小距離が残る。
/// これを snap して reduced→元空間 KKT ギャップを縮める意図だったが、
/// dual KKT が悪化するケース (DUAL1/3, STCQP2) で退行が発生したため無効化。
#[allow(dead_code)]
const X_SNAP_TOL: f64 = 1e-6;

/// 1 回の IPM 呼出 + 後処理。元空間の解と残差を返す。
pub fn run_ipm(
    orig_problem: &QpProblem,
    presolve_result: &QpPresolveResult,
    opts: &SolverOptions,
) -> IpmOutcome {
    let reduced = &presolve_result.reduced;
    let mut result = crate::qp::ipm::solve_qp_ippmm(reduced, opts);

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
            numerical_failure: true,
        };
    }

    // dual の post-process refinement (LSQ): scaled 空間で動かす方が IPM 出力との整合性が高い。
    if reduced.num_constraints > 0 {
        crate::qp::refine_dual_lsq(reduced, &mut result);
    }

    // Ruiz unscale: presolve が scaling 適用済みの場合のみ。
    if let Some(scaler) = &presolve_result.ruiz_scaler {
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

    // postsolve: reduced 空間 → 元問題空間
    let mut final_sol = postsolve_qp(presolve_result, &result);

    // bound_duals を元問題空間に remap
    if presolve_result.was_reduced {
        final_sol.bound_duals = crate::qp::remap_bound_duals_to_orig(
            presolve_result,
            &orig_problem.bounds,
            &final_sol.bound_duals,
        );
    }

    // bounds clip (Ruiz unscale 増幅由来の微小違反補正)
    for (xi, &(lb, ub)) in final_sol.solution.iter_mut().zip(orig_problem.bounds.iter()) {
        if lb.is_finite() {
            *xi = xi.max(lb);
        }
        if ub.is_finite() {
            *xi = xi.min(ub);
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

    // x bound snap (bound 近傍 x_j を bound に強制) は試行したが、
    // primal pres は OK でも dual KKT が悪化する (Q*x の変化を z, y が吸収しきれない) ため
    // DUAL1/DUAL3/STCQP2 で PASS→DFEAS_FAIL の退行が発生 (60s bench 99 PASS, baseline 102)。
    // 構造的に「post-processing を 2 回走らせて KKT 改善時のみ採用」の guard が必要だが
    // 実装コスト高のため現状は無効化。X_SNAP_TOL 定数は将来の参考用に保持。

    // 元空間で dual refinement: y を LSQ refine + active-set z-refit。
    // 各ステップで KKT 比較し、改善時のみ採用 (LSQ は L2 vs KKT は max-rel で目的関数が違うため)。
    let kkt = if !final_sol.solution.is_empty() {
        let mut current_kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        let active_tol = 1e-3_f64; // ε-active 変数を捕捉する許容差 (より緩和: x=1e-4 程度の準 active も拾う)

        // y refine (constraints ありの時のみ)
        if orig_problem.num_constraints > 0 {
            let pre_y = final_sol.dual_solution.clone();
            let pre_z = final_sol.bound_duals.clone();
            crate::qp::refine_dual_lsq(orig_problem, &mut final_sol);
            let post_kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
            if post_kkt <= current_kkt {
                current_kkt = post_kkt;
            } else {
                final_sol.dual_solution = pre_y;
                final_sol.bound_duals = pre_z;
            }
        }

        // active-set z-refit: x_j が bound 近傍 (距離 < active_tol) の変数で z を再計算する。
        let pre_z = final_sol.bound_duals.clone();
        refit_z_active_set(orig_problem, &final_sol.solution, &final_sol.dual_solution, &mut final_sol.bound_duals, active_tol);
        let post_kkt = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
        if post_kkt < current_kkt {
            current_kkt = post_kkt;
            // z 更新後にもう一度 y refine する (z 変更で y の最適点が動く可能性)
            if orig_problem.num_constraints > 0 {
                let pre_y2 = final_sol.dual_solution.clone();
                let pre_z2 = final_sol.bound_duals.clone();
                crate::qp::refine_dual_lsq(orig_problem, &mut final_sol);
                let post_kkt2 = kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals);
                if post_kkt2 < current_kkt {
                    current_kkt = post_kkt2;
                } else {
                    final_sol.dual_solution = pre_y2;
                    final_sol.bound_duals = pre_z2;
                }
            }
        } else {
            final_sol.bound_duals = pre_z;
        }

        // KKT system Newton refinement: y と z_active を NNLS-style で同時最適化。
        // y/z 単独 LSQ では捉えきれない結合を解く。
        // active_tol=1e-3 と同じ ε-active 判定を使用 (refit_z_active_set と一貫)。
        let pre_y = final_sol.dual_solution.clone();
        let pre_z = final_sol.bound_duals.clone();
        let mut new_y = final_sol.dual_solution.clone();
        let mut new_z = final_sol.bound_duals.clone();
        if dual_solve_kkt_lsq(orig_problem, &final_sol.solution, &mut new_y, &mut new_z, active_tol) {
            let post_kkt = kkt_residual_rel(&view, &final_sol.solution, &new_y, &new_z);
            if post_kkt < current_kkt {
                final_sol.dual_solution = new_y;
                final_sol.bound_duals = new_z;
                current_kkt = post_kkt;
            } else {
                final_sol.dual_solution = pre_y;
                final_sol.bound_duals = pre_z;
            }
        }

        current_kkt
    } else {
        kkt_residual_rel(&view, &final_sol.solution, &final_sol.dual_solution, &final_sol.bound_duals)
    };

    let pres = primal_residual_rel(&view, &final_sol.solution);
    let bv = bound_violation(orig_problem.bounds.as_slice(), &final_sol.solution);

    IpmOutcome {
        solution: final_sol.solution,
        dual_solution: final_sol.dual_solution,
        bound_duals: final_sol.bound_duals,
        objective: final_sol.objective,
        iterations: result.iterations,
        kkt_residual_rel: kkt,
        primal_residual_rel: pres,
        bound_violation: bv,
        numerical_failure: false,
    }
}

/// LP solver から得た primal x と reduced_costs/dual を QP として post-processing する。
///
/// 既存 IPM の後半 (bounds clip + dual refinement + active-set z-refit + KKT Newton)
/// と同じ流れを LP 解に適用し、元空間 KKT/primal/bound 残差付きの IpmOutcome を返す。
///
/// 初期 z は LP の reduced_costs (variable reduced cost, length n) から QP の
/// bound_duals レイアウト ([lb有限変数の z_lb; ub有限変数の z_ub]) に変換する。
/// reduced_cost_j > 0 → z_lb_j = rc_j (active at lb), < 0 → z_ub_j = -rc_j (active at ub)。
/// LP の y もそのまま QP の dual_solution として初期化する。
pub fn run_lp_postprocess(
    orig_problem: &QpProblem,
    mut x: Vec<f64>,
    lp_dual: Vec<f64>,
    lp_reduced_costs: Vec<f64>,
) -> IpmOutcome {
    let n = orig_problem.num_vars;
    if x.len() != n {
        return IpmOutcome::empty();
    }

    // bounds clip (LP solver からの数値誤差を補正)
    for (xi, &(lb, ub)) in x.iter_mut().zip(orig_problem.bounds.iter()) {
        if lb.is_finite() {
            *xi = xi.max(lb);
        }
        if ub.is_finite() {
            *xi = xi.min(ub);
        }
    }

    let n_lb = orig_problem.bounds.iter().filter(|(lb, _)| lb.is_finite()).count();
    let n_ub = orig_problem.bounds.iter().filter(|(_, ub)| ub.is_finite()).count();

    // LP の reduced_costs から QP の bound_duals 初期値を構築
    let mut z_init = vec![0.0_f64; n_lb + n_ub];
    if lp_reduced_costs.len() == n {
        let mut lb_idx = 0usize;
        let mut ub_idx = 0usize;
        for (j, &(lb, ub)) in orig_problem.bounds.iter().enumerate() {
            let rc_j = lp_reduced_costs[j];
            if lb.is_finite() {
                if rc_j > 0.0 {
                    z_init[lb_idx] = rc_j;
                }
                lb_idx += 1;
            }
            if ub.is_finite() {
                if rc_j < 0.0 {
                    z_init[n_lb + ub_idx] = -rc_j;
                }
                ub_idx += 1;
            }
        }
    }

    // LP の dual もそのまま QP の y 初期値として採用 (constraint の dual はレイアウト共通)
    let y_init = if lp_dual.len() == orig_problem.num_constraints {
        lp_dual
    } else {
        vec![0.0; orig_problem.num_constraints]
    };

    let mut tmp = crate::problem::SolverResult {
        status: SolveStatus::Optimal,
        solution: x.clone(),
        dual_solution: y_init,
        bound_duals: z_init,
        ..Default::default()
    };

    let view = ProblemView {
        q: &orig_problem.q,
        a: &orig_problem.a,
        c: &orig_problem.c,
        b: &orig_problem.b,
        bounds: &orig_problem.bounds,
        constraint_types: &orig_problem.constraint_types,
    };

    let active_tol = 1e-3_f64;

    // y refine (z 初期値固定で A^T y ≈ -(Qx + c + bound_contrib) を解く)
    if orig_problem.num_constraints > 0 {
        crate::qp::refine_dual_lsq(orig_problem, &mut tmp);
    }
    // active-set z-refit
    refit_z_active_set(orig_problem, &tmp.solution, &tmp.dual_solution, &mut tmp.bound_duals, active_tol);
    if orig_problem.num_constraints > 0 {
        crate::qp::refine_dual_lsq(orig_problem, &mut tmp);
    }

    let mut current_kkt = kkt_residual_rel(&view, &tmp.solution, &tmp.dual_solution, &tmp.bound_duals);

    // KKT Newton refinement (KKT-guard 付き)
    let mut new_y = tmp.dual_solution.clone();
    let mut new_z = tmp.bound_duals.clone();
    if dual_solve_kkt_lsq(orig_problem, &tmp.solution, &mut new_y, &mut new_z, active_tol) {
        let post_kkt = kkt_residual_rel(&view, &tmp.solution, &new_y, &new_z);
        if post_kkt < current_kkt {
            tmp.dual_solution = new_y;
            tmp.bound_duals = new_z;
            current_kkt = post_kkt;
        }
    }

    let pres = primal_residual_rel(&view, &tmp.solution);
    let bv = bound_violation(orig_problem.bounds.as_slice(), &tmp.solution);

    // 元 QP の objective を計算 (LP solver の objective は Q=0 として計算されている)
    let qx = orig_problem.q.mat_vec_mul(&tmp.solution).unwrap_or(vec![0.0; n]);
    let obj = 0.5 * qx.iter().zip(tmp.solution.iter()).map(|(a, b)| a * b).sum::<f64>()
        + orig_problem.c.iter().zip(tmp.solution.iter()).map(|(a, b)| a * b).sum::<f64>();

    IpmOutcome {
        solution: tmp.solution,
        dual_solution: tmp.dual_solution,
        bound_duals: tmp.bound_duals,
        objective: obj,
        iterations: 0,
        kkt_residual_rel: current_kkt,
        primal_residual_rel: pres,
        bound_violation: bv,
        numerical_failure: false,
    }
}

/// active-set ベースで bound_duals (z) を再計算する。
///
/// 変数 j の active 判定: |x_j - lb_j| < active_tol → active at lb、|x_j - ub_j| < active_tol → active at ub。
/// active at lb のみ → z_lb_j = max(0, r_j); z_ub_j = 0
/// active at ub のみ → z_ub_j = max(0, -r_j); z_lb_j = 0
/// FX (上下両 active) → 残差の符号で振り分け
/// inactive → z_lb_j = z_ub_j = 0
/// ここで r_j = (Q*x + c + A^T*y)_j。
fn refit_z_active_set(
    problem: &QpProblem,
    x: &[f64],
    y: &[f64],
    bound_duals: &mut Vec<f64>,
    active_tol: f64,
) {
    let n = problem.num_vars;
    if x.len() != n {
        return;
    }
    let qx = match problem.q.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return,
    };
    let aty = if problem.num_constraints > 0 && !y.is_empty() {
        match problem.a.transpose().mat_vec_mul(y) {
            Ok(v) => v,
            Err(_) => return,
        }
    } else {
        vec![0.0; n]
    };

    let n_lb = problem.bounds.iter().filter(|(lb, _)| lb.is_finite()).count();
    let n_ub = problem.bounds.iter().filter(|(_, ub)| ub.is_finite()).count();
    let mut new_bd = vec![0.0_f64; n_lb + n_ub];

    let mut lb_idx = 0;
    let mut ub_idx = 0;
    for j in 0..n {
        let (lb, ub) = problem.bounds[j];
        let r_j = qx[j] + problem.c[j] + aty[j];
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();
        let at_lb = lb_finite && (x[j] - lb).abs() < active_tol;
        let at_ub = ub_finite && (ub - x[j]).abs() < active_tol;

        if lb_finite {
            new_bd[lb_idx] = if at_lb && !at_ub {
                r_j.max(0.0)
            } else if at_lb && at_ub {
                r_j.max(0.0)
            } else {
                0.0
            };
            lb_idx += 1;
        }
        if ub_finite {
            new_bd[n_lb + ub_idx] = if at_ub && !at_lb {
                (-r_j).max(0.0)
            } else if at_lb && at_ub {
                (-r_j).max(0.0)
            } else {
                0.0
            };
            ub_idx += 1;
        }
    }
    *bound_duals = new_bd;
}

/// 明示 active 集合を渡す版の KKT system Newton refinement (現状未使用、将来の LP-hint 統合用)。
#[allow(dead_code)]
fn dual_solve_kkt_lsq_with_active_set(
    problem: &QpProblem,
    x: &[f64],
    active_lb_var: Vec<usize>,
    active_ub_var: Vec<usize>,
    y: &mut Vec<f64>,
    bound_duals: &mut Vec<f64>,
) -> bool {
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if x.len() != n {
        return false;
    }
    let bounds = &problem.bounds;

    let qx = match problem.q.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let r: Vec<f64> = (0..n).map(|j| -(qx[j] + problem.c[j])).collect();

    let na_lb = active_lb_var.len();
    let na_ub = active_ub_var.len();
    let nv = m + na_lb + na_ub;
    if nv == 0 {
        return false;
    }

    let at = problem.a.transpose();
    if at.nrows != n {
        return false;
    }
    let mut col_ptr: Vec<usize> = Vec::with_capacity(nv + 1);
    let mut row_ind: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    col_ptr.push(0);
    for j in 0..m {
        let (rows, vals) = match at.get_column(j) {
            Ok(v) => v,
            Err(_) => return false,
        };
        for (i, &row) in rows.iter().enumerate() {
            row_ind.push(row);
            values.push(vals[i]);
        }
        col_ptr.push(row_ind.len());
    }
    for &j in &active_lb_var {
        row_ind.push(j);
        values.push(-1.0);
        col_ptr.push(row_ind.len());
    }
    for &j in &active_ub_var {
        row_ind.push(j);
        values.push(1.0);
        col_ptr.push(row_ind.len());
    }
    let m_mat = CscMatrix {
        col_ptr,
        row_ind,
        values,
        nrows: n,
        ncols: nv,
    };

    let nz = na_lb + na_ub;
    let mut fixed_zero = vec![false; nz];
    let mut v_full: Vec<f64> = vec![0.0; nv];

    for _iter in 0..KKT_NNLS_MAX_ITER {
        let free_z: Vec<usize> = (0..nz).filter(|&k| !fixed_zero[k]).collect();
        let nf = m + free_z.len();
        if nf == 0 {
            return false;
        }
        let mut col_ptr_f: Vec<usize> = Vec::with_capacity(nf + 1);
        let mut row_ind_f: Vec<usize> = Vec::new();
        let mut values_f: Vec<f64> = Vec::new();
        col_ptr_f.push(0);
        for j in 0..m {
            let s = m_mat.col_ptr[j];
            let e = m_mat.col_ptr[j + 1];
            for p in s..e {
                row_ind_f.push(m_mat.row_ind[p]);
                values_f.push(m_mat.values[p]);
            }
            col_ptr_f.push(row_ind_f.len());
        }
        for &k in &free_z {
            let src_col = m + k;
            let s = m_mat.col_ptr[src_col];
            let e = m_mat.col_ptr[src_col + 1];
            for p in s..e {
                row_ind_f.push(m_mat.row_ind[p]);
                values_f.push(m_mat.values[p]);
            }
            col_ptr_f.push(row_ind_f.len());
        }
        let mf = CscMatrix {
            col_ptr: col_ptr_f,
            row_ind: row_ind_f,
            values: values_f,
            nrows: n,
            ncols: nf,
        };

        let mft = mf.transpose();
        let mtr_f = match mft.mat_vec_mul(&r) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let mtm_f = match crate::qp::build_aat_upper_csc(&mft, n, nf) {
            Some(v) => v,
            None => return false,
        };
        let factor = match crate::linalg::ldl::factorize(&mtm_f) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mut v_f = vec![0.0; nf];
        factor.solve(&mtr_f, &mut v_f);
        if v_f.iter().any(|x| !x.is_finite()) {
            return false;
        }
        for j in 0..m {
            v_full[j] = v_f[j];
        }
        for k in 0..nz {
            v_full[m + k] = 0.0;
        }
        for (idx, &k) in free_z.iter().enumerate() {
            v_full[m + k] = v_f[m + idx];
        }
        let mut worst_neg: Option<(usize, f64)> = None;
        for &k in &free_z {
            let val = v_full[m + k];
            if val < 0.0 {
                match worst_neg {
                    None => worst_neg = Some((k, val)),
                    Some((_, prev)) if val < prev => worst_neg = Some((k, val)),
                    _ => {}
                }
            }
        }
        match worst_neg {
            Some((k, _)) => {
                fixed_zero[k] = true;
                v_full[m + k] = 0.0;
            }
            None => break,
        }
    }
    for k in 0..nz {
        if v_full[m + k] < 0.0 {
            v_full[m + k] = 0.0;
        }
    }

    let n_lb_total = bounds.iter().filter(|(lb, _)| lb.is_finite()).count();
    let n_ub_total = bounds.iter().filter(|(_, ub)| ub.is_finite()).count();
    let mut new_bd = vec![0.0_f64; n_lb_total + n_ub_total];
    let mut lb_pos = vec![0usize; n];
    let mut ub_pos = vec![0usize; n];
    {
        let mut li = 0usize;
        for j in 0..n {
            if bounds[j].0.is_finite() {
                lb_pos[j] = li;
                li += 1;
            }
        }
        let mut ui = 0usize;
        for j in 0..n {
            if bounds[j].1.is_finite() {
                ub_pos[j] = ui;
                ui += 1;
            }
        }
    }
    for (k, &j) in active_lb_var.iter().enumerate() {
        new_bd[lb_pos[j]] = v_full[m + k];
    }
    for (k, &j) in active_ub_var.iter().enumerate() {
        new_bd[n_lb_total + ub_pos[j]] = v_full[m + na_lb + k];
    }

    *y = v_full[..m].to_vec();
    *bound_duals = new_bd;
    true
}

/// KKT system Newton refinement: y と active な z を NNLS-style の active-set 法で同時最適化する。
///
/// stationarity `Q*x + c + A^T y - z_lb + z_ub = 0` を、x 固定下で
/// 自由変数 [y; z_lb_active; z_ub_active] の線形 LSQ として解く (z は ≥0 制約)。
///
/// アルゴリズム:
/// 1. ε-active 判定: x_j が lb 近傍 → active_lb_var に追加、ub 近傍 → active_ub_var に追加
/// 2. 線形系 M v = r を構築 (r = -(Q*x + c)、M は A^T と active 変数の indicator を結合)
/// 3. NNLS-style: 全自由 LSQ → 負の z を active set から外す → 再 solve、最大 KKT_NNLS_MAX_ITER 回反復
///
/// inactive な z は 0 に固定 (補完性)。FX 変数は KKT 評価から除外されるため active_set からも除外。
///
/// 戻り値: 改善試行に成功 (有限値で解けた) なら true。呼出側は KKT-guard で改善時のみ採用すべき。
fn dual_solve_kkt_lsq(
    problem: &QpProblem,
    x: &[f64],
    y: &mut Vec<f64>,
    bound_duals: &mut Vec<f64>,
    active_tol: f64,
) -> bool {
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if x.len() != n {
        return false;
    }
    let bounds = &problem.bounds;

    let qx = match problem.q.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let r: Vec<f64> = (0..n).map(|j| -(qx[j] + problem.c[j])).collect();

    let mut active_lb_var: Vec<usize> = Vec::new();
    let mut active_ub_var: Vec<usize> = Vec::new();
    for j in 0..n {
        let (lb, ub) = bounds[j];
        if lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL {
            continue;
        }
        if lb.is_finite() && (x[j] - lb).abs() < active_tol {
            active_lb_var.push(j);
        }
        if ub.is_finite() && (ub - x[j]).abs() < active_tol {
            active_ub_var.push(j);
        }
    }
    let na_lb = active_lb_var.len();
    let na_ub = active_ub_var.len();
    let nv = m + na_lb + na_ub;

    if nv == 0 {
        return false;
    }

    // M を CSC で構築 (n × nv)
    // - col 0..m: A^T の各列 (= A の各行を転置, n 行)
    // - col m..m+na_lb: -e_{active_lb_var[k]} (n 行のうち active_lb_var[k] のみ -1)
    // - col m+na_lb..nv: +e_{active_ub_var[k]}
    let at = problem.a.transpose();
    if at.nrows != n {
        return false;
    }
    let mut col_ptr: Vec<usize> = Vec::with_capacity(nv + 1);
    let mut row_ind: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    col_ptr.push(0);
    for j in 0..m {
        let (rows, vals) = match at.get_column(j) {
            Ok(v) => v,
            Err(_) => return false,
        };
        for (i, &row) in rows.iter().enumerate() {
            row_ind.push(row);
            values.push(vals[i]);
        }
        col_ptr.push(row_ind.len());
    }
    for &j in &active_lb_var {
        row_ind.push(j);
        values.push(-1.0);
        col_ptr.push(row_ind.len());
    }
    for &j in &active_ub_var {
        row_ind.push(j);
        values.push(1.0);
        col_ptr.push(row_ind.len());
    }
    let m_mat = CscMatrix {
        col_ptr,
        row_ind,
        values,
        nrows: n,
        ncols: nv,
    };

    // active set から固定除外する z を保持する集合 (NNLS-style)
    // fixed_zero[k]: k 番目の z (k ∈ [0, na_lb+na_ub)) が active set から外されたか
    let nz = na_lb + na_ub;
    let mut fixed_zero = vec![false; nz];

    let mut v_full: Vec<f64> = vec![0.0; nv];

    for _iter in 0..KKT_NNLS_MAX_ITER {
        // 自由変数: y (m 個全て自由) + z のうち !fixed_zero
        let free_z: Vec<usize> = (0..nz).filter(|&k| !fixed_zero[k]).collect();
        let nf = m + free_z.len();
        if nf == 0 {
            // 全 z 固定 + m=0: 退避
            return false;
        }

        // 自由 sub-system M_f を抽出 (n × nf)
        let mut col_ptr_f: Vec<usize> = Vec::with_capacity(nf + 1);
        let mut row_ind_f: Vec<usize> = Vec::new();
        let mut values_f: Vec<f64> = Vec::new();
        col_ptr_f.push(0);
        // y 部分 (col 0..m はそのまま M の col 0..m)
        for j in 0..m {
            let s = m_mat.col_ptr[j];
            let e = m_mat.col_ptr[j + 1];
            for p in s..e {
                row_ind_f.push(m_mat.row_ind[p]);
                values_f.push(m_mat.values[p]);
            }
            col_ptr_f.push(row_ind_f.len());
        }
        // 自由な z 部分
        for &k in &free_z {
            let src_col = m + k;
            let s = m_mat.col_ptr[src_col];
            let e = m_mat.col_ptr[src_col + 1];
            for p in s..e {
                row_ind_f.push(m_mat.row_ind[p]);
                values_f.push(m_mat.values[p]);
            }
            col_ptr_f.push(row_ind_f.len());
        }
        let mf = CscMatrix {
            col_ptr: col_ptr_f,
            row_ind: row_ind_f,
            values: values_f,
            nrows: n,
            ncols: nf,
        };

        let mft = mf.transpose();
        let mtr_f = match mft.mat_vec_mul(&r) {
            Ok(v) => v,
            Err(_) => return false,
        };

        // M_f^T M_f (上三角, ε regularization 付き) — build_aat_upper_csc(M_f^T) で計算
        let mtm_f = match crate::qp::build_aat_upper_csc(&mft, n, nf) {
            Some(v) => v,
            None => return false,
        };

        let factor = match crate::linalg::ldl::factorize(&mtm_f) {
            Ok(f) => f,
            Err(_) => return false,
        };

        let mut v_f = vec![0.0; nf];
        factor.solve(&mtr_f, &mut v_f);
        if v_f.iter().any(|x| !x.is_finite()) {
            return false;
        }

        // v_full に書き戻し
        for j in 0..m {
            v_full[j] = v_f[j];
        }
        for k in 0..nz {
            v_full[m + k] = 0.0;
        }
        for (idx, &k) in free_z.iter().enumerate() {
            v_full[m + k] = v_f[m + idx];
        }

        // 最も負の z を見つけて active set から外す。なければ break。
        let mut worst_neg: Option<(usize, f64)> = None;
        for &k in &free_z {
            let val = v_full[m + k];
            if val < 0.0 {
                match worst_neg {
                    None => worst_neg = Some((k, val)),
                    Some((_, prev)) if val < prev => worst_neg = Some((k, val)),
                    _ => {}
                }
            }
        }
        match worst_neg {
            Some((k, _)) => {
                fixed_zero[k] = true;
                v_full[m + k] = 0.0;
            }
            None => break,
        }
    }

    // 残った負成分を最終的に 0 へ (max_iter 到達時の保険)
    for k in 0..nz {
        if v_full[m + k] < 0.0 {
            v_full[m + k] = 0.0;
        }
    }

    // bound_duals に書き戻し ([lb 有限な変数の z_lb; ub 有限な変数の z_ub] レイアウト)
    let n_lb_total = bounds.iter().filter(|(lb, _)| lb.is_finite()).count();
    let n_ub_total = bounds.iter().filter(|(_, ub)| ub.is_finite()).count();
    let mut new_bd = vec![0.0_f64; n_lb_total + n_ub_total];

    let mut lb_pos = vec![0usize; n];
    let mut ub_pos = vec![0usize; n];
    {
        let mut li = 0usize;
        for j in 0..n {
            if bounds[j].0.is_finite() {
                lb_pos[j] = li;
                li += 1;
            }
        }
        let mut ui = 0usize;
        for j in 0..n {
            if bounds[j].1.is_finite() {
                ub_pos[j] = ui;
                ui += 1;
            }
        }
    }
    for (k, &j) in active_lb_var.iter().enumerate() {
        new_bd[lb_pos[j]] = v_full[m + k];
    }
    for (k, &j) in active_ub_var.iter().enumerate() {
        new_bd[n_lb_total + ub_pos[j]] = v_full[m + na_lb + k];
    }

    *y = v_full[..m].to_vec();
    *bound_duals = new_bd;
    true
}

