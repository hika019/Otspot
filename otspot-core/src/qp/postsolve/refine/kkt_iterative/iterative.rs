//! Wilkinson 流 KKT iterative refinement。

use super::bound_refit::refit_bound_duals_kkt;
use crate::qp::postsolve::dual_recovery::dual_recovery_progress_tol;
use crate::qp::postsolve::postprocess::{run_dual_recovery_postprocess, try_dual_only_ir};
use crate::qp::problem::QpProblem;

pub(crate) fn refine_kkt_iterative(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    eliminated_cols: &[bool],
    max_iters: usize,
    target_pf: f64,
    deadline: Option<std::time::Instant>,
) -> usize {
    use crate::presolve::bound_contrib_at_var;
    use crate::problem::ConstraintType;
    use crate::qp::ipm_solver::kkt::kkt_residual_rel;

    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return 0;
    }

    let n = problem.num_vars;
    let m = problem.num_constraints;
    if m == 0 || result.solution.len() != n {
        return 0;
    }
    if result.dual_solution.len() != m {
        return 0;
    }

    // KKT 反復 refinement の時間予算 proxy。saddle-point K の factorize は
    // deadline-aware (下の factorize_quasidefinite_with_amd) だが、巨大問題では
    // 単発 factorize が deadline を空費する。post-processing 段で K factorize を
    // 行うか否かの規模ガード (n+m で判定)。
    const REFINE_KKT_SIZE_LIMIT: usize = 50_000;
    if n + m > REFINE_KKT_SIZE_LIMIT {
        return 0;
    }

    // Dual-only IR (x 不変 / y のみ更新) を target_pf 達成まで反復。
    // saddle-point K の ill-conditioned (1,1) ブロックで dx が暴走する問題を回避。
    let mut n_dual_total = 0_usize;
    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
        eliminated_cols,
    };
    let mut prev_kkt = kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    let mut best_kkt = prev_kkt;
    let mut best_result = result.clone();
    for _outer in 0..max_iters.max(1) {
        let mut outer_made_progress = false;
        let n_dual = try_dual_only_ir(problem, result, eliminated_cols, target_pf, deadline);
        if n_dual > 0 {
            n_dual_total += n_dual;
            outer_made_progress = true;
            let _ = run_dual_recovery_postprocess(problem, &view, result, deadline);
        } else {
            let pre_cleanup_kkt = kkt_residual_rel(
                &view,
                &result.solution,
                &result.dual_solution,
                &result.bound_duals,
            );
            let post_cleanup_kkt =
                run_dual_recovery_postprocess(problem, &view, result, deadline);
            if post_cleanup_kkt + dual_recovery_progress_tol(pre_cleanup_kkt, post_cleanup_kkt, target_pf)
                < pre_cleanup_kkt
            {
                outer_made_progress = true;
            }
        }
        if !outer_made_progress {
            break;
        }
        let cur_kkt = kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        if cur_kkt < best_kkt {
            best_kkt = cur_kkt;
            best_result = result.clone();
        }
        if cur_kkt < target_pf {
            break;
        }
        let progress_tol = dual_recovery_progress_tol(prev_kkt, cur_kkt, target_pf);
        if cur_kkt + progress_tol >= prev_kkt {
            break;
        }
        prev_kkt = cur_kkt;
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
    }
    if n_dual_total > 0 {
        *result = best_result;
        if best_kkt < target_pf || deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return n_dual_total;
        }
    }
    // dual-only で改善できない / 不十分なら saddle-point IR に fall-through。

    // K = [Q+δp·I, A^T; A, -δd·I] の対角正則化。十分小さく IR で eps·‖K‖ まで refine 可。
    const DELTA_P_DEFAULT: f64 = 1e-10;
    const DELTA_D_DEFAULT: f64 = 1e-10;
    let (delta_p, delta_d) = (DELTA_P_DEFAULT, DELTA_D_DEFAULT);

    let sigma_zero = vec![0.0_f64; m];
    let mut k_mat = crate::qp::ipm_core::kkt::build_augmented_system(
        &problem.q,
        &problem.a,
        &sigma_zero,
        delta_p,
        delta_d,
    );

    let diag_on = std::env::var("REFINE_KKT_DIAG").ok().as_deref() == Some("1");

    // bound-active 変数の dx を K 対角 penalty で抑制 (近似 active set fix)。
    const ACTIVE_TOL: f64 = 1e-8;
    const ACTIVE_PENALTY_RATIO: f64 = 1e8;
    let active_fix_enabled = true;
    if active_fix_enabled {
        let mut k_diag_max = 0.0_f64;
        for j in 0..(n + m) {
            let cs = k_mat.col_ptr[j];
            let ce = k_mat.col_ptr[j + 1];
            for k in cs..ce {
                if k_mat.row_ind[k] == j {
                    k_diag_max = k_diag_max.max(k_mat.values[k].abs());
                    break;
                }
            }
        }
        let active_penalty = (k_diag_max * ACTIVE_PENALTY_RATIO).max(ACTIVE_PENALTY_RATIO);
        let mut penalized = 0_usize;
        for j in 0..n {
            let x = result.solution[j];
            let (lb, ub) = problem.bounds[j];
            let is_active = (lb.is_finite() && (x - lb).abs() < ACTIVE_TOL)
                || (ub.is_finite() && (ub - x).abs() < ACTIVE_TOL);
            if !is_active {
                continue;
            }
            let col_start = k_mat.col_ptr[j];
            let col_end = k_mat.col_ptr[j + 1];
            for k in col_start..col_end {
                if k_mat.row_ind[k] == j {
                    k_mat.values[k] += active_penalty;
                    penalized += 1;
                    break;
                }
            }
        }
        if diag_on && penalized > 0 {
            eprintln!("REFINE_KKT bound-active fix: penalized {} vars (PENALTY={:.2e}, K_diag_max={:.2e})",
                penalized, active_penalty, k_diag_max);
        }
    }
    if diag_on {
        let mut diag_top_min = f64::INFINITY;
        let mut diag_top_max = f64::NEG_INFINITY;
        let mut diag_top_abs_min = f64::INFINITY;
        let mut diag_bot_min = f64::INFINITY;
        let mut diag_bot_max = f64::NEG_INFINITY;
        let mut diag_bot_abs_min = f64::INFINITY;
        for j in 0..(n + m) {
            let col_start = k_mat.col_ptr[j];
            let col_end = k_mat.col_ptr[j + 1];
            for k in col_start..col_end {
                if k_mat.row_ind[k] == j {
                    let v = k_mat.values[k];
                    if j < n {
                        diag_top_min = diag_top_min.min(v);
                        diag_top_max = diag_top_max.max(v);
                        diag_top_abs_min = diag_top_abs_min.min(v.abs());
                    } else {
                        diag_bot_min = diag_bot_min.min(v);
                        diag_bot_max = diag_bot_max.max(v);
                        diag_bot_abs_min = diag_bot_abs_min.min(v.abs());
                    }
                    break;
                }
            }
        }
        eprintln!(
            "REFINE_KKT_DIAG K_diag top(Q+δp·I)=[min={:.3e} max={:.3e} abs_min={:.3e}] bot(-δd·I)=[min={:.3e} max={:.3e} abs_min={:.3e}]",
            diag_top_min, diag_top_max, diag_top_abs_min,
            diag_bot_min, diag_bot_max, diag_bot_abs_min
        );
        let abs_max = k_mat.values.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let abs_min_nz = k_mat
            .values
            .iter()
            .filter(|&&v| v != 0.0)
            .fold(f64::INFINITY, |a, &v| a.min(v.abs()));
        eprintln!(
            "REFINE_KKT_DIAG K_all abs_max={:.3e} abs_min_nz={:.3e} ratio={:.3e}",
            abs_max,
            abs_min_nz,
            abs_max / abs_min_nz.max(1e-300)
        );
    }
    // On SingularOrIndefinite, grow δ by FACTOR_RETRY_GROWTH and retry until
    // factorization succeeds; the first success is the smallest δ that works
    // (deltas grow monotonically). Deadline guards against large-K stalls.

    const FACTOR_RETRY_GROWTH: f64 = 10.0;
    const FACTOR_RETRY_MAX: usize = 6;
    let factor = {
        let mut current_delta_p = delta_p;
        let mut current_delta_d = delta_d;
        let mut current_k = k_mat.clone();
        let mut result_factor: Option<crate::linalg::ldl::LdlFactorizationAmd> = None;
        let mut retry_count = 0usize;
        loop {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                if diag_on {
                    eprintln!(
                        "REFINE_KKT factorize abandoned due to deadline at retry={}",
                        retry_count
                    );
                }
                break;
            }
            match crate::linalg::ldl::factorize_quasidefinite_with_amd(&current_k, deadline) {
                Ok(f) => {
                    result_factor = Some(f);
                    break;
                }
                Err(e) => {
                    if retry_count >= FACTOR_RETRY_MAX {
                        if diag_on {
                            eprintln!("REFINE_KKT factorize failed after {} retries: {:?} (last delta_p={:.1e} delta_d={:.1e})",
                                retry_count, e, current_delta_p, current_delta_d);
                        }
                        break;
                    }
                    retry_count += 1;
                    current_delta_p *= FACTOR_RETRY_GROWTH;
                    current_delta_d *= FACTOR_RETRY_GROWTH;
                    current_k = crate::qp::ipm_core::kkt::build_augmented_system(
                        &problem.q,
                        &problem.a,
                        &sigma_zero,
                        current_delta_p,
                        current_delta_d,
                    );
                    if active_fix_enabled {
                        let mut k_diag_max_retry = 0.0_f64;
                        for j in 0..(n + m) {
                            let cs = current_k.col_ptr[j];
                            let ce = current_k.col_ptr[j + 1];
                            for k in cs..ce {
                                if current_k.row_ind[k] == j {
                                    k_diag_max_retry =
                                        k_diag_max_retry.max(current_k.values[k].abs());
                                    break;
                                }
                            }
                        }
                        let active_penalty_retry =
                            (k_diag_max_retry * ACTIVE_PENALTY_RATIO).max(ACTIVE_PENALTY_RATIO);
                        for j in 0..n {
                            let x = result.solution[j];
                            let (lb, ub) = problem.bounds[j];
                            let is_active = (lb.is_finite() && (x - lb).abs() < ACTIVE_TOL)
                                || (ub.is_finite() && (ub - x).abs() < ACTIVE_TOL);
                            if !is_active {
                                continue;
                            }
                            let cs = current_k.col_ptr[j];
                            let ce = current_k.col_ptr[j + 1];
                            for k in cs..ce {
                                if current_k.row_ind[k] == j {
                                    current_k.values[k] += active_penalty_retry;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        if diag_on && retry_count > 0 && result_factor.is_some() {
            eprintln!("REFINE_KKT factorize succeeded after {} retries (final delta_p={:.1e} delta_d={:.1e})",
                retry_count, current_delta_p, current_delta_d);
        }
        match result_factor {
            Some(f) => f,
            None => return 0,
        }
    };
    if diag_on {
        // cond 代理: ||K^-1·r||_∞ / ||r||_∞ (xorshift64 RHS)。
        let mut rng_state: u64 = 0x9E3779B97F4A7C15;
        let mut rhs = vec![0.0_f64; n + m];
        for v in rhs.iter_mut() {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            *v = ((rng_state as f64) / (u64::MAX as f64)) * 2.0 - 1.0;
        }
        let rhs_inf = rhs.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let mut sol = vec![0.0_f64; n + m];
        factor.solve(&rhs, &mut sol);
        let sol_inf = sol.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let any_nan = sol.iter().any(|v| !v.is_finite());
        eprintln!(
            "REFINE_KKT_DIAG cond_proxy: ||K^-1·rand||_∞ / ||rand||_∞ = {:.3e} / {:.3e} = {:.3e} nan={}",
            sol_inf, rhs_inf, sol_inf / rhs_inf.max(1e-300), any_nan
        );
    }

    // Exclude FX vars (lb≈ub) and presolve-eliminated columns from stationarity.
    use crate::tolerances::FX_TOL;
    let use_elim_mask = eliminated_cols.len() == n;
    let exclude_var: Vec<bool> = (0..n)
        .map(|j| {
            let (lb, ub) = problem.bounds[j];
            if lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL {
                return true;
            }
            if use_elim_mask && eliminated_cols[j] {
                return true;
            }
            false
        })
        .collect();

    // (r_d, r_p, pf_abs, df_abs, pf_rel, df_rel) を返す。pf_rel/df_rel は OSQP-style componentwise。
    let compute_residuals =
        |x: &[f64], y: &[f64], z: &[f64]| -> (Vec<f64>, Vec<f64>, f64, f64, f64, f64) {
            let qx = problem.q.mat_vec_mul(x).unwrap_or_else(|_| vec![0.0; n]);
            let aty = problem
                .a
                .transpose()
                .mat_vec_mul(y)
                .unwrap_or_else(|_| vec![0.0; n]);
            let mut r_d = vec![0.0_f64; n];
            let mut df_rel_componentwise = 0.0_f64;
            for j in 0..n {
                if exclude_var[j] {
                    continue;
                }
                let bc = bound_contrib_at_var(&problem.bounds, z, j);
                r_d[j] = qx[j] + problem.c[j] + aty[j] + bc;
                let scale_j = 1.0 + qx[j].abs() + problem.c[j].abs() + aty[j].abs() + bc.abs();
                let rel_j = r_d[j].abs() / scale_j;
                if rel_j > df_rel_componentwise {
                    df_rel_componentwise = rel_j;
                }
            }
            let ax = problem.a.mat_vec_mul(x).unwrap_or_else(|_| vec![0.0; m]);
            let mut r_p = vec![0.0_f64; m];
            let mut pf_abs = 0.0_f64;
            let mut pf_rel_componentwise = 0.0_f64;
            for i in 0..m {
                let raw = ax[i] - problem.b[i];
                let v = match problem.constraint_types[i] {
                    ConstraintType::Eq => raw,
                    ConstraintType::Ge => {
                        if raw < 0.0 {
                            raw
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
                r_p[i] = v;
                pf_abs = pf_abs.max(v.abs());
                let scale_i = 1.0 + ax[i].abs() + problem.b[i].abs();
                let rel_i = v.abs() / scale_i;
                if rel_i > pf_rel_componentwise {
                    pf_rel_componentwise = rel_i;
                }
            }
            let df_abs = r_d.iter().fold(0.0_f64, |a, &r| a.max(r.abs()));
            (
                r_d,
                r_p,
                pf_abs,
                df_abs,
                pf_rel_componentwise,
                df_rel_componentwise,
            )
        };

    // Wilkinson IR の "double the working precision": Qx, A^T y, Ax を TwoFloat (DD) で積算し
    // residual を f64 limit 以下に精密化。LDL solve は f64 のまま。
    let dd_mode = true;
    let compute_residuals_dd =
        |x: &[f64], y: &[f64], z: &[f64]| -> (Vec<f64>, Vec<f64>, f64, f64, f64, f64) {
            use twofloat::TwoFloat;
            let zero_dd = TwoFloat::from(0.0);
            // Q は全要素格納 (上下三角両方)、symmetric duplication せず CSC 全走査。
            let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
            for j in 0..n {
                let xv = x[j];
                let cs = problem.q.col_ptr[j];
                let ce = problem.q.col_ptr[j + 1];
                for k in cs..ce {
                    let row = problem.q.row_ind[k];
                    let v = problem.q.values[k];
                    qx_dd[row] += TwoFloat::new_mul(v, xv);
                }
            }
            let mut aty_dd: Vec<TwoFloat> = vec![zero_dd; n];
            for col in 0..n {
                let cs = problem.a.col_ptr[col];
                let ce = problem.a.col_ptr[col + 1];
                for k in cs..ce {
                    let row = problem.a.row_ind[k];
                    let v = problem.a.values[k];
                    aty_dd[col] += TwoFloat::new_mul(v, y[row]);
                }
            }
            let mut r_d = vec![0.0_f64; n];
            for j in 0..n {
                if exclude_var[j] {
                    continue;
                }
                let bc = bound_contrib_at_var(&problem.bounds, z, j);
                let r = qx_dd[j] + TwoFloat::from(problem.c[j]) + aty_dd[j] + TwoFloat::from(bc);
                r_d[j] = f64::from(r);
            }
            let mut ax_dd: Vec<TwoFloat> = vec![zero_dd; m];
            for col in 0..n {
                let cs = problem.a.col_ptr[col];
                let ce = problem.a.col_ptr[col + 1];
                for k in cs..ce {
                    let row = problem.a.row_ind[k];
                    let v = problem.a.values[k];
                    ax_dd[row] += TwoFloat::new_mul(v, x[col]);
                }
            }
            let mut r_p = vec![0.0_f64; m];
            let mut pf_abs = 0.0_f64;
            let mut pf_rel_componentwise = 0.0_f64;
            for i in 0..m {
                let raw_dd = ax_dd[i] - TwoFloat::from(problem.b[i]);
                let raw = f64::from(raw_dd);
                let v = match problem.constraint_types[i] {
                    ConstraintType::Eq => raw,
                    ConstraintType::Ge => {
                        if raw < 0.0 {
                            raw
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
                r_p[i] = v;
                pf_abs = pf_abs.max(v.abs());
                let ax_i_abs = f64::from(ax_dd[i]).abs();
                let scale_i = 1.0 + ax_i_abs + problem.b[i].abs();
                let rel_i = v.abs() / scale_i;
                if rel_i > pf_rel_componentwise {
                    pf_rel_componentwise = rel_i;
                }
            }
            let df_abs = r_d.iter().fold(0.0_f64, |a, &r| a.max(r.abs()));
            // componentwise が必須 (全体相対化は ill-scaled で 1 成分外れを見逃す)。
            let mut df_rel_componentwise = 0.0_f64;
            for j in 0..n {
                if exclude_var[j] {
                    continue;
                }
                let qx_j = f64::from(qx_dd[j]).abs();
                let aty_j = f64::from(aty_dd[j]).abs();
                let bc = bound_contrib_at_var(&problem.bounds, z, j);
                let scale_j = 1.0 + qx_j + problem.c[j].abs() + aty_j + bc.abs();
                let rel_j = r_d[j].abs() / scale_j;
                if rel_j > df_rel_componentwise {
                    df_rel_componentwise = rel_j;
                }
            }
            (
                r_d,
                r_p,
                pf_abs,
                df_abs,
                pf_rel_componentwise,
                df_rel_componentwise,
            )
        };

    let pre_z = result.bound_duals.clone();
    let (_, _, _pre_pf, _pre_df, pre_pf_rel, pre_df_rel) = if dd_mode {
        compute_residuals_dd(&result.solution, &result.dual_solution, &pre_z)
    } else {
        compute_residuals(&result.solution, &result.dual_solution, &pre_z)
    };
    if pre_pf_rel < target_pf && pre_df_rel < target_pf {
        return 0;
    }

    let mut accepted = n_dual_total;
    // 残差悪化許容: max(pre_rel × 2, target_pf × 100) を超えたら revert。
    const RESID_TOLERANCE_FACTOR: f64 = 2.0;
    const RESID_FLOOR_RATIO: f64 = 100.0;
    let resid_floor = target_pf * RESID_FLOOR_RATIO;
    let pf_limit = (pre_pf_rel * RESID_TOLERANCE_FACTOR).max(resid_floor);
    let df_limit = (pre_df_rel * RESID_TOLERANCE_FACTOR).max(resid_floor);

    for iter in 0..max_iters {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        let (r_d, r_p, pf_abs_cur, df_abs_cur, pf_cur, df_cur) = if dd_mode {
            compute_residuals_dd(&result.solution, &result.dual_solution, &result.bound_duals)
        } else {
            compute_residuals(&result.solution, &result.dual_solution, &result.bound_duals)
        };
        if pf_cur < target_pf && df_cur < target_pf {
            break;
        }
        let _ = (pf_abs_cur, df_abs_cur);

        let mut rhs = vec![0.0_f64; n + m];
        for j in 0..n {
            rhs[j] = -r_d[j];
        }
        for i in 0..m {
            rhs[n + i] = -r_p[i];
        }

        let mut sol = vec![0.0_f64; n + m];
        factor.solve(&rhs, &mut sol);
        if sol.iter().any(|v| !v.is_finite()) {
            break;
        }

        let mut x_new = result.solution.clone();
        let mut y_new = result.dual_solution.clone();
        let mut clip_amt = 0.0_f64;
        let mut clip_count = 0_usize;
        let mut clip_top: Vec<(usize, f64)> = Vec::new();
        for j in 0..n {
            let raw = x_new[j] + sol[j];
            let (lb, ub) = problem.bounds[j];
            let mut clipped = raw;
            if lb.is_finite() {
                clipped = clipped.max(lb);
            }
            if ub.is_finite() {
                clipped = clipped.min(ub);
            }
            let amt = (raw - clipped).abs();
            clip_amt = clip_amt.max(amt);
            if amt > 0.0 {
                clip_count += 1;
                if diag_on {
                    clip_top.push((j, amt));
                }
            }
            x_new[j] = clipped;
        }
        if diag_on && !clip_top.is_empty() {
            clip_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top5: Vec<String> = clip_top
                .iter()
                .take(5)
                .map(|(j, a)| format!("x[{}]={:.2e}", j, a))
                .collect();
            eprintln!(
                "REFINE_KKT_DIAG iter={} clip_count={}/{} clip_max={:.3e} top5: {}",
                iter,
                clip_count,
                n,
                clip_amt,
                top5.join(", ")
            );
        }
        for i in 0..m {
            y_new[i] += sol[n + i];
        }

        let mut tmp = result.clone();
        tmp.solution = x_new;
        tmp.dual_solution = y_new;
        refit_bound_duals_kkt(problem, &mut tmp);

        let (_, _, _pf_abs_new, _df_abs_new, pf_new, df_new) = if dd_mode {
            compute_residuals_dd(&tmp.solution, &tmp.dual_solution, &tmp.bound_duals)
        } else {
            compute_residuals(&tmp.solution, &tmp.dual_solution, &tmp.bound_duals)
        };

        // 採用: max(pf_rel, df_rel) strict 減少 + 両者 guardrail 内。
        let score_cur = pf_cur.max(df_cur);
        let score_new = pf_new.max(df_new);
        let progress = score_new < score_cur;
        let pf_safe = pf_new < pf_limit;
        let df_safe = df_new < df_limit;
        if progress && pf_safe && df_safe {
            *result = tmp;
            accepted += 1;
        } else {
            break;
        }
    }

    accepted
}
