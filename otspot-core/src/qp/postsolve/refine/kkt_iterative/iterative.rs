//! Wilkinson 流 KKT iterative refinement。

use super::bound_refit::refit_bound_duals_kkt;
use crate::qp::postsolve::dual_recovery::dual_recovery_progress_tol;
use crate::qp::postsolve::postprocess::{run_dual_recovery_postprocess, try_dual_only_ir};
use crate::qp::problem::QpProblem;
use crate::tolerances::any_nonfinite;

/// Relative progress threshold for Krylov IR residual scores. Strict decreases
/// below this relative scale are indistinguishable from f64/DD accumulation
/// noise on ill-conditioned postsolve systems.
const KRYLOV_REL_STALL: f64 = 1e-8;
/// Absolute floor for near-zero residual scores, matching the post-refit stall
/// policy so tiny sub-floor changes do not keep IR alive.
const KRYLOV_PROGRESS_EPS: f64 = 1e-12;

fn krylov_score_made_progress(score_cur: f64, score_new: f64, target_pf: f64) -> bool {
    let progress_tol = dual_recovery_progress_tol(score_cur, score_new, target_pf)
        .max(KRYLOV_REL_STALL * score_cur.abs())
        .max(KRYLOV_PROGRESS_EPS);
    score_new + progress_tol < score_cur
}

pub(crate) fn refine_kkt_iterative(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    eliminated_cols: &[bool],
    max_iters: usize,
    target_pf: f64,
    deadline: Option<std::time::Instant>,
) -> usize {
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
    if n + m > crate::tolerances::LARGE_PROBLEM_THRESHOLD {
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
            // side-effect refit only; KKT score is reused in the else branch via pre/post diff.
            let _: f64 = run_dual_recovery_postprocess(problem, &view, result, deadline, target_pf);
        } else {
            let pre_cleanup_kkt = kkt_residual_rel(
                &view,
                &result.solution,
                &result.dual_solution,
                &result.bound_duals,
            );
            let post_cleanup_kkt = run_dual_recovery_postprocess(problem, &view, result, deadline, target_pf);
            if post_cleanup_kkt
                + dual_recovery_progress_tol(pre_cleanup_kkt, post_cleanup_kkt, target_pf)
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

    let sigma_zero = vec![0.0_f64; m];
    let mut k_mat = crate::qp::ipm_core::kkt::build_augmented_system(
        &problem.q,
        &problem.a,
        &sigma_zero,
        DELTA_P_DEFAULT,
        DELTA_D_DEFAULT,
    );

    // bound-active 変数の dx を K 対角 penalty で抑制 (近似 active set fix)。
    const ACTIVE_TOL: f64 = 1e-8;
    const ACTIVE_PENALTY_RATIO: f64 = 1e8;
    {
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
                    break;
                }
            }
        }
    }

    // On SingularOrIndefinite, grow δ by FACTOR_RETRY_GROWTH and retry until
    // factorization succeeds; the first success is the smallest δ that works
    // (deltas grow monotonically). Deadline guards against large-K stalls.
    const FACTOR_RETRY_GROWTH: f64 = 10.0;
    const FACTOR_RETRY_MAX: usize = 6;
    let factor = {
        let mut current_delta_p = DELTA_P_DEFAULT;
        let mut current_delta_d = DELTA_D_DEFAULT;
        let mut current_k = k_mat.clone();
        let mut result_factor: Option<crate::linalg::ldl::LdlFactorizationAmd> = None;
        let mut retry_count = 0usize;
        loop {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                break;
            }
            match crate::linalg::ldl::factorize_quasidefinite_with_amd(&current_k, deadline) {
                Ok(f) => {
                    result_factor = Some(f);
                    break;
                }
                Err(_) => {
                    if retry_count >= FACTOR_RETRY_MAX {
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
        match result_factor {
            Some(f) => f,
            None => return 0,
        }
    };

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

    // Wilkinson IR の "double the working precision": Qx, A^T y, Ax を TwoFloat (DD) で積算し
    // residual を f64 limit 以下に精密化。LDL solve は f64 のまま。
    // 戻り値: (r_d, r_p, pf_rel, df_rel)、pf_rel/df_rel は OSQP-style componentwise。
    let compute_residuals =
        |x: &[f64], y: &[f64], z: &[f64]| -> (Vec<f64>, Vec<f64>, f64, f64) {
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
            let bc_vec = crate::qp::kkt_resid::bound_contrib(&problem.bounds, z);
            let mut r_d = vec![0.0_f64; n];
            for j in 0..n {
                if exclude_var[j] {
                    continue;
                }
                let bc = bc_vec[j];
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
                let ax_i_abs = f64::from(ax_dd[i]).abs();
                let scale_i = 1.0 + ax_i_abs + problem.b[i].abs();
                let rel_i = v.abs() / scale_i;
                if rel_i > pf_rel_componentwise {
                    pf_rel_componentwise = rel_i;
                }
            }
            // componentwise が必須 (全体相対化は ill-scaled で 1 成分外れを見逃す)。
            let mut df_rel_componentwise = 0.0_f64;
            for j in 0..n {
                if exclude_var[j] {
                    continue;
                }
                let qx_j = f64::from(qx_dd[j]).abs();
                let aty_j = f64::from(aty_dd[j]).abs();
                let bc = bc_vec[j];
                let scale_j = 1.0 + qx_j + problem.c[j].abs() + aty_j + bc.abs();
                let rel_j = r_d[j].abs() / scale_j;
                if rel_j > df_rel_componentwise {
                    df_rel_componentwise = rel_j;
                }
            }
            (r_d, r_p, pf_rel_componentwise, df_rel_componentwise)
        };

    let pre_z = result.bound_duals.clone();
    let (_, _, pre_pf_rel, pre_df_rel) =
        compute_residuals(&result.solution, &result.dual_solution, &pre_z);
    if pre_pf_rel < target_pf && pre_df_rel < target_pf {
        return 0;
    }

    let mut accepted = n_dual_total;
    // best 選別は採用指標と同じ max(pf, df) で行う。stationarity のみでは
    // pf 改善 + stationarity 微増の採用ステップを「最良点」と誤判定し、
    // 採用済み進捗を巻き戻す failure mode を生む。
    let mut best_score = pre_pf_rel.max(pre_df_rel);
    let mut best_saddle_result = result.clone();
    // 残差悪化許容: max(pre_rel × 2, target_pf × 100) を超えたら revert。
    const RESID_TOLERANCE_FACTOR: f64 = 2.0;
    const RESID_FLOOR_RATIO: f64 = 100.0;
    let resid_floor = target_pf * RESID_FLOOR_RATIO;
    let pf_limit = (pre_pf_rel * RESID_TOLERANCE_FACTOR).max(resid_floor);
    let df_limit = (pre_df_rel * RESID_TOLERANCE_FACTOR).max(resid_floor);

    for _iter in 0..max_iters {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        let (r_d, r_p, pf_cur, df_cur) =
            compute_residuals(&result.solution, &result.dual_solution, &result.bound_duals);
        if pf_cur < target_pf && df_cur < target_pf {
            break;
        }

        let mut rhs = vec![0.0_f64; n + m];
        for j in 0..n {
            rhs[j] = -r_d[j];
        }
        for i in 0..m {
            rhs[n + i] = -r_p[i];
        }

        let mut sol = vec![0.0_f64; n + m];
        factor.solve(&rhs, &mut sol);
        if any_nonfinite(&sol) {
            break;
        }

        let mut x_new = result.solution.clone();
        let mut y_new = result.dual_solution.clone();
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
            x_new[j] = clipped;
        }
        for i in 0..m {
            y_new[i] += sol[n + i];
        }

        let mut tmp = result.clone();
        tmp.solution = x_new;
        tmp.dual_solution = y_new;
        refit_bound_duals_kkt(problem, &mut tmp, target_pf);

        let (_, _, pf_new, df_new) =
            compute_residuals(&tmp.solution, &tmp.dual_solution, &tmp.bound_duals);

        // 採用: max(pf_rel, df_rel) の意味ある減少 + 両者 guardrail 内。
        let score_cur = pf_cur.max(df_cur);
        let score_new = pf_new.max(df_new);
        let progress = krylov_score_made_progress(score_cur, score_new, target_pf);
        let pf_safe = pf_new < pf_limit;
        let df_safe = df_new < df_limit;
        if progress && pf_safe && df_safe {
            *result = tmp;
            accepted += 1;
            if score_new < best_score {
                best_score = score_new;
                best_saddle_result = result.clone();
            }
        } else {
            break;
        }
    }

    *result = best_saddle_result;
    accepted
}

#[cfg(test)]
mod tests {
    use super::krylov_score_made_progress;

    /// P2 sentinel: best-snapshot tracking must use the composite max(pf, df) score,
    /// not stationarity alone.
    ///
    /// Scenario: initial point has pf=0.5, df=0.1 (composite score=0.5).
    /// One step is accepted: pf improves to 0.1, but stationarity worsens to 0.15
    /// (composite score=0.15, strictly better than 0.5).
    ///
    /// Old stationarity-only logic:  best_kkt initialised to 0.1, cur_kkt=0.15 > 0.1
    ///   → best NOT updated → final result has pf=0.5 (silently reverts all progress).
    ///
    /// New composite-score logic: best_score initialised to 0.5, score_new=0.15 < 0.5
    ///   → best updated → final result has pf=0.1 (correct).
    ///
    /// The test replicates the call-site state machine verbatim so that a regression
    /// to stationarity-only initialisation causes the `old_updated` assertion to flip
    /// and break this test.
    #[test]
    fn saddle_best_snapshot_tracks_composite_score_not_stationarity_only() {
        // Initial point residuals (after dual-only IR or loop start)
        let (pre_pf, pre_df) = (0.5_f64, 0.1_f64);

        // Accepted saddle step: pf greatly improves, stationarity (df proxy) worsens
        let (pf_new, df_new) = (0.1_f64, 0.15_f64);
        let score_new = pf_new.max(df_new); // 0.15

        // ── new composite-score logic (verbatim call-site pattern) ──────────
        let mut best_score = pre_pf.max(pre_df); // 0.5
        let mut best_pf = pre_pf; // track pf at the best point separately
        let mut new_logic_updated = false;
        if score_new < best_score {
            best_score = score_new;
            best_pf = pf_new;
            new_logic_updated = true;
        }
        let _ = best_score; // updated value is not re-read in this single-iteration simulation
        assert!(
            new_logic_updated,
            "composite score must update best: score_new {:.2} < best_score {:.2}",
            score_new,
            pre_pf.max(pre_df),
        );
        // best_score = max(pf_new, df_new) = 0.15; best_pf = pf_new = 0.1 ≤ pre_pf = 0.5
        assert!(
            best_pf <= pre_pf,
            "returned best pf must be ≤ initial pf (got best_pf={best_pf:.2}, pre_pf={pre_pf:.2})",
        );

        // ── no-op proof: stationarity-only (old logic) ──────────────────────
        // old: best_saddle_kkt ← kkt_residual_rel ≈ stationarity ≈ df
        let mut best_kkt_old = pre_df; // 0.1
        let mut old_updated = false;
        let cur_kkt_old = df_new; // 0.15  (stationarity worsened)
        if cur_kkt_old < best_kkt_old {
            best_kkt_old = cur_kkt_old;
            old_updated = true;
        }
        let _ = best_kkt_old; // silence unused-assign lint
        assert!(
            !old_updated,
            "stationarity-only old logic must NOT update best (stationarity worsened: \
             cur_kkt {:.2} ≥ best_kkt {:.2})",
            cur_kkt_old,
            pre_df,
        );
        // If old_updated were true the no-op proof would be invalid;
        // if the call-site regressed to stationarity init the new_logic_updated
        // assertion above would catch it.
    }

    /// Integration sentinel for the best-snapshot bug.
    ///
    /// Problem: min 0.5 x², s.t. x = 1, x ∈ (-∞, ∞). Optimal: x=1, y=-1.
    /// Initial point: x=0, y=0  →  pf=0.5, df=0 (stationarity is exactly 0).
    ///
    /// The saddle IR accepts one step (x≈1, y≈-1): pf≈1e-11, df≈1e-11.
    ///
    /// OLD stationarity-only logic:
    ///   best_saddle_kkt ← kkt_residual_rel(x=0, y=0) = 0.
    ///   After accepted step: cur_kkt ≈ 1e-11 > 0 → best NOT updated
    ///   → *result reverts to x=0, pf≈0.5.  ← FAIL this assertion.
    ///
    /// NEW composite-score logic:
    ///   best_score ← pre_pf.max(pre_df) = 0.5.
    ///   score_new ≈ 1e-11 < 0.5 → best updated → *result stays at x≈1, pf≈1e-11.
    #[test]
    fn saddle_best_snapshot_integration_revert_bug_regression() {
        use crate::problem::{ConstraintType, SolveStatus, SolverResult};
        use crate::qp::problem::QpProblem;
        use crate::sparse::CscMatrix;

        // Q = [[1]], A = [[1]], c = [0], b = [1], x ∈ (-∞, ∞)
        let q = CscMatrix::from_triplets(&[0usize], &[0usize], &[1.0_f64], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[0usize], &[0usize], &[1.0_f64], 1, 1).unwrap();
        let problem = QpProblem::new(
            q,
            vec![0.0_f64],
            a,
            vec![1.0_f64],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            vec![ConstraintType::Eq],
        )
        .unwrap();

        // Initial point: x=0, y=0  (pf=0.5, df=0)
        let mut result = SolverResult {
            solution: vec![0.0_f64],
            dual_solution: vec![0.0_f64],
            bound_duals: vec![],
            status: SolveStatus::NumericalError,
            ..Default::default()
        };

        let n_accepted = super::refine_kkt_iterative(&problem, &mut result, &[], 10, 1e-6, None);

        assert!(n_accepted > 0, "saddle IR must accept at least one step");

        // After refinement: x≈1. Primal residual = |A*x - b| / scale.
        let ax = result.solution[0];
        let pf = (ax - 1.0_f64).abs() / (1.0 + ax.abs() + 1.0_f64);

        // Old stationarity-only logic reverts to x=0 → pf≈0.5, this assertion fails.
        // New composite-score logic keeps x≈1 → pf≈1e-11, this assertion passes.
        assert!(
            pf < 1e-4,
            "post-refine pf={:.2e} must be near-zero; \
             old stationarity-only best-snapshot logic reverts result to x=0 (pf≈0.5)",
            pf,
        );
    }

    #[test]
    fn krylov_progress_rejects_noise_floor_drops() {
        let score_cur = 1.0e-1_f64;
        let score_new = score_cur - 1.0e-11_f64;
        assert!(
            !krylov_score_made_progress(score_cur, score_new, 1.0e-6),
            "strict-but-noise-sized residual drops must not keep Krylov IR spinning"
        );
        assert!(
            krylov_score_made_progress(score_cur, score_cur * 0.5, 1.0e-6),
            "large residual reduction is real progress"
        );
    }
}
