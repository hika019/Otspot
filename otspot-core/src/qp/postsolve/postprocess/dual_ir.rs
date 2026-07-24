//! dual-only IR: x 固定で y のみ更新し r_d_free を厳密に 0 にする。
//! A_free^T δy = -r_d_free の最小ノルム解 δy = -A_free α、G α = r_d_free (G SPD)。

use crate::qp::postsolve::dual_recovery::{
    collect_dual_recovery_cluster_rows, collect_dual_recovery_free_columns,
    compute_dual_recovery_row_activity, compute_dual_recovery_row_bounds,
    row_is_active_for_dual_recovery, select_dual_recovery_local_bounds,
    DUAL_RECOVERY_ACTIVE_TOL_REL,
};
use crate::qp::postsolve::refine::kkt_iterative::refit_bound_duals_kkt;
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;
use crate::tolerances::any_nonfinite;

/// G + δ·I regularization: prevents F64 round-off cancellation.
/// δ × ‖α‖ acts as a floor for the new r_d_free (typically 1e-12 × 1e2 = 1e-10, well below target 1e-6).
const DUAL_IR_REG_DEFAULT: f64 = 1e-12;
/// Maximum dense local normal-equation dimension in dual-only IR. The local
/// cluster solve is O(k^3); larger clusters are not a local refinement anymore
/// and should fall back to the outer guarded postsolve path.
const DUAL_IR_MAX_DENSE_DIM: usize = 256;

fn dual_ir_dense_dim_allowed(dim: usize) -> bool {
    dim <= DUAL_IR_MAX_DENSE_DIM
}

pub(crate) fn try_dual_only_ir(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    eliminated_cols: &[bool],
    target_pf: f64,
    deadline: Option<std::time::Instant>,
) -> usize {
    use crate::qp::kkt_resid::bound_contrib;
    use twofloat::TwoFloat;

    let m = problem.num_constraints;
    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
        eliminated_cols,
    };
    let kkt_pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );

    let dual_ir_reg = DUAL_IR_REG_DEFAULT;

    // 1. free 変数の特定 (active = bound 近傍 or A col 空)
    let free_eval_idx = collect_dual_recovery_free_columns(problem, result, eliminated_cols);
    let n_free_eval = free_eval_idx.len();
    if n_free_eval == 0 {
        return 0;
    }

    // 2. r_d_free を DD で計算
    //    r_d[j] = c[j] + (A^T y)[j] + bound_contrib[j]
    //    free var の bound_contrib は通常 0 (z=0) だが念のため計算
    let mut r_d_eval = vec![0.0_f64; n_free_eval];
    let mut r_d_rel_eval = vec![0.0_f64; n_free_eval];
    let mut df_rel_pre = 0.0_f64;
    let bc_eval = bound_contrib(&problem.bounds, &result.bound_duals);
    for (fi, &j) in free_eval_idx.iter().enumerate() {
        // r_d_free 用に Q x も加算する必要 (Q≠0 の QP で正確性必須)
        let mut qx = TwoFloat::from(0.0);
        for k in problem.q.col_ptr[j]..problem.q.col_ptr[j + 1] {
            let row = problem.q.row_ind[k];
            qx += TwoFloat::new_mul(problem.q.values[k], result.solution[row]);
        }
        let qx_f = f64::from(qx);
        let mut aty = TwoFloat::from(0.0);
        for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
            let r = problem.a.row_ind[k];
            aty += TwoFloat::new_mul(problem.a.values[k], result.dual_solution[r]);
        }
        let aty_f = f64::from(aty);
        let bc = bc_eval[j];
        let r_d = qx_f + problem.c[j] + aty_f + bc;
        r_d_eval[fi] = r_d;
        let scale = 1.0 + qx_f.abs() + problem.c[j].abs() + aty_f.abs() + bc.abs();
        let rel = r_d.abs() / scale;
        r_d_rel_eval[fi] = rel;
        if rel > df_rel_pre {
            df_rel_pre = rel;
        }
    }
    if df_rel_pre < target_pf {
        return 0;
    }

    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return 0;
    }

    let row_bounds = match compute_dual_recovery_row_bounds(problem, &result.solution) {
        Some(v) => v,
        None => return 0,
    };
    let (proj_lower, proj_upper) = (&row_bounds.0, &row_bounds.1);

    let (ax, row_abs_activity) = compute_dual_recovery_row_activity(problem, &result.solution);
    let Some((worst_j, active_rows)) = collect_dual_recovery_cluster_rows(
        problem,
        &free_eval_idx,
        &r_d_rel_eval,
        &ax,
        &row_abs_activity,
    ) else {
        return 0;
    };
    let mut seed_rows = Vec::new();
    for k in problem.a.col_ptr[worst_j]..problem.a.col_ptr[worst_j + 1] {
        let row = problem.a.row_ind[k];
        if row_is_active_for_dual_recovery(
            problem,
            row,
            &ax,
            &row_abs_activity,
            DUAL_RECOVERY_ACTIVE_TOL_REL,
        ) {
            seed_rows.push(row);
        }
    }
    seed_rows.sort_unstable();
    seed_rows.dedup();

    let mut active_rows = active_rows;
    let mut active_row_pos = vec![usize::MAX; m];
    for (pos, &row) in active_rows.iter().enumerate() {
        active_row_pos[row] = pos;
    }
    let m_active = active_rows.len();
    if m_active == 0 {
        return 0;
    }

    let mut free_idx = Vec::new();
    for &j in &free_eval_idx {
        let touches_cluster = (problem.a.col_ptr[j]..problem.a.col_ptr[j + 1])
            .any(|k| active_row_pos[problem.a.row_ind[k]] != usize::MAX);
        if touches_cluster {
            free_idx.push(j);
        }
    }
    if !seed_rows.is_empty()
        && free_idx.len() * 2 > n_free_eval
        && active_rows.len() > seed_rows.len()
    {
        active_rows = seed_rows;
        active_row_pos.fill(usize::MAX);
        for (pos, &row) in active_rows.iter().enumerate() {
            active_row_pos[row] = pos;
        }
        free_idx.clear();
        for &j in &free_eval_idx {
            let touches_cluster = (problem.a.col_ptr[j]..problem.a.col_ptr[j + 1])
                .any(|k| active_row_pos[problem.a.row_ind[k]] != usize::MAX);
            if touches_cluster {
                free_idx.push(j);
            }
        }
    }
    let n_free = free_idx.len();
    if n_free == 0 {
        return 0;
    }

    // y/z を [row duals ; active bound duals] 連成で局所 LS。row-only は bound 押し返しで悪化。
    // y は DD 精度で保持 (unscale で y≈1e10 級になると f64 累積では |dy|<2e-6 が切り捨てられる)。
    let mut tmp = result.clone();
    let mut y_dd: Vec<TwoFloat> = tmp
        .dual_solution
        .iter()
        .map(|&v| TwoFloat::from(v))
        .collect();
    let mut df_rel_post = df_rel_pre;
    let mut accepted_iters = 0;
    let mut current_r_d_free: Vec<f64> = free_idx
        .iter()
        .map(|&j| {
            let pos = free_eval_idx
                .iter()
                .position(|&jj| jj == j)
                .expect("free cluster column must exist in eval set");
            r_d_eval[pos]
        })
        .collect();
    const DUAL_IR_ACCEPT_REL_TOL: f64 = 1e-12;
    const DUAL_IR_MIN_PROGRESS_RATIO: f64 = 1e-4;
    loop {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        let mut provisional_residual = vec![0.0_f64; problem.num_vars];
        for (fi, &j) in free_idx.iter().enumerate() {
            provisional_residual[j] = current_r_d_free[fi];
        }
        let (local_bounds, bound_pos_of_var) = select_dual_recovery_local_bounds(
            problem,
            &tmp.solution,
            &tmp.bound_duals,
            &free_idx,
            &provisional_residual,
        );
        let ulen = m_active + local_bounds.len();
        if ulen == 0 {
            break;
        }
        if !dual_ir_dense_dim_allowed(ulen) {
            break;
        }
        let mut gram = vec![0.0_f64; ulen * ulen];
        let mut rhs = vec![0.0_f64; ulen];
        let bc_tmp = bound_contrib(&problem.bounds, &tmp.bound_duals);
        for (fi, &j) in free_idx.iter().enumerate() {
            let residual = current_r_d_free[fi];

            // 1/scale[j]^2 で重み付けし min Σ (r_d[j]/scale[j])² を解く
            // (重み無しの abs LS は componentwise max を悪化させる)。
            let mut qx_j = 0.0_f64;
            for k in problem.q.col_ptr[j]..problem.q.col_ptr[j + 1] {
                qx_j += problem.q.values[k] * tmp.solution[problem.q.row_ind[k]];
            }
            let mut aty_j = 0.0_f64;
            for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
                aty_j += problem.a.values[k] * f64::from(y_dd[problem.a.row_ind[k]]);
            }
            let bc_j = bc_tmp[j];
            let scale_j =
                (1.0 + qx_j.abs() + problem.c[j].abs() + aty_j.abs() + bc_j.abs()).max(1.0);
            let inv_scale2 = 1.0 / (scale_j * scale_j);

            let mut col_vec = vec![0.0_f64; ulen];
            for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
                let r = problem.a.row_ind[k];
                let pos = active_row_pos[r];
                if pos != usize::MAX {
                    col_vec[pos] = problem.a.values[k];
                }
            }
            let bpos = bound_pos_of_var[j];
            if bpos != usize::MAX {
                col_vec[m_active + bpos] = local_bounds[bpos].coeff();
            }
            for i in 0..ulen {
                rhs[i] -= col_vec[i] * residual * inv_scale2;
                for j2 in i..ulen {
                    gram[i * ulen + j2] += col_vec[i] * col_vec[j2] * inv_scale2;
                }
            }
        }
        for i in 0..ulen {
            gram[i * ulen + i] += dual_ir_reg;
        }
        let mut col_ptr: Vec<usize> = vec![0; ulen + 1];
        let mut row_ind: Vec<usize> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for j in 0..ulen {
            for i in 0..=j {
                let v = gram[i * ulen + j];
                if v != 0.0 {
                    row_ind.push(i);
                    values.push(v);
                }
            }
            col_ptr[j + 1] = row_ind.len();
        }
        let gram_csc = CscMatrix {
            col_ptr,
            row_ind,
            values,
            nrows: ulen,
            ncols: ulen,
        };
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        let factor = match crate::linalg::ldl::factorize(&gram_csc) {
            Ok(f) => f,
            Err(_) => break,
        };
        let mut delta = vec![0.0_f64; ulen];
        factor.solve(&rhs, &mut delta);
        if any_nonfinite(&delta) {
            break;
        }
        let mut dy_dd = vec![TwoFloat::from(0.0); m];
        for (pos, &row) in active_rows.iter().enumerate() {
            dy_dd[row] = TwoFloat::from(delta[pos]);
        }
        let dy_inf = dy_dd
            .iter()
            .fold(0.0_f64, |a, v| a.max(f64::from(*v).abs()));
        if !dy_inf.is_finite() {
            break;
        }

        let mut accepted = false;
        let mut accepted_df_rel = df_rel_post;
        let mut accepted_r_d_free = current_r_d_free.clone();
        let mut accepted_y_dd = y_dd.clone();
        let mut accepted_bound_duals = tmp.bound_duals.clone();
        let mut step_scale = 1.0_f64;
        while step_scale > 0.0 {
            let mut y_dd_new: Vec<TwoFloat> = y_dd
                .iter()
                .zip(dy_dd.iter())
                .map(|(&y, &d)| y + d * step_scale)
                .collect();
            let mut bound_duals_new = tmp.bound_duals.clone();
            // dy_dd は active_rows のみ更新する。非アクティブ行の y_dd_new は y_dd と同値のため
            // クランプ不要。全行クランプすると非アクティブ行の y が 0 に強制され、
            // df_rel_pre (非クランプ y で計算) との比較が不整合になり、正当なステップが棄却される。
            for &row in &active_rows {
                let val = f64::from(y_dd_new[row]);
                let lo = proj_lower[row];
                let hi = proj_upper[row];
                let clamped = if lo <= hi { val.clamp(lo, hi) } else { val };
                y_dd_new[row] = TwoFloat::from(clamped);
            }
            for (pos, &bound) in local_bounds.iter().enumerate() {
                let slot = bound.slot();
                if slot >= bound_duals_new.len() {
                    continue;
                }
                let z = tmp.bound_duals[slot] + step_scale * delta[m_active + pos];
                bound_duals_new[slot] = z.max(0.0);
            }

            // 新 r_d_free を y_dd_new から DD 精度で計算 (Q x は変化なし、aty のみ更新)
            let mut new_r_d_free = vec![0.0_f64; n_free];
            let mut new_df_rel = 0.0_f64;
            let bc_new = bound_contrib(&problem.bounds, &bound_duals_new);
            for &j in &free_eval_idx {
                let mut qx = TwoFloat::from(0.0);
                for k in problem.q.col_ptr[j]..problem.q.col_ptr[j + 1] {
                    let row = problem.q.row_ind[k];
                    qx += TwoFloat::new_mul(problem.q.values[k], tmp.solution[row]);
                }
                let mut aty = TwoFloat::from(0.0);
                for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
                    let r = problem.a.row_ind[k];
                    aty += y_dd_new[r] * problem.a.values[k];
                }
                let bc = bc_new[j];
                let r_d = f64::from(qx + TwoFloat::from(problem.c[j]) + aty + TwoFloat::from(bc));
                if let Some(local_pos) = free_idx.iter().position(|&jj| jj == j) {
                    new_r_d_free[local_pos] = r_d;
                }
                let qx_f = f64::from(qx);
                let aty_f = f64::from(aty);
                let scale = 1.0 + qx_f.abs() + problem.c[j].abs() + aty_f.abs() + bc.abs();
                let rel = r_d.abs() / scale;
                if rel > new_df_rel {
                    new_df_rel = rel;
                }
            }
            if new_df_rel <= df_rel_post + DUAL_IR_ACCEPT_REL_TOL * (1.0 + df_rel_post) {
                accepted = true;
                accepted_df_rel = new_df_rel;
                accepted_r_d_free = new_r_d_free;
                accepted_y_dd = y_dd_new;
                accepted_bound_duals = bound_duals_new;
                break;
            }
            let next_step_scale = step_scale * 0.5;
            if next_step_scale == step_scale {
                break;
            }
            step_scale = next_step_scale;
        }
        if !accepted {
            break;
        }

        let rel_improvement = (df_rel_post - accepted_df_rel).max(0.0);
        let progress_ratio = if df_rel_post > 0.0 {
            rel_improvement / df_rel_post
        } else {
            0.0
        };
        if accepted_iters > 0 && progress_ratio <= DUAL_IR_MIN_PROGRESS_RATIO {
            break;
        }

        y_dd = accepted_y_dd;
        for i in 0..m {
            tmp.dual_solution[i] = f64::from(y_dd[i]);
        }
        tmp.bound_duals = accepted_bound_duals;
        current_r_d_free = accepted_r_d_free;
        df_rel_post = accepted_df_rel;
        accepted_iters += 1;
        // 早期 break: target を達成したら終了
        if df_rel_post < target_pf {
            break;
        }
    }
    for i in 0..m {
        tmp.dual_solution[i] = f64::from(y_dd[i]);
    }
    // 採用判定前に z を取り直す (y-only 更新を stale な z で評価すると改善候補を落とす)。
    refit_bound_duals_kkt(problem, &mut tmp, target_pf);

    let kkt_post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &tmp.solution,
        &tmp.dual_solution,
        &tmp.bound_duals,
    );
    if df_rel_post < df_rel_pre && kkt_post <= kkt_pre {
        *result = tmp;
        accepted_iters
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::{dual_ir_dense_dim_allowed, DUAL_IR_MAX_DENSE_DIM};

    #[test]
    fn dense_cluster_cap_rejects_nonlocal_dual_ir_blocks() {
        assert!(dual_ir_dense_dim_allowed(DUAL_IR_MAX_DENSE_DIM));
        assert!(
            !dual_ir_dense_dim_allowed(DUAL_IR_MAX_DENSE_DIM + 1),
            "dual-only IR must not build unbounded dense local normal equations"
        );
    }
}
