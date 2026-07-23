//! worst residual 列に接続する active cluster を局所的に再最適化。
//! [active row duals ; active bound duals] 連成で解く (row dual 単独では bound 押し返しで悪化)。

use crate::qp::kkt_resid;
use crate::qp::postsolve::dual_recovery::{
    compute_dual_recovery_row_activity, compute_dual_recovery_row_bounds,
    row_is_active_for_dual_recovery, select_dual_recovery_local_bounds,
    DUAL_RECOVERY_ACTIVE_TOL_REL,
};
use crate::qp::problem::QpProblem;
use crate::qp::FX_TOL;
use crate::sparse::CscMatrix;
use crate::tolerances::any_nonfinite;

/// Maximum dense local normal-equation dimension for worst-active correction.
///
/// This stage is a local repair pass. Its cost is cubic in the number of active
/// row/bound variables in the selected cluster, so clusters above this bound are
/// not "local" in the algorithmic sense and must not monopolize postsolve until
/// the global deadline.
const WORST_ACTIVE_MAX_DENSE_DIM: usize = 256;

fn worst_active_dense_dim_allowed(dim: usize) -> bool {
    dim <= WORST_ACTIVE_MAX_DENSE_DIM
}

pub(crate) fn refine_dual_worst_active_block(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    eliminated_cols: &[bool],
    deadline: Option<std::time::Instant>,
) {
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if result.solution.len() != n || result.dual_solution.len() != m {
        return;
    }

    let qx = problem.q.mat_vec_mul(&result.solution).expect(
        "q.ncols() == solution.len() == num_vars: QpProblem::new() enforces \
         q.ncols() == num_vars, and solution.len() == n is checked above",
    );
    let aty = if problem.a.nrows > 0 {
        problem
            .a
            .transpose()
            .mat_vec_mul(&result.dual_solution)
            .expect(
                "a.transpose().ncols() == a.nrows() == num_constraints == dual_solution.len(): \
             QpProblem::new() enforces a.nrows() == num_constraints, and \
             dual_solution.len() == m is checked above",
            )
    } else {
        vec![0.0_f64; n]
    };
    let (ax, row_abs_activity) = compute_dual_recovery_row_activity(problem, &result.solution);
    let bound_contrib = kkt_resid::bound_contrib(&problem.bounds, &result.bound_duals);

    let use_elim_mask = eliminated_cols.len() == n;
    let mut worst_j = None;
    let mut worst_rel = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = problem.bounds[j];
        let is_fx = lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL;
        let is_eliminated = use_elim_mask && eliminated_cols[j];
        if is_fx || is_eliminated {
            continue;
        }
        let r = qx[j] + problem.c[j] + aty[j] + bound_contrib[j];
        let scale = 1.0 + qx[j].abs() + problem.c[j].abs() + aty[j].abs() + bound_contrib[j].abs();
        let rel = r.abs() / scale;
        if rel > worst_rel {
            worst_rel = rel;
            worst_j = Some(j);
        }
    }
    let Some(worst_j) = worst_j else {
        return;
    };
    let mut rows = Vec::new();
    for k in problem.a.col_ptr[worst_j]..problem.a.col_ptr[worst_j + 1] {
        let row = problem.a.row_ind[k];
        if row_is_active_for_dual_recovery(
            problem,
            row,
            &ax,
            &row_abs_activity,
            DUAL_RECOVERY_ACTIVE_TOL_REL,
        ) {
            rows.push(row);
        }
    }
    if rows.is_empty() {
        return;
    }
    rows.sort_unstable();
    rows.dedup();
    let rlen = rows.len();
    if !worst_active_dense_dim_allowed(rlen) {
        return;
    }

    let mut row_pos = vec![usize::MAX; m];
    for (pos, &row) in rows.iter().enumerate() {
        row_pos[row] = pos;
    }

    let mut row_only_gram = vec![0.0_f64; rlen * rlen];
    let mut row_only_rhs = vec![0.0_f64; rlen];
    let mut current_local_residual = vec![0.0_f64; n];
    for col in 0..n {
        let residual = qx[col] + problem.c[col] + aty[col] + bound_contrib[col];
        current_local_residual[col] = residual;
        let mut col_vec = vec![0.0_f64; rlen];
        let mut touches = false;
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let row = problem.a.row_ind[k];
            let pos = row_pos[row];
            if pos != usize::MAX {
                col_vec[pos] = problem.a.values[k];
                touches = true;
            }
        }
        if !touches {
            continue;
        }
        for i in 0..rlen {
            row_only_rhs[i] -= col_vec[i] * residual;
            for j in i..rlen {
                row_only_gram[i * rlen + j] += col_vec[i] * col_vec[j];
            }
        }
    }
    let row_only_sol = {
        let row_diag_max = (0..rlen)
            .map(|i| row_only_gram[i * rlen + i].abs())
            .fold(0.0_f64, f64::max);
        let row_reg = f64::EPSILON * (1.0 + row_diag_max);
        let mut row_col_ptr = vec![0usize; rlen + 1];
        let mut row_ind = Vec::new();
        let mut row_values = Vec::new();
        for j in 0..rlen {
            for i in 0..=j {
                let mut v = row_only_gram[i * rlen + j];
                if i == j {
                    v += row_reg;
                }
                if v != 0.0 {
                    row_ind.push(i);
                    row_values.push(v);
                }
            }
            row_col_ptr[j + 1] = row_ind.len();
        }
        let row_csc = CscMatrix {
            col_ptr: row_col_ptr,
            row_ind,
            values: row_values,
            nrows: rlen,
            ncols: rlen,
        };
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return;
        }
        crate::linalg::ldl::factorize(&row_csc)
            .ok()
            .map(|factor| {
                let mut sol = vec![0.0_f64; rlen];
                factor.solve(&row_only_rhs, &mut sol);
                sol
            })
            .filter(|sol| sol.iter().all(|v| v.is_finite()))
    };
    let mut provisional_residual = current_local_residual.clone();
    if let Some(ref delta_row) = row_only_sol {
        for col in 0..n {
            let mut delta = 0.0_f64;
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                let row = problem.a.row_ind[k];
                let pos = row_pos[row];
                if pos != usize::MAX {
                    delta += problem.a.values[k] * delta_row[pos];
                }
            }
            provisional_residual[col] += delta;
        }
    }

    let mut cols = Vec::new();
    for col in 0..n {
        let mut touches = false;
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            if row_pos[problem.a.row_ind[k]] != usize::MAX {
                touches = true;
                break;
            }
        }
        if touches {
            cols.push(col);
        }
    }
    if cols.is_empty() {
        return;
    }

    let (local_bounds, bound_pos_of_var) = select_dual_recovery_local_bounds(
        problem,
        &result.solution,
        &result.bound_duals,
        &cols,
        &provisional_residual,
    );

    let ulen = rlen + local_bounds.len();
    if ulen == 0 {
        return;
    }
    if !worst_active_dense_dim_allowed(ulen) {
        return;
    }
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let mut gram = vec![0.0_f64; ulen * ulen];
    let mut rhs = vec![0.0_f64; ulen];
    let mut local_aty = vec![0.0_f64; cols.len()];
    let mut local_bound_contrib = vec![0.0_f64; cols.len()];
    for (ci, &col) in cols.iter().enumerate() {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let row = problem.a.row_ind[k];
            let pos = row_pos[row];
            if pos != usize::MAX {
                local_aty[ci] += problem.a.values[k] * result.dual_solution[row];
            }
        }
        let bpos = bound_pos_of_var[col];
        if bpos != usize::MAX {
            let bound = local_bounds[bpos];
            if let Some(&z) = result.bound_duals.get(bound.slot()) {
                local_bound_contrib[ci] += bound.coeff() * z;
            }
        }
    }

    for &col in &cols {
        let residual = qx[col] + problem.c[col] + aty[col] + bound_contrib[col];
        let mut col_vec = vec![0.0_f64; ulen];
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let row = problem.a.row_ind[k];
            let pos = row_pos[row];
            if pos != usize::MAX {
                col_vec[pos] = problem.a.values[k];
            }
        }
        let bpos = bound_pos_of_var[col];
        if bpos != usize::MAX {
            col_vec[rlen + bpos] = local_bounds[bpos].coeff();
        }
        for i in 0..ulen {
            rhs[i] -= col_vec[i] * residual;
            for j in i..ulen {
                gram[i * ulen + j] += col_vec[i] * col_vec[j];
            }
        }
    }

    let diag_max = (0..ulen)
        .map(|i| gram[i * ulen + i].abs())
        .fold(0.0_f64, f64::max);
    let reg = f64::EPSILON * (1.0 + diag_max);
    let mut col_ptr = vec![0usize; ulen + 1];
    let mut row_ind = Vec::new();
    let mut values = Vec::new();
    for j in 0..ulen {
        for i in 0..=j {
            let mut v = gram[i * ulen + j];
            if i == j {
                v += reg;
            }
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
        return;
    }
    let Ok(factor) = crate::linalg::ldl::factorize(&gram_csc) else {
        return;
    };
    let mut block_sol = vec![0.0_f64; ulen];
    factor.solve(&rhs, &mut block_sol);
    if any_nonfinite(&block_sol) {
        return;
    }

    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
        eliminated_cols,
    };
    let pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    let Some((row_lower, row_upper)) = compute_dual_recovery_row_bounds(problem, &result.solution)
    else {
        return;
    };
    let mut best = result.clone();
    let mut best_kkt = pre;
    let mut step = 1.0_f64;
    while step > 0.0 {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return;
        }
        let mut tmp = result.clone();
        for (pos, &row) in rows.iter().enumerate() {
            let mut v = result.dual_solution[row] + step * block_sol[pos];
            let lo = row_lower[row];
            let hi = row_upper[row];
            if lo <= hi {
                v = v.clamp(lo, hi);
            }
            tmp.dual_solution[row] = v;
        }
        for (pos, &bound) in local_bounds.iter().enumerate() {
            let slot = bound.slot();
            if slot >= tmp.bound_duals.len() {
                continue;
            }
            let z = result.bound_duals[slot] + step * block_sol[rlen + pos];
            tmp.bound_duals[slot] = z.max(0.0);
        }
        let post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &tmp.solution,
            &tmp.dual_solution,
            &tmp.bound_duals,
        );
        if post < best_kkt {
            best = tmp;
            best_kkt = post;
            break;
        }
        let next_step = step * 0.5;
        if next_step == step {
            break;
        }
        step = next_step;
    }
    if best_kkt < pre {
        result.dual_solution = best.dual_solution;
        result.bound_duals = best.bound_duals;
    }
}

#[cfg(test)]
mod tests {
    use super::{worst_active_dense_dim_allowed, WORST_ACTIVE_MAX_DENSE_DIM};

    #[test]
    fn dense_cluster_cap_rejects_nonlocal_worst_active_blocks() {
        assert!(worst_active_dense_dim_allowed(WORST_ACTIVE_MAX_DENSE_DIM));
        assert!(
            !worst_active_dense_dim_allowed(WORST_ACTIVE_MAX_DENSE_DIM + 1),
            "worst-active local correction must not build an unbounded dense Gram"
        );
    }
}
