//! CSC accessors, Kahan-compensated accumulators, and the standalone
//! transforms that do not need the per-pass workspace (early infeasibility,
//! block-structure detection, large-coefficient rescaling).

use super::state::Workspace;
use crate::qp::QpProblem;
use crate::sparse::CscMatrix;
use crate::tolerances::ZERO_TOL;
use super::state::QpPresolveStatus;

pub(super) fn q_diagonal(q: &CscMatrix, j: usize) -> f64 {
    let start = q.col_ptr[j];
    let end = q.col_ptr[j + 1];
    for k in start..end {
        if q.row_ind[k] == j {
            return q.values[k];
        }
    }
    0.0
}

/// Kahan-compensated `*sum += delta` to keep presolve-induced rounding noise
/// well below tight user-eps targets.
#[inline]
pub(super) fn kahan_add(sum: &mut f64, comp: &mut f64, delta: f64) {
    let y = delta - *comp;
    let t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}

/// Fix variable `j` to `val` and update `c`, `obj_offset`, `b` in place (Kahan-compensated).
/// The caller must mark `removed_cols[j] = true` and push the postsolve step.
pub(super) fn apply_fixed_variable(j: usize, val: f64, prob: &QpProblem, ws: &mut Workspace) {
    let n = prob.num_vars;
    let m = prob.num_constraints;

    let q_jj = q_diagonal(&prob.q, j);
    kahan_add(&mut ws.obj_offset, &mut ws.obj_offset_comp, 0.5 * q_jj * val * val);
    kahan_add(&mut ws.obj_offset, &mut ws.obj_offset_comp, ws.c[j] * val);

    // c[k] += Q[k,j]·val for k ≠ j (symmetric Q stored in full).
    let start = prob.q.col_ptr[j];
    let end = prob.q.col_ptr[j + 1];
    for idx in start..end {
        let k = prob.q.row_ind[idx];
        if k != j && k < n && !ws.removed_cols[k] {
            let delta = prob.q.values[idx] * val;
            kahan_add(&mut ws.c[k], &mut ws.c_comp[k], delta);
        }
    }

    // b[i] -= A[i,j]·val on every active row.
    let col_start = prob.a.col_ptr[j];
    let col_end = prob.a.col_ptr[j + 1];
    for idx in col_start..col_end {
        let row = prob.a.row_ind[idx];
        if row < m && !ws.removed_rows[row] {
            let delta = -prob.a.values[idx] * val;
            kahan_add(&mut ws.b[row], &mut ws.b_comp[row], delta);
        }
    }
}

/// Early infeasibility / unboundedness checks: inverted bounds, or a fully
/// unconstrained problem with negative-definite Q.
pub(super) fn early_infeasibility_check(prob: &QpProblem) -> Option<QpPresolveStatus> {
    for &(lb, ub) in &prob.bounds {
        if lb > ub + ZERO_TOL {
            return Some(QpPresolveStatus::Infeasible);
        }
    }

    if prob.num_constraints == 0
        && prob.bounds.iter().all(|&(lb, ub)| lb.is_infinite() && ub.is_infinite())
    {
        let all_q_diag_neg = (0..prob.num_vars).all(|j| q_diagonal(&prob.q, j) < -ZERO_TOL);
        if all_q_diag_neg && prob.num_vars > 0 {
            return Some(QpPresolveStatus::Unbounded);
        }
    }

    None
}

/// Count connected variable blocks via Union-Find over the Q+A nonzero pattern.
pub(super) fn count_block_components(q: &CscMatrix, a: &CscMatrix, n: usize) -> usize {
    if n == 0 {
        return 0;
    }

    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut Vec<usize>, x: usize, y: usize) {
        let rx = find(parent, x);
        let ry = find(parent, y);
        if rx != ry {
            parent[rx] = ry;
        }
    }

    for j in 0..n {
        let start = q.col_ptr[j];
        let end = q.col_ptr[j + 1];
        for k in start..end {
            let row = q.row_ind[k];
            if row < n && row != j && q.values[k].abs() > ZERO_TOL {
                union(&mut parent, j, row);
            }
        }
    }

    let m = a.nrows;
    let mut row_vars: Vec<Vec<usize>> = vec![vec![]; m];
    for j in 0..n.min(a.ncols) {
        let start = a.col_ptr[j];
        let end = a.col_ptr[j + 1];
        for k in start..end {
            let row = a.row_ind[k];
            if row < m && a.values[k].abs() > ZERO_TOL {
                row_vars[row].push(j);
            }
        }
    }
    for vars in &row_vars {
        if vars.len() >= 2 {
            let first = vars[0];
            for &v in &vars[1..] {
                union(&mut parent, first, v);
            }
        }
    }

    let mut roots = std::collections::HashSet::new();
    for j in 0..n {
        roots.insert(find(&mut parent, j));
    }
    roots.len()
}

/// True when every Q off-diagonal entry is below 1e-10 in magnitude.
pub(super) fn is_diagonal_q(q: &CscMatrix, n: usize) -> bool {
    for j in 0..n {
        let start = q.col_ptr[j];
        let end = q.col_ptr[j + 1];
        for k in start..end {
            let row = q.row_ind[k];
            if row != j && q.values[k].abs() > 1e-10 {
                return false;
            }
        }
    }
    true
}

/// If A contains entries `|a_ij| > 1e6`, scale each affected row by
/// `σ_i = 1/√(max|A[i,*]|)` (capped at `SIGMA_FLOOR`) so subsequent Ruiz / IPM is
/// well-conditioned. Returns the per-row scales for dual unscaling.
pub(super) fn apply_large_coeff_rescaling(a: &mut CscMatrix, b: &mut [f64], n: usize) -> Vec<f64> {
    let m = a.nrows;
    let has_large = a.values.iter().chain(std::iter::empty()).any(|&v| v.abs() > 1e6);
    if !has_large {
        return vec![1.0; m];
    }

    let mut row_max = vec![0.0f64; m];
    for col in 0..n.min(a.ncols) {
        let start = a.col_ptr[col];
        let end = a.col_ptr[col + 1];
        for k in start..end {
            let row = a.row_ind[k];
            let v = a.values[k].abs();
            if v > row_max[row] {
                row_max[row] = v;
            }
        }
    }

    // Cap per-row amplification at 1/SIGMA_FLOOR so the composite scaling
    // (phase1 · phase2 · Ruiz) stays within the IPM's achievable scaled accuracy.
    const SIGMA_FLOOR: f64 = 1e-3;
    let row_scales: Vec<f64> = row_max
        .iter()
        .map(|&mx| if mx > 1.0 { (1.0 / mx.sqrt()).max(SIGMA_FLOOR) } else { 1.0 })
        .collect();

    for col in 0..n.min(a.ncols) {
        let start = a.col_ptr[col];
        let end = a.col_ptr[col + 1];
        for k in start..end {
            let row = a.row_ind[k];
            a.values[k] *= row_scales[row];
        }
    }

    for i in 0..m {
        b[i] *= row_scales[i];
    }

    row_scales
}

/// Per-step skip hook for tests: returns `true` when `QP_PRESOLVE_SKIP` contains `n`.
///
/// In non-test builds this always returns `false` — the env var is not read in
/// production code.  Tests use `std::env::set_var("QP_PRESOLVE_SKIP", "9")` to
/// exercise no-op proofs without a public Options field.
pub(super) fn skip_step(n: usize) -> bool {
    #[cfg(test)]
    {
        std::env::var("QP_PRESOLVE_SKIP")
            .ok()
            .map(|v| v.split(',').any(|s| s.trim().parse::<usize>().ok() == Some(n)))
            .unwrap_or(false)
    }
    #[cfg(not(test))]
    {
        let _ = n;
        false
    }
}
