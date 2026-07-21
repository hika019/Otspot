//! CSC accessors, Kahan-compensated accumulators, and the standalone
//! transforms that do not need the per-pass workspace (early infeasibility,
//! block-structure detection, large-coefficient rescaling).

use super::state::QpPresolveStatus;
use super::state::Workspace;
use crate::qp::QpProblem;
use crate::sparse::CscMatrix;
use crate::tolerances::{
    LARGE_A_COEFF_TRIGGER, Q_OFFDIAG_REL, SCALING_SIGMA_FLOOR, UNDERFLOW_GUARD, ZERO_TOL,
};

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

/// 列 `j` が構造的な二次項を持つか（pure-LP 列分類の単一判定点）。
///
/// LP-style な固定/消去 (empty/singleton/dual-fixing/free-singleton) を行う前段で
/// 「この列に Q 構造があるか」を判定する。**数値閾値ではなく構造的ゼロ**で判定する:
/// `|q| > ZERO_TOL` 等の閾値は微小 Q (例 1e-13) 列を pure-LP と誤分類し、曲率最適
/// 解を境界へ固定して suboptimal を産む。`from_triplets` が `|v| ≤ DROP_TOL` を構築
/// 時に落とすため、stored 値は構造的非ゼロである。全 step がこの単一述語を共有し、
/// 横展開漏れ (閾値ドリフト) を構造的に封じる。
pub(super) fn col_has_structural_q(q: &CscMatrix, j: usize) -> bool {
    (q.col_ptr[j]..q.col_ptr[j + 1]).any(|k| q.values[k] != 0.0)
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
    kahan_add(
        &mut ws.obj_offset,
        &mut ws.obj_offset_comp,
        0.5 * q_jj * val * val,
    );
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
        && prob
            .bounds
            .iter()
            .all(|&(lb, ub)| lb.is_infinite() && ub.is_infinite())
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
            // 構造的非ゼロ pattern で連結成分を数える (doc 通り)。微小 off-diag Q を
            // 閾値で「結合なし」と誤分類しないよう構造的ゼロ判定に統一。
            if row < n && row != j && q.values[k] != 0.0 {
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

/// True when every Q off-diagonal entry is below the scale-relative threshold.
///
/// Threshold: `eps_q = Q_OFFDIAG_REL * q_abs_max + UNDERFLOW_GUARD`.
pub(super) fn is_diagonal_q(q: &CscMatrix, n: usize) -> bool {
    let q_abs_max = q.values.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let eps_q = Q_OFFDIAG_REL * q_abs_max + UNDERFLOW_GUARD;
    for j in 0..n {
        let start = q.col_ptr[j];
        let end = q.col_ptr[j + 1];
        for k in start..end {
            let row = q.row_ind[k];
            if row != j && q.values[k].abs() > eps_q {
                return false;
            }
        }
    }
    true
}

/// If A contains entries `|a_ij| > LARGE_A_COEFF_TRIGGER`, scale each affected row by
/// `σ_i = 1/√(max|A[i,*]|)` (capped at `SCALING_SIGMA_FLOOR`) so subsequent Ruiz / IPM is
/// well-conditioned. Returns the per-row scales for dual unscaling.
pub(super) fn apply_large_coeff_rescaling(a: &mut CscMatrix, b: &mut [f64], n: usize) -> Vec<f64> {
    let m = a.nrows;
    let has_large = a.values.iter().any(|&v| v.abs() > LARGE_A_COEFF_TRIGGER);
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

    // Cap per-row amplification at 1/SCALING_SIGMA_FLOOR so the composite scaling
    // (phase1 · phase2 · Ruiz) stays within the IPM's achievable scaled accuracy.
    let row_scales: Vec<f64> = row_max
        .iter()
        .map(|&mx| {
            if mx > 1.0 {
                (1.0 / mx.sqrt()).max(SCALING_SIGMA_FLOOR)
            } else {
                1.0
            }
        })
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

/// Per-step skip hook for tests: returns `true` when step `n` is currently
/// marked as skipped in this thread.
///
/// In non-test builds this always returns `false`. Tests inject the mask via
/// [`with_skip_steps`] so parallel tests stay isolated (the previous
/// `QP_PRESOLVE_SKIP` env-var hook was process-global and caused test flake).
pub(super) fn skip_step(n: usize) -> bool {
    #[cfg(test)]
    {
        if n < 64 {
            return SKIP_STEPS_MASK.with(|c| (c.get() >> n) & 1 == 1);
        }
        false
    }
    #[cfg(not(test))]
    {
        let _ = n;
        false
    }
}

#[cfg(test)]
thread_local! {
    static SKIP_STEPS_MASK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Run `f` with the given presolve step indices marked as skipped on this
/// thread. Restores the previous mask on return (including panic unwind).
#[cfg(test)]
pub(crate) fn with_skip_steps<R>(steps: &[usize], f: impl FnOnce() -> R) -> R {
    let mut mask: u64 = 0;
    for &n in steps {
        debug_assert!(n < 64, "presolve step index {n} out of range");
        mask |= 1u64 << n;
    }
    let prev = SKIP_STEPS_MASK.with(|c| c.replace(mask));
    struct Restore(u64);
    impl Drop for Restore {
        fn drop(&mut self) {
            SKIP_STEPS_MASK.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(prev);
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sentinel: `is_diagonal_q` uses scale-relative threshold — q_abs_max=1e6, off-diag=1e-8
    /// is correctly classified as diagonal (1e-8 < Q_OFFDIAG_REL*1e6 = 1e-6).
    ///
    /// **Sentinel**: reverting to absolute `1e-10` makes eps_q=1e-10, and
    /// 1e-8 > 1e-10 → returns `false` → this test FAIL.
    #[test]
    fn is_diagonal_q_relative_threshold_sentinel() {
        let q =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1e6, 1e-8, 1e-8, 1e6], 2, 2)
                .unwrap();
        assert!(
            is_diagonal_q(&q, 2),
            "off-diag 1e-8 should be below eps_q (Q_OFFDIAG_REL*1e6≈1e-6) → diagonal"
        );
    }
}
