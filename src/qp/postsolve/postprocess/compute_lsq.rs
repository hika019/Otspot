//! 元問題空間で A^T y = -(Qx + c + bound_contrib) の最小二乗 y を計算。
//!
//! 正規方程式 (A·Aᵀ + εI) y = A·target を陰的 CG で解く。
//! A·Aᵀ を明示的に構築しないため O(k·nnz) (k = CG 収束イテレーション数)。
//! 直接法 (BTreeMap + LDL) は LASSO 等の密 A·Aᵀ 問題で O(m²·nnz/col) となり
//! 79-96% wall を支配していた。CG はその回避策 (疎性活用・全体 LSQ 回避)。
//!
//! CG が CG_MAX_ITERS 内に収束しなかった場合は direct LDL 経路にフォールバックする。

use crate::qp::linalg::{build_aat_upper_csc, compute_bound_contrib, AAT_REG_FACTOR, LSQ_DUAL_SIZE_LIMIT};
use crate::qp::problem::QpProblem;
use crate::qp::FX_TOL;
use crate::sparse::CscMatrix;
use crate::tolerances::COMP_SLACK_REL_TOL;

/// CG フルバジェット。直接法 (LDL) が memory-budget 超の場合のみ使用。
/// κ=1e6 問題でも √κ=1000 反復以内に収束する想定。
const CG_MAX_ITERS: usize = 1000;

/// CG 相対収束判定 (||r||² / ||r0||² < tol)。LSQ 解精度として十分。
const CG_TOL_SQ: f64 = 1e-20;

/// (A·Aᵀ + ε·I) を p (m_sub 次元) に適用して m_sub 次元ベクトルを返す。
/// A_sub は CSC 形式 (nrows=m_sub, ncols=n)、reg = ε。
fn aat_apply(
    a_sub: &CscMatrix,
    n: usize,
    m_sub: usize,
    p: &[f64],
    reg: f64,
    tmp: &mut Vec<f64>, // length n scratch
) -> Vec<f64> {
    // Step 1: atp = Aᵀ·p  (n 次元)
    tmp.iter_mut().for_each(|v| *v = 0.0);
    for col in 0..n {
        let cs = a_sub.col_ptr[col];
        let ce = a_sub.col_ptr[col + 1];
        let mut s = 0.0f64;
        for k in cs..ce {
            s += a_sub.values[k] * p[a_sub.row_ind[k]];
        }
        tmp[col] = s;
    }
    // Step 2: ap = A·atp + reg·p  (m_sub 次元)
    let mut ap = vec![0.0f64; m_sub];
    for col in 0..n {
        let cs = a_sub.col_ptr[col];
        let ce = a_sub.col_ptr[col + 1];
        let tv = tmp[col];
        if tv == 0.0 {
            continue;
        }
        for k in cs..ce {
            ap[a_sub.row_ind[k]] += a_sub.values[k] * tv;
        }
    }
    for i in 0..m_sub {
        ap[i] += reg * p[i];
    }
    ap
}

/// 陰的 CG で正規方程式 (A·Aᵀ + ε·I)·y = A·target を解く。
/// 収束フラグも返す (false = CG_MAX_ITERS 消費、y は best-effort)。
fn solve_aat_cg(
    a_sub: &CscMatrix,
    n: usize,
    m_sub: usize,
    target_dd: &[twofloat::TwoFloat],
    perf_trace: bool,
) -> (Option<Vec<f64>>, bool) {
    use twofloat::TwoFloat;
    let zero = TwoFloat::from(0.0);

    // RHS b = A_sub · target  (m_sub 次元、DD 精度で計算して f64 に落とす)
    let mut rhs_dd: Vec<TwoFloat> = vec![zero; m_sub];
    for col in 0..n {
        let cs = a_sub.col_ptr[col];
        let ce = a_sub.col_ptr[col + 1];
        let tv = target_dd[col];
        let tv_hi = f64::from(tv);
        let tv_lo = f64::from(tv - TwoFloat::from(tv_hi));
        for k in cs..ce {
            let row = a_sub.row_ind[k];
            let aval = a_sub.values[k];
            rhs_dd[row] = rhs_dd[row]
                + TwoFloat::new_mul(aval, tv_hi)
                + TwoFloat::new_mul(aval, tv_lo);
        }
    }
    let rhs: Vec<f64> = rhs_dd.iter().map(|&v| f64::from(v)).collect();

    // 正則化: max_diag(A·Aᵀ) = max_i Σ_k A[i,k]²  (O(nnz))
    let mut row_sq = vec![0.0f64; m_sub];
    for col in 0..n {
        for k in a_sub.col_ptr[col]..a_sub.col_ptr[col + 1] {
            let r = a_sub.row_ind[k];
            row_sq[r] += a_sub.values[k] * a_sub.values[k];
        }
    }
    let max_diag = row_sq.iter().cloned().fold(0.0f64, f64::max).max(1.0);
    let reg = AAT_REG_FACTOR * max_diag;

    // CG 初期化: y=0, r=b, p=b
    let mut y = vec![0.0f64; m_sub];
    let mut r = rhs;
    let r0_sq: f64 = r.iter().map(|&x| x * x).sum();
    if r0_sq < 1e-200 {
        return (Some(y), true);
    }
    let mut p = r.clone();
    let mut rdr = r0_sq;
    let mut tmp = vec![0.0f64; n];
    let mut iters = 0usize;
    let mut converged = false;

    for _ in 0..CG_MAX_ITERS {
        let ap = aat_apply(a_sub, n, m_sub, &p, reg, &mut tmp);
        let pap: f64 = p.iter().zip(ap.iter()).map(|(&a, &b)| a * b).sum();
        if pap <= 0.0 {
            break;
        }
        let alpha = rdr / pap;
        for i in 0..m_sub {
            y[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rdr_new: f64 = r.iter().map(|&x| x * x).sum();
        if !rdr_new.is_finite() {
            break;
        }
        iters += 1;
        if rdr_new <= CG_TOL_SQ * r0_sq {
            converged = true;
            rdr = rdr_new;
            break;
        }
        let beta = rdr_new / rdr;
        for i in 0..m_sub {
            p[i] = r[i] + beta * p[i];
        }
        rdr = rdr_new;
    }

    if perf_trace {
        let rel = (rdr / r0_sq).sqrt();
        eprintln!(
            "PERF_TRACE [compute_lsq] cg({m_sub}x{n}, nnz={}): iters={iters} converged={converged} rel_res={rel:.2e}",
            a_sub.col_ptr[n],
        );
    }

    if y.iter().any(|v| !v.is_finite()) {
        return (None, false);
    }
    (Some(y), converged)
}

/// 直接法: A·Aᵀ (上三角 CSC) + LDL + 反復精密化 (IR) で y を解く。
/// A·Aᵀ が memory budget 超なら None を返す (caller は CG にフォールバック)。
fn solve_aat_direct_ir(
    a_sub: &CscMatrix,
    n: usize,
    m_sub: usize,
    target_dd: &[twofloat::TwoFloat],
    deadline: Option<std::time::Instant>,
) -> Option<Vec<f64>> {
    use twofloat::TwoFloat;
    let zero = TwoFloat::from(0.0);

    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return None;
    }
    let aat = build_aat_upper_csc(a_sub, n, m_sub)?;
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return None;
    }
    let factor = crate::linalg::ldl::factorize(&aat).ok()?;

    let build_rhs = |v_dd: &[TwoFloat]| -> Vec<f64> {
        let mut acc: Vec<TwoFloat> = vec![zero; m_sub];
        for col in 0..n {
            let cs = a_sub.col_ptr[col];
            let ce = a_sub.col_ptr[col + 1];
            for k in cs..ce {
                let row = a_sub.row_ind[k];
                let v_f64 = f64::from(v_dd[col]);
                let lo = v_dd[col] - TwoFloat::from(v_f64);
                acc[row] = acc[row]
                    + TwoFloat::new_mul(a_sub.values[k], v_f64)
                    + TwoFloat::new_mul(a_sub.values[k], f64::from(lo));
            }
        }
        acc.iter().map(|&v| f64::from(v)).collect()
    };

    let rhs0 = build_rhs(target_dd);
    let mut y = vec![0.0_f64; m_sub];
    factor.solve(&rhs0, &mut y);
    if y.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // IR: AᵀA·y 残差を DD で計算し不足分を追加ソルブ
    const IR_STAGNATE_RATIO: f64 = 0.5;
    const IR_PROGRESS_EPS: f64 = 1e-18;
    let mut prev_r_inf = f64::INFINITY;
    loop {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        let mut aty_dd: Vec<TwoFloat> = vec![zero; n];
        for col in 0..n {
            let cs = a_sub.col_ptr[col];
            let ce = a_sub.col_ptr[col + 1];
            for k in cs..ce {
                let row = a_sub.row_ind[k];
                aty_dd[col] = aty_dd[col] + TwoFloat::new_mul(a_sub.values[k], y[row]);
            }
        }
        let r_dd: Vec<TwoFloat> = (0..n).map(|j| target_dd[j] - aty_dd[j]).collect();
        let r_inf = r_dd.iter().fold(0.0_f64, |a, &v| a.max(f64::from(v).abs()));
        if !r_inf.is_finite() {
            break;
        }
        if prev_r_inf.is_finite() && r_inf + IR_PROGRESS_EPS >= prev_r_inf {
            break;
        }
        if prev_r_inf.is_finite() && r_inf > prev_r_inf * IR_STAGNATE_RATIO {
            break;
        }
        prev_r_inf = r_inf;
        let rhs_dy = build_rhs(&r_dd);
        let mut dy = vec![0.0_f64; m_sub];
        factor.solve(&rhs_dy, &mut dy);
        if dy.iter().any(|v| !v.is_finite()) {
            break;
        }
        for i in 0..m_sub {
            y[i] += dy[i];
        }
    }
    Some(y)
}

/// A·Aᵀ LSQ を解く。
///
/// 戦略:
/// 1. CG (CG_MAX_ITERS): 陰的 matvec のみ。LASSO 等の密 A·Aᵀ で高速かつ安定。
///    CG が有限 y を返したらそれを採用 (収束/未収束に関わらず)。
///    ill-conditioned 問題でも CG の Tikhonov 正則化 が LDL より安定なことが多い。
/// 2. Direct LDL+IR フォールバック: CG が NaN/Inf を返した場合のみ。
///    A·Aᵀ が budget 超なら None を返す。
fn solve_aat(
    a_sub: &CscMatrix,
    n: usize,
    m_sub: usize,
    target_dd: &[twofloat::TwoFloat],
    deadline: Option<std::time::Instant>,
    perf_trace: bool,
) -> Option<Vec<f64>> {
    // Step 1: CG (primary path)
    let (y_cg, converged) = solve_aat_cg(a_sub, n, m_sub, target_dd, perf_trace);
    if let Some(ref y) = y_cg {
        // CG returned finite y — use it regardless of convergence.
        // Unconverged y (rel_res > CG_TOL_SQ) is still useful: caller's DD/KKT guard
        // rejects it if it doesn't improve the objective.
        let _ = converged;
        return Some(y.clone());
    }
    // Step 2: direct LDL+IR fallback (only when CG diverged to NaN/Inf)
    if perf_trace {
        eprintln!(
            "PERF_TRACE [compute_lsq] cg({m_sub}x{n}, nnz={}): NaN/Inf, falling back to direct LDL",
            a_sub.col_ptr[n],
        );
    }
    solve_aat_direct_ir(a_sub, n, m_sub, target_dd, deadline)
}

pub(crate) fn compute_lsq_dual_y(
    problem: &QpProblem,
    result: &crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
) -> Option<Vec<f64>> {
    use twofloat::TwoFloat;
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if m == 0 || result.solution.len() != n {
        return None;
    }
    if n + m > LSQ_DUAL_SIZE_LIMIT {
        return None;
    }
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return None;
    }
    let x = &result.solution;

    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = x[col];
        let cs = problem.q.col_ptr[col];
        let ce = problem.q.col_ptr[col + 1];
        for k in cs..ce {
            let row = problem.q.row_ind[k];
            qx_dd[row] = qx_dd[row] + TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    let target_dd: Vec<TwoFloat> = (0..n)
        .map(|j| -(qx_dd[j] + TwoFloat::from(problem.c[j]) + TwoFloat::from(bound_contrib[j])))
        .collect();

    let mut proj_lower = vec![f64::NEG_INFINITY; m];
    let mut proj_upper = vec![f64::INFINITY; m];
    for (i, ct) in problem.constraint_types.iter().enumerate() {
        match ct {
            crate::problem::ConstraintType::Le => proj_lower[i] = 0.0,
            crate::problem::ConstraintType::Ge => proj_upper[i] = 0.0,
            crate::problem::ConstraintType::Eq => {}
        }
    }
    for j in 0..n {
        let cs = problem.a.col_ptr[j];
        let ce = problem.a.col_ptr[j + 1];
        if ce - cs != 1 {
            continue;
        }
        let row = problem.a.row_ind[cs];
        let aij = problem.a.values[cs];
        if !aij.is_finite() || aij == 0.0 {
            continue;
        }
        let (lb, ub) = problem.bounds[j];
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();
        if lb_finite && ub_finite && (lb - ub).abs() < FX_TOL {
            continue;
        }
        let qxj = f64::from(qx_dd[j]);
        let rhs = -(qxj + problem.c[j]) / aij;
        if !rhs.is_finite() {
            continue;
        }
        match (lb_finite, ub_finite) {
            (true, false) => {
                if aij > 0.0 {
                    proj_lower[row] = proj_lower[row].max(rhs);
                } else {
                    proj_upper[row] = proj_upper[row].min(rhs);
                }
            }
            (false, true) => {
                if aij > 0.0 {
                    proj_upper[row] = proj_upper[row].min(rhs);
                } else {
                    proj_lower[row] = proj_lower[row].max(rhs);
                }
            }
            _ => {}
        }
    }
    let mut fixed_y: Vec<Option<f64>> = vec![None; m];
    let mut n_fixed = 0usize;
    for i in 0..m {
        let lo = proj_lower[i];
        let hi = proj_upper[i];
        if lo.is_finite() && hi.is_finite() {
            let scale = 1.0 + lo.abs().max(hi.abs());
            if (lo - hi).abs() < 1e-10 * scale {
                fixed_y[i] = Some((lo + hi) * 0.5);
                n_fixed += 1;
            }
        }
    }
    // Complementary slackness: rows whose primal is strictly non-binding (slack
    // > COMP_SLACK_REL_TOL relative to the row magnitudes) must have y_i = 0.
    // Without this clamp LSQ is free to assign sign-feasible but
    // slackness-violating duals — the same drift root #45 fixed for
    // `recover_removed_row_dual`. Overwrite (rather than skip) any existing
    // `fixed_y[i]` so LSQ cannot resurrect a non-zero dual on a non-binding row.
    let mut ax = vec![0.0_f64; m];
    for col in 0..n {
        let cs = problem.a.col_ptr[col];
        let ce = problem.a.col_ptr[col + 1];
        let xv = x[col];
        for k in cs..ce {
            ax[problem.a.row_ind[k]] += problem.a.values[k] * xv;
        }
    }
    for i in 0..m {
        if problem.constraint_types[i] == crate::problem::ConstraintType::Eq {
            continue;
        }
        let b_i = problem.b[i];
        let ax_i = ax[i];
        let slack = match problem.constraint_types[i] {
            crate::problem::ConstraintType::Le => b_i - ax_i,
            crate::problem::ConstraintType::Ge => ax_i - b_i,
            crate::problem::ConstraintType::Eq => 0.0,
        };
        let scale = 1.0 + b_i.abs() + ax_i.abs();
        if slack > COMP_SLACK_REL_TOL * scale {
            if fixed_y[i].is_none() {
                n_fixed += 1;
            }
            fixed_y[i] = Some(0.0);
        }
    }

    let perf_trace = std::env::var("POSTSOLVE_PERF_TRACE").ok().as_deref() == Some("1");

    if n_fixed == 0 {
        return solve_aat(&problem.a, n, m, &target_dd, deadline, perf_trace);
    }

    let mut free_row_local = vec![usize::MAX; m];
    let mut free_rows: Vec<usize> = Vec::with_capacity(m - n_fixed);
    for (i, fy) in fixed_y.iter().enumerate() {
        if fy.is_none() {
            free_row_local[i] = free_rows.len();
            free_rows.push(i);
        }
    }
    let m_free = free_rows.len();
    if m_free == 0 {
        return Some(fixed_y.iter().map(|fy| fy.unwrap_or(0.0)).collect());
    }

    let mut a_free_col_ptr = vec![0usize; n + 1];
    let mut a_free_row_ind: Vec<usize> = Vec::new();
    let mut a_free_values: Vec<f64> = Vec::new();
    for col in 0..n {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let orig_row = problem.a.row_ind[k];
            let local_row = free_row_local[orig_row];
            if local_row != usize::MAX {
                a_free_row_ind.push(local_row);
                a_free_values.push(problem.a.values[k]);
            }
        }
        a_free_col_ptr[col + 1] = a_free_row_ind.len();
    }
    let a_free = CscMatrix {
        col_ptr: a_free_col_ptr,
        row_ind: a_free_row_ind,
        values: a_free_values,
        nrows: m_free,
        ncols: n,
    };

    let mut target_adj_dd = target_dd.clone();
    for col in 0..n {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let orig_row = problem.a.row_ind[k];
            if let Some(yfi) = fixed_y[orig_row] {
                if yfi != 0.0 {
                    target_adj_dd[col] =
                        target_adj_dd[col] - TwoFloat::new_mul(problem.a.values[k], yfi);
                }
            }
        }
    }

    let y_free = match solve_aat(&a_free, n, m_free, &target_adj_dd, deadline, perf_trace) {
        Some(v) => v,
        None => return solve_aat(&problem.a, n, m, &target_dd, deadline, perf_trace),
    };

    let mut y_full = vec![0.0_f64; m];
    for (local_idx, &orig_row) in free_rows.iter().enumerate() {
        y_full[orig_row] = y_free[local_idx];
    }
    for (i, fy) in fixed_y.iter().enumerate() {
        if let Some(v) = fy {
            y_full[i] = *v;
        }
    }
    Some(y_full)
}

#[cfg(test)]
mod comp_slackness_tests {
    //! LSQ comp slackness sentinels — non-binding rows must return y_i = 0.
    //!
    //! Without the clamp, LSQ minimises ||A^T y + c|| subject only to the
    //! sign convention on y; nothing stops it from absorbing residual into a
    //! slack-positive row. Removing the `if slack > COMP_SLACK_REL_TOL` branch
    //! flips these tests to FAIL (the LSQ y becomes non-zero on the loose row).
    use super::*;
    use crate::problem::{ConstraintType, SolverResult};
    use crate::sparse::CscMatrix;

    /// Threshold for declaring "y is zero" — well below COMP_SLACK_REL_TOL.
    const Y_ZERO_TOL: f64 = 1e-9;

    fn lp_qp(
        n: usize, m: usize,
        c: Vec<f64>, a: CscMatrix, b: Vec<f64>,
        bounds: Vec<(f64, f64)>, cts: Vec<ConstraintType>,
    ) -> QpProblem {
        let q = CscMatrix::new(n, n);
        let _ = m;
        QpProblem::new(q, c, a, b, bounds, cts).unwrap()
    }

    fn run_lsq(problem: &QpProblem, x: Vec<f64>) -> Vec<f64> {
        let result = SolverResult { solution: x, ..Default::default() };
        compute_lsq_dual_y(problem, &result, None)
            .expect("LSQ should succeed on a tiny well-conditioned fixture")
    }

    /// Fixture A: 2 rows, both Le; row 0 binding at the chosen primal, row 1
    /// strictly loose. With comp clamp, row 1's y must be 0 regardless of how
    /// the LSQ residual would prefer to split.
    #[test]
    fn lsq_le_loose_row_clamped_to_zero() {
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 1).unwrap();
        let qp = lp_qp(
            1, 2,
            vec![1.0], a, vec![1.0, 10.0],
            vec![(0.0, f64::INFINITY)],
            vec![ConstraintType::Le, ConstraintType::Le],
        );
        let y = run_lsq(&qp, vec![1.0]);
        assert_eq!(y.len(), 2);
        assert!(
            y[1].abs() < Y_ZERO_TOL,
            "loose Le row y[1]={:.3e} should be clamped to 0",
            y[1],
        );
    }

    /// Fixture B: 2 rows, both Ge; row 0 binding, row 1 loose. Mirrors A on
    /// the Ge branch (proj_upper instead of proj_lower).
    #[test]
    fn lsq_ge_loose_row_clamped_to_zero() {
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 1).unwrap();
        let qp = lp_qp(
            1, 2,
            vec![-1.0], a, vec![1.0, -5.0],
            vec![(f64::NEG_INFINITY, 1.0)],
            vec![ConstraintType::Ge, ConstraintType::Ge],
        );
        let y = run_lsq(&qp, vec![1.0]);
        assert_eq!(y.len(), 2);
        assert!(
            y[1].abs() < Y_ZERO_TOL,
            "loose Ge row y[1]={:.3e} should be clamped to 0",
            y[1],
        );
    }

    /// Fixture C: mixed Le + Ge, both loose.
    #[test]
    fn lsq_mixed_loose_rows_all_clamped_to_zero() {
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2,
        ).unwrap();
        let qp = lp_qp(
            2, 2,
            vec![1.0, -1.0], a, vec![100.0, -50.0],
            vec![(0.0, 5.0), (0.0, 5.0)],
            vec![ConstraintType::Le, ConstraintType::Ge],
        );
        let y = run_lsq(&qp, vec![1.0, 1.0]);
        for i in 0..2 {
            assert!(
                y[i].abs() < Y_ZERO_TOL,
                "loose row {} y={:.3e} should be 0 (all rows non-binding at this primal)",
                i, y[i],
            );
        }
    }

    /// Fixture D: binding row keeps its y free.
    #[test]
    fn lsq_binding_row_y_is_not_clamped() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let qp = lp_qp(
            1, 1,
            vec![-1.0], a, vec![1.0],
            vec![(0.0, 1.0)],
            vec![ConstraintType::Le],
        );
        let y = run_lsq(&qp, vec![1.0]);
        assert!(
            y[0].abs() > Y_ZERO_TOL,
            "binding Le row y[0]={:.3e} should NOT be clamped to 0",
            y[0],
        );
    }
}
