//! QP ソルバー: min 1/2 x'Qx + c'x  s.t. Ax (≤|=|≥) b, lb ≤ x ≤ ub
//! (OSQP/qpOASES 標準の「1/2 あり」規約)

pub mod diagnose;
pub(crate) mod ipm_core;
pub mod ipm_solver;
mod lp_dispatch;
mod problem;
pub use crate::problem::SolverResult;
pub use diagnose::{
    diagnose, DiagnosticCode, DiagnosticReport, DiagnosticWarning, ProblemInfo, Severity,
};
pub(crate) use lp_dispatch::solve_as_lp_pub;
pub use problem::{QpProblem, QpWarmStart};

use crate::options::SolverOptions;
use crate::sparse::CscMatrix;

/// Q (上三角 CSC) が PSD か。n>CHECK_SIZE_LIMIT は O(n³) を避けスキップ (true 返却)。
/// 対角負値は ‖Q‖_max 相対許容、Cholesky regularization は QPS 6 桁丸めを救う。
#[cfg(test)]
pub(crate) fn check_q_positive_semidefinite(q: &CscMatrix) -> bool {
    let n = q.nrows;
    if n == 0 {
        return true;
    }

    let mut q_abs_max = 0.0_f64;
    for &v in q.values.iter() {
        let a = v.abs();
        if a > q_abs_max {
            q_abs_max = a;
        }
    }

    const QPS_NEG_TOL_RATIO: f64 = 1e-6;
    let neg_tol = (q_abs_max * QPS_NEG_TOL_RATIO).max(1e-12);
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            if q.row_ind[k] == col && q.values[k] < -neg_tol {
                return false;
            }
        }
    }

    const CHECK_SIZE_LIMIT: usize = 1000;
    if n > CHECK_SIZE_LIMIT {
        return true;
    }

    const CHOL_EPS_RATIO: f64 = 1e-4;
    let eps = (q_abs_max * CHOL_EPS_RATIO).max(1e-8);

    let mut a = vec![0.0f64; n * n];
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            if row <= col {
                let v = q.values[k];
                a[row * n + col] = v;
                if row != col {
                    a[col * n + row] = v;
                }
            }
        }
    }
    for i in 0..n {
        a[i * n + i] += eps;
    }

    // 密 L L^T 分解。負ピボット → non-PSD。
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 0.0 {
            return false;
        }
        let sqrt_d = d.sqrt();
        a[j * n + j] = sqrt_d;
        for i in (j + 1)..n {
            let mut l_ij = a[i * n + j];
            for k in 0..j {
                l_ij -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = l_ij / sqrt_d;
        }
    }
    true
}

/// QP をデフォルト設定で解く。
pub fn solve_qp(problem: &QpProblem) -> SolverResult {
    solve_qp_with(problem, &SolverOptions::default())
}

#[deprecated(note = "to_all_le()廃止に伴いcollapse_extended_dualを使用")]
#[allow(dead_code, deprecated)]
pub(crate) fn collapse_le_expansion_dual(
    dual_expanded: &[f64],
    le_map: &crate::qp::problem::LeExpansionMap,
    orig_types: &[crate::problem::ConstraintType],
) -> Vec<f64> {
    use crate::problem::ConstraintType;
    let m_orig = orig_types.len();
    let total_expanded: usize = le_map
        .original_to_expanded
        .iter()
        .map(|rows| rows.len())
        .sum();
    if dual_expanded.len() < total_expanded {
        return dual_expanded.to_vec();
    }
    let mut collapsed = vec![0.0f64; m_orig];
    for (i, (ct, rows)) in orig_types
        .iter()
        .zip(le_map.original_to_expanded.iter())
        .enumerate()
    {
        collapsed[i] = match ct {
            ConstraintType::Le => dual_expanded[rows[0]],
            ConstraintType::Ge => -dual_expanded[rows[0]],
            ConstraintType::Eq => {
                let mu1 = dual_expanded[rows[0]];
                let mu2 = if rows.len() > 1 {
                    dual_expanded[rows[1]]
                } else {
                    0.0
                };
                mu1 - mu2
            }
        };
    }
    collapsed
}

/// faer supernodal Cholesky の deepest stack 要求 + マージン。Rust thread デフォルト 2 MB では
/// BOYD1 級 (n=93261) で overflow するため、入口で必ずこのサイズの scoped thread に載せる。
pub(crate) const SOLVE_STACK_SIZE: usize = 8 * 1024 * 1024;

/// QP をカスタム設定で解く (8 MB scoped thread で stack overflow を防ぐ)。
pub fn solve_qp_with(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    std::thread::scope(|s| {
        let handle = std::thread::Builder::new()
            .stack_size(SOLVE_STACK_SIZE)
            .spawn_scoped(s, || dispatch_solve_qp(problem, options))
            .expect("spawn QP solver thread");
        handle.join().expect("QP solver thread panicked")
    })
}

/// Q=0 (LP) は Simplex に委譲、その他は IPPMM。
fn dispatch_solve_qp(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    if problem.is_zero_q() {
        return solve_as_lp_pub(problem, options);
    }
    ipm_solver::solve_ipm(problem, options)
}

/// FX (固定) 変数判定: |lb − ub| < FX_TOL。
pub(crate) const FX_TOL: f64 = 1e-12;

/// reduced bound_duals を元問題空間に展開。除去変数の bound_dual は 0.0 で埋める。
pub(crate) fn remap_bound_duals_to_orig(
    presolve_result: &crate::presolve::QpPresolveResult,
    orig_bounds: &[(f64, f64)],
    reduced_bound_duals: &[f64],
) -> Vec<f64> {
    let n_lb_orig = orig_bounds.iter().filter(|(lb, _)| lb.is_finite()).count();
    let n_ub_orig = orig_bounds.iter().filter(|(_, ub)| ub.is_finite()).count();
    if n_lb_orig + n_ub_orig == 0 {
        return Vec::new();
    }
    let reduced_bounds = &presolve_result.reduced.bounds;
    let n_lb_reduced = reduced_bounds
        .iter()
        .filter(|(lb, _)| lb.is_finite())
        .count();
    let n_reduced = reduced_bounds.len();

    let mut lb_bd_idx: Vec<Option<usize>> = vec![None; n_reduced];
    let mut ub_bd_idx: Vec<Option<usize>> = vec![None; n_reduced];
    {
        let mut li = 0usize;
        for (jj, &(lb, _)) in reduced_bounds.iter().enumerate() {
            if lb.is_finite() {
                lb_bd_idx[jj] = Some(li);
                li += 1;
            }
        }
        let mut ui = 0usize;
        for (jj, &(_, ub)) in reduced_bounds.iter().enumerate() {
            if ub.is_finite() {
                ub_bd_idx[jj] = Some(n_lb_reduced + ui);
                ui += 1;
            }
        }
    }

    let mut new_bd = vec![0.0_f64; n_lb_orig + n_ub_orig];
    if !reduced_bound_duals.is_empty() {
        let mut orig_li = 0usize;
        for (j, &(lb, _)) in orig_bounds.iter().enumerate() {
            if lb.is_finite() {
                if let Some(jj) = presolve_result.col_map[j] {
                    if let Some(bd_idx) = lb_bd_idx[jj] {
                        if bd_idx < reduced_bound_duals.len() {
                            new_bd[orig_li] = reduced_bound_duals[bd_idx];
                        }
                    }
                }
                orig_li += 1;
            }
        }
        let mut orig_ui = 0usize;
        for (j, &(_, ub)) in orig_bounds.iter().enumerate() {
            if ub.is_finite() {
                if let Some(jj) = presolve_result.col_map[j] {
                    if let Some(bd_idx) = ub_bd_idx[jj] {
                        if bd_idx < reduced_bound_duals.len() {
                            new_bd[n_lb_orig + orig_ui] = reduced_bound_duals[bd_idx];
                        }
                    }
                }
                orig_ui += 1;
            }
        }
    }
    new_bd
}

/// AAT 対角ε 正則化倍率 (rank-deficient 対策)。f64 eps より十分上、LDL dynamic reg より十分下。
const AAT_REG_FACTOR: f64 = 1e-12;

/// LSQ dual の size guard。AAT (m×m) LDL は m=186k で 30+ GB メモリを確保するため skip。
const LSQ_DUAL_SIZE_LIMIT: usize = 50_000;

/// primal x から KKT を満たす dual y を A^T y = -(Qx + c + bound_contrib) の最小二乗で再計算。
/// 正規方程式 (A·A^T) y = A·r を LDL で解き、KKT 残差改善時のみ採用 (退行防止)。
/// deadline 経過時は no-op。
pub(crate) fn refine_dual_lsq(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
) {
    use twofloat::TwoFloat;
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let Some(y_new) = compute_lsq_dual_y(problem, result) else {
        return;
    };
    let n = problem.num_vars;
    // ill-conditioned (cond~1e12) では f64 mat_vec の cancellation noise が真残差を
    // 上回り IPM の正しい y が LSQ y に置換される。DD で比較する。
    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = result.solution[col];
        let cs = problem.q.col_ptr[col];
        let ce = problem.q.col_ptr[col + 1];
        for k in cs..ce {
            let row = problem.q.row_ind[k];
            qx_dd[row] = qx_dd[row] + TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let aty_dd = |y: &[f64]| -> Vec<TwoFloat> {
        let mut acc: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            let cs = problem.a.col_ptr[col];
            let ce = problem.a.col_ptr[col + 1];
            for k in cs..ce {
                let row = problem.a.row_ind[k];
                acc[col] = acc[col] + TwoFloat::new_mul(problem.a.values[k], y[row]);
            }
        }
        acc
    };
    let aty_old_dd = aty_dd(&result.dual_solution);
    let aty_new_dd = aty_dd(&y_new);
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    // componentwise rel = |r_j| / (1 + |Qx_j| + |c_j| + |Aty_j| + |z_j|) で比較。
    // abs max では ill-scaled 問題で外れ残差が巨大スケールに埋もれる。
    let mut max_rel_old = 0.0_f64;
    let mut max_rel_new = 0.0_f64;
    for j in 0..n {
        let (lbj, ubj) = problem.bounds[j];
        if lbj.is_finite() && ubj.is_finite() && (lbj - ubj).abs() < FX_TOL {
            continue;
        }
        if problem.a.col_ptr.len() > j + 1 && problem.a.col_ptr[j + 1] - problem.a.col_ptr[j] == 0 {
            continue;
        }
        let r_old_dd = qx_dd[j]
            + TwoFloat::from(problem.c[j])
            + aty_old_dd[j]
            + TwoFloat::from(bound_contrib[j]);
        let r_new_dd = qx_dd[j]
            + TwoFloat::from(problem.c[j])
            + aty_new_dd[j]
            + TwoFloat::from(bound_contrib[j]);
        let qx_j = f64::from(qx_dd[j]).abs();
        let aty_old_j = f64::from(aty_old_dd[j]).abs();
        let aty_new_j = f64::from(aty_new_dd[j]).abs();
        let scale_old = 1.0 + qx_j + problem.c[j].abs() + aty_old_j + bound_contrib[j].abs();
        let scale_new = 1.0 + qx_j + problem.c[j].abs() + aty_new_j + bound_contrib[j].abs();
        let rel_old = f64::from(r_old_dd).abs() / scale_old;
        let rel_new = f64::from(r_new_dd).abs() / scale_new;
        if rel_old > max_rel_old {
            max_rel_old = rel_old;
        }
        if rel_new > max_rel_new {
            max_rel_new = rel_new;
        }
    }
    if max_rel_new < max_rel_old {
        result.dual_solution = y_new;
    }
}

/// singleton column の停留性から row dual の feasible interval を作り、現在 y を射影する。
/// unconstrained LSQ refine では one-sided bound 列で「非負 z で補正不能な y」が出るのを補正。
pub(crate) fn project_duals_from_singleton_columns(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
) {
    let Some((lower, upper)) = compute_dual_recovery_row_bounds(problem, &result.solution) else {
        return;
    };
    if result.dual_solution.len() != problem.num_constraints {
        return;
    }
    for row in 0..problem.num_constraints {
        let lo = lower[row];
        let hi = upper[row];
        if lo > hi {
            continue;
        }
        let y = &mut result.dual_solution[row];
        if *y < lo {
            *y = lo;
        } else if *y > hi {
            *y = hi;
        }
    }
}

fn compute_dual_recovery_row_bounds(
    problem: &QpProblem,
    solution: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if solution.len() != n {
        return None;
    }

    let qx = problem.q.mat_vec_mul(solution).ok()?;
    let (ax, row_abs_activity) = compute_dual_recovery_row_activity(problem, solution)?;

    let mut lower = vec![f64::NEG_INFINITY; m];
    let mut upper = vec![f64::INFINITY; m];

    for (row, ct) in problem.constraint_types.iter().enumerate() {
        match ct {
            crate::problem::ConstraintType::Le => lower[row] = 0.0,
            crate::problem::ConstraintType::Ge => upper[row] = 0.0,
            crate::problem::ConstraintType::Eq => {}
        }
    }
    const SLACK_TOL_REL: f64 = 1e-8;
    for i in 0..m {
        let slack = match problem.constraint_types[i] {
            crate::problem::ConstraintType::Le => problem.b[i] - ax[i],
            crate::problem::ConstraintType::Ge => ax[i] - problem.b[i],
            crate::problem::ConstraintType::Eq => continue,
        };
        let tol = dual_recovery_row_slack_tol(problem, i, ax[i], row_abs_activity[i], SLACK_TOL_REL);
        if slack > tol {
            lower[i] = 0.0;
            upper[i] = 0.0;
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
        let is_fx = lb_finite && ub_finite && (lb - ub).abs() < FX_TOL;
        if is_fx {
            continue;
        }

        let rhs = -(qx[j] + problem.c[j]) / aij;
        if !rhs.is_finite() {
            continue;
        }

        match (lb_finite, ub_finite) {
            // lb-only: qx + c + a*y = z_lb >= 0
            (true, false) => {
                if aij > 0.0 {
                    lower[row] = lower[row].max(rhs);
                } else {
                    upper[row] = upper[row].min(rhs);
                }
            }
            // ub-only: qx + c + a*y = -z_ub <= 0
            (false, true) => {
                if aij > 0.0 {
                    upper[row] = upper[row].min(rhs);
                } else {
                    lower[row] = lower[row].max(rhs);
                }
            }
            _ => {}
        }
    }

    Some((lower, upper))
}

fn compute_dual_recovery_row_activity(
    problem: &QpProblem,
    solution: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let ax = problem.a.mat_vec_mul(solution).ok()?;
    let mut row_abs_activity = vec![0.0_f64; problem.num_constraints];
    for j in 0..problem.num_vars {
        let xabs = solution[j].abs();
        if xabs == 0.0 {
            continue;
        }
        for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
            let row = problem.a.row_ind[k];
            row_abs_activity[row] += problem.a.values[k].abs() * xabs;
        }
    }
    Some((ax, row_abs_activity))
}

fn dual_recovery_row_slack_tol(
    problem: &QpProblem,
    row: usize,
    ax: f64,
    row_abs_activity: f64,
    rel: f64,
) -> f64 {
    rel * (1.0 + problem.b[row].abs() + ax.abs() + row_abs_activity)
}

fn dual_recovery_progress_tol(prev_kkt: f64, cur_kkt: f64, target_pf: f64) -> f64 {
    let scale = prev_kkt
        .abs()
        .max(cur_kkt.abs())
        .max(target_pf.abs())
        .max(1.0);
    64.0 * f64::EPSILON * scale
}

fn row_is_active_for_dual_recovery(
    problem: &QpProblem,
    row: usize,
    ax: &[f64],
    row_abs_activity: &[f64],
    slack_tol_rel: f64,
) -> bool {
    match problem.constraint_types[row] {
        crate::problem::ConstraintType::Eq => true,
        crate::problem::ConstraintType::Le => {
            let slack = problem.b[row] - ax[row];
            let tol =
                dual_recovery_row_slack_tol(problem, row, ax[row], row_abs_activity[row], slack_tol_rel);
            slack.abs() <= tol
        }
        crate::problem::ConstraintType::Ge => {
            let slack = ax[row] - problem.b[row];
            let tol =
                dual_recovery_row_slack_tol(problem, row, ax[row], row_abs_activity[row], slack_tol_rel);
            slack.abs() <= tol
        }
    }
}

fn collect_dual_recovery_cluster_rows(
    problem: &QpProblem,
    candidate_cols: &[usize],
    candidate_rel: &[f64],
    ax: &[f64],
    row_abs_activity: &[f64],
    _target_pf: f64,
) -> Option<(usize, Vec<usize>)> {
    debug_assert_eq!(candidate_cols.len(), candidate_rel.len());
    if candidate_cols.is_empty() {
        return None;
    }

    let mut order: Vec<usize> = (0..candidate_cols.len()).collect();
    order.sort_by(|&lhs, &rhs| {
        candidate_rel[rhs]
            .partial_cmp(&candidate_rel[lhs])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let worst_pos = order[0];
    let worst_j = candidate_cols[worst_pos];
    let worst_rel = candidate_rel[worst_pos];
    if !worst_rel.is_finite() || worst_rel <= 0.0 {
        return None;
    }

    const CLUSTER_REL_CUTOFF_RATIO: f64 = 0.25;
    let rel_cutoff = worst_rel * CLUSTER_REL_CUTOFF_RATIO;

    let m = problem.num_constraints;
    let mut in_cluster = vec![false; m];
    let mut rows = Vec::new();
    let push_active_rows = |col: usize, in_cluster: &mut [bool], rows: &mut Vec<usize>| {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let row = problem.a.row_ind[k];
            if !row_is_active_for_dual_recovery(
                problem,
                row,
                ax,
                row_abs_activity,
                DUAL_RECOVERY_ACTIVE_TOL_REL,
            ) {
                continue;
            }
            if !in_cluster[row] {
                in_cluster[row] = true;
                rows.push(row);
            }
        }
    };
    push_active_rows(worst_j, &mut in_cluster, &mut rows);
    if rows.is_empty() {
        return None;
    }

    let mut changed = true;
    while changed {
        changed = false;
        for &pos in &order {
            if candidate_rel[pos] < rel_cutoff {
                break;
            }
            let col = candidate_cols[pos];
            let touches_cluster = (problem.a.col_ptr[col]..problem.a.col_ptr[col + 1])
                .any(|k| in_cluster[problem.a.row_ind[k]]);
            if !touches_cluster {
                continue;
            }
            let before = rows.len();
            push_active_rows(col, &mut in_cluster, &mut rows);
            if rows.len() > before {
                changed = true;
            }
        }
    }

    rows.sort_unstable();
    Some((worst_j, rows))
}

#[derive(Clone, Copy)]
enum DualRecoveryBoundVar {
    Lower { var: usize, slot: usize },
    Upper { var: usize, slot: usize },
}

impl DualRecoveryBoundVar {
    fn var(self) -> usize {
        match self {
            DualRecoveryBoundVar::Lower { var, .. } | DualRecoveryBoundVar::Upper { var, .. } => var,
        }
    }

    fn slot(self) -> usize {
        match self {
            DualRecoveryBoundVar::Lower { slot, .. } | DualRecoveryBoundVar::Upper { slot, .. } => slot,
        }
    }

    fn coeff(self) -> f64 {
        match self {
            DualRecoveryBoundVar::Lower { .. } => -1.0,
            DualRecoveryBoundVar::Upper { .. } => 1.0,
        }
    }
}

fn select_dual_recovery_local_bounds(
    problem: &QpProblem,
    solution: &[f64],
    bound_duals: &[f64],
    cols: &[usize],
    provisional_residual: &[f64],
) -> (Vec<DualRecoveryBoundVar>, Vec<usize>) {
    let n = problem.num_vars;
    let n_lb = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    let mut lb_slot_of_var = vec![None; n];
    let mut ub_slot_of_var = vec![None; n];
    let mut lb_slot = 0usize;
    let mut ub_slot = n_lb;
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        if lb.is_finite() {
            lb_slot_of_var[j] = Some(lb_slot);
            lb_slot += 1;
        }
        if ub.is_finite() {
            ub_slot_of_var[j] = Some(ub_slot);
            ub_slot += 1;
        }
    }

    let mut local_bounds = Vec::new();
    for &col in cols {
        let xj = solution[col];
        let tol = DUAL_RECOVERY_ACTIVE_TOL_REL * (1.0 + xj.abs());
        let (lb, ub) = problem.bounds[col];
        let is_fx = lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL;
        if is_fx {
            continue;
        }
        let lb_active = lb.is_finite()
            && ((xj - lb).abs() <= tol
                || lb_slot_of_var[col]
                    .and_then(|slot| bound_duals.get(slot))
                    .is_some_and(|&z| z > 0.0));
        let ub_active = ub.is_finite()
            && ((ub - xj).abs() <= tol
                || ub_slot_of_var[col]
                    .and_then(|slot| bound_duals.get(slot))
                    .is_some_and(|&z| z > 0.0));
        let residual_j = provisional_residual[col];
        let lb_can_help = residual_j > 0.0
            || lb_slot_of_var[col]
                .and_then(|slot| bound_duals.get(slot))
                .is_some_and(|&z| z > 0.0);
        let ub_can_help = residual_j < 0.0
            || ub_slot_of_var[col]
                .and_then(|slot| bound_duals.get(slot))
                .is_some_and(|&z| z > 0.0);
        match (lb_active, ub_active) {
            (true, false) => {
                if lb_can_help {
                    if let Some(slot) = lb_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Lower { var: col, slot });
                    }
                }
            }
            (false, true) => {
                if ub_can_help {
                    if let Some(slot) = ub_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Upper { var: col, slot });
                    }
                }
            }
            (true, true) => {
                if lb_can_help && !ub_can_help {
                    if let Some(slot) = lb_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Lower { var: col, slot });
                    }
                } else if ub_can_help && !lb_can_help {
                    if let Some(slot) = ub_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Upper { var: col, slot });
                    }
                } else if lb_can_help && ub_can_help {
                    let lb_dist = (xj - lb).abs();
                    let ub_dist = (ub - xj).abs();
                    if lb_dist <= ub_dist {
                        if let Some(slot) = lb_slot_of_var[col] {
                            local_bounds.push(DualRecoveryBoundVar::Lower { var: col, slot });
                        }
                    } else if let Some(slot) = ub_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Upper { var: col, slot });
                    }
                }
            }
            (false, false) => {}
        }
    }

    let mut bound_pos_of_var = vec![usize::MAX; n];
    for (pos, &bound) in local_bounds.iter().enumerate() {
        bound_pos_of_var[bound.var()] = pos;
    }
    (local_bounds, bound_pos_of_var)
}

/// 明確に slack ある不等式行の dual を相補性から 0 にする。LSQ/IR は stationarity のみ見るため
/// slack 行に dual が残る場合がある。
pub(crate) fn zero_inactive_inequality_duals(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
) {
    if result.solution.len() != problem.num_vars
        || result.dual_solution.len() != problem.num_constraints
    {
        return;
    }
    let Some((ax, row_abs_activity)) = compute_dual_recovery_row_activity(problem, &result.solution) else {
        return;
    };
    const SLACK_TOL_REL: f64 = 1e-8;
    for i in 0..problem.num_constraints {
        let slack = match problem.constraint_types[i] {
            crate::problem::ConstraintType::Le => problem.b[i] - ax[i],
            crate::problem::ConstraintType::Ge => ax[i] - problem.b[i],
            crate::problem::ConstraintType::Eq => continue,
        };
        let tol = dual_recovery_row_slack_tol(problem, i, ax[i], row_abs_activity[i], SLACK_TOL_REL);
        if slack > tol {
            result.dual_solution[i] = 0.0;
        }
    }
}

const DUAL_RECOVERY_ACTIVE_TOL_REL: f64 = 1e-8;

fn collect_dual_recovery_free_columns(
    problem: &QpProblem,
    result: &crate::problem::SolverResult,
) -> Vec<usize> {
    let n = problem.num_vars;
    let mut free_idx: Vec<usize> = Vec::with_capacity(n);
    for j in 0..n {
        let xj = result.solution[j];
        let tol = DUAL_RECOVERY_ACTIVE_TOL_REL * (1.0 + xj.abs());
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() && (xj - lb).abs() < tol {
            continue;
        }
        if ub.is_finite() && (ub - xj).abs() < tol {
            continue;
        }
        if problem.a.col_ptr[j + 1] - problem.a.col_ptr[j] == 0 {
            continue;
        }
        free_idx.push(j);
    }
    free_idx
}

/// 不等式符号制約と inactive 0 制約を守りつつ ‖A^T y - target‖² を projected gradient で下げる。
pub(crate) fn refine_dual_projected_gradient(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
) {
    use twofloat::TwoFloat;
    let trace = std::env::var("REFINE_DUAL_PG_TRACE").ok().as_deref() == Some("1");
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if result.solution.len() != n || result.dual_solution.len() != m {
        return;
    }
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = result.solution[col];
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            qx_dd[problem.q.row_ind[k]] =
                qx_dd[problem.q.row_ind[k]] + TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let qx: Vec<f64> = qx_dd.iter().map(|&v| f64::from(v)).collect();
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    let target: Vec<f64> = (0..n)
        .map(|j| -(qx[j] + problem.c[j] + bound_contrib[j]))
        .collect();

    let objective = |y: &[f64]| -> Option<(f64, Vec<f64>)> {
        let aty = if problem.a.nrows > 0 {
            problem.a.transpose().mat_vec_mul(y).ok()?
        } else {
            vec![0.0_f64; n]
        };
        let mut residual = vec![0.0_f64; n];
        let mut obj = 0.0_f64;
        for j in 0..n {
            residual[j] = aty[j] - target[j];
            obj += 0.5 * residual[j] * residual[j];
        }
        Some((obj, residual))
    };

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
        let rhs = -(qx[j] + problem.c[j]) / aij;
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
    for i in 0..m {
        if proj_lower[i] > proj_upper[i] {
            let (lo, hi) = match problem.constraint_types[i] {
                crate::problem::ConstraintType::Le => (0.0, f64::INFINITY),
                crate::problem::ConstraintType::Ge => (f64::NEG_INFINITY, 0.0),
                crate::problem::ConstraintType::Eq => (f64::NEG_INFINITY, f64::INFINITY),
            };
            proj_lower[i] = lo;
            proj_upper[i] = hi;
        }
    }

    let project_feasible = |y: &mut [f64]| {
        for (i, ct) in problem.constraint_types.iter().enumerate() {
            match ct {
                crate::problem::ConstraintType::Le => y[i] = y[i].max(0.0),
                crate::problem::ConstraintType::Ge => y[i] = y[i].min(0.0),
                crate::problem::ConstraintType::Eq => {}
            }
        }
        for i in 0..m {
            y[i] = y[i].clamp(proj_lower[i], proj_upper[i]);
        }
    };

    let mut y_start = result.dual_solution.clone();
    project_feasible(&mut y_start);
    let Some((mut obj_curr, mut residual_curr)) = objective(&y_start) else {
        return;
    };
    let mut y_curr = y_start;
    let mut y_best = y_curr.clone();
    let mut obj_best = obj_curr;
    let mut prev_obj = obj_curr;

    let pg_max_iters = m.saturating_mul(2).clamp(200, 2000);
    const ACCEPT_TOL_REL: f64 = 1e-12;
    let obj_converge_thresh = 1e-16 * (n as f64).max(1.0);
    const STAGNATE_MIN_RATIO: f64 = 1e-7;

    for iter in 0..pg_max_iters {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        if obj_curr < obj_converge_thresh {
            break;
        }
        let grad = match problem.a.mat_vec_mul(&residual_curr) {
            Ok(v) => v,
            Err(_) => break,
        };
        let grad_inf = grad.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        if !grad_inf.is_finite() || grad_inf < 1e-14 {
            break;
        }
        let grad_sq = grad.iter().map(|v| v * v).sum::<f64>();
        if !grad_sq.is_finite() || grad_sq < 1e-28 {
            break;
        }
        let aty_grad = match problem.a.transpose().mat_vec_mul(&grad) {
            Ok(v) => v,
            Err(_) => break,
        };
        let curvature = aty_grad.iter().map(|v| v * v).sum::<f64>();
        if !curvature.is_finite() || curvature < 1e-28 {
            break;
        }
        let base_step = (grad_sq / curvature).clamp(1e-14, 1e8);
        let mut accepted = false;
        let mut step = base_step;
        while step > 0.0 {
            let mut y_try = y_curr.clone();
            for i in 0..m {
                y_try[i] -= step * grad[i];
            }
            project_feasible(&mut y_try);
            let Some((obj_try, residual_try)) = objective(&y_try) else {
                continue;
            };
            if obj_try <= obj_curr + ACCEPT_TOL_REL * (1.0 + obj_curr) {
                if trace {
                    eprintln!(
                        "DUAL_PG iter={} step={:.3e} base={:.3e} obj {:.3e}->{:.3e} grad_inf={:.3e}",
                        iter, step, base_step, obj_curr, obj_try, grad_inf
                    );
                }
                y_curr = y_try;
                obj_curr = obj_try.min(obj_curr);
                residual_curr = residual_try;
                if obj_curr < obj_best {
                    y_best = y_curr.clone();
                    obj_best = obj_curr;
                }
                accepted = true;
                break;
            }
            let next_step = step * 0.5;
            if next_step == step {
                break;
            }
            step = next_step;
        }
        if !accepted {
            if trace {
                eprintln!(
                    "DUAL_PG iter={} no acceptable step obj={:.3e} grad_inf={:.3e} base={:.3e}",
                    iter, obj_curr, grad_inf, base_step
                );
            }
            break;
        }
        let relative_improvement = if prev_obj > 0.0 {
            (prev_obj - obj_curr) / prev_obj
        } else {
            0.0
        };
        if relative_improvement < STAGNATE_MIN_RATIO {
            break;
        }
        prev_obj = obj_curr;
    }

    let mut tmp = result.clone();
    tmp.dual_solution = y_best;
    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
    };
    let pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    let post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &tmp.solution,
        &tmp.dual_solution,
        &tmp.bound_duals,
    );
    if trace {
        eprintln!("DUAL_PG final kkt {:.3e}->{:.3e}", pre, post);
    }
    if post < pre {
        result.dual_solution = tmp.dual_solution;
    }
}

/// worst residual 列に接続する active cluster を局所的に再最適化。
/// [active row duals ; active bound duals] 連成で解く (row dual 単独では bound 押し返しで悪化)。
pub(crate) fn refine_dual_worst_active_block(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
) {
    let trace = std::env::var("REFINE_DUAL_BLOCK_TRACE").ok().as_deref() == Some("1");
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if result.solution.len() != n || result.dual_solution.len() != m {
        return;
    }

    let Ok(qx) = problem.q.mat_vec_mul(&result.solution) else {
        return;
    };
    let aty = if problem.a.nrows > 0 {
        match problem.a.transpose().mat_vec_mul(&result.dual_solution) {
            Ok(v) => v,
            Err(_) => return,
        }
    } else {
        vec![0.0_f64; n]
    };
    let Some((ax, row_abs_activity)) = compute_dual_recovery_row_activity(problem, &result.solution) else {
        return;
    };
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);

    let mut worst_j = None;
    let mut worst_rel = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = problem.bounds[j];
        let is_fx = lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL;
        let is_empty_col = problem.a.col_ptr[j + 1] == problem.a.col_ptr[j];
        if is_fx || is_empty_col {
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

    if trace {
        eprintln!(
            "DUAL_BLOCK worst_j={} worst_rel={:.3e} active_rows={} touched_cols={} local_bounds={}",
            worst_j,
            worst_rel,
            rows.len(),
            cols.len(),
            local_bounds.len()
        );
    }

    let ulen = rlen + local_bounds.len();
    if ulen == 0 {
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
    let Ok(factor) = crate::linalg::ldl::factorize(&gram_csc) else {
        return;
    };
    let mut block_sol = vec![0.0_f64; ulen];
    factor.solve(&rhs, &mut block_sol);
    if block_sol.iter().any(|v| !v.is_finite()) {
        return;
    }

    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
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
    if trace {
        eprintln!("DUAL_BLOCK kkt {:.3e}->{:.3e}", pre, best_kkt);
    }
    if best_kkt < pre {
        result.dual_solution = best.dual_solution;
        result.bound_duals = best.bound_duals;
    }
}

/// A^T y = -(Qx + c + bound_contrib) の最小二乗 y を (A·A^T) y = A·target の LDL で求め、
/// DD (TwoFloat) 残差で Wilkinson 流 iterative refinement する。ill-conditioned で
/// cond(A·A^T)·ε の限界を超えて refine するために IR が必須。
pub(crate) fn compute_lsq_dual_y(
    problem: &QpProblem,
    result: &crate::problem::SolverResult,
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

    let solve_lsq_ir = |a_sub: &CscMatrix, m_sub: usize, v_dd: &[TwoFloat]| -> Option<Vec<f64>> {
        let aat_sub = build_aat_upper_csc(a_sub, n, m_sub)?;
        let factor = crate::linalg::ldl::factorize(&aat_sub).ok()?;
        let build_rhs_sub = |v_dd: &[TwoFloat]| -> Vec<f64> {
            let mut acc: Vec<TwoFloat> = vec![zero_dd; m_sub];
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
        let rhs0 = build_rhs_sub(v_dd);
        let mut y_sub = vec![0.0_f64; m_sub];
        factor.solve(&rhs0, &mut y_sub);
        if y_sub.iter().any(|v| !v.is_finite()) {
            return None;
        }
        const IR_STAGNATE_RATIO: f64 = 0.5;
        const IR_PROGRESS_EPS: f64 = 1e-18;
        let mut prev_r_inf = f64::INFINITY;
        loop {
            let mut atysub_dd: Vec<TwoFloat> = vec![zero_dd; n];
            for col in 0..n {
                let cs = a_sub.col_ptr[col];
                let ce = a_sub.col_ptr[col + 1];
                for k in cs..ce {
                    let row = a_sub.row_ind[k];
                    atysub_dd[col] =
                        atysub_dd[col] + TwoFloat::new_mul(a_sub.values[k], y_sub[row]);
                }
            }
            let r_dd: Vec<TwoFloat> = (0..n).map(|j| v_dd[j] - atysub_dd[j]).collect();
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
            let rhs_dy = build_rhs_sub(&r_dd);
            let mut dy = vec![0.0_f64; m_sub];
            factor.solve(&rhs_dy, &mut dy);
            if dy.iter().any(|v| !v.is_finite()) {
                break;
            }
            for i in 0..m_sub {
                y_sub[i] += dy[i];
            }
        }
        Some(y_sub)
    };

    if n_fixed == 0 {
        return solve_lsq_ir(&problem.a, m, &target_dd);
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

    let y_free = match solve_lsq_ir(&a_free, m_free, &target_adj_dd) {
        Some(v) => v,
        None => return solve_lsq_ir(&problem.a, m, &target_dd),
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

/// borderline pf を violating 制約方向に最小ノルム射影で押し込む post-processing。
/// (A A^T) λ = v_active を LDL + DD-IR で解き δ = A^T λ、pf 改善時のみ採用。
pub(crate) fn refine_primal_lsq(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
) {
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if m == 0 || result.solution.len() != n {
        return;
    }
    let x = &mut result.solution;

    // ill-conditioned 系で f64 sum の cancellation を防ぐため Ax を DD で積算。
    use crate::problem::ConstraintType;
    use twofloat::TwoFloat;
    let zero_dd = TwoFloat::from(0.0);
    let mut ax_dd: Vec<TwoFloat> = vec![zero_dd; m];
    for col in 0..n {
        let xv = x[col];
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            ax_dd[problem.a.row_ind[k]] =
                ax_dd[problem.a.row_ind[k]] + TwoFloat::new_mul(problem.a.values[k], xv);
        }
    }
    let ax: Vec<f64> = ax_dd.iter().map(|&v| f64::from(v)).collect();
    const PRIMAL_VIOLATION_TOL: f64 = 1e-12;
    let mut v = vec![0.0_f64; m];
    let mut max_v_pre = 0.0_f64;
    for i in 0..m {
        let raw = match problem.constraint_types[i] {
            ConstraintType::Eq => ax[i] - problem.b[i],
            ConstraintType::Ge => -(ax[i] - problem.b[i]),
            ConstraintType::Le => ax[i] - problem.b[i],
        };
        if raw > PRIMAL_VIOLATION_TOL {
            v[i] = raw;
            max_v_pre = max_v_pre.max(raw);
        }
    }
    if max_v_pre <= PRIMAL_VIOLATION_TOL {
        return;
    }
    // target = ax − b で A δ = target を解く (Le/Ge/Eq とも一貫した符号)。
    let target: Vec<f64> = (0..m)
        .map(|i| {
            match problem.constraint_types[i] {
                ConstraintType::Eq => ax[i] - problem.b[i],
                ConstraintType::Ge => {
                    let r = ax[i] - problem.b[i];
                    if r < -PRIMAL_VIOLATION_TOL {
                        r
                    } else {
                        0.0
                    }
                }
                ConstraintType::Le => {
                    let r = ax[i] - problem.b[i];
                    if r > PRIMAL_VIOLATION_TOL {
                        r
                    } else {
                        0.0
                    }
                }
            }
        })
        .collect();
    let target_inf = target.iter().map(|t| t.abs()).fold(0.0_f64, f64::max);
    if target_inf <= PRIMAL_VIOLATION_TOL {
        return;
    }

    // (A A^T) λ = target を LDL + DD-IR (cond(AAT)≈1e13 の暴走を抑制)。
    let aat = match build_aat_upper_csc(&problem.a, n, m) {
        Some(mat) => mat,
        None => return,
    };
    let factor = match crate::linalg::ldl::factorize(&aat) {
        Ok(f) => f,
        Err(_) => return,
    };
    let mut lambda = vec![0.0_f64; m];
    factor.solve(&target, &mut lambda);
    if lambda.iter().any(|v| !v.is_finite()) {
        return;
    }
    const IR_STAGNATE_RATIO: f64 = 0.5;
    const IR_PROGRESS_EPS: f64 = 1e-18;
    let mut prev_r_inf = f64::INFINITY;
    loop {
        let mut atl_dd: Vec<TwoFloat> = vec![zero_dd; n];
        for j in 0..n {
            for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
                let i = problem.a.row_ind[k];
                if i < m {
                    atl_dd[j] = atl_dd[j] + TwoFloat::new_mul(problem.a.values[k], lambda[i]);
                }
            }
        }
        let mut r_dd: Vec<TwoFloat> = (0..m).map(|i| TwoFloat::from(target[i])).collect();
        for j in 0..n {
            let atl_j_f64 = f64::from(atl_dd[j]);
            let atl_j_lo = atl_dd[j] - TwoFloat::from(atl_j_f64);
            for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
                let i = problem.a.row_ind[k];
                if i < m {
                    r_dd[i] = r_dd[i]
                        - TwoFloat::new_mul(problem.a.values[k], atl_j_f64)
                        - TwoFloat::new_mul(problem.a.values[k], f64::from(atl_j_lo));
                }
            }
        }
        let r_f64: Vec<f64> = r_dd.iter().map(|&v| f64::from(v)).collect();
        let r_inf = r_f64.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
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
        let mut dlambda = vec![0.0_f64; m];
        factor.solve(&r_f64, &mut dlambda);
        if dlambda.iter().any(|v| !v.is_finite()) {
            break;
        }
        for i in 0..m {
            lambda[i] += dlambda[i];
        }
    }

    let mut delta_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for j in 0..n {
        let s = problem.a.col_ptr[j];
        let e = problem.a.col_ptr[j + 1];
        for k in s..e {
            let i = problem.a.row_ind[k];
            if i < m {
                delta_dd[j] = delta_dd[j] + TwoFloat::new_mul(problem.a.values[k], lambda[i]);
            }
        }
    }
    let delta: Vec<f64> = delta_dd.iter().map(|&v| f64::from(v)).collect();
    if delta.iter().any(|v| !v.is_finite()) {
        return;
    }

    let mut x_new = x.clone();
    for j in 0..n {
        x_new[j] -= delta[j];
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() {
            x_new[j] = x_new[j].max(lb);
        }
        if ub.is_finite() {
            x_new[j] = x_new[j].min(ub);
        }
    }

    // 成分相対化での max rel violation で改善判定 (abs では ill-scaled で見逃す)。
    let ax_new = match problem.a.mat_vec_mul(&x_new) {
        Ok(v) => v,
        Err(_) => return,
    };
    let mut max_rel_pre = 0.0_f64;
    let mut max_rel_post = 0.0_f64;
    for i in 0..m {
        let raw_pre = match problem.constraint_types[i] {
            ConstraintType::Eq => (ax[i] - problem.b[i]).abs(),
            ConstraintType::Ge => (problem.b[i] - ax[i]).max(0.0),
            ConstraintType::Le => (ax[i] - problem.b[i]).max(0.0),
        };
        let raw_post = match problem.constraint_types[i] {
            ConstraintType::Eq => (ax_new[i] - problem.b[i]).abs(),
            ConstraintType::Ge => (problem.b[i] - ax_new[i]).max(0.0),
            ConstraintType::Le => (ax_new[i] - problem.b[i]).max(0.0),
        };
        let scale_pre = 1.0 + ax[i].abs() + problem.b[i].abs();
        let scale_post = 1.0 + ax_new[i].abs() + problem.b[i].abs();
        let rel_pre = raw_pre / scale_pre;
        let rel_post = raw_post / scale_post;
        if rel_pre > max_rel_pre {
            max_rel_pre = rel_pre;
        }
        if rel_post > max_rel_post {
            max_rel_post = rel_post;
        }
    }
    if max_rel_post < max_rel_pre {
        *x = x_new;
    }
}

/// dual-only IR: x 固定で y のみ更新し r_d_free を厳密に 0 にする。
/// A_free^T δy = -r_d_free の最小ノルム解 δy = -A_free α、G α = r_d_free
/// (G = A_free^T A_free SPD)。active 変数の z は後段 refit_bound_duals_kkt で取り直す。
/// 戻り値: 1 (改善採用) / 0 (skip)。
fn try_dual_only_ir(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    target_pf: f64,
    deadline: Option<std::time::Instant>,
) -> usize {
    use crate::presolve::bound_contrib_at_var;
    use twofloat::TwoFloat;

    let m = problem.num_constraints;
    let trace = std::env::var("REFINE_KKT_TRACE").ok().as_deref() == Some("1");
    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
    };
    let kkt_pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );

    // G + δ·I の正則化。F64 round-off の cancellation を防ぐ最小値。
    // δ × ‖α‖ が new r_d_free の floor (典型 1e-12 × 1e2 = 1e-10、target 1e-6 を十分下回る)。
    let dual_ir_reg = std::env::var("DUAL_IR_REG")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1e-12);

    // 1. free 変数の特定 (active = bound 近傍 or A col 空)
    let free_eval_idx = collect_dual_recovery_free_columns(problem, result);
    let n_free_eval = free_eval_idx.len();
    if n_free_eval == 0 {
        if trace {
            eprintln!("DUAL_IR skip: n_free=0");
        }
        return 0;
    }

    // 2. r_d_free を DD で計算
    //    r_d[j] = c[j] + (A^T y)[j] + bound_contrib[j]
    //    free var の bound_contrib は通常 0 (z=0) だが念のため計算
    let mut r_d_eval = vec![0.0_f64; n_free_eval];
    let mut r_d_rel_eval = vec![0.0_f64; n_free_eval];
    let mut df_rel_pre = 0.0_f64;
    let mut df_abs_pre = 0.0_f64;
    let mut worst_idx = 0;
    let mut worst_qx = 0.0_f64;
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
        let bc = bound_contrib_at_var(&problem.bounds, &result.bound_duals, j);
        let r_d = qx_f + problem.c[j] + aty_f + bc;
        r_d_eval[fi] = r_d;
        let scale = 1.0 + qx_f.abs() + problem.c[j].abs() + aty_f.abs() + bc.abs();
        let rel = r_d.abs() / scale;
        r_d_rel_eval[fi] = rel;
        if rel > df_rel_pre {
            df_rel_pre = rel;
            worst_idx = j;
            worst_qx = qx_f;
        }
        if r_d.abs() > df_abs_pre {
            df_abs_pre = r_d.abs();
        }
    }
    if df_rel_pre < target_pf {
        if trace {
            eprintln!(
                "DUAL_IR skip: df_rel_pre={:.3e} < target {:.3e}",
                df_rel_pre, target_pf
            );
        }
        return 0;
    }
    if trace {
        eprintln!(
            "DUAL_IR pre: n_free_eval={} df_abs_max={:.3e} df_rel_max={:.3e} worst_j={} qx={:.3e}",
            n_free_eval, df_abs_pre, df_rel_pre, worst_idx, worst_qx
        );
    }

    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return 0;
    }

    let row_bounds = match compute_dual_recovery_row_bounds(problem, &result.solution) {
        Some(v) => v,
        None => return 0,
    };
    let (proj_lower, proj_upper) = (&row_bounds.0, &row_bounds.1);

    let Some((ax, row_abs_activity)) = compute_dual_recovery_row_activity(problem, &result.solution) else {
        return 0;
    };
    let Some((worst_j, active_rows)) = collect_dual_recovery_cluster_rows(
        problem,
        &free_eval_idx,
        &r_d_rel_eval,
        &ax,
        &row_abs_activity,
        target_pf,
    ) else {
        if trace {
            eprintln!("DUAL_IR skip: no active row cluster");
        }
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
        if trace {
            eprintln!("DUAL_IR skip: m_active=0");
        }
        return 0;
    }
    if trace {
        eprintln!(
            "DUAL_IR cluster_rows={}/{} worst_j={} seed_worst_j={}",
            m_active, m, worst_idx, worst_j
        );
    }

    let mut free_idx = Vec::new();
    for &j in &free_eval_idx {
        let touches_cluster = (problem.a.col_ptr[j]..problem.a.col_ptr[j + 1])
            .any(|k| active_row_pos[problem.a.row_ind[k]] != usize::MAX);
        if touches_cluster {
            free_idx.push(j);
        }
    }
    if !seed_rows.is_empty() && free_idx.len() * 2 > n_free_eval && active_rows.len() > seed_rows.len() {
        if trace {
            eprintln!(
                "DUAL_IR cluster fallback: expanded_rows={} expanded_free={} seed_rows={}",
                active_rows.len(),
                free_idx.len(),
                seed_rows.len()
            );
        }
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
        if trace {
            eprintln!("DUAL_IR skip: cluster has no free columns");
        }
        return 0;
    }
    if trace {
        eprintln!("DUAL_IR cluster_free={}/{}", n_free, n_free_eval);
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
    let mut df_abs_post = df_abs_pre;
    let mut total_dy_inf = 0.0_f64;
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
    let mut inner = 0usize;
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
        let mut gram = vec![0.0_f64; ulen * ulen];
        let mut rhs = vec![0.0_f64; ulen];
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
            let bc_j = bound_contrib_at_var(&problem.bounds, &tmp.bound_duals, j);
            let scale_j = (1.0 + qx_j.abs() + problem.c[j].abs() + aty_j.abs() + bc_j.abs()).max(1.0);
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
        let factor = match crate::linalg::ldl::factorize(&gram_csc) {
            Ok(f) => f,
            Err(e) => {
                if trace {
                    eprintln!("DUAL_IR factorize failed: {:?}", e);
                }
                break;
            }
        };
        let mut delta = vec![0.0_f64; ulen];
        factor.solve(&rhs, &mut delta);
        if delta.iter().any(|v| !v.is_finite()) {
            if trace {
                eprintln!("DUAL_IR inner={} solve NaN, abort", inner);
            }
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
        total_dy_inf = total_dy_inf.max(dy_inf);

        let mut accepted = false;
        let mut accepted_df_rel = df_rel_post;
        let mut accepted_df_abs = df_abs_post;
        let mut accepted_r_d_free = current_r_d_free.clone();
        let mut accepted_y_dd = y_dd.clone();
        let mut accepted_bound_duals = tmp.bound_duals.clone();
        let mut accepted_step_scale = 0.0_f64;
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
            let mut new_df_abs = 0.0_f64;
            for &j in &free_eval_idx {
                let mut qx = TwoFloat::from(0.0);
                for k in problem.q.col_ptr[j]..problem.q.col_ptr[j + 1] {
                    let row = problem.q.row_ind[k];
                    qx += TwoFloat::new_mul(problem.q.values[k], tmp.solution[row]);
                }
                let mut aty = TwoFloat::from(0.0);
                for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
                    let r = problem.a.row_ind[k];
                    aty = aty + y_dd_new[r] * problem.a.values[k];
                }
                let bc = bound_contrib_at_var(&problem.bounds, &bound_duals_new, j);
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
                if r_d.abs() > new_df_abs {
                    new_df_abs = r_d.abs();
                }
            }
            if new_df_rel <= df_rel_post + DUAL_IR_ACCEPT_REL_TOL * (1.0 + df_rel_post) {
                accepted = true;
                accepted_df_rel = new_df_rel;
                accepted_df_abs = new_df_abs;
                accepted_r_d_free = new_r_d_free;
                accepted_y_dd = y_dd_new;
                accepted_bound_duals = bound_duals_new;
                accepted_step_scale = step_scale;
                break;
            }
            let next_step_scale = step_scale * 0.5;
            if next_step_scale == step_scale {
                break;
            }
            step_scale = next_step_scale;
        }
        if !accepted {
            if trace {
                eprintln!(
                    "DUAL_IR inner={} regression, breaking (rel {:.3e} -> rejected all backtracks)",
                    inner, df_rel_post
                );
            }
            break;
        }

        let rel_improvement = (df_rel_post - accepted_df_rel).max(0.0);
        let progress_ratio = if df_rel_post > 0.0 {
            rel_improvement / df_rel_post
        } else {
            0.0
        };
        if accepted_iters > 0 && progress_ratio <= DUAL_IR_MIN_PROGRESS_RATIO {
            if trace {
                eprintln!(
                    "DUAL_IR inner={} stagnated: df_rel {:.3e} -> {:.3e} ratio={:.3e}",
                    inner, df_rel_post, accepted_df_rel, progress_ratio
                );
            }
            break;
        }

        y_dd = accepted_y_dd;
        for i in 0..m {
            tmp.dual_solution[i] = f64::from(y_dd[i]);
        }
        tmp.bound_duals = accepted_bound_duals;
        current_r_d_free = accepted_r_d_free;
        df_rel_post = accepted_df_rel;
        df_abs_post = accepted_df_abs;
        accepted_iters += 1;
        inner += 1;
        if trace && accepted_step_scale < 1.0 {
            eprintln!(
                "DUAL_IR inner={} accepted with step_scale={:.3e}",
                inner, accepted_step_scale
            );
        }
        // 早期 break: target を達成したら終了
        if df_rel_post < target_pf {
            break;
        }
    }
    for i in 0..m {
        tmp.dual_solution[i] = f64::from(y_dd[i]);
    }
    // 採用判定前に z を取り直す (y-only 更新を stale な z で評価すると改善候補を落とす)。
    refit_bound_duals_kkt(problem, &mut tmp);

    let kkt_post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &tmp.solution,
        &tmp.dual_solution,
        &tmp.bound_duals,
    );
    if trace {
        eprintln!(
            "DUAL_IR cluster_free={} df_abs {:.3e}->{:.3e} df_rel {:.3e}->{:.3e} dy_inf={:.3e} iters={}",
            n_free, df_abs_pre, df_abs_post, df_rel_pre, df_rel_post, total_dy_inf, accepted_iters
        );
        eprintln!("DUAL_IR kkt {:.3e}->{:.3e}", kkt_pre, kkt_post);
    }
    if df_rel_post < df_rel_pre && kkt_post <= kkt_pre {
        *result = tmp;
        accepted_iters
    } else {
        if trace {
            eprintln!(
                "DUAL_IR rejected: df_improved={} kkt_safe={}",
                df_rel_post < df_rel_pre,
                kkt_post <= kkt_pre
            );
        }
        0
    }
}

fn run_dual_recovery_postprocess(
    problem: &QpProblem,
    view: &crate::qp::ipm_solver::outcome::ProblemView<'_>,
    result: &mut crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
    trace: bool,
) -> f64 {
    let pre_cleanup = result.clone();
    let kkt_before_cleanup = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    zero_inactive_inequality_duals(problem, result);
    if trace {
        let kkt_after_zero = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        eprintln!(
            "DUAL_IR outer: after zero_inactive kkt {:.3e}",
            kkt_after_zero
        );
    }
    project_duals_from_singleton_columns(problem, result);
    if trace {
        let kkt_after_singleton = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        eprintln!(
            "DUAL_IR outer: after singleton projection kkt {:.3e}",
            kkt_after_singleton
        );
    }
    refine_dual_projected_gradient(problem, result, deadline);
    if trace {
        let kkt_after_pg = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        eprintln!(
            "DUAL_IR outer: after projected gradient kkt {:.3e}",
            kkt_after_pg
        );
    }
    refine_dual_worst_active_block(problem, result, deadline);
    if trace {
        let kkt_after_block = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        eprintln!(
            "DUAL_IR outer: after local block kkt {:.3e}",
            kkt_after_block
        );
    }

    let pre_z = result.bound_duals.clone();
    let pre_refit_kkt = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    refit_bound_duals_kkt(problem, result);
    let post_refit_kkt = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    if post_refit_kkt > pre_refit_kkt {
        result.bound_duals = pre_z;
        if trace {
            eprintln!(
                "DUAL_IR z-refit rejected: kkt {:.3e} -> {:.3e}",
                pre_refit_kkt, post_refit_kkt
            );
        }
    } else if trace {
        eprintln!(
            "DUAL_IR z-refit accepted: kkt {:.3e} -> {:.3e}",
            pre_refit_kkt, post_refit_kkt
        );
    }

    let kkt_after_cleanup = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    if kkt_after_cleanup > kkt_before_cleanup {
        if trace {
            eprintln!(
                "DUAL_IR cleanup reverted: kkt {:.3e} -> {:.3e}",
                kkt_before_cleanup, kkt_after_cleanup
            );
        }
        *result = pre_cleanup;
        kkt_before_cleanup
    } else {
        kkt_after_cleanup
    }
}

/// 戻り値: 採用された refinement iter 数 (0 = no-op)
pub(crate) fn refine_kkt_iterative(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
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
    };
    let mut prev_kkt = kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    let start_kkt = prev_kkt;
    let mut best_kkt = prev_kkt;
    let mut best_result = result.clone();
    let trace = std::env::var("REFINE_KKT_TRACE").ok().as_deref() == Some("1");
    for _outer in 0..max_iters.max(1) {
        let mut outer_made_progress = false;
        let n_dual = try_dual_only_ir(problem, result, target_pf, deadline);
        if n_dual > 0 {
            n_dual_total += n_dual;
            outer_made_progress = true;
            let kkt_after_dual_ir = kkt_residual_rel(
                &view,
                &result.solution,
                &result.dual_solution,
                &result.bound_duals,
            );
            if trace {
                eprintln!(
                    "DUAL_IR outer: after try_dual_only_ir kkt {:.3e}",
                    kkt_after_dual_ir
                );
            }
            let _ = run_dual_recovery_postprocess(problem, &view, result, deadline, trace);
        } else {
            let pre_cleanup_kkt = kkt_residual_rel(
                &view,
                &result.solution,
                &result.dual_solution,
                &result.bound_duals,
            );
            let post_cleanup_kkt =
                run_dual_recovery_postprocess(problem, &view, result, deadline, trace);
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
        if trace {
            eprintln!(
                "DUAL_IR outer: best_kkt {:.3e} (start {:.3e})",
                best_kkt, start_kkt
            );
        }
        if best_kkt < target_pf || deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return n_dual_total;
        }
    }
    // dual-only で改善できない / 不十分なら saddle-point IR に fall-through。

    // K = [Q+δp·I, A^T; A, -δd·I] の対角正則化。十分小さく IR で eps·‖K‖ まで refine 可。
    // env REFINE_KKT_DELTA で上書き可。
    const DELTA_P_DEFAULT: f64 = 1e-10;
    const DELTA_D_DEFAULT: f64 = 1e-10;
    let (delta_p, delta_d) = match std::env::var("REFINE_KKT_DELTA")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
    {
        Some(v) if v > 0.0 => (v, v),
        _ => (DELTA_P_DEFAULT, DELTA_D_DEFAULT),
    };

    let sigma_zero = vec![0.0_f64; m];
    let mut k_mat = crate::qp::ipm_core::kkt::build_augmented_system(
        &problem.q,
        &problem.a,
        &sigma_zero,
        delta_p,
        delta_d,
    );

    let trace_pre = std::env::var("REFINE_KKT_TRACE").ok().as_deref() == Some("1");
    let diag_on = std::env::var("REFINE_KKT_DIAG").ok().as_deref() == Some("1");

    // bound-active 変数の dx を K 対角 penalty で抑制 (近似 active set fix)。
    // K は bound 制約を陽に持たず active 変数にも dx 生成、bound 超過後の clip で
    // K·u=-r を破り reject される。env REFINE_KKT_REDUCED=0 で無効化可能。
    const ACTIVE_TOL: f64 = 1e-8;
    const ACTIVE_PENALTY_RATIO: f64 = 1e8;
    let active_fix_enabled = std::env::var("REFINE_KKT_REDUCED").ok().as_deref() != Some("0");
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
        if (trace_pre || diag_on) && penalized > 0 {
            eprintln!("REFINE_KKT bound-active fix: penalized {} vars (PENALTY={:.2e}, K_diag_max={:.2e})",
                penalized, active_penalty, k_diag_max);
        }
    }
    if trace_pre {
        eprintln!(
            "REFINE_KKT pre-factorize: n={} m={} K_nnz={} delta_p={:.1e} delta_d={:.1e}",
            n,
            m,
            k_mat.values.len(),
            delta_p,
            delta_d
        );
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
    // SingularOrIndefinite なら δ を段階的に上げて再試行 (factorize 成立する最小 δ を採用)。
    // deadline 必須: 大規模 K factorize は単発 80s 級に達する。
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
                if trace_pre || diag_on {
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
                        if trace_pre || diag_on {
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
        if (trace_pre || diag_on) && retry_count > 0 && result_factor.is_some() {
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

    // FX (lb≈ub) と EmptyCol は bound_dual=0 慣例で stationarity 評価から除外。
    // 含めると orig 空間で huge cancellation noise が出て IR が壊れる。
    const FX_TOL_REFINE: f64 = 1e-12;
    let exclude_var: Vec<bool> = (0..n)
        .map(|j| {
            let (lb, ub) = problem.bounds[j];
            if lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL_REFINE {
                return true;
            }
            if problem.a.col_ptr.len() > j + 1
                && problem.a.col_ptr[j + 1] - problem.a.col_ptr[j] == 0
            {
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
    // residual を f64 limit 以下に精密化。LDL solve は f64 のまま。env REFINE_KKT_DD=0 で f64 fallback。
    let dd_mode = std::env::var("REFINE_KKT_DD").ok().as_deref() != Some("0");
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
                    qx_dd[row] = qx_dd[row] + TwoFloat::new_mul(v, xv);
                }
            }
            let mut aty_dd: Vec<TwoFloat> = vec![zero_dd; n];
            for col in 0..n {
                let cs = problem.a.col_ptr[col];
                let ce = problem.a.col_ptr[col + 1];
                for k in cs..ce {
                    let row = problem.a.row_ind[k];
                    let v = problem.a.values[k];
                    aty_dd[col] = aty_dd[col] + TwoFloat::new_mul(v, y[row]);
                }
            }
            let mut r_d = vec![0.0_f64; n];
            let mut max_qx = 0.0_f64;
            let mut max_c = 0.0_f64;
            let mut max_aty = 0.0_f64;
            let mut max_bnd = 0.0_f64;
            for j in 0..n {
                if exclude_var[j] {
                    continue;
                }
                let bc = bound_contrib_at_var(&problem.bounds, z, j);
                let r = qx_dd[j] + TwoFloat::from(problem.c[j]) + aty_dd[j] + TwoFloat::from(bc);
                r_d[j] = f64::from(r);
                max_qx = max_qx.max(f64::from(qx_dd[j]).abs());
                max_c = max_c.max(problem.c[j].abs());
                max_aty = max_aty.max(f64::from(aty_dd[j]).abs());
                max_bnd = max_bnd.max(bc.abs());
            }
            let mut ax_dd: Vec<TwoFloat> = vec![zero_dd; m];
            for col in 0..n {
                let cs = problem.a.col_ptr[col];
                let ce = problem.a.col_ptr[col + 1];
                for k in cs..ce {
                    let row = problem.a.row_ind[k];
                    let v = problem.a.values[k];
                    ax_dd[row] = ax_dd[row] + TwoFloat::new_mul(v, x[col]);
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
            let _ = max_qx;
            let _ = max_c;
            let _ = max_aty;
            let _ = max_bnd;
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
    let (_, _, pre_pf, pre_df, pre_pf_rel, pre_df_rel) = if dd_mode {
        compute_residuals_dd(&result.solution, &result.dual_solution, &pre_z)
    } else {
        compute_residuals(&result.solution, &result.dual_solution, &pre_z)
    };
    let trace = std::env::var("REFINE_KKT_TRACE").ok().as_deref() == Some("1");
    if trace {
        eprintln!(
            "REFINE_KKT entry: n={} m={} pre_pf={:.3e} pre_df={:.3e} target_pf={:.3e} dd_mode={}",
            n, m, pre_pf, pre_df, target_pf, dd_mode
        );
    }
    if pre_pf_rel < target_pf && pre_df_rel < target_pf {
        if trace {
            eprintln!(
                "REFINE_KKT skip: pre_pf_rel={:.3e} pre_df_rel={:.3e} both < target_pf",
                pre_pf_rel, pre_df_rel
            );
        }
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
            if trace {
                eprintln!("REFINE_KKT iter={} deadline reached", iter);
            }
            break;
        }
        let (r_d, r_p, pf_abs_cur, df_abs_cur, pf_cur, df_cur) = if dd_mode {
            compute_residuals_dd(&result.solution, &result.dual_solution, &result.bound_duals)
        } else {
            compute_residuals(&result.solution, &result.dual_solution, &result.bound_duals)
        };
        if pf_cur < target_pf && df_cur < target_pf {
            if trace {
                eprintln!(
                    "REFINE_KKT iter={} early: pf_rel={:.3e} df_rel={:.3e} both < target",
                    iter, pf_cur, df_cur
                );
            }
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
            if trace {
                eprintln!("REFINE_KKT iter={} solve produced NaN", iter);
            }
            break;
        }

        let dx_inf: f64 = sol[..n].iter().fold(0.0, |a, &v| a.max(v.abs()));
        let dy_inf: f64 = sol[n..].iter().fold(0.0, |a, &v| a.max(v.abs()));

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

        if trace {
            eprintln!("REFINE_KKT iter={} pf_rel={:.3e}->{:.3e} df_rel={:.3e}->{:.3e} dx_inf={:.3e} dy_inf={:.3e} clip={:.3e}",
                iter, pf_cur, pf_new, df_cur, df_new, dx_inf, dy_inf, clip_amt);
        }

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
            if trace {
                eprintln!("REFINE_KKT iter={} REJECTED (progress={} pf_safe={} df_safe={} score:{:.3e}->{:.3e})",
                    iter, progress, pf_safe, df_safe, score_cur, score_new);
            }
            break;
        }
    }

    accepted
}

/// x, y を不変としたまま bound_duals を KKT stationarity から再計算 (postsolve 後の 0 埋め解消)。
/// bound_contrib = -z_lb + z_ub = -(Qx+c+A^T y) より符号で z_lb/z_ub を候補化、per-col guard 採用。
pub(crate) fn refit_bound_duals_kkt(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
) {
    let n = problem.num_vars;
    if result.solution.len() != n {
        return;
    }
    use twofloat::TwoFloat;
    let x = &result.solution;
    // Q*x と A^T*y は DD で積算 (f64 cancellation 防止)。
    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = x[col];
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            qx_dd[problem.q.row_ind[k]] =
                qx_dd[problem.q.row_ind[k]] + TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let qx: Vec<f64> = qx_dd.iter().map(|&v| f64::from(v)).collect();
    let aty: Vec<f64> = if problem.a.nrows > 0 && !result.dual_solution.is_empty() {
        let mut acc: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                acc[col] = acc[col]
                    + TwoFloat::new_mul(
                        problem.a.values[k],
                        result.dual_solution[problem.a.row_ind[k]],
                    );
            }
        }
        acc.iter().map(|&v| f64::from(v)).collect()
    } else {
        vec![0.0_f64; n]
    };

    let n_lb = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    let n_ub = problem
        .bounds
        .iter()
        .filter(|&&(_, ub)| ub.is_finite())
        .count();
    if n_lb + n_ub == 0 {
        return;
    }

    let mut new_bd = vec![0.0_f64; n_lb + n_ub];
    // 候補値: target = -(Qx+c+Aty) の符号で z_lb/z_ub を提示。後段 per-col guard で採用判定。
    let mut lb_idx = 0_usize;
    let mut ub_idx = n_lb;
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        let target = -(qx[j] + problem.c[j] + aty[j]);
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();

        if lb_finite && ub_finite {
            // FX (lb==ub) は postsolve 慣例で 0 埋め、KKT 評価からも除外。
            if (lb - ub).abs() >= FX_TOL {
                if target > 0.0 {
                    new_bd[ub_idx] = target;
                } else {
                    new_bd[lb_idx] = -target;
                }
            }
            lb_idx += 1;
            ub_idx += 1;
        } else if lb_finite {
            new_bd[lb_idx] = (-target).max(0.0);
            lb_idx += 1;
        } else if ub_finite {
            new_bd[ub_idx] = target.max(0.0);
            ub_idx += 1;
        }
    }

    // per-col guard: col 単位で改善時のみ採用 (max ベース guard は 1 col 悪化で全 reject になる)。
    let pre_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    let post_contrib = compute_bound_contrib(&problem.bounds, &new_bd, n);
    let mut accepted_bd = result.bound_duals.clone();
    if accepted_bd.len() < new_bd.len() {
        accepted_bd.resize(new_bd.len(), 0.0);
    }
    let mut updated_lb = 0usize;
    let mut updated_ub = 0usize;
    let mut lb_slot = 0usize;
    let mut ub_slot = n_lb;
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        let is_fx = lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL;
        let r_pre = if is_fx {
            0.0
        } else {
            (qx[j] + problem.c[j] + aty[j] + pre_contrib[j]).abs()
        };
        let r_post = if is_fx {
            0.0
        } else {
            (qx[j] + problem.c[j] + aty[j] + post_contrib[j]).abs()
        };
        let take_new = !is_fx && r_post <= r_pre;
        if lb.is_finite() {
            if take_new && lb_slot < new_bd.len() {
                if accepted_bd[lb_slot] != new_bd[lb_slot] {
                    updated_lb += 1;
                }
                accepted_bd[lb_slot] = new_bd[lb_slot];
            }
            lb_slot += 1;
        }
        if ub.is_finite() {
            if take_new && ub_slot < new_bd.len() {
                if accepted_bd[ub_slot] != new_bd[ub_slot] {
                    updated_ub += 1;
                }
                accepted_bd[ub_slot] = new_bd[ub_slot];
            }
            ub_slot += 1;
        }
    }
    if std::env::var("REFIT_BD_TRACE").ok().as_deref() == Some("1") {
        eprintln!(
            "REFIT_BD per-col: updated_lb={} updated_ub={} (n={})",
            updated_lb, updated_ub, n
        );
    }
    result.bound_duals = accepted_bd;
}

/// IRLS で y を componentwise rel 最小化 (L∞ 漸近) する。各 iter で残差 rel に応じて
/// 重みを上げて weighted LSQ (A·diag(w)·A^T) y = A·diag(w)·target を解く。
/// A の列 k を √w_k でスケールして既存 build_aat_upper_csc 経路を流用。
pub(crate) fn refine_dual_lsq_irls(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    eps_target: f64,
    max_iters: usize,
    deadline: Option<std::time::Instant>,
) {
    use twofloat::TwoFloat;
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if m == 0 || result.solution.len() != n {
        return;
    }
    if n + m > LSQ_DUAL_SIZE_LIMIT {
        return;
    }
    if result.dual_solution.len() != m {
        return;
    }

    let zero_dd = TwoFloat::from(0.0);

    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = result.solution[col];
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            qx_dd[problem.q.row_ind[k]] =
                qx_dd[problem.q.row_ind[k]] + TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let qx: Vec<f64> = qx_dd.iter().map(|&v| f64::from(v)).collect();
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    let target: Vec<f64> = (0..n)
        .map(|j| -(qx[j] + problem.c[j] + bound_contrib[j]))
        .collect();

    let exclude: Vec<bool> = (0..n)
        .map(|j| {
            let (lb, ub) = problem.bounds[j];
            if lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL {
                return true;
            }
            problem.a.col_ptr.len() > j + 1 && problem.a.col_ptr[j + 1] - problem.a.col_ptr[j] == 0
        })
        .collect();

    let compute_aty = |y: &[f64]| -> Vec<f64> {
        let mut acc: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                let row = problem.a.row_ind[k];
                acc[col] = acc[col] + TwoFloat::new_mul(problem.a.values[k], y[row]);
            }
        }
        acc.iter().map(|&v| f64::from(v)).collect()
    };

    let max_rel_with_aty = |aty_v: &[f64]| -> f64 {
        let mut max_rel = 0.0_f64;
        for j in 0..n {
            if exclude[j] {
                continue;
            }
            let r = qx[j] + problem.c[j] + aty_v[j] + bound_contrib[j];
            let scale =
                1.0 + qx[j].abs() + problem.c[j].abs() + aty_v[j].abs() + bound_contrib[j].abs();
            let rel = r.abs() / scale;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        max_rel
    };

    let mut y_curr = result.dual_solution.clone();
    let initial_aty = compute_aty(&y_curr);
    let initial_max_rel = max_rel_with_aty(&initial_aty);
    if initial_max_rel < eps_target {
        return;
    }

    let mut best_y = y_curr.clone();
    let mut best_max_rel = initial_max_rel;
    let mut prev_max_rel = initial_max_rel;

    /// 単一成分の重み上限 (rel/eps)。> 1e4 で他成分悪化との oscillation が出る。
    const MAX_WEIGHT_RATIO: f64 = 1e4;
    const STAGNATE_RATIO: f64 = 0.95;

    for irls_iter in 0..max_iters {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }

        // weight = (rel/eps)² ( LSQ 内部の √w 倍に対し二乗で componentwise 効果を得る )。
        let aty_v = compute_aty(&y_curr);
        let mut weights: Vec<f64> = vec![1.0; n];
        for j in 0..n {
            if exclude[j] {
                continue;
            }
            let r = qx[j] + problem.c[j] + aty_v[j] + bound_contrib[j];
            let scale =
                1.0 + qx[j].abs() + problem.c[j].abs() + aty_v[j].abs() + bound_contrib[j].abs();
            let rel = r.abs() / scale;
            if rel > eps_target {
                let ratio = (rel / eps_target).min(MAX_WEIGHT_RATIO);
                weights[j] = ratio * ratio;
            }
        }

        let mut a_scaled = problem.a.clone();
        for k in 0..n {
            let s = weights[k].sqrt();
            if (s - 1.0).abs() < 1e-15 {
                continue;
            }
            let cs = a_scaled.col_ptr[k];
            let ce = a_scaled.col_ptr[k + 1];
            for idx in cs..ce {
                a_scaled.values[idx] *= s;
            }
        }

        let aat_w = match build_aat_upper_csc(&a_scaled, n, m) {
            Some(mat) => mat,
            None => break,
        };
        let factor = match crate::linalg::ldl::factorize(&aat_w) {
            Ok(f) => f,
            Err(_) => break,
        };

        let mut rhs_dd: Vec<TwoFloat> = vec![zero_dd; m];
        for col in 0..n {
            let wt = weights[col] * target[col];
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                let row = problem.a.row_ind[k];
                rhs_dd[row] = rhs_dd[row] + TwoFloat::new_mul(problem.a.values[k], wt);
            }
        }
        let rhs: Vec<f64> = rhs_dd.iter().map(|&v| f64::from(v)).collect();

        let mut y_new = vec![0.0_f64; m];
        factor.solve(&rhs, &mut y_new);
        if y_new.iter().any(|v| !v.is_finite()) {
            break;
        }

        let aty_new = compute_aty(&y_new);
        let new_max_rel = max_rel_with_aty(&aty_new);

        if new_max_rel < best_max_rel {
            best_y = y_new.clone();
            best_max_rel = new_max_rel;
        }

        if best_max_rel < eps_target {
            break;
        }
        if irls_iter > 0 && new_max_rel >= prev_max_rel * STAGNATE_RATIO {
            break;
        }
        prev_max_rel = new_max_rel;
        y_curr = y_new;
    }

    if best_max_rel < initial_max_rel {
        result.dual_solution = best_y;
    }
}

/// KKT bound 寄与 (-z_lb + z_ub)。bound_duals layout は [lb 有限の z_lb; ub 有限の z_ub]。
fn compute_bound_contrib(bounds: &[(f64, f64)], bound_duals: &[f64], n: usize) -> Vec<f64> {
    let mut contrib = vec![0.0_f64; n];
    if bound_duals.is_empty() {
        return contrib;
    }
    let mut idx = 0usize;
    for (j, &(lb, _)) in bounds.iter().enumerate() {
        if lb.is_finite() && idx < bound_duals.len() {
            contrib[j] -= bound_duals[idx];
            idx += 1;
        }
    }
    for (j, &(_, ub)) in bounds.iter().enumerate() {
        if ub.is_finite() && idx < bound_duals.len() {
            contrib[j] += bound_duals[idx];
            idx += 1;
        }
    }
    contrib
}

/// BTreeMap ノードあたり実測バイト数 (key 16 + value 8 + node overhead)。memory budget の係数。
const AAT_BUILD_BYTES_PER_ENTRY: u128 = 80;

/// A·A^T (m×m, 上三角 CSC) + 対角 ε 正則化 (rank-deficient でも factorize 可)。
/// nnz upper bound × 80 B が memory_budget 超なら None を返し caller は skip。
pub(crate) fn build_aat_upper_csc(a: &CscMatrix, n: usize, m: usize) -> Option<CscMatrix> {
    use std::collections::BTreeMap;
    let m_u = m as u128;
    let mut col_pair_sum: u128 = 0;
    for k in 0..n {
        let c_k = (a.col_ptr[k + 1] - a.col_ptr[k]) as u128;
        col_pair_sum = col_pair_sum.saturating_add(c_k.saturating_mul(c_k + 1) / 2);
    }
    let nnz_upper_bound = (m_u.saturating_mul(m_u + 1) / 2).min(col_pair_sum);
    let bytes_estimate = nnz_upper_bound.saturating_mul(AAT_BUILD_BYTES_PER_ENTRY);
    if bytes_estimate > crate::linalg::kkt_solver::memory_budget_bytes() as u128 {
        return None;
    }
    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for k in 0..n {
        let start = a.col_ptr[k];
        let end = a.col_ptr[k + 1];
        let cols_in_k: Vec<(usize, f64)> =
            (start..end).map(|p| (a.row_ind[p], a.values[p])).collect();
        for (idx_a, &(i, v_i)) in cols_in_k.iter().enumerate() {
            for &(j, v_j) in &cols_in_k[idx_a..] {
                let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
                *acc.entry((hi, lo)).or_insert(0.0) += v_i * v_j;
            }
        }
    }
    let max_diag = (0..m)
        .filter_map(|i| acc.get(&(i, i)).copied())
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let reg = AAT_REG_FACTOR * max_diag;
    for i in 0..m {
        *acc.entry((i, i)).or_insert(0.0) += reg;
    }
    let mut col_ptr = vec![0_usize; m + 1];
    let mut row_ind: Vec<usize> = Vec::with_capacity(acc.len());
    let mut values: Vec<f64> = Vec::with_capacity(acc.len());
    for ((col, row), val) in acc {
        row_ind.push(row);
        values.push(val);
        col_ptr[col + 1] = row_ind.len();
    }
    for i in 1..=m {
        if col_ptr[i] < col_ptr[i - 1] {
            col_ptr[i] = col_ptr[i - 1];
        }
    }
    Some(CscMatrix {
        col_ptr,
        row_ind,
        values,
        nrows: m,
        ncols: m,
    })
}

#[deprecated(since = "0.1.0", note = "use `solve_qp_with` instead")]
pub fn solve_qp_with_options(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    solve_qp_with(problem, options)
}

/// Warm-start 付きで QP を解く (qpOASES `hotstart` 相当)。
///
/// IPM は warm_start 未対応 (initial_point/active_set 共に無視)。solve_qp_with に委譲。
pub fn solve_qp_warm(
    problem: &QpProblem,
    _warm_start: &QpWarmStart,
    options: &SolverOptions,
) -> SolverResult {
    solve_qp_with(problem, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::SolveStatus;
    use crate::sparse::CscMatrix;

    const EPS: f64 = 1e-2;

    fn assert_close(a: f64, b: f64, eps: f64, name: &str) {
        assert!(
            (a - b).abs() < eps,
            "{}: expected {:.8}, got {:.8} (diff={:.2e})",
            name,
            b,
            a,
            (a - b).abs()
        );
    }

    /// min x²+y² s.t. x+y ≥ 1 → x*=y*=0.5, obj=0.5
    #[test]
    fn test_basic_qp_2vars() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 0.5, EPS, "x[0]");
        assert_close(result.solution[1], 0.5, EPS, "x[1]");
        assert_close(result.objective, 0.5, EPS, "obj");
        assert!(result.bound_duals.is_empty());
        assert_eq!(result.dual_solution.len(), 1);
    }

    /// min x²+y² s.t. x+y=1 → x*=y*=0.5
    #[test]
    fn test_qp_equality_constraint() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, -1.0, -1.0], 2, 2)
                .unwrap();
        let b = vec![1.0, -1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 0.5, EPS, "x[0]");
        assert_close(result.solution[1], 0.5, EPS, "x[1]");
        assert_close(result.objective, 0.5, EPS, "obj");
    }

    /// Q=0 (LP): min x+2y s.t. x,y≥0, x+y≤4, 2x+y≤6 → obj=0
    #[test]
    fn test_qp_degenerate_lp_case() {
        let n = 2;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0, 2.0];
        let a = CscMatrix::from_triplets(
            &[0, 1, 2, 2, 3, 3],
            &[0, 1, 0, 1, 0, 1],
            &[-1.0, -1.0, 1.0, 1.0, 2.0, 1.0],
            4,
            2,
        )
        .unwrap();
        let b = vec![0.0, 0.0, 4.0, 6.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.objective, 0.0, EPS, "obj");
    }

    /// 制約なし: min (x-3)²+(y-4)² → x*=3, y*=4
    #[test]
    fn test_qp_unconstrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-6.0, -8.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 3.0, EPS, "x[0]");
        assert_close(result.solution[1], 4.0, EPS, "x[1]");
        assert_close(result.objective, -25.0, EPS, "obj");
    }

    /// warm-start: IPM は warm-start を無視するため同一解が返る。
    #[test]
    fn test_warm_start_consistency() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a.clone(), b.clone(), bounds.clone()).unwrap();
        let problem2 = QpProblem::new_all_le(
            CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap(),
            vec![0.0, 0.0],
            a,
            b,
            bounds,
        )
        .unwrap();

        let result1 = solve_qp(&problem);
        assert_eq!(result1.status, SolveStatus::Optimal);

        let ws = crate::qp::QpWarmStart {
            initial_active_set: vec![],
            initial_point: Some(result1.solution.clone()),
        };
        let result2 = solve_qp_warm(&problem2, &ws, &SolverOptions::default());

        assert_eq!(result2.status, SolveStatus::Optimal);
        assert_close(result2.solution[0], 0.5, EPS, "x[0]");
        assert_close(result2.solution[1], 0.5, EPS, "x[1]");
    }

    /// 矛盾制約 (x≥1 ∧ x≤0) → Infeasible
    #[test]
    fn test_qp_infeasible() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[-1.0, 1.0], 2, 1).unwrap();
        let b = vec![-1.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Infeasible);
    }

    /// Markowitz 平均分散ポートフォリオ。
    #[test]
    fn test_qp_portfolio_markowitz() {
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[2.0, 2.0, 2.0], 3, 3).unwrap();
        let c = vec![0.0, 0.0, 0.0];
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
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal);
        let w_sum = result.solution[0] + result.solution[1] + result.solution[2];
        assert_close(w_sum, 1.0, EPS, "w_sum");
        assert_close(result.solution[0], 1.0 / 3.0, EPS, "w[0]");
        assert_close(result.solution[1], 1.0 / 3.0, EPS, "w[1]");
        assert_close(result.solution[2], 1.0 / 3.0, EPS, "w[2]");
        assert_close(result.objective, 1.0 / 3.0, EPS, "obj");
    }

    /// Least Squares。
    #[test]
    fn test_qp_least_squares() {
        let q =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[10.0, 8.0, 8.0, 10.0], 2, 2)
                .unwrap();
        let c = vec![-28.0, -26.0];
        let a = CscMatrix::new(0, 2);
        let b_vec = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b_vec, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 2.0, EPS, "x[0]");
        assert_close(result.solution[1], 1.0, EPS, "x[1]");
        assert_close(result.objective, -41.0, EPS, "obj");
    }

    /// Q=0 → LP 退化。
    #[test]
    fn test_qp_degenerate_to_lp() {
        let n = 2;
        let q = CscMatrix::new(n, n);
        let c = vec![2.0, 1.0];
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[-1.0, -1.0, -1.0, -1.0],
            3,
            2,
        )
        .unwrap();
        let b = vec![-1.0, 0.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 0.0, EPS, "x[0]");
        assert_close(result.solution[1], 1.0, EPS, "x[1]");
        assert_close(result.objective, 1.0, EPS, "obj");
    }

    /// 等式 + 不等式 mixed。
    #[test]
    fn test_qp_mixed_constraints() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-2.0, -4.0];
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2],
            &[0, 1, 0, 1, 0],
            &[1.0, 1.0, -1.0, -1.0, -1.0],
            3,
            2,
        )
        .unwrap();
        let b = vec![2.0, -2.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 0.5, EPS, "x[0]");
        assert_close(result.solution[1], 1.5, EPS, "x[1]");
        assert_close(result.objective, -4.5, EPS, "obj");
    }

    /// Box: 上界 active。
    #[test]
    fn test_qp_box_constrained_upper_bound() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-4.0, -4.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert!(matches!(
            result.status,
            SolveStatus::Optimal | SolveStatus::SuboptimalSolution
        ), "got {:?}", result.status);
        assert_close(result.solution[0], 1.0, EPS, "x[0]");
        assert_close(result.solution[1], 1.0, EPS, "x[1]");
        assert_close(result.objective, -6.0, EPS, "obj");
    }

    /// Box: 下界 active。
    #[test]
    fn test_qp_box_constrained_lower_bound() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![4.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 0.0, EPS, "x[0]");
        assert_close(result.solution[1], 0.0, EPS, "x[1]");
        assert_close(result.objective, 0.0, EPS, "obj");
    }

    /// timeout=0 で Timeout or Optimal。
    #[test]
    fn test_timeout_returns_timeout_status() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(0.0),
            ..Default::default()
        };

        let result = solve_qp_with(&problem, &opts);
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "got {:?}", result.status
        );
    }

    /// 強制 IPM (小規模)。
    #[test]
    fn test_force_ipm_small() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.solution[0] - 0.5).abs() < 1e-4);
        assert!((result.solution[1] - 0.5).abs() < 1e-4);
        assert!((result.objective - 0.5).abs() < 1e-4);
    }

    /// parallel feature 有効時の IPPMM dispatch smoke test
    #[cfg(feature = "parallel")]
    #[test]
    fn test_concurrent_solver_basic() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.solution[0] - 0.5).abs() < EPS);
        assert!((result.solution[1] - 0.5).abs() < EPS);
        assert!((result.objective - 0.5).abs() < EPS);
    }

    /// 大行ノルム制約での Ruiz scaling 耐性 (元空間で pfeas 評価)。
    #[test]
    fn test_presolve_pfeas_large_row_norm() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1000.0], 1, 1).unwrap();
        let b = vec![500.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal);
        let ax = problem.a.mat_vec_mul(&result.solution).unwrap();
        let pfeas = ax
            .iter()
            .zip(problem.b.iter())
            .map(|(&ax_i, &b_i)| (ax_i - b_i).max(0.0))
            .fold(0.0_f64, f64::max);
        let norm_b = problem
            .b
            .iter()
            .fold(0.0_f64, |a, &bi| a.max(bi.abs()))
            .max(1.0);
        let eps = opts.ipm_eps();
        assert!(pfeas < eps * (1.0 + norm_b), "pfeas={pfeas:.2e}");
    }

    /// bounds 付き問題で post-postsolve bfeas check が誤降格しないこと。
    #[test]
    fn test_presolve_bfeas_bounded_problem() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal);
        let x = result.solution[0];
        assert!(x >= -1e-4, "x >= lb=0, got {x}");
        assert!(x <= 1.0 + 1e-4, "x <= ub=1, got {x}");
    }

    /// 正常解で post-postsolve pfeas+bfeas check が Optimal を維持。
    #[test]
    fn test_presolve_pfeas_bfeas_ok() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let b = vec![1.0];
        let bounds = vec![(0.0_f64, 0.5_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal);
    }

    /// presolve=true で post-unscaling check が正常問題に影響しないこと。
    #[test]
    fn test_solve_qp_with_presolve_path_verified() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();
        assert!(opts.presolve);
        let result = solve_qp_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal);
        let eps = 1e-3_f64;
        assert!((result.solution[0] - 0.5).abs() < eps);
        assert!((result.solution[1] - 0.5).abs() < eps);
    }

    /// 不定 Q (対角負値) → 慣性修正 IPM で NonConvex を返さないこと。
    #[test]
    fn test_qp_nonconvex_indefinite_q() {
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1.0, 1.0, 1.0], 3, 3).unwrap();
        let c = vec![0.0, 0.0, 0.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 3).unwrap();
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert!(
            !matches!(result.status, SolveStatus::NonConvex(_)),
            "got {:?}", result.status
        );
        assert!(
            matches!(
                result.status,
                SolveStatus::LocallyOptimal | SolveStatus::Optimal
                | SolveStatus::Unbounded | SolveStatus::Timeout
                | SolveStatus::SuboptimalSolution | SolveStatus::NumericalError
            ),
            "got {:?}", result.status
        );
    }

    /// 不定 Q + bounds → LocallyOptimal/Optimal/Suboptimal。
    #[test]
    fn test_qp_nonconvex_with_bounds() {
        let q = CscMatrix::from_triplets(
            &[0, 1],
            &[0, 1],
            &[-2.0, 2.0],
            2,
            2,
        ).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
        let b = vec![];
        let bounds = vec![(-1.0_f64, 1.0_f64); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds.clone()).unwrap();

        let opts = SolverOptions { timeout_secs: Some(10.0), ..Default::default() };
        let result = solve_qp_with(&problem, &opts);

        assert!(
            !matches!(result.status, SolveStatus::NonConvex(_)),
            "got {:?}", result.status
        );
        assert!(
            matches!(result.status, SolveStatus::LocallyOptimal | SolveStatus::Optimal
                | SolveStatus::SuboptimalSolution | SolveStatus::Timeout),
            "got {:?}", result.status
        );
        if !result.solution.is_empty() {
            for (&xi, &(lb, ub)) in result.solution.iter().zip(bounds.iter()) {
                assert!(xi >= lb - 1e-4 && xi <= ub + 1e-4);
            }
        }
    }

    /// 半正定値 Q (min eig=0) は PSD 判定。
    #[test]
    fn test_qp_psd_semidefinite_q() {
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.0, 1.0, 1.0], 3, 3).unwrap();
        assert!(check_q_positive_semidefinite(&q));
    }

    /// SolveStatus::NonConvex の Display。
    #[test]
    fn test_solve_status_display_nonconvex() {
        let msg = "Q matrix is indefinite".to_string();
        let status = SolveStatus::NonConvex(msg.clone());
        assert_eq!(format!("{}", status), format!("NonConvex({})", msg));
    }

    /// n>1000 対角負値 → NonPSD 検出。
    #[test]
    fn test_qp_nonconvex_large_diagonal_negative() {
        let n = 1001_usize;
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        let vals: Vec<f64> = std::iter::once(-1.0_f64)
            .chain(std::iter::repeat(1.0_f64).take(n - 1))
            .collect();
        let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        assert!(!check_q_positive_semidefinite(&q));
    }

    /// n>1000 対角全正値 → PSD (偽陽性防止)。
    #[test]
    fn test_qp_psd_large_diagonal_positive() {
        let n = 1001_usize;
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        let vals: Vec<f64> = vec![1.0_f64; n];
        let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        assert!(check_q_positive_semidefinite(&q));
    }

    /// 閾値 ‖Q‖_max × 1e-6 内の僅かな負対角値は PSD 扱い (QPS encoding noise)。
    #[test]
    fn test_qp_diagonal_boundary_below_threshold() {
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1e-11_f64, 1.0, 1.0], 3, 3)
            .unwrap();
        assert!(check_q_positive_semidefinite(&q));
    }

    /// noise floor (Q[0,0]=-1e-7, ‖Q‖_max=1) は PSD。
    #[test]
    fn test_qp_diagonal_boundary_at_noise_floor() {
        let q =
            CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1e-7_f64, 1.0, 1.0], 3, 3).unwrap();
        assert!(check_q_positive_semidefinite(&q));
    }

    /// 閾値 |‖Q‖_max × 1e-6| 超 (Q[0,0]=-1e-4) → NonConvex。
    #[test]
    fn test_qp_diagonal_boundary_above_threshold() {
        let q =
            CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1e-4_f64, 1.0, 1.0], 3, 3).unwrap();
        assert!(!check_q_positive_semidefinite(&q));
    }

    /// UBH1 (n=18009) の Q が sparse LDL で non-PSD と判定されるかを実証 (n>1000 で
    /// dense Cholesky skip のため対角正値だけでは検出不能)。
    #[test]
    fn test_ubh1_q_psd_diagnose() {
        use crate::io::qps::parse_qps;
        use crate::linalg::ldl;
        use std::path::Path;
        use std::time::Instant;

        let path = Path::new("data/maros_meszaros/UBH1.QPS");
        if !path.exists() {
            eprintln!("UBH1.QPS not found, skipping");
            return;
        }
        let prob = parse_qps(path).expect("parse UBH1");
        eprintln!(
            "UBH1: n={}, m={}, Q.nnz={}",
            prob.num_vars,
            prob.num_constraints,
            prob.q.values.len()
        );

        for eps in &[0.0_f64, 1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-3, 1.0] {
            let q_reg = build_q_with_diag_reg(&prob.q, *eps);
            let t = Instant::now();
            match ldl::factorize(&q_reg) {
                Ok(_) => eprintln!(
                    "  eps={:.0e}: factorize OK (Q+εI PSD), {:.2}s",
                    eps,
                    t.elapsed().as_secs_f64()
                ),
                Err(e) => eprintln!(
                    "  eps={:.0e}: factorize FAILED ({:?}), {:.2}s",
                    eps,
                    e,
                    t.elapsed().as_secs_f64()
                ),
            }
        }
    }

    /// HS268 (n=5, m=5) で IPPMM 出力の dual 残差を成分ごと表示する診断テスト。
    #[test]
    fn test_hs268_dual_residual_diagnose() {
        use crate::io::qps::parse_qps;
        use crate::options::SolverOptions;
        use std::path::Path;

        let path = Path::new("data/maros_meszaros/HS268.QPS");
        if !path.exists() {
            eprintln!("HS268.QPS not found, skipping");
            return;
        }
        let prob = parse_qps(path).expect("parse HS268");
        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let result = solve_qp_with(&prob, &opts);
        eprintln!(
            "HS268 status={:?} obj={:.6e}",
            result.status, result.objective
        );
        let x = &result.solution;
        let y = &result.dual_solution;
        let bd = &result.bound_duals;
        eprintln!("  x = {:?}", x);
        eprintln!("  y = {:?}", y);
        eprintln!("  bound_duals = {:?} (len={})", bd, bd.len());
        // 各成分の KKT 残差: Qx + c + A^T y + bound_contrib
        let qx = prob.q.mat_vec_mul(x).unwrap();
        let aty = if !y.is_empty() {
            prob.a.transpose().mat_vec_mul(y).unwrap()
        } else {
            vec![0.0; prob.num_vars]
        };
        for j in 0..prob.num_vars {
            let r = qx[j] + prob.c[j] + aty[j];
            eprintln!(
                "    j={}: Qx={:.3e} c={:.3e} (A^Ty)={:.3e} sum={:.3e}",
                j, qx[j], prob.c[j], aty[j], r
            );
        }
        let n = prob.num_vars;
        let m = prob.num_constraints;
        let mut at_dense = vec![vec![0.0_f64; m]; n];
        for j in 0..n {
            for k in prob.a.col_ptr[j]..prob.a.col_ptr[j + 1] {
                let i = prob.a.row_ind[k];
                let v = prob.a.values[k];
                if i < m {
                    at_dense[j][i] = v;
                }
            }
        }
        let rhs: Vec<f64> = (0..n).map(|j| -(qx[j] + prob.c[j])).collect();
        let mut aug = at_dense.clone();
        let mut b = rhs.clone();
        for k in 0..n.min(m) {
            let mut max_row = k;
            for i in (k + 1)..n {
                if aug[i][k].abs() > aug[max_row][k].abs() {
                    max_row = i;
                }
            }
            aug.swap(k, max_row);
            b.swap(k, max_row);
            if aug[k][k].abs() < 1e-15 {
                eprintln!("  singular at k={}", k);
                return;
            }
            for i in (k + 1)..n {
                let factor = aug[i][k] / aug[k][k];
                for j in k..m {
                    aug[i][j] -= factor * aug[k][j];
                }
                b[i] -= factor * b[k];
            }
        }
        let mut y_recon = vec![0.0_f64; m];
        for k in (0..n.min(m)).rev() {
            let mut sum = b[k];
            for j in (k + 1)..m {
                sum -= aug[k][j] * y_recon[j];
            }
            y_recon[k] = sum / aug[k][k];
        }
        eprintln!("  reconstructed y (LSQ): {:?}", y_recon);
        eprintln!("  ratio (solver_y / recon_y):");
        for i in 0..m.min(y.len()) {
            if y_recon[i].abs() > 1e-15 {
                eprintln!("    i={}: ratio={:.4}", i, y[i] / y_recon[i]);
            }
        }
    }

    /// Q の対角に ε を加算した CSC を返す (UBH1 PSD 診断用)。
    #[cfg(test)]
    fn build_q_with_diag_reg(q: &CscMatrix, eps_q: f64) -> CscMatrix {
        let n = q.ncols;
        let mut new_col_ptr = vec![0_usize; n + 1];
        let mut new_row_ind: Vec<usize> = Vec::with_capacity(q.values.len() + n);
        let mut new_values: Vec<f64> = Vec::with_capacity(q.values.len() + n);
        for col in 0..n {
            new_col_ptr[col] = new_row_ind.len();
            let start = q.col_ptr[col];
            let end = q.col_ptr[col + 1];
            let mut diag_added = false;
            for ptr in start..end {
                let row = q.row_ind[ptr];
                let val = q.values[ptr];
                if row == col {
                    new_row_ind.push(row);
                    new_values.push(val + eps_q);
                    diag_added = true;
                } else {
                    new_row_ind.push(row);
                    new_values.push(val);
                }
            }
            if !diag_added {
                new_row_ind.push(col);
                new_values.push(eps_q);
            }
        }
        new_col_ptr[n] = new_row_ind.len();
        CscMatrix {
            col_ptr: new_col_ptr,
            row_ind: new_row_ind,
            values: new_values,
            nrows: n,
            ncols: n,
        }
    }

    /// solve_as_lp が NumericalError を返さないこと。
    #[test]
    fn test_qp001_solve_as_lp_no_numerical_error() {
        let q = CscMatrix::from_triplets(&[], &[], &[], 2, 2).unwrap();
        let c = vec![-1.0, -1.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![4.0];
        let bounds = vec![(0.0f64, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_ne!(result.status, SolveStatus::NumericalError);
    }

    /// timeout_secs=None で有限ステップ収束。
    #[test]
    fn test_a2t03_qp_no_deadline_converges() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            timeout_secs: None,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
    }

    /// cancel_flag 事前設定で Timeout。
    #[test]
    fn test_a3c02_cancel_flag_preset_qp_returns_timeout() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let cancel = Arc::new(AtomicBool::new(true));
        let opts = SolverOptions {
            cancel_flag: Some(Arc::clone(&cancel)),
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Timeout);
    }

    /// presolve 有無で解が一致 (透過性)。
    #[test]
    fn test_a4p01_presolve_transparency_qp() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts_with = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let opts_without = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
        let result_with = solve_qp_with(&problem, &opts_with);
        let result_without = solve_qp_with(&problem, &opts_without);
        assert_eq!(result_with.status, SolveStatus::Optimal);
        assert_eq!(result_without.status, SolveStatus::Optimal);
        assert!((result_with.solution[0] - result_without.solution[0]).abs() < 1e-3);
        assert!((result_with.solution[1] - result_without.solution[1]).abs() < 1e-3);
    }

    /// n>1000 では Cholesky skip。対角負値は検出、非対角の非 PSD は skip (既知制限)。
    #[test]
    fn test_a6i03_nonconvex_skip_for_large_n() {
        let n = 1001usize;
        let mut rows = vec![0usize];
        let mut cols = vec![0usize];
        let mut vals = vec![-1e-3_f64];
        for i in 1..n {
            rows.push(i);
            cols.push(i);
            vals.push(1.0);
        }
        let q1 = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        assert!(!check_q_positive_semidefinite(&q1));

        let mut rows2: Vec<usize> = (0..n).collect();
        let mut cols2: Vec<usize> = (0..n).collect();
        let mut vals2: Vec<f64> = vec![1.0; n];
        rows2.push(0);
        cols2.push(1);
        vals2.push(-2.0);
        let q2 = CscMatrix::from_triplets(&rows2, &cols2, &vals2, n, n).unwrap();
        assert!(check_q_positive_semidefinite(&q2));
    }

    /// A7-CS02: concurrent solver スレッド安全性（cancel_flag 経由の停止）
    #[cfg(feature = "parallel")]
    #[test]
    fn test_a7cs02_concurrent_cancel_flag_thread_safety() {
        // SPEC: A7-CS02
        // concurrent solver で Optimal を発見したとき cancel_flag でリソースリーク・
        // データ競合なしに停止することを確認（10回繰り返してクラッシュなし）
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        for _ in 0..10 {
            let opts = SolverOptions::default();
            let result = solve_qp_with(&problem, &opts);
            assert_eq!(result.status, SolveStatus::Optimal);
        }
    }

    /// 全スレッド Timeout → Timeout。
    #[cfg(feature = "parallel")]
    #[test]
    fn test_a7cs03_concurrent_all_timeout_returns_timeout() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            timeout_secs: Some(0.0),
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Timeout);
    }

    /// concurrent solver で cancel_flag=true → Timeout。
    #[cfg(feature = "parallel")]
    #[test]
    fn test_a3c01_cancel_flag_concurrent_returns_timeout() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let cancel = Arc::new(AtomicBool::new(true));
        let opts = SolverOptions {
            cancel_flag: Some(Arc::clone(&cancel)),
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Timeout);
    }

    /// presolve OFF 基準線。
    #[test]
    fn test_postsolve_t1_presolve_off_baseline() {
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![2.0, 3.0];
        let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, n).unwrap();
        let b = vec![4.0, 3.0];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        let tol = 1e-8_f64;
        assert!((result.solution[0]).abs() < tol);
        assert!((result.solution[1]).abs() < tol);
        assert!((result.objective).abs() < tol);
        assert_eq!(result.slack.len(), 2);
        assert!((result.slack[0] - 4.0).abs() < tol);
        assert!((result.slack[1] - 3.0).abs() < tol);
        assert_eq!(result.reduced_costs.len(), n);
        assert!((result.reduced_costs[0] - 2.0).abs() < tol);
        assert!((result.reduced_costs[1] - 3.0).abs() < tol);
    }

    /// FixedVar + col_map リマップ (rc[2]=0 で展開されること)。
    #[test]
    fn test_postsolve_t2_fixed_var_col_map() {
        let n = 3usize;
        let q = CscMatrix::new(n, n);
        let c = vec![2.0, 3.0, 1.0];
        let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 2.0], 2, n)
            .unwrap();
        let b = vec![4.0, 6.0];
        let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (5.0, 5.0)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        let tol = 1e-6_f64;
        assert_eq!(result.solution.len(), 3);
        assert!((result.solution[0]).abs() < tol);
        assert!((result.solution[1]).abs() < tol);
        assert!((result.solution[2] - 5.0).abs() < tol);
        assert!((result.objective - 5.0).abs() < tol);
        assert_eq!(result.reduced_costs.len(), 3);
        assert!((result.reduced_costs[0] - 2.0).abs() < tol);
        assert!((result.reduced_costs[1] - 3.0).abs() < tol);
        assert!((result.reduced_costs[2] - 1.0).abs() < tol);
        assert_eq!(result.slack.len(), 2);
        assert!((result.slack[0] - 4.0).abs() < tol);
        assert!((result.slack[1] - 6.0).abs() < tol);
        // 自由変数 (x, y) のみ複ementarity 検査 (固定 z は lb/ub の dual を持ち得る)。
        for j in 0..2 {
            assert!((result.solution[j] * result.reduced_costs[j]).abs() < 1e-7);
        }
    }

    /// SingletonRow + row_map: x=2 (Eq) + y≤3。
    #[test]
    fn test_postsolve_t3_singleton_row() {
        use crate::problem::ConstraintType;
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0];
        // x=2 (Eq), y<=3 (Le)
        let rows = &[0usize, 1usize];
        let cols = &[0usize, 1usize];
        let vals = &[1.0, 1.0];
        let a = CscMatrix::from_triplets(rows, cols, vals, 2, n).unwrap();
        let b = vec![2.0, 3.0];
        let bounds = vec![(0.0, f64::INFINITY); n];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![ConstraintType::Eq, ConstraintType::Le],
        )
        .unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        let tol = 1e-6_f64;
        assert_eq!(result.solution.len(), 2);
        assert!((result.solution[0] - 2.0).abs() < tol);
        assert!((result.solution[1]).abs() < tol);
        assert_eq!(result.slack.len(), 2);
        assert!((result.slack[0]).abs() < tol);
        assert_eq!(result.reduced_costs.len(), 2);
    }

    /// Ruiz + FixedVar 複合。
    #[test]
    fn test_postsolve_t4_ruiz_fixed_var() {
        let n = 3usize;
        let q = CscMatrix::new(n, n);
        let c = vec![2.0, 3.0, 1.0];
        let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[10.0, 1.0, 1.0], 2, n).unwrap();
        let b = vec![10.0, 3.0];
        let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (5.0, 5.0)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        let tol = 1e-6_f64;
        assert_eq!(result.solution.len(), 3);
        assert!((result.solution[0]).abs() < tol);
        assert!((result.solution[1]).abs() < tol);
        assert!((result.solution[2] - 5.0).abs() < tol);
        assert!((result.objective - 5.0).abs() < tol);
        assert_eq!(result.slack.len(), 2);
        assert!((result.slack[0] - 10.0).abs() < tol);
        assert!((result.slack[1] - 3.0).abs() < tol);
        assert_eq!(result.reduced_costs.len(), 3);
        assert!((result.reduced_costs[2] - 1.0).abs() < tol);
    }

    /// LCS (1e7 係数) + Ruiz + FixedVar: slack を元空間 b-Ax で再計算する精度確認。
    #[test]
    fn test_postsolve_t5_lcs_ruiz_fixed_var() {
        let n = 3usize;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0, 1.0];
        let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1e7, 1.0, 1.0, 1.0], 2, n)
            .unwrap();
        let b = vec![1e7, 2.0];
        let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.5, 0.5)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        let x = result.solution[0];
        let y = result.solution[1];
        assert_eq!(result.slack.len(), 2);
        let slack0_expected = 1e7 - 1e7 * x - y;
        let slack1_expected = 2.0 - x - y;
        let tol_rel = 1e-5_f64;
        assert!((result.slack[0] - slack0_expected).abs() <= tol_rel * slack0_expected.abs().max(1.0));
        assert!((result.slack[1] - slack1_expected).abs() <= tol_rel * slack1_expected.abs().max(1.0));
        assert_eq!(result.reduced_costs.len(), 3);
        assert!((result.reduced_costs[2] - 1.0).abs() < 1e-6);
    }

    /// EmptyCol (z 制約行ゼロ) → z=lb=0 に固定。
    #[test]
    fn test_postsolve_t6_empty_col() {
        let n = 3usize;
        let q = CscMatrix::new(n, n);
        let c = vec![2.0, 3.0, 1.0];
        let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, n).unwrap();
        let b = vec![4.0, 3.0];
        let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.0, 3.0)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        let tol = 1e-8_f64;
        assert_eq!(result.solution.len(), 3);
        assert!((result.solution[0]).abs() < tol);
        assert!((result.solution[1]).abs() < tol);
        assert!((result.solution[2]).abs() < tol);
        assert!((result.objective).abs() < tol);
        assert_eq!(result.slack.len(), 2);
        assert!((result.slack[0] - 4.0).abs() < tol);
        assert!((result.slack[1] - 3.0).abs() < tol);
        assert_eq!(result.reduced_costs.len(), 3);
        assert!((result.reduced_costs[2] - 1.0).abs() < tol);
    }

    /// QP IPM 経路では slack=[], reduced_costs=[]。
    #[test]
    fn test_postsolve_t7_qp_ipm_empty_slack_rc() {
        let n = 2usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![2.0];
        let bounds = vec![(0.0, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(result.slack.is_empty());
        assert!(result.reduced_costs.is_empty());
    }

    /// 全変数 FixedVar。
    #[test]
    fn test_postsolve_e1_all_vars_fixed() {
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(1.0_f64, 1.0_f64), (2.0_f64, 2.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_eq!(result.solution.len(), 2);
        assert_eq!(result.reduced_costs.len(), 2);
        assert_eq!(result.slack.len(), 0);
    }

    /// 制約なし問題: slack=0, rc=n。
    #[test]
    fn test_postsolve_e2_no_constraints() {
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![2.0, 3.0];
        let a = CscMatrix::new(0, n);
        let b: Vec<f64> = vec![];
        let bounds = vec![(0.0, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        let tol = 1e-8_f64;
        assert_eq!(result.slack.len(), 0);
        assert_eq!(result.reduced_costs.len(), n);
        assert!((result.solution[0]).abs() < tol);
        assert!((result.solution[1]).abs() < tol);
    }

    /// presolve=true でも reduction 発動なし → col_map identity。
    #[test]
    fn test_postsolve_e3_presolve_no_reduction() {
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![2.0];
        let bounds = vec![(0.0, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_eq!(result.reduced_costs.len(), n);
        assert_eq!(result.slack.len(), 1);
        let tol = 1e-8_f64;
        assert!((result.slack[0] - 2.0).abs() < tol);
    }

    /// LCS 発動 + presolve 変数除去なし: slack を b-Ax 元空間再計算。
    #[test]
    fn test_postsolve_e4_lcs_no_presolve_elimination() {
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1e7, 1.0], 1, n).unwrap();
        let b = vec![1e7];
        let bounds = vec![(0.0, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        let x = result.solution[0];
        let y = result.solution[1];
        assert_eq!(result.slack.len(), 1);
        let slack_expected = 1e7 - 1e7 * x - y;
        let tol_rel = 1e-5_f64;
        assert!((result.slack[0] - slack_expected).abs() <= tol_rel * slack_expected.abs().max(1.0));
        assert_eq!(result.reduced_costs.len(), n);
    }

    /// Q=0 (LP) で reduced_costs が理論値と一致 (Simplex 経路保持)。
    #[test]
    fn test_solve_as_lp_preserves_reduced_costs() {
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0, 2.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, n).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal);
        assert_eq!(result.reduced_costs.len(), n);
        assert_eq!(result.slack.len(), 1);
    }

    /// BD-T1: baseline (presolve OFF, 全変数 box) → bound_duals.len()=4。
    #[test]
    fn test_bd_t1_baseline_presolve_off() {
        let n = 2usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.001, 0.001], n, n).unwrap();
        let c = vec![1.0, 1.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![10.0];
        let bounds = vec![(0.0_f64, 5.0_f64); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        let sol_tol = 1e-3_f64;
        let tol = 1e-4_f64;
        assert!((result.solution[0]).abs() < sol_tol);
        assert!((result.solution[1]).abs() < sol_tol);
        assert_eq!(result.bound_duals.len(), 4);
        assert!(result.bound_duals[0] > tol);
        assert!(result.bound_duals[1] > tol);
        assert!(result.bound_duals[2].abs() < tol);
        assert!(result.bound_duals[3].abs() < tol);
    }

    /// BD-T2: FixedVar + bound_duals リマップ (z 除去 → bound_duals.len()=6, lb_x≠lb_y で順序検証)。
    #[test]
    fn test_bd_t2_fixed_var_remap_core() {
        let n = 3usize;
        let q =
            CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], n, n).unwrap();
        let c = vec![2.0, 1.0, 1.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![10.0];
        let bounds = vec![(0.0_f64, 5.0_f64), (0.0_f64, 5.0_f64), (3.0_f64, 3.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        let sol_tol = 5e-3_f64;
        let tol = 1e-4_f64;
        assert!((result.solution[0]).abs() < sol_tol);
        assert!((result.solution[1]).abs() < sol_tol);
        assert!((result.solution[2] - 3.0).abs() < sol_tol);
        assert_eq!(result.bound_duals.len(), 6);
        assert!(result.bound_duals[0] > tol);
        assert!(result.bound_duals[1] > tol);
        // lb_x ≠ lb_y で変数順序バグを検出。
        assert!((result.bound_duals[0] - result.bound_duals[1]).abs() > tol);
        assert!((result.bound_duals[2]).abs() < tol);
        assert!(result.bound_duals[3].abs() < 5e-3);
        assert!(result.bound_duals[4].abs() < 5e-3);
        assert!((result.bound_duals[5]).abs() < tol);
        let dual = if result.dual_solution.is_empty() {
            0.0
        } else {
            result.dual_solution[0]
        };
        let kkt_x = 2.0 - dual - result.bound_duals[0] + result.bound_duals[3];
        assert!(kkt_x.abs() < 1e-3);
        let kkt_y = 1.0 - dual - result.bound_duals[1] + result.bound_duals[4];
        assert!(kkt_y.abs() < 1e-3);
    }

    /// BD-T3: FixedVar + lb_only 変数 → bound_duals.len()=3。
    #[test]
    fn test_bd_t3_fixed_var_lb_only() {
        let n = 2usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.001, 0.001], n, n).unwrap();
        let c = vec![1.0, 1.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![10.0];
        let bounds = vec![(0.0_f64, f64::INFINITY), (2.0_f64, 2.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_eq!(result.bound_duals.len(), 3);
    }

    /// BD-T4: EmptyCol の bound_duals を KKT で復元 (refit_bound_duals_kkt が 0 埋めを修復)。
    #[test]
    fn test_bd_t4_empty_col_kkt_recovered() {
        let n = 3usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.001, 0.001], n, n).unwrap();
        let c = vec![-1.0, -1.0, 1.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![4.0];
        let bounds = vec![
            (f64::NEG_INFINITY, f64::INFINITY),
            (f64::NEG_INFINITY, f64::INFINITY),
            (0.0_f64, 3.0_f64),
        ];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_eq!(result.bound_duals.len(), 2);
        let z_lb = result.bound_duals[0];
        let z_ub = result.bound_duals[1];
        assert!((z_lb - 1.0).abs() < 1e-3, "z_lb={z_lb}");
        assert!(z_ub.abs() < 1e-3, "z_ub={z_ub}");
    }

    /// 全変数 ±∞ → bound_duals 空。
    #[test]
    fn test_bd_t5_unbounded_vars_empty() {
        let n = 2usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![10.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(result.bound_duals.is_empty());
    }

    /// BD-T6: FixedVar + ub 活性変数 (ub_dual 非ゼロ × presolve 残存)。
    #[test]
    fn test_bd_t6_ub_active_with_presolve() {
        let n = 3usize;
        let q =
            CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], n, n).unwrap();
        let c = vec![-1.0, -1.0, 1.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![10.0];
        let bounds = vec![(0.0_f64, 3.0_f64), (0.0_f64, 5.0_f64), (2.0_f64, 2.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        let sol_tol = 1e-3_f64;
        let tol = 1e-4_f64;
        assert!((result.solution[0] - 3.0).abs() < sol_tol);
        assert!((result.solution[1] - 5.0).abs() < sol_tol);
        assert!((result.solution[2] - 2.0).abs() < sol_tol);
        assert_eq!(result.bound_duals.len(), 6);
        assert!(result.bound_duals[0].abs() < tol);
        assert!(result.bound_duals[1].abs() < tol);
        assert!((result.bound_duals[2]).abs() < tol);
        assert!(result.bound_duals[3] > tol);
        assert!(result.bound_duals[4] > tol);
        assert!((result.bound_duals[5]).abs() < tol);
    }

    /// BD-T7: constraint active × lb_dual nonzero × KKT 照合 (x*=2, y*=1)。
    #[test]
    fn test_bd_t7_constraint_active_lb_dual_nonzero_kkt() {
        let n = 2usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, n).unwrap();
        let b = vec![-3.0];
        let bounds = vec![(2.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        let sol_tol = 1e-3_f64;
        let tol = 1e-4_f64;
        assert!((result.solution[0] - 2.0).abs() < sol_tol);
        assert!((result.solution[1] - 1.0).abs() < sol_tol);
        assert_eq!(result.bound_duals.len(), 2);
        let dual = if result.dual_solution.is_empty() {
            0.0
        } else {
            result.dual_solution[0]
        };
        assert!(dual > tol);
        assert!(result.bound_duals[0] > tol);
        assert!(result.bound_duals[1].abs() < tol);
        let kkt_x = result.solution[0] - dual - result.bound_duals[0];
        assert!(kkt_x.abs() < 1e-3);
        let kkt_y = result.solution[1] - dual - result.bound_duals[1];
        assert!(kkt_y.abs() < 1e-3);
    }

    /// row_infinity_norms 基本。
    #[test]
    fn test_row_infinity_norms_basic() {
        let a = CscMatrix::from_triplets(
            &[0, 1, 0],
            &[0, 1, 2],
            &[1.0, 2.5, -3.0],
            2,
            3,
        )
        .unwrap();
        let norms = a.row_infinity_norms();
        assert_eq!(norms.len(), 2);
        assert!((norms[0] - 3.0).abs() < 1e-15);
        assert!((norms[1] - 2.5).abs() < 1e-15);
    }

    /// 大/小係数行 mixed で行ノルム正規化 pfeas が偽 SubOptimal を防ぐ。
    #[test]
    fn test_pfeas_row_norm_mixed_scale() {
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1000.0], 2, 1).unwrap();
        let norms = a.row_infinity_norms();
        assert!((norms[0] - 1.0).abs() < 1e-15);
        assert!((norms[1] - 1000.0).abs() < 1e-15);

        let b: Vec<f64> = vec![1.0, 1000.0];
        let x_val: f64 = 1.0 + 1e-7;
        let ax: Vec<f64> = vec![x_val, 1000.0 * x_val];
        let eps: f64 = 1e-6;

        let pfeas_old = ax
            .iter()
            .zip(b.iter())
            .map(|(&ax_i, &b_i)| (ax_i - b_i).max(0.0))
            .fold(0.0_f64, f64::max);
        assert!(pfeas_old > 1e-5);

        let pfeas_normalized = ax
            .iter()
            .zip(b.iter())
            .zip(norms.iter())
            .map(|((&ax_i, &b_i), &rn)| {
                let violation = (ax_i - b_i).max(0.0);
                violation / (1.0 + rn + b_i.abs())
            })
            .fold(0.0_f64, f64::max);
        assert!(pfeas_normalized < eps);
    }

    /// b=0 大係数行で正規化 pfeas が偽 SubOptimal を防ぐ。
    #[test]
    fn test_pfeas_row_norm_false_suboptimal_prevention() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1e6], 1, 1).unwrap();
        let norms = a.row_infinity_norms();
        assert!((norms[0] - 1e6).abs() < 1e-9);

        let b_val: f64 = 0.0;
        let ax_val: f64 = 1e6 * 1e-9;
        let eps: f64 = 1e-6;

        let norm_b = b_val.abs().max(1.0);
        let pfeas_old = (ax_val - b_val).abs();
        assert!(pfeas_old >= eps * (1.0 + norm_b));

        let pfeas_norm = (ax_val - b_val).abs() / (1.0 + norms[0] + b_val.abs());
        assert!(pfeas_norm < eps);
    }

    /// Ge 制約 (ConstraintType::Ge) で Optimal 到達。
    #[test]
    fn test_qp_ge_defensive() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 0.5, EPS, "x[0]");
        assert_close(result.solution[1], 0.5, EPS, "x[1]");
    }

    /// Mixed Ge+Le 防御 (presolve=false でソルバ本体の正確さ; mixed presolve bug 既知)。
    #[test]
    fn test_qp_mixed_ge_le_defensive() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // Row 0: x+y≥0.5 (Ge), Row 1: x-y≤1 (Le)
        let a =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2)
                .unwrap();
        let b = vec![0.5, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![ConstraintType::Ge, ConstraintType::Le],
        )
        .unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            presolve: false,
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0, "D: wall-clock 6秒超過");
        assert_eq!(result.status, SolveStatus::Optimal, "D: status");
        assert_close(result.solution[0], 0.25, EPS, "D: x[0]");
        assert_close(result.solution[1], 0.25, EPS, "D: x[1]");
    }

    /// Concurrent Eq 制約。
    #[cfg(feature = "parallel")]
    #[test]
    fn test_concurrent_eq_constraint() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem =
            QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 0.5, EPS, "x[0]");
        assert_close(result.solution[1], 0.5, EPS, "x[1]");
    }

    /// Concurrent Ge 制約。
    #[cfg(feature = "parallel")]
    #[test]
    fn test_concurrent_ge_constraint() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 0.5, EPS, "x[0]");
        assert_close(result.solution[1], 0.5, EPS, "x[1]");
    }

    /// Concurrent Box 制約。
    #[cfg(feature = "parallel")]
    #[test]
    fn test_concurrent_box_constraint() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 0.0, EPS, "x[0]");
        assert_close(result.solution[1], 0.0, EPS, "x[1]");
    }

    /// Concurrent Mixed (Le+Eq)。
    #[cfg(feature = "parallel")]
    #[test]
    fn test_concurrent_mixed_constraint() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, 2).unwrap();
        let b = vec![1.0, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![ConstraintType::Eq, ConstraintType::Le],
        )
        .unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 0.5, EPS, "x[0]");
        assert_close(result.solution[1], 0.5, EPS, "x[1]");
    }

    /// Concurrent 無制約。
    #[cfg(feature = "parallel")]
    #[test]
    fn test_concurrent_unconstrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-2.0, -2.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 1.0, EPS, "x[0]");
        assert_close(result.solution[1], 1.0, EPS, "x[1]");
    }

    /// 全変数固定退化ケース (presolve=false で本体検証)。
    #[test]
    fn test_qp_all_vars_fixed() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        let a = CscMatrix::new(0, 1);
        let b: Vec<f64> = vec![];
        let bounds = vec![(1.0_f64, 1.0_f64)];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![]).unwrap();

        let mut opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        opts.presolve = false;
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0);
        assert_eq!(result.status, SolveStatus::Optimal, "got {:?}", result.status);
        assert_close(result.solution[0], 1.0, EPS, "x[0]");
    }

    /// SuboptimalSolution mapping: MaxIterations/NumericalError が外部に漏れないこと。
    #[test]
    fn test_suboptimal_to_optimal_mapping() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(2.0),
            ipm: crate::options::IpmOptions {
                max_iter: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_ne!(result.status, SolveStatus::MaxIterations);
        assert!(matches!(
            result.status,
            SolveStatus::Optimal | SolveStatus::Timeout | SolveStatus::SuboptimalSolution
        ), "got {:?}", result.status);
    }

    /// MaxIterations が外部 API に漏れないこと。
    #[test]
    fn test_max_iterations_to_timeout_mapping() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ipm: crate::options::IpmOptions {
                max_iter: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_ne!(result.status, SolveStatus::MaxIterations);
        assert!(matches!(
            result.status,
            SolveStatus::Optimal | SolveStatus::Timeout | SolveStatus::SuboptimalSolution
        ), "got {:?}", result.status);
    }

    /// Eq 制約 presolve ON/OFF で解一致。
    #[test]
    fn test_presolve_qp_eq_on_off_consistency() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem =
            QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();

        let opts_on = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        let mut opts_off = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        opts_off.presolve = false;

        let result_on = solve_qp_with(&problem, &opts_on);
        let result_off = solve_qp_with(&problem, &opts_off);

        assert_eq!(result_on.status, SolveStatus::Optimal);
        assert_eq!(result_off.status, SolveStatus::Optimal);
        assert!((result_on.solution[0] - result_off.solution[0]).abs() < 1e-4);
        assert!((result_on.solution[1] - result_off.solution[1]).abs() < 1e-4);
    }

    /// Box 制約 presolve ON/OFF で解一致。
    #[test]
    fn test_presolve_qp_box_on_off_consistency() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 2.0_f64); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts_on = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        let mut opts_off = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        opts_off.presolve = false;

        let result_on = solve_qp_with(&problem, &opts_on);
        let result_off = solve_qp_with(&problem, &opts_off);

        assert_eq!(result_on.status, SolveStatus::Optimal);
        assert_eq!(result_off.status, SolveStatus::Optimal);
        assert_close(result_on.solution[0], 0.0, EPS, "ON x[0]");
        assert_close(result_on.solution[1], 0.0, EPS, "ON x[1]");
        assert_close(result_off.solution[0], 0.0, EPS, "OFF x[0]");
        assert_close(result_off.solution[1], 0.0, EPS, "OFF x[1]");
    }

    /// Ge 制約 + presolve ON。
    #[test]
    fn test_qp_ge_constraint_with_presolve() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 0.5, EPS, "x[0]");
        assert_close(result.solution[1], 0.5, EPS, "x[1]");
    }

    /// Mixed (Ge+Le) presolve=false (mixed presolve バグ既知)。
    #[test]
    fn test_qp_mixed_ge_with_presolve() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2)
                .unwrap();
        let b = vec![0.5, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![ConstraintType::Ge, ConstraintType::Le],
        )
        .unwrap();

        let mut opts = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        opts.presolve = false;
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_close(result.solution[0], 0.25, EPS, "x[0]");
        assert_close(result.solution[1], 0.25, EPS, "x[1]");
    }

    /// Mixed (Ge+Le) presolve=ON + Ruiz=ON: pfeas Ge 違反検出 regression。
    #[test]
    fn test_qp_mixed_ge_le_presolve_ruiz_regression() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2)
                .unwrap();
        let b = vec![0.5, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![ConstraintType::Ge, ConstraintType::Le],
        )
        .unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "got {:?}", result.status);
        assert_close(result.solution[0], 0.25, EPS, "x[0]");
        assert_close(result.solution[1], 0.25, EPS, "x[1]");
        let pfeas = {
            let x = &result.solution;
            let ge_viol = (0.5_f64 - (x[0] + x[1])).max(0.0);
            let le_viol = (x[0] - x[1] - 1.0_f64).max(0.0);
            ge_viol.max(le_viol)
        };
        assert!(pfeas < 1e-6, "pfeas={:e}", pfeas);

        let opts_no_presolve = SolverOptions {
            timeout_secs: Some(10.0),
            presolve: false,
            ..Default::default()
        };
        let result_no_presolve = solve_qp_with(&problem, &opts_no_presolve);
        assert_eq!(result_no_presolve.status, SolveStatus::Optimal);
        assert_close(result_no_presolve.solution[0], 0.25, EPS, "no-presolve x[0]");
        assert_close(result_no_presolve.solution[1], 0.25, EPS, "no-presolve x[1]");
    }

    /// 正常解で dfeas check が Optimal を維持。
    #[test]
    fn test_dfeas_optimal_preserved() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(0.0, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal);
    }

    /// スケール不変性 (1e6 倍) で Optimal 維持。
    #[test]
    fn test_dfeas_scale_invariant() {
        let scale = 1e6_f64;
        let q = CscMatrix::from_triplets(
            &[0, 1],
            &[0, 1],
            &[2.0 * scale * scale, 2.0 * scale * scale],
            2,
            2,
        )
        .unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-scale, -scale], 1, 2).unwrap();
        let b = vec![-scale];
        let bounds = vec![(0.0, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "got {:?}", result.status);
        assert_close(result.solution[0], 0.5, 1e-4, "x[0]");
        assert_close(result.solution[1], 0.5, 1e-4, "x[1]");
    }

    /// dfeas 悪化解の SuboptimalSolution 降格 (check_dfeas_status 直接呼出)。
    #[test]
    fn test_dfeas_bad_solution_downgraded() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        // 最適 x=y=0, dfeas=0。bad: x=y=1 で Qx+c=[2,2], dfeas=2.0。
        let bad_x = vec![1.0, 1.0];
        let bad_y: Vec<f64> = vec![];
        let bad_bd: Vec<f64> = vec![];

        let status = ipm_core::check_dfeas_status(&problem, &bad_x, &bad_y, &bad_bd, 1e-6);
        assert_eq!(status, SolveStatus::SuboptimalSolution);
        let status_ok = ipm_core::check_dfeas_status(&problem, &bad_x, &bad_y, &bad_bd, 10.0);
        assert_eq!(status_ok, SolveStatus::Optimal);

        let status_rel =
            ipm_core::check_dfeas_status_relative(&problem, &bad_x, &bad_y, &bad_bd, 0.01);
        assert_eq!(status_rel, SolveStatus::SuboptimalSolution);
        let status_rel_ok =
            ipm_core::check_dfeas_status_relative(&problem, &bad_x, &bad_y, &bad_bd, 1.0);
        assert_eq!(status_rel_ok, SolveStatus::Optimal);
    }

    /// 大 KKT スケール (2e12) でも相対閾値が正規化。
    #[test]
    fn test_dfeas_relative_threshold_large_kkt() {
        let n = 1usize;
        let q = CscMatrix::from_triplets(&[0], &[0], &[2e12], n, n).unwrap();
        let c = vec![-1e6];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "got {:?}", result.status);
        assert!((result.solution[0] - 5e-7).abs() < 1e-9, "x*=5e-7, got {:.2e}", result.solution[0]);
    }

    /// 巨大項キャンセレーション (Qx ≈ -A^Ty): 成分相対なら正確に判定。
    #[test]
    fn test_dfeas_cancellation_pattern() {
        let n = 2usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let big_x = vec![5e9, 5e9];
        let empty_y: Vec<f64> = vec![];
        let empty_bd: Vec<f64> = vec![];
        let status =
            ipm_core::check_dfeas_status_relative(&problem, &big_x, &empty_y, &empty_bd, 0.01);
        assert_eq!(status, SolveStatus::SuboptimalSolution);

        let good_x = vec![1e-12, 1e-12];
        let status_good =
            ipm_core::check_dfeas_status_relative(&problem, &good_x, &empty_y, &empty_bd, 1e-8);
        assert_eq!(status_good, SolveStatus::Optimal);
    }

    /// REFIT-T1: lb 活性 + c>0 で y_lb = c を復元。
    #[test]
    fn test_refit_bound_duals_lb_only_active() {
        let n = 1usize;
        let q = CscMatrix::new(n, n);
        let c = vec![2.5_f64];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![],
            bound_duals: vec![0.0],
            objective: 0.0,
            ..SolverResult::default()
        };
        refit_bound_duals_kkt(&problem, &mut result);
        assert!((result.bound_duals[0] - 2.5).abs() < 1e-9, "got {}", result.bound_duals[0]);
    }

    /// REFIT-T2: ub 活性 + c<0 で y_ub = -c。
    #[test]
    fn test_refit_bound_duals_ub_only_active() {
        let n = 1usize;
        let q = CscMatrix::new(n, n);
        let c = vec![-3.0_f64];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, 5.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![5.0],
            dual_solution: vec![],
            bound_duals: vec![0.0],
            objective: 0.0,
            ..SolverResult::default()
        };
        refit_bound_duals_kkt(&problem, &mut result);
        assert!((result.bound_duals[0] - 3.0).abs() < 1e-9, "got {}", result.bound_duals[0]);
    }

    /// REFIT-T3: 内点では y_lb=y_ub=0 維持。
    #[test]
    fn test_refit_bound_duals_interior_keeps_zero() {
        let n = 1usize;
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], n, n).unwrap();
        let c = vec![-4.0_f64];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(0.0_f64, 5.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![2.0],
            dual_solution: vec![],
            bound_duals: vec![0.0, 0.0],
            objective: -4.0,
            ..SolverResult::default()
        };
        refit_bound_duals_kkt(&problem, &mut result);
        assert!(result.bound_duals[0].abs() < 1e-9);
        assert!(result.bound_duals[1].abs() < 1e-9);
    }

    /// REFIT-T4: KKT-guard が改善なし更新を revert (既値維持)。
    #[test]
    fn test_refit_bound_duals_kkt_guard_no_regression() {
        let n = 1usize;
        let q = CscMatrix::new(n, n);
        let c = vec![2.0_f64];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![],
            bound_duals: vec![2.0],
            objective: 0.0,
            ..SolverResult::default()
        };
        refit_bound_duals_kkt(&problem, &mut result);
        assert!((result.bound_duals[0] - 2.0).abs() < 1e-9, "got {}", result.bound_duals[0]);
    }

    /// REFIT-T5: 制約あり (A^T y 非ゼロ) で bound_dual 計算。
    #[test]
    fn test_refit_bound_duals_with_constraint() {
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0_f64, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![5.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0, 0.0],
            dual_solution: vec![0.0],
            bound_duals: vec![0.0, 0.0],
            objective: 0.0,
            ..SolverResult::default()
        };
        refit_bound_duals_kkt(&problem, &mut result);
        assert!((result.bound_duals[0] - 1.0).abs() < 1e-9);
        assert!(result.bound_duals[1].abs() < 1e-9);
    }

    /// 不可能な正 Le dual を singleton column interval {0} に projection。
    #[test]
    fn test_project_duals_from_singleton_columns_clamps_infeasible_positive_le_dual() {
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![0.0_f64, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0_f64, 1.0], 1, n).unwrap();
        let b = vec![0.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0, 0.0],
            dual_solution: vec![5.0],
            bound_duals: vec![0.0, 0.0],
            ..SolverResult::default()
        };

        project_duals_from_singleton_columns(&problem, &mut result);
        refit_bound_duals_kkt(&problem, &mut result);

        assert!(result.dual_solution[0].abs() < 1e-12);
        assert!(result.bound_duals.iter().all(|v| v.abs() < 1e-12));
    }

    /// lb-only singleton column の lower bound から y を必要値まで引き上げ。
    #[test]
    fn test_project_duals_from_singleton_columns_respects_lb_only_lower_bound() {
        let n = 1usize;
        let q = CscMatrix::new(n, n);
        let c = vec![-2.0_f64];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, n).unwrap();
        let b = vec![0.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![0.0],
            bound_duals: vec![0.0],
            ..SolverResult::default()
        };

        project_duals_from_singleton_columns(&problem, &mut result);
        refit_bound_duals_kkt(&problem, &mut result);

        assert!((result.dual_solution[0] - 2.0).abs() < 1e-12);
        assert!(result.bound_duals[0].abs() < 1e-12);
    }

    #[test]
    fn test_zero_inactive_inequality_duals_clears_slack_le_rows() {
        let n = 1usize;
        let q = CscMatrix::new(n, n);
        let c = vec![0.0_f64];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, n).unwrap();
        let b = vec![10.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![3.0],
            dual_solution: vec![7.0],
            bound_duals: vec![0.0],
            ..SolverResult::default()
        };

        zero_inactive_inequality_duals(&problem, &mut result);

        assert!(result.dual_solution[0].abs() < 1e-12);
    }

    #[test]
    fn test_refine_dual_projected_gradient_uses_curvature_scaled_step() {
        let n = 1usize;
        let q = CscMatrix::new(n, n);
        let c = vec![-1.0_f64];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1000.0_f64], 1, n).unwrap();
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0e-3],
            dual_solution: vec![0.0],
            bound_duals: vec![0.0],
            ..SolverResult::default()
        };

        refine_dual_projected_gradient(&problem, &mut result, None);

        assert!((result.dual_solution[0] - 1.0e-3).abs() < 1e-9);
    }

    #[test]
    fn test_refine_dual_worst_active_block_updates_row_and_bound_duals_together() {
        let q = CscMatrix::from_triplets(&[1], &[1], &[2.0_f64], 2, 2).unwrap();
        let c = vec![-1.0_f64, 0.0_f64];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0_f64, 1.0_f64], 1, 2).unwrap();
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY)];
        let problem =
            QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();
        let view = crate::qp::ipm_solver::outcome::ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
        };
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0_f64, 0.0_f64],
            dual_solution: vec![0.0_f64],
            bound_duals: vec![0.0_f64, 0.0_f64],
            ..SolverResult::default()
        };

        let pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        refine_dual_worst_active_block(&problem, &mut result, None);
        let post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );

        assert!(post < pre);
        assert!(post < 1e-12);
        assert!((result.dual_solution[0] - 1.0).abs() < 1e-9);
        assert!(result.bound_duals[0].abs() < 1e-12);
        assert!((result.bound_duals[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_dual_recovery_postprocess_can_improve_without_dual_ir() {
        let q = CscMatrix::from_triplets(&[1], &[1], &[2.0_f64], 2, 2).unwrap();
        let c = vec![-1.0_f64, 0.0_f64];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0_f64, 1.0_f64], 1, 2).unwrap();
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY)];
        let problem =
            QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();
        let view = crate::qp::ipm_solver::outcome::ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
        };
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0_f64, 0.0_f64],
            dual_solution: vec![0.0_f64],
            bound_duals: vec![0.0_f64, 0.0_f64],
            ..SolverResult::default()
        };

        let pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        let post = run_dual_recovery_postprocess(&problem, &view, &mut result, None, false);

        assert!(post < pre);
        assert!(post < 1e-12);
    }

    #[test]
    fn test_dual_only_ir_uses_active_rows_and_keeps_inactive_le_zero() {
        let q = CscMatrix::new(1, 1);
        let c = vec![-1.0_f64];
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0_f64, 1.0_f64], 2, 1).unwrap();
        let b = vec![1.0_f64, 10.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![
                crate::problem::ConstraintType::Eq,
                crate::problem::ConstraintType::Le,
            ],
        )
        .unwrap();
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0_f64],
            dual_solution: vec![0.0_f64, 0.0_f64],
            bound_duals: vec![],
            ..SolverResult::default()
        };

        let accepted = try_dual_only_ir(&problem, &mut result, 1e-8, None);

        assert!(accepted > 0);
        assert!((result.dual_solution[0] - 1.0).abs() < 1e-9);
        assert!(result.dual_solution[1].abs() < 1e-12);
    }

    #[test]
    fn test_dual_only_ir_couples_row_and_bound_duals() {
        let q = CscMatrix::from_triplets(&[1], &[1], &[2.0_f64], 2, 2).unwrap();
        let c = vec![-1.0_f64, 0.0_f64];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0_f64, 1.0_f64], 1, 2).unwrap();
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY)];
        let problem =
            QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();
        let view = crate::qp::ipm_solver::outcome::ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
        };
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0_f64, 0.0_f64],
            dual_solution: vec![0.0_f64],
            bound_duals: vec![0.0_f64, 0.0_f64],
            ..SolverResult::default()
        };

        let pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        let accepted = try_dual_only_ir(&problem, &mut result, 1e-8, None);
        let post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );

        assert!(accepted > 0);
        assert!(post < pre);
        assert!((result.dual_solution[0] - 1.0).abs() < 1e-6);
        assert!((result.bound_duals[1] - 1.0).abs() < 1e-6);
    }

    /// 加重 Gram (1/scale²) が componentwise 最悪 j を優先削減 (無加重では r_rel 悪化)。
    #[test]
    fn test_dual_only_ir_weighted_gram_prioritizes_worst_component() {
        let q = CscMatrix::from_triplets(&[1], &[1], &[1.0_f64], 2, 2).unwrap();
        let c = vec![0.0_f64, 3.0_f64];
        let a = CscMatrix::from_triplets(
            &[0usize, 1, 0, 1],
            &[0usize, 0, 1, 1],
            &[-1.0_f64, 1.0, -2.0, 1.0],
            2,
            2,
        )
        .unwrap();
        let b = vec![-10.0_f64, 5.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![
                crate::problem::ConstraintType::Eq,
                crate::problem::ConstraintType::Eq,
            ],
        )
        .unwrap();

        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0_f64, 5.0_f64],
            dual_solution: vec![8.0_f64, 8.0_f64 + 1e-6_f64],
            bound_duals: vec![],
            ..SolverResult::default()
        };

        let target_pf = 5e-7;
        let accepted = try_dual_only_ir(&problem, &mut result, target_pf, None);

        assert!(accepted > 0);

        let view = crate::qp::ipm_solver::outcome::ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
        };
        let df_rel = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        assert!(df_rel < target_pf, "got {:.3e}", df_rel);
    }

    /// rank-deficient Q (e e^T) + 多解で duality gap が偽 Optimal を弾く。
    #[test]
    fn test_duality_gap_rejects_rank_deficient_false_optimal() {
        use crate::sparse::CscMatrix;
        let n = 2usize;
        let q = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[1.0, 1.0, 1.0], n, n).unwrap();
        let c = vec![-1.0_f64, 0.0];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap();
        let b = vec![3.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        if result.status == SolveStatus::Optimal {
            assert!((result.objective - (-0.5)).abs() < 1e-3, "got {}", result.objective);
        }
    }

    /// EmptyCol 変数の bound_dual を統合経路で KKT 復元 (presolve ON)。
    #[test]
    fn test_refit_integration_emptycol_recovery() {
        let n = 3usize;
        let q =
            CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], n, n).unwrap();
        let c = vec![-1.0, -1.0, 2.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![5.0_f64];
        let bounds = vec![
            (0.0_f64, f64::INFINITY),
            (0.0_f64, f64::INFINITY),
            (0.0_f64, 10.0_f64),
        ];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_eq!(result.bound_duals.len(), 4);
        let z_lb_z = result.bound_duals[2];
        assert!((z_lb_z - 2.0).abs() < 1e-2, "got {}", z_lb_z);
    }

    /// 1×1 well-conditioned で compute_lsq_dual_y が解析解 y=-3 を再現。
    #[test]
    fn compute_lsq_dual_y_recovers_exact_solution_on_well_conditioned() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[2.0_f64], 1, 1).unwrap();
        let q = CscMatrix::new(1, 1);
        let c = vec![6.0_f64];
        let b = vec![0.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![0.0],
            bound_duals: vec![],
            ..SolverResult::default()
        };
        let y = compute_lsq_dual_y(&problem, &result).expect("LSQ should succeed");
        assert!((y[0] - (-3.0)).abs() < 1e-12, "got {}", y[0]);
    }

    /// ill-conditioned (cond(AAT)≈1e16) で IR が residual を f64 1-shot 限界以下に縮める。
    #[test]
    fn compute_lsq_dual_y_ir_improves_ill_conditioned_problem() {
        let delta = 1e-8;
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0_f64, 1.0, 1.0, 1.0 + delta],
            2,
            2,
        )
        .unwrap();
        let q = CscMatrix::new(2, 2);
        let c = vec![-1.0_f64, -1.0];
        let b = vec![0.0_f64; 2];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c.clone(), a.clone(), b, bounds).unwrap();
        let result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0, 0.0],
            dual_solution: vec![0.0, 0.0],
            bound_duals: vec![],
            ..SolverResult::default()
        };
        let y = compute_lsq_dual_y(&problem, &result).expect("LSQ should succeed");

        use twofloat::TwoFloat;
        let target = [1.0_f64, 1.0];
        let mut max_abs_res = 0.0_f64;
        for col in 0..2 {
            let mut s = TwoFloat::from(0.0);
            for k in a.col_ptr[col]..a.col_ptr[col + 1] {
                s = s + TwoFloat::new_mul(a.values[k], y[a.row_ind[k]]);
            }
            let r = (f64::from(s) - target[col]).abs();
            max_abs_res = max_abs_res.max(r);
        }
        // f64 1-shot solve は cond²·ε ≈ 2 で打ち止め。IR で <1e-7 に到達できる。
        assert!(max_abs_res < 1e-7, "got {:.3e}", max_abs_res);
    }

    #[test]
    fn compute_lsq_dual_y_respects_singleton_row_fixed_value() {
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-1.0_f64, 1.0], 2, 2).unwrap();
        let q = CscMatrix::new(2, 2);
        let c = vec![0.0_f64, 5.0];
        let b = vec![0.0_f64; 2];
        let bounds = vec![(0.0_f64, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0, 0.0],
            dual_solution: vec![50.0, 0.0],
            bound_duals: vec![],
            ..SolverResult::default()
        };

        let y = compute_lsq_dual_y(&problem, &result).expect("LSQ should succeed");

        assert_eq!(y.len(), 2);
        assert!(y[0].abs() < 1e-10, "got {}", y[0]);
        assert!((y[1] - (-5.0)).abs() < 1e-8, "got {}", y[1]);
    }

    /// refine_dual_lsq の DD-guard が改善なし y_new を rejection (現状維持)。
    #[test]
    fn refine_dual_lsq_keeps_y_when_lsq_does_not_strictly_improve() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let q = CscMatrix::new(1, 1);
        let c = vec![0.0_f64];
        let b = vec![0.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![0.0_f64],
            bound_duals: vec![],
            ..SolverResult::default()
        };
        refine_dual_lsq(&problem, &mut result, None);
        assert!(result.dual_solution[0].abs() < 1e-12, "got {}", result.dual_solution[0]);
    }
}
