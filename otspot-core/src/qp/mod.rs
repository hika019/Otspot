//! QP ソルバー: min 1/2 x'Qx + c'x  s.t. Ax (≤|=|≥) b, lb ≤ x ≤ ub
//! (OSQP/qpOASES 標準の「1/2 あり」規約)

pub mod certificate;
pub mod diagnose;
pub mod global;
pub(crate) mod ipm_core;
pub mod ipm_solver;
pub mod kkt_resid;
pub(crate) mod linalg;
pub(crate) mod lp_dispatch;
pub mod multistart;
pub(crate) mod postsolve;
mod problem;
pub use crate::problem::SolverResult;
pub use diagnose::{
    diagnose, DiagnosticCode, DiagnosticReport, DiagnosticWarning, ProblemInfo, Severity,
};
pub use global::{solve_qp_global, solve_qp_global_with_stats, GlobalStats};
pub use multistart::solve_qp_multistart;
pub(crate) use lp_dispatch::solve_as_lp_pub;
#[doc(hidden)]
pub use lp_dispatch::pick_best_ipm_or_simplex;
/// Public accessor for the LP→IPM size gate (used by qps_benchmark for label
/// reporting). Returns `true` when an LP of size `(n, m)` will be routed via
/// IPM-first in `solve_as_lp_pub`.
pub fn lp_dispatch_prefers_ipm(n: usize, m: usize) -> bool {
    lp_dispatch::prefer_ipm_for_size(n, m)
}
pub(crate) use postsolve::bound_dual::{
    project_duals_from_singleton_columns, remap_bound_duals_to_orig, zero_inactive_inequality_duals,
};
pub(crate) use postsolve::postprocess::compute_lsq_dual_y;
pub(crate) use postsolve::refine::kkt_iterative::{refine_kkt_iterative, refit_bound_duals_kkt};
pub(crate) use postsolve::refine::lsq::{refine_dual_lsq, refine_dual_lsq_irls};
pub(crate) use postsolve::refine::primal_lsq::refine_primal_lsq;
pub(crate) use postsolve::refine::projected_gradient::refine_dual_projected_gradient;
pub(crate) use postsolve::refine::worst_active::refine_dual_worst_active_block;
pub use problem::{QcqpMatrix, QpProblem, QpProblemError, QpWarmStart};

use crate::options::SolverOptions;
#[cfg(test)]
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

/// faer supernodal Cholesky の deepest stack 要求 + マージン。Rust thread デフォルト 2 MB では
/// BOYD1 級 (n=93261) で overflow するため、入口で必ずこのサイズの scoped thread に載せる。
pub(crate) const SOLVE_STACK_SIZE: usize = 8 * 1024 * 1024;

/// QP をカスタム設定で解く (8 MB scoped thread で stack overflow を防ぐ)。
///
/// `options.multistart.is_some() && n_starts >= 2` のとき multi-start に委譲する。
/// multistart 内部の各 solve は同じ entry に戻るが options.multistart を None に剥がして
/// 再入を断ち切る。
///
/// Returns [`SolveStatus::NumericalError`] immediately if `options` fails
/// validation (invalid tolerance, zero threads, etc.).
pub fn solve_qp_with(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    if options.validate().is_err() {
        return SolverResult::numerical_error();
    }
    if let Some(cfg) = options.multistart.as_ref() {
        if cfg.n_starts >= 2 {
            return multistart::solve_qp_multistart(problem, options, cfg);
        }
    }
    std::thread::scope(|s| {
        let handle = std::thread::Builder::new()
            .stack_size(SOLVE_STACK_SIZE)
            .spawn_scoped(s, || dispatch_solve_qp(problem, options))
            .expect("spawn QP solver thread");
        handle.join().expect("QP solver thread panicked")
    })
}

/// Q=0 forwards to the LP entry (kept for backward compat — callers
/// should prefer `crate::lp::solve_lp_with` directly); Q≠0 goes to IPPMM.
fn dispatch_solve_qp(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    use crate::problem::{SolveRoute, SolveStatus};
    if problem.has_qcqp_constraints() {
        return SolverResult::not_supported(
            "QCQP (quadratic constraints) is not yet supported; only QP/LP problems are accepted",
        );
    }
    if problem.is_zero_q() {
        return solve_as_lp_pub(problem, options);
    }
    let mut result = ipm_solver::solve_ipm(problem, options);
    result.stats.route = SolveRoute::QpIpm;
    result.stats.deadline_triggered = matches!(result.status, SolveStatus::Timeout);
    result
}

pub(crate) use crate::tolerances::FX_TOL;

/// Warm-start 付きで QP を解く (B&B node 間引継ぎなど)。
///
/// `warm_start` を `options.warm_start_qp` に注入して `solve_qp_with` へ委譲する。
/// 既に options 側で warm を組み立てているなら `solve_qp_with` を直接呼べばよい。
pub fn solve_qp_warm(
    problem: &QpProblem,
    warm_start: &QpWarmStart,
    options: &SolverOptions,
) -> SolverResult {
    let mut opts = options.clone();
    opts.warm_start_qp = Some(warm_start.clone());
    solve_qp_with(problem, &opts)
}

#[cfg(test)]
mod tests;
