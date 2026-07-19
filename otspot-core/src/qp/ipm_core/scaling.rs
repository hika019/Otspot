//! Ruiz スケーリングラッパーとアンスケール。
//!
//! status には一切触れない: 解品質の判定 (Optimal mint / 降格) は
//! `ipm_solver::attempt::finalize_outcome` の `prove_optimal` 一本に集約されている。
//! ここでは scaled 空間の解を元空間へ写像するだけ。

use crate::linalg::ruiz::RuizScaler;
use crate::options::SolverOptions;
use crate::problem::SolverResult;
use crate::qp::problem::QpProblem;

/// `user_eps / amplification` の machine-noise floor。
/// `IPM_EPS_NOISE_FLOOR` (ipm_core/mod.rs) と整合: core.rs の σ-tightening と
/// scaling.rs の amp-tightening は両方とも IPM convergence eps の下押しで、
/// amp > 100 が起きると 1×EPS 旧 floor が core 側 floor を defeat してしまう。
pub(crate) const EPS_FLOOR: f64 = super::IPM_EPS_NOISE_FLOOR;

/// solve_qp_ippmm 用の Ruiz スケーリングラッパー。
pub(crate) fn solve_with_ruiz_scaling<F>(
    problem: &QpProblem,
    options: &SolverOptions,
    inner_solver: F,
) -> SolverResult
where
    F: Fn(&QpProblem, &SolverOptions, f64) -> SolverResult,
{
    if options.use_ruiz_scaling && problem.num_vars > 0 {
        let n = problem.num_vars;
        let m = problem.num_constraints;

        let mut scaler = RuizScaler::new(n, m);
        scaler.compute_with_rhs(&problem.q, &problem.a, &problem.c, &[]);

        let (q_s, a_s, c_s, b_s, bounds_s) = scaler.scale_problem(
            &problem.q,
            &problem.a,
            &problem.c,
            &problem.b,
            &problem.bounds,
        );

        if let Ok(mut scaled_problem) = QpProblem::new(
            q_s,
            c_s,
            a_s,
            b_s,
            bounds_s,
            problem.constraint_types.clone(),
        ) {
            scaled_problem.obj_offset = problem.obj_offset;
            // unscale 後に元空間 eps を保証するため scaled 空間 eps を amp 倍 tighten。
            let amplification = compute_amplification(&scaler);
            let mut adjusted_opts = options.clone();
            adjusted_opts.ipm.eps = (options.ipm_eps() / amplification).max(EPS_FLOOR);
            // warm start: user 空間 (x, y) を scaled 空間に変換 (Ruiz: x = D·x_s, y = E·y_s/c)
            if let Some(ws) = adjusted_opts.warm_start_qp.as_mut() {
                if ws.x.len() == n && ws.y.len() == m {
                    for j in 0..n {
                        ws.x[j] /= scaler.d[j];
                    }
                    for i in 0..m {
                        ws.y[i] = scaler.c * ws.y[i] / scaler.e[i];
                    }
                } else {
                    log::warn!(
                        "warm_start_qp ignored: ruiz dim mismatch (x: {}/{}, y: {}/{})",
                        ws.x.len(),
                        n,
                        ws.y.len(),
                        m
                    );
                    adjusted_opts.warm_start_qp = None;
                }
            }

            let scaled_result = inner_solver(&scaled_problem, &adjusted_opts, options.ipm_eps());
            return unscale_ipm_result(scaled_result, &scaler, problem);
        }
    }

    inner_solver(problem, options, options.ipm_eps())
}

/// unscale 残差増幅率 = max(1/min(e), 1/(c·min(d)))。MIN_POSITIVE で div0 防護。
pub(crate) fn compute_amplification(scaler: &RuizScaler) -> f64 {
    let e_min = if scaler.e.is_empty() {
        1.0
    } else {
        scaler
            .e
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .max(f64::MIN_POSITIVE)
    };
    let d_min = if scaler.d.is_empty() {
        1.0
    } else {
        scaler
            .d
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .max(f64::MIN_POSITIVE)
    };
    (1.0 / e_min).max(1.0 / (scaler.c * d_min))
}

/// スケール済み IPM 結果を元スケールへ純粋に写像する (status は不変)。
///
/// 解を持たない結果 (NumericalError / Infeasible / Unbounded / 空 Timeout) は
/// そのまま返す。`final_residuals` は scaled 空間の診断値のまま残る (元空間の
/// 残差は `run_ipm_with` が独立に再計算する)。
pub(crate) fn unscale_ipm_result(
    result: SolverResult,
    scaler: &RuizScaler,
    problem: &QpProblem,
) -> SolverResult {
    if result.solution.is_empty() {
        return result;
    }
    let (x, y) = scaler.unscale_solution(&result.solution, &result.dual_solution);
    let bound_duals = if result.bound_duals.is_empty() {
        result.bound_duals
    } else {
        scaler.unscale_bound_duals(&result.bound_duals, &problem.bounds)
    };
    SolverResult {
        objective: result.objective / scaler.c,
        solution: x,
        dual_solution: y,
        bound_duals,
        ..result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::ruiz::RuizScaler;
    use crate::problem::SolveStatus;

    #[test]
    fn compute_amplification_includes_dual_side() {
        let mut scaler = RuizScaler::new(2, 2);
        scaler.e = vec![0.01, 1.0];
        scaler.d = vec![0.001, 1.0];
        scaler.c = 0.1;
        let amp = compute_amplification(&scaler);
        assert!((amp - 10000.0).abs() < 1.0, "got {:.3e}", amp);
    }

    #[test]
    fn compute_amplification_primal_dominant() {
        let mut scaler = RuizScaler::new(2, 2);
        scaler.e = vec![1e-5, 1.0];
        scaler.d = vec![0.5, 1.0];
        scaler.c = 1.0;
        let amp = compute_amplification(&scaler);
        assert!((amp - 1e5).abs() < 10.0, "got {:.3e}", amp);
    }

    fn one_var_problem() -> QpProblem {
        let q = crate::sparse::CscMatrix::new(1, 1);
        let a = crate::sparse::CscMatrix::new(0, 1);
        crate::qp::problem::QpProblem::new_all_le(
            q,
            vec![1.0],
            a,
            vec![],
            vec![(0.0_f64, f64::INFINITY)],
        )
        .unwrap()
    }

    /// Sentinel: unscale は status を書き換えない (旧実装の Suboptimal→Optimal 昇格 /
    /// MaxIterations remap を復活させると FAIL する)。手計算オラクル:
    /// d=[2], c=0.5 で x_orig = d·x_s = 2·3 = 6, obj_orig = obj_s / c = 1/0.5 = 2。
    #[test]
    fn unscale_is_pure_and_status_preserving() {
        let problem = one_var_problem();
        let mut scaler = RuizScaler::new(1, 0);
        scaler.d = vec![2.0];
        scaler.e = vec![];
        scaler.c = 0.5;
        for status in [
            SolveStatus::Optimal,
            SolveStatus::Stalled,
            SolveStatus::MaxIterations,
            SolveStatus::Timeout,
            SolveStatus::LocallyOptimal,
        ] {
            let result = SolverResult {
                status: status.clone(),
                objective: 1.0,
                solution: vec![3.0],
                dual_solution: vec![],
                bound_duals: vec![],
                ..Default::default()
            };
            let un = unscale_ipm_result(result, &scaler, &problem);
            assert_eq!(un.status, status, "status must pass through unchanged");
            assert!((un.solution[0] - 6.0).abs() < 1e-15, "x = d·x_s = 6");
            assert!((un.objective - 2.0).abs() < 1e-15, "obj = obj_s/c = 2");
        }
    }

    /// Sentinel: 解なし結果 (NumericalError 等) は無変換で返る。
    #[test]
    fn unscale_passes_through_empty_solution() {
        let problem = one_var_problem();
        let mut scaler = RuizScaler::new(1, 0);
        scaler.d = vec![2.0];
        scaler.e = vec![];
        scaler.c = 0.5;
        let result = SolverResult {
            status: SolveStatus::NumericalError,
            objective: f64::INFINITY,
            ..Default::default()
        };
        let un = unscale_ipm_result(result, &scaler, &problem);
        assert_eq!(un.status, SolveStatus::NumericalError);
        assert!(un.solution.is_empty());
        assert!(un.objective.is_infinite());
    }
}
