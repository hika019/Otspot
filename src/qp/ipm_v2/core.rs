//! Mehrotra IPM core への薄いラッパー。
//!
//! 既存 `ipm::solve_ippmm` を呼び出し、結果を `IpmOutcome` に変換する。
//! 既存実装を破壊しないため、IPM 数値カーネル自体は流用する。
//! v2 の新規性は wrapping レイヤー (retry 統合・status 単一化・KKT 元空間判定) にある。

use crate::options::SolverOptions;
use crate::problem::SolveStatus;
use crate::qp::problem::QpProblem;
use super::outcome::{IpmOutcome, ProblemView};
use super::kkt::{kkt_residual_rel, primal_residual_rel, bound_violation};

/// 1 回の IPM 呼出を実行し、内部 status を捨てて元空間 KKT 残差ベースの `IpmOutcome` を返す。
pub fn run_ipm(prob: &QpProblem, opts: &SolverOptions) -> IpmOutcome {
    let mut result = crate::qp::ipm::solve_qp_ippmm(prob, opts);

    // 数値エラー判定
    let nan_or_empty = result.solution.is_empty()
        || result.solution.iter().any(|v| !v.is_finite())
        || matches!(result.status, SolveStatus::NumericalError);

    if nan_or_empty {
        return IpmOutcome {
            solution: Vec::new(),
            dual_solution: Vec::new(),
            bound_duals: Vec::new(),
            objective: f64::INFINITY,
            iterations: result.iterations,
            kkt_residual_rel: f64::INFINITY,
            primal_residual_rel: f64::INFINITY,
            bound_violation: f64::INFINITY,
            numerical_failure: true,
        };
    }

    // dual の post-process refinement (LSQ): 旧 ippmm の scaled 空間判定で
    // 偽 dual が出た場合に元空間 KKT を満たす dual を最小二乗で再計算する。
    if prob.num_constraints > 0 {
        crate::qp::refine_dual_lsq(prob, &mut result);
    }

    let view = ProblemView {
        q: &prob.q,
        a: &prob.a,
        c: &prob.c,
        b: &prob.b,
        bounds: &prob.bounds,
        constraint_types: &prob.constraint_types,
    };
    let kkt = kkt_residual_rel(&view, &result.solution, &result.dual_solution, &result.bound_duals);
    let pres = primal_residual_rel(&view, &result.solution);
    let bv = bound_violation(prob.bounds.as_slice(), &result.solution);

    IpmOutcome {
        solution: result.solution,
        dual_solution: result.dual_solution,
        bound_duals: result.bound_duals,
        objective: result.objective,
        iterations: result.iterations,
        kkt_residual_rel: kkt,
        primal_residual_rel: pres,
        bound_violation: bv,
        numerical_failure: false,
    }
}
