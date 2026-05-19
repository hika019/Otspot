//! Q=0 (LP) dispatch.
//!
//! 中規模以下は `crate::lp::solve_lp_forwarded_from_qp` (telemetry 付き simplex) に
//! forward する。`n > LP_IPM_FIRST_N` または `m > LP_IPM_FIRST_M` を満たす大規模 LP は
//! IPM を先行し、収束しなければ残時間で simplex にフォールバック。
//!
//! IPM 呼び出し時は QP presolve を無効化する。Empty-Column 解析が pure LP で
//! false Unbounded を返す既知バグ (別途追跡) を回避するため。LP の不有界/不可解は
//! simplex/IPM 本体で判定可能で presolve なしでも検出できる。
//!
//! `LP_DISPATCH_NOOP=1` は sentinel 用 (no-op proof) で IPM 経路を無効化する。

use std::time::Instant;

use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolveStatus, SolverResult};

use super::{ipm_solver, QpProblem};

/// IPM を先に走らせる変数数閾値。Netlib 中央値 n≈800 の約 4 倍。
const LP_IPM_FIRST_N: usize = 3_000;
/// IPM を先に走らせる制約数閾値。LU 再因子分解 O(m·nnz(L)) を回避する。
const LP_IPM_FIRST_M: usize = 2_000;

pub(crate) fn prefer_ipm_for_size(n: usize, m: usize) -> bool {
    n > LP_IPM_FIRST_N || m > LP_IPM_FIRST_M
}

pub(crate) fn solve_as_lp_pub(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    let opts_with_deadline;
    let options: &SolverOptions = if options.deadline.is_none() {
        if let Some(secs) = options.timeout_secs {
            opts_with_deadline = {
                let mut o = options.clone();
                o.deadline = Some(Instant::now() + std::time::Duration::from_secs_f64(secs));
                o.timeout_secs = None;
                o
            };
            &opts_with_deadline
        } else {
            options
        }
    } else {
        options
    };

    let lp = match LpProblem::new_general(
        problem.c.clone(),
        problem.a.clone(),
        problem.b.clone(),
        problem.constraint_types.clone(),
        problem.bounds.clone(),
        None,
    ) {
        Ok(lp) => lp,
        Err(_) => return SolverResult::infeasible(),
    };

    // 大規模 LP: IPM 先行、Timeout/NumericalError/Unbounded/MaxIter は simplex 再試行。
    // Optimal/LocallyOptimal/Infeasible は確定的 → 即返却。
    // Unbounded は IPM 側 Q=0 数値リスクがあるため simplex で再確認。
    // LP_DISPATCH_NOOP=1 は sentinel 用 (no-op proof) で IPM 経路を無効化する。
    let dispatch_disabled = std::env::var("LP_DISPATCH_NOOP").ok().as_deref() == Some("1");
    if !dispatch_disabled && prefer_ipm_for_size(problem.num_vars, problem.num_constraints) {
        let ipm_opts = ipm_opts_for_lp(options);
        let ipm_result = ipm_solver::solve_ipm(problem, &ipm_opts);
        // ipm_solver は内部で obj_offset を加算済み → そのまま返す。
        match ipm_result.status {
            SolveStatus::Optimal | SolveStatus::LocallyOptimal | SolveStatus::Infeasible => {
                return ipm_result;
            }
            SolveStatus::Unbounded
            | SolveStatus::Timeout
            | SolveStatus::NumericalError
            | SolveStatus::SuboptimalSolution
            | SolveStatus::MaxIterations => {
                if options.deadline.is_some_and(|d| Instant::now() >= d) {
                    return ipm_result;
                }
                // 残時間で simplex 再試行。
            }
            SolveStatus::NonConvex(_)
            | SolveStatus::NonconvexLocal
            | SolveStatus::NonconvexGlobal => {
                // LP dispatch は Q=0 前提 → 非凸 status は本経路には出ないが、
                // non-exhaustive match を防ぎ safety net として simplex に倒す。
            }
        }
    }

    // simplex (LpProblem) は obj_offset を含まないため明示的に加算。
    let mut result = crate::lp::solve_lp_forwarded_from_qp(&lp, options);
    result.objective += problem.obj_offset;
    result
}

/// LP→IPM 呼び出し時に presolve を無効化したオプションを生成。
fn ipm_opts_for_lp(options: &SolverOptions) -> SolverOptions {
    let mut o = options.clone();
    o.presolve = false;
    o
}
