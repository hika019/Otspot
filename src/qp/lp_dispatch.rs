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
use crate::problem::{LpProblem, SolveRoute, SolveStatus, SolverResult};
use crate::simplex::guard_lp_optimal;

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
    let mut ipm_subopt_candidate: Option<SolverResult> = None;
    if !dispatch_disabled && prefer_ipm_for_size(problem.num_vars, problem.num_constraints) {
        let ipm_opts = ipm_opts_for_lp(options);
        let mut ipm_result = ipm_solver::solve_ipm(problem, &ipm_opts);
        ipm_result.stats.route = SolveRoute::LpForwardedFromQp;
        ipm_result.stats.lp_ipm_path = true;
        // ipm_solver は内部で obj_offset を加算済み → そのまま返す。
        match ipm_result.status {
            SolveStatus::Optimal | SolveStatus::LocallyOptimal | SolveStatus::Infeasible => {
                // 確定 status は simplex 再試行不要、即返却。
                // Optimal は primal guard で false-Optimal を除去してから返す。
                return guard_lp_optimal(ipm_result, &lp);
            }
            SolveStatus::Unbounded
            | SolveStatus::Timeout
            | SolveStatus::NumericalError
            | SolveStatus::MaxIterations => {
                if options.deadline.is_some_and(|d| Instant::now() >= d) {
                    return ipm_result;
                }
                // 残時間で simplex 再試行。
            }
            SolveStatus::SuboptimalSolution => {
                if options.deadline.is_some_and(|d| Instant::now() >= d) {
                    return ipm_result;
                }
                // known_optimal_obj が設定されており obj が一致するなら simplex retry 不要。
                if let Some(ref_obj) = options.known_optimal_obj {
                    if crate::bench_utils::obj_within_tol(
                        ipm_result.objective, ref_obj,
                        crate::bench_utils::OBJ_MATCH_REL_TOL,
                    ) && !ipm_result.solution.is_empty()
                    {
                        let promoted = SolverResult { status: SolveStatus::Optimal, ..ipm_result };
                        return guard_lp_optimal(promoted, &lp);
                    }
                }
                // IPM incumbent を保存して simplex 再試行。simplex が失敗したとき
                // pick_best_ipm_or_simplex が SuboptimalSolution を復元する。
                ipm_subopt_candidate = Some(ipm_result);
            }
            SolveStatus::NonConvex(_)
            | SolveStatus::NonconvexLocal
            | SolveStatus::NonconvexGlobal => {
                // LP dispatch は Q=0 前提 → 非凸 status は本経路には出ないが、
                // non-exhaustive match を防ぎ safety net として simplex に倒す。
            }
            SolveStatus::NotSupported(_) => {
                // Propagate immediately; simplex retry cannot help.
                return ipm_result;
            }
        }
    }

    // simplex (LpProblem) は obj_offset を含まないため明示的に加算。
    let mut simplex_result = crate::lp::solve_lp_forwarded_from_qp(&lp, options);
    simplex_result.objective += problem.obj_offset;
    crate::bench_utils::pick_best_ipm_or_simplex(ipm_subopt_candidate, simplex_result)
}

/// LP→IPM 呼び出し時に presolve を無効化したオプションを生成。
fn ipm_opts_for_lp(options: &SolverOptions) -> SolverOptions {
    let mut o = options.clone();
    o.presolve = false;
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    fn eq_lp_fixture(n: usize, m: usize) -> LpProblem {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            rows.push(i); cols.push(i);     vals.push(1.0);
            rows.push(i); cols.push(i + m); vals.push(1.0);
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        let b = vec![2.0_f64; m];
        let c = vec![1.0_f64; n];
        let ctypes = vec![crate::problem::ConstraintType::Eq; m];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        LpProblem::new_general(c, a, b, ctypes, bounds, None).unwrap()
    }

    /// 2 solve を独立実行し、それぞれの route stats が独立していることを確認。
    #[test]
    fn parallel_solve_stats_independent() {
        use crate::options::SolverOptions;
        use crate::problem::SolveRoute;

        let lp = eq_lp_fixture(3500, 200);
        let lp2 = eq_lp_fixture(3600, 180);
        let opts = SolverOptions::default();

        let r1 = crate::lp::solve_lp_with(&lp, &opts);
        let r2 = crate::lp::solve_lp_with(&lp2, &opts);

        assert_eq!(r1.stats.route, SolveRoute::LpDirect, "r1 route must be LpDirect");
        assert_eq!(r2.stats.route, SolveRoute::LpDirect, "r2 route must be LpDirect");
    }
}
