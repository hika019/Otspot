//! Q=0 (LP) ディスパッチ。
//!
//! 中規模以下は simplex。`n > LP_IPM_FIRST_N` または `m > LP_IPM_FIRST_M` を満たす
//! 大規模 LP は IPM を先に試し、収束しなければ残り時間で simplex にフォールバック。
//!
//! 根拠: simplex の主反復数は O(m) で各反復は pricing/BTRAN/FTRAN ~ O(n+nnz)。
//! 一方 IPM は問題サイズに依らずほぼ固定の 30-100 反復で済む。Netlib の ken-13
//! (m≈28k) や ken-18 (m≈105k) では simplex は 1000 秒で truth から 1 桁離れた
//! incumbent しか返せないが、IPM は数十秒で Optimal に到達する (#33 profile)。
//!
//! IPM 呼び出し時は **QP presolve を無効化**する。`qp_transforms.rs:594-600` の
//! Empty-Column 解析が pure LP では dfl001 / ken-13 で false Unbounded を返す
//! 既知バグ (#34 として別途追跡) があり、これにより large LP の IPM 経路が壊れ
//! ていた。LP の不有界性/不可解性は simplex/IPM 本体が判定可能で、presolve なし
//! でも IPM は ken-13 を 2 秒で Optimal、dfl001 は 120 秒で truth ±0.03% に到達。
//!
//! 閾値は commit 66e2c9d で導入されたもの (n>3000 || m>2000) を踏襲。

use std::time::Instant;

use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::simplex;

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

    // 大規模 LP: IPM 先行、Timeout/NumericalError なら simplex 再試行。
    // Optimal/LocallyOptimal/Infeasible は確定的 → 即返却。
    // Unbounded は IPM 側の Q=0 数値リスクがあるため simplex で再確認。
    // LP_DISPATCH_NOOP=1 は sentinel 用 (no-op proof) で IPM 経路を無効化する。
    let dispatch_disabled =
        std::env::var("LP_DISPATCH_NOOP").ok().as_deref() == Some("1");
    if !dispatch_disabled
        && prefer_ipm_for_size(problem.num_vars, problem.num_constraints)
    {
        let ipm_opts = ipm_opts_for_lp(options);
        let ipm_result = ipm_solver::solve_ipm(problem, &ipm_opts);
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
            SolveStatus::NonConvex(_) => {
                // Q=0 では発生しない設計。安全策で simplex に倒す。
            }
        }
    }

    let mut result = simplex::solve_with(&lp, options);
    result.objective += problem.obj_offset;
    result
}

/// LP→IPM 呼び出し時に presolve を無効化したオプションを生成。
fn ipm_opts_for_lp(options: &SolverOptions) -> SolverOptions {
    let mut o = options.clone();
    o.presolve = false;
    o
}
