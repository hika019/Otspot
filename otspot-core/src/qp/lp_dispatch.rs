//! Q=0 (LP) dispatch.
//!
//! Q=0 の QP は LP として `crate::lp::solve_lp_forwarded_from_qp` (telemetry 付き
//! simplex) に forward する。QP presolve は使わず、LP presolve を先に通した上で
//! 縮約後の LP を simplex で解く。

use std::time::Instant;

use super::certificate::guard_lp_optimal;
use crate::options::SolverOptions;
use crate::presolve;
#[cfg(test)]
use crate::problem::ConstraintType;
use crate::problem::{LpProblem, SolveRoute, SolveStatus, SolverResult};
#[cfg(test)]
use crate::sparse::CscMatrix;
#[cfg(test)]
use crate::tolerances::any_nonfinite;

use super::QpProblem;
#[cfg(test)]
use super::ipm_solver;

// Test-only hook: forces solve_reduced_lp_from_qp to treat the reduced solve
// as if it returned Timeout with a reduced-space solution, bypassing wall-clock.
#[cfg(test)]
thread_local! {
    static INJECT_REDUCED_TIMEOUT_QP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn timeout_result_lp_dispatch() -> SolverResult {
    let mut timeout = SolverResult {
        status: SolveStatus::Timeout,
        objective: f64::INFINITY,
        ..Default::default()
    };
    timeout.stats.route = SolveRoute::LpForwardedFromQp;
    timeout.stats.deadline_triggered = true;
    timeout
}

pub(crate) fn solve_as_lp(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
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
        Err(_) => {
            let mut r = SolverResult::numerical_error();
            r.stats.route = SolveRoute::LpForwardedFromQp;
            return r;
        }
    };

    if options.presolve {
        let t_presolve = Instant::now();
        match presolve::run_presolve(&lp, options.deadline) {
            Err(presolve::PresolveStatus::Infeasible) => {
                let mut result = SolverResult::infeasible();
                result.timing_breakdown = Some(crate::problem::TimingBreakdown {
                    presolve_us: t_presolve.elapsed().as_micros() as u64,
                    ..Default::default()
                });
                result.stats.route = SolveRoute::LpForwardedFromQp;
                return result;
            }
            Err(presolve::PresolveStatus::Unbounded) => {
                let mut result = SolverResult {
                    status: SolveStatus::Unbounded,
                    objective: f64::NEG_INFINITY,
                    ..Default::default()
                };
                result.timing_breakdown = Some(crate::problem::TimingBreakdown {
                    presolve_us: t_presolve.elapsed().as_micros() as u64,
                    ..Default::default()
                });
                result.stats.route = SolveRoute::LpForwardedFromQp;
                return result;
            }
            Ok(presolve_result) if presolve_result.was_reduced => {
                let presolve_us = t_presolve.elapsed().as_micros() as u64;
                return solve_reduced_lp_from_qp(
                    &lp,
                    problem.obj_offset,
                    presolve_result,
                    options,
                    presolve_us,
                );
            }
            Ok(_) => {
                let presolve_us = t_presolve.elapsed().as_micros() as u64;
                if options.deadline.is_some_and(|d| Instant::now() >= d) {
                    let mut timeout = timeout_result_lp_dispatch();
                    timeout.timing_breakdown = Some(crate::problem::TimingBreakdown {
                        presolve_us,
                        solve_us: 0,
                        postsolve_us: 0,
                        ..Default::default()
                    });
                    return timeout;
                }
                let t_solve = Instant::now();
                let mut no_presolve_opts = options.clone();
                no_presolve_opts.presolve = false;
                let mut result = solve_unpresolved_lp_from_qp(&lp, problem, &no_presolve_opts);
                if result.timing_breakdown.is_none() {
                    result.timing_breakdown = Some(crate::problem::TimingBreakdown {
                        presolve_us,
                        solve_us: t_solve.elapsed().as_micros() as u64,
                        postsolve_us: 0,
                        ..Default::default()
                    });
                }
                return result;
            }
        }
    }

    if options.deadline.is_some_and(|d| Instant::now() >= d) {
        return timeout_result_lp_dispatch();
    }

    solve_unpresolved_lp_from_qp(&lp, problem, options)
}

fn solve_unpresolved_lp_from_qp(
    lp: &LpProblem,
    problem: &QpProblem,
    options: &SolverOptions,
) -> SolverResult {
    // QpProblem → LpProblem 変換時に lp.obj_offset=0.0 になるため、
    // QpProblem.obj_offset を別経路で加算する。
    let mut result = crate::lp::solve_lp_forwarded_from_qp(lp, options);
    if matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::SuboptimalSolution | SolveStatus::Timeout
    ) {
        result.objective += problem.obj_offset;
    }
    fill_lp_reduced_costs_from_dual(&mut result, lp);
    result
}

fn solve_reduced_lp_from_qp(
    original_lp: &LpProblem,
    qp_obj_offset: f64,
    presolve_result: presolve::transforms::PresolveResult,
    options: &SolverOptions,
    presolve_us: u64,
) -> SolverResult {
    let reduced_lp = &presolve_result.reduced_problem;
    let mut reduced_opts = options.clone();
    reduced_opts.presolve = false;
    reduced_opts.warm_start = None;
    reduced_opts.warm_start_lp = None;

    let t_solve = Instant::now();
    let raw = solve_lp_backend_no_presolve(reduced_lp, &reduced_opts);
    let solve_us = t_solve.elapsed().as_micros() as u64;
    // Test hook: override raw with Timeout carrying a reduced-space solution.
    #[cfg(test)]
    let raw = if INJECT_REDUCED_TIMEOUT_QP.with(|v| v.get()) {
        SolverResult {
            status: SolveStatus::Timeout,
            solution: vec![0.0; reduced_lp.num_vars],
            ..Default::default()
        }
    } else {
        raw
    };
    if (raw.status == SolveStatus::NumericalError
        || (raw.status == SolveStatus::SuboptimalSolution && raw.solution.is_empty()))
        && options.deadline.is_none_or(|d| Instant::now() < d)
    {
        let mut fallback_opts = options.clone();
        fallback_opts.presolve = false;
        fallback_opts.warm_start = None;
        fallback_opts.warm_start_lp = None;
        let t_fallback = Instant::now();
        let mut fallback = solve_original_lp_direct_retry(original_lp, &fallback_opts);
        fill_lp_reduced_costs_from_dual(&mut fallback, original_lp);
        add_qp_obj_offset(&mut fallback, qp_obj_offset);
        fallback.timing_breakdown = Some(crate::problem::TimingBreakdown {
            presolve_us,
            solve_us: t_fallback.elapsed().as_micros() as u64,
            postsolve_us: 0,
            ..Default::default()
        });
        return fallback;
    }

    if matches!(
        raw.status,
        SolveStatus::Optimal
            | SolveStatus::LocallyOptimal
            | SolveStatus::SuboptimalSolution
            | SolveStatus::Timeout
    ) && (!raw.solution.is_empty() || reduced_lp.num_vars == 0)
    {
        let deadline_expired = options.deadline.is_some_and(|d| Instant::now() >= d);
        #[cfg(test)]
        let deadline_expired = deadline_expired || INJECT_REDUCED_TIMEOUT_QP.with(|v| v.get());
        if raw.status == SolveStatus::Timeout && deadline_expired {
            // `raw.solution` is in the reduced variable space; propagating it would
            // violate the SolverResult contract (solution must be in the original
            // variable space or empty). Return an empty Timeout result.
            let mut timeout = SolverResult {
                status: SolveStatus::Timeout,
                objective: f64::INFINITY,
                solution: vec![],
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
                timing_breakdown: Some(crate::problem::TimingBreakdown {
                    presolve_us,
                    solve_us,
                    postsolve_us: 0,
                    ..Default::default()
                }),
                ..Default::default()
            };
            timeout.stats.route = SolveRoute::LpForwardedFromQp;
            timeout.stats.deadline_triggered = true;
            return timeout;
        }
        let t_postsolve = Instant::now();
        let mut lifted = presolve::postsolve::run_postsolve(
            &raw,
            &presolve_result,
            original_lp,
            options.deadline,
            options.recover_warm_start_basis,
        );
        lifted.stats.route = SolveRoute::LpForwardedFromQp;
        lifted.stats.deadline_triggered = matches!(lifted.status, SolveStatus::Timeout);
        lifted = guard_lp_optimal(lifted, original_lp);
        let postsolve_us = t_postsolve.elapsed().as_micros() as u64;
        lifted.timing_breakdown = Some(crate::problem::TimingBreakdown {
            presolve_us,
            solve_us,
            postsolve_us,
            ..Default::default()
        });
        if postsolved_lp_needs_direct_retry(&lifted)
            && options.deadline.is_none_or(|d| Instant::now() < d)
        {
            let keep_lifted = lifted.clone();
            let mut fallback_opts = options.clone();
            fallback_opts.presolve = false;
            fallback_opts.warm_start = None;
            fallback_opts.warm_start_lp = None;
            let t_fallback = Instant::now();
            let mut fallback = solve_original_lp_direct_retry(original_lp, &fallback_opts);
            fill_lp_reduced_costs_from_dual(&mut fallback, original_lp);
            add_qp_obj_offset(&mut fallback, qp_obj_offset);
            if fallback.status == SolveStatus::Optimal {
                fallback.timing_breakdown = Some(crate::problem::TimingBreakdown {
                    presolve_us,
                    solve_us: t_fallback.elapsed().as_micros() as u64,
                    postsolve_us,
                    ..Default::default()
                });
                return fallback;
            }
            lifted = keep_lifted;
        }
        add_qp_obj_offset(&mut lifted, qp_obj_offset);
        return lifted;
    }

    raw
}

fn solve_lp_backend_no_presolve(lp: &LpProblem, options: &SolverOptions) -> SolverResult {
    let mut result = crate::lp::solve_lp_forwarded_from_qp(lp, options);
    fill_lp_reduced_costs_from_dual(&mut result, lp);
    result
}

fn solve_original_lp_direct_retry(lp: &LpProblem, options: &SolverOptions) -> SolverResult {
    let mut retry_opts = options.clone();
    retry_opts.presolve = false;
    retry_opts.simplex_method = crate::options::SimplexMethod::Primal;
    crate::lp::solve_lp_forwarded_from_qp(lp, &retry_opts)
}

fn add_qp_obj_offset(result: &mut SolverResult, qp_obj_offset: f64) {
    if matches!(
        result.status,
        SolveStatus::Optimal
            | SolveStatus::LocallyOptimal
            | SolveStatus::SuboptimalSolution
            | SolveStatus::Timeout
    ) {
        result.objective += qp_obj_offset;
    }
}

fn fill_lp_reduced_costs_from_dual(result: &mut SolverResult, problem: &LpProblem) {
    if result.reduced_costs.len() == problem.num_vars {
        return;
    }
    if result.solution.len() != problem.num_vars
        || result.dual_solution.len() != problem.num_constraints
    {
        return;
    }
    if !matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::LocallyOptimal | SolveStatus::SuboptimalSolution
    ) {
        return;
    }
    let Some(rc) =
        best_lp_reduced_costs_from_dual(problem, &result.solution, &result.dual_solution)
    else {
        result.reduced_costs.clear();
        return;
    };
    result.reduced_costs = rc;
    result.bound_duals.clear();
}

fn best_lp_reduced_costs_from_dual(
    problem: &LpProblem,
    solution: &[f64],
    dual_solution: &[f64],
) -> Option<Vec<f64>> {
    if solution.len() != problem.num_vars || dual_solution.len() != problem.num_constraints {
        return None;
    }
    let mut rc_minus = problem.c.clone();
    let mut rc_plus = problem.c.clone();
    for j in 0..problem.num_vars {
        let Ok((rows, vals)) = problem.a.get_column(j) else {
            return None;
        };
        for (k, &row) in rows.iter().enumerate() {
            let term = vals[k] * dual_solution[row];
            rc_minus[j] -= term;
            rc_plus[j] += term;
        }
    }
    let df_minus = lp_reduced_cost_bound_violation(problem, solution, &rc_minus);
    let df_plus = lp_reduced_cost_bound_violation(problem, solution, &rc_plus);
    Some(if df_plus < df_minus {
        rc_plus
    } else {
        rc_minus
    })
}

fn lp_reduced_cost_bound_violation(problem: &LpProblem, x: &[f64], rc: &[f64]) -> f64 {
    let n = problem.num_vars.min(x.len()).min(rc.len());
    let mut max_rel = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = problem.bounds[j];
        let fixed = lb.is_finite() && ub.is_finite() && (ub - lb).abs() < 1e-6;
        if fixed {
            continue;
        }
        let at_lb = lb.is_finite() && (x[j] - lb).abs() < 1e-6;
        let at_ub = ub.is_finite() && (x[j] - ub).abs() < 1e-6;
        let r = rc[j];
        let viol = if at_lb && !at_ub {
            f64::max(0.0, -r)
        } else if at_ub && !at_lb {
            f64::max(0.0, r)
        } else {
            0.0
        };
        let scale = 1.0 + r.abs() + problem.c[j].abs();
        max_rel = max_rel.max(viol / scale);
    }
    max_rel
}

fn postsolved_lp_needs_direct_retry(result: &SolverResult) -> bool {
    matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::SuboptimalSolution
    ) && (result
        .postsolve_dfeas
        .is_some_and(|d| d > crate::tolerances::PIVOT_TOL)
        || result.status == SolveStatus::SuboptimalSolution)
}

/// LP→IPM 呼び出し時に presolve を無効化したオプションを生成 (Farkas cert 検証専用)。
#[cfg(test)]
fn ipm_opts_for_lp(options: &SolverOptions) -> SolverOptions {
    let mut o = options.clone();
    o.presolve = false;
    o
}

/// Try a normalized Farkas certificate after simplex Phase I stalls.
/// This stays on nonnegative variables so bounds need no certificate terms.
#[cfg(test)]
fn verified_farkas_timeout_fallback(problem: &QpProblem, options: &SolverOptions) -> bool {
    if !problem
        .bounds
        .iter()
        .all(|&(lb, ub)| lb == 0.0 && ub == f64::INFINITY)
    {
        return false;
    }

    // Convert user rows to Cx >= d. Equality rows need both directions.
    let (cert_cols_by_row, cert_rhs) = normalized_farkas_rows(problem);
    if cert_rhs.is_empty() {
        return false;
    }

    // y >= 0, C^T y <= 0, d^T y >= 1 certifies Cx >= d, x >= 0 infeasible.
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..problem.num_vars {
        let Ok((a_rows, a_vals)) = problem.a.get_column(j) else {
            return false;
        };
        for (k, &i) in a_rows.iter().enumerate() {
            for &(cert_col, sign) in &cert_cols_by_row[i] {
                rows.push(j);
                cols.push(cert_col);
                vals.push(sign * a_vals[k]);
            }
        }
    }
    for (cert_col, &rhs) in cert_rhs.iter().enumerate() {
        rows.push(problem.num_vars);
        cols.push(cert_col);
        vals.push(rhs);
    }
    let Ok(cert_a) =
        CscMatrix::from_triplets(&rows, &cols, &vals, problem.num_vars + 1, cert_rhs.len())
    else {
        return false;
    };
    let mut cert_b = vec![0.0; problem.num_vars];
    cert_b.push(1.0);
    let mut cert_types = vec![ConstraintType::Le; problem.num_vars];
    cert_types.push(ConstraintType::Ge);
    let Ok(cert_qp) = QpProblem::new(
        CscMatrix::new(cert_rhs.len(), cert_rhs.len()),
        vec![0.0; cert_rhs.len()],
        cert_a,
        cert_b,
        vec![(0.0, f64::INFINITY); cert_rhs.len()],
        cert_types,
    ) else {
        return false;
    };

    let result = ipm_solver::solve_ipm(&cert_qp, &ipm_opts_for_lp(options));
    result.status == SolveStatus::Optimal
        && result.solution.len() == cert_rhs.len()
        && verify_normalized_farkas(problem, &cert_cols_by_row, &cert_rhs, &result.solution)
}

/// ユーザ行 (Ge/Le/Eq) を `Cx ≥ d` 形へ正規化し、行 i ごとの (cert_col, sign) と
/// 正規化済 RHS d を返す。Eq は両向き (±) で 2 列。
#[cfg(test)]
fn normalized_farkas_rows(problem: &QpProblem) -> (Vec<Vec<(usize, f64)>>, Vec<f64>) {
    let mut cert_cols_by_row = vec![Vec::<(usize, f64)>::new(); problem.num_constraints];
    let mut cert_rhs = Vec::new();
    for (i, &kind) in problem.constraint_types.iter().enumerate() {
        let mut push_col = |sign: f64| {
            let col = cert_rhs.len();
            cert_cols_by_row[i].push((col, sign));
            cert_rhs.push(sign * problem.b[i]);
        };
        match kind {
            ConstraintType::Ge => push_col(1.0),
            ConstraintType::Le => push_col(-1.0),
            ConstraintType::Eq => {
                push_col(1.0);
                push_col(-1.0);
            }
        }
    }
    (cert_cols_by_row, cert_rhs)
}

/// 正規化制約 dᵀy ≥ 1 の許容下限。1 は cert LP の正規化定数 (データスケール非依存)
/// なので絶対 tol で安全。
#[cfg(test)]
const FARKAS_NORM_TOL: f64 = 1e-7;

/// 内積 Σ sign·a·y の f64 累積丸め誤差を見積もる 1 項あたりの後退誤差係数。
///
/// floor は IPM 収束 tol ではなく f64 の**真の丸め境界**に置く。n 項の積和の丸め
/// 誤差は後退誤差解析で ≲ n·u·Σ|項| (u = ε/2 は unit roundoff)。各項は積 1 回 +
/// 和 1 回で最大 2u = ε の相対誤差を負うため、floor を `n_terms·ε·term_mag` とする。
///
/// これを IPM tol と分離する理由: cert IPM は infeasible な cert LP の残差を自身の
/// 収束 tol (~1e-11) まで潰し、Eq の ± 二方向で Cᵀy = y0−y1 ≈ 1/K の微小な**正の
/// slack** を持つ偽証明を作る。floor を 1e-11 級に置くとこれを noise と誤判定し
/// feasible (`x1+x2=K`, K≳1e11) を false-infeasible に認定する。floor を丸め境界
/// (~n·1e-16) に締めると、IPM 残差由来の偽証明 (Cᵀy~1e-11..1e-13) は floor の数桁
/// 上で reject される。klein3 の genuine cert (Cᵀy<0、厳密に負) と真の丸め
/// (~n·ε·term_mag 以下) は通過する。soundness 最優先: 偽 accept を出さないことを、
/// 境界際 genuine cert を取りこぼす (honest Timeout 化) より優先する。
#[cfg(test)]
const FARKAS_CTY_ROUNDOFF_PER_TERM: f64 = f64::EPSILON;

/// 正の slack `aty = (Cᵀy)_j` が内積丸め誤差の範囲内か (= Cᵀy ≤ 0 を f64 精度で
/// 満たすか)。`term_mag = Σ_k |sign·a·y|` はその成分の内積項 magnitude、
/// `n_terms` は加算した項数。floor を `n_terms·ε·term_mag` とし、scale 不変かつ
/// IPM tol から独立な丸め境界で判定する。
#[cfg(test)]
fn cty_slack_within_noise(aty: f64, term_mag: f64, n_terms: usize) -> bool {
    let roundoff_floor = (n_terms as f64) * FARKAS_CTY_ROUNDOFF_PER_TERM * term_mag;
    aty <= roundoff_floor
}

#[cfg(test)]
fn verify_normalized_farkas(
    problem: &QpProblem,
    cert_cols_by_row: &[Vec<(usize, f64)>],
    cert_rhs: &[f64],
    y: &[f64],
) -> bool {
    if y.len() != cert_rhs.len() || any_nonfinite(y) {
        return false;
    }
    // 厳密な非負部分 y⁺ = max(y, 0) で検証する。IPM の僅かな負 slack を許容しても
    // y⁺ ≥ 0 が厳密に成り立つので Farkas の健全性 (dᵀy⁺ ≤ xᵀCᵀy⁺) を崩さない。
    let yp = |col: usize| y[col].max(0.0);
    let rhs_dot = cert_rhs
        .iter()
        .enumerate()
        .map(|(col, &d)| d * yp(col))
        .sum::<f64>();
    if !rhs_dot.is_finite() || rhs_dot < 1.0 - FARKAS_NORM_TOL {
        return false;
    }
    for j in 0..problem.num_vars {
        let Ok((a_rows, a_vals)) = problem.a.get_column(j) else {
            return false;
        };
        let mut aty = 0.0;
        let mut term_mag = 0.0;
        let mut n_terms = 0usize;
        for (k, &i) in a_rows.iter().enumerate() {
            for &(cert_col, sign) in &cert_cols_by_row[i] {
                let term = sign * a_vals[k] * yp(cert_col);
                aty += term;
                term_mag += term.abs();
                n_terms += 1;
            }
        }
        if !aty.is_finite() || !cty_slack_within_noise(aty, term_mag, n_terms) {
            return false;
        }
    }
    true
}

/// 旧 IPM dispatch を発火させた規模閾値。IPM 撤廃後は production では未使用で、
/// 大規模 LP が simplex 経路を通ることを確認する test fixture としてのみ保持する。
#[cfg(test)]
const LP_IPM_FIRST_N: usize = 3_000;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;
    use std::time::Duration;

    fn eq_lp_fixture(n: usize, m: usize) -> LpProblem {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            rows.push(i);
            cols.push(i);
            vals.push(1.0);
            rows.push(i);
            cols.push(i + m);
            vals.push(1.0);
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

        assert_eq!(
            r1.stats.route,
            SolveRoute::LpDirect,
            "r1 route must be LpDirect"
        );
        assert_eq!(
            r2.stats.route,
            SolveRoute::LpDirect,
            "r2 route must be LpDirect"
        );
    }

    /// Q=0 QP entry must run LP presolve before reaching the simplex backend.
    ///
    /// The unreduced problem has `n > LP_IPM_FIRST_N`. LP presolve fixes the
    /// singleton row and empty positive-cost columns, leaving a zero-size reduced
    /// LP that postsolves back to the original space. `lp_ipm_path` stays false
    /// (IPM 経路は撤廃済み)。
    #[test]
    fn qp_zero_path_presolve_reduces_before_ipm_dispatch() {
        use crate::options::SolverOptions;
        use crate::problem::{SolveRoute, SolveStatus};

        let n = LP_IPM_FIRST_N + 1;
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap();
        let mut problem = QpProblem::new(
            CscMatrix::new(n, n),
            vec![2.0; n],
            a,
            vec![1.0],
            vec![(0.0, f64::INFINITY); n],
            vec![ConstraintType::Eq],
        )
        .unwrap();
        problem.obj_offset = 5.0;

        let result = solve_as_lp(&problem, &SolverOptions::default());

        assert_eq!(result.status, SolveStatus::Optimal);
        assert_eq!(result.stats.route, SolveRoute::LpForwardedFromQp);
        assert!(
            !result.stats.lp_ipm_path,
            "presolve must reduce; LP path never sets lp_ipm_path"
        );
        assert_eq!(result.solution.len(), n);
        assert!((result.solution[0] - 1.0).abs() < 1e-9);
        assert!(result.solution[1..].iter().all(|&x| x.abs() < 1e-9));
        assert!(
            (result.objective - 7.0).abs() < 1e-9,
            "objective must include presolve contribution and QP obj_offset"
        );
        let timing = result
            .timing_breakdown
            .expect("LP-dispatched QP presolve/postsolve path must keep timing");
        assert!(timing.presolve_us > 0, "presolve timing must be recorded");
        assert!(timing.postsolve_us > 0, "postsolve timing must be recorded");
    }

    /// 大規模 LP (n > 旧 IPM 閾値) でも IPM を経由せず simplex 一本で解くこと。
    /// IPM 撤廃の load-bearing sentinel: `lp_ipm_path` は常に false、route は
    /// `LpForwardedFromQp`。presolve を切って simplex backend を直接通す。
    #[test]
    fn large_lp_dispatch_stays_on_simplex_path() {
        use crate::options::SolverOptions;
        use crate::problem::{SolveRoute, SolveStatus};

        // n = 旧 IPM dispatch 発火規模。単一等式 x_0 = 1、min Σ x_i → x_0=1 他 0、obj=1。
        let n = LP_IPM_FIRST_N + 1;
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap();
        let problem = QpProblem::new(
            CscMatrix::new(n, n),
            vec![1.0; n],
            a,
            vec![1.0],
            vec![(0.0, f64::INFINITY); n],
            vec![ConstraintType::Eq],
        )
        .unwrap();
        let opts = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };

        let result = solve_as_lp(&problem, &opts);

        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "large LP must solve via simplex; got {:?}",
            result.status
        );
        assert_eq!(result.stats.route, SolveRoute::LpForwardedFromQp);
        assert!(
            !result.stats.lp_ipm_path,
            "IPM 撤廃後 LP は simplex 一本: lp_ipm_path must be false for n>{LP_IPM_FIRST_N}"
        );
        assert_eq!(result.solution.len(), n);
        assert!((result.objective - 1.0).abs() < 1e-6);
    }

    #[test]
    fn qp_zero_path_expired_deadline_after_presolve_returns_timeout() {
        use crate::options::SolverOptions;
        use crate::problem::{SolveRoute, SolveStatus};

        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let problem = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![1.0],
            a,
            vec![1.0],
            vec![(0.0, f64::INFINITY)],
            vec![ConstraintType::Eq],
        )
        .unwrap();
        let opts = SolverOptions {
            deadline: Some(Instant::now() - Duration::from_millis(1)),
            presolve: true,
            ..SolverOptions::default()
        };

        let result = solve_as_lp(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Timeout);
        assert_eq!(result.stats.route, SolveRoute::LpForwardedFromQp);
        assert!(result.stats.deadline_triggered);
    }

    /// 非負変数の QP/LP を密行で構築するヘルパー (Farkas 検証 sentinel 用)。
    fn nonneg_qp(a_rows: &[Vec<f64>], b: &[f64], types: &[ConstraintType]) -> QpProblem {
        let m = a_rows.len();
        let n = a_rows.first().map_or(0, |r| r.len());
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (i, row) in a_rows.iter().enumerate() {
            assert_eq!(row.len(), n, "rows must be rectangular");
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    rows.push(i);
                    cols.push(j);
                    vals.push(v);
                }
            }
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        QpProblem::new(
            CscMatrix::new(n, n),
            vec![0.0; n],
            a,
            b.to_vec(),
            vec![(0.0, f64::INFINITY); n],
            types.to_vec(),
        )
        .unwrap()
    }

    /// 旧 IPM-tol 級 floor。sentinel が「floor を IPM tol (1e-11) に戻すと
    /// 偽証明を accept する」ことを明示するための参照値 (実装側には残っていない)。
    const LEGACY_IPM_TOL_FLOOR: f64 = 1e-11;

    /// 丸め境界 floor `n_terms·ε·term_mag` を複数パターンで cover。
    /// 偽証明 (IPM 残差級の正 slack) は reject、真の負残差/丸め以下は accept。
    /// 同一 aty でも n_terms / term_mag で floor がスケールすることを確認。
    #[test]
    fn cty_slack_within_noise_separates_real_slack_from_roundoff() {
        // (aty, term_mag, n_terms, expect_within_noise, label)
        let cases = [
            // IPM 残差級の正 slack: floor (~n·ε·mag) の数桁上 → reject。
            (1e-9, 8.76, 2, false, "d=1e9 normalized feasible"),
            (1.8626e-9, 2.0, 2, false, "d=1e9 (dᵀy≈1.86)"),
            (1.49e-8, 2.0, 2, false, "d=1e8 normalized feasible"),
            (1.455e-11, 2.0, 2, false, "K=1e11 false-cert residual"),
            (1.137e-13, 2.0, 2, false, "K=1e13 false-cert residual"),
            // klein3 genuine: 残差は厳密に負。
            (-4.1e-6, 986.0, 4, true, "klein3 genuine cert"),
            (-1.0, 3.0, 2, true, "strict negative residual"),
            (0.0, 5.0, 2, true, "exact zero residual"),
            // f64 内積丸めレベル: noise として accept。floor = 2·ε·8.76 ≈ 3.9e-15。
            (1e-15, 8.76, 2, true, "roundoff-level positive"),
            (2e-16, 1.0, 2, true, "near machine eps"),
            // n_terms スケール: 同 aty=1e-12 でも項数で floor が動く。
            (1e-12, 1.0, 2, false, "small n: above roundoff floor"),
            (
                1e-12,
                1.0,
                10_000,
                true,
                "large n: within accumulated roundoff",
            ),
        ];
        for (aty, mag, n_terms, expect, label) in cases {
            assert_eq!(
                cty_slack_within_noise(aty, mag, n_terms),
                expect,
                "case `{label}`: aty={aty:e}, mag={mag:e}, n_terms={n_terms}",
            );
        }

        // load-bearing: floor を IPM tol (1e-11) に戻すと K≳1e11 の偽証明残差
        // (1.455e-11 / 1.137e-13) を noise と誤判定する。丸め境界 floor はこれを
        // reject する。両 floor が分岐することを実証 (sentinel が no-op で FAIL)。
        for &(aty, mag, n_terms) in &[(1.455e-11, 2.0, 2usize), (1.137e-13, 2.0, 2)] {
            assert!(
                aty <= LEGACY_IPM_TOL_FLOOR * mag,
                "premise: IPM-tol floor would have accepted aty={aty:e}",
            );
            assert!(
                !cty_slack_within_noise(aty, mag, n_terms),
                "roundoff floor must reject IPM-residual slack aty={aty:e}",
            );
        }
    }

    /// 大 magnitude feasible (`x1+x2=K`) が Infeasible 認定されないこと。
    /// 偽証明 y は正規化 dᵀy≥1 を満たすが Cᵀy≈dᵀy/K の本物の正 slack を持つ。
    /// load-bearing: K≳1e11 の偽残差は旧 IPM-tol floor (1e-11) では accept される。
    #[test]
    fn farkas_rejects_large_magnitude_feasible() {
        // (K, g, legacy_would_accept): g は y0-y1 (2 のべきで厳密表現)。dᵀy=K·g≥1。
        // legacy_would_accept = 旧 IPM-tol floor (1e-11·term_mag) が Cᵀy=g を誤 accept
        // するか。K=1e9 の残差 (~1.86e-9) は旧 floor でも既に reject されるため非 load-
        // bearing、K≳1e11 (~1.46e-11..1.14e-13) が新 floor 固有の reject。
        let patterns = [
            (1e9, 2.0_f64.powi(-29), false), // Cᵀy = g ≈ 1.863e-9
            (1e11, 2.0_f64.powi(-36), true), // Cᵀy = g ≈ 1.455e-11
            (1e12, 2.0_f64.powi(-39), true), // Cᵀy = g ≈ 1.819e-12
            (1e13, 2.0_f64.powi(-43), true), // Cᵀy = g ≈ 1.137e-13
        ];
        for (k, g, legacy_would_accept) in patterns {
            let problem = nonneg_qp(&[vec![1.0, 1.0]], &[k], &[ConstraintType::Eq]);
            let (cols, rhs) = normalized_farkas_rows(&problem);
            assert_eq!(rhs, vec![k, -k], "Eq → ±K の cert RHS");
            // y0 = 1 + g, y1 = 1。Cᵀy = y0 - y1 = g (正)、dᵀy = K·g ≥ 1。
            let y = vec![1.0 + g, 1.0];
            let cty = g; // y0 - y1
            let dty = k * g;
            let term_mag = (1.0 + g) + 1.0; // |y0| + |y1|
            assert!(
                dty >= 1.0 - FARKAS_NORM_TOL,
                "premise: dᵀy={dty} must clear norm"
            );
            assert_eq!(
                cty <= LEGACY_IPM_TOL_FLOOR * term_mag,
                legacy_would_accept,
                "premise: IPM-tol floor accept(Cᵀy={cty:e}) for K={k:e}",
            );
            assert!(
                !verify_normalized_farkas(&problem, &cols, &rhs, &y),
                "feasible x1+x2={k:e} must NOT be certified infeasible",
            );
        }
    }

    /// reviewer 再現を端から潰す: cert IPM を実際に走らせる end-to-end gate。
    /// feasible 問題 (`x1+x2=K`, single-var Ge `2x1≥K`) は cert LP 自体が
    /// infeasible なので、IPM が残差を tol まで潰した偽証明を返しても
    /// 丸め境界 floor が reject し、Infeasible 認定されてはならない。
    #[test]
    fn verified_farkas_rejects_feasible_large_magnitude_end_to_end() {
        use crate::options::SolverOptions;
        let opts = SolverOptions::default();
        for k in [1e9, 1e11, 1e12, 1e13] {
            // x1+x2=K, x≥0 は feasible (例 x=(K,0))。
            let eq = nonneg_qp(&[vec![1.0, 1.0]], &[k], &[ConstraintType::Eq]);
            assert!(
                !verified_farkas_timeout_fallback(&eq, &opts),
                "feasible x1+x2={k:e} must NOT be certified infeasible",
            );
            // 2x1 ≥ K, x≥0 は feasible (x1=K/2)。
            let ge = nonneg_qp(&[vec![2.0]], &[k], &[ConstraintType::Ge]);
            assert!(
                !verified_farkas_timeout_fallback(&ge, &opts),
                "feasible 2x1 ≥ {k:e} must NOT be certified infeasible",
            );
        }
    }

    /// genuine infeasible (`x1≥1` かつ `-2x1≥1`) は証明書が通り続ける。
    /// klein3 と同型: max Cᵀy < 0 (厳密に負)、dᵀy ≫ 1。
    #[test]
    fn farkas_certifies_genuine_infeasible() {
        let problem = nonneg_qp(
            &[vec![1.0], vec![-2.0]],
            &[1.0, 1.0],
            &[ConstraintType::Ge, ConstraintType::Ge],
        );
        let (cols, rhs) = normalized_farkas_rows(&problem);
        assert_eq!(rhs, vec![1.0, 1.0]);
        let y = vec![1.0, 1.0];
        // Cᵀy = 1·1 + (-2)·1 = -1 < 0、dᵀy = 2。
        assert!(
            verify_normalized_farkas(&problem, &cols, &rhs, &y),
            "genuine infeasible must remain certified",
        );
    }

    /// Infeasible LP dispatched via QP path must return `f64::INFINITY` as objective,
    /// regardless of `problem.obj_offset`.
    ///
    /// Sentinel: removing `objective: f64::INFINITY` from any simplex Infeasible arm
    /// (e.g. reverting to `objective: 0.0`) causes the assert to fail.
    #[test]
    fn infeasible_lp_dispatch_obj_offset_not_added() {
        use crate::options::SolverOptions;
        use crate::problem::SolveStatus;
        // Infeasible: x >= 2 AND x <= 1 (empty feasible set), obj_offset = 42.5
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 1).unwrap();
        let mut problem = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![1.0],
            a,
            vec![2.0, 1.0],
            vec![(0.0, f64::INFINITY)],
            vec![ConstraintType::Ge, ConstraintType::Le],
        )
        .unwrap();
        problem.obj_offset = 42.5;
        let result = solve_as_lp(&problem, &SolverOptions::default());
        assert_eq!(
            result.status,
            SolveStatus::Infeasible,
            "expected Infeasible, got {:?}",
            result.status
        );
        assert!(
            result.objective.is_infinite() && result.objective.is_sign_positive(),
            "Infeasible objective must be +INFINITY (convention); got {} (obj_offset={})",
            result.objective,
            problem.obj_offset,
        );
    }

    /// 小 magnitude feasible は元々誤認定されない (正 slack が大きく floor 超過)。
    /// 相対化が小規模問題を退化させないことの確認。
    #[test]
    fn farkas_rejects_modest_feasible() {
        let problem = nonneg_qp(&[vec![1.0, 1.0]], &[2.0], &[ConstraintType::Eq]);
        let (cols, rhs) = normalized_farkas_rows(&problem);
        // dᵀy=1 → y0-y1=0.5 (大きな正 slack)。
        let y = vec![0.5, 0.0];
        assert!(
            !verify_normalized_farkas(&problem, &cols, &rhs, &y),
            "modest feasible must NOT be certified infeasible",
        );
    }

    // ── verified_farkas_timeout_fallback 早期 false return ────────────

    /// 非負制約 (lb=0, ub=∞) を持たない問題は Farkas 経路に入れない → false。
    ///
    /// sentinel: 各入力は制約 `-x ≥ 1` (x ≤ -1) を使う。nonneg 解釈では infeasible
    /// なので cert LP が Optimal を返す。境界チェックを削除すると true を返し fail。
    #[test]
    fn farkas_false_on_non_nonneg_bounds() {
        use crate::options::SolverOptions;
        let opts = SolverOptions::default();

        // lb < 0: 問題は feasible (x=-2 で -(-2)=2≥1)、nonneg 解釈では infeasible。
        let neg_lb = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![0.0],
            CscMatrix::from_triplets(&[0], &[0], &[-1.0], 1, 1).unwrap(),
            vec![1.0],
            vec![(-2.0, f64::INFINITY)],
            vec![ConstraintType::Ge],
        )
        .unwrap();
        assert!(
            !verified_farkas_timeout_fallback(&neg_lb, &opts),
            "lb < 0 must return false (non-nonneg bounds)",
        );

        // finite ub: 境界チェック除去後 cert LP が Optimal → sentinel。
        let finite_ub = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![0.0],
            CscMatrix::from_triplets(&[0], &[0], &[-1.0], 1, 1).unwrap(),
            vec![1.0],
            vec![(0.0, 10.0)],
            vec![ConstraintType::Ge],
        )
        .unwrap();
        assert!(
            !verified_farkas_timeout_fallback(&finite_ub, &opts),
            "finite ub must return false (non-nonneg bounds)",
        );

        // lb > 0 (lb=0.5): lb=0 でない非負でない境界も同様。
        let lb_positive = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![0.0],
            CscMatrix::from_triplets(&[0], &[0], &[-1.0], 1, 1).unwrap(),
            vec![1.0],
            vec![(0.5, f64::INFINITY)],
            vec![ConstraintType::Ge],
        )
        .unwrap();
        assert!(
            !verified_farkas_timeout_fallback(&lb_positive, &opts),
            "lb=0.5 must return false (non-nonneg bounds)",
        );
    }

    /// zero-Q QpProblem が simplex 経由で Timeout を返した場合、`obj_offset` が
    /// 加算されることを確認する。
    ///
    /// sentinel: `SolveStatus::Timeout` を match から削除すると
    /// `result.objective += problem.obj_offset` が実行されず、
    /// `result.objective == 0.0` のまま → assert FAIL。
    ///
    /// `c = [0.0]` により c^T x* = 0 (incumbent 不定でも)。cancel_flag=true で
    /// 初回イテレーション即キャンセル → Timeout with initial BFS objective = 0。
    #[test]
    fn test_qp_simplex_dispatch_timeout_includes_obj_offset() {
        use std::sync::{atomic::AtomicBool, Arc};

        const OBJ_OFFSET: f64 = 42.0;

        // min 0·x s.t. x >= 1, x in [0, ∞).  c=0 → c^T x* = 0 for any incumbent.
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let mut problem = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![0.0],
            a,
            vec![1.0],
            vec![(0.0, f64::INFINITY)],
            vec![ConstraintType::Ge],
        )
        .unwrap();
        problem.obj_offset = OBJ_OFFSET;

        // cancel_flag=true + presolve=false: simplex fires cancel at first iteration,
        // returns Timeout with initial BFS (objective = 0.0 before offset).
        let opts = SolverOptions {
            cancel_flag: Some(Arc::new(AtomicBool::new(true))),
            presolve: false,
            ..SolverOptions::default()
        };

        let result = solve_as_lp(&problem, &opts);

        assert_eq!(
            result.status,
            SolveStatus::Timeout,
            "cancel_flag=true must produce Timeout; got {:?}",
            result.status,
        );
        // c^T x* = 0 (zero cost), so objective must equal obj_offset exactly.
        // Sentinel: removing SolveStatus::Timeout from the match leaves
        // objective = 0.0 (no offset added) → assert fails.
        assert!(
            (result.objective - OBJ_OFFSET).abs() < 1e-9,
            "Timeout objective must include obj_offset {OBJ_OFFSET}; got {} \
             (sentinel: removing Timeout from match yields 0.0 ≠ {OBJ_OFFSET})",
            result.objective,
        );
    }

    /// 制約がゼロ本の問題は cert_rhs が空になり早期 false を返す (regression)。
    ///
    /// `cert_rhs.is_empty()` ガードの除去後は 0 変数 cert LP が IPM に渡り、
    /// Infeasible 返却になるため no-op では fail しない (sentinel 要件非充足)。
    /// 既知 early-exit 動作の文書化テスト。
    #[test]
    fn farkas_false_on_empty_constraints() {
        use crate::options::SolverOptions;
        let opts = SolverOptions::default();

        let zero_constraints = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![0.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![(0.0, f64::INFINITY)],
            vec![],
        )
        .unwrap();
        assert!(
            !verified_farkas_timeout_fallback(&zero_constraints, &opts),
            "zero constraints → empty cert_rhs must return false",
        );
    }

    /// Sentinel: `solve_reduced_lp_from_qp` must not return a reduced-space solution on Timeout.
    ///
    /// Pre-fix (lines 202-212): returned `raw` which had `solution.len() == reduced_num_vars`.
    /// Post-fix: returns `solution: vec![]`.
    ///
    /// LP fixture: 3-var problem where presolve fixes x via a singleton Eq row,
    /// leaving a 2-var sub-problem (y,z). `orig_n=3`, `reduced_n=2`.
    /// A reduced-space Timeout solution (len=2) is visibly wrong (not 0, not 3).
    ///
    ///   row 0: 1.0*x = 5          (singleton Eq — presolve fixes x=5)
    ///   row 1: 2.0*y + 1.0*z >= 3
    ///   row 2: 1.0*y + 2.0*z >= 3
    #[test]
    fn presolve_timeout_solution_never_leaks_reduced_space_in_qp_path() {
        use crate::options::SolverOptions;
        use crate::problem::SolveStatus;

        let a = CscMatrix::from_triplets(
            &[0, 1, 2, 1, 2],
            &[0, 1, 1, 2, 2],
            &[1.0, 2.0, 1.0, 1.0, 2.0],
            3,
            3,
        )
        .unwrap();
        let problem = QpProblem::new(
            CscMatrix::new(3, 3),
            vec![1.0, 1.0, 1.0],
            a,
            vec![5.0, 3.0, 3.0],
            vec![(0.0, f64::INFINITY); 3],
            vec![ConstraintType::Eq, ConstraintType::Ge, ConstraintType::Ge],
        )
        .unwrap();
        let orig_n = problem.num_vars; // 3

        // 1. Normal solve: solution must be in original space.
        let r = solve_as_lp(&problem, &SolverOptions { presolve: true, ..Default::default() });
        assert_eq!(r.status, SolveStatus::Optimal);
        assert_eq!(
            r.solution.len(),
            orig_n,
            "Optimal: solution.len() must equal orig_n={orig_n}"
        );

        // 2. Inject hook: force reduced-space Timeout and bypass wall-clock deadline check.
        // Pre-fix: returns raw.solution.len() == reduced_n (2) — FAIL.
        // Post-fix: returns solution: vec![] — PASS.
        INJECT_REDUCED_TIMEOUT_QP.with(|v| v.set(true));
        let r = solve_as_lp(&problem, &SolverOptions { presolve: true, ..Default::default() });
        INJECT_REDUCED_TIMEOUT_QP.with(|v| v.set(false));
        let n = r.solution.len();
        assert_eq!(r.status, SolveStatus::Timeout, "injected path must return Timeout");
        assert!(
            n == 0 || n == orig_n,
            "injected Timeout: solution.len()={n} must be 0 or {orig_n}, \
             never 2 (reduced — lp_dispatch reduced-space leak)",
        );
    }
}
