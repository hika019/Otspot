//! Primal-to-dual crossover: reconstruct an optimal basis from a known primal solution.

use super::super::pricing::SteepestEdgePricing;
use super::super::{build_standard_form, extract_dual_info, SimplexOutcome};
use super::core::revised_simplex_core;
use super::reconcile::{
    extract_solution, pivot_out_degenerate_artificials, reconcile_final_basis_state,
};
use crate::basis::{BasisManager, LuBasis};
use crate::options::SolverOptions;
use crate::problem::LpProblem;
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::{COMP_SLACK_REL_TOL, PIVOT_TOL};

/// Relative tolerance below which a standard-form column value is treated as
/// at-bound (zero) when seeding the crossover basis from `x_star`.
const CROSSOVER_ZERO_TOL: f64 = 1e-9;
const CROSSOVER_PHASE1_CLEANUP_SECS: f64 = 30.0;
const CROSSOVER_PHASE2_CLEANUP_SECS: f64 = 60.0;

#[allow(clippy::print_stderr)]
fn trace_crossover(msg: std::fmt::Arguments<'_>) {
    if crossover_trace_enabled() {
        eprintln!("[crossover] {msg}");
    }
}

fn crossover_trace_enabled() -> bool {
    std::env::var_os("OTSPOT_CROSSOVER_TRACE").is_some()
}

fn cleanup_deadline(
    caller_deadline: Option<std::time::Instant>,
    local_cap_secs: f64,
) -> Option<std::time::Instant> {
    let local = std::time::Instant::now() + std::time::Duration::from_secs_f64(local_cap_secs);
    Some(caller_deadline.map_or(local, |global| local.min(global)))
}

/// Bound-aware dual infeasibility of `y` against the reported primal `x_star`:
/// the worst per-variable reduced-cost sign violation. `0` iff `y` is KKT
/// dual-feasible and complementary with `x_star` (the metric `postsolve` and
/// `guard_lp_optimal` ultimately gate on). Used to pick the crossover dual that
/// is actually complementary with the *reported* primal.
fn crossover_dual_infeasibility(problem: &LpProblem, x_star: &[f64], y: &[f64]) -> f64 {
    let n = problem.num_vars;
    let mut max_viol = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = problem.bounds[j];
        let lb_s = if lb.is_finite() { lb.abs() } else { 0.0 };
        let ub_s = if ub.is_finite() { ub.abs() } else { 0.0 };
        let fixed = lb.is_finite()
            && ub.is_finite()
            && (ub - lb).abs() < COMP_SLACK_REL_TOL * (1.0 + lb_s.max(ub_s));
        if fixed {
            continue;
        }
        let at_lb = lb.is_finite() && (x_star[j] - lb).abs() < COMP_SLACK_REL_TOL * (1.0 + lb_s);
        let at_ub = ub.is_finite() && (x_star[j] - ub).abs() < COMP_SLACK_REL_TOL * (1.0 + ub_s);
        let mut rc = problem.c[j];
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                rc -= vals[k] * y[row];
            }
        }
        let viol = if at_lb && !at_ub {
            f64::max(0.0, -rc)
        } else if at_ub && !at_lb {
            f64::max(0.0, rc)
        } else {
            rc.abs()
        };
        if viol > max_viol {
            max_viol = viol;
        }
    }
    max_viol
}

fn crossover_dual_infeasibility_detail(
    problem: &LpProblem,
    x_star: &[f64],
    y: &[f64],
) -> Option<(usize, f64, f64, f64, f64, bool, bool)> {
    let n = problem.num_vars;
    let mut worst: Option<(usize, f64, f64, f64, f64, bool, bool)> = None;
    for j in 0..n {
        let (lb, ub) = problem.bounds[j];
        let lb_s = if lb.is_finite() { lb.abs() } else { 0.0 };
        let ub_s = if ub.is_finite() { ub.abs() } else { 0.0 };
        let fixed = lb.is_finite()
            && ub.is_finite()
            && (ub - lb).abs() < COMP_SLACK_REL_TOL * (1.0 + lb_s.max(ub_s));
        if fixed {
            continue;
        }
        let at_lb = lb.is_finite() && (x_star[j] - lb).abs() < COMP_SLACK_REL_TOL * (1.0 + lb_s);
        let at_ub = ub.is_finite() && (x_star[j] - ub).abs() < COMP_SLACK_REL_TOL * (1.0 + ub_s);
        let mut rc = problem.c[j];
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                rc -= vals[k] * y[row];
            }
        }
        let viol = if at_lb && !at_ub {
            f64::max(0.0, -rc)
        } else if at_ub && !at_lb {
            f64::max(0.0, rc)
        } else {
            rc.abs()
        };
        if worst.is_none_or(|(_, best, _, _, _, _, _)| viol > best) {
            worst = Some((j, viol, rc, x_star[j], lb, at_lb, at_ub));
        }
    }
    worst
}

/// Derive a globally dual-feasible dual for `problem` from its known optimal
/// primal `x_star` (postsolved original-space optimum) via primal crossover:
/// reconstruct an optimal basis *at* `x_star` and read `y = B⁻ᵀ c_B`.
///
///   1. Standard form + `x_star` → standard-form primal `x_std`.
///   2. Initial basis = slacks ± one artificial per `needs_artificial` row (a
///      permuted ±identity, provably non-singular).
///   3. Seat every support column (`x_std > 0`) via FTRAN pivots, so `B⁻¹b =
///      x_star` represents the optimal vertex (`B` stays non-singular).
///   4. Phase I drives residual artificials out (degenerate at feasible x*).
///   5. A no-perturbation Phase II takes only degenerate (step-0) pivots,
///      walking bases at the fixed vertex to a dual-feasible one.
///
/// Any optimal basis yields a dual-feasible dual, so this is degeneracy-robust
/// where incremental per-transform recovery can strand. Returns `(dual,
/// reduced_costs)` in original space, or `None` if the crossover cannot complete.
pub(crate) fn crossover_dual_from_primal(
    problem: &LpProblem,
    x_star: &[f64],
    deadline: Option<std::time::Instant>,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let t_total = std::time::Instant::now();
    let sf = build_standard_form(problem);
    let m = sf.m;
    let n_orig = problem.num_vars;
    let n_total = sf.n_total;
    let n_shifted = sf.n_shifted;
    if x_star.len() != n_orig || m == 0 {
        return None;
    }
    trace_crossover(format_args!(
        "start m={} n_orig={} n_total={} nnz={}",
        m,
        n_orig,
        n_total,
        sf.a.values.len()
    ));

    let options = SolverOptions {
        deadline,
        warm_start: None,
        ..Default::default()
    };

    // (1) x_star → standard-form primal x_std (variable shifts / free-var splits).
    let mut x_std = vec![0.0_f64; n_total];
    for j in 0..n_orig {
        let info = &sf.orig_var_info[j];
        let xj = x_star[j];
        if info.new_vars.len() == 2 {
            x_std[info.new_vars[0].0] = xj.max(0.0);
            x_std[info.new_vars[1].0] = (-xj).max(0.0);
        } else {
            let (idx, coeff) = info.new_vars[0];
            let val = if coeff > 0.0 {
                xj - info.offset
            } else {
                info.offset - xj
            };
            x_std[idx] = val.max(0.0);
        }
    }
    // Slack values from the structural row sums (each slack has one entry).
    let mut row_struct_sum = vec![0.0_f64; m];
    for j in 0..n_shifted {
        if x_std[j].abs() < CROSSOVER_ZERO_TOL {
            continue;
        }
        if let Ok((rows, vals)) = sf.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                row_struct_sum[row] += vals[k] * x_std[j];
            }
        }
    }
    for j in n_shifted..n_total {
        if let Ok((rows, vals)) = sf.a.get_column(j) {
            if rows.len() == 1 && vals[0].abs() > 0.0 {
                let i = rows[0];
                x_std[j] = ((sf.b[i] - row_struct_sum[i]) / vals[0]).max(0.0);
            }
        }
    }
    trace_crossover(format_args!(
        "x_std {:.3}s",
        t_total.elapsed().as_secs_f64()
    ));

    // (2) a_ext = A plus one artificial unit column per row with no slack basis
    // column. The basis (slacks ± artificials) is a permuted ±identity, hence
    // provably non-singular — unlike a partial LTSF crash, whose covered block
    // can be rank-deficient when columns are assigned with active count > 1.
    let mut basis = sf.initial_basis.clone();
    let mut tr: Vec<usize> = Vec::new();
    let mut tc: Vec<usize> = Vec::new();
    let mut tv: Vec<f64> = Vec::new();
    for j in 0..n_total {
        if let Ok((rows, vals)) = sf.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                tr.push(row);
                tc.push(j);
                tv.push(vals[k]);
            }
        }
    }
    let mut art = n_total;
    for i in 0..m {
        if sf.needs_artificial[i] {
            tr.push(i);
            tc.push(art);
            tv.push(1.0);
            basis[i] = art;
            art += 1;
        }
    }
    let n_ext = art;
    let a_ext = CscMatrix::from_triplets(&tr, &tc, &tv, m, n_ext).ok()?;
    trace_crossover(format_args!(
        "a_ext n_ext={} arts={} {:.3}s",
        n_ext,
        n_ext.saturating_sub(n_total),
        t_total.elapsed().as_secs_f64()
    ));

    // (3) x_star-driven refinement via FTRAN pivots: seat every support column
    // (x_std > 0 — structurals AND slacks) into the basis, displacing 0-valued
    // slacks / artificials. Pivoting on a nonzero (B⁻¹aⱼ)ᵢ keeps B non-singular
    // (a blind index swap does not). A non-binding Ge surplus slack starts
    // nonbasic, so seating slacks too is required or B⁻¹b ≠ x*.
    {
        let t_seat = std::time::Instant::now();
        let mut basis_mgr = LuBasis::new_timed(&a_ext, &basis, options.max_etas, deadline).ok()?;
        let mut is_basic = vec![false; n_ext];
        for &col in basis.iter() {
            is_basic[col] = true;
        }
        let removable = |col: usize| -> bool {
            col >= n_total || (col >= n_shifted && x_std[col] <= CROSSOVER_ZERO_TOL)
        };
        let mut active: Vec<(f64, usize)> = (0..n_total)
            .filter(|&j| x_std[j] > CROSSOVER_ZERO_TOL && !is_basic[j])
            .map(|j| (x_std[j], j))
            .collect();
        active.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let active_len = active.len();
        let mut seat_pivots = 0usize;
        for (_xj, j) in active {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                break;
            }
            let Ok((col_rows, col_vals)) = a_ext.get_column(j) else {
                continue;
            };
            let mut d_sv = SparseVec {
                indices: col_rows.to_vec(),
                values: col_vals.to_vec(),
                len: m,
            };
            basis_mgr.ftran(&mut d_sv);
            let mut best_row: Option<usize> = None;
            let mut best_abs = PIVOT_TOL;
            for (k, &row) in d_sv.indices.iter().enumerate() {
                let abs = d_sv.values[k].abs();
                if abs > best_abs && removable(basis[row]) {
                    best_abs = abs;
                    best_row = Some(row);
                }
            }
            if let Some(row) = best_row {
                is_basic[basis[row]] = false;
                is_basic[j] = true;
                basis_mgr.update(j, row, &d_sv);
                basis[row] = j;
                seat_pivots += 1;
                basis_mgr.refactor_if_needed_timed(&a_ext, &basis, deadline);
            }
        }
        trace_crossover(format_args!(
            "seat active={} pivots={} stage={:.3}s total={:.3}s",
            active_len,
            seat_pivots,
            t_seat.elapsed().as_secs_f64(),
            t_total.elapsed().as_secs_f64()
        ));
    }

    // (4) Reconcile x_B = B⁻¹b from a fresh LU (also detects a singular basis).
    let t_reconcile = std::time::Instant::now();
    let mut x_b = vec![0.0_f64; m];
    let mut y_tmp = vec![0.0_f64; m];
    let mut c_phase1 = vec![0.0_f64; n_ext];
    c_phase1[n_total..].fill(1.0);
    reconcile_final_basis_state(
        &a_ext,
        &sf.b,
        &c_phase1,
        &basis,
        &mut x_b,
        &mut y_tmp,
        options.max_etas,
        deadline,
    )
    .ok()?;
    for v in x_b.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    trace_crossover(format_args!(
        "reconcile1 stage={:.3}s total={:.3}s",
        t_reconcile.elapsed().as_secs_f64(),
        t_total.elapsed().as_secs_f64()
    ));

    // Phase I: drive any residual artificials out (degenerate at the feasible x*).
    if basis.iter().any(|&col| col >= n_total) {
        let t_phase1 = std::time::Instant::now();
        for i in 0..m {
            if basis[i] >= n_total && x_b[i].abs() <= PIVOT_TOL {
                x_b[i] = PIVOT_TOL * (i as f64 + 1.0);
            }
        }
        let mut pricing1 = SteepestEdgePricing::new(n_ext);
        let mut iters = 0usize;
        let mut phase1_options = options.clone();
        phase1_options.deadline = cleanup_deadline(deadline, CROSSOVER_PHASE1_CLEANUP_SECS);
        match revised_simplex_core(
            &a_ext,
            &mut x_b,
            &c_phase1,
            &sf.b,
            &mut basis,
            m,
            n_ext,
            n_ext,
            &mut pricing1,
            &phase1_options,
            &mut iters,
            true,
            Some(n_total),
            false,
            None,
        ) {
            SimplexOutcome::Optimal(_, _) => {}
            _ => return None,
        }
        trace_crossover(format_args!(
            "phase1 iters={} stage={:.3}s total={:.3}s",
            iters,
            t_phase1.elapsed().as_secs_f64(),
            t_total.elapsed().as_secs_f64()
        ));
        for i in 0..m {
            if basis[i] >= n_total {
                x_b[i] = 0.0;
            }
        }
        // Artificials are zeroed and c_phase1 is zero for structurals, so
        // phase1_obj is identically 0 — basis singularity is caught by
        // reconcile_final_basis_state below.
        let t_pivot_out = std::time::Instant::now();
        pivot_out_degenerate_artificials(&a_ext, &mut basis, &x_b, &sf, &options);
        trace_crossover(format_args!(
            "pivot_out_arts stage={:.3}s total={:.3}s",
            t_pivot_out.elapsed().as_secs_f64(),
            t_total.elapsed().as_secs_f64()
        ));
    }

    // (5) Read the dual at the x*-representing basis. Its BFS is x*, so its dual
    // is KKT-complementary with x*. When x* is a degenerate vertex this basis may
    // not yet be dual-feasible; a perturbation-free Phase II then walks the bases
    // at the fixed vertex (degenerate, step-0 pivots) to a dual-feasible one.
    let mut c_phase2 = vec![0.0_f64; n_ext];
    c_phase2[..n_total].copy_from_slice(&sf.c[..n_total]);
    let row_scale = vec![1.0_f64; m];

    let mut y = vec![0.0_f64; m];
    let t_dual1 = std::time::Instant::now();
    if reconcile_final_basis_state(
        &a_ext,
        &sf.b,
        &c_phase2,
        &basis,
        &mut x_b,
        &mut y,
        options.max_etas,
        deadline,
    )
    .is_err()
    {
        return None;
    }
    let vertex1 = extract_solution(&sf, &basis, &x_b, &[]);
    let (dual1, rc1, _) = extract_dual_info(&sf, problem, &y, &vertex1, &row_scale);
    let df1 = crossover_dual_infeasibility(problem, &vertex1, &dual1);
    trace_crossover(format_args!(
        "dual1 df={:.3e} stage={:.3}s total={:.3}s",
        df1,
        t_dual1.elapsed().as_secs_f64(),
        t_total.elapsed().as_secs_f64()
    ));
    if crossover_trace_enabled() {
        trace_crossover(format_args!(
            "dual1 detail {:?}",
            crossover_dual_infeasibility_detail(problem, &vertex1, &dual1)
        ));
    }
    if df1 <= crate::qp::certificate::LP_CERT_TOL {
        return Some((vertex1, dual1, rc1));
    }

    for v in x_b.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    let mut pricing2 = SteepestEdgePricing::new(n_ext);
    let mut iters2 = 0usize;
    let t_phase2 = std::time::Instant::now();
    let mut phase2_options = options.clone();
    phase2_options.deadline = cleanup_deadline(deadline, CROSSOVER_PHASE2_CLEANUP_SECS);
    let phase2 = revised_simplex_core(
        &a_ext,
        &mut x_b,
        &c_phase2,
        &sf.b,
        &mut basis,
        m,
        n_ext,
        n_total,
        &mut pricing2,
        &phase2_options,
        &mut iters2,
        false,
        None,
        true,
        Some(crate::qp::certificate::LP_CERT_TOL),
    );
    match phase2 {
        SimplexOutcome::Optimal(_, mut y2) => {
            trace_crossover(format_args!(
                "phase2 optimal iters={} stage={:.3}s total={:.3}s",
                iters2,
                t_phase2.elapsed().as_secs_f64(),
                t_total.elapsed().as_secs_f64()
            ));
            if reconcile_final_basis_state(
                &a_ext,
                &sf.b,
                &c_phase2,
                &basis,
                &mut x_b,
                &mut y2,
                options.max_etas,
                deadline,
            )
            .is_err()
            {
                return Some((vertex1, dual1, rc1));
            }
            let vertex2 = extract_solution(&sf, &basis, &x_b, &[]);
            let (dual2, rc2, _) = extract_dual_info(&sf, problem, &y2, &vertex2, &row_scale);
            let df2 = crossover_dual_infeasibility(problem, &vertex2, &dual2);
            trace_crossover(format_args!(
                "finish df1={:.3e} df2={:.3e} total={:.3}s",
                df1,
                df2,
                t_total.elapsed().as_secs_f64()
            ));
            if crossover_trace_enabled() {
                trace_crossover(format_args!(
                    "finish detail {:?}",
                    crossover_dual_infeasibility_detail(problem, &vertex2, &dual2)
                ));
            }
            if df2 < df1 {
                Some((vertex2, dual2, rc2))
            } else {
                Some((vertex1, dual1, rc1))
            }
        }
        _ => {
            let mut y_partial = vec![0.0_f64; m];
            let partial = if reconcile_final_basis_state(
                &a_ext,
                &sf.b,
                &c_phase2,
                &basis,
                &mut x_b,
                &mut y_partial,
                options.max_etas,
                deadline,
            )
            .is_ok()
            {
                let vertex_p = extract_solution(&sf, &basis, &x_b, &[]);
                let (dual_p, rc_p, _) =
                    extract_dual_info(&sf, problem, &y_partial, &vertex_p, &row_scale);
                let df_p = crossover_dual_infeasibility(problem, &vertex_p, &dual_p);
                trace_crossover(format_args!(
                    "phase2 partial iters={} df={:.3e} stage={:.3}s total={:.3}s",
                    iters2,
                    df_p,
                    t_phase2.elapsed().as_secs_f64(),
                    t_total.elapsed().as_secs_f64()
                ));
                if crossover_trace_enabled() {
                    trace_crossover(format_args!(
                        "phase2 partial detail {:?}",
                        crossover_dual_infeasibility_detail(problem, &vertex_p, &dual_p)
                    ));
                }
                Some((df_p, vertex_p, dual_p, rc_p))
            } else {
                None
            };
            trace_crossover(format_args!(
                "phase2 fallback iters={} stage={:.3}s total={:.3}s df1={:.3e}",
                iters2,
                t_phase2.elapsed().as_secs_f64(),
                t_total.elapsed().as_secs_f64(),
                df1
            ));
            if let Some((df_p, vertex_p, dual_p, rc_p)) = partial {
                if df_p < df1 {
                    Some((vertex_p, dual_p, rc_p))
                } else {
                    Some((vertex1, dual1, rc1))
                }
            } else {
                Some((vertex1, dual1, rc1))
            }
        }
    }
}

pub(crate) fn crossover_dual_from_primal_with_dual_warm_start(
    problem: &LpProblem,
    x_star: &[f64],
    ipm_dual_prove: Option<&[f64]>,
    deadline: Option<std::time::Instant>,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let warm = ipm_dual_prove.and_then(|y_prove| {
        if y_prove.len() != problem.num_constraints || x_star.len() != problem.num_vars {
            return None;
        }
        let y_simplex: Vec<f64> = y_prove.iter().map(|&v| -v).collect();
        let mut rc = problem.c.clone();
        for (j, rc_j) in rc.iter_mut().enumerate().take(problem.num_vars) {
            let Ok((rows, vals)) = problem.a.get_column(j) else {
                return None;
            };
            for (k, &row) in rows.iter().enumerate() {
                *rc_j -= vals[k] * y_simplex[row];
            }
        }
        let df = crossover_dual_infeasibility(problem, x_star, &y_simplex);
        trace_crossover(format_args!("warm_ipm_dual df={:.3e}", df));
        Some((df, x_star.to_vec(), y_simplex, rc))
    });

    if let Some((df, vertex, dual, rc)) = warm.as_ref() {
        if *df <= crate::qp::certificate::LP_CERT_TOL {
            trace_crossover(format_args!("warm_ipm_dual accept df={:.3e}", df));
            return Some((vertex.clone(), dual.clone(), rc.clone()));
        }
    }

    let cold = crossover_dual_from_primal(problem, x_star, deadline);
    match (warm, cold) {
        (
            Some((warm_df, warm_vertex, warm_dual, warm_rc)),
            Some((cold_vertex, cold_dual, cold_rc)),
        ) => {
            let cold_df = crossover_dual_infeasibility(problem, &cold_vertex, &cold_dual);
            trace_crossover(format_args!(
                "warm_vs_basis warm_df={:.3e} basis_df={:.3e}",
                warm_df, cold_df
            ));
            if warm_df < cold_df {
                Some((warm_vertex, warm_dual, warm_rc))
            } else {
                Some((cold_vertex, cold_dual, cold_rc))
            }
        }
        (None, cold) => cold,
        (Some((_warm_df, warm_vertex, warm_dual, warm_rc)), None) => {
            Some((warm_vertex, warm_dual, warm_rc))
        }
    }
}

#[cfg(test)]
mod crossover_tests {
    //! `crossover_dual_from_primal` reconstructs an optimal basis at a known
    //! primal optimum `x*` and reads `y = B⁻ᵀ c_B`. The contract: the returned
    //! dual is KKT dual-feasible AND complementary with `x*`
    //! (`crossover_dual_infeasibility ≈ 0`), across constraint senses, free
    //! variables, finite upper bounds, and non-binding Ge rows.
    use super::{crossover_dual_from_primal, crossover_dual_infeasibility};
    use crate::problem::{ConstraintType, LpProblem};
    use crate::sparse::CscMatrix;

    /// Tolerance for "dual-feasible & complementary with x*".
    const DF_TOL: f64 = 1e-7;

    fn assert_crossover_complementary(problem: &LpProblem, x_star: &[f64], label: &str) {
        let (_vertex, y, rc) = crossover_dual_from_primal(problem, x_star, None)
            .unwrap_or_else(|| panic!("{label}: crossover returned None"));
        assert_eq!(y.len(), problem.num_constraints, "{label}: dual length");
        assert_eq!(rc.len(), problem.num_vars, "{label}: rc length");
        let df = crossover_dual_infeasibility(problem, x_star, &y);
        assert!(
            df < DF_TOL,
            "{label}: dual infeasibility {df:.3e} must be ~0 — the crossover dual \
             must be KKT-feasible and complementary with x* (y={y:?})"
        );
    }

    fn lp(
        c: Vec<f64>,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        b: Vec<f64>,
        ct: Vec<ConstraintType>,
        bounds: Vec<(f64, f64)>,
    ) -> LpProblem {
        let m = b.len();
        let a = CscMatrix::from_triplets(rows, cols, vals, m, c.len()).unwrap();
        LpProblem::new_general(c, a, b, ct, bounds, None).unwrap()
    }

    /// Le-only LP, unique optimum. min -x1-x2 s.t. x1+2x2<=4, 3x1+x2<=6.
    /// Optimum x*=(1.6, 1.2): both Le binding, both x interior.
    #[test]
    fn crossover_le_unique_optimum() {
        let p = lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 2.0, 3.0, 1.0],
            vec![4.0, 6.0],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
        );
        assert_crossover_complementary(&p, &[1.6, 1.2], "le_unique");
    }

    /// Equality constraint (artificial in standard form) + free variable (± split).
    /// min x1 + x2 s.t. x1 + x2 = 3 (Eq), x1 free, x2 >= 0. Optimum x*=(3,0).
    #[test]
    fn crossover_eq_with_free_var() {
        let p = lp(
            vec![1.0, 1.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            vec![3.0],
            vec![ConstraintType::Eq],
            vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, f64::INFINITY)],
        );
        assert_crossover_complementary(&p, &[3.0, 0.0], "eq_free");
    }

    /// Finite upper bound (UB row in standard form). min -x1 s.t. x1+x2<=3,
    /// x1 ∈ [0,2], x2 ∈ [0,5]. Optimum x*=(2,0): x1 at UB.
    #[test]
    fn crossover_finite_upper_bound() {
        let p = lp(
            vec![-1.0, 0.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            vec![3.0],
            vec![ConstraintType::Le],
            vec![(0.0, 2.0), (0.0, 5.0)],
        );
        assert_crossover_complementary(&p, &[2.0, 0.0], "finite_ub");
    }

    /// SENTINEL (support-slack seating): a Ge row that is NON-binding at the
    /// optimum, so its surplus slack is a support column (value > 0) that starts
    /// NONBASIC (the Ge row is seeded with an artificial). If the refinement seats
    /// only structural support columns (not slacks), B⁻¹b ≠ x* and the dual is
    /// wrong. min -x1 s.t. x1<=2 (Le), x1+x2>=1 (Ge, surplus=1>0 at opt),
    /// x1,x2 ∈ [0,10]. Optimum x*=(2,0): y0=-1 (Le), y1=0 (Ge non-binding).
    #[test]
    fn crossover_seats_support_slack_on_nonbinding_ge() {
        let p = lp(
            vec![-1.0, 0.0],
            &[0, 1, 1],
            &[0, 0, 1],
            &[1.0, 1.0, 1.0],
            vec![2.0, 1.0],
            vec![ConstraintType::Le, ConstraintType::Ge],
            vec![(0.0, 10.0), (0.0, 10.0)],
        );
        assert_crossover_complementary(&p, &[2.0, 0.0], "ge_nonbinding");
    }

    /// Degenerate optimum: x*=(1,1) with THREE binding constraints (x1<=1,
    /// x2<=1, x1+x2<=2) but only 2 structurals — a degenerate vertex represented
    /// by several bases; the crossover must reach a dual-feasible one. min -x1-x2.
    #[test]
    fn crossover_degenerate_vertex() {
        let p = lp(
            vec![-1.0, -1.0],
            &[0, 1, 2, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 2.0],
            vec![ConstraintType::Le, ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
        );
        assert_crossover_complementary(&p, &[1.0, 1.0], "degenerate");
    }
}
