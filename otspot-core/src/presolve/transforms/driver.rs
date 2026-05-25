//! Orchestrator: fixpoint loop over Steps 1–11 and the final reduced-problem build.

use super::bounds::step5_bounds_tightening;
use super::doubleton::step6_doubleton_equation;
use super::empty_redundant::{step3a_empty_row, step3b_empty_column, step4_redundant_constraint};
use super::fixed::step1_fixed_variable;
use super::free::{step7_free_var_substitution, step8_free_singleton_col};
use super::singleton::step2_singleton_row;
use super::state::{PresolveFlags, PresolveResult, PresolveState, PresolveStatus};
use crate::problem::{ConstraintType, LpProblem};
use crate::sparse::CscMatrix;
use crate::tolerances::ZERO_TOL;

pub fn run_presolve(
    problem: &LpProblem,
    deadline: Option<std::time::Instant>,
) -> Result<PresolveResult, PresolveStatus> {
    run_presolve_with_flags(problem, deadline, PresolveFlags::default())
}

/// Variant of `run_presolve` with per-transform flags. Production callers use
/// the default-flag wrapper above; sentinel / bench-gating callers vary flags
/// to isolate each transform's contribution.
pub fn run_presolve_with_flags(
    problem: &LpProblem,
    deadline: Option<std::time::Instant>,
    flags: PresolveFlags,
) -> Result<PresolveResult, PresolveStatus> {
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return Ok(PresolveResult::no_reduction(problem));
    }

    let n = problem.num_vars;
    let m = problem.num_constraints;
    let mut st = PresolveState::from_problem(problem);

    // Loop until reduction == 0. Each step removes finitely many elements, so this
    // terminates; the per-step deadline check is the only safety bound.
    loop {
        let prev_removed = st.removed_cols.iter().filter(|&&r| r).count()
            + st.removed_rows.iter().filter(|&&r| r).count();
        let mut new_fixed_by_step5 = 0usize;
        let mut new_subst_steps = 0usize;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step1_fixed_variable(&mut st)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step2_singleton_row(&mut st)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step3a_empty_row(&mut st)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step3b_empty_column(&mut st)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step4_redundant_constraint(&mut st)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step5_bounds_tightening(&mut st, &mut new_fixed_by_step5)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step6_doubleton_equation(&mut st, &mut new_subst_steps)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step7_free_var_substitution(&mut st, &mut new_subst_steps)?;

        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(PresolveResult::no_reduction(problem));
        }
        step8_free_singleton_col(&mut st, &mut new_subst_steps)?;

        if flags.enable_parallel_row {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return Ok(PresolveResult::no_reduction(problem));
            }
            crate::presolve::transforms_dup::step9_parallel_row(&mut st)?;
        }
        if flags.enable_dup_dom_col {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return Ok(PresolveResult::no_reduction(problem));
            }
            crate::presolve::transforms_dup::step10_dup_dom_col(&mut st, &mut new_fixed_by_step5)?;
        }
        if flags.enable_dual_fixing {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return Ok(PresolveResult::no_reduction(problem));
            }
            crate::presolve::transforms_dup::step11_dual_fixing(&mut st, &mut new_fixed_by_step5)?;
        }

        let curr_removed = st.removed_cols.iter().filter(|&&r| r).count()
            + st.removed_rows.iter().filter(|&&r| r).count();
        let reduction = curr_removed - prev_removed;
        if reduction == 0 && new_fixed_by_step5 == 0 && new_subst_steps == 0 {
            break;
        }
    }

    build_reduced_result(problem, st, n, m)
}

fn build_reduced_result(
    problem: &LpProblem,
    st: PresolveState,
    n: usize,
    m: usize,
) -> Result<PresolveResult, PresolveStatus> {
    let mut col_map = vec![None; n];
    let mut new_col_idx = 0usize;
    for j in 0..n {
        if !st.removed_cols[j] {
            col_map[j] = Some(new_col_idx);
            new_col_idx += 1;
        }
    }
    let n_new = new_col_idx;

    let mut row_map = vec![None; m];
    let mut new_row_idx = 0usize;
    for i in 0..m {
        if !st.removed_rows[i] {
            row_map[i] = Some(new_row_idx);
            new_row_idx += 1;
        }
    }
    let m_new = new_row_idx;

    let was_reduced = n_new < n || m_new < m;

    let mut c_new = vec![0.0f64; n_new];
    let mut bounds_new = vec![(0.0f64, f64::INFINITY); n_new];
    for j in 0..n {
        if let Some(jj) = col_map[j] {
            c_new[jj] = st.c[j];
            bounds_new[jj] = st.bounds[j];
        }
    }

    let mut b_new = vec![0.0f64; m_new];
    let mut ct_new = vec![ConstraintType::Le; m_new];
    for i in 0..m {
        if let Some(ii) = row_map[i] {
            b_new[ii] = st.b[i];
            ct_new[ii] = st.constraint_types[i];
        }
    }

    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
    for j in 0..n {
        if st.removed_cols[j] {
            continue;
        }
        let jj = col_map[j].unwrap();
        for &(row, val) in &st.col_entries[j] {
            if st.removed_rows[row] || val.abs() < ZERO_TOL {
                continue;
            }
            let ii = row_map[row].unwrap();
            trip_rows.push(ii);
            trip_cols.push(jj);
            trip_vals.push(val);
        }
    }

    let a_new = if trip_rows.is_empty() {
        CscMatrix::new(m_new, n_new)
    } else {
        CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_new, n_new)
            .unwrap_or_else(|_| CscMatrix::new(m_new, n_new))
    };

    let reduced_problem = LpProblem::new_general(
        c_new,
        a_new,
        b_new,
        ct_new,
        bounds_new,
        problem.name.clone(),
    )
    .expect("presolve: reduced problem construction failed");

    Ok(PresolveResult {
        reduced_problem,
        postsolve_stack: st.postsolve_stack,
        orig_num_vars: n,
        orig_num_constraints: m,
        col_map,
        row_map,
        was_reduced,
        obj_offset: st.obj_offset,
    })
}
