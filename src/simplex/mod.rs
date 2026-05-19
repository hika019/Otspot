//! Two-phase revised simplex with LU-based basis updates.

pub(crate) mod crash;
pub mod dual;
pub mod dual_advanced;
mod dual_common;
mod entry;
pub mod pricing;
pub(crate) mod primal;
mod standard_form;

pub use entry::{solve, solve_with};
pub(crate) use entry::solve_without_presolve;

pub(crate) use primal::{extract_solution, revised_simplex_core, two_phase_simplex};
#[cfg(test)]
pub(crate) use primal::reconcile_final_basis_state;

pub(crate) use standard_form::{
    extract_dual_info, timeout_result_with_incumbent, SimplexOutcome, StandardForm,
};
pub(crate) use standard_form::build_standard_form;
#[cfg(test)]
pub(crate) use standard_form::OrigVarInfo;
#[allow(unused_imports)]
pub(crate) use standard_form::{build_bounded_standard_form, scale_upper_bounds, wrap_to_legacy, BoundedStandardForm};

/// crash basis → IPM warm-start helper (`lp_dispatch.rs` 経由)。
///
/// `LpProblem` から simplex 標準形を構築し、structural 列で artificial 行を
/// cover した crash basis を返す。crash が artificial を 1 つも減らせなければ
/// `None` (caller は Mehrotra cold init に着地する)。
///
/// 戻り値 `(sf, basis)`:
/// - `sf`: 標準形 (n_orig→n_total/m_ext 写像情報を保持)
/// - `basis`: 長さ m_ext、各 i は行 i を被覆する列 index
///   (`basis[i] < sf.n_total` なら structural/slack、>= は artificial placeholder)
pub(crate) fn crash_basis_for_ipm_warm(
    lp: &crate::problem::LpProblem,
) -> Option<(StandardForm, Vec<usize>)> {
    let sf = build_standard_form(lp);
    let num_art_in = sf.num_artificial;
    if num_art_in == 0 {
        // 全行 slack cover 済 → crash 不要 (cold init で十分 primal-feasible)。
        return None;
    }
    let (basis, _needs_art, num_art_out) = crash::compute_crash_basis(
        &sf.a,
        &sf.b,
        sf.m,
        sf.n_shifted,
        &sf.initial_basis,
        &sf.needs_artificial,
    );
    if num_art_out >= num_art_in {
        return None;
    }
    Some((sf, basis))
}

/// `extract_solution` を crate 内 LP→IPM crash wiring 用に再公開。
///
/// 既存 simplex Phase II 経路と共有 (TwoFloat split-var cancellation 対策含む)。
pub(crate) fn extract_solution_for_ipm_warm(
    sf: &StandardForm,
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
) -> Vec<f64> {
    extract_solution(sf, basis, x_b, col_scale)
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_dual_advanced;
#[cfg(test)]
mod tests_bounded_form;
