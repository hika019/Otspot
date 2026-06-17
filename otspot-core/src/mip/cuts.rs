//! Root Gomory Mixed-Integer (GMI) cutting planes for MILP.
//!
//! Cuts tighten the LP relaxation without removing integer-feasible points.
//! Uses the primal-simplex legacy standard form (`build_standard_form`) where
//! every nonbasic sits at 0; the bounded/Ruiz-scaled dual path would place
//! nonbasics at upper bounds and scaled values, complicating the GMI formula
//! and risking invalid cuts.
//!
//! Pipeline per round:
//!   1. Solve the LP (primal simplex, no presolve) for a standard-form basis.
//!   2. Recompute `beta = B^{-1} b_std` via FTRAN.
//!   3. For each fractional integer basic, form the tableau row (BTRAN + dot)
//!      and emit a GMI cut over the nonbasic columns.
//!   4. Back-substitute to original variables (`v_j = d_j + g_j·x`, `G·x >= rhs`).
//!
//! GMI validity (coefficients >= 0, integer hull) is structural; a brute-force
//! sentinel in tests checks every integer-feasible point satisfies every cut.

use crate::basis::{BasisManager, LuBasis};
use crate::options::{MipConfig, SimplexMethod, SolverOptions, DEFAULT_MAX_CUT_ROUNDS};
use crate::problem::{ConstraintType, LpProblem, SolveStatus};
use crate::simplex::{build_standard_form, StandardForm};
use crate::sparse::CscMatrix;
use crate::tolerances::{feas_rel_tol, ZERO_TOL};

use super::problem::MilpProblem;

/// Maximum GMI cuts added per round. Bounds LP growth and numerical pollution;
/// the most-fractional sources (closest to 0.5) are kept first.
const MAX_CUTS_PER_ROUND: usize = 64;
/// Relative LP-bound improvement below which cut rounds stop (diminishing return).
const MIN_OBJ_IMPROVEMENT_REL: f64 = 1e-4;
/// Reject a cut whose coefficient magnitudes span more than this ratio: such a
/// row is numerically unreliable in f64 (≈9 orders is already at the edge).
const GMI_MAX_COEF_DYNAMISM: f64 = 1e9;
/// Fraction of the remaining solve budget root cut generation may consume. The
/// rest is reserved for branch-and-bound, so cuts never starve the search (an
/// unbudgeted loop on a large LP can eat the whole deadline → B&B runs 0 nodes).
const CUT_TIME_FRACTION: f64 = 0.3;

/// A generated cut `coeffs · x >= rhs` (always stored as a `Ge` row over the
/// original variable space).
struct CutRow {
    /// Dense coefficient vector over the original variables (length `num_vars`).
    coeffs: Vec<f64>,
    rhs: f64,
}

/// Classification of an original variable's structural standard-form column.
#[derive(Clone, Copy)]
enum StructKind {
    /// `x_std = x_p - lb`; integer iff `x_p` integer and `lb` integer.
    LbShift,
    /// `x_std = ub - x_p` (lb = -inf, ub finite); integer iff `x_p`, `ub` integer.
    UbOnly,
    /// Half of a free-variable split (`x_p = x_plus - x_minus`); no single-var
    /// affine image, so any cut whose support touches it is rejected.
    FreeSplit,
}

/// Per-structural-column metadata (length `n_shifted`).
#[derive(Clone, Copy)]
struct StructCol {
    var: usize,
    offset: f64,
    kind: StructKind,
    /// `x_std` is integer-constrained (var is integer and the shift is integral).
    integral: bool,
}

/// What a standard-form slack column measures, in original variables.
#[derive(Clone, Copy)]
enum SlackKind {
    /// Original `Le` row `i`: `v = b_i - A_i·x`.
    ConstraintLe(usize),
    /// Original `Ge` row `i`: `v = A_i·x - b_i`.
    ConstraintGe(usize),
    /// Upper-bound row for bounded var `p`: `v = ub_p - x_p`.
    UbRow(usize),
}

/// Append GMI cuts found at the root to `milp`, returning the augmented problem.
///
/// When no usable cut is found (or the LP cannot be solved cleanly) the input is
/// returned unchanged. The integer-variable set is preserved; only inequality
/// rows are added, all valid for the integer hull, so the MILP optimum is
/// unchanged (verified by the optimality-invariance sentinel).
pub(crate) fn add_root_cuts(
    milp: &MilpProblem,
    options: &SolverOptions,
    cfg: &MipConfig,
) -> MilpProblem {
    let integer_mask = super::integer_mask(milp.lp.num_vars, &milp.integer_vars);
    let max_rounds = if cfg.max_cut_rounds == 0 {
        DEFAULT_MAX_CUT_ROUNDS
    } else {
        cfg.max_cut_rounds
    };

    // `committed` is always an LP that solves cleanly; cuts are committed only
    // after the candidate is re-validated with the *same* solver the B&B uses, so
    // the returned root is never one B&B cannot solve. (Higher-round GMI cuts can
    // degrade numerical conditioning; without this guard an unsolvable root would
    // be handed to B&B and surface as a spurious Timeout.)
    let mut committed = milp.lp.clone();
    let mut prev_obj: Option<f64> = None;
    // Hard ceiling on total cut rows so the relaxation never more than doubles in
    // size — beyond this, every downstream B&B-node LP solve slows enough to starve
    // the search (observed: 6+ rounds on p0201 drive B&B to 0 nodes).
    let max_total_cuts = milp.lp.num_constraints.max(MAX_CUTS_PER_ROUND);
    let mut total_cuts = 0usize;

    // Bound cut work to a fraction of the remaining budget (reserves time for B&B).
    let cut_deadline = options.deadline.map(|d| {
        let now = std::time::Instant::now();
        now + d.saturating_duration_since(now).mul_f64(CUT_TIME_FRACTION)
    });

    for _ in 0..max_rounds {
        if deadline_passed(cut_deadline) {
            break;
        }
        // Primal solve (no presolve) gives a legacy-standard-form basis for GMI.
        let res = solve_cut_lp(&committed, options, cut_deadline);
        if res.status != SolveStatus::Optimal {
            break;
        }
        let Some(ws) = res.warm_start_basis.as_ref() else {
            break;
        };
        if let Some(po) = prev_obj {
            let scale = 1.0_f64.max(po.abs());
            if (res.objective - po).abs() <= MIN_OBJ_IMPROVEMENT_REL * scale {
                break;
            }
        }
        prev_obj = Some(res.objective);

        let cuts = generate_round(&committed, &integer_mask, &res.solution, &ws.basis);
        if cuts.is_empty() {
            break;
        }
        let candidate = append_ge_rows(&committed, &cuts);
        // Reject the round if the augmented LP no longer solves to Optimal under
        // the B&B solver (numerically unstable cuts): keep the last good LP.
        let check = solve_validate(&candidate, options, cut_deadline);
        if check.status != SolveStatus::Optimal {
            break;
        }
        committed = candidate;
        total_cuts += cuts.len();
        if total_cuts >= max_total_cuts {
            break;
        }
    }

    MilpProblem {
        lp: committed,
        integer_vars: milp.integer_vars.clone(),
    }
}

/// Solve the cut-augmented candidate with the solver / presolve the B&B will use
/// at the root, to confirm the added cuts did not make the relaxation
/// numerically unsolvable.
fn solve_validate(
    lp: &LpProblem,
    options: &SolverOptions,
    deadline: Option<std::time::Instant>,
) -> crate::problem::SolverResult {
    let opts = SolverOptions {
        recover_warm_start_basis: false,
        warm_start: None,
        warm_start_lp: None,
        deadline,
        timeout_secs: None,
        primal_tol: options.primal_tol,
        dual_tol: options.dual_tol,
        threads: options.threads,
        tolerance: options.tolerance,
        ..SolverOptions::default()
    };
    crate::lp::solve_lp_with(lp, &opts)
}

/// Solve the cut-generation LP via the primal simplex with presolve disabled so
/// the returned basis lives in `build_standard_form(lp)`'s column space.
fn solve_cut_lp(
    lp: &LpProblem,
    options: &SolverOptions,
    deadline: Option<std::time::Instant>,
) -> crate::problem::SolverResult {
    let opts = SolverOptions {
        presolve: false,
        simplex_method: SimplexMethod::Primal,
        recover_warm_start_basis: true,
        warm_start: None,
        warm_start_lp: None,
        deadline,
        timeout_secs: None,
        primal_tol: options.primal_tol,
        dual_tol: options.dual_tol,
        threads: options.threads,
        ..SolverOptions::default()
    };
    crate::lp::solve_lp_with(lp, &opts)
}

fn deadline_passed(deadline: Option<std::time::Instant>) -> bool {
    deadline.is_some_and(|d| std::time::Instant::now() >= d)
}

/// Generate one round of GMI cuts from the optimal `basis` (legacy std-form
/// column indices) and LP solution `x_star`.
fn generate_round(
    lp: &LpProblem,
    integer_mask: &[bool],
    x_star: &[f64],
    basis: &[usize],
) -> Vec<CutRow> {
    let sf = build_standard_form(lp);
    if basis.len() != sf.m {
        return Vec::new();
    }
    // An artificial still basic (index >= n_total) means the basis is not a clean
    // structural+slack basis; skip rather than mis-index A_std.
    if basis.iter().any(|&j| j >= sf.n_total) {
        return Vec::new();
    }

    let frac_tol = feas_rel_tol();
    let struct_cols = classify_struct_cols(&sf, integer_mask);
    let slack_kinds = classify_slack_cols(lp, &sf);
    let rows = row_lists(&lp.a, lp.num_constraints);

    let mut in_basis = vec![false; sf.n_total];
    for &j in basis {
        in_basis[j] = true;
    }

    // Reconstruct the basis factorization and recompute the unscaled basic values
    // beta = B^{-1} b_std (FTRAN). The basis column SET is scale-invariant, so the
    // primal path's Ruiz scaling does not affect which columns are basic.
    let Ok(mut lu) = LuBasis::new_timed(&sf.a, basis, 0, None) else {
        return Vec::new();
    };
    let mut beta = sf.b.clone();
    lu.ftran_dense(&mut beta);

    // Candidate source rows: basic, structural, integer-constrained, fractional.
    let mut sources: Vec<(usize, f64)> = Vec::new();
    for (i, &col) in basis.iter().enumerate() {
        if col >= sf.n_shifted {
            continue;
        }
        let sc = struct_cols[col];
        if !sc.integral {
            continue;
        }
        let b = beta[i];
        let f0 = b - b.floor();
        if f0 <= frac_tol || f0 >= 1.0 - frac_tol {
            continue;
        }
        sources.push((i, (f0 - 0.5).abs()));
    }
    // Most-fractional first (closest to 0.5), capped per round.
    sources.sort_by(|a, b| a.1.total_cmp(&b.1));
    sources.truncate(MAX_CUTS_PER_ROUND);

    let mut cuts = Vec::new();
    for (i, _) in sources {
        if let Some(cut) = build_gmi_cut(
            &sf,
            &mut lu,
            &beta,
            i,
            &in_basis,
            &struct_cols,
            &slack_kinds,
            &rows,
            lp,
            x_star,
            frac_tol,
        ) {
            cuts.push(cut);
        }
    }
    cuts
}

/// Build the GMI cut from basis row `i` (whose basic variable is fractional).
/// Returns `None` when the cut is unusable (free-var support, degenerate, weak,
/// or numerically unstable).
#[allow(clippy::too_many_arguments)]
fn build_gmi_cut(
    sf: &StandardForm,
    lu: &mut LuBasis,
    beta: &[f64],
    i: usize,
    in_basis: &[bool],
    struct_cols: &[StructCol],
    slack_kinds: &[Option<SlackKind>],
    rows: &[Vec<(usize, f64)>],
    lp: &LpProblem,
    x_star: &[f64],
    frac_tol: f64,
) -> Option<CutRow> {
    let f0 = {
        let b = beta[i];
        b - b.floor()
    };
    // f0 ∈ (frac_tol, 1-frac_tol) guaranteed by the caller's source filter.
    let one_minus_f0 = 1.0 - f0;

    // rho = B^{-T} e_i (BTRAN of the i-th unit vector).
    let mut rho = vec![0.0; sf.m];
    rho[i] = 1.0;
    lu.btran_dense(&mut rho);

    // Accumulate the cut G·x >= 1 - D from gamma_j · v_j over nonbasic columns.
    let mut g = vec![0.0; lp.num_vars];
    let mut d = 0.0_f64;

    for j in 0..sf.n_total {
        if in_basis[j] {
            continue;
        }
        let alpha = column_dot(&sf.a, j, &rho);
        if alpha.abs() <= ZERO_TOL {
            continue;
        }
        let integral = j < sf.n_shifted && struct_cols[j].integral;
        let gamma = gmi_coeff(alpha, f0, one_minus_f0, integral);
        if gamma <= ZERO_TOL {
            continue;
        }
        if !accumulate_column(j, gamma, sf, struct_cols, slack_kinds, rows, lp, &mut g, &mut d) {
            // Free-variable split column in the support: no affine image → reject.
            return None;
        }
    }

    let rhs = 1.0 - d;
    finalize_cut(g, rhs, x_star, frac_tol)
}

/// GMI coefficient (always >= 0) for a nonbasic column with tableau entry
/// `alpha`. `integral` selects the integer (fractional-part) formula; otherwise
/// the continuous formula is used (always valid, possibly weaker).
fn gmi_coeff(alpha: f64, f0: f64, one_minus_f0: f64, integral: bool) -> f64 {
    if integral {
        let f = (alpha - alpha.floor()).clamp(0.0, 1.0);
        if f <= f0 {
            f / f0
        } else {
            (1.0 - f) / one_minus_f0
        }
    } else if alpha > 0.0 {
        alpha / f0
    } else {
        -alpha / one_minus_f0
    }
}

/// Add `gamma · v_j` (where `v_j = d_j + g_j·x`) into the accumulators `g`, `d`.
/// Returns `false` if the column is a free-variable split (no affine image).
#[allow(clippy::too_many_arguments)]
fn accumulate_column(
    j: usize,
    gamma: f64,
    sf: &StandardForm,
    struct_cols: &[StructCol],
    slack_kinds: &[Option<SlackKind>],
    rows: &[Vec<(usize, f64)>],
    lp: &LpProblem,
    g: &mut [f64],
    d: &mut f64,
) -> bool {
    if j < sf.n_shifted {
        let sc = struct_cols[j];
        match sc.kind {
            StructKind::LbShift => {
                // v = x_p - lb  (d = -lb, g_p = +1)
                g[sc.var] += gamma;
                *d += gamma * (-sc.offset);
            }
            StructKind::UbOnly => {
                // v = ub - x_p  (d = ub, g_p = -1)
                g[sc.var] -= gamma;
                *d += gamma * sc.offset;
            }
            StructKind::FreeSplit => return false,
        }
    } else {
        match slack_kinds[j - sf.n_shifted] {
            Some(SlackKind::ConstraintLe(r)) => {
                // v = b_r - A_r·x  (d = b_r, g = -A_r)
                *d += gamma * lp.b[r];
                for &(c, v) in &rows[r] {
                    g[c] -= gamma * v;
                }
            }
            Some(SlackKind::ConstraintGe(r)) => {
                // v = A_r·x - b_r  (d = -b_r, g = A_r)
                *d += gamma * (-lp.b[r]);
                for &(c, v) in &rows[r] {
                    g[c] += gamma * v;
                }
            }
            Some(SlackKind::UbRow(p)) => {
                // v = ub_p - x_p  (d = ub_p, g_p = -1)
                let ub = lp.bounds[p].1;
                g[p] -= gamma;
                *d += gamma * ub;
            }
            None => {
                // A slack-range column that maps to None means the slack/kind
                // table is inconsistent (should be unreachable). Dropping the
                // term would silently strengthen the cut and risk invalidity.
                return false;
            }
        }
    }
    true
}

/// Dot product of standard-form column `j` with the dense vector `rho`.
fn column_dot(a: &CscMatrix, j: usize, rho: &[f64]) -> f64 {
    let (rows, vals) = a.get_column(j).expect("valid std-form column index");
    rows.iter()
        .zip(vals)
        .map(|(&r, &v)| v * rho[r])
        .sum::<f64>()
}

/// Apply density / strength / stability guards and build the final `Ge` cut.
fn finalize_cut(g: Vec<f64>, rhs: f64, x_star: &[f64], frac_tol: f64) -> Option<CutRow> {
    if !rhs.is_finite() || g.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut max_abs = 0.0_f64;
    let mut min_abs = f64::INFINITY;
    for &v in &g {
        let a = v.abs();
        if a > ZERO_TOL {
            max_abs = max_abs.max(a);
            min_abs = min_abs.min(a);
        }
    }
    // Degenerate (all-zero) cut: 0 >= rhs. Dropping it is correct — a valid GMI
    // can only degenerate by cancellation, never to a true `0 >= positive`.
    if max_abs <= ZERO_TOL {
        return None;
    }
    if max_abs / min_abs > GMI_MAX_COEF_DYNAMISM {
        return None;
    }
    // Effectiveness: the cut must actually violate the current LP optimum.
    let lhs: f64 = g.iter().zip(x_star).map(|(&gi, &xi)| gi * xi).sum();
    let violation = rhs - lhs; // Ge cut violated when g·x* < rhs
    if violation <= frac_tol * (1.0 + rhs.abs()) {
        return None;
    }
    Some(CutRow { coeffs: g, rhs })
}

/// Build per-structural-column metadata.
fn classify_struct_cols(sf: &StandardForm, integer_mask: &[bool]) -> Vec<StructCol> {
    let mut cols = vec![
        StructCol {
            var: 0,
            offset: 0.0,
            kind: StructKind::LbShift,
            integral: false,
        };
        sf.n_shifted
    ];
    for (p, info) in sf.orig_var_info.iter().enumerate() {
        let is_int = integer_mask[p];
        if info.new_vars.len() == 2 {
            // Free-variable split (x_plus, x_minus).
            for &(idx, _) in &info.new_vars {
                cols[idx] = StructCol {
                    var: p,
                    offset: 0.0,
                    kind: StructKind::FreeSplit,
                    integral: false,
                };
            }
            continue;
        }
        let (idx, coeff) = info.new_vars[0];
        let kind = if coeff > 0.0 {
            StructKind::LbShift // x_std = x_p - lb
        } else {
            StructKind::UbOnly // x_std = ub - x_p
        };
        // Integral in std space iff the variable is integer AND the shift is an
        // integer (so x_std integrality ⇔ x_p integrality).
        let shift_integral = (info.offset - info.offset.round()).abs() <= ZERO_TOL;
        cols[idx] = StructCol {
            var: p,
            offset: info.offset,
            kind,
            integral: is_int && shift_integral,
        };
    }
    cols
}

/// Build the slack-column → meaning map, replaying `build_standard_form`'s slack
/// assignment order over the extended row set (original rows then UB rows).
fn classify_slack_cols(lp: &LpProblem, sf: &StandardForm) -> Vec<Option<SlackKind>> {
    let n_slack = sf.n_total - sf.n_shifted;
    let mut kinds = vec![None; n_slack];

    // UB rows are appended for variables with both bounds finite, in var order.
    let ub_row_vars: Vec<usize> = (0..lp.num_vars)
        .filter(|&p| {
            let (lo, hi) = lp.bounds[p];
            lo.is_finite() && hi.is_finite()
        })
        .collect();

    let mut s = 0usize;
    // Original constraint rows.
    for (r, &ct) in lp.constraint_types.iter().enumerate() {
        match ct {
            ConstraintType::Le => {
                kinds[s] = Some(SlackKind::ConstraintLe(r));
                s += 1;
            }
            ConstraintType::Ge => {
                kinds[s] = Some(SlackKind::ConstraintGe(r));
                s += 1;
            }
            ConstraintType::Eq => { /* no slack column */ }
        }
    }
    // UB rows (all `Le`, always slacked).
    for &p in &ub_row_vars {
        assert!(
            s < n_slack,
            "UB-row count exceeds slack column count: s={s} >= n_slack={n_slack}"
        );
        kinds[s] = Some(SlackKind::UbRow(p));
        s += 1;
    }
    debug_assert_eq!(s, n_slack, "slack count mismatch vs standard form");
    kinds
}

/// Sparse rows of `A` (row → list of (col, value)).
fn row_lists(a: &CscMatrix, num_rows: usize) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); num_rows];
    for c in 0..a.ncols {
        let (rs, vs) = a.get_column(c).expect("valid column");
        for (&r, &v) in rs.iter().zip(vs) {
            rows[r].push((c, v));
        }
    }
    rows
}

/// Return a new LP with the cut rows appended as `Ge` constraints.
fn append_ge_rows(lp: &LpProblem, cuts: &[CutRow]) -> LpProblem {
    let m_old = lp.num_constraints;
    let n = lp.num_vars;
    let m_new = m_old + cuts.len();

    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
    for c in 0..lp.a.ncols {
        let (rs, vs) = lp.a.get_column(c).expect("valid column");
        for (&r, &v) in rs.iter().zip(vs) {
            trip_rows.push(r);
            trip_cols.push(c);
            trip_vals.push(v);
        }
    }
    for (k, cut) in cuts.iter().enumerate() {
        let r = m_old + k;
        for (col, &v) in cut.coeffs.iter().enumerate() {
            if v.abs() > ZERO_TOL {
                trip_rows.push(r);
                trip_cols.push(col);
                trip_vals.push(v);
            }
        }
    }
    let a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_new, n)
        .expect("cut-augmented A is well-formed");

    let mut b = lp.b.clone();
    let mut ctypes = lp.constraint_types.clone();
    for cut in cuts {
        b.push(cut.rhs);
        ctypes.push(ConstraintType::Ge);
    }

    let mut out = LpProblem::new_general(lp.c.clone(), a, b, ctypes, lp.bounds.clone(), lp.name.clone())
        .expect("cut-augmented LP is valid");
    out.obj_offset = lp.obj_offset;
    out
}

#[cfg(test)]
mod tests;
