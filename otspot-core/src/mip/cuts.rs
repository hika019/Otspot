//! Root Gomory Mixed-Integer (GMI) and Mixed-Integer Rounding (MIR) cutting
//! planes for MILP.
//!
//! Cuts tighten the LP relaxation without removing integer-feasible points.
//! Uses the primal-simplex standard form (`build_standard_form`) so every
//! nonbasic sits at 0, simplifying the tableau row formulae.
//!
//! Pipeline per round:
//!   1. Solve the LP relaxation (primal simplex, no presolve).
//!   2. For each fractional integer basic, form the tableau row and emit a cut.
//!   3. Back-substitute to original variables (`G·x >= rhs`).
//!
//! GMI vs MIR: differ only in the coefficient for continuous nonbasics with a
//! negative tableau entry α — GMI uses `−α / (1 − f₀)`, MIR drops the term.

use crate::basis::{BasisManager, LuBasis};
use crate::linalg::timeout::deadline_reached;
use crate::options::{MipConfig, SimplexMethod, SolverOptions, DEFAULT_MAX_CUT_ROUNDS};
use crate::problem::{ConstraintType, LpProblem, SolveStatus};
use crate::simplex::{build_standard_form, StandardForm};
use crate::sparse::CscMatrix;
use crate::tolerances::{feas_rel_tol, ZERO_TOL};

use super::problem::MilpProblem;

/// Maximum cuts added per round.
const MAX_CUTS_PER_ROUND: usize = 64;
/// Relative LP-bound improvement below which cut rounds stop.
const MIN_OBJ_IMPROVEMENT_REL: f64 = 1e-4;
/// Reject a cut whose coefficient magnitudes span more than this ratio.
const GMI_MAX_COEF_DYNAMISM: f64 = 1e9;
/// Fraction of the remaining solve budget root cut generation may consume.
const CUT_TIME_FRACTION: f64 = 0.3;

/// A generated cut `coeffs · x >= rhs` over the original variable space.
struct CutRow {
    coeffs: Vec<f64>,
    rhs: f64,
}

/// Classification of an original variable's structural standard-form column.
#[derive(Clone, Copy)]
enum StructKind {
    LbShift,
    UbOnly,
    FreeSplit,
}

/// Per-structural-column metadata (length `n_shifted`).
#[derive(Clone, Copy)]
struct StructCol {
    var: usize,
    offset: f64,
    kind: StructKind,
    integral: bool,
}

/// What a standard-form slack column measures, in original variables.
#[derive(Clone, Copy)]
enum SlackKind {
    ConstraintLe(usize),
    ConstraintGe(usize),
    UbRow(usize),
}

/// Which cutting-plane formula to apply.
#[derive(Clone, Copy)]
enum CutKind {
    Gmi,
    Mir,
}

/// Append GMI and MIR cuts at the root, returning the augmented problem.
///
/// Rounds alternate GMI (even) / MIR (odd). Multi-round generation uses Ge rows
/// internally (preserving the original simplex path and numerical properties).
/// After all rounds, added Ge cut rows are converted to Le (`−g·x ≤ −rhs`) before
/// returning, so B&B node solves (primal simplex, presolve=false) use slack
/// variables (coeff +1) rather than surplus variables (coeff −1).
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
    let m_orig = milp.lp.num_constraints;
    let mut committed = milp.lp.clone();
    let mut prev_obj: Option<f64> = None;
    let max_total_cuts = m_orig.max(MAX_CUTS_PER_ROUND);
    let mut total_cuts = 0usize;

    let cut_deadline = options.deadline.map(|d| {
        let now = std::time::Instant::now();
        now + d.saturating_duration_since(now).mul_f64(CUT_TIME_FRACTION)
    });

    for round_idx in 0..max_rounds {
        if deadline_reached(cut_deadline) {
            break;
        }
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

        let kind = if round_idx % 2 == 0 { CutKind::Gmi } else { CutKind::Mir };
        let cuts = generate_round(&committed, &integer_mask, &res.solution, &ws.basis, kind);
        if cuts.is_empty() {
            break;
        }
        let candidate = append_ge_rows(&committed, &cuts);
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

    // Convert added Ge cut rows to Le before handing to B&B, so B&B node
    // solves (presolve=false, primal simplex) use slacks (coeff +1) rather than
    // surplus variables (coeff −1). Multi-round generation above uses Ge internally
    // (same simplex path as the original single-kind loop) and only the final LP
    // passed to B&B changes representation.
    let lp = convert_cuts_to_le(committed, m_orig);

    MilpProblem {
        lp,
        integer_vars: milp.integer_vars.clone(),
    }
}

/// Validate the augmented LP (still as Ge) to confirm the added cuts did not
/// make the relaxation numerically unsolvable.
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

/// Solve the cut-generation LP via primal simplex with presolve disabled so
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

/// Generate one round of cuts from the optimal `basis` and LP solution `x_star`.
fn generate_round(
    lp: &LpProblem,
    integer_mask: &[bool],
    x_star: &[f64],
    basis: &[usize],
    kind: CutKind,
) -> Vec<CutRow> {
    let sf = build_standard_form(lp);
    if basis.len() != sf.m {
        return Vec::new();
    }
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

    let Ok(mut lu) = LuBasis::new_timed(&sf.a, basis, 0, None) else {
        return Vec::new();
    };
    let mut beta = sf.b.clone();
    lu.ftran_dense(&mut beta);

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
    sources.sort_by(|a, b| a.1.total_cmp(&b.1));
    sources.truncate(MAX_CUTS_PER_ROUND);

    let mut cuts = Vec::new();
    for (i, _) in sources {
        if let Some(cut) = build_cut(
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
            kind,
        ) {
            cuts.push(cut);
        }
    }
    cuts
}

/// Build a cut from basis row `i`.
/// Returns `None` when the cut is unusable (free-var support, degenerate, weak,
/// or numerically unstable).
#[allow(clippy::too_many_arguments)]
fn build_cut(
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
    kind: CutKind,
) -> Option<CutRow> {
    let f0 = {
        let b = beta[i];
        b - b.floor()
    };
    let one_minus_f0 = 1.0 - f0;

    let mut rho = vec![0.0; sf.m];
    rho[i] = 1.0;
    lu.btran_dense(&mut rho);

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
        // MIR's `max(alpha,0)/f0` suppression for continuous nonbasics is only
        // valid for original-problem structural variables. Slack columns (of any
        // constraint, including UB rows and previous cut rows) must use GMI to
        // guarantee integer hull membership.
        let effective_kind = if matches!(kind, CutKind::Mir) && j >= sf.n_shifted {
            CutKind::Gmi
        } else {
            kind
        };
        let gamma = match effective_kind {
            CutKind::Gmi => gmi_coeff(alpha, f0, one_minus_f0, integral),
            CutKind::Mir => mir_coeff(alpha, f0, one_minus_f0, integral),
        };
        if gamma <= ZERO_TOL {
            continue;
        }
        if !accumulate_column(j, gamma, sf, struct_cols, slack_kinds, rows, lp, &mut g, &mut d) {
            return None;
        }
    }

    let rhs = 1.0 - d;
    finalize_cut(g, rhs, x_star, frac_tol)
}

/// GMI coefficient (always >= 0) for a nonbasic column with tableau entry `alpha`.
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

/// MIR coefficient (always >= 0). Identical to GMI for integer variables; for
/// continuous variables negative tableau entries contribute 0 (vs. GMI's
/// `−α / (1 − f₀)`), producing a weaker but complementary cut.
fn mir_coeff(alpha: f64, f0: f64, one_minus_f0: f64, integral: bool) -> f64 {
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
        0.0
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
                g[sc.var] += gamma;
                *d += gamma * (-sc.offset);
            }
            StructKind::UbOnly => {
                g[sc.var] -= gamma;
                *d += gamma * sc.offset;
            }
            StructKind::FreeSplit => return false,
        }
    } else {
        match slack_kinds[j - sf.n_shifted] {
            Some(SlackKind::ConstraintLe(r)) => {
                *d += gamma * lp.b[r];
                for &(c, v) in &rows[r] {
                    g[c] -= gamma * v;
                }
            }
            Some(SlackKind::ConstraintGe(r)) => {
                *d += gamma * (-lp.b[r]);
                for &(c, v) in &rows[r] {
                    g[c] += gamma * v;
                }
            }
            Some(SlackKind::UbRow(p)) => {
                let ub = lp.bounds[p].1;
                g[p] -= gamma;
                *d += gamma * ub;
            }
            None => return false,
        }
    }
    true
}

fn column_dot(a: &CscMatrix, j: usize, rho: &[f64]) -> f64 {
    let (rows, vals) = a.get_column(j).expect("valid std-form column index");
    rows.iter()
        .zip(vals)
        .map(|(&r, &v)| v * rho[r])
        .sum::<f64>()
}

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
    if max_abs <= ZERO_TOL {
        return None;
    }
    if max_abs / min_abs > GMI_MAX_COEF_DYNAMISM {
        return None;
    }
    let lhs: f64 = g.iter().zip(x_star).map(|(&gi, &xi)| gi * xi).sum();
    let violation = rhs - lhs;
    if violation <= frac_tol * (1.0 + rhs.abs()) {
        return None;
    }
    Some(CutRow { coeffs: g, rhs })
}

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
            StructKind::LbShift
        } else {
            StructKind::UbOnly
        };
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

fn classify_slack_cols(lp: &LpProblem, sf: &StandardForm) -> Vec<Option<SlackKind>> {
    let n_slack = sf.n_total - sf.n_shifted;
    let mut kinds = vec![None; n_slack];

    let ub_row_vars: Vec<usize> = (0..lp.num_vars)
        .filter(|&p| {
            let (lo, hi) = lp.bounds[p];
            lo.is_finite() && hi.is_finite()
        })
        .collect();

    let mut s = 0usize;
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
            ConstraintType::Eq => {}
        }
    }
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

/// Append cuts as Ge rows during multi-round generation (preserves original
/// simplex path; same numerical behavior as the original code).
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

/// Convert the Ge cut rows (indices `m_orig..`) to equivalent Le rows (`−g·x ≤ −rhs`).
///
/// B&B node solves run with presolve=false (primal simplex). Le slacks (coeff +1)
/// are numerically better in that setting than Ge surplus variables (coeff −1).
/// Multi-round generation ran on the Ge form (same simplex path as original code);
/// only the LP handed to B&B has the Le representation.
fn convert_cuts_to_le(lp: LpProblem, m_orig: usize) -> LpProblem {
    if lp.num_constraints == m_orig {
        return lp;
    }
    let m_total = lp.num_constraints;
    let n = lp.num_vars;

    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
    for c in 0..lp.a.ncols {
        let (rs, vs) = lp.a.get_column(c).expect("valid column");
        for (&r, &v) in rs.iter().zip(vs) {
            trip_rows.push(r);
            trip_cols.push(c);
            // Negate the coefficient for cut rows.
            trip_vals.push(if r >= m_orig { -v } else { v });
        }
    }
    let a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_total, n)
        .expect("cut-Le conversion is well-formed");

    let mut b = lp.b[..m_orig].to_vec();
    let mut ctypes = lp.constraint_types[..m_orig].to_vec();
    for i in m_orig..m_total {
        b.push(-lp.b[i]);
        ctypes.push(ConstraintType::Le);
    }

    let mut out = LpProblem::new_general(lp.c.clone(), a, b, ctypes, lp.bounds.clone(), lp.name.clone())
        .expect("cut-Le LP is valid");
    out.obj_offset = lp.obj_offset;
    out
}

#[cfg(test)]
mod tests;
