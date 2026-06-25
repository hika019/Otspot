//! Root cutting planes for MILP: GMI, MIR, cover, clique, and implied-bound cuts.
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
//! After the GMI/MIR rounds, a structural cut phase runs cover, clique, and
//! implied-bound cuts directly from the constraint matrix and LP solution.

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
/// Rounds alternate GMI (even) / MIR (odd). After the tableau-based rounds,
/// a structural phase adds cover, clique, and implied-bound cuts.
/// Multi-round generation uses Ge rows internally (preserving the original
/// simplex path). After all rounds, added Ge cut rows are converted to Le
/// (`−g·x ≤ −rhs`) before returning, so B&B node solves use slack variables
/// (coeff +1) rather than surplus variables (coeff −1).
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
    // after the candidate is re-validated with the *same* solver the B&B uses.
    let m_orig = milp.lp.num_constraints;
    let mut committed = milp.lp.clone();
    let mut prev_obj: Option<f64> = None;
    let max_total_cuts = m_orig.max(MAX_CUTS_PER_ROUND);
    let mut total_cuts = 0usize;

    let cut_deadline = options.deadline.map(|d| {
        let now = std::time::Instant::now();
        now + d.saturating_duration_since(now).mul_f64(CUT_TIME_FRACTION)
    });

    // GMI / MIR tableau-based rounds.
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

    // Structural cut phase: cover, clique, implied-bound.
    if total_cuts < max_total_cuts && !deadline_reached(cut_deadline) {
        let res = solve_cut_lp(&committed, options, cut_deadline);
        if res.status == SolveStatus::Optimal {
            let budget = max_total_cuts.saturating_sub(total_cuts);
            let mut structural: Vec<CutRow> = Vec::new();
            structural.extend(generate_cover_cuts(&committed, &integer_mask, &res.solution));
            structural.extend(generate_clique_cuts(&committed, &integer_mask, &res.solution));
            structural.extend(generate_implied_bound_cuts(
                &committed,
                &integer_mask,
                &res.solution,
            ));
            structural.truncate(budget);
            if !structural.is_empty() {
                let candidate = append_ge_rows(&committed, &structural);
                let check = solve_validate(&candidate, options, cut_deadline);
                if check.status == SolveStatus::Optimal {
                    committed = candidate;
                }
            }
        }
    }

    // Convert added Ge cut rows to Le before handing to B&B.
    let lp = convert_cuts_to_le(committed, m_orig);

    let le_check = solve_validate(&lp, options, cut_deadline);
    let final_lp = if le_check.status == SolveStatus::Optimal {
        lp
    } else {
        milp.lp.clone()
    };

    MilpProblem {
        lp: final_lp,
        integer_vars: milp.integer_vars.clone(),
    }
}

fn solve_validate(
    lp: &LpProblem,
    options: &SolverOptions,
    deadline: Option<std::time::Instant>,
) -> crate::problem::SolverResult {
    let opts = SolverOptions {
        presolve: false,
        recover_warm_start_basis: false,
        warm_start: None,
        warm_start_lp: None,
        deadline,
        timeout_secs: None,
        primal_tol: options.primal_tol,
        dual_tol: options.dual_tol,
        threads: options.threads,
        tolerance: options.tolerance,
        cancel_flag: options.cancel_flag.clone(),
        ..SolverOptions::default()
    };
    crate::lp::solve_lp_with(lp, &opts)
}

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
        cancel_flag: options.cancel_flag.clone(),
        ..SolverOptions::default()
    };
    crate::lp::solve_lp_with(lp, &opts)
}

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

/// MIR coefficient — identical to GMI for all cases.
fn mir_coeff(alpha: f64, f0: f64, one_minus_f0: f64, integral: bool) -> f64 {
    gmi_coeff(alpha, f0, one_minus_f0, integral)
}

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

// ── Structural cuts (cover, clique, implied bound) ──────────────────────────

/// Returns `true` when variable `j` is a binary integer (bounds within [0,1]).
fn is_binary(j: usize, integer_mask: &[bool], bounds: &[(f64, f64)]) -> bool {
    j < integer_mask.len()
        && integer_mask[j]
        && bounds[j].0 >= -ZERO_TOL
        && bounds[j].1 <= 1.0 + ZERO_TOL
}

/// Cover cuts for 0-1 knapsack Le constraints.
///
/// For each Le row whose support is entirely non-negative binary, finds a
/// minimal cover C (Σ_{j∈C} a_j > b) and emits −Σ_{j∈C} x_j ≥ −(|C|−1).
fn generate_cover_cuts(
    lp: &LpProblem,
    integer_mask: &[bool],
    x_star: &[f64],
) -> Vec<CutRow> {
    let frac_tol = feas_rel_tol();
    let rows = row_lists(&lp.a, lp.num_constraints);
    let mut cuts = Vec::new();

    'row: for r in 0..lp.num_constraints {
        if cuts.len() >= MAX_CUTS_PER_ROUND {
            break;
        }
        if lp.constraint_types[r] != ConstraintType::Le {
            continue;
        }
        let b = lp.b[r];
        if b <= ZERO_TOL {
            continue;
        }
        let row = &rows[r];
        if row.len() < 2 {
            continue;
        }
        // Every nonzero entry must be a positive-coefficient binary variable.
        for &(j, v) in row {
            if v <= ZERO_TOL || !is_binary(j, integer_mask, &lp.bounds) {
                continue 'row;
            }
        }

        // Sort by coefficient descending; greedy cover.
        let mut sorted: Vec<(usize, f64)> = row.to_vec();
        sorted.sort_by(|a, b_| b_.1.total_cmp(&a.1));

        let mut cover: Vec<usize> = Vec::new();
        let mut sum = 0.0_f64;
        for &(j, v) in &sorted {
            cover.push(j);
            sum += v;
            if sum > b {
                break;
            }
        }
        if sum <= b {
            continue; // all variables needed; no cover exists
        }

        // Minimise: remove smallest-coefficient elements that keep sum > b.
        let mut k = cover.len();
        while k > 0 {
            k -= 1;
            let j = cover[k];
            let coeff_j = sorted.iter().find(|&&(jj, _)| jj == j).map_or(0.0, |&(_, v)| v);
            if sum - coeff_j > b {
                sum -= coeff_j;
                cover.swap_remove(k);
            }
        }

        if cover.len() < 2 {
            continue;
        }

        let mut g = vec![0.0; lp.num_vars];
        for &j in &cover {
            g[j] = -1.0;
        }
        let rhs = -((cover.len() - 1) as f64);
        if let Some(cut) = finalize_cut(g, rhs, x_star, frac_tol) {
            cuts.push(cut);
        }
    }
    cuts
}

/// Clique cuts via global pairwise conflict graph.
///
/// Two binary variables i, j "conflict" when some Le row has a_i + a_j > b_r
/// (both being 1 would violate that constraint). We build this conflict graph
/// from all rows, then for each fractional binary variable greedily extend to a
/// clique in the conflict graph. Cliques of size ≥ 3 that the LP solution
/// violates (Σ x_star[j] > 1) are emitted as −Σ_{j∈clique} x_j ≥ −1.
fn generate_clique_cuts(
    lp: &LpProblem,
    integer_mask: &[bool],
    x_star: &[f64],
) -> Vec<CutRow> {
    let frac_tol = feas_rel_tol();
    let rows = row_lists(&lp.a, lp.num_constraints);
    let n = lp.num_vars;

    // Build conflict adjacency: conflicts[i] contains all j that conflict with i.
    let mut conflicts: Vec<Vec<usize>> = vec![Vec::new(); n];
    'row: for r in 0..lp.num_constraints {
        if lp.constraint_types[r] != ConstraintType::Le {
            continue;
        }
        let b = lp.b[r];
        if b <= ZERO_TOL {
            continue;
        }
        let row = &rows[r];
        if row.len() < 2 {
            continue;
        }
        // Pairwise conflict a_i + a_j > b is only valid when the residual
        // min-activity of all other variables is ≥ 0. Require every entry to be
        // a positive-coefficient binary variable (same guard as cover cuts).
        for &(j, v) in row {
            if v <= ZERO_TOL || !is_binary(j, integer_mask, &lp.bounds) {
                continue 'row;
            }
        }
        let bin_entries: Vec<(usize, f64)> = row.to_vec();
        for pi in 0..bin_entries.len() {
            for pj in (pi + 1)..bin_entries.len() {
                let (i, ai) = bin_entries[pi];
                let (j, aj) = bin_entries[pj];
                if ai + aj > b + ZERO_TOL {
                    if !conflicts[i].contains(&j) {
                        conflicts[i].push(j);
                    }
                    if !conflicts[j].contains(&i) {
                        conflicts[j].push(i);
                    }
                }
            }
        }
    }

    // For each fractional binary variable, greedily grow a clique in the conflict
    // graph and emit a cut if the LP solution violates Σ x_j ≤ 1.
    let mut cuts = Vec::new();
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();

    for seed in 0..n {
        if cuts.len() >= MAX_CUTS_PER_ROUND {
            break;
        }
        if !is_binary(seed, integer_mask, &lp.bounds) {
            continue;
        }
        let x_seed = if seed < x_star.len() { x_star[seed] } else { continue };
        if x_seed <= frac_tol {
            continue; // seed is at zero, no incentive to include
        }
        if conflicts[seed].is_empty() {
            continue;
        }

        // Start clique with seed; extend by adding neighbours that conflict with all.
        let mut clique: Vec<usize> = vec![seed];
        // Candidate: neighbours of seed, sorted by x_star descending (greedy).
        let mut candidates: Vec<usize> = conflicts[seed].clone();
        candidates.sort_by(|&a, &b| {
            let xa = if a < x_star.len() { x_star[a] } else { 0.0 };
            let xb = if b < x_star.len() { x_star[b] } else { 0.0 };
            xb.total_cmp(&xa)
        });
        for cand in candidates {
            // cand must conflict with every current clique member.
            if clique.iter().all(|&m| conflicts[cand].contains(&m)) {
                clique.push(cand);
            }
        }

        if clique.len() < 3 {
            continue;
        }

        // Deduplicate.
        let mut key_vec = clique.clone();
        key_vec.sort_unstable();
        let key: u64 = key_vec
            .iter()
            .take(5)
            .fold(0u64, |acc, &j| acc.wrapping_mul(1_000_003).wrapping_add(j as u64 + 1));
        if !seen.insert(key) {
            continue;
        }

        let mut g = vec![0.0; lp.num_vars];
        for &j in &clique {
            g[j] = -1.0;
        }
        if let Some(cut) = finalize_cut(g, -1.0, x_star, frac_tol) {
            cuts.push(cut);
        }
    }
    cuts
}

/// Implied bound cuts derived from constraint activity bounds.
///
/// For integer variable i, the continuous implied bound is rounded to the
/// tightest integer value (floor for upper bounds, ceil for lower bounds).
/// This creates a violation gap when the LP solution is fractional between
/// the integer implied bound and the variable's original bound.
///
/// Le row, integer var i with a_i > 0:
///   implied_ub_int = floor((b − activity_min_without_i) / a_i)
///   Cut: −x_i ≥ −implied_ub_int
///
/// Ge row, integer var i with a_i > 0:
///   implied_lb_int = ceil((b − activity_max_without_i) / a_i)
///   Cut: x_i ≥ implied_lb_int
fn generate_implied_bound_cuts(
    lp: &LpProblem,
    integer_mask: &[bool],
    x_star: &[f64],
) -> Vec<CutRow> {
    use crate::tolerances::INT_ROUND_TOL;
    let frac_tol = feas_rel_tol();
    let rows = row_lists(&lp.a, lp.num_constraints);
    let mut cuts = Vec::new();

    'row: for r in 0..lp.num_constraints {
        if cuts.len() >= MAX_CUTS_PER_ROUND {
            break;
        }
        let b = lp.b[r];
        let row = &rows[r];
        if row.is_empty() {
            continue;
        }

        match lp.constraint_types[r] {
            ConstraintType::Le => {
                // activity_min: a_j * lb_j (a_j > 0) + a_j * ub_j (a_j < 0)
                let mut activity_min = 0.0_f64;
                for &(j, v) in row {
                    let (lb, ub) = lp.bounds[j];
                    let bound = if v > 0.0 { lb } else { ub };
                    if !bound.is_finite() {
                        continue 'row;
                    }
                    activity_min += v * bound;
                }

                for &(i, ai) in row {
                    if cuts.len() >= MAX_CUTS_PER_ROUND {
                        break;
                    }
                    if ai <= ZERO_TOL || !integer_mask[i] {
                        continue;
                    }
                    let (lb_i, ub_i) = lp.bounds[i];
                    if !ub_i.is_finite() {
                        continue;
                    }
                    let lb_i_val = if lb_i.is_finite() { lb_i } else { continue };
                    // Remove var i's own contribution (positive coeff → used lb_i).
                    let activity_min_i = activity_min - ai * lb_i_val;
                    let raw_ub = (b - activity_min_i) / ai;
                    // Floor to integer (INT_ROUND_TOL guards floating-point drift).
                    let implied_ub = (raw_ub + INT_ROUND_TOL).floor();
                    if implied_ub >= ub_i - frac_tol {
                        continue; // not tighter than current bound
                    }
                    let mut g = vec![0.0; lp.num_vars];
                    g[i] = -1.0;
                    if let Some(cut) = finalize_cut(g, -implied_ub, x_star, frac_tol) {
                        cuts.push(cut);
                    }
                }
            }
            ConstraintType::Ge => {
                // activity_max: a_j * ub_j (a_j > 0) + a_j * lb_j (a_j < 0)
                let mut activity_max = 0.0_f64;
                for &(j, v) in row {
                    let (lb, ub) = lp.bounds[j];
                    let bound = if v > 0.0 { ub } else { lb };
                    if !bound.is_finite() {
                        continue 'row;
                    }
                    activity_max += v * bound;
                }

                for &(i, ai) in row {
                    if cuts.len() >= MAX_CUTS_PER_ROUND {
                        break;
                    }
                    if ai <= ZERO_TOL || !integer_mask[i] {
                        continue;
                    }
                    let (lb_i, ub_i) = lp.bounds[i];
                    if !lb_i.is_finite() {
                        continue;
                    }
                    let ub_i_val = if ub_i.is_finite() { ub_i } else { continue };
                    let activity_max_i = activity_max - ai * ub_i_val;
                    let raw_lb = (b - activity_max_i) / ai;
                    // Ceil to integer.
                    let implied_lb = (raw_lb - INT_ROUND_TOL).ceil();
                    if implied_lb <= lb_i + frac_tol {
                        continue;
                    }
                    let mut g = vec![0.0; lp.num_vars];
                    g[i] = 1.0;
                    if let Some(cut) = finalize_cut(g, implied_lb, x_star, frac_tol) {
                        cuts.push(cut);
                    }
                }
            }
            ConstraintType::Eq => {}
        }
    }
    cuts
}

#[cfg(test)]
mod tests;
