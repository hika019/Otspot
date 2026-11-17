//! Root-stage feasibility-based bound tightening (FBBT) for the nonconvex
//! QCQP route (Phase 2, plan §4.1).
//!
//! [`derive_root_box`] propagates the declared box through the problem's
//! *true* constraint rows and returns a derived root box. It never rewrites
//! `qp.lb`/`qp.ub` (plan §4.2 line 136: derived bounds travel only in
//! `GlobalSeed.root_lb`/`root_ub`, so an incumbent inside `feas_tol` of the
//! declared box is never rejected by `feasible()`), and it never widens: the
//! derived box is an interval-wise subset of the declared box by construction.
//!
//! Two row families are propagated (plan §4.1 / research doc §4 step 2):
//!
//! - **Zero-quadratic linear rows** (`g_lin` / `a_eq` — the route only puts a
//!   constraint here when its `QcqpMatrix` has `nnz() == 0`, so a quadratic
//!   row's linear part can never masquerade as linear; plan §4.1 warning,
//!   `qcqp_route.rs:239-241`). Propagation reuses the shared linear machinery
//!   `crate::presolve::activity::propagate_row_bounds` — the same row
//!   arithmetic `mip::presolve::tighten_bounds_linear` runs — with the
//!   continuous (no-integer-rounding) mask, and each implied bound is
//!   outward-rounded before it is stored (see below).
//! - **Single-variable (diagonal-only) quadratic rows** from `quad`: rows
//!   whose stored quadratic part carries no off-diagonal entry are evaluated
//!   as Σᵢ (aᵢ xᵢ² + bᵢ xᵢ) + r ≤ 0 with the completed-square batch rule
//!   (plan §4.1 line 122: per-term interval propagation cannot bound a free
//!   `x` — the 8605 `Σ aᵢ xᵢ² = c` shape needs the row-as-ellipsoid
//!   evaluation). Rows with any off-diagonal entry are skipped entirely (no
//!   bilinear / `w_ij` propagation: the L5 subsolver-dependence warning,
//!   research doc §3.3, forbids it in this phase).
//!
//! Soundness contract:
//!
//! - Every stored implied bound is relaxed outward by
//!   `feas_tol·(1+|v|) + ROOT_FBBT_ULP_SLACK·|v|` before use, so the derived
//!   box always contains the declared-box-feasible set up to `feas_tol`
//!   (propagation amplifies a row violation by `1/|a_j|`, which is exactly
//!   why the plan keeps the derived box out of `qp.lb`/`qp.ub` and why the
//!   margin must be there; plan §4.1 line 124).
//! - **No infeasibility is ever reported.** A row whose implied bound crosses
//!   the current box, or whose completed-square discriminant is genuinely
//!   negative, marks the row as a *contradiction* and contributes no
//!   tightening; the returned type has no infeasible state. False
//!   `Infeasible` verdicts have a measured precedent in this solver's LP
//!   layer (the `PRESOLVE_ESCALATION` audit in `nonconvex.rs` exists because
//!   of one), and root FBBT ships no equivalent audit — so per research doc
//!   §3.3 the only admissible contract is to *refuse* the verdict: the B&B
//!   relaxation, with its own escalation-audited infeasible arm, remains the
//!   sole emptiness certificate. Contradictions are only counted in
//!   [`RootFbbtBox::contradiction_rows`] for measurement.
//! - Fixed-point rounds are capped by [`ROOT_FBBT_MAX_ROUNDS`]; a box that
//!   has not converged by then is returned partially tightened (sound, just
//!   less tight) — termination is never delegated to "run until stable".

use super::nonconvex::NonconvexQcqp;
use crate::presolve::activity::propagate_row_bounds;
use crate::problem::ConstraintType;
use otspot_num::sparse::CscMatrix;
use otspot_num::{run_fixpoint, SolveControl};

/// Maximum fixed-point rounds of [`derive_root_box`].
///
/// Same treatment as `mip::presolve::MAX_PRESOLVE_ROUNDS` (plan §4.1 line
/// 125: the cap is a named const justified like the existing one): interval
/// propagation is a monotone contraction of a finite set of boxes, so runs
/// that need more than a handful of chained round-trips are pathological,
/// and the root cost is `rounds × O(nnz)` regardless of problem size. 10
/// mirrors the MIP presolve choice; the B&B pays the residual slack itself
/// via branching, so an unconverged box costs search nodes, never
/// correctness.
pub const ROOT_FBBT_MAX_ROUNDS: usize = 10;

/// Relative outward slack applied to every derived bound *before* the
/// `feas_tol` margin, in units of `f64::EPSILON`.
///
/// A derived bound passes through a division by the row coefficient and (for
/// quadratic rows) a `sqrt`, each ~1–2 ulp of relative error against the
/// exact implied bound; `8·EPSILON` bounds that chain with margin. The
/// outward direction is the safety-critical one: a bound rounded *inward*
/// could cut a feasible point by a few ulp, and the B&B would then prove a
/// wrong optimum of the truncated problem.
pub const ROOT_FBBT_ULP_SLACK: f64 = 8.0 * f64::EPSILON;

/// Relative slack below which a negative completed-square discriminant is
/// treated as the tangent case (`D = 0`, one-point bound) instead of a row
/// contradiction. `D = b² + 4aK` loses its leading digits when the vertex
/// value equals `K` almost exactly (the tangent-ellipsoid fixture), and a
/// tangent row legitimately fixes a variable — dropping it as a
/// contradiction would only lose tightening, but clamping `D` at 0 keeps the
/// (valid) fixing. `64·EPSILON` absorbs the accumulation error of `K`'s
/// compensated sum plus the `b²`/`4aK` squaring roundings.
const ROOT_FBBT_DISCRIM_SLACK: f64 = 64.0 * f64::EPSILON;

/// Derived root box plus propagation statistics (measurement output for the
/// Phase-0 width study; the route consumes only `lb`/`ub`).
#[derive(Debug, Clone)]
pub struct RootFbbtBox {
    /// Derived lower bounds: interval-wise inside `qp.lb`/`qp.ub`, never
    /// NaN, never `lb > ub` (empty intervals are refused, never created).
    pub lb: Vec<f64>,
    /// Derived upper bounds, same contract as [`RootFbbtBox::lb`].
    pub ub: Vec<f64>,
    /// Fixed-point passes executed (≤ [`ROOT_FBBT_MAX_ROUNDS`]).
    pub rounds: usize,
    /// Variables whose bound moved at least once.
    pub tightened: usize,
    /// Zero-quadratic linear rows available for propagation.
    pub linear_rows: usize,
    /// Diagonal-only quadratic rows available for propagation.
    pub quad_diag_rows: usize,
    /// Rows that contradicted the current box during propagation. Counted
    /// for measurement only — the verdict is *not* reported (module-doc
    /// never-infeasible contract).
    pub contradiction_rows: usize,
}

impl RootFbbtBox {
    /// Whether the derived box is strictly tighter than the declared box
    /// somewhere (the route only attaches a seed when this is `true`, so a
    /// no-op derivation keeps the solver path bit-identical to the seedless
    /// one).
    pub fn is_tightening(&self, qp: &NonconvexQcqp) -> bool {
        self.lb
            .iter()
            .zip(&self.ub)
            .zip(qp.lb.iter().zip(&qp.ub))
            .any(|((&l, &u), (&pl, &pu))| l != pl || u != pu)
    }
}

/// Outward margin for a derived bound value `v` at feasibility tolerance
/// `feas_tol` (plan §4.1 line 124: round outward, *then* relax by `feas_tol`).
fn bound_margin(v: f64, feas_tol: f64) -> f64 {
    feas_tol * (1.0 + v.abs()) + ROOT_FBBT_ULP_SLACK * v.abs()
}

/// Linear row in the row-list form `propagate_row_bounds` consumes.
struct LinRow {
    entries: Vec<(usize, f64)>,
    ct: ConstraintType,
    rhs: f64,
}

/// CSR-style row lists of the zero-quadratic rows (`g_lin` are all `Le`
/// after the route's normalization, `a_eq` are `Eq`). The quadratic rows'
/// linear parts live inside `quad[k].q` and never enter these lists —
/// plan §4.1's warning that a quadratic row's linear part must not be fed to
/// linear FBBT is structural here.
fn build_lin_rows(qp: &NonconvexQcqp) -> Vec<LinRow> {
    let mut rows = Vec::with_capacity(qp.h_lin.len() + qp.b_eq.len());
    let gl = qp.g_lin.transpose();
    let ae = qp.a_eq.transpose();
    for (i, &h) in qp.h_lin.iter().enumerate() {
        let (cols, vals) = gl.column(i);
        rows.push(LinRow {
            entries: cols.iter().copied().zip(vals.iter().copied()).collect(),
            ct: ConstraintType::Le,
            rhs: h,
        });
    }
    for (i, &b) in qp.b_eq.iter().enumerate() {
        let (cols, vals) = ae.column(i);
        rows.push(LinRow {
            entries: cols.iter().copied().zip(vals.iter().copied()).collect(),
            ct: ConstraintType::Eq,
            rhs: b,
        });
    }
    rows
}

/// One diagonal term `(a, b)` of `a·x² + b·x` for a single variable.
#[derive(Clone, Copy)]
struct DiagTerm {
    var: usize,
    a: f64,
    b: f64,
}

/// A diagonal-only quadratic row: `Σ (aᵢ xᵢ² + bᵢ xᵢ) + r ≤ 0`.
struct DiagQuadRow {
    terms: Vec<DiagTerm>,
    r: f64,
}

/// Extract the diagonal-only rows. A row is diagonal-only when no
/// off-diagonal `P` entry is stored at all (composed-zero cancellation
/// `P_ij = -P_ji` is treated as off-diagonal — conservative: such rows are
/// rare, and misclassifying a real bilinear pair as diagonal would break the
/// containment argument). Non-finite coefficients disqualify the row.
fn build_diag_quad_rows(qp: &NonconvexQcqp) -> Vec<DiagQuadRow> {
    qp.quad
        .iter()
        .filter_map(|qc| diag_terms(&qc.p, &qc.q, qc.r))
        .collect()
}

/// `Some(row)` for a diagonal-only row, `None` when the row must be skipped.
fn diag_terms(p: &CscMatrix, q: &[f64], r: f64) -> Option<DiagQuadRow> {
    if !r.is_finite() {
        return None;
    }
    let mut diag: Vec<(usize, f64)> = Vec::new();
    let cp = p.col_ptr();
    let ri = p.row_ind();
    let va = p.values();
    for j in 0..p.ncols() {
        for k in cp[j]..cp[j + 1] {
            if ri[k] != j {
                return None;
            }
            // Composed diagonal coefficient of the row value
            // `(1/2) x^T P x`: `0.5 * P[j][j]` (qcqp_classify::lift_coeffs
            // convention).
            let a = 0.5 * va[k];
            if !a.is_finite() {
                return None;
            }
            if a != 0.0 {
                diag.push((j, a));
            }
        }
    }
    // Linear parts of the row: every nonzero q entry (a variable with only a
    // linear member still contributes a bound to the row minimum — dropping
    // it would over-estimate the slack and derive an unsound bound).
    // `diag` is ascending in `var` (CSC column order); merge the two lists.
    let mut terms: Vec<DiagTerm> = Vec::with_capacity(diag.len() + 8);
    let mut next = 0usize;
    for (j, &b) in q.iter().enumerate() {
        if b == 0.0 {
            continue;
        }
        if !b.is_finite() {
            return None;
        }
        while next < diag.len() && diag[next].0 < j {
            let (v, a) = diag[next];
            terms.push(DiagTerm { var: v, a, b: q[v] });
            next += 1;
        }
        if next < diag.len() && diag[next].0 == j {
            let (v, a) = diag[next];
            terms.push(DiagTerm { var: v, a, b });
            next += 1;
        } else {
            terms.push(DiagTerm { var: j, a: 0.0, b });
        }
    }
    for &(v, a) in &diag[next..] {
        terms.push(DiagTerm { var: v, a, b: q[v] });
    }
    Some(DiagQuadRow { terms, r })
}

/// `min` of `a·x² + b·x` over `[l, u]` in the extended reals, as
/// `(value, sign)` with `sign` ∈ {-1, 0, +1} for `-inf` / finite / `+inf`.
///
/// The completed-square evaluation (plan §4.1 line 122) is what lets a free
/// endpoint carry a *finite* vertex value (`a > 0` and `v ∈ [l, u]`): the
/// 0·±∞-style term products of per-coefficient interval arithmetic are never
/// formed, so no NaN can appear and free variables do not poison the row
/// minimum (plan §5 sentinel 4's failure mode).
fn term_min(a: f64, b: f64, l: f64, u: f64) -> (f64, i8) {
    // `eval` extends `a x² + b x` to ±inf: the `x²` term dominates (`a == 0`
    // falls back to the sign of `b x`), so no `0 * inf` product is formed.
    let eval = |x: f64| -> (f64, i8) {
        if !x.is_infinite() {
            return (a * x * x + b * x, 0);
        }
        if a > 0.0 {
            (f64::INFINITY, 1)
        } else if a < 0.0 {
            (f64::NEG_INFINITY, -1)
        } else if b != 0.0 {
            // `b` is finite and nonzero here (`diag_terms` rejects non-finite
            // coefficients), so the sign of `b * x` decides.
            let v = b * x;
            (v, if v > 0.0 { 1 } else { -1 })
        } else {
            (0.0, 0)
        }
    };
    if a > 0.0 {
        let v = -b / (2.0 * a);
        if v <= l {
            eval(l)
        } else if v >= u {
            eval(u)
        } else {
            (a * v * v + b * v, 0)
        }
    } else if a < 0.0 {
        // Concave: the minimum sits at an endpoint; -inf beats everything.
        let (gl, sl) = eval(l);
        let (gu, su) = eval(u);
        if sl < 0 {
            (gl, sl)
        } else if su < 0 {
            (gu, su)
        } else {
            (gl.min(gu), if gl <= gu { sl } else { su })
        }
    } else if b > 0.0 {
        eval(l)
    } else if b < 0.0 {
        eval(u)
    } else {
        (0.0, 0)
    }
}

/// Neumaier-compensated accumulator: row minima sum thousands of terms
/// (`Σ aᵢ xᵢ² = c` rows on 5000-variable instances), and `K_j` is the
/// difference of two such sums, so plain accumulation would put
/// `O(nnz·ulp)` error directly into every derived bound's margin budget.
struct CompSum {
    sum: f64,
    comp: f64,
}

impl CompSum {
    fn new() -> Self {
        CompSum {
            sum: 0.0,
            comp: 0.0,
        }
    }
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.comp += (self.sum - t) + x;
        } else {
            self.comp += (x - t) + self.sum;
        }
        self.sum = t;
    }
    fn total(&self) -> f64 {
        self.sum + self.comp
    }
}

/// One fixed-point pass over the diagonal-only quadratic rows. Returns the
/// number of bound moves; contradictions are counted into `contradictions`
/// and never reported (module-doc contract).
fn diag_quad_pass(
    rows: &[DiagQuadRow],
    bounds: &mut [(f64, f64)],
    feas_tol: f64,
    contradictions: &mut usize,
) -> usize {
    let mut moves = 0usize;
    'rows: for row in rows {
        if row.terms.is_empty() {
            continue;
        }
        // Row minimum per term + the ±inf bookkeeping (a `-inf` in the rest
        // makes K infinite -> no bound; a `+inf` in the row minimum means the
        // row is violated everywhere -> contradiction).
        let mut mins = vec![(0.0f64, 0i8); row.terms.len()];
        let mut acc = CompSum::new();
        let mut neg_inf = 0usize;
        let mut pos_inf = 0usize;
        for (t, term) in row.terms.iter().enumerate() {
            let (l, u) = bounds[term.var];
            let (v, s) = term_min(term.a, term.b, l, u);
            mins[t] = (v, s);
            match s {
                -1 => neg_inf += 1,
                1 => pos_inf += 1,
                _ => acc.add(v),
            }
        }
        if pos_inf > 0 {
            *contradictions += 1;
            continue;
        }
        let total = acc.total();
        for (t, term) in row.terms.iter().enumerate() {
            let (v, s) = mins[t];
            let rest_neg_inf = neg_inf - usize::from(s == -1);
            if rest_neg_inf > 0 {
                // Some other term stays -inf whatever x_j does: no bound.
                continue;
            }
            let rest = total - if s == 0 { v } else { 0.0 };
            // x_j must satisfy `a·x_j² + b·x_j <= k`.
            let k = -row.r - rest;
            let (j, a, b) = (term.var, term.a, term.b);
            if a > 0.0 {
                let d = b * b + 4.0 * a * k;
                let scale = b * b + 4.0 * a * k.abs();
                if !d.is_finite() {
                    continue;
                }
                if d < -ROOT_FBBT_DISCRIM_SLACK * scale.max(1.0) {
                    // Even the vertex violates the row: contradiction for the
                    // whole row (never an infeasibility verdict).
                    *contradictions += 1;
                    continue 'rows;
                }
                let sq = d.max(0.0).sqrt();
                let lo = (-b - sq) / (2.0 * a);
                let hi = (-b + sq) / (2.0 * a);
                moves += apply_outward(
                    bounds,
                    j,
                    lo - bound_margin(lo, feas_tol),
                    hi + bound_margin(hi, feas_tol),
                );
            } else if a == 0.0 && b != 0.0 {
                // Single-term linear row member `b·x_j <= k`.
                let implied = k / b;
                if !implied.is_finite() {
                    continue;
                }
                if b > 0.0 {
                    moves += apply_outward(
                        bounds,
                        j,
                        f64::NEG_INFINITY,
                        implied + bound_margin(implied, feas_tol),
                    );
                } else {
                    moves += apply_outward(
                        bounds,
                        j,
                        implied - bound_margin(implied, feas_tol),
                        f64::INFINITY,
                    );
                }
            }
            // a < 0: `{a x² + b x <= k}` is the *outside* of two roots and
            // bounds neither side of x_j — derive nothing (a naive linear-
            // part reading of this row is exactly plan §4.1's forbidden move).
        }
    }
    moves
}

/// Apply one outward-relaxed implied interval `[new_lb, new_ub]` to
/// `bounds[j]` (infinite sides are "no constraint from this derivation").
///
/// A side tightens only if strictly inside the current interval, and is
/// clamped at the opposite side: an interval is never emptied
/// (never-infeasible contract), a `NaN`/non-finite candidate is never
/// applied, and no side ever widens.
fn apply_outward(bounds: &mut [(f64, f64)], j: usize, new_lb: f64, new_ub: f64) -> usize {
    let (l, u) = bounds[j];
    let mut moves = 0usize;
    if new_lb.is_finite() && new_lb > l && new_lb <= u {
        bounds[j].0 = new_lb;
        moves += 1;
    }
    let l = bounds[j].0;
    if new_ub.is_finite() && new_ub < u && new_ub >= l {
        bounds[j].1 = new_ub;
        moves += 1;
    }
    moves
}

/// Run root-stage FBBT on the declared box of `qp` and return the derived
/// root box (module doc for the full contract). `feas_tol` is the solver's
/// incumbent feasibility tolerance (`GlobalOptions::feas_tol`) — the same
/// quantity `feasible()` accepts at, so the derived box contains every point
/// the acceptance rule can produce.
///
/// The returned box is interval-wise inside the declared box; if the caller's
/// `qp` fails its shape validation the declared box is returned unchanged
/// (a malformed problem is the route's/`global_core`'s error to report, not
/// this function's).
pub fn derive_root_box(qp: &NonconvexQcqp, feas_tol: f64) -> RootFbbtBox {
    let declared = (qp.lb.clone(), qp.ub.clone());
    if qp.validate().is_err() {
        return RootFbbtBox {
            lb: declared.0,
            ub: declared.1,
            rounds: 0,
            tightened: 0,
            linear_rows: 0,
            quad_diag_rows: 0,
            contradiction_rows: 0,
        };
    }
    let feas_tol = if feas_tol.is_finite() && feas_tol > 0.0 {
        feas_tol
    } else {
        0.0
    };
    let lin_rows = build_lin_rows(qp);
    let quad_rows = build_diag_quad_rows(qp);
    let mut bounds: Vec<(f64, f64)> = qp.lb.iter().zip(&qp.ub).map(|(&l, &u)| (l, u)).collect();
    let mut moved_vars = vec![false; qp.n];
    let mut contradictions = 0usize;
    let mut rounds = 0usize;
    let _ = run_fixpoint::<()>(ROOT_FBBT_MAX_ROUNDS, SolveControl::default(), |_| {
        rounds += 1;
        let mut moves = 0usize;
        for row in &lin_rows {
            match propagate_row_bounds(&row.entries, &bounds, row.ct, row.rhs, None) {
                Some(updates) => {
                    for (j, nlb, nub) in updates {
                        // Outward relax before storing (module doc): the
                        // stored bound is the relaxed one, so later rows in
                        // the same pass propagate from sound intervals.
                        let lo = nlb - bound_margin(nlb, feas_tol);
                        let hi = nub + bound_margin(nub, feas_tol);
                        moves += apply_outward(&mut bounds, j, lo, hi);
                    }
                }
                None => contradictions += 1,
            }
        }
        moves += diag_quad_pass(&quad_rows, &mut bounds, feas_tol, &mut contradictions);
        for (j, m) in moved_vars.iter_mut().enumerate() {
            *m |= bounds[j] != (qp.lb[j], qp.ub[j]);
        }
        Ok::<_, ()>(moves > 0)
    });
    RootFbbtBox {
        lb: bounds.iter().map(|&(l, _)| l).collect(),
        ub: bounds.iter().map(|&(_, u)| u).collect(),
        rounds,
        tightened: moved_vars.iter().filter(|m| **m).count(),
        linear_rows: lin_rows.len(),
        quad_diag_rows: quad_rows.len(),
        contradiction_rows: contradictions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic::GQuadConstraint;

    fn csc(n: usize, entries: Vec<(usize, usize, f64)>) -> CscMatrix {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (r, c, v) in entries {
            rows.push(r);
            cols.push(c);
            vals.push(v);
        }
        CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
    }

    fn empty(n: usize) -> CscMatrix {
        CscMatrix::new(0, n)
    }

    /// Row-shaped linear constraint matrix (`h_lin.len()` rows, `n` cols) —
    /// `validate` demands the row counts to match.
    fn linc(rows: usize, cols: usize, entries: Vec<(usize, usize, f64)>) -> CscMatrix {
        let mut r = Vec::new();
        let mut c = Vec::new();
        let mut v = Vec::new();
        for (i, j, val) in entries {
            r.push(i);
            c.push(j);
            v.push(val);
        }
        CscMatrix::from_triplets(&r, &c, &v, rows, cols).unwrap()
    }

    fn qcqp(n: usize, quad: Vec<GQuadConstraint>, lb: Vec<f64>, ub: Vec<f64>) -> NonconvexQcqp {
        NonconvexQcqp {
            n,
            p0: None,
            q0: vec![0.0; n],
            quad,
            g_lin: empty(n),
            h_lin: vec![],
            a_eq: empty(n),
            b_eq: vec![],
            lb,
            ub,
        }
    }

    /// Diagonal-only row from composed `(var, a, b)` terms + constant:
    /// `Σ (aᵢ xᵢ² + bᵢ xᵢ) + r ≤ 0` (a stored `P[j][j] = 2a` under the
    /// `(1/2) xᵀP x` row convention).
    fn diag_row(n: usize, terms: &[(usize, f64, f64)], r: f64) -> GQuadConstraint {
        let p = csc(
            n,
            terms
                .iter()
                .filter(|(_, a, _)| *a != 0.0)
                .map(|&(j, a, _)| (j, j, 2.0 * a))
                .collect(),
        );
        let mut q = vec![0.0; n];
        for &(j, _, b) in terms {
            q[j] = b;
        }
        GQuadConstraint { p, q, r }
    }

    fn lin_qp(
        n: usize,
        g_lin: CscMatrix,
        h_lin: Vec<f64>,
        lb: Vec<f64>,
        ub: Vec<f64>,
    ) -> NonconvexQcqp {
        NonconvexQcqp {
            n,
            p0: None,
            q0: vec![0.0; n],
            quad: vec![],
            g_lin,
            h_lin,
            a_eq: empty(n),
            b_eq: vec![],
            lb,
            ub,
        }
    }

    /// Independent oracle for the linear machinery: hand-computed implied
    /// bounds (nothing here is recomputed through the propagation code).
    /// `x0 + x1 <= 6`, `x0 - x1 <= 2`, `x1 in [0,4]`, `x0` free:
    /// `x0 <= 6` (row 1 at `x1 = 0`, row 2 at `x1 = 4` agree), and no row
    /// bounds `x0` below.
    #[test]
    fn linear_rows_derive_hand_computed_bounds() {
        let n = 2;
        let g_lin = linc(
            2,
            n,
            vec![(0, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, -1.0)],
        );
        let qp = lin_qp(
            n,
            g_lin,
            vec![6.0, 2.0],
            vec![f64::NEG_INFINITY, 0.0],
            vec![f64::INFINITY, 4.0],
        );
        let out = derive_root_box(&qp, 1e-6);
        assert!(
            out.ub[0] >= 6.0 && out.ub[0] <= 6.0 + 1e-4,
            "hand oracle: x0 <= 6 (outward-rounded only): {:?}",
            out.ub
        );
        assert_eq!(out.lb[0], f64::NEG_INFINITY, "no row bounds x0 below");
        assert_eq!(out.tightened, 1);
        // One productive pass plus one stable pass (the cap never fires here).
        assert_eq!(out.rounds, 2);
    }

    /// Plan §5 sentinel 7 (quadratic row's linear part must never be read as
    /// linear): row `-x0² + x1 <= 0` (true set: `x1 <= x0²`), `x0 ∈ [1,2]`,
    /// `x1` free. Reading the linear part alone would derive `x1 <= 0` and
    /// cut `x1 = 4` (feasible at `x0 = 2`); the completed-square rule uses
    /// the *whole* diagonal term and derives the sound bound `x1 <= 4`
    /// (tight at `x0 = 2`) — both facts pinned together, so a revert to the
    /// forbidden reading fails the lower assertion.
    #[test]
    fn quadratic_linear_part_never_propagated_as_linear() {
        let n = 2;
        let row = diag_row(n, &[(0, -1.0, 0.0), (1, 0.0, 1.0)], 0.0);
        let qp = qcqp(
            n,
            vec![row],
            vec![1.0, f64::NEG_INFINITY],
            vec![2.0, f64::INFINITY],
        );
        let out = derive_root_box(&qp, 1e-6);
        assert!(
            out.ub[1] >= 4.0,
            "the feasible point x1 = 4 (at x0 = 2) must stay inside the box; \
             the forbidden linear-part reading derives x1 <= 0: ub[1] = {}",
            out.ub[1]
        );
        assert!(
            out.ub[1] < 4.0 + 1e-3,
            "the completed-square bound x1 <= 4 must be derived: ub[1] = {}",
            out.ub[1]
        );
        assert_eq!(out.lb[1], f64::NEG_INFINITY);
        assert_eq!(out.tightened, 1);
    }

    /// Plan §4.1 line 123 (the 8605 shape): the equality `Σ xᵢ² = 0.5` splits
    /// into the `<=`/`>=` pair; the `<=` half is an ellipsoid and bounds every
    /// member variable, hand oracle `|xᵢ| <= sqrt(0.5)`. Per-term interval
    /// arithmetic (0·±inf) cannot achieve this from free `x`.
    #[test]
    fn completing_the_square_bounds_free_vars_of_ellipsoid_equality() {
        let n = 3;
        let plus = diag_row(n, &[(0, 1.0, 0.0), (1, 1.0, 0.0), (2, 1.0, 0.0)], -0.5);
        let minus = diag_row(n, &[(0, -1.0, 0.0), (1, -1.0, 0.0), (2, -1.0, 0.0)], 0.5);
        let qp = qcqp(
            n,
            vec![plus, minus],
            vec![f64::NEG_INFINITY; n],
            vec![f64::INFINITY; n],
        );
        let out = derive_root_box(&qp, 1e-6);
        let root = 0.5f64.sqrt();
        for j in 0..n {
            assert!(
                out.lb[j] <= -root && out.lb[j] > -root - 1e-3,
                "lb[{j}] must contain -sqrt(0.5) outward: {}",
                out.lb[j]
            );
            assert!(
                out.ub[j] >= root && out.ub[j] < root + 1e-3,
                "ub[{j}] must contain +sqrt(0.5) outward: {}",
                out.ub[j]
            );
        }
        // The `-P` half (a<0) contributes nothing and contradicts nothing.
        assert_eq!(out.contradiction_rows, 0);
    }

    /// Plan §5 sentinel 4 at the diagonal level: a variable fixed at 0
    /// multiplied into the row (`x0 ∈ [0,0]` with a quadratic member) must
    /// contribute min 0 — no 0·±inf NaN, no false contradiction, and the
    /// free partner is still bounded (`x1² <= 1 - 0` -> `|x1| <= 1`).
    #[test]
    fn fixed_zero_times_free_term_no_false_infeasible() {
        let n = 2;
        let row = diag_row(n, &[(0, 1.0, 0.0), (1, 1.0, 0.0)], -1.0);
        let qp = qcqp(
            n,
            vec![row],
            vec![0.0, f64::NEG_INFINITY],
            vec![0.0, f64::INFINITY],
        );
        let out = derive_root_box(&qp, 1e-6);
        assert_eq!(out.contradiction_rows, 0, "no false contradiction");
        assert!(out.ub[1] >= 1.0 && out.ub[1] < 1.0 + 1e-3, "{:?}", out.ub);
        assert!(out.lb[1] <= -1.0 && out.lb[1] > -1.0 - 1e-3, "{:?}", out.lb);
    }

    /// Sentinel (task item d): the outward rounding + feas_tol margin must
    /// contain the genuinely feasible point `x0 = 2e7` of
    /// `1e-8·x0 + x1 <= 0.3`, `x1 = 0.1`: float evaluates `0.3 - 0.1 =
    /// 0.19999999999999998`, and the raw implied bound
    /// `19999999.99999999...` sits BELOW the feasible point `x0 = 2e7`
    /// (whose float residual `4e-17` is well inside `feas_tol = 1e-6`).
    /// Without the guard the derived `ub[0]` excludes it (revert-fails);
    /// with it, `ub[0] >= 2e7` while staying far inside the declared `1e9`.
    #[test]
    fn outward_rounding_contains_feasible_amplified_point() {
        let n = 2;
        let g_lin = linc(1, n, vec![(0, 0, 1e-8), (0, 1, 1.0)]);
        let qp = lin_qp(n, g_lin, vec![0.3], vec![0.0, 0.1], vec![1e9, 0.1]);
        let naive = (0.3f64 - 0.1) / 1e-8;
        assert!(
            naive < 2e7,
            "test premise: float division alone must sit below 2e7, got {naive}"
        );
        let out = derive_root_box(&qp, 1e-6);
        assert!(
            out.ub[0] >= 2e7,
            "derived ub must OUTWARD-round past the feasible point 2e7 \
             (naive bound {naive} excludes it): ub[0] = {}",
            out.ub[0]
        );
        assert!(
            out.ub[0] < 1e9,
            "the tightening must survive: ub[0] = {} stayed at declared",
            out.ub[0]
        );
        assert!(out.is_tightening(&qp));
    }

    /// The named fixed-point cap is load-bearing: a 13-link `x_k <= x_{k+1}`
    /// lower-bound chain propagates one link per pass against the adversarial
    /// row order, so after `ROOT_FBBT_MAX_ROUNDS` passes exactly `10` links
    /// have tightened and the tail is untouched (the cap terminates; never
    /// "until stable"; a partially-converged box is still sound).
    #[test]
    fn fixpoint_cap_terminates_long_chain() {
        let n = 13;
        // Row n-2-k: x_k - x_{k+1} <= 0 — the pass order only lets the
        // declared lb of x0 advance one link per pass.
        let mut trip = Vec::new();
        for k in 0..(n - 1) {
            trip.push((n - 2 - k, k, 1.0));
            trip.push((n - 2 - k, k + 1, -1.0));
        }
        let mut lb = vec![f64::NEG_INFINITY; n];
        lb[0] = 5.0;
        let qp = lin_qp(
            n,
            linc(n - 1, n, trip),
            vec![0.0; n - 1],
            lb,
            vec![f64::INFINITY; n],
        );
        let out = derive_root_box(&qp, 1e-6);
        assert_eq!(out.rounds, ROOT_FBBT_MAX_ROUNDS);
        let deep = out.lb[ROOT_FBBT_MAX_ROUNDS];
        assert!(
            deep > 4.99 && deep <= 5.0,
            "link 10 must carry the chain (margin-relaxed): {deep}"
        );
        assert_eq!(
            out.lb[ROOT_FBBT_MAX_ROUNDS + 2],
            f64::NEG_INFINITY,
            "cap must stop before full convergence on this adversarial chain"
        );
    }

    /// Task item (e): `x0² + 1 <= 0` is empty, and root FBBT must *not*
    /// report it — no infeasible state exists to report, the box comes back
    /// exactly as declared (a false `Infeasible` from propagation has
    /// precedent in this solver's LP layer; the emptiness certificate stays
    /// with the relaxation's escalation-audited arm).
    #[test]
    fn contradiction_rows_are_counted_never_reported() {
        let n = 1;
        let row = diag_row(n, &[(0, 1.0, 0.0)], 1.0);
        let qp = qcqp(n, vec![row], vec![-1.0], vec![1.0]);
        let out = derive_root_box(&qp, 1e-6);
        assert_eq!(out.lb, vec![-1.0]);
        assert_eq!(out.ub, vec![1.0]);
        assert_eq!(out.tightened, 0);
        assert!(
            out.contradiction_rows >= 1,
            "the contradiction must at least be countable for measurement"
        );
    }

    /// Refine-only: an implied bound that would WIDEN (slack far outside the
    /// declared box) leaves the box bit-identical, and `is_tightening` stays
    /// false so the route attaches no seed (bit-identical solver path).
    #[test]
    fn never_widens_and_reports_no_change() {
        let n = 2;
        let g_lin = linc(1, n, vec![(0, 0, 1.0), (0, 1, 1.0)]);
        let qp = lin_qp(n, g_lin, vec![100.0], vec![0.0, 0.0], vec![1.0, 1.0]);
        let out = derive_root_box(&qp, 1e-6);
        assert_eq!(out.lb, qp.lb);
        assert_eq!(out.ub, qp.ub);
        assert!(!out.is_tightening(&qp));
        assert_eq!(out.tightened, 0);
    }

    /// A negative discriminant at the tangent (`x0² <= 0` from
    /// `x0² + x1² <= 1` with `x1` fixed at 1) clamps to the one-point bound
    /// instead of dropping the row: the derived interval contains `x0 = 0`
    /// and is finite on both sides.
    #[test]
    fn tangent_clamps_to_point_bound() {
        let n = 2;
        let row = diag_row(n, &[(0, 1.0, 0.0), (1, 1.0, 0.0)], -1.0);
        let qp = qcqp(
            n,
            vec![row],
            vec![f64::NEG_INFINITY, 1.0],
            vec![f64::INFINITY, 1.0],
        );
        let out = derive_root_box(&qp, 1e-6);
        assert!(out.lb[0] <= 0.0 && out.ub[0] >= 0.0);
        assert!(out.lb[0].is_finite() && out.ub[0].is_finite());
        assert_eq!(out.contradiction_rows, 0);
    }

    #[test]
    fn rows_with_offdiag_entries_are_skipped_entirely() {
        // `diag_terms` refuses any row with a stored off-diagonal entry
        // (composed-zero cancellation counts as off-diagonal — plan §4.1:
        // no bilinear/w-ij propagation this phase). Removing the refusal
        // makes this row masquerade as two diagonal terms and fabricate a
        // bound out of `-2 x0 x1` (revert-fail sentinel, report §6 S2).
        let n = 2;
        let p = csc(n, vec![(1, 0, 2.0), (0, 1, 2.0)]);
        let row = GQuadConstraint {
            p,
            q: vec![0.0, 0.0],
            r: 0.0,
        };
        let qp = qcqp(
            n,
            vec![row],
            vec![f64::NEG_INFINITY; n],
            vec![f64::INFINITY; n],
        );
        let out = derive_root_box(&qp, 1e-6);
        assert_eq!(out.quad_diag_rows, 0, "off-diagonal rows must not be read");
        assert_eq!(out.tightened, 0, "a skipped row cannot move any bound");
        assert_eq!(out.lb, vec![f64::NEG_INFINITY; n]);
        assert_eq!(out.ub, vec![f64::INFINITY; n]);
    }

    #[test]
    fn tangent_clamp_survives_accumulated_roundoff() {
        // Same tangent shape as `tangent_clamps_to_point_bound`, but the
        // fixed side `s = sqrt(0.7)` squares to `0.7000000000000001`, so the
        // compensated row budget `K = 0.7 - s^2` lands a few ulp *below* zero
        // through rounding alone. The `ROOT_FBBT_DISCRIM_SLACK` arm is what
        // keeps this a tangent (clamped `D = 0`, one-point fixing) instead of
        // a row contradiction: with the slack removed the assertions below
        // fail (revert-fail sentinel, report §6 T3).
        let n = 2;
        let s = 0.7f64.sqrt();
        let row = diag_row(n, &[(0, 1.0, 0.0), (1, 1.0, 0.0)], -0.7);
        let qp = qcqp(
            n,
            vec![row],
            vec![f64::NEG_INFINITY, s],
            vec![f64::INFINITY, s],
        );
        let out = derive_root_box(&qp, 1e-6);
        assert_eq!(
            out.contradiction_rows, 0,
            "a ulp-negative row budget is the tangent case, never a contradiction"
        );
        assert!(
            out.lb[0].is_finite() && out.ub[0].is_finite(),
            "tangent fixing lost: [{}, {}]",
            out.lb[0],
            out.ub[0]
        );
    }
}
