//! §3.1 term classification for the bound-requirement set `R` of a nonconvex
//! QCQP, shared by the two entry points that used to reject *any* non-finite
//! variable bound (`crate::qp::qcqp_route::nonconvex_from_qp_problem` and
//! `otspot-model`'s `Model` QCQP fallback).
//!
//! The McCormick spatial B&B only needs finite bounds for variables in the
//! lifted, interval-estimated quadratic terms. Rows are judged per `<=`-row
//! after normalization (`>=` rows are sign-flipped, `=` rows split into the
//! `P` and `-P` pair), objective is minimize:
//!
//! - Rule 1: a row (or the objective) whose whole symmetric quadratic form is
//!   PSD-certified stays convex and contributes nothing to `R`. A PSD `=`
//!   row still loses its bounds through the `-P` half of the split.
//! - Rule 2: a diagonal `x_i^2` whose lifted coefficient `0.5·P[i][i]` is
//!   `>= 0` in every nonconvex row and objective is under-estimated by
//!   `w_ii >= x_i^2` alone, so `x_i` is not required. Excluding rule-1 rows
//!   from the scan is behavior-neutral: a certified-PSD row's stored diagonal
//!   is `>= 0`, so it can never veto rule 2.
//! - Rule 3: an off-diagonal pair `(i, j)` whose composed coefficient
//!   `0.5·(P[i][j] + P[j][i])` is nonzero is McCormick-lifted, so both
//!   endpoints join `R`.
//! - A diagonal failing rule 2 (some negative composed coefficient) joins `R`.
//! - Variables appearing only in linear terms are never required.
//!
//! Phase 0 b1 reconciliation (QPLIB_2546 divergence): see
//! [`classify_bound_requirements`].

use super::nonconvex::GQuadConstraint;
use otspot_num::sparse::CscMatrix;
use std::collections::{BTreeMap, BTreeSet};

/// Classification output of [`classify_bound_requirements`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundRequirement {
    /// Membership mask (length `n`) of the bound-requirement set `R`.
    pub required: Vec<bool>,
    /// `convex_rows[k]`: quadratic part of `quad[k]` is PSD-certified
    /// (rule 1), so the row is kept convex and lifts nothing.
    pub convex_rows: Vec<bool>,
    /// `rule2_diag[i]`: the diagonal lift coefficient of `x_i^2` is `>= 0` in
    /// every nonconvex row and the objective, so only the under-estimator
    /// `w_ii >= x_i^2` is valid (a variable without any diagonal is vacuously
    /// `true` — there is nothing to lift).
    pub rule2_diag: Vec<bool>,
}

/// Composed lift coefficients of one quadratic matrix `P`, keyed by the sorted
/// pair `(i, j)`, `i <= j`: diagonal `(i, i)` holds `0.5·P[i][i]`, off-diagonal
/// `(i, j)` holds `0.5·(P[i][j] + P[j][i])` — the exact `w_ij` coefficient
/// convention of `nonconvex::quad_pair_coeffs`, including its drop of composed
/// zeros (`P[i][j] = -P[j][i]` cancellation).
fn lift_coeffs(p: &CscMatrix) -> BTreeMap<(usize, usize), f64> {
    let cp = p.col_ptr();
    let ri = p.row_ind();
    let va = p.values();
    let mut coeffs: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for j in 0..p.ncols() {
        for k in cp[j]..cp[j + 1] {
            let i = ri[k];
            *coeffs
                .entry(if i <= j { (i, j) } else { (j, i) })
                .or_insert(0.0) += 0.5 * va[k];
        }
    }
    coeffs.retain(|_, v| *v != 0.0);
    coeffs
}

/// The principal submatrix `S_T` of the symmetric composition `S = (P + Pᵀ)/2`
/// over `support` (renumbered `0..support.len()`), built from the lift-coeff
/// map: off-diagonal `S[a][b] = S[b][a] = c_ab`, diagonal `S[a][a] = 2·c_aa`
/// so that `½ xᵀ S x` reproduces the row's quadratic form. Indices outside the
/// support carry only zeros, and `S` is PSD iff `S_T` is, so the PSD test
/// costs `O(nnz(support))` instead of the full `n`-column factorization sweep.
/// `from_triplets` sums the two emitted triangles exactly and drops composed
/// magnitudes `<= DROP_TOL` (`1e-15`), far below the Cholesky contract's
/// absolute jitter floor (`1e-14`), so the certificate is unchanged; it also
/// rejects non-finite coefficients, which an unvalidated public caller can
/// smuggle in, hence the `Result`.
fn support_submatrix(
    coeffs: &BTreeMap<(usize, usize), f64>,
    support: &[usize],
) -> Result<CscMatrix, String> {
    let pos = |idx: usize| {
        support
            .binary_search(&idx)
            .expect("support is collected from exactly these key indices")
    };
    let mut rows = Vec::with_capacity(coeffs.len() * 2);
    let mut cols = Vec::with_capacity(coeffs.len() * 2);
    let mut vals = Vec::with_capacity(coeffs.len() * 2);
    for (&(i, j), &c) in coeffs {
        let (a, b) = (pos(i), pos(j));
        if i == j {
            rows.push(a);
            cols.push(a);
            vals.push(2.0 * c);
        } else {
            rows.extend_from_slice(&[a, b]);
            cols.extend_from_slice(&[b, a]);
            vals.extend_from_slice(&[c, c]);
        }
    }
    let m = support.len();
    CscMatrix::from_triplets(&rows, &cols, &vals, m, m)
        .map_err(|e| format!("PSD support submatrix: {e:?}"))
}

/// Sufficient positive-definiteness certificate: strict diagonal dominance.
/// `S` is symmetric with both triangles stored, so one accumulation pass over
/// columns yields each Gershgorin radius exactly, and `diag[j] > radius[j]`
/// bounds every eigenvalue below by `min_j (diag[j] - radius[j]) > 0`.
fn gershgorin_pd(s: &CscMatrix) -> bool {
    let m = s.ncols();
    let mut diag = vec![0.0; m];
    let mut radius = vec![0.0; m];
    for j in 0..m {
        let (rows, vals) = s.column(j);
        for (&r, &v) in rows.iter().zip(vals) {
            if r == j {
                diag[j] = v;
            } else {
                radius[j] += v.abs();
            }
        }
    }
    (0..m).all(|j| diag[j] > radius[j])
}

/// Rule-1 test: the quadratic form described by `coeffs` is PSD-certified.
/// `true` = convex; the Cholesky jitter-band clamp (`convexity_unproven`) and
/// a negative-minor rejection both stay on the nonconvex side (conservative).
fn certifiably_psd(coeffs: &BTreeMap<(usize, usize), f64>) -> Result<bool, String> {
    if coeffs.is_empty() {
        return Ok(true);
    }
    let mut support = BTreeSet::new();
    for &(i, j) in coeffs.keys() {
        support.insert(i);
        support.insert(j);
    }
    let support: Vec<usize> = support.into_iter().collect();
    let s = support_submatrix(coeffs, &support)?;
    Ok(gershgorin_pd(&s) || super::qcqp::chol_certifies_psd(&s, support.len()))
}

/// Fold a nonconvex quadratic form into the classification: nonzero
/// off-diagonal pairs add both endpoints to `R` (rule 3), diagonals feed the
/// rule-2 running minimum.
fn mark_nonconvex_terms(
    coeffs: &BTreeMap<(usize, usize), f64>,
    n: usize,
    required: &mut [bool],
    min_diag: &mut [f64],
) -> Result<(), String> {
    for (&(i, j), &c) in coeffs {
        if i >= n || j >= n {
            return Err(format!(
                "quadratic index ({i}, {j}) out of range for n = {n}"
            ));
        }
        if i == j {
            if c < min_diag[i] {
                min_diag[i] = c;
            }
        } else {
            required[i] = true;
            required[j] = true;
        }
    }
    Ok(())
}

fn check_shape(field: &str, p: &CscMatrix, n: usize) -> Result<(), String> {
    if p.nrows() != n || p.ncols() != n {
        return Err(format!(
            "{field} shape ({}, {}) != (n, n) = ({n}, {n})",
            p.nrows(),
            p.ncols()
        ));
    }
    Ok(())
}

/// Classify the quadratic terms of a normalized (all `<=` rows, minimize
/// objective) nonconvex QCQP per the module-doc rules and return the bound
/// requirement. Dimension mismatches are reported rather than trusted: this
/// is `pub` on public struct fields, and the PSD/`R` walk reads them directly.
///
/// Phase 0 b1 reconciliation (plan §2/§3.1; table
/// `/tmp/opencode/qcqp_phase0/REPORT.md`, external Gershgorin→eigvalsh tol
/// `1e-9·|S|max`; measured 2026-11-13, pinned by
/// `tests/qcqp_classify_qplib_phase0.rs`): this classifier agrees with b1 on
/// QPLIB 8595 (2500), 8605 (5000), 8685 (772). QPLIB_2546 diverges — b1 415
/// vs `|R|` 799 — and the divergence is by design (§3.1: PSD judgement shares
/// the `sparse_cholesky_lower` contract; unproven/clamped is never convex).
/// The constraint-derived part is exactly 415; the objective `P0` (n 1015,
/// 808 composed terms, support 424) fails Gershgorin and
/// `sparse_cholesky_lower` returns `Err` (certifiably negative 2×2 minor) on
/// its 424×424 support submatrix, so rule 3 adds its 384 nonzero-pair
/// endpoints (measured `neg_diag = 0`; the other 40 support variables are
/// rule-2 diagonals). The external eigvalsh tol counts that minor as convex;
/// the repo contract does not, and no b1 match is required for such rows.
pub fn classify_bound_requirements(
    n: usize,
    p0: Option<&CscMatrix>,
    quad: &[GQuadConstraint],
) -> Result<BoundRequirement, String> {
    let mut required = vec![false; n];
    let mut min_diag = vec![0.0f64; n];
    let mut convex_rows = Vec::with_capacity(quad.len());
    for (k, qc) in quad.iter().enumerate() {
        check_shape(&format!("quad[{k}].p"), &qc.p, n)?;
        let coeffs = lift_coeffs(&qc.p);
        if certifiably_psd(&coeffs)? {
            convex_rows.push(true);
            continue;
        }
        convex_rows.push(false);
        mark_nonconvex_terms(&coeffs, n, &mut required, &mut min_diag)?;
    }
    if let Some(p0) = p0 {
        check_shape("p0", p0, n)?;
        let coeffs = lift_coeffs(p0);
        if !certifiably_psd(&coeffs)? {
            mark_nonconvex_terms(&coeffs, n, &mut required, &mut min_diag)?;
        }
    }
    let rule2_diag: Vec<bool> = min_diag.iter().map(|&c| c >= 0.0).collect();
    for (i, &under) in rule2_diag.iter().enumerate() {
        if !under {
            required[i] = true;
        }
    }
    Ok(BoundRequirement {
        required,
        convex_rows,
        rule2_diag,
    })
}

/// D1-conservative rule-1 exemption: [`classify_bound_requirements`] trusts a
/// PSD-certified row/objective to need no bounds; this filter additionally
/// requires every variable of the row's off-diagonal lift support to have a
/// finite raw bound. A row failing that test is demoted to the nonconvex side
/// (rule 3 marks its off-diagonal endpoints, its diagonals feed the rule-2
/// minimum scan), so its support joins `R`.
///
/// Plan §3.1 footnote ("同じ変数が他の行で持ち上げられても包含は崩れない") only
/// covers variables *other* rows already lift: the shared `w` column is
/// constrained only where interval/McCormick rows exist, and an exempt PSD row
/// over a variable with a non-finite raw bound lifts pairs that emit no such
/// rows (`build_relax` skips envelope rows for non-finite endpoints, plan
/// §3.2), so the parent-approved conservative reading (decision D1: "迷ったら
/// 非凸側") gates the exemption on raw-bound finiteness of that row's own
/// off-diagonal support. Diagonal-only PSD rows are never demoted: their
/// single diagonal pair carries the rule-2 under-estimator `w_ii >= 0` cut,
/// which needs no box.
///
/// Demotion can never flip `rule2_diag` through a *negative* diagonal (a
/// PSD-certified composed form has `0.5·S_ii >= 0` on its support), but the
/// minimum scan is still fed so the function does not rely on that lemma.
///
/// The result is the acceptance set for Phase 2 bound derivation ("変数 R 入り
/// → 最終的に Phase 2 導出後も非有限なら拒否"); the current G2 acceptance gate
/// (`NonconvexQcqp::validate`, plan §3.2 / decision D8 / §5 sentinel 1) uses
/// the raw classification so plan §5 example 1 (PSD ellipsoid row over free
/// variables) is not rejected on the bound requirement itself -- its eventual
/// NotSupported comes from the missing `x̃` base point (plan line 110), which
/// is a Phase 2 local-solver obligation, not a bound-requirement error.
pub fn effective_bound_requirement(
    n: usize,
    p0: Option<&CscMatrix>,
    quad: &[GQuadConstraint],
    lb: &[f64],
    ub: &[f64],
) -> Result<BoundRequirement, String> {
    if lb.len() != n {
        return Err(format!("lb length {} != n {n}", lb.len()));
    }
    if ub.len() != n {
        return Err(format!("ub length {} != n {n}", ub.len()));
    }
    let mut req = classify_bound_requirements(n, p0, quad)?;
    let mut min_diag = vec![0.0f64; n];
    let has_free_support = |coeffs: &BTreeMap<(usize, usize), f64>| {
        coeffs.iter().any(|(key, _)| {
            let &(i, j) = key;
            i != j
                && !(lb[i].is_finite()
                    && ub[i].is_finite()
                    && lb[j].is_finite()
                    && ub[j].is_finite())
        })
    };
    for (k, qc) in quad.iter().enumerate() {
        if !req.convex_rows[k] {
            continue;
        }
        let coeffs = lift_coeffs(&qc.p);
        if !has_free_support(&coeffs) {
            continue;
        }
        req.convex_rows[k] = false;
        mark_nonconvex_terms(&coeffs, n, &mut req.required, &mut min_diag)?;
    }
    if let Some(p0) = p0 {
        let coeffs = lift_coeffs(p0);
        if has_free_support(&coeffs) {
            mark_nonconvex_terms(&coeffs, n, &mut req.required, &mut min_diag)?;
        }
    }
    for (i, under) in req.rule2_diag.iter_mut().enumerate() {
        *under &= min_diag[i] >= 0.0;
    }
    for (i, &under) in req.rule2_diag.iter().enumerate() {
        if !under {
            req.required[i] = true;
        }
    }
    Ok(req)
}

/// Split `bounds` into `lb`/`ub`, rejecting the problem when any variable of
/// the §3.1 bound-requirement set `R` has a non-finite bound.
///
/// G1 gate boundary: `R` is the raw classification — Phase 2 bound derivation
/// (FBBT, objective cutoff) does not exist yet, so acceptance is stricter than
/// §3.2's final rule ("reject only non-finite remaining after derivation") and
/// the downstream relaxation, which is R-aware (decision D8):
/// `NonconvexQcqp::validate` demands finiteness on exactly `R` as well, and
/// `build_relax` keeps infinite boxes out of every LP by skipping box rows
/// and envelope rows on non-finite endpoints.
///
/// The all-finite fast path returns without classification, keeping the
/// current solver behavior (and cost) bit-identical for problems that the old
/// every-variable guard accepted. The rejection message reports the number of
/// non-finite variables inside `R` and the first such index.
pub fn require_finite_bounds(
    n: usize,
    p0: Option<&CscMatrix>,
    quad: &[GQuadConstraint],
    bounds: &[(f64, f64)],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if bounds.len() != n {
        return Err(format!("bounds length {} != n {n}", bounds.len()));
    }
    if bounds.iter().all(|&(l, u)| l.is_finite() && u.is_finite()) {
        let lb = bounds.iter().map(|&(l, _)| l).collect();
        let ub = bounds.iter().map(|&(_, u)| u).collect();
        return Ok((lb, ub));
    }
    let req = classify_bound_requirements(n, p0, quad)?;
    let bad: Vec<(usize, f64, f64)> = bounds
        .iter()
        .enumerate()
        .filter(|&(j, _)| req.required[j])
        .filter_map(|(j, &(l, u))| (!(l.is_finite() && u.is_finite())).then_some((j, l, u)))
        .collect();
    if let Some(&(j, l, u)) = bad.first() {
        return Err(format!(
            "nonconvex QCQP requires finite bounds on every bound-required \
             variable set R (plan §3.1); {} of the {} required variables \
             have a non-finite bound (first: variable {j} with bounds ({l}, {u}))",
            bad.len(),
            req.required.iter().filter(|b| **b).count()
        ));
    }
    let lb = bounds.iter().map(|&(l, _)| l).collect();
    let ub = bounds.iter().map(|&(_, u)| u).collect();
    Ok((lb, ub))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic::NonconvexQcqp;

    /// `(1/2) x' P x + q' x + r` row from explicit composed coefficients.
    fn quad_row(n: usize, p: Vec<(usize, usize, f64)>, q: Vec<f64>, r: f64) -> GQuadConstraint {
        GQuadConstraint {
            p: dense(n, p),
            q,
            r,
        }
    }

    fn dense(n: usize, entries: Vec<(usize, usize, f64)>) -> CscMatrix {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (i, j, v) in entries {
            rows.push(i);
            cols.push(j);
            vals.push(v);
        }
        CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
    }

    /// Plan §5 example 1 (rule 1): `min x1 s.t. x0^2 - x0*x1 + x1^2 <= 1`,
    /// `x` free; true optimum `-2/sqrt(3)` at `x0 = -1/sqrt(3)`. The row
    /// matrix `[[2,-1],[-1,2]]` is PSD, so the row is convex, nothing lifts,
    /// and `R = {}` even with free variables.
    #[test]
    fn rule1_psd_constraint_row_needs_no_bounds() {
        let row = quad_row(
            2,
            vec![(0, 0, 2.0), (1, 1, 2.0), (0, 1, -1.0), (1, 0, -1.0)],
            vec![0.0, 0.0],
            -1.0,
        );
        let req = classify_bound_requirements(2, None, std::slice::from_ref(&row)).unwrap();
        assert_eq!(req.convex_rows, vec![true], "PSD row must classify convex");
        assert_eq!(req.required, vec![false, false], "R must be empty");
        // Interpretation pin (rule-2 scan skips convex rows): the row's
        // diagonals are positive anyway (a certified-PSD diagonal can never be
        // negative), so vacuous rule-2 membership is the observable outcome.
        assert_eq!(req.rule2_diag, vec![true, true]);
        let free = [(f64::NEG_INFINITY, f64::INFINITY); 2];
        let (lb, ub) = require_finite_bounds(2, None, &[row], &free).unwrap();
        assert!(lb.iter().all(|v| *v == f64::NEG_INFINITY));
        assert!(ub.iter().all(|v| *v == f64::INFINITY));
    }

    /// Plan §5 example 2 (rule 1 on the objective): `min x0^2 + x1 s.t.
    /// x0 - x1 = 0`, `x` free; true optimum `-1/4` at `(-1/2, -1/2)`. The PSD
    /// objective needs no lifting, the equality is linear, and `x1` is
    /// linear-only: `R = {}`.
    #[test]
    fn rule1_psd_objective_and_linear_only_var_need_no_bounds() {
        let p0 = dense(2, vec![(0, 0, 2.0)]);
        let req = classify_bound_requirements(2, Some(&p0), &[]).unwrap();
        assert_eq!(req.required, vec![false, false]);
        assert!(req.rule2_diag[0], "x0^2 is a rule-2 under-diagonal");
        let free = [(f64::NEG_INFINITY, f64::INFINITY); 2];
        require_finite_bounds(2, Some(&p0), &[], &free).unwrap();
    }

    /// Plan §5 example 3 (rule 2 + rule 3): `min x0^2 - x2*x3` with `x2, x3`
    /// free and `x0` free; true optimum `-4` on the box `x2, x3 in [0,2]`.
    /// The off-diagonal `-x2*x3` makes the objective indefinite, so `x2, x3`
    /// join `R` (rule 3); the objective diagonal `+x0^2` is rule 2.
    #[test]
    fn rule2_under_diagonal_excludes_it_from_r() {
        let p0 = dense(4, vec![(0, 0, 2.0), (2, 3, -1.0), (3, 2, -1.0)]);
        let req = classify_bound_requirements(4, Some(&p0), &[]).unwrap();
        assert_eq!(req.required, vec![false, false, true, true]);
        assert!(req.rule2_diag[0], "diagonal coeff +1 >= 0 -> rule 2");
    }

    /// Plan §5 example 4 (rule 2 veto from a flipped `>=` row, rule 3-style
    /// diagonal): `min x1 s.t. x0^2 - x1 >= 0` normalizes to `-x0^2 + x1 <= 0`
    /// with diagonal lift coeff `-1 < 0`, so `R = {x0}` and rule 2 does not
    /// fire; true optimum `0` (Phase 2 FBBT would derive `x0 in [-1, 1]` from
    /// `x1 <= 1`). Sentinel for plan §5 sentinel 2: accepting the pre-flip
    /// `+1` coefficient here would classify `x0` as rule 2 and drop it from R.
    #[test]
    fn negative_flipped_diagonal_vetoes_rule2_and_requires_bound() {
        let row = quad_row(2, vec![(0, 0, -2.0)], vec![0.0, 1.0], 0.0);
        let req = classify_bound_requirements(2, None, &[row]).unwrap();
        assert_eq!(req.convex_rows, vec![false]);
        assert_eq!(req.required, vec![true, false]);
        assert!(!req.rule2_diag[0]);
    }

    /// Plan §5 example 5 (equality split, the 8605 shape): `min x0 + x1 s.t.
    /// x0^2 + 2*x1^2 = 2`, `x` free; true optimum `-sqrt(3)`. The `P` half is
    /// a convex row but the `-P` half is nonconvex with negative diagonals,
    /// so both variables join `R` (§3.1 table note).
    #[test]
    fn equality_split_moves_psd_quadratic_into_r_via_neg_half() {
        let p_half = quad_row(2, vec![(0, 0, 2.0), (1, 1, 4.0)], vec![0.0, 0.0], -2.0);
        let n_half = quad_row(2, vec![(0, 0, -2.0), (1, 1, -4.0)], vec![0.0, 0.0], 2.0);
        let req = classify_bound_requirements(2, None, &[p_half, n_half]).unwrap();
        assert_eq!(req.convex_rows, vec![true, false]);
        assert_eq!(req.required, vec![true, true]);
    }

    /// Rule 1 covers a PSD *objective* with nonzero off-diagonals:
    /// `min 1/2 (x0 + x1)^2` has composed matrix `[[1,1],[1,1]]` — not
    /// strictly diagonally dominant (Gershgorin fails), yet the Cholesky
    /// factorizes it rank-deficient *without* clamping, which certifies PSD.
    /// Clamped (`convexity_unproven`) factorizations must NOT count, so this
    /// also pins the `Ok ∧ !clamped` half of the contract.
    #[test]
    fn rank_deficient_unclamped_psd_objective_is_convex() {
        let p0 = dense(2, vec![(0, 0, 1.0), (1, 1, 1.0), (0, 1, 1.0), (1, 0, 1.0)]);
        let req = classify_bound_requirements(2, Some(&p0), &[]).unwrap();
        assert_eq!(req.required, vec![false, false]);
    }

    /// A diagonal sitting inside the Cholesky jitter band (e.g. `-2e-10`,
    /// the `qcqp_guard::jitter_band_indefinite_qcqp_falls_back_to_global`
    /// shape) is clamped by `sparse_cholesky_lower`, i.e. convexity-unproven:
    /// the row must fall on the nonconvex side and its negative diagonal
    /// member requires a bound.
    #[test]
    fn clamped_jitter_band_row_is_not_certified_convex() {
        let row = quad_row(1, vec![(0, 0, -2e-10)], vec![1.0], 0.0);
        let req = classify_bound_requirements(1, None, &[row]).unwrap();
        assert_eq!(req.convex_rows, vec![false], "clamped != certified PSD");
        assert_eq!(req.required, vec![true]);
    }

    /// Fast path: an all-finite box returns `lb`/`ub` unchanged without ever
    /// consulting `R` — pinned by passing a deliberately mis-shaped `p0`
    /// alongside the finite box (classification would reject it, the fast path
    /// never runs it).
    #[test]
    fn require_finite_bounds_fast_path_passes_finite_box() {
        let p0 = dense(3, vec![(0, 1, 1.0), (1, 0, 1.0)]);
        let bounds = [(0.1, 3.0), (0.1, 5.0)];
        let (lb, ub) = require_finite_bounds(2, Some(&p0), &[], &bounds).unwrap();
        assert_eq!(lb, vec![0.1, 0.1]);
        assert_eq!(ub, vec![3.0, 5.0]);
    }

    /// An unbounded variable *inside* `R` is rejected with its count and first
    /// index (`x0*x1 >= 1` flipped: both endpoints of the bilinear pair are
    /// required).
    #[test]
    fn require_finite_bounds_rejects_unbounded_required_variable() {
        let row = quad_row(2, vec![(0, 1, -1.0), (1, 0, -1.0)], vec![0.0, 0.0], 1.0);
        let bounds = [(f64::NEG_INFINITY, f64::INFINITY), (0.1, 3.0)];
        let err = require_finite_bounds(2, None, &[row], &bounds).unwrap_err();
        assert!(err.contains("1 of the 2 required"), "{err}");
        assert!(err.contains("variable 0"), "{err}");
    }

    /// A non-finite bound on a variable *outside* `R` (linear-only `x1`)
    /// passes the G1 gate **and** `NonconvexQcqp::validate` (decision D8):
    /// the R-aware relaxation (`build_relax` skips box rows on non-finite
    /// node bounds) is the consumer that keeps infinite boxes out of the LP,
    /// so free non-`R` variables are a supported input, not a second-line
    /// rejection. Free variables *inside* `R` stay rejected by both layers
    /// (`require_finite_bounds_rejects_unbounded_required_variable` and
    /// `validate_requires_finite_bounds_only_on_r` in nonconvex.rs).
    #[test]
    fn require_finite_bounds_defers_non_required_variables_to_validate() {
        let row = quad_row(2, vec![(0, 0, -2.0)], vec![0.0, 1.0], 0.0);
        let bounds = [(-2.0, 2.0), (f64::NEG_INFINITY, f64::INFINITY)];
        let (lb, ub) = require_finite_bounds(2, None, std::slice::from_ref(&row), &bounds).unwrap();
        assert_eq!(lb, vec![-2.0, f64::NEG_INFINITY]);
        assert_eq!(ub, vec![2.0, f64::INFINITY]);
        let nc = NonconvexQcqp {
            n: 2,
            p0: None,
            q0: vec![1.0, 0.0],
            quad: vec![row],
            g_lin: CscMatrix::new(0, 2),
            h_lin: vec![],
            a_eq: CscMatrix::new(0, 2),
            b_eq: vec![],
            lb,
            ub,
        };
        nc.validate()
            .expect("x1 outside R may stay free end-to-end (D8)");
    }

    /// Dimension guards on the public entry: a mismatched `p0`/`bounds` is an
    /// `Err`, not an out-of-range panic.
    #[test]
    fn classify_and_gate_reject_mismatched_dimensions() {
        let big = dense(3, vec![(0, 1, 1.0), (1, 0, 1.0)]);
        let row = GQuadConstraint {
            p: big.clone(),
            q: vec![0.0, 0.0, 0.0],
            r: 0.0,
        };
        let err = classify_bound_requirements(2, None, &[row]).unwrap_err();
        assert!(err.contains("quad[0].p shape"), "{err}");
        let err = classify_bound_requirements(2, Some(&big), &[]).unwrap_err();
        assert!(err.contains("p0 shape"), "{err}");
        let small: [(f64, f64); 1] = [(0.0, 1.0)];
        let err = require_finite_bounds(2, None, &[], &small).unwrap_err();
        assert!(err.contains("bounds length"), "{err}");
    }

    /// D1: a PSD row whose *off-diagonal* support touches a non-finite raw
    /// bound is demoted to the nonconvex side and its endpoints join `R`
    /// (plan §3.1 footnote guards only variables other rows already lift; the
    /// parent decision D1 takes the conservative "迷ったら非凸側" reading).
    #[test]
    fn d1_demotes_psd_row_with_free_off_diagonal_support() {
        // (x0 - x1)^2 <= 1: PSD, off-diagonal support {x0, x1}.
        let row = quad_row(
            2,
            vec![(0, 0, 2.0), (1, 1, 2.0), (0, 1, -2.0), (1, 0, -2.0)],
            vec![0.0, 0.0],
            -1.0,
        );
        let free = [f64::NEG_INFINITY; 2];
        let finite = [-1.0; 2];
        let eff = effective_bound_requirement(2, None, std::slice::from_ref(&row), &free, &finite)
            .unwrap();
        assert_eq!(eff.convex_rows, vec![false], "free endpoint demotes row");
        assert_eq!(eff.required, vec![true, true], "demoted endpoints join R");
        // With finite raw bounds the exemption stands (raw classification).
        let kept =
            effective_bound_requirement(2, None, std::slice::from_ref(&row), &finite, &finite)
                .unwrap();
        assert_eq!(kept, classify_bound_requirements(2, None, &[row]).unwrap());
    }

    /// D1: a diagonal-only PSD row is never demoted -- its single diagonal
    /// pair carries the rule-2 `w_ii >= 0` cut, which needs no box. This is
    /// what keeps plan §3.0-B's `min -x s.t. x^2 <= 1` (x free) admissible.
    #[test]
    fn d1_keeps_diagonal_only_psd_row_exempt_over_free_vars() {
        // x0^2 - 1 <= 0: PSD (diag-only), x0 free.
        let row = quad_row(2, vec![(0, 0, 2.0)], vec![0.0, 0.0], -1.0);
        let free = [(f64::NEG_INFINITY, f64::INFINITY); 2];
        let eff = effective_bound_requirement(
            2,
            None,
            std::slice::from_ref(&row),
            &[free[0].0, -1.0],
            &[free[0].1, 1.0],
        )
        .unwrap();
        assert_eq!(eff.convex_rows, vec![true]);
        assert_eq!(eff.required, vec![false, false]);
        assert!(eff.rule2_diag[0], "diagonal +1 stays rule 2");
    }

    /// D1 applies to a PSD objective the same way: a rank-deficient PSD
    /// objective with free off-diagonal support is demoted (its endpoints join
    /// `R`); the diagonal-only PSD objective stays exempt.
    #[test]
    fn d1_demotes_psd_objective_with_free_off_diagonal_support() {
        let p0 = dense(2, vec![(0, 0, 1.0), (1, 1, 1.0), (0, 1, 1.0), (1, 0, 1.0)]);
        let free = [(f64::NEG_INFINITY, f64::INFINITY); 2];
        let lb: Vec<f64> = free.iter().map(|&(l, _)| l).collect();
        let ub: Vec<f64> = free.iter().map(|&(_, u)| u).collect();
        let eff = effective_bound_requirement(2, Some(&p0), &[], &lb, &ub).unwrap();
        assert_eq!(eff.required, vec![true, true]);
        let diag = dense(2, vec![(0, 0, 2.0)]);
        let kept = effective_bound_requirement(2, Some(&diag), &[], &lb, &ub).unwrap();
        assert_eq!(kept.required, vec![false, false]);
    }

    /// Dimension guard of the D1 entry itself.
    #[test]
    fn effective_bound_requirement_rejects_mismatched_bounds() {
        let err = effective_bound_requirement(2, None, &[], &[0.0], &[0.0, 1.0]).unwrap_err();
        assert!(err.contains("lb length"), "{err}");
        let err = effective_bound_requirement(2, None, &[], &[0.0, 0.0], &[1.0]).unwrap_err();
        assert!(err.contains("ub length"), "{err}");
    }
}
