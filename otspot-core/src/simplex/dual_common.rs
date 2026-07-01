//! Shared simplex primitives (primal + dual).
//!
//! `compute_dual_vars` / `compute_reduced_costs` were duplicated byte-for-byte
//! between `dual.rs` and `dual_advanced/core.rs`; the inline reduced-cost loop
//! in `primal.rs::revised_simplex_core` was a third copy. Three sites meant any
//! future correction (e.g. dual reconstruction drift) had to land thrice — the
//! kind of bug-magnet the DRY audit flagged.
//!
//! `_into` variants let the primal core reuse pre-allocated buffers across the
//! hot-loop; the allocating wrappers are kept for dual paths that refresh
//! reduced costs less frequently.
//!
//! `basic_obj` consolidates `c_B^T x_B`, repeated 19× across the simplex tree
//! for objective reporting on Optimal/Timeout/SingularBasis exits.

use super::{extract_dual_info, extract_solution, SimplexOutcome, StandardForm};
use crate::basis::{BasisManager, LuBasis};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::sparse::{CscMatrix, SparseVec};
use std::sync::atomic::{AtomicBool, Ordering};

/// y = B^{-T} c_B written into the caller's buffer. `y_out.len()` is the basis
/// dimension m; the caller owns the allocation so a hot loop can reuse it.
pub(super) fn compute_dual_vars_into(
    c: &[f64],
    basis_mgr: &mut LuBasis,
    basis: &[usize],
    y_out: &mut [f64],
) {
    debug_assert_eq!(y_out.len(), basis.len());
    for (i, slot) in y_out.iter_mut().enumerate() {
        *slot = c[basis[i]];
    }
    basis_mgr.btran_dense(y_out);
}

/// y = B^{-T} c_B. Allocating wrapper; callers in cold paths (or those that
/// only need y once) use this. Hot loops should use `_into` + a reused buffer.
pub(super) fn compute_dual_vars(
    c: &[f64],
    basis_mgr: &mut LuBasis,
    basis: &[usize],
    m: usize,
) -> Vec<f64> {
    let mut y = vec![0.0f64; m];
    debug_assert_eq!(basis.len(), m);
    compute_dual_vars_into(c, basis_mgr, basis, &mut y);
    y
}

/// r_j = c_j − y^T a_j written into `rc_out`. Basic columns are zeroed; the
/// caller must pre-size `rc_out` to `n_price` and supply a `y_buf` of length m
/// (reused across iterations). `y_buf` is overwritten on every call.
pub(super) fn compute_reduced_costs_into(
    a: &CscMatrix,
    c: &[f64],
    basis_mgr: &mut LuBasis,
    is_basic: &[bool],
    n_price: usize,
    basis: &[usize],
    y_buf: &mut [f64],
    rc_out: &mut [f64],
) {
    debug_assert_eq!(rc_out.len(), n_price);
    compute_dual_vars_into(c, basis_mgr, basis, y_buf);
    for j in 0..n_price {
        if is_basic[j] {
            rc_out[j] = 0.0;
            continue;
        }
        let (rows, vals) = a.get_column(j).unwrap();
        let mut ya = 0.0;
        for (k, &row) in rows.iter().enumerate() {
            ya += y_buf[row] * vals[k];
        }
        rc_out[j] = c[j] - ya;
    }
}

/// r_j = c_j − y^T a_j with y = B^{-T} c_B. Basic columns are skipped
/// (r_j ≡ 0). Allocating wrapper used by dual paths that refresh r only after
/// a refactor.
pub(super) fn compute_reduced_costs(
    a: &CscMatrix,
    c: &[f64],
    basis_mgr: &mut LuBasis,
    is_basic: &[bool],
    n_price: usize,
    m: usize,
    basis: &[usize],
) -> Vec<f64> {
    let mut y = vec![0.0f64; m];
    let mut reduced_costs = vec![0.0f64; n_price];
    compute_reduced_costs_into(
        a,
        c,
        basis_mgr,
        is_basic,
        n_price,
        basis,
        &mut y,
        &mut reduced_costs,
    );
    reduced_costs
}

/// Standard-form dual infeasibility from reduced costs: maximum negative
/// reduced-cost violations over nonbasic priced columns. In the primal core all
/// nonbasic variables are at their lower bound, so dual feasibility is
/// `r_j >= 0`; the magnitude, not the count, is the progress signal.
pub(super) fn reduced_cost_dual_infeasibility(
    reduced_costs: &[f64],
    is_basic: &[bool],
    n_price: usize,
) -> f64 {
    let limit = n_price.min(reduced_costs.len()).min(is_basic.len());
    let mut infeas = 0.0;
    for j in 0..limit {
        if is_basic[j] {
            continue;
        }
        let rc = reduced_costs[j];
        if !rc.is_finite() {
            return f64::INFINITY;
        }
        let viol = (-rc).max(0.0);
        if viol > infeas {
            infeas = viol;
        }
    }
    infeas
}

/// Convert a `SimplexOutcome` to a `SolverResult`.
///
/// Single implementation shared by all three simplex dispatch paths:
/// - warm-start dual (`dual.rs`): `dual_unbounded_is_infeasible = true`
/// - cold-start primal phase-II (`dual.rs`): `dual_unbounded_is_infeasible = false`
/// - advanced dual warm-start (`dual_advanced`): `dual_unbounded_is_infeasible = true`
///
/// The only semantic difference between callers is the `Unbounded` arm:
/// dual simplex yields dual-unbounded ⇒ primal-infeasible; primal simplex
/// yields a genuinely unbounded objective.
#[allow(clippy::too_many_arguments)]
pub(super) fn outcome_to_result(
    outcome: SimplexOutcome,
    sf: &StandardForm,
    problem: &LpProblem,
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
    row_scale: &[f64],
    dual_unbounded_is_infeasible: bool,
) -> SolverResult {
    match outcome {
        SimplexOutcome::Optimal(obj, y) => {
            let solution = extract_solution(sf, basis, x_b, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info(sf, problem, &y, &solution, row_scale);
            let ws = WarmStartBasis {
                basis: basis.to_vec(),
                x_b: x_b.to_vec(),
            };
            SolverResult {
                status: SolveStatus::Optimal,
                objective: obj + sf.obj_offset,
                solution,
                dual_solution,
                reduced_costs,
                slack,
                warm_start_basis: Some(ws),
                ..Default::default()
            }
        }
        SimplexOutcome::Unbounded => {
            if dual_unbounded_is_infeasible {
                SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: f64::INFINITY,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
                    ..Default::default()
                }
            } else {
                SolverResult {
                    status: SolveStatus::Unbounded,
                    objective: f64::NEG_INFINITY,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
                    ..Default::default()
                }
            }
        }
        SimplexOutcome::Timeout(obj) => {
            let solution = extract_solution(sf, basis, x_b, col_scale);
            SolverResult {
                status: SolveStatus::Timeout,
                objective: obj + sf.obj_offset,
                solution,
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
                warm_start_basis: None,
                ..Default::default()
            }
        }
        SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
    }
}

/// Verify an LP `Unbounded` exit against a re-derived recession ray (symmetric
/// to the Phase-I Farkas gate: unverified ray ⇒ honest Timeout).
///
/// Eta drift can falsely read `B⁻¹a_q ≤ 0`; this rebuilds a clean LU and
/// confirms column `q < n_enter` is improving (`r_q < −dual_tol`) AND unbounded:
///   - structural/slack rows: `(B⁻¹a_q)[i] ≤ ray_floor` (each finite UB is a
///     slack row, so a real UB limit surfaces as a positive component);
///   - artificial rows (`basis[i] ≥ n_enter`): `|(B⁻¹a_q)[i]| ≤ ray_floor`
///     (a `< 0` entry increases the artificial off 0 ⇒ not an original-LP ray).
///
/// `ray_floor = EPSILON·max(1,‖B⁻¹a_q‖∞)` — machine-noise scale, not `PIVOT_TOL`
/// (a real positive pivot is a leaving row; the looser tol leaks false Unbounded).
/// `n_enter` excludes Big-M artificials; pure-slack paths pass `n_enter = n_price`.
/// Empty witness ⇒ Timeout.
pub(super) fn lp_unbounded_ray_verified(
    a: &CscMatrix,
    basis: &[usize],
    c: &[f64],
    m: usize,
    n_price: usize,
    n_enter: usize,
    options: &SolverOptions,
) -> bool {
    let mut basis_mgr = match LuBasis::new_timed(a, basis, options.max_etas, options.deadline) {
        Ok(bm) => bm,
        Err(_) => return false,
    };
    let mut in_basis = vec![false; n_price];
    for &col in basis {
        if col < n_price {
            in_basis[col] = true;
        }
    }
    // y = B⁻ᵀ c_B
    let mut y: Vec<f64> = basis.iter().map(|&col| c[col]).collect();
    basis_mgr.btran_dense(&mut y);

    for q in 0..n_enter {
        if in_basis[q] {
            continue;
        }
        let Ok((rows, vals)) = a.get_column(q) else {
            continue;
        };
        let rc = c[q]
            - rows
                .iter()
                .zip(vals.iter())
                .map(|(&r, &v)| v * y[r])
                .sum::<f64>();
        if rc >= -options.dual_tol {
            continue;
        }
        let mut d_sv = SparseVec {
            indices: rows.to_vec(),
            values: vals.to_vec(),
            len: m,
        };
        basis_mgr.ftran(&mut d_sv);
        // A recession ray needs every basic structural/slack component ≤ 0
        // (increasing x_q from lb=0 with ub=∞ never drives a basic variable below
        // its lower bound). The floor is machine-noise scale (`EPSILON·max(1,‖d‖∞)`),
        // matching the simplex's own last-chance ratio test: any *real* positive
        // pivot — even one below `PIVOT_TOL` (0 < d_i < 1e-8) — is a genuine leaving
        // row, so the direction is bounded, NOT a ray. Using `PIVOT_TOL` here instead
        // would re-admit those small-positive pivots and leak a false-Unbounded.
        //
        // Basic artificials (basis[i] ≥ n_enter) need the stricter |d_i| ≤ floor:
        // d_i < 0 there *increases* the artificial off 0, so the direction stays
        // feasible only in the augmented system, not the original LP — it is not a
        // recession ray. Allowing it re-admits a false-Unbounded whenever a
        // degenerate artificial lingers in the Phase-II basis (the symptom this gate
        // exists to prevent).
        let d = d_sv.to_dense();
        let scale = d.iter().map(|v| v.abs()).fold(1.0_f64, f64::max);
        let ray_floor = f64::EPSILON * scale;
        let ray_ok = d.iter().enumerate().all(|(i, &di)| {
            if basis[i] >= n_enter {
                di.abs() <= ray_floor
            } else {
                di <= ray_floor
            }
        });
        if ray_ok {
            return true;
        }
    }
    false
}

/// c_B^T x_B = Σ c[basis[i]] · x_B[i]. Shared between primal/dual simplex for
/// objective reporting on Optimal / Timeout / SingularBasis exits.
pub(super) fn basic_obj(c: &[f64], basis: &[usize], x_b: &[f64]) -> f64 {
    debug_assert_eq!(basis.len(), x_b.len());
    basis.iter().zip(x_b.iter()).map(|(&j, &v)| c[j] * v).sum()
}

/// Anti-cycling: enter Bland mode after K = `(NO_PROGRESS_TRIGGER_FACTOR * m).max(NO_PROGRESS_MIN)`
/// consecutive no-progress iterations.
pub(super) const NO_PROGRESS_TRIGGER_FACTOR: usize = 3;
pub(super) const NO_PROGRESS_MIN: usize = 100;

/// Relative improvement threshold for progress detection:
/// improvement is counted only when `best - current > |best| * NO_PROGRESS_REL_EPS`.
pub(super) const NO_PROGRESS_REL_EPS: f64 = 1e-12;

/// Returns `true` when `current` is strictly better than `best` by more than
/// the relative noise floor `best.abs().max(floor) * NO_PROGRESS_REL_EPS`.
///
/// `floor = 0.0`: noise floor scales with `|best|` (use in dual anti-cycling).
/// `floor = 1.0`: guards against noise resets when `best ≈ 0`, preventing
/// sub-eps improvements from counting as real progress (primal Phase I).
pub(super) fn made_progress_with_floor(best: f64, current: f64, floor: f64) -> bool {
    best - current > best.abs().max(floor) * NO_PROGRESS_REL_EPS
}

/// Periodic deadline-check interval inside the m-BTRAN gamma loop.
/// Checked every `GAMMA_DEADLINE_CHECK_INTERVAL` rows; large enough to
/// amortize the `Instant::now()` syscall but small enough to catch an
/// expired deadline before a full O(m²) budget overrun on large warm-starts.
const GAMMA_DEADLINE_CHECK_INTERVAL: usize = 10;

/// γ_i = ||(B^{-1})_{i,:}||² for each basis row i via m BTRANs.
///
/// Used by DSE leaving strategies at warm-start and after refactor to set
/// the exact initial weights. Cost O(m²); called once per warm-start solve
/// or refactor boundary.
///
/// Returns `None` when `deadline` is `Some` and the deadline expires inside
/// the loop (checked every `GAMMA_DEADLINE_CHECK_INTERVAL` rows), or when
/// `cancel_flag` is `Some` and the flag is set. Callers must propagate this
/// as a `Timeout` outcome so the solver stays within its time budget on large
/// warm-start solves.
pub(super) fn recompute_gamma_truth(
    basis_mgr: &mut LuBasis,
    m: usize,
    deadline: Option<std::time::Instant>,
    cancel_flag: Option<&AtomicBool>,
) -> Option<Vec<f64>> {
    let mut gamma_truth = vec![0.0f64; m];
    let mut e_i = vec![0.0f64; m];
    let mut rho_i = vec![0.0f64; m];
    for i in 0..m {
        if i % GAMMA_DEADLINE_CHECK_INTERVAL == 0 {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return None;
            }
            if cancel_flag.is_some_and(|f| f.load(Ordering::Relaxed)) {
                return None;
            }
        }
        e_i.iter_mut().for_each(|v| *v = 0.0);
        e_i[i] = 1.0;
        let mut sv = SparseVec::from_dense(&e_i);
        basis_mgr.btran(&mut sv);
        sv.to_dense_into(&mut rho_i);
        gamma_truth[i] = rho_i.iter().map(|&v| v * v).sum();
    }
    Some(gamma_truth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::LuBasis;
    use crate::sparse::CscMatrix;

    /// A = [I_m | extras]. Extra column (m + k) is a single +2.0 at row (k mod m)
    /// so r_{m+k} = c_{m+k} − 2·c[k mod m] under the identity basis.
    fn make_identity_plus(n: usize, m: usize) -> (CscMatrix, Vec<f64>, Vec<usize>) {
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for j in 0..m {
            rows.push(j);
            cols.push(j);
            vals.push(1.0);
        }
        for j in m..n {
            rows.push((j - m) % m);
            cols.push(j);
            vals.push(2.0);
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        let basis: Vec<usize> = (0..m).collect();
        let c: Vec<f64> = (0..n).map(|j| (j as f64) + 1.0).collect();
        (a, c, basis)
    }

    #[test]
    fn dual_vars_identity_basis_returns_c_b() {
        let m = 4;
        let (a, c, basis) = make_identity_plus(m + 3, m);
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        let y = compute_dual_vars(&c, &mut bm, &basis, m);
        for i in 0..m {
            assert!(
                (y[i] - c[i]).abs() < 1e-12,
                "y[{}] = {} expected {}",
                i,
                y[i],
                c[i]
            );
        }
    }

    /// `_into` must agree byte-for-byte with the allocating wrapper. Multiple
    /// invocations into the same buffer must not leave stale state.
    #[test]
    fn dual_vars_into_matches_allocating_and_is_reuse_safe() {
        let m = 4;
        let (a, c, basis) = make_identity_plus(m + 3, m);
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        let y_alloc = compute_dual_vars(&c, &mut bm, &basis, m);

        let mut y_into = vec![999.0f64; m];
        compute_dual_vars_into(&c, &mut bm, &basis, &mut y_into);
        for i in 0..m {
            assert!((y_into[i] - y_alloc[i]).abs() < 1e-14);
        }
        // Second call into the same buffer (stale sentinel left from above) must
        // still produce the canonical answer — covers buffer-reuse correctness.
        for slot in y_into.iter_mut() {
            *slot = -42.0;
        }
        compute_dual_vars_into(&c, &mut bm, &basis, &mut y_into);
        for i in 0..m {
            assert!((y_into[i] - y_alloc[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn reduced_costs_identity_basis_match_closed_form() {
        let m = 3;
        let n = m + 3;
        let (a, c, basis) = make_identity_plus(n, m);
        let is_basic: Vec<bool> = (0..n).map(|j| j < m).collect();
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        let r = compute_reduced_costs(&a, &c, &mut bm, &is_basic, n, m, &basis);

        for j in 0..m {
            assert_eq!(r[j], 0.0);
        }
        for j in m..n {
            let expected = c[j] - 2.0 * c[(j - m) % m];
            assert!(
                (r[j] - expected).abs() < 1e-12,
                "r[{}] = {} expected {}",
                j,
                r[j],
                expected
            );
        }
    }

    /// `_into` variant must produce identical reduced costs and must zero
    /// basic-column slots even when the caller's buffer holds stale non-zero
    /// values (primal hot loop reuses `rc_vec` across iterations).
    #[test]
    fn reduced_costs_into_matches_allocating_and_clears_basic_slots() {
        let m = 3;
        let n = m + 3;
        let (a, c, basis) = make_identity_plus(n, m);
        let is_basic: Vec<bool> = (0..n).map(|j| j < m).collect();
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();

        let r_alloc = compute_reduced_costs(&a, &c, &mut bm, &is_basic, n, m, &basis);

        let mut y_buf = vec![0.0f64; m];
        // Pre-fill rc_out with garbage to ensure basic-slot zeroing happens.
        let mut rc_out = vec![123.456f64; n];
        compute_reduced_costs_into(
            &a,
            &c,
            &mut bm,
            &is_basic,
            n,
            &basis,
            &mut y_buf,
            &mut rc_out,
        );
        for j in 0..n {
            assert!((rc_out[j] - r_alloc[j]).abs() < 1e-14, "j={}", j);
        }
        for j in 0..m {
            assert_eq!(rc_out[j], 0.0, "basic slot {} not zeroed", j);
        }
    }

    #[test]
    fn reduced_costs_zero_cost_yields_zero_vector() {
        let m = 3;
        let n = m + 2;
        let (a, _c, basis) = make_identity_plus(n, m);
        let c = vec![0.0f64; n];
        let is_basic: Vec<bool> = (0..n).map(|j| j < m).collect();
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        let r = compute_reduced_costs(&a, &c, &mut bm, &is_basic, n, m, &basis);
        for &rj in &r {
            assert!(rj.abs() < 1e-14, "r = {:?} should be all zero", r);
        }
    }

    /// Permuted basis: confirm `c[basis[i]]` indexing is honoured end-to-end.
    /// With B = P (a permutation of I), y^T a_{basis[i]} must equal c[basis[i]].
    #[test]
    fn dual_vars_permuted_basis_uses_basis_indexing() {
        let m = 3;
        let n = m;
        let rows: Vec<usize> = (0..m).collect();
        let cols: Vec<usize> = (0..m).collect();
        let vals: Vec<f64> = vec![1.0; m];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        let basis = vec![2usize, 0, 1];
        let c = vec![10.0, 20.0, 30.0];
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        let y = compute_dual_vars(&c, &mut bm, &basis, m);

        for i in 0..m {
            let (rs, vs) = a.get_column(basis[i]).unwrap();
            let mut dot = 0.0;
            for (k, &row) in rs.iter().enumerate() {
                dot += y[row] * vs[k];
            }
            assert!(
                (dot - c[basis[i]]).abs() < 1e-12,
                "y^T a_{{basis[{}]}} = {} expected {}",
                i,
                dot,
                c[basis[i]]
            );
        }
    }

    #[test]
    fn basic_obj_identity_basis() {
        let m = 4;
        let (_a, c, basis) = make_identity_plus(m + 2, m);
        // x_B = [1, 2, 3, 4], c[basis[i]] = c[i] = i+1 → obj = 1·1 + 2·2 + 3·3 + 4·4 = 30
        let x_b = vec![1.0, 2.0, 3.0, 4.0];
        let obj = basic_obj(&c, &basis, &x_b);
        let expected: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
        assert!((obj - expected).abs() < 1e-14);
        assert!((obj - 30.0).abs() < 1e-14);
    }

    /// basic_obj on a permuted basis must respect `basis[i]` indexing.
    #[test]
    fn basic_obj_permuted_basis() {
        let basis = vec![2usize, 0, 1];
        let c = vec![10.0, 20.0, 30.0];
        let x_b = vec![1.0, 2.0, 3.0];
        // obj = c[2]·x_b[0] + c[0]·x_b[1] + c[1]·x_b[2] = 30+20+60 = 110
        let obj = basic_obj(&c, &basis, &x_b);
        assert!((obj - 110.0).abs() < 1e-14);
    }

    /// Empty basis edge case: must return 0.0, not panic.
    #[test]
    fn basic_obj_empty_basis() {
        let c = vec![1.0, 2.0, 3.0];
        let basis: Vec<usize> = vec![];
        let x_b: Vec<f64> = vec![];
        assert_eq!(basic_obj(&c, &basis, &x_b), 0.0);
    }

    /// Negative x_B (dual simplex pre-feasible state) must accumulate signs
    /// correctly — not abs-summed by accident.
    #[test]
    fn basic_obj_negative_x_b_signs_preserved() {
        let c = vec![1.0, 2.0, 3.0];
        let basis = vec![0usize, 1, 2];
        let x_b = vec![-1.0, 2.0, -3.0];
        // obj = -1 + 4 + -9 = -6
        assert!((basic_obj(&c, &basis, &x_b) - (-6.0)).abs() < 1e-14);
    }

    /// An already-expired deadline must short-circuit the BTRAN loop immediately
    /// (checked at i=0) and return `None`.
    ///
    /// no-op proof: removing the deadline check inside the loop (reverting
    /// `recompute_gamma_truth` to a `-> Vec<f64>` that ignores `deadline`) makes
    /// this test fail — it would return `Some(...)` instead of `None`.
    #[test]
    fn dse_expired_deadline_during_gamma_loop_returns_timeout() {
        let m = 4;
        let (a, _c, basis) = make_identity_plus(m + 2, m);
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        // deadline 1 second in the past — already expired before the first BTRAN
        let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let result = recompute_gamma_truth(&mut bm, m, Some(past), None);
        assert!(
            result.is_none(),
            "expired deadline must short-circuit the BTRAN loop and return None; \
             got Some (deadline check is missing or not firing at i=0)"
        );
        // No-deadline call must always succeed.
        let result_no_dl = recompute_gamma_truth(&mut bm, m, None, None);
        assert!(
            result_no_dl.is_some(),
            "None deadline must always return Some"
        );
    }

    /// A pre-set cancel_flag must abort the BTRAN loop and return `None`.
    ///
    /// no-op proof: removing the cancel_flag check inside the loop makes this
    /// test return `Some(...)` instead of `None` → test fails.
    #[test]
    fn recompute_gamma_truth_cancellation_during_sweep() {
        use std::sync::Arc;
        let m = 4;
        let (a, _c, basis) = make_identity_plus(m + 2, m);
        let mut bm = LuBasis::new(&a, &basis, 32).unwrap();
        // Pre-set cancel flag: already true before the sweep starts.
        let flag = Arc::new(AtomicBool::new(true));
        let result = recompute_gamma_truth(&mut bm, m, None, Some(&flag));
        assert!(
            result.is_none(),
            "pre-set cancel_flag must abort the BTRAN sweep and return None; \
             got Some (cancel_flag check is missing or not firing)"
        );
        // Cleared flag: sweep must complete.
        flag.store(false, Ordering::Relaxed);
        let result_ok = recompute_gamma_truth(&mut bm, m, None, Some(&flag));
        assert!(
            result_ok.is_some(),
            "cleared cancel_flag must allow sweep to complete"
        );
        // No flag at all: must always succeed.
        let result_no_flag = recompute_gamma_truth(&mut bm, m, None, None);
        assert!(
            result_no_flag.is_some(),
            "None cancel_flag must always return Some"
        );
    }

    /// `made_progress_with_floor(_, _, 0.0)` must return true iff improvement
    /// exceeds the relative noise floor `|best| * NO_PROGRESS_REL_EPS`.
    ///
    /// no-op proof: stubbing `made_progress_with_floor` to always return `false`
    /// makes the second and third assertions fail → test fails.
    #[test]
    fn made_progress_threshold_boundary() {
        // Clear improvement far above noise floor.
        assert!(
            made_progress_with_floor(1.0, 0.0, 0.0),
            "1.0 → 0.0 is clear improvement; must return true"
        );
        // Improvement exactly at the noise floor is NOT counted as progress.
        let eps = NO_PROGRESS_REL_EPS;
        let best = 1.0_f64;
        let at_boundary = best - best.abs() * eps;
        assert!(
            !made_progress_with_floor(best, at_boundary, 0.0),
            "improvement == eps * |best| is not strictly above threshold; must return false"
        );
        // Improvement just above the noise floor IS counted.
        let above_boundary = best - best.abs() * eps - 1e-15;
        assert!(
            made_progress_with_floor(best, above_boundary, 0.0),
            "improvement slightly above eps * |best| must return true"
        );
        // No improvement at all.
        assert!(
            !made_progress_with_floor(1.0, 1.0, 0.0),
            "no improvement must return false"
        );
        // best == 0, current == 0: no improvement → false.
        assert!(
            !made_progress_with_floor(0.0, 0.0, 0.0),
            "best == 0, current == 0: no improvement must return false"
        );
    }

    /// `made_progress_with_floor`: floor=1.0 keeps near-zero `best` from treating
    /// sub-eps values as noise-free progress; floor=0.0 uses `|best|` as the scale.
    ///
    /// no-op proof: stubbing `made_progress_with_floor` to always return `false`
    /// means `best_obj` in primal Phase I never updates → `OBJ_PROGRESS_RESET_COUNT`
    /// stays 0 → `b2_obj_progress_reset_fires_on_improving_objective` FAIL.
    #[test]
    fn made_progress_with_floor_protects_near_zero_best() {
        // floor=1.0: improvement < floor*eps = 1e-12 is rejected even when best ≈ 0.
        assert!(
            !made_progress_with_floor(0.0, -0.5e-12, 1.0),
            "near-zero best, floor=1.0: improvement below floor*eps must be rejected"
        );
        // best=0, current<0 (improvement), floor=0.0:
        //   0 - (-0.5e-12) = 0.5e-12 > 0.0 * eps = 0 → true.
        //   Note: best==0 does NOT imply false when floor=0; only best==current==0 gives false.
        assert!(
            made_progress_with_floor(0.0, -0.5e-12, 0.0),
            "best=0, current=-0.5e-12, floor=0: any positive improvement must return true"
        );
        // floor=1.0 with improvement above the floor*eps threshold.
        assert!(
            made_progress_with_floor(0.0, -1.5e-12, 1.0),
            "floor=1.0, improvement > floor*eps must pass"
        );
        // floor=0.0, typical values: improvement above |best|*eps.
        assert!(
            made_progress_with_floor(1.0, 0.0, 0.0),
            "floor=0.0, best=1.0, current=0.0: clear improvement must return true"
        );
    }

    /// LOAD-BEARING sentinel for the verified-Unbounded gate. Pins both directions:
    /// a genuine recession ray must verify (else true-Unbounded → false Timeout),
    /// and a bounded basis must NOT (else an eta-drift false-Unbounded survives).
    /// Also pins the `n_enter` exclusion (artificials never enter a ray).
    ///
    /// no-op proof: a gate hard-wired to `true` fails the bounded assertion; one
    /// hard-wired to `false` fails the genuine-ray assertion.
    #[test]
    fn lp_unbounded_ray_verified_distinguishes_genuine_bounded_and_artificial() {
        let opts = crate::options::SolverOptions::default();

        // Genuine unbounded: A=[[-1, 1]], basis={col1}, c=[-1,0]. col0 improving
        // (r0=-1) with B⁻¹a_0=[-1] ≤ 0 ⇒ no leaving row ⇒ unbounded ray.
        let a_unb = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, 1.0], 1, 2).unwrap();
        assert!(
            lp_unbounded_ray_verified(&a_unb, &[1], &[-1.0, 0.0], 1, 2, 2, &opts),
            "genuine unbounded ray must verify (else true-Unbounded demoted to Timeout)"
        );

        // Bounded: A=[[1, 1]], same basis/c. col0 improving but B⁻¹a_0=[1] > 0
        // ⇒ leaving row exists ⇒ no recession ray.
        let a_bnd = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        assert!(
            !lp_unbounded_ray_verified(&a_bnd, &[1], &[-1.0, 0.0], 1, 2, 2, &opts),
            "bounded LP must NOT verify a ray (else eta-drift false-Unbounded survives)"
        );

        // n_enter exclusion: A=[[-1, 1, -1]], basis={col1}, c=[0,0,-1], n_enter=2.
        // Only col2 is an unbounded direction, but it is ≥ n_enter (artificial) ⇒
        // excluded ⇒ no verified ray.
        let a_art =
            CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[-1.0, 1.0, -1.0], 1, 3).unwrap();
        assert!(
            !lp_unbounded_ray_verified(&a_art, &[1], &[0.0, 0.0, -1.0], 1, 3, 2, &opts),
            "a column ≥ n_enter (artificial) must be excluded from the recession ray"
        );

        // CONCERN B verification: bounded direction with 0 < pivot < PIVOT_TOL must NOT verify.
        let a_smallpiv = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[5e-9, 1.0], 1, 2).unwrap();
        assert!(
            !lp_unbounded_ray_verified(&a_smallpiv, &[1], &[-1.0, 0.0], 1, 2, 2, &opts),
            "CONCERN B: a small positive pivot (0<d<PIVOT_TOL) is a leaving row ⇒ bounded, not a ray"
        );
    }

    /// LOAD-BEARING sentinel for the basic-artificial ray check. A direction that
    /// drives no structural variable below its lower bound but *increases* a basic
    /// artificial off 0 is feasible only in the augmented system, not the original
    /// LP — so it is NOT a recession ray and must be rejected.
    ///
    /// Setup (m=2, cols [struct0 | struct1 | artificial2], n_enter=2, n_price=3):
    ///   A col0=(row0,1), col1=(row1,-1), col2=(row1,1); basis={col0, col2}=I, c=[0,-1,0].
    ///   Entering col1 (rc=-1) gives d=B⁻¹a_1=[0,-1]: row0 structural d=0 ≤ floor (ok),
    ///   row1 is the basic artificial (basis[1]=2 ≥ n_enter=2) with d=-1 < 0 ⇒ the
    ///   artificial grows to +t ⇒ not an original-LP ray.
    ///
    /// no-op proof: dropping the `basis[i] ≥ n_enter ⇒ |d_i| ≤ floor` branch (i.e.
    /// reverting to the plain `d_i ≤ floor` test) admits this direction as a ray and
    /// this assertion flips to a false-Unbounded → the test fails.
    #[test]
    fn lp_unbounded_ray_verified_rejects_ray_that_increases_basic_artificial() {
        let opts = crate::options::SolverOptions::default();
        let a = CscMatrix::from_triplets(&[0, 1, 1], &[0, 1, 2], &[1.0, -1.0, 1.0], 2, 3).unwrap();
        assert!(
            !lp_unbounded_ray_verified(&a, &[0, 2], &[0.0, -1.0, 0.0], 2, 3, 2, &opts),
            "a direction that increases a basic artificial off 0 must NOT verify as a recession ray"
        );
    }
}
