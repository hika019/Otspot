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

use crate::basis::{BasisManager, LuBasis};
use crate::options::WarmStartBasis;
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::sparse::CscMatrix;
use super::{extract_dual_info, extract_solution, SimplexOutcome, StandardForm};

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
        a, c, basis_mgr, is_basic, n_price, basis, &mut y, &mut reduced_costs,
    );
    reduced_costs
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
            let ws = WarmStartBasis { basis: basis.to_vec(), x_b: x_b.to_vec() };
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
                    objective: 0.0,
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

/// c_B^T x_B = Σ c[basis[i]] · x_B[i]. Shared between primal/dual simplex for
/// objective reporting on Optimal / Timeout / SingularBasis exits.
pub(super) fn basic_obj(c: &[f64], basis: &[usize], x_b: &[f64]) -> f64 {
    debug_assert_eq!(basis.len(), x_b.len());
    basis
        .iter()
        .zip(x_b.iter())
        .map(|(&j, &v)| c[j] * v)
        .sum()
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
            assert!((y[i] - c[i]).abs() < 1e-12, "y[{}] = {} expected {}", i, y[i], c[i]);
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
        for slot in y_into.iter_mut() { *slot = -42.0; }
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
            assert!((r[j] - expected).abs() < 1e-12, "r[{}] = {} expected {}", j, r[j], expected);
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
        compute_reduced_costs_into(&a, &c, &mut bm, &is_basic, n, &basis, &mut y_buf, &mut rc_out);
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
            assert!((dot - c[basis[i]]).abs() < 1e-12,
                "y^T a_{{basis[{}]}} = {} expected {}", i, dot, c[basis[i]]);
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
}
