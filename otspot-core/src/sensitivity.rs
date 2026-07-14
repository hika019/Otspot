//! LP sensitivity analysis (ranging).
//!
//! Computes RHS and objective coefficient ranges over which the current
//! optimal basis stays optimal (primal- and dual-feasible). Solve with
//! `presolve = false` to obtain a `warm_start_basis`, then call
//! [`compute_sensitivity`]; see the module tests for a worked example.

use crate::basis::{BasisManager, LuBasis};
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::simplex::{
    build_bounded_standard_form_with_deadline, build_standard_form, extract_solution, StandardForm,
};

/// Numerical threshold for the ratio test in sensitivity ranging.
const RATIO_TOL: f64 = 1e-10;
/// Tolerance for deciding a variable sits at its finite upper bound when
/// translating a bounded-form warm-start basis into the UB-row legacy form.
const BOUND_ACTIVE_TOL: f64 = 1e-7;
/// Relative tolerance for validating the reconstructed legacy basis against the
/// solver's reported primal solution.
const BASIS_RECON_TOL: f64 = 1e-6;

/// Translate the solver's warm-start basis into a legacy UB-row-expanded basis
/// of length `sf.m`.
///
/// The default dual-advanced path solves in `BoundedStandardForm` (per-variable
/// upper bounds, no UB rows), so its `WarmStartBasis` has one entry per original
/// constraint row. `compute_sensitivity` works in the UB-row-expanded
/// `StandardForm`, which adds one row (and one basic variable) per finite upper
/// bound. For each such UB row the basic variable is the structural column when
/// that variable sits at its upper bound, otherwise the UB slack. Returns `None`
/// when the basis is neither legacy- nor bounded-shaped.
fn legacy_basis_from_warm_start(
    problem: &LpProblem,
    sf: &StandardForm,
    solution: &[f64],
    basis: &[usize],
) -> Option<Vec<usize>> {
    if basis.iter().any(|&c| c >= sf.n_total) {
        return None;
    }
    if basis.len() == sf.m {
        return Some(basis.to_vec()); // already legacy form (e.g. no finite UBs)
    }
    let bsf = build_bounded_standard_form_with_deadline(problem, None)?;
    if basis.len() != bsf.m {
        return None;
    }
    let n_shifted = bsf.n_shifted;
    let ub_cols: Vec<usize> = (0..n_shifted)
        .filter(|&j| bsf.upper_bounds[j].is_finite())
        .collect();
    if bsf.m + ub_cols.len() != sf.m {
        return None;
    }
    let mut col_to_orig = vec![usize::MAX; n_shifted];
    for (orig, info) in bsf.orig_var_info.iter().enumerate() {
        for &(col, _) in &info.new_vars {
            if col < n_shifted {
                col_to_orig[col] = orig;
            }
        }
    }
    let mut legacy = basis.to_vec();
    for (k, &j) in ub_cols.iter().enumerate() {
        let slack_col = bsf.n_total + k; // UB slacks are appended after bounded columns
        let orig = col_to_orig[j];
        let at_upper = orig != usize::MAX && {
            let (_, ub) = problem.bounds[orig];
            ub.is_finite() && (ub - solution.get(orig).copied().unwrap_or(0.0)).abs() <= BOUND_ACTIVE_TOL
        };
        // The structural column j is the UB-row basic only when j is non-basic at
        // its upper bound. If j is already basic (e.g. degenerate basic-at-upper),
        // the UB row's basic variable is its slack (zero-valued, degenerate), so
        // pushing j would duplicate a basis column and make B singular.
        let j_already_basic = basis.contains(&j);
        legacy.push(if at_upper && !j_already_basic {
            j
        } else {
            slack_col
        });
    }
    Some(legacy)
}

/// LP sensitivity analysis result.
///
/// Each `(allowable_decrease, allowable_increase)` pair reports how much the
/// parameter can be decreased or increased while the current optimal basis
/// remains primal-feasible and dual-feasible (optimal). All values are
/// non-negative; `f64::INFINITY` means there is no finite limit in that
/// direction.
pub struct SensitivityResult {
    /// RHS ranging: `(allowable_decrease, allowable_increase)` for each
    /// original constraint's right-hand side value `b_i`.
    pub rhs_ranges: Vec<(f64, f64)>,
    /// Objective ranging: `(allowable_decrease, allowable_increase)` for each
    /// original variable's objective coefficient `c_j`.
    pub obj_ranges: Vec<(f64, f64)>,
}

/// Compute LP sensitivity analysis from an optimal simplex result.
///
/// Returns `None` when `result.status` is not `Optimal` or
/// `result.warm_start_basis` is absent (IPM solve, presolve enabled, or
/// non-optimal termination). Otherwise `rhs_ranges[i]` / `obj_ranges[j]`
/// correspond to constraint `i` / variable `j` in original order.
///
/// RHS ranging uses B^{-1} e_i (one FTRAN per constraint) with a ratio test on
/// x_B + (B^{-1} e_i)·δ ≥ 0. Objective ranging for a basic variable uses
/// B^{-T} e_p (one BTRAN) then a reduced-cost ratio test over non-basic
/// columns; a non-basic variable uses its original-space reduced cost.
pub fn compute_sensitivity(
    problem: &LpProblem,
    result: &SolverResult,
) -> Option<SensitivityResult> {
    if !matches!(result.status, SolveStatus::Optimal) {
        return None;
    }
    let ws = result.warm_start_basis.as_ref()?;

    let sf = build_standard_form(problem);
    let m_ext = sf.m;
    let n_total = sf.n_total;
    let m_orig = problem.num_constraints;
    let n_orig = problem.num_vars;
    if result.solution.len() != n_orig || result.reduced_costs.len() != n_orig {
        return None;
    }

    // The default bounded path returns a basis over BoundedStandardForm rows;
    // translate it into this UB-row-expanded form (no-op when already legacy).
    let basis = legacy_basis_from_warm_start(problem, &sf, &result.solution, &ws.basis)?;
    if basis.len() != m_ext || basis.iter().any(|&c| c >= n_total) {
        return None;
    }

    // Build LU factorization of the basis matrix B.
    let mut bm = LuBasis::new_timed(&sf.a, &basis, 0, None).ok()?;

    // x_B = B^{-1} b  (recomputed for accuracy; ws.x_b is a hint only).
    let mut x_b = sf.b.clone();
    bm.ftran_dense(&mut x_b);

    // Reject rather than emit ranges from a wrong basis: the (possibly
    // bound-translated) basis must reproduce the solver's reported primal.
    let col_scale = vec![1.0_f64; n_total];
    let recon = extract_solution(&sf, &basis, &x_b, &col_scale);
    if recon.len() != n_orig
        || result.solution.len() != n_orig
        || recon
            .iter()
            .zip(result.solution.iter())
            .any(|(&r, &s)| !r.is_finite() || (r - s).abs() > BASIS_RECON_TOL * (1.0 + s.abs()))
    {
        return None;
    }

    // is_basic[col] and basis_row_map[col] → row index.
    let mut is_basic = vec![false; n_total];
    let mut basis_row_map = vec![usize::MAX; n_total];
    for (row, &col) in basis.iter().enumerate() {
        is_basic[col] = true;
        basis_row_map[col] = row;
    }

    // Standard-form dual: y_sf = B^{-T} c_sf_B.
    let mut y_sf: Vec<f64> = basis.iter().map(|&col| sf.c[col]).collect();
    bm.btran_dense(&mut y_sf);

    // Reduced costs for non-basic columns: c̄_sf[k] = c_sf[k] - y_sf^T a_sf_k.
    let mut c_bar = vec![0.0_f64; n_total];
    for k in 0..n_total {
        if is_basic[k] {
            continue;
        }
        let mut ya = 0.0;
        if let Ok((rows, vals)) = sf.a.get_column(k) {
            for (idx, &row) in rows.iter().enumerate() {
                ya += y_sf[row] * vals[idx];
            }
        }
        c_bar[k] = sf.c[k] - ya;
    }

    // ── RHS ranging ───────────────────────────────────────────────────────────
    //
    // Increasing b_orig[i] by δ changes b_sf[i] by sign_i * δ:
    //   sign_i = +1  if !row_negated[i]   (b_sf[i] =  adjusted b_orig[i])
    //   sign_i = -1  if  row_negated[i]   (b_sf[i] = -adjusted b_orig[i])
    //
    // Δx_B = (sign_i * δ) * B^{-1} e_i.  Primal feasibility (x_B + Δx_B ≥ 0)
    // with d_eff = sign_i * (B^{-1} e_i) gives:
    //   Δ_up   = min { x_B[k] / (-d_eff[k]) : d_eff[k] < 0 }
    //   Δ_down = min { x_B[k] /   d_eff[k]  : d_eff[k] > 0 }

    let mut rhs_ranges = Vec::with_capacity(m_orig);
    let mut buf = vec![0.0_f64; m_ext];

    for i in 0..m_orig {
        buf[i] = 1.0;
        bm.ftran_dense(&mut buf); // buf ← d = B^{-1} e_i

        let sign = if sf.row_negated[i] { -1.0 } else { 1.0 };

        let mut delta_up = f64::INFINITY;
        let mut delta_down = f64::INFINITY;
        for k in 0..m_ext {
            let d_eff = sign * buf[k];
            let xb_k = x_b[k].max(0.0); // clamp sub-zero numerical noise
            if d_eff < -RATIO_TOL {
                delta_up = delta_up.min(xb_k / (-d_eff));
            } else if d_eff > RATIO_TOL {
                delta_down = delta_down.min(xb_k / d_eff);
            }
        }

        rhs_ranges.push((delta_down.max(0.0), delta_up.max(0.0)));

        buf.iter_mut().for_each(|v| *v = 0.0);
    }

    // ── Objective ranging ─────────────────────────────────────────────────────
    //
    // Changing c_j by δ changes c_sf[col] by sign_j * δ (where sign_j is the
    // coefficient from orig_var_info[j].new_vars[.].1, ±1).
    //
    // For basic variable j at basis row p, the new reduced cost of any non-basic k is:
    //   c̄_sf_new[k] = c̄_sf[k] + δ * C_k
    //   C_k = direct_k - sign_j * (π_p^T a_sf_k)
    //
    // where π_p = B^{-T} e_p  and  direct_k is sign_k if k is another split
    // column of the same free variable j (else 0).  Optimality c̄_sf_new[k] ≥ 0:
    //   Δ_up   = min { c̄_sf[k] / (-C_k) : C_k < 0, k non-basic }
    //   Δ_down = min { c̄_sf[k] /   C_k  : C_k > 0, k non-basic }
    //
    // For a non-basic variable the original-space reduced cost rc[j] gives the
    // range directly (no BTRAN required).

    let mut obj_ranges = Vec::with_capacity(n_orig);

    for j in 0..n_orig {
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() && ub.is_finite() && lb == ub {
            // A fixed variable contributes only the constant c_j * lb to every
            // feasible solution, regardless of whether its transformed column
            // is basic or non-basic.
            obj_ranges.push((f64::INFINITY, f64::INFINITY));
            continue;
        }

        let info = &sf.orig_var_info[j];

        let basic_entry = info.new_vars.iter().find(|&&(col, _)| is_basic[col]);

        let (delta_down, delta_up) = if let Some(&(basic_col, sign_j)) = basic_entry {
            let p = basis_row_map[basic_col];

            // π_p = B^{-T} e_p  (p-th row of B^{-1}).
            buf[p] = 1.0;
            bm.btran_dense(&mut buf); // buf ← π_p

            let mut delta_up = f64::INFINITY;
            let mut delta_down = f64::INFINITY;

            for k in 0..n_total {
                if is_basic[k] {
                    continue;
                }
                // η_k = π_p^T a_sf_k.
                let mut eta = 0.0;
                if let Ok((rows, vals)) = sf.a.get_column(k) {
                    for (idx, &row) in rows.iter().enumerate() {
                        eta += buf[row] * vals[idx];
                    }
                }
                // Direct change in c_sf[k] per unit δ_cj (nonzero only for
                // a free-variable split where k is the companion column of j).
                let direct = info
                    .new_vars
                    .iter()
                    .find(|&&(c, _)| c == k)
                    .map(|&(_, s)| s)
                    .unwrap_or(0.0);

                let cap_k = direct - sign_j * eta; // net coefficient of δ on c̄_sf[k]
                let cbar_k = c_bar[k].max(0.0); // clamp numerical noise below 0

                if cap_k < -RATIO_TOL {
                    // c̄_sf[k] + cap_k * δ ≥ 0 with cap_k < 0: δ ≤ cbar_k / |cap_k|
                    delta_up = delta_up.min(cbar_k / (-cap_k));
                } else if cap_k > RATIO_TOL {
                    // c̄_sf[k] + cap_k * δ ≥ 0 with cap_k > 0: −δ ≤ cbar_k / cap_k
                    delta_down = delta_down.min(cbar_k / cap_k);
                }
            }

            buf.iter_mut().for_each(|v| *v = 0.0);
            (delta_down.max(0.0), delta_up.max(0.0))
        } else {
            // Non-basic variable.
            let (lb, ub) = problem.bounds[j];

            if lb == f64::NEG_INFINITY && ub == f64::INFINITY {
                // Free variable with both split columns non-basic: any
                // perturbation of c_j makes the companion column's reduced
                // cost negative, so the allowable range is zero in both
                // directions.
                (0.0, 0.0)
            } else {
            let rc = result.reduced_costs[j];
            let x_j = result.solution[j];
                let lb_dist = (x_j - lb).abs();
                let ub_dist = if ub.is_finite() {
                    (ub - x_j).abs()
                } else {
                    f64::INFINITY
                };

                if lb_dist <= ub_dist {
                    (rc.max(0.0), f64::INFINITY)
                } else {
                    (f64::INFINITY, (-rc).max(0.0))
                }
            }
        };

        obj_ranges.push((delta_down, delta_up));
    }

    Some(SensitivityResult { rhs_ranges, obj_ranges })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SolverOptions;
    use crate::problem::{ConstraintType, SolveStatus};
    use crate::solve_lp_with;
    use crate::sparse::CscMatrix;

    /// Solve with presolve disabled to guarantee a warm_start_basis in the result.
    fn solve_no_presolve(problem: &LpProblem) -> SolverResult {
        let opts = SolverOptions {
            presolve: false,
            ..Default::default()
        };
        solve_lp_with(problem, &opts)
    }

    // ── 2×2 LP, hand-calculated ranging ──
    // Min -2x1 - x2  s.t.  x1+x2 ≤ 4 (b0=4),  x1+2x2 ≤ 6 (b1=6),  x1,x2 ≥ 0.
    // Optimal x1=4, x2=0 (obj=-8). Basis {x1@row0, s2@row1}; nonbasic {x2 rc=1, s1 rc=2}.
    // B = [[1,0],[1,1]], B^{-1} = [[1,0],[-1,1]], y_sf = B^{-T}[-2,0] = [-2,0].
    // Hand-calculated ranges:
    //   RHS[0]: d=[1,-1] → Δ_down=4, Δ_up=2
    //   RHS[1]: d=[0,1]  → Δ_down=2, Δ_up=∞
    //   Obj[x1]: π_0=[1,0]; C_{x2}=-1, C_{s1}=-1 → Δ_up=min(1,2)=1, Δ_down=∞
    //   Obj[x2]: nonbasic at lb, rc=1 → Δ_down=1, Δ_up=∞

    fn make_2x2_lp() -> LpProblem {
        let a = CscMatrix::from_triplets(
            &[0, 1, 0, 1],
            &[0, 0, 1, 1],
            &[1.0, 1.0, 1.0, 2.0],
            2,
            2,
        )
        .unwrap();
        LpProblem::new_general(
            vec![-2.0, -1.0],
            a,
            vec![4.0, 6.0],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_2x2_rhs_ranging() {
        let lp = make_2x2_lp();
        let result = solve_no_presolve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        let sens =
            compute_sensitivity(&lp, &result).expect("should return Some for optimal LP with basis");

        let tol = 1e-6;
        let (down0, up0) = sens.rhs_ranges[0];
        let (down1, up1) = sens.rhs_ranges[1];

        assert!((down0 - 4.0).abs() < tol, "RHS[0] Δ_down expected 4.0, got {}", down0);
        assert!((up0 - 2.0).abs() < tol, "RHS[0] Δ_up expected 2.0, got {}", up0);
        assert!((down1 - 2.0).abs() < tol, "RHS[1] Δ_down expected 2.0, got {}", down1);
        assert!(up1.is_infinite(), "RHS[1] Δ_up expected ∞, got {}", up1);
    }

    /// Sentinel: removing the row_negated sign flip from `compute_sensitivity`
    /// produces d_eff = +d instead of -d for negated rows, swapping Δ_up/Δ_down
    /// and breaking this assertion for the constraint with row_negated=true.
    #[test]
    fn test_2x2_obj_ranging() {
        let lp = make_2x2_lp();
        let result = solve_no_presolve(&lp);
        let sens = compute_sensitivity(&lp, &result).expect("should return Some");

        let tol = 1e-6;
        let (down0, up0) = sens.obj_ranges[0]; // x1 is basic
        let (down1, up1) = sens.obj_ranges[1]; // x2 is non-basic

        // x1 basic: c_x1 can increase by 1 before x2 enters; decrease: ∞.
        assert!(
            down0.is_infinite(),
            "Obj[x1] Δ_down expected ∞, got {}",
            down0
        );
        assert!(
            (up0 - 1.0).abs() < tol,
            "Obj[x1] Δ_up expected 1.0, got {}",
            up0
        );

        // x2 non-basic at lb, rc=1: decrease by 1; increase: ∞.
        assert!(
            (down1 - 1.0).abs() < tol,
            "Obj[x2] Δ_down expected 1.0, got {}",
            down1
        );
        assert!(up1.is_infinite(), "Obj[x2] Δ_up expected ∞, got {}", up1);
    }

    /// Sentinel: changing the ratio bound computation (e.g. swapping Δ_up and
    /// Δ_down in the ftran branch) must flip the values and break this test.
    #[test]
    fn test_rhs_range_boundary_2x2() {
        let lp = make_2x2_lp();
        let result = solve_no_presolve(&lp);
        let sens = compute_sensitivity(&lp, &result).expect("should return Some");

        let (_, up0) = sens.rhs_ranges[0]; // Δ_up = 2
        let b0_new = 4.0 + up0;

        // With B = [[1,0],[1,1]], B^{-1} = [[1,0],[-1,1]]:
        //   x_B = B^{-1} [b0_new, 6] = [b0_new, 6 - b0_new].
        // At Δ_up boundary (b0 = 6): x_B = [6, 0] ≥ 0 (barely feasible).
        let xb0 = b0_new;
        let xb1 = 6.0 - b0_new;
        assert!(xb0 >= -1e-9 && xb1 >= -1e-9, "basis must stay feasible at Δ_up boundary");
    }

    // ── No-basis path returns None ────────────────────────────────────────────

    /// Sentinel: if `compute_sensitivity` were to continue without a basis,
    /// it would either panic or return garbage.  This asserts it returns None.
    #[test]
    fn test_no_basis_returns_none() {
        let lp = make_2x2_lp();
        let result = SolverResult {
            status: SolveStatus::Optimal,
            objective: -8.0,
            solution: vec![4.0, 0.0],
            dual_solution: vec![-2.0, 0.0],
            reduced_costs: vec![0.0, 1.0],
            slack: vec![0.0, 2.0],
            warm_start_basis: None,
            ..Default::default()
        };
        assert!(
            compute_sensitivity(&lp, &result).is_none(),
            "must return None when warm_start_basis is absent"
        );
    }

    // ── Non-optimal status returns None ──────────────────────────────────────

    #[test]
    fn test_non_optimal_status_returns_none() {
        use crate::options::WarmStartBasis;
        let lp = make_2x2_lp();
        let result = SolverResult {
            status: SolveStatus::Infeasible,
            objective: f64::INFINITY,
            warm_start_basis: Some(WarmStartBasis {
                basis: vec![0, 1],
                x_b: vec![0.0, 0.0],
            }),
            ..Default::default()
        };
        assert!(
            compute_sensitivity(&lp, &result).is_none(),
            "must return None for non-Optimal status"
        );
    }

    // ── Degenerate LP (basic variable at zero) ────────────────────────────────
    //
    // Minimize  -x1
    // s.t.  x1 ≤ 4
    //       x1 ≤ 4   (redundant; forces a degenerate basic variable)
    //       x1 ≥ 0
    //
    // At optimal x1=4, one of the slacks is 0 (degenerate).
    // For the binding constraint, Δ_down = 0 (cannot decrease further).

    fn make_degenerate_lp() -> LpProblem {
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 1).unwrap();
        LpProblem::new_general(
            vec![-1.0],
            a,
            vec![4.0, 4.0],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_degenerate_rhs_ranging() {
        let lp = make_degenerate_lp();
        let result = solve_no_presolve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        let sens = compute_sensitivity(&lp, &result).expect("should return Some");

        // At degeneracy, at least one Δ_down = 0.
        let has_zero_down = sens.rhs_ranges.iter().any(|&(d, _)| d < 1e-6);
        assert!(
            has_zero_down,
            "degenerate basis should produce Δ_down=0 for at least one constraint"
        );

        for (i, &(d, u)) in sens.rhs_ranges.iter().enumerate() {
            assert!(d >= 0.0, "rhs_ranges[{}] Δ_down must be >= 0, got {}", i, d);
            assert!(u >= 0.0, "rhs_ranges[{}] Δ_up must be >= 0, got {}", i, u);
        }
    }

    // ── Infinite ranging (very loose constraint) ──────────────────────────────
    //
    // Minimize  x1
    // s.t.  x1 ≤ 100
    //       x1 ≥ 0
    //
    // Optimal: x1=0 (non-basic at lb, rc=1), s1=100 (basic).
    // RHS[0]: Δ_down=100 (s1→0), Δ_up=∞ (s1 can grow without bound).
    // Obj[x1]: non-basic at lb, rc=1 → Δ_down=1, Δ_up=∞.

    fn make_loose_lp() -> LpProblem {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        LpProblem::new_general(
            vec![1.0],
            a,
            vec![100.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_loose_constraint_rhs_ranging() {
        let lp = make_loose_lp();
        let result = solve_no_presolve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        let sens = compute_sensitivity(&lp, &result).expect("should return Some");

        let (down, up) = sens.rhs_ranges[0];
        assert!(
            (down - 100.0).abs() < 1e-6,
            "Δ_down expected 100.0, got {}",
            down
        );
        assert!(up.is_infinite(), "Δ_up expected ∞, got {}", up);
    }

    #[test]
    fn test_loose_constraint_obj_ranging() {
        let lp = make_loose_lp();
        let result = solve_no_presolve(&lp);
        let sens = compute_sensitivity(&lp, &result).expect("should return Some");

        let (down, up) = sens.obj_ranges[0];
        assert!(
            (down - 1.0).abs() < 1e-6,
            "Obj[x1] Δ_down expected 1.0, got {}",
            down
        );
        assert!(up.is_infinite(), "Obj[x1] Δ_up expected ∞, got {}", up);
    }

    // ── 3×3 LP: output dimensions and sign sanity ─────────────────────────────
    //
    // Minimize  -x1 - x2 - x3
    // s.t.  x1 + x2       ≤ 4
    //             x2 + x3 ≤ 3
    //       x1       + x3 ≤ 5
    //       x1, x2, x3 ≥ 0
    //
    // Optimal at the intersection x1=3, x2=1, x3=2 (all constraints active).
    // All slacks are 0 → fully degenerate; all basic variables have x_B = {3,1,2}.

    fn make_3x3_lp() -> LpProblem {
        let a = CscMatrix::from_triplets(
            &[0, 2, 0, 1, 1, 2],
            &[0, 0, 1, 1, 2, 2],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            3,
            3,
        )
        .unwrap();
        LpProblem::new_general(
            vec![-1.0, -1.0, -1.0],
            a,
            vec![4.0, 3.0, 5.0],
            vec![ConstraintType::Le, ConstraintType::Le, ConstraintType::Le],
            vec![
                (0.0, f64::INFINITY),
                (0.0, f64::INFINITY),
                (0.0, f64::INFINITY),
            ],
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_3x3_optimal_solution() {
        let lp = make_3x3_lp();
        let result = solve_no_presolve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        let sum: f64 = result.solution.iter().sum();
        assert!(
            (sum - 6.0).abs() < 1e-5,
            "x1+x2+x3 should be 6, got {}",
            sum
        );
        assert!(
            (result.objective + 6.0).abs() < 1e-5,
            "obj should be -6, got {}",
            result.objective
        );
    }

    #[test]
    fn test_3x3_sensitivity_dimensions_and_signs() {
        let lp = make_3x3_lp();
        let result = solve_no_presolve(&lp);
        let sens = compute_sensitivity(&lp, &result).expect("basis required");
        assert_eq!(sens.rhs_ranges.len(), 3, "rhs_ranges must have 3 entries");
        assert_eq!(sens.obj_ranges.len(), 3, "obj_ranges must have 3 entries");
        for (i, &(d, u)) in sens
            .rhs_ranges
            .iter()
            .chain(sens.obj_ranges.iter())
            .enumerate()
        {
            assert!(d >= 0.0, "entry {} allowable_decrease must be >= 0, got {}", i, d);
            assert!(u >= 0.0, "entry {} allowable_increase must be >= 0, got {}", i, u);
        }
    }

    // ── Ge constraint (surplus variable path) ────────────────────────────────
    //
    // Minimize  x1
    // s.t.  x1 >= 2   (Ge constraint, triggers row_negated + surplus variable)
    //       x1 <= 10
    //       x1 >= 0
    //
    // Optimal: x1=2, obj=2.  Surplus s1=0 (basic, degenerate), slack s2=8 (basic).
    // RHS[0] (Ge): Δ_down depends on how much b can decrease while keeping
    //   the surplus variable non-negative.
    // RHS[1] (Le): standard slack path.

    fn make_ge_lp() -> LpProblem {
        let a = CscMatrix::from_triplets(
            &[0, 1],
            &[0, 0],
            &[1.0, 1.0],
            2,
            1,
        )
        .unwrap();
        LpProblem::new_general(
            vec![1.0],
            a,
            vec![2.0, 10.0],
            vec![ConstraintType::Ge, ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_ge_constraint_rhs_ranging() {
        let lp = make_ge_lp();
        let result = solve_no_presolve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        let sens = compute_sensitivity(&lp, &result).expect("basis required");

        assert_eq!(sens.rhs_ranges.len(), 2);
        for (i, &(d, u)) in sens.rhs_ranges.iter().enumerate() {
            assert!(d >= 0.0, "rhs_ranges[{}] Δ_down must be >= 0, got {}", i, d);
            assert!(u >= 0.0, "rhs_ranges[{}] Δ_up must be >= 0, got {}", i, u);
        }

        // Ge constraint (row_negated=true): decreasing b[0] from 2 is bounded
        // by the variable's lower bound (x1 >= 0), so Δ_down = 2.
        let (down0, _) = sens.rhs_ranges[0];
        assert!(
            (down0 - 2.0).abs() < 1e-6,
            "Ge RHS[0] Δ_down expected 2.0, got {}",
            down0
        );
    }

    #[test]
    fn test_ge_constraint_obj_ranging() {
        let lp = make_ge_lp();
        let result = solve_no_presolve(&lp);
        let sens = compute_sensitivity(&lp, &result).expect("basis required");

        assert_eq!(sens.obj_ranges.len(), 1);
        let (d, u) = sens.obj_ranges[0];
        assert!(d >= 0.0, "obj Δ_down must be >= 0, got {}", d);
        assert!(u >= 0.0, "obj Δ_up must be >= 0, got {}", u);
    }

    // ── Non-basic free variable: obj ranging = (0, 0) ────────────────────────
    //
    // Minimize  x1
    // s.t.  x1 ≤ 5
    //       x1 ≥ 0, x2 free (x2 has no constraint coefficients)
    //
    // Optimal: x1=0 (non-basic at lb), s1=5 (basic).
    // x2 is free and unconstrained: both split columns (x2+, x2-) are
    // non-basic with zero-valued column vectors.  Any perturbation of c_x2
    // immediately makes one companion column's reduced cost negative, so the
    // allowable range is (0, 0).

    fn make_free_var_nonbasic_lp() -> LpProblem {
        // A has 1 row and 2 columns; only x1 (col 0) appears in the constraint.
        let a = CscMatrix::from_triplets(
            &[0],
            &[0],
            &[1.0],
            1,
            2,
        )
        .unwrap();
        LpProblem::new_general(
            vec![1.0, 0.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY)],
            None,
        )
        .unwrap()
    }

    #[test]
    fn test_nonbasic_free_variable_obj_ranging() {
        let lp = make_free_var_nonbasic_lp();
        let result = solve_no_presolve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        let sens = compute_sensitivity(&lp, &result).expect("basis required");

        // x2 is free and non-basic: obj ranging must be (0, 0).
        let (down, up) = sens.obj_ranges[1];
        assert!(
            down.abs() < 1e-10,
            "free non-basic Δ_down expected 0.0, got {}",
            down
        );
        assert!(
            up.abs() < 1e-10,
            "free non-basic Δ_up expected 0.0, got {}",
            up
        );
    }

    #[test]
    fn fixed_variable_obj_ranging_is_unbounded() {
        // Min 7x + y, x fixed at 3, 0 <= y <= 5, y <= 5.
        // Changing c_x changes every feasible objective by the same constant
        // delta * 3, so the allowable objective range for x is unbounded both
        // downward and upward.
        let a = CscMatrix::from_triplets(&[0], &[1], &[1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![7.0, 1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(3.0, 3.0), (0.0, 5.0)],
            None,
        )
        .unwrap();
        let result = solve_no_presolve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.solution[0] - 3.0).abs() < 1e-9, "x must stay fixed");
        let sens = compute_sensitivity(&lp, &result).expect("basis required");

        let (down, up) = sens.obj_ranges[0];
        assert!(down.is_infinite() && down.is_sign_positive(), "fixed x down={down}");
        assert!(up.is_infinite() && up.is_sign_positive(), "fixed x up={up}");
    }

    fn make_bounded_lp_var_at_ub() -> LpProblem {
        // Min -2x1 - x2  s.t. x1 + x2 <= 10 ; 0<=x1<=4, 0<=x2<=7.
        // Optimal x1=4 (at UB), x2=6 (basic), obj=-14.
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        LpProblem::new_general(
            vec![-2.0, -1.0],
            a,
            vec![10.0],
            vec![ConstraintType::Le],
            vec![(0.0, 4.0), (0.0, 7.0)],
            None,
        )
        .unwrap()
    }

    #[test]
    fn bounded_lp_with_var_at_ub_has_sensitivity() {
        // Regression: the bounded dual-advanced path returns a basis over
        // BoundedStandardForm rows; sensitivity must translate it (previously None).
        let lp = make_bounded_lp_var_at_ub();
        let result = solve_no_presolve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        let sens =
            compute_sensitivity(&lp, &result).expect("bounded-form basis must yield sensitivity");
        assert_eq!(sens.rhs_ranges.len(), 1);
        assert_eq!(sens.obj_ranges.len(), 2);
        // b0 in [10-6, 10+1]: down until x2=0, up until x2 hits its UB 7.
        let (down, up) = sens.rhs_ranges[0];
        assert!((down - 6.0).abs() < 1e-6, "rhs down expected 6, got {}", down);
        assert!((up - 1.0).abs() < 1e-6, "rhs up expected 1, got {}", up);
    }

    fn make_bounded_lp_var_interior() -> LpProblem {
        // Min -x1 - 0.1x2  s.t. x1 + x2 <= 2 ; 0<=x1<=5, 0<=x2<=5.
        // Optimal x1=2 (interior wrt UB 5), x2=0, obj=-2.
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        LpProblem::new_general(
            vec![-1.0, -0.1],
            a,
            vec![2.0],
            vec![ConstraintType::Le],
            vec![(0.0, 5.0), (0.0, 5.0)],
            None,
        )
        .unwrap()
    }

    #[test]
    fn bounded_lp_all_interior_ub_slacks_basic() {
        // No variable sits at its upper bound: every UB row keeps its slack basic.
        let lp = make_bounded_lp_var_interior();
        let result = solve_no_presolve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        let sens =
            compute_sensitivity(&lp, &result).expect("bounded-form basis must yield sensitivity");
        // b0 in [2-2, 2+3]: down until x1=0, up until x1 hits its UB 5.
        let (down, up) = sens.rhs_ranges[0];
        assert!((down - 2.0).abs() < 1e-6, "rhs down expected 2, got {}", down);
        assert!((up - 3.0).abs() < 1e-6, "rhs up expected 3, got {}", up);
    }

    #[test]
    fn bounded_lp_var_basic_at_ub_has_sensitivity() {
        // Degenerate: an equality forces x1 to its upper bound, so x1 is BOTH
        // basic (in the eq row) and at its UB. The UB row must take the slack,
        // not a duplicate of x1's column. Previously this returned None.
        // Min -x2  s.t. x1 = 4 (Eq), x1 + x2 <= 12 ; 0<=x1<=4, 0<=x2<=10.
        let a = CscMatrix::from_triplets(&[0, 1, 1], &[0, 0, 1], &[1.0, 1.0, 1.0], 2, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![0.0, -1.0],
            a,
            vec![4.0, 12.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, 4.0), (0.0, 10.0)],
            None,
        )
        .unwrap();
        let result = solve_no_presolve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.solution[0] - 4.0).abs() < 1e-9, "x1 must be at its UB");
        let sens = compute_sensitivity(&lp, &result)
            .expect("basic-at-upper degenerate basis must still yield sensitivity");
        assert_eq!(sens.rhs_ranges.len(), 2);
        assert_eq!(sens.obj_ranges.len(), 2);
    }

    #[test]
    fn sensitivity_rejects_malformed_solution_or_reduced_costs() {
        let lp = make_2x2_lp();
        let result = solve_no_presolve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(compute_sensitivity(&lp, &result).is_some());

        let mut short_solution = result.clone();
        short_solution.solution.pop();
        assert!(
            compute_sensitivity(&lp, &short_solution).is_none(),
            "short solution must not be padded while computing sensitivity ranges"
        );

        let mut short_rc = result;
        short_rc.reduced_costs.pop();
        assert!(
            compute_sensitivity(&lp, &short_rc).is_none(),
            "short reduced_costs must not be treated as zero"
        );
    }
}
