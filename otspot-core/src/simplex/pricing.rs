//! Pricing strategies for the Revised Simplex method
//!
//! Provides a trait and two implementations:
//! - `DantzigPricing`: classic most-negative-reduced-cost rule (test-only)
//! - `SteepestEdgePricing`: Devex approximate steepest-edge pricing

use crate::basis::LuBasis;

const EPS: f64 = 1e-8;

/// Minimum weight floor to keep `sqrt(γ)` safe (prevents div-by-zero in score).
const GAMMA_FLOOR: f64 = 1e-10;

/// Strategy for selecting the entering variable in the revised simplex.
pub(crate) trait PricingStrategy {
    /// Select the entering column index.
    ///
    /// `reduced_costs` contains pre-computed reduced costs for columns 0..n_basic.
    /// Entries for basic variables should be set to 0.0 by the caller.
    /// Returns `None` if no improving column exists (optimality).
    fn select_entering(&self, reduced_costs: &[f64], n_basic: usize) -> Option<usize>;

    /// Update internal weights after a pivot.
    ///
    /// `entering` = column index entering the basis.
    /// `leaving`  = column index leaving the basis (not the row index).
    /// `eta`      = B⁻¹ * a_entering (FTRAN of entering column, dense).
    fn update_weights(
        &mut self,
        basis: &LuBasis,
        entering: usize,
        leaving: usize,
        eta: &[f64],
    );
}

/// Classic Dantzig pricing: select the column with the most negative reduced cost.
///
/// Reference implementation retained for pricing strategy comparison tests.
/// Production uses `SteepestEdgePricing`.
#[cfg(test)]
pub(crate) struct DantzigPricing;

#[cfg(test)]
impl PricingStrategy for DantzigPricing {
    fn select_entering(&self, reduced_costs: &[f64], n_basic: usize) -> Option<usize> {
        let limit = n_basic.min(reduced_costs.len());
        let mut min_rc = -EPS;
        let mut entering = None;
        for (j, &rc) in reduced_costs.iter().enumerate().take(limit) {
            if rc < min_rc {
                min_rc = rc;
                entering = Some(j);
            }
        }
        entering
    }

    fn update_weights(&mut self, _: &LuBasis, _: usize, _: usize, _: &[f64]) {}
}

/// Devex approximate steepest-edge pricing (Harris 1973 / Price 1987).
///
/// Maintains γ[j] ≈ ‖B⁻¹ a_j‖² for each non-basic column j.
/// Selects the entering variable maximising `-rc_j / sqrt(γ_j)`.
///
/// Weight update after a pivot:
///   γ[leaving] ← max(γ[leaving], ‖η‖² / γ[entering])
///   γ[entering] ← 1.0   (reset; entering becomes basic)
///   γ[other j]  unchanged (approximation; exact update requires per-column FTRAN)
///
/// The numerically exact leaving-column weight is `‖η‖²/pivot²` (Sherman-Morrison
/// update of B⁻¹).  Multiple variants of that formula produced bench regressions
/// when tested (see retreat history below), so we use `γ[entering]` as the
/// denominator.  When |pivot| is small, γ[entering] tends to be large (the
/// entering column had high steepest-edge weight), providing implicit damping
/// that keeps γ[leaving] finite without any explicit cap.
///
/// **Retreat history — pivot² formula variants:**
///
/// **#75 (`‖η‖²/pivot²`, unguarded):** wood1p NumericalError. Mechanism (#178
/// verified 2026-05-30): pricing distortion → false-Optimal at infeasible vertex
/// → `check_eq_feasibility` FAIL (wood1p iter 3106, 676 γ-blowup, γ=6.6e17);
/// Phase 2 LU instability (maros). Prior claim "permanent column exclusion →
/// SingularBasis" was incorrect.
///
/// **#146 (pivot² with DEVEX_WEIGHT_CAP, three variants):**
///
/// Attempt 1 (per-column cap at 1·m): wood1p 14s → 2.7s, but maros (m=846)
/// DFEAS_FAIL.  Normal pivots produce weights in [m, 100·m]; capping at m
/// distorts pricing for those problems.
///
/// Attempt 2 (per-column cap at 100·m): wood1p PASS, grow22 PASS. Retreat
/// claim "maros FAIL:Infeasible (0.7s)" **does not reproduce in current codebase**
/// (post-2026-05-30, #178 Agent C verified). cap-100m is strictly better in
/// tested 3: wood1p 17.6× faster, grow22 1.66× faster, maros parity.  Phase 1 /
/// dual_advanced / postsolve / KKT guard improvements since #146 absorbed the
/// perturbation.  Full Netlib 109 + Maros 138 validation pending (#182).
///
/// Attempt 3 (global weight reset when any weight > 100·m): maros PASS, but
/// grow22 PFEAS_FAIL.  A full reset wipes pricing history for all columns,
/// causing Dantzig-like selection mid-solve that reaches a different optimal
/// vertex and exposes a postsolve bound-check failure.
///
/// **#165 (per-row Charnes perturbation with DEGENERATE_ROW_THRESHOLD gate):**
/// grow22 PFEAS_FAIL. Mechanism (#178 verified 2026-05-30, Agent B): bfeas=1.957e-3,
/// x_b_neg=4 (basis truly primal-infeasible). ε addition skews ratio test →
/// ineligible leaving row selected → reconciled basis primal-infeasible. Algorithm
/// invalid (Scenario D pure); retreat confirmed. 事実化済 (#178 検証, Agent B,
/// 2026-05-30).
///
/// **Future pivot² guard:** A guard `pivot_sq > 1e-16` is f64-boundary weak:
/// `(1e-8)² = 1.0000000000000001e-16 > 1e-16` passes the guard. wood1p
/// observation: col=77, pivot=1.48e-8 → γ=2.19e16 (blowup directly above
/// guard). Stronger guard: `pivot.abs() > PIVOT_TOL` (compare before squaring).
///
/// γ[entering] formula confirmed: 109/109 PASS, eps=1e-6, timeout=1000s.
/// Retreat decisions can become stale as the codebase evolves; re-evaluate
/// pivot² variants against current main before dismissing.
///
/// Weights start at 1.0 and are non-decreasing per column (the `max` prevents
/// reducing a weight). Columns that never leave the basis retain γ = 1.0 and
/// are scored as `-rc_j` (Dantzig-equivalent for those columns).
pub(crate) struct SteepestEdgePricing {
    weights: Vec<f64>,
}

impl SteepestEdgePricing {
    pub fn new(n_vars: usize) -> Self {
        Self {
            weights: vec![1.0; n_vars],
        }
    }

    /// Reset all weights to 1.0. Exposed for tests only.
    #[cfg(test)]
    pub(crate) fn reset_weights(&mut self, n_vars: usize) {
        if self.weights.len() != n_vars {
            self.weights = vec![1.0; n_vars];
        } else {
            self.weights.fill(1.0);
        }
    }
}

impl PricingStrategy for SteepestEdgePricing {
    fn select_entering(&self, reduced_costs: &[f64], n_basic: usize) -> Option<usize> {
        let limit = n_basic.min(reduced_costs.len());
        let mut best_score = -EPS;
        let mut entering = None;

        for (j, &rc) in reduced_costs.iter().enumerate().take(limit) {
            if rc < -EPS {
                let gamma = self.weights.get(j).copied().unwrap_or(1.0).max(GAMMA_FLOOR);
                let score = -rc / gamma.sqrt();
                if score > best_score {
                    best_score = score;
                    entering = Some(j);
                }
            }
        }
        entering
    }

    /// Devex weight update using γ[entering] as denominator.
    ///
    /// γ[leaving] ← max(γ[leaving], ‖η‖² / γ[entering])
    /// γ[entering] ← 1.0
    ///
    /// See struct docstring for the retreat history documenting why the
    /// numerically exact pivot² formula was abandoned.
    fn update_weights(
        &mut self,
        _basis: &LuBasis,
        entering: usize,
        leaving: usize,
        eta: &[f64],
    ) {
        if leaving < self.weights.len() {
            let eta_norm_sq: f64 = eta.iter().map(|&x| x * x).sum();
            let gamma_e = self.weights.get(entering).copied().unwrap_or(1.0).max(GAMMA_FLOOR);
            let new_weight = (eta_norm_sq / gamma_e).max(GAMMA_FLOOR);
            self.weights[leaving] = self.weights[leaving].max(new_weight);
        }

        if entering < self.weights.len() {
            self.weights[entering] = 1.0;
        }
    }
}

/// Dual Simplex leaving-variable selection.
///
/// `basis` is the current basic-column index per row. Standard `MostInfeasibleLeaving`
/// ignores it; Big-M Phase I (`dual_advanced/phase1.rs`) uses it to detect artificials
/// still in the basis. Stateful strategies (DSE) use `&mut self` + hooks to maintain
/// per-iter weights; stateless strategies leave the hooks as no-ops.
pub(crate) trait DualLeavingStrategy {
    /// Row index of a basic variable to leave (None = primal feasible).
    fn select_leaving(&mut self, x_b: &[f64], primal_tol: f64, basis: &[usize]) -> Option<usize>;

    /// Anti-cycling fallback used when `dual_simplex_core_advanced` enters
    /// bland_mode. Default = pure Bland (smallest `basis[i]` index among rows
    /// with `x_B[i] < -primal_tol`). Strategies with auxiliary objectives
    /// (e.g. artificial removal in Big-M Phase I) must override so bland_mode
    /// does not mask their secondary priority.
    fn bland_leaving(&mut self, x_b: &[f64], primal_tol: f64, basis: &[usize]) -> Option<usize> {
        let mut best_row: Option<usize> = None;
        let mut best_var = usize::MAX;
        for (i, &v) in x_b.iter().enumerate() {
            if v < -primal_tol && basis[i] < best_var {
                best_var = basis[i];
                best_row = Some(i);
            }
        }
        best_row
    }

    /// Progress metric (smaller = closer to goal). bland_mode triggers when
    /// this fails to improve for `k_trigger` consecutive iterations. Default
    /// = sum of negative parts of x_B (standard "primal infeasibility").
    /// Strategies that drive auxiliary quantities (e.g. artificial values)
    /// to zero must include those in the metric; otherwise Big-M Phase I
    /// entry condition `x_B = b ≥ 0` makes `best = 0` and threshold = 0 →
    /// every iter judged no-progress → bland_mode 誤起動.
    fn progress_metric(&mut self, x_b: &[f64], _basis: &[usize]) -> f64 {
        x_b.iter().map(|&v| (-v).max(0.0)).sum()
    }

    /// Whether the strategy needs σ = B^{-1} ρ_p passed to `after_pivot`.
    /// Stateless strategies return false → core skips the extra FTRAN.
    /// DSE returns true.
    fn needs_sigma(&self) -> bool {
        false
    }

    /// Post-pivot hook. Stateful weight updates (DSE) wire this; stateless
    /// strategies leave the default no-op. `sigma` is only valid when
    /// `needs_sigma()` returned true.
    fn after_pivot(
        &mut self,
        _leaving_row: usize,
        _alpha: &[f64],
        _sigma: &[f64],
        _pivot: f64,
    ) {
    }

    /// Called after the basis is refactored. Stateful weights that may drift
    /// (DSE) reset here to identity; stateless strategies leave the no-op.
    fn after_refactor(&mut self, _m: usize) {}

    /// Initialise γ_i = ||(B^{-1})_{i,:}||² for arbitrary starting basis.
    /// `gamma_truth[i]` is supplied by the core loop after m BTRANs. DSE
    /// overrides; stateless strategies ignore. The default no-op means the
    /// core loop may skip the (O(m²)) BTRAN sweep for stateless callers.
    fn set_initial_gamma(&mut self, _gamma_truth: &[f64]) {}
}

/// Most Infeasible Rule: 最も負のx_B[i]を選択
pub(crate) struct MostInfeasibleLeaving;

impl DualLeavingStrategy for MostInfeasibleLeaving {
    fn select_leaving(&mut self, x_b: &[f64], primal_tol: f64, _basis: &[usize]) -> Option<usize> {
        let mut best_row = None;
        let mut max_violation = primal_tol;

        for (i, &val) in x_b.iter().enumerate() {
            if val < -max_violation {
                max_violation = -val;
                best_row = Some(i);
            }
        }
        best_row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dantzig_selects_most_negative() {
        let pricing = DantzigPricing;
        let rc = vec![0.0, -0.5, -1.0, 0.2, -0.3];
        let entering = pricing.select_entering(&rc, rc.len());
        assert_eq!(entering, Some(2)); // index 2 has rc = -1.0 (most negative)
    }

    #[test]
    fn test_dantzig_returns_none_when_optimal() {
        let pricing = DantzigPricing;
        let rc = vec![0.0, 0.1, 0.5, 0.0];
        assert_eq!(pricing.select_entering(&rc, rc.len()), None);
    }

    #[test]
    fn test_dantzig_respects_n_basic_limit() {
        let pricing = DantzigPricing;
        let rc = vec![-0.1, -0.5, -1.0]; // most negative is index 2
        // Only look at first 2
        let entering = pricing.select_entering(&rc, 2);
        assert_eq!(entering, Some(1)); // index 1 within limit
    }

    #[test]
    fn test_steepest_edge_scoring() {
        let pricing = SteepestEdgePricing {
            weights: vec![1.0, 4.0, 1.0], // γ[1] = 4 → score divides by 2
        };
        // rc = [-1.0, -2.0, -0.5]
        // scores: 1.0/1.0=1.0, 2.0/2.0=1.0, 0.5/1.0=0.5
        // tie: both index 0 and 1 have score 1.0 → first wins
        let rc = vec![-1.0, -2.0, -0.5];
        let entering = pricing.select_entering(&rc, rc.len());
        // Index 0: score = 1.0/sqrt(1.0) = 1.0
        // Index 1: score = 2.0/sqrt(4.0) = 1.0 (tie, first found wins)
        // Index 2: score = 0.5/sqrt(1.0) = 0.5
        assert!(entering == Some(0) || entering == Some(1));
    }

    #[test]
    fn test_steepest_edge_prefers_better_score() {
        let pricing2 = SteepestEdgePricing {
            weights: vec![1.0, 100.0, 1.0],
        };
        let rc = vec![-1.0, -10.0, -0.5];
        // Index 0: 1.0/1.0 = 1.0
        // Index 1: 10.0/10.0 = 1.0 (tie)
        // Index 2: 0.5/1.0 = 0.5
        let entering = pricing2.select_entering(&rc, rc.len());
        assert!(entering == Some(0) || entering == Some(1));
    }

    fn make_identity_basis_2x2() -> crate::basis::LuBasis {
        let a = crate::sparse::CscMatrix::from_triplets(
            &[0, 1], &[0, 1], &[1.0, 1.0], 2, 2,
        ).unwrap();
        crate::basis::LuBasis::new(&a, &[0, 1], 50).unwrap()
    }

    /// Sentinel: `update_weights` uses γ[entering] in the denominator (not pivot²).
    ///
    /// Formula: γ[leaving] ← max(γ[leaving], ‖η‖² / γ[entering])
    ///
    /// With η = [0, 3, 4, 0, 0] (‖η‖² = 25), γ[entering] = 7.0:
    ///   γ[entering] formula: 25 / 7 ≈ 3.571
    ///
    /// A no-op implementation (e.g. returning early without updating) would leave
    /// γ[leaving] = 1.0, which is strictly less than 3.571 and fails the assertion.
    #[test]
    fn devex_leaving_weight_uses_gamma_entering() {
        let mut pricing = SteepestEdgePricing::new(5);
        pricing.weights[2] = 7.0; // γ[entering] = 7.0
        let eta = vec![0.0, 3.0, 4.0, 0.0, 0.0]; // ‖η‖² = 25
        let expected = 25.0_f64 / 7.0_f64; // γ[entering] formula

        let basis_id = make_identity_basis_2x2();
        pricing.update_weights(&basis_id, 2, 3, &eta);

        assert!(
            (pricing.weights[3] - expected).abs() < 1e-12,
            "expected γ[leaving] = {:.6} (‖η‖²/γ[entering]), got {:.6}",
            expected,
            pricing.weights[3]
        );
        assert_eq!(pricing.weights[2], 1.0, "entering column weight must reset to 1");
    }

    /// Sentinel: degenerate-pivot scenario produces a finite γ[leaving] (γ[entering] damps).
    ///
    /// With η = [0, 0.01, 4, 0, 0] (‖η‖² ≈ 16.0001), γ[entering] = 2.0:
    ///   γ[entering] formula: 16.0001 / 2.0 ≈ 8.0  (finite, no cap needed)
    ///   For contrast: pivot² with pivot=1e-5 gives 16.0001 / 1e-10 ≈ 1.6e11  (blow-up)
    ///
    /// Other column weights must remain untouched (no global reset side-effect).
    #[test]
    fn devex_degenerate_pivot_no_blowup() {
        let mut pricing = SteepestEdgePricing::new(5);
        pricing.weights[0] = 3.0;
        pricing.weights[1] = 5.0;
        pricing.weights[2] = 2.0; // γ[entering] = 2.0
        let eta = vec![0.0, 0.01_f64, 4.0, 0.0, 0.0]; // ‖η‖² ≈ 16.0001
        let eta_norm_sq: f64 = eta.iter().map(|&x| x * x).sum();
        let expected = eta_norm_sq / 2.0; // γ[entering]=2.0

        let basis_id = make_identity_basis_2x2();
        pricing.update_weights(&basis_id, 2, 3, &eta);

        assert!(
            (pricing.weights[3] - expected).abs() < 1e-10,
            "degenerate pivot: expected γ[leaving]={:.6} (finite), got {:.6}",
            expected,
            pricing.weights[3]
        );
        assert!(
            (pricing.weights[0] - 3.0).abs() < 1e-15,
            "γ[0] must stay at 3.0 (no global reset), got {:.6}",
            pricing.weights[0]
        );
        assert!(
            (pricing.weights[1] - 5.0).abs() < 1e-15,
            "γ[1] must stay at 5.0 (no global reset), got {:.6}",
            pricing.weights[1]
        );
    }

    /// Sentinel: `reset_weights` clears all weights to 1.0.
    ///
    /// Removing the body of `reset_weights` leaves stale weights, failing this test.
    #[test]
    fn reset_weights_resets_all_to_one() {
        let mut pricing = SteepestEdgePricing::new(4);
        pricing.weights[0] = 5.0;
        pricing.weights[2] = 7.3;
        pricing.reset_weights(4);
        assert!(
            pricing.weights.iter().all(|&w| (w - 1.0).abs() < 1e-15),
            "reset_weights must clear all weights to 1.0, got {:?}",
            pricing.weights
        );
    }

    #[test]
    fn reset_weights_resizes_if_n_changes() {
        let mut pricing = SteepestEdgePricing::new(3);
        pricing.weights[1] = 9.9;
        pricing.reset_weights(5); // new size
        assert_eq!(pricing.weights.len(), 5);
        assert!(pricing.weights.iter().all(|&w| (w - 1.0).abs() < 1e-15));
    }
}
