//! Pricing strategies for the Revised Simplex method
//!
//! Provides a trait and two implementations:
//! - `DantzigPricing`: classic most-negative-reduced-cost rule (test-only)
//! - `SteepestEdgePricing`: Devex approximate steepest-edge pricing

use crate::basis::LuBasis;

const EPS: f64 = 1e-8;

/// Minimum weight floor, preventing division blow-up and keeping sqrt safe.
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
    /// `entering`     = column index entering the basis.
    /// `leaving`      = column index leaving the basis.
    /// `leaving_row`  = row index of the leaving variable.
    /// `eta`          = B⁻¹ * a_entering (FTRAN of entering column, dense, length m).
    fn update_weights(
        &mut self,
        basis: &LuBasis,
        entering: usize,
        leaving: usize,
        leaving_row: usize,
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

    fn update_weights(&mut self, _: &LuBasis, _: usize, _: usize, _: usize, _: &[f64]) {}
}

/// Devex approximate steepest-edge pricing (Harris 1973 / Price 1987).
///
/// Maintains γ[j] ≈ ‖B⁻¹ a_j‖² for each non-basic column j.
/// Selects the entering variable maximising `-rc_j / sqrt(γ_j)`.
///
/// Weight update after a pivot with pivot column η = B⁻¹ a_entering and
/// pivot element p = η[leaving_row]:
///   γ[leaving] ← max(γ[leaving], ‖η‖² / p²)
///   γ[entering] ← 1.0   (reset; entering becomes basic)
///   γ[other j]  unchanged (approximation; exact update requires per-column FTRAN)
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

    /// Reset all weights to 1.0 (e.g. after LU refactor to wipe drift).
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

    /// Devex weight update for the leaving column.
    ///
    /// γ[leaving] ← max(γ[leaving], ‖η‖² / η[leaving_row]²)
    ///
    /// `‖η‖² / pivot²` is the Devex approximation to `‖B_new^{-1} a_leaving‖²`.
    /// Taking the max keeps weights non-decreasing, preventing score inflation
    /// from a temporarily small pivot.
    fn update_weights(
        &mut self,
        _basis: &LuBasis,
        entering: usize,
        leaving: usize,
        leaving_row: usize,
        eta: &[f64],
    ) {
        let pivot = eta.get(leaving_row).copied().unwrap_or(0.0);
        let pivot_sq = pivot * pivot;

        if leaving < self.weights.len() && pivot_sq > GAMMA_FLOOR {
            let eta_norm_sq: f64 = eta.iter().map(|&x| x * x).sum();
            let new_weight = (eta_norm_sq / pivot_sq).max(GAMMA_FLOOR);
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

    /// Sentinel: `update_weights` uses pivot² (not γ[entering]) in the denominator.
    ///
    /// Old formula: γ[leaving] ← max(γ[leaving], ‖η‖² / γ[entering])
    /// New formula: γ[leaving] ← max(γ[leaving], ‖η‖² / η[leaving_row]²)
    ///
    /// With γ[entering]=1.0 and pivot=3, these differ: old gives 25, new gives 25/9.
    /// Removing the leaving_row parameter or reusing γ[entering] fails this test.
    #[test]
    fn devex_leaving_weight_uses_pivot_squared() {
        let mut pricing = SteepestEdgePricing::new(5);
        // η = [0, 3, 4, 0, 0], leaving_row=1 → pivot=3, pivot²=9.
        // ‖η‖² = 9 + 16 = 25.
        // Expected: γ[leaving=3] = max(1.0, 25/9) = 25/9 ≈ 2.778.
        // Old formula: max(1.0, 25/γ[entering=1.0]) = 25.
        let eta = vec![0.0, 3.0, 4.0, 0.0, 0.0];

        let a_id = crate::sparse::CscMatrix::from_triplets(
            &[0, 1], &[0, 1], &[1.0, 1.0], 2, 2,
        ).unwrap();
        let basis_id = crate::basis::LuBasis::new(&a_id, &[0, 1], 50).unwrap();

        pricing.update_weights(
            &basis_id,
            2, // entering
            3, // leaving col
            1, // leaving_row (pivot = η[1] = 3.0)
            &eta,
        );
        let expected = 25.0_f64 / 9.0_f64;
        assert!(
            (pricing.weights[3] - expected).abs() < 1e-12,
            "expected γ[leaving] = {:.6}, got {:.6}",
            expected,
            pricing.weights[3]
        );
        assert_eq!(pricing.weights[2], 1.0, "entering column weight must reset to 1");
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
