//! Pricing strategies for the Revised Simplex method
//!
//! Provides a trait and two implementations:
//! - `DantzigPricing`: classic most-negative-reduced-cost rule (test-only)
//! - `SteepestEdgePricing`: Devex approximate steepest-edge pricing

use crate::basis::LuBasis;
use crate::tolerances::PIVOT_TOL;

const EPS: f64 = 1e-8;

/// Minimum weight floor to keep `sqrt(γ)` safe (prevents div-by-zero in score).
pub(crate) const GAMMA_FLOOR: f64 = 1e-10;
/// Cap for the pivot² Devex update. Normal non-degenerate weights are O(m);
/// capping prevents tiny pivots from permanently distorting pricing.
pub(crate) const CAP_MULT_OF_M: f64 = 100.0;

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
    /// `leaving_row` = row index of the leaving basic variable, so pivot =
    ///                 `eta[leaving_row]`.
    /// `eta`         = B⁻¹ * a_entering (FTRAN of entering column, dense).
    fn update_weights(
        &mut self,
        basis: &LuBasis,
        entering: usize,
        leaving: usize,
        leaving_row: usize,
        eta: &[f64],
    );

    /// Reset all weights to 1.0 (cycle-breaking anti-degeneracy reset).
    fn reset_weights(&mut self, n_vars: usize);
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
    fn reset_weights(&mut self, _n_vars: usize) {}
}

/// Devex approximate steepest-edge pricing (Harris 1973 / Price 1987).
///
/// Maintains γ[j] ≈ ‖B⁻¹ a_j‖² for each non-basic column j.
/// Selects the entering variable maximising `-rc_j / sqrt(γ_j)`.
///
/// Weight update after a pivot (pivot² formula with per-column cap):
///   raw          = ‖η‖² / pivot², pivot = η[leaving_row]
///   γ[leaving]  = max(γ[leaving], min(raw, 100·m))
///   γ[entering] = 1.0
///   γ[other j]  unchanged (per-column FTRAN avoided)
///
/// The pivot guard compares `pivot.abs()` before squaring. A squared guard at
/// `1e-16` admits boundary pivots around 1e-8 due to f64 rounding and can
/// produce weights around 1e17; the cap keeps those cases from dominating
/// the rest of the solve while preserving normal O(m)..O(100m) weights.
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

    /// Devex weight update — pivot² formula with a per-column cap at 100·m.
    ///
    ///   raw          = ‖η‖² / pivot², pivot = η[leaving_row]
    ///   γ[leaving]  = max(γ[leaving], min(raw, 100·m))
    ///   γ[entering] = 1.0
    ///   γ[other j]  unchanged (per-column FTRAN avoided)
    ///
    /// When |pivot| ≤ PIVOT_TOL, the weight is not updated (degenerate pivot
    /// guard). The cap prevents tiny-pivot weight blow-up from permanently
    /// distorting pricing.
    fn update_weights(
        &mut self,
        _basis: &LuBasis,
        entering: usize,
        leaving: usize,
        leaving_row: usize,
        eta: &[f64],
    ) {
        if leaving < self.weights.len() {
            let pivot = eta.get(leaving_row).copied().unwrap_or(0.0);
            if pivot.abs() > PIVOT_TOL {
                let eta_norm_sq: f64 = eta.iter().map(|&x| x * x).sum();
                let cap = CAP_MULT_OF_M * (eta.len() as f64);
                let raw = eta_norm_sq / (pivot * pivot);
                let new_weight = raw.min(cap).max(GAMMA_FLOOR);
                self.weights[leaving] = self.weights[leaving].max(new_weight);
            }
        }

        if entering < self.weights.len() {
            // After leaving the basis, penalize re-entry: set entering weight
            // to max(1, gamma_leaving / eta_norm_sq) so a column with high
            // leaving weight gets higher entering weight → lower selection score
            // → implicit anti-cycling for recently-left columns (old formula
            // property retained). Columns not recently leaving start at 1.0.
            let eta_norm_sq: f64 = eta.iter().map(|&x| x * x).sum();
            let gamma_leaving = if leaving < self.weights.len() {
                self.weights[leaving]
            } else {
                1.0
            };
            let new_entering_w = if eta_norm_sq > GAMMA_FLOOR {
                (gamma_leaving / eta_norm_sq).max(1.0)
            } else {
                1.0
            };
            self.weights[entering] = new_entering_w;
        }
    }

    fn reset_weights(&mut self, n_vars: usize) {
        if self.weights.len() != n_vars {
            self.weights = vec![1.0; n_vars];
        } else {
            self.weights.fill(1.0);
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
    /// bland_mode. Default = pure Bland on the leaving row: select the
    /// smallest row index with `x_B[i] < -primal_tol`. Strategies with
    /// auxiliary objectives (e.g. artificial removal in Big-M Phase I) must
    /// override so bland_mode does not mask their secondary priority.
    fn bland_leaving(&mut self, x_b: &[f64], primal_tol: f64, basis: &[usize]) -> Option<usize> {
        // Bland's leaving rule: select the basic variable with the smallest
        // column index (basis[i]) among those with x_B[i] < -primal_tol.
        // Row-index selection breaks the anti-cycling guarantee.
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
    fn after_pivot(&mut self, _leaving_row: usize, _alpha: &[f64], _sigma: &[f64], _pivot: f64) {}

    /// Called after the basis is refactored. Stateful weights (DSE) keep the
    /// rank-1-updated values across refactor (Forrest-Goldfarb 1992); identity
    /// reset only fires on CEILING-flagged drift or size mismatch. Stateless
    /// strategies leave the no-op.
    fn after_refactor(&mut self, _m: usize) {}

    /// Initialise γ_i = ||(B^{-1})_{i,:}||² for arbitrary starting basis.
    /// `gamma_truth[i]` is supplied by the core loop after m BTRANs. DSE
    /// overrides; stateless strategies ignore. The default no-op means the
    /// core loop may skip the (O(m²)) BTRAN sweep for stateless callers.
    fn set_initial_gamma(&mut self, _gamma_truth: &[f64]) {}

    /// Whether the core may flip `trow` signs to repair an lb-violation
    /// (x_B[r] < 0) at a warm-start leaving row.  Default `true`: standard
    /// MI/DSE strategies allow lb-repair pivots. Big-M Phase I overrides to
    /// `false`: lb-violations there arise from LU eta drift in natural rows,
    /// not genuine warm-start infeasibilities, and the sign flip routes the
    /// ratio test into a direction that returns no candidates → false Unbounded.
    fn allows_lb_repair(&self) -> bool {
        true
    }
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
        let a =
            crate::sparse::CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
        crate::basis::LuBasis::new(&a, &[0, 1], 50).unwrap()
    }

    /// Sentinel: `update_weights` uses the pivot² Sherman-Morrison form.
    ///
    /// With η = [0, 3, 4, 0, 0] (‖η‖² = 25), pivot = η[1] = 3:
    ///   pivot² formula: 25 / 9 ≈ 2.778
    ///
    /// A no-op implementation would leave γ[leaving] = 1.0 < 2.778 and fail.
    #[test]
    fn devex_leaving_weight_uses_pivot_squared() {
        let mut pricing = SteepestEdgePricing::new(5);
        let eta = vec![0.0, 3.0, 4.0, 0.0, 0.0]; // ‖η‖² = 25
        let expected = 25.0_f64 / 9.0_f64; // pivot = eta[1] = 3

        let basis_id = make_identity_basis_2x2();
        pricing.update_weights(&basis_id, 2, 3, 1, &eta);

        assert!(
            (pricing.weights[3] - expected).abs() < 1e-12,
            "expected γ[leaving] = {:.6} (‖η‖²/pivot²), got {:.6}",
            expected,
            pricing.weights[3]
        );
        // With γ[leaving=3]=2.778 and ‖η‖²=25: entering weight = max(1, 2.778/25)
        // = max(1, 0.111) = 1.0 (no penalty since leaving weight < eta_norm_sq).
        assert_eq!(
            pricing.weights[2], 1.0,
            "entering weight = 1.0 when γ[leaving] < ‖η‖² (no cycle-penalty case)"
        );
    }

    /// Sentinel: tiny pivots below `PIVOT_TOL` do not update the leaving weight.
    ///
    /// Other column weights must remain untouched (no global reset side-effect).
    #[test]
    fn devex_small_pivot_guard_skips_update() {
        let mut pricing = SteepestEdgePricing::new(5);
        pricing.weights[0] = 3.0;
        pricing.weights[1] = 5.0;
        let eta = vec![0.0, 1e-9_f64, 4.0, 0.0, 0.0];

        let basis_id = make_identity_basis_2x2();
        pricing.update_weights(&basis_id, 2, 3, 1, &eta);

        assert_eq!(pricing.weights[3], 1.0, "small pivot: weight must remain 1.0");
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

    /// Sentinel: entering weight is penalized proportional to γ[leaving] / ‖η‖².
    ///
    /// This gives anti-cycling memory: columns that just LEFT with a high weight
    /// (were hard to displace) are penalized on re-entry, reducing the chance of
    /// re-entering the same column in a short cycle.
    ///
    /// With η = [0, 2e-8, 4, 0, 0] (cap fires), γ[leaving=3] = 500 (cap),
    /// ‖η‖² ≈ 16: entering weight = max(1, 500/16) ≈ 31.25 > 1.0 (penalized).
    ///
    /// no-op proof: if entering weight is always 1.0, assert(31.25 > 1.1) fails.
    #[test]
    fn devex_entering_weight_penalized_when_leaving_large() {
        let mut pricing = SteepestEdgePricing::new(5);
        let eta = vec![0.0, 2e-8_f64, 4.0, 0.0, 0.0]; // cap fires for leaving
        let basis_id = make_identity_basis_2x2();
        pricing.update_weights(&basis_id, 2, 3, 1, &eta);

        // entering=2 should be penalized (weight > 1)
        assert!(
            pricing.weights[2] > 1.1,
            "entering weight should be > 1 when γ[leaving] >> ‖η‖², got {:.6}",
            pricing.weights[2]
        );
    }

    /// Sentinel: the per-column cap clamps pivot² blow-up at 100·m.
    #[test]
    fn devex_pivot_squared_update_is_capped() {
        let mut pricing = SteepestEdgePricing::new(5);
        let eta = vec![0.0, 2e-8_f64, 4.0, 0.0, 0.0];
        let basis_id = make_identity_basis_2x2();
        pricing.update_weights(&basis_id, 2, 3, 1, &eta);

        let cap = CAP_MULT_OF_M * eta.len() as f64;
        assert!(
            (pricing.weights[3] - cap).abs() <= 1e-9,
            "expected capped γ[leaving]={cap:.3e}, got {:.3e}",
            pricing.weights[3]
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

    /// Sentinel: `bland_leaving` selects the row whose *column index* (basis[i])
    /// is smallest among all infeasible rows — Bland's rule, not row-index order.
    ///
    /// basis = [5, 1, 3], x_b = [-1.0, -2.0, -0.5]:
    ///   row 0: basis=5, infeasible (-1.0)
    ///   row 1: basis=1, infeasible (-2.0)  ← smallest column index
    ///   row 2: basis=3, infeasible (-0.5)
    /// Bland selects row 1 (column 1 is smallest). Row-index selection would
    /// pick row 0 (first infeasible), which breaks the anti-cycling guarantee.
    #[test]
    fn bland_leaving_uses_smallest_column_index() {
        let mut strat = MostInfeasibleLeaving;
        let x_b = vec![-1.0_f64, -2.0, -0.5];
        let basis = vec![5_usize, 1, 3];
        let pick = strat.bland_leaving(&x_b, 1e-8, &basis);
        assert_eq!(
            pick,
            Some(1),
            "Bland leaving must select row with smallest basis[i]=1 (col 1), not row 0 (col 5)"
        );
    }
}
