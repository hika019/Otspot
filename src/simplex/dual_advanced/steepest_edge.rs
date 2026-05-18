//! Dual Steepest Edge (DSE) weight management.
//!
//! γ_i = ||(B^{-1})_{i,:}||² for each basis row i. Selecting the leaving row
//! that maximises x_B[i]² / γ_i (over rows with x_B[i] < -tol) gives the
//! "steepest" descent step in dual space (Forrest-Goldfarb 1992).
//!
//! Update after pivot at (row p, column q):
//!   α = B^{-1} a_q                      (already computed for the pivot)
//!   ρ_p = B^{-T} e_p                    (already computed for PRICE)
//!   σ = B^{-1} ρ_p                      (one extra FTRAN per iter)
//!   γ_p ← γ_p / α_p²
//!   γ_i ← γ_i − 2 (α_i/α_p) σ_i + (α_i/α_p)² γ_p   for i ≠ p
//!
//! Identity for σ: <(B^{-1})_{i,:}, (B^{-1})_{p,:}> = (B^{-1} ρ_p)_i = σ_i,
//! so the update is exact (not an approximation).

use super::super::pricing::DualLeavingStrategy;

/// Lower clamp on γ_i. Prevents division blow-up in the score
/// `x_B[i]² / γ_i` and keeps sqrt(γ_i) numerically safe.
/// 1e-10 is well above f64 noise (~1e-16) and below any legitimate
/// row-norm value for a non-degenerate basis (= 1.0 at identity).
const DSE_GAMMA_FLOOR: f64 = 1e-10;

/// Upper clamp on γ_i. If the rank-1 update ever produces γ_i above this,
/// drift has accumulated past trustworthy precision; the weight is clamped
/// and `needs_reset` is flagged so the next refactor resets to identity.
/// 1e10 = 10 orders above the natural unit scale, generous enough not to
/// fire on legitimately large-norm rows but tight enough to catch runaway.
const DSE_GAMMA_CEILING: f64 = 1e10;

/// Stateful DSE weight vector.
///
/// On `new(m)` initialises γ_i = 1 (correct for the identity basis used at
/// dual-simplex cold start with slack basis). After every pivot the caller
/// invokes `update_after_pivot`. On full refactor the caller invokes
/// `reset_to_identity` (γ drift wipe-out, cheaper than re-running m BTRANs).
pub(crate) struct DseWeights {
    gamma: Vec<f64>,
    /// Set by `update_after_pivot` when clamp ceiling fires; the next refactor
    /// hook honours it via `reset_to_identity`. Independent flag rather than
    /// auto-reset because resets happen at refactor boundaries (tied to LU
    /// rebuild) for amortised cost.
    needs_reset: bool,
}

impl DseWeights {
    pub(crate) fn new(m: usize) -> Self {
        Self {
            gamma: vec![1.0; m],
            needs_reset: false,
        }
    }

    pub(crate) fn gamma(&self, i: usize) -> f64 {
        self.gamma[i].max(DSE_GAMMA_FLOOR)
    }

    pub(crate) fn reset_to_identity(&mut self) {
        self.gamma.fill(1.0);
        self.needs_reset = false;
    }

    /// Apply the Forrest-Goldfarb rank-1 update.
    ///
    /// `alpha` = α dense (length m), `sigma` = σ = B^{-1} ρ_p dense (length m),
    /// `pivot` = α[leaving_row]. Caller must guarantee `pivot.abs() >= PIVOT_TOL`.
    pub(crate) fn update_after_pivot(
        &mut self,
        leaving_row: usize,
        alpha: &[f64],
        sigma: &[f64],
        pivot: f64,
    ) {
        let inv_p = 1.0 / pivot;
        let gamma_p = self.gamma[leaving_row];
        for i in 0..self.gamma.len() {
            if i == leaving_row {
                continue;
            }
            let ratio = alpha[i] * inv_p;
            let new_gamma = self.gamma[i] - 2.0 * ratio * sigma[i] + ratio * ratio * gamma_p;
            self.gamma[i] = clamp_gamma(new_gamma, &mut self.needs_reset);
        }
        let new_gamma_p = gamma_p * inv_p * inv_p;
        self.gamma[leaving_row] = clamp_gamma(new_gamma_p, &mut self.needs_reset);
    }
}

fn clamp_gamma(v: f64, needs_reset: &mut bool) -> f64 {
    if !v.is_finite() || v > DSE_GAMMA_CEILING {
        *needs_reset = true;
        DSE_GAMMA_CEILING.min(1.0_f64.max(v.abs()))
    } else if v < DSE_GAMMA_FLOOR {
        DSE_GAMMA_FLOOR
    } else {
        v
    }
}

/// Env switch that freezes the γ update to all-1 for the no-op proof
/// sentinel. With γ_i ≡ 1 the DSE score reduces to x_B[i]² → row selection
/// is identical to `MostInfeasibleLeaving`, so the sentinel's "DSE faster"
/// assertion *must* fail when this flag is set (memory:
/// feedback_sentinel_must_fail_under_noop).
const DSE_DISABLE_ENV: &str = "DSE_DISABLE_GAMMA_UPDATE";

fn gamma_update_disabled() -> bool {
    std::env::var(DSE_DISABLE_ENV)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// `DualLeavingStrategy` impl that uses DSE-scored leaving selection.
///
/// Score for row i (with x_B[i] < -primal_tol): x_B[i]² / γ_i.
/// `select_leaving` returns argmax. Stateless strategies short-circuit
/// `after_pivot` / `after_refactor` to no-ops; DSE wires both.
pub(crate) struct DualSteepestEdgeLeaving {
    weights: DseWeights,
}

impl DualSteepestEdgeLeaving {
    pub(crate) fn new(m: usize) -> Self {
        Self {
            weights: DseWeights::new(m),
        }
    }
}

impl DualLeavingStrategy for DualSteepestEdgeLeaving {
    fn select_leaving(&mut self, x_b: &[f64], primal_tol: f64, _basis: &[usize]) -> Option<usize> {
        let mut best_row: Option<usize> = None;
        let mut best_score = 0.0_f64;
        for (i, &val) in x_b.iter().enumerate() {
            if val < -primal_tol {
                let score = val * val / self.weights.gamma(i);
                if score > best_score {
                    best_score = score;
                    best_row = Some(i);
                }
            }
        }
        best_row
    }

    fn needs_sigma(&self) -> bool {
        true
    }

    fn after_pivot(&mut self, leaving_row: usize, alpha: &[f64], sigma: &[f64], pivot: f64) {
        if gamma_update_disabled() {
            return;
        }
        self.weights.update_after_pivot(leaving_row, alpha, sigma, pivot);
    }

    fn after_refactor(&mut self, m: usize) {
        // Refactor wipes accumulated eta drift; γ accumulates similarly so
        // we reset to identity for stability. The DSE literature accepts
        // periodic resets — between resets the rank-1 update is exact.
        if self.weights.gamma.len() != m {
            self.weights = DseWeights::new(m);
        } else if self.weights.needs_reset {
            self.weights.reset_to_identity();
        }
    }

    fn set_initial_gamma(&mut self, gamma_truth: &[f64]) {
        if gamma_update_disabled() {
            // No-op proof: γ stays at identity so score = x_B[i]² and DSE
            // collapses to MostInfeasible (same row choice). The proof would
            // leak if we accepted the BTRAN-derived truth here.
            if self.weights.gamma.len() != gamma_truth.len() {
                self.weights = DseWeights::new(gamma_truth.len());
            }
            self.weights.reset_to_identity();
            return;
        }
        if self.weights.gamma.len() != gamma_truth.len() {
            self.weights = DseWeights::new(gamma_truth.len());
        }
        for (slot, &v) in self.weights.gamma.iter_mut().zip(gamma_truth.iter()) {
            *slot = v.max(DSE_GAMMA_FLOOR);
        }
        self.weights.needs_reset = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_initialised_to_identity() {
        let w = DseWeights::new(4);
        for i in 0..4 {
            assert_eq!(w.gamma(i), 1.0);
        }
    }

    #[test]
    fn update_pivot_row_inverse_square() {
        // pivot = 2 → γ_p_new = 1 / 4 = 0.25, then floor doesn't apply.
        let mut w = DseWeights::new(3);
        w.update_after_pivot(1, &[0.0, 2.0, 0.0], &[0.0, 1.0, 0.0], 2.0);
        assert!((w.gamma(1) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn update_other_rows_apply_rank1_correction() {
        // Setup: m=2, γ = [1, 1], leaving=1, α=[3, 4], σ=[5, 7], pivot=4.
        // ratio_0 = 3/4 = 0.75.
        // γ_0_new = 1 - 2*0.75*5 + 0.75² * 1 = 1 - 7.5 + 0.5625 = -5.9375 → clamp.
        let mut w = DseWeights::new(2);
        w.update_after_pivot(1, &[3.0, 4.0], &[5.0, 7.0], 4.0);
        assert!(w.gamma(0) >= DSE_GAMMA_FLOOR, "γ floored");
        assert!((w.gamma(1) - (1.0_f64 / 16.0)).abs() < 1e-12, "pivot row 1/α²");
    }

    #[test]
    fn dse_strategy_selects_max_score() {
        let mut s = DualSteepestEdgeLeaving::new(3);
        // γ = [1, 1, 1] → score = x_B[i]². x_B = [-2, -3, -1] → row 1 wins.
        let pick = s.select_leaving(&[-2.0, -3.0, -1.0], 1e-9, &[0, 1, 2]);
        assert_eq!(pick, Some(1));
    }

    #[test]
    fn dse_strategy_respects_gamma_weighting() {
        // x_B = [-2, -3], γ = [1, 100]. Score_0 = 4, score_1 = 9/100 = 0.09 → row 0 wins.
        let mut s = DualSteepestEdgeLeaving::new(2);
        s.weights.gamma[1] = 100.0;
        let pick = s.select_leaving(&[-2.0, -3.0], 1e-9, &[0, 1]);
        assert_eq!(pick, Some(0));
    }

    #[test]
    fn dse_returns_none_when_primal_feasible() {
        let mut s = DualSteepestEdgeLeaving::new(3);
        let pick = s.select_leaving(&[0.5, 1.0, 0.0], 1e-9, &[0, 1, 2]);
        assert_eq!(pick, None);
    }

    #[test]
    fn ceiling_clamp_marks_for_reset() {
        let mut w = DseWeights::new(2);
        // Force a huge update: pivot = 1e-8, γ_p_new = 1 / 1e-16 = 1e16 → clamp ceiling.
        w.update_after_pivot(0, &[1e-8, 0.0], &[0.0, 0.0], 1e-8);
        assert!(w.needs_reset, "ceiling clamp must flag reset");
    }
}
