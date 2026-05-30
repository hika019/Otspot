//! Proof-carrying certificate types. [`OptimalCertificate`] / [`BoundGapCertificate`]
//! は内部ファクトリ経由でのみ生成され、保持自体が proof token になる。

/// KKT optimality certificate — minted only by [`crate::qp::certificate::prove_optimal`].
#[derive(Debug, Clone)]
pub struct OptimalCertificate {
    stationarity_rel: f64,
    primal_residual_rel: f64,
    dual_sign_violation: f64,
    duality_gap_rel: f64,
    tol: f64,
}

impl OptimalCertificate {
    pub(crate) fn new(
        stationarity_rel: f64,
        primal_residual_rel: f64,
        dual_sign_violation: f64,
        duality_gap_rel: f64,
        tol: f64,
    ) -> Self {
        Self {
            stationarity_rel,
            primal_residual_rel,
            dual_sign_violation,
            duality_gap_rel,
            tol,
        }
    }

    /// Componentwise relative stationarity: `max_j |Qx+c+Aᵀy+z|_j / scale_j`.
    pub fn stationarity_rel(&self) -> f64 {
        self.stationarity_rel
    }

    /// Componentwise relative primal violation: `max_i viol_i / scale_i`.
    pub fn primal_residual_rel(&self) -> f64 {
        self.primal_residual_rel
    }

    /// Dual-sign violation: `max_k viol_k / (1 + |v_k|)` over sign-constrained duals.
    pub fn dual_sign_violation(&self) -> f64 {
        self.dual_sign_violation
    }

    /// Relative duality gap: `|p_obj − d_obj| / max(|p|, |d|, 1)`.
    pub fn duality_gap_rel(&self) -> f64 {
        self.duality_gap_rel
    }

    /// Tolerance used when the certificate was issued.
    pub fn tol(&self) -> f64 {
        self.tol
    }
}

/// Reason why `prove_optimal` failed: observed residuals + failing condition names.
#[derive(Debug, Clone)]
pub struct NotProven {
    pub stationarity_rel: f64,
    pub primal_residual_rel: f64,
    pub bound_violation: f64,
    pub complementarity_rel: f64,
    pub dual_sign_violation: f64,
    pub duality_gap_rel: f64,
    pub tol: f64,
    pub failing_conditions: Vec<&'static str>,
}

/// B&B gap certificate — minted by the B&B driver when no region was abandoned
/// (`proof_uncertain == false`) and `gap_rel ≤ gap_tol`. External code cannot
/// construct it.
#[derive(Debug, Clone)]
pub struct BoundGapCertificate {
    incumbent_obj: f64,
    lower_bound: f64,
    gap_rel: f64,
    gap_tol: f64,
}

impl BoundGapCertificate {
    pub(crate) fn new(incumbent_obj: f64, lower_bound: f64, gap_rel: f64, gap_tol: f64) -> Self {
        Self {
            incumbent_obj,
            lower_bound,
            gap_rel,
            gap_tol,
        }
    }

    /// Best integer-feasible objective found by the search.
    pub fn incumbent_obj(&self) -> f64 {
        self.incumbent_obj
    }

    /// Authenticated lower bound on the global optimum at termination.
    pub fn lower_bound(&self) -> f64 {
        self.lower_bound
    }

    /// Relative gap: `(incumbent_obj - lower_bound) / max(1, |incumbent_obj|)`.
    pub fn gap_rel(&self) -> f64 {
        self.gap_rel
    }

    /// Tolerance against which the gap was checked when the certificate was issued.
    pub fn gap_tol(&self) -> f64 {
        self.gap_tol
    }
}
