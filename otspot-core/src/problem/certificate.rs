//! Proof-carrying certificate types for solver outcomes.
//!
//! An [`OptimalCertificate`] can only be obtained by running [`prove_optimal`]
//! (defined in `crate::qp::certificate`), which verifies all KKT conditions.
//! Holding a certificate is a proof token: it is impossible to construct one
//! without passing the full verifier.

/// KKT optimality certificate — all fields private, minted only by [`prove_optimal`].
///
/// Construction is only possible via the `pub(crate) new` method called from
/// `crate::qp::certificate::prove_optimal`. External code cannot build this
/// struct directly, eliminating the "proof-less Optimal" anti-pattern.
#[derive(Debug, Clone)]
pub struct OptimalCertificate {
    stationarity_rel: f64,
    primal_residual_rel: f64,
    bound_violation: f64,
    complementarity_rel: f64,
    dual_sign_violation: f64,
    duality_gap_rel: f64,
    tol: f64,
}

impl OptimalCertificate {
    /// Internal constructor — only `crate::qp::certificate::prove_optimal` calls this.
    pub(crate) fn new(
        stationarity_rel: f64,
        primal_residual_rel: f64,
        bound_violation: f64,
        complementarity_rel: f64,
        dual_sign_violation: f64,
        duality_gap_rel: f64,
        tol: f64,
    ) -> Self {
        Self {
            stationarity_rel,
            primal_residual_rel,
            bound_violation,
            complementarity_rel,
            dual_sign_violation,
            duality_gap_rel,
            tol,
        }
    }

    /// Componentwise relative stationarity: `max_j |Qx+c+Aᵀy+z|_j / scale_j`.
    pub fn stationarity_rel(&self) -> f64 { self.stationarity_rel }

    /// Componentwise relative primal violation: `max_i viol_i / scale_i`.
    pub fn primal_residual_rel(&self) -> f64 { self.primal_residual_rel }

    /// Primal bound violation: `max_j max(lb_j−x_j, x_j−ub_j, 0) / scale_j`.
    pub fn bound_violation(&self) -> f64 { self.bound_violation }

    /// Complementarity: `max(|y_i · slack_i|, |z_j · (x_j−bnd_j)|) / normaliser`.
    pub fn complementarity_rel(&self) -> f64 { self.complementarity_rel }

    /// Dual-sign violation: `max_k viol_k / (1 + |v_k|)` over sign-constrained duals.
    pub fn dual_sign_violation(&self) -> f64 { self.dual_sign_violation }

    /// Relative duality gap: `|p_obj − d_obj| / max(|p|, |d|, 1)`.
    pub fn duality_gap_rel(&self) -> f64 { self.duality_gap_rel }

    /// Tolerance used when the certificate was issued.
    pub fn tol(&self) -> f64 { self.tol }
}

/// Reason why [`prove_optimal`] failed to issue an [`OptimalCertificate`].
///
/// Contains the observed residuals and the names of the conditions that exceeded `tol`.
#[derive(Debug, Clone)]
pub struct NotProven {
    /// Stationarity residual observed.
    pub stationarity_rel: f64,
    /// Primal residual observed.
    pub primal_residual_rel: f64,
    /// Bound violation observed.
    pub bound_violation: f64,
    /// Complementarity residual observed.
    pub complementarity_rel: f64,
    /// Dual-sign violation observed.
    pub dual_sign_violation: f64,
    /// Duality-gap residual observed.
    pub duality_gap_rel: f64,
    /// Tolerance against which residuals were compared.
    pub tol: f64,
    /// Names of conditions that exceeded `tol`.
    pub failing_conditions: Vec<&'static str>,
}

/// Branch-and-bound gap certificate — minted only by the B&B driver when:
/// - No region was abandoned due to a non-Optimal relaxation (`proof_uncertain == false`).
/// - `(incumbent_obj - lower_bound) / max(1, |incumbent_obj|) ≤ gap_tol`.
///
/// Holding this value is a proof token: the incumbent is within `gap_tol` of the global
/// optimum. External code cannot construct it (constructor is `pub(crate)`).
#[derive(Debug, Clone)]
pub struct BoundGapCertificate {
    incumbent_obj: f64,
    lower_bound: f64,
    gap_rel: f64,
    gap_tol: f64,
}

impl BoundGapCertificate {
    pub(crate) fn new(incumbent_obj: f64, lower_bound: f64, gap_rel: f64, gap_tol: f64) -> Self {
        Self { incumbent_obj, lower_bound, gap_rel, gap_tol }
    }

    /// Best integer-feasible objective found by the search.
    pub fn incumbent_obj(&self) -> f64 { self.incumbent_obj }

    /// Authenticated lower bound on the global optimum at termination.
    pub fn lower_bound(&self) -> f64 { self.lower_bound }

    /// Relative gap: `(incumbent_obj - lower_bound) / max(1, |incumbent_obj|)`.
    pub fn gap_rel(&self) -> f64 { self.gap_rel }

    /// Tolerance against which the gap was checked when the certificate was issued.
    pub fn gap_tol(&self) -> f64 { self.gap_tol }
}

/// Placeholder for Farkas infeasibility certificate (Phase 2+).
#[derive(Debug, Clone)]
pub struct FarkasCertificate;

/// Placeholder for unbounded-ray certificate (Phase 2+).
#[derive(Debug, Clone)]
pub struct UnboundedRayCertificate;

/// Typed solver outcome carrying certificates.
///
/// Phase 1 adds the types; existing solver entry points are not yet wired here
/// (connection happens in Phase 2+). The enum is defined now so that downstream
/// code can import and pattern-match against it.
#[derive(Debug)]
pub enum SolveOutcome {
    /// Globally optimal (PSD Q or LP); KKT verified.
    Optimal {
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<f64>,
        objective: f64,
        cert: OptimalCertificate,
    },
    /// KKT-point for a non-convex QP (inertia-corrected IPM converged).
    LocalOptimal {
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<f64>,
        objective: f64,
        kkt_cert: OptimalCertificate,
    },
    /// Primal infeasible; Farkas certificate attached.
    Infeasible { cert: FarkasCertificate },
    /// Primal unbounded; ray certificate attached.
    Unbounded { cert: UnboundedRayCertificate },
    /// Solver stopped before proof completion.
    Incomplete {
        incumbent: Option<Vec<f64>>,
        reason: IncompleteReason,
    },
}

/// Why a solve completed without a full optimality proof.
#[derive(Debug, Clone)]
pub enum IncompleteReason {
    MaxIterations,
    Timeout,
    NumericalError,
    SuboptimalSolution,
    NonConvex(String),
    NotSupported(String),
}
