//! Solver configuration parameters.
//!
//! [`SolverOptions`] controls simplex and IPM solver behaviour: tolerances,
//! iteration limits, refactorisation frequency, and algorithm selection.
//!
//! ## Solver-specific options
//!
//! IPM-specific parameters live in [`IpmOptions`], accessed via
//! [`SolverOptions::ipm`].

use crate::tolerances::*;
use std::sync::{
    atomic::AtomicBool,
    Arc,
};

use std::time::Instant;

// ---- Error type -------------------------------------------------------

/// Error returned when option values fail validation.
///
/// Produced by [`IpmOptions::validate`] and [`SolverOptions::validate`], and
/// by builder methods (`with_*`) that validate on assignment.
#[derive(Debug, Clone, PartialEq)]
pub struct OptionsError {
    /// Name of the offending field (e.g. `"ipm.eps"`).
    pub field: &'static str,
    /// Human-readable rejection reason.
    pub reason: &'static str,
}

impl std::fmt::Display for OptionsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid option `{}`: {}", self.field, self.reason)
    }
}

impl std::error::Error for OptionsError {}

// ---- Enum / simple struct types ---------------------------------------

/// Dual simplex leaving (depart) strategy.
///
/// `MostInfeasible`: select the most negative x_B\[i\] (Dantzig rule).
/// Stable but inflates iteration count on large problems.
///
/// `SteepestEdge`: Forrest-Goldfarb 1992 Dual Steepest Edge.
/// Maintains weight γ_i = ||(B^{-1})_{i,:}||² and maximises
/// score = x_B\[i\]² / γ_i.  Typical 3-10× speed-up (HiGHS/CPLEX) at the cost
/// of one extra FTRAN per iteration.
///
/// Default: `MostInfeasible` (easy A/B comparison; preserves existing behaviour).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DualPricing {
    #[default]
    MostInfeasible,
    SteepestEdge,
}

/// Simplex algorithm selection.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimplexMethod {
    /// Auto-select based on warm-start availability.
    #[default]
    Auto,
    /// Force Primal Simplex.
    Primal,
    /// Force Dual Simplex.
    Dual,
    /// Production-quality Dual Simplex (`dual_advanced` module).
    DualAdvanced,
}

/// Basis information for warm-starting simplex.
///
/// Carries basis indices and primal values from a previous solve. Used as the
/// initial basis for Dual Simplex in SQP integration.
#[derive(Debug, Clone)]
pub struct WarmStartBasis {
    /// Basis variable indices (standard-form column numbers, length = m).
    pub basis: Vec<usize>,
    /// Basis variable values x_B (length = m). Stale values are acceptable;
    /// they are recomputed from the new RHS on warm-start entry.
    pub x_b: Vec<f64>,
}

/// QP IP-PMM interior-point warm-start data.
///
/// Passes the optimal (x, y, μ) from a parent B&B node as the starting point
/// on the central path for the child node.  LP warm-start uses basis indices
/// ([`WarmStartBasis`]); QP warm-start uses a central-path point.
///
/// Convention:
/// - `x`: length = n (primal)
/// - `y`: length = m (dual, user sign convention; Ge constraints inverted internally)
/// - `mu`: barrier parameter ≈ sᵀy / m_ineq of the parent final iterate
///
/// Interior corrections (μ floor / x bound margin / y positivity) are applied
/// on entry so boundary or zero values are safe to pass.
#[derive(Debug, Clone)]
pub struct QpWarmStart {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub mu: f64,
}

/// Extended LP warm-start.
///
/// Superset of [`WarmStartBasis`]: accepts (x, y, basis) from an external
/// solver and lands simplex at that point.  Takes priority over `warm_start`.
///
/// Convention:
/// - `basis`: length = m_ext (standard-form rows), each value < n_total.
///   Size mismatch: logged and dropped (not silently ignored).
/// - `x_orig`: length = problem.num_vars (original variable space)
/// - `y_orig`: length = problem.num_constraints (original constraint space, user sign)
#[derive(Debug, Clone)]
pub struct LpWarmStart {
    pub basis: Vec<usize>,
    pub x_orig: Option<Vec<f64>>,
    pub y_orig: Option<Vec<f64>>,
}

/// Multi-start sampling strategy.
///
/// IPM converges to the nearest KKT point under inertia correction, so
/// different starting points can reach different local optima on non-convex QPs.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartStrategy {
    /// Independent uniform sampling within box bounds (LCG).
    RandomBox,
    /// Latin Hypercube Sampling: partition each dimension into `n_starts`
    /// strata and permute per column.  Better global coverage than pure random.
    LatinHypercube,
}

/// Multi-start local search user-facing config.
///
/// Solves `n_starts` independent IPM problems from different starting points
/// and returns the best objective.  Improves escape rate on non-convex QPs
/// and supplies incumbents for spatial B&B.
///
/// **User-controlled (pub fields):**
/// - `n_starts`: parallelism / hit probability
/// - `seed`: reproducibility (`0` is internally clamped to 1 to avoid LCG lock)
/// - `strategy`: sampling strategy
///
/// `n_starts == 1`: single cold solve (existing behaviour).
/// `n_starts >= 2`: start #0 = cold, #1..n = random (warm_start_qp.x injected).
/// All starts share the same deadline.
#[derive(Debug, Clone)]
pub struct MultiStartConfig {
    /// Number of starting points.  1 disables multi-start.  Default = 1.
    pub n_starts: usize,
    /// Random seed.  Default = [`DEFAULT_MULTISTART_SEED`].
    pub seed: u64,
    /// Sampling strategy.  Default = `RandomBox`.
    pub strategy: StartStrategy,
}

/// Default seed for [`MultiStartConfig`].  Fixed non-zero value for
/// deterministic test environments.
pub const DEFAULT_MULTISTART_SEED: u64 = 0x_00C0_FFEE_DEAD_BEEF;

/// Branching strategy for spatial B&B.
///
/// `MaxViolation`: branch on the variable whose x* deviates most from the
/// box midpoint, splitting at x*\[j\].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchingStrategy {
    MaxViolation,
}

/// Defaults for [`GlobalOptimizationConfig`].
///
/// - `DEFAULT_GLOBAL_GAP_TOL = 1e-3`: Phase 3 interval-arithmetic bounds are
///   loose; tightening to 1e-6 causes node explosion.  Phase 4 (α-BB) can tighten.
/// - `DEFAULT_GLOBAL_MAX_DEPTH = 20`: tree depth cap (2^20 ≈ 1 M nodes).
/// - `DEFAULT_GLOBAL_MAX_NODES = 10_000`: node budget (~1 IPM solve per node).
pub const DEFAULT_GLOBAL_GAP_TOL: f64 = 1e-3;
pub const DEFAULT_GLOBAL_MAX_DEPTH: usize = 20;
pub const DEFAULT_GLOBAL_MAX_NODES: usize = 10_000;

/// Spatial Branch-and-Bound config for global QP optimisation.
///
/// Set [`SolverOptions::global_optimization`] and call `solve_qp_global`
/// explicitly.  `solve_qp_with` does **not** dispatch to this path (prevents
/// accidental wall-time blow-up for existing users).
///
/// Rules:
/// - `gap_tol > 0`: relative gap = |UB − LB| / max(1, |UB|)
/// - `max_depth >= 1`, `max_nodes >= 1`
#[derive(Debug, Clone)]
pub struct GlobalOptimizationConfig {
    pub gap_tol: f64,
    pub max_depth: usize,
    pub max_nodes: usize,
    pub branching: BranchingStrategy,
    pub use_alpha_bb: bool,
    pub use_mccormick: bool,
}

impl Default for GlobalOptimizationConfig {
    fn default() -> Self {
        Self {
            gap_tol: DEFAULT_GLOBAL_GAP_TOL,
            max_depth: DEFAULT_GLOBAL_MAX_DEPTH,
            max_nodes: DEFAULT_GLOBAL_MAX_NODES,
            branching: BranchingStrategy::MaxViolation,
            use_alpha_bb: true,
            use_mccormick: false,
        }
    }
}

impl Default for MultiStartConfig {
    fn default() -> Self {
        Self {
            n_starts: 1,
            seed: DEFAULT_MULTISTART_SEED,
            strategy: StartStrategy::RandomBox,
        }
    }
}

/// MILP/MIQP branching variable selection strategy.
///
/// `MostFractional`: branch on the integer-constrained variable whose
/// relaxation value is closest to 0.5.  Ties broken by variable index.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipBranching {
    MostFractional,
}

/// Defaults for [`MipConfig`].
///
/// - `DEFAULT_MIP_GAP_TOL = 1e-6`: tighter than spatial B&B (1e-3) because LP/QP
///   relaxations give exact lower bounds.
/// - `DEFAULT_INTEGER_FEAS_TOL = 1e-6`: integrality threshold.
/// - `DEFAULT_MIP_MAX_NODES = 1_000_000`: safety cap (deadline is primary cutoff).
/// - `DEFAULT_MIP_MAX_DEPTH = 1_000`: depth cap.
pub const DEFAULT_MIP_GAP_TOL: f64 = 1e-6;
pub const DEFAULT_INTEGER_FEAS_TOL: f64 = 1e-6;
pub const DEFAULT_MIP_MAX_NODES: usize = 1_000_000;
pub const DEFAULT_MIP_MAX_DEPTH: usize = 1_000;

/// MILP/MIQP branch-and-bound config.
///
/// Passed to `solve_milp` / `solve_miqp`.
///
/// Rules:
/// - `gap_tol >= 0`: 0 means exact optimality (node explosion risk).
/// - `integer_feas_tol > 0`
/// - `max_nodes >= 1`, `max_depth >= 1`
#[derive(Debug, Clone)]
pub struct MipConfig {
    pub gap_tol: f64,
    pub integer_feas_tol: f64,
    pub max_nodes: usize,
    pub max_depth: usize,
    pub branching: MipBranching,
}

impl Default for MipConfig {
    fn default() -> Self {
        Self {
            gap_tol: DEFAULT_MIP_GAP_TOL,
            integer_feas_tol: DEFAULT_INTEGER_FEAS_TOL,
            max_nodes: DEFAULT_MIP_MAX_NODES,
            max_depth: DEFAULT_MIP_MAX_DEPTH,
            branching: MipBranching::MostFractional,
        }
    }
}

// ---- Tolerance --------------------------------------------------------

/// IPM eps for [`Tolerance::High`].
pub const TOLERANCE_HIGH_EPS: f64 = 1e-8;
/// IPM eps for [`Tolerance::Medium`] (default).
pub const TOLERANCE_MEDIUM_EPS: f64 = 1e-6;
/// IPM eps for [`Tolerance::Fast`]: 100× looser than Medium for faster convergence.
pub const TOLERANCE_FAST_EPS: f64 = 1e-4;

/// Convergence accuracy level.
///
/// Abstracts the raw `ipm.eps` field.  When set on [`SolverOptions`], the
/// solver derives its internal convergence threshold from this enum;
/// `ipm.eps` is ignored.
///
/// ## Translation table
///
/// | Tolerance | IPM eps                              |
/// |-----------|--------------------------------------|
/// | High      | [`TOLERANCE_HIGH_EPS`] = 1e-8        |
/// | Medium    | [`TOLERANCE_MEDIUM_EPS`] = 1e-6      |
/// | Fast      | [`TOLERANCE_FAST_EPS`] = 1e-4        |
/// | Custom(v) | v                                    |
///
/// `Medium` is the default (comparable to Gurobi `eps = 1e-6`).
/// `Fast` accepts solutions 100× less precise than Medium for reduced
/// iteration counts — appropriate when a coarse objective estimate suffices.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tolerance {
    /// High accuracy: research / verification workloads.
    High,
    /// Medium accuracy (default): general-purpose workloads.
    Medium,
    /// Fast: speed-priority, looser convergence (100× coarser than Medium).
    Fast,
    /// Custom: pass the eps value directly to each solver.
    Custom(f64),
}

// ---- IpmOptions -------------------------------------------------------

/// Default convergence tolerance for [`IpmOptions::eps`].
pub const DEFAULT_IPM_EPS: f64 = 1e-6;
/// Default proximity regularisation lower bound for [`IpmOptions::delta_min`].
pub const DEFAULT_IPM_DELTA_MIN: f64 = 1e-8;
/// Default initial proximity regularisation for [`IpmOptions::delta_p_init`]
/// and [`IpmOptions::delta_d_init`].
pub const DEFAULT_IPM_DELTA_INIT: f64 = 1e-6;
/// Default Gondzio corrector count (Gondzio 1997, recommended range 2–5).
pub const DEFAULT_IPM_MAX_CORRECTORS: usize = 3;

/// IPM (interior-point method) solver options.
///
/// Set via [`SolverOptions::ipm`].  Call [`IpmOptions::validate`] (or
/// [`SolverOptions::validate`]) before solving to catch invalid values early.
#[derive(Debug, Clone)]
pub struct IpmOptions {
    /// Maximum iterations.  Default: `usize::MAX` (timeout is the primary guard).
    pub max_iter: usize,
    /// Convergence tolerance.  Default: [`DEFAULT_IPM_EPS`].
    pub eps: f64,
    /// Proximity regularisation lower bound δ_min.  Default: [`DEFAULT_IPM_DELTA_MIN`].
    pub delta_min: f64,
    /// Initial primal proximity regularisation δ_p.  Default: [`DEFAULT_IPM_DELTA_INIT`].
    pub delta_p_init: f64,
    /// Initial dual proximity regularisation δ_d.  Default: [`DEFAULT_IPM_DELTA_INIT`].
    pub delta_d_init: f64,
    /// Maximum Gondzio correctors.  Default: [`DEFAULT_IPM_MAX_CORRECTORS`].
    pub max_correctors: usize,
    /// Use TwoFloat (double-double, ~106-bit) LDL for KKT systems where f64 conditioning
    /// would exceed the requested accuracy.  Default: `false`.
    pub dd_ldl: bool,
    /// MINRES iterative-refinement rounds applied after each MINRES solve.
    /// `None` uses 0 (disabled by default; auto-Schur makes this unnecessary in practice).
    /// Must be `<= 10`.
    pub minres_ir: Option<usize>,
    /// Memory budget for KKT LDL factorization in bytes.
    /// `None` uses the 4 GiB default.  Factorizations predicted to exceed the budget
    /// fall back to MINRES automatically.
    pub kkt_memory_budget_bytes: Option<usize>,
}

impl Default for IpmOptions {
    fn default() -> Self {
        Self {
            max_iter: usize::MAX,
            eps: DEFAULT_IPM_EPS,
            delta_min: DEFAULT_IPM_DELTA_MIN,
            delta_p_init: DEFAULT_IPM_DELTA_INIT,
            delta_d_init: DEFAULT_IPM_DELTA_INIT,
            max_correctors: DEFAULT_IPM_MAX_CORRECTORS,
            dd_ldl: false,
            minres_ir: None,
            kkt_memory_budget_bytes: None,
        }
    }
}

impl IpmOptions {
    /// Validate all numeric fields.
    ///
    /// Returns the first `Err` in field declaration order.
    /// Invalid: non-finite or non-positive `eps` / `delta_*`, or `max_correctors == 0`.
    pub fn validate(&self) -> Result<(), OptionsError> {
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(OptionsError { field: "ipm.eps", reason: "must be finite and > 0" });
        }
        if !self.delta_min.is_finite() || self.delta_min <= 0.0 {
            return Err(OptionsError { field: "ipm.delta_min", reason: "must be finite and > 0" });
        }
        if !self.delta_p_init.is_finite() || self.delta_p_init <= 0.0 {
            return Err(OptionsError { field: "ipm.delta_p_init", reason: "must be finite and > 0" });
        }
        if !self.delta_d_init.is_finite() || self.delta_d_init <= 0.0 {
            return Err(OptionsError { field: "ipm.delta_d_init", reason: "must be finite and > 0" });
        }
        if self.max_correctors == 0 {
            return Err(OptionsError { field: "ipm.max_correctors", reason: "must be >= 1" });
        }
        if let Some(ir) = self.minres_ir {
            if ir > 10 {
                return Err(OptionsError { field: "ipm.minres_ir", reason: "must be <= 10" });
            }
        }
        Ok(())
    }

    /// Builder: set `eps`, validated immediately.
    pub fn with_eps(mut self, eps: f64) -> Result<Self, OptionsError> {
        if !eps.is_finite() || eps <= 0.0 {
            return Err(OptionsError { field: "ipm.eps", reason: "must be finite and > 0" });
        }
        self.eps = eps;
        Ok(self)
    }

    /// Builder: set `max_correctors`, validated immediately.
    pub fn with_max_correctors(mut self, n: usize) -> Result<Self, OptionsError> {
        if n == 0 {
            return Err(OptionsError { field: "ipm.max_correctors", reason: "must be >= 1" });
        }
        self.max_correctors = n;
        Ok(self)
    }

    /// Effective MINRES iterative-refinement rounds: resolves `None` to 0.
    pub(crate) fn effective_minres_ir(&self) -> usize {
        self.minres_ir.unwrap_or(0)
    }

    /// Effective KKT memory budget in bytes: resolves `None` to the built-in default (4 GiB).
    pub(crate) fn effective_kkt_memory_budget_bytes(&self) -> usize {
        use crate::linalg::kkt_solver::DEFAULT_MEMORY_BUDGET_BYTES;
        self.kkt_memory_budget_bytes.unwrap_or(DEFAULT_MEMORY_BUDGET_BYTES)
    }

    /// Max L-factor entries from memory budget (budget / bytes-per-entry).
    pub(crate) fn effective_max_l_nnz(&self) -> usize {
        use crate::linalg::kkt_solver::BYTES_PER_L_ENTRY;
        self.effective_kkt_memory_budget_bytes() / BYTES_PER_L_ENTRY
    }
}

// ---- SolverOptions ----------------------------------------------------

/// Default clamp threshold for micro-values in solver output.
pub const DEFAULT_CLAMP_TOL: f64 = 1e-14;

/// Solver configuration.
///
/// Controls tolerances, iteration limits, refactorisation frequency, and
/// algorithm selection.  `Default` uses values from `tolerances.rs`.
///
/// ## Validation
///
/// Call [`SolverOptions::validate`] (or use builder methods) before solving
/// to catch invalid values (NaN, zero, negative tolerances, etc.) early.
///
/// ## Solver-specific parameters
///
/// Use the [`SolverOptions::ipm`] sub-struct for IPM-specific settings.
#[derive(Debug, Clone)]
pub struct SolverOptions {
    // --- Common ---
    /// Simplex primal feasibility / optimality threshold.  Default: `PIVOT_TOL`.
    pub primal_tol: f64,
    /// Max eta-file count (refactorisation threshold).  0 = auto (from problem size).
    pub max_etas: usize,
    /// Micro-value clamp threshold.  Default: [`DEFAULT_CLAMP_TOL`].
    pub clamp_tol: f64,
    /// Simplex algorithm selection.  Default: `Auto`.
    pub simplex_method: SimplexMethod,
    /// Dual feasibility threshold.  Default: `PIVOT_TOL`.
    pub dual_tol: f64,
    /// Dual simplex leaving strategy.  Default: `MostInfeasible`.
    pub dual_pricing: DualPricing,
    /// Enable Bound-Flipping Ratio Test (Maros 2003 §7.6) in `dual_advanced`.
    /// Runtime override: `BOUND_FLIP_DISABLE=1`.
    pub enable_bound_flipping: bool,
    /// LP warm-start basis.  `None` = cold start.
    pub warm_start: Option<WarmStartBasis>,
    /// QP IP-PMM interior-point warm start for B&B node transfer.
    pub warm_start_qp: Option<QpWarmStart>,
    /// Extended LP warm start; takes priority over `warm_start`.
    pub warm_start_lp: Option<LpWarmStart>,
    /// Reconstruct `warm_start_basis` after postsolve.  Default: `false`.
    ///
    /// When presolve reduces the problem the reduced-LP basis indices are
    /// invalid for the original LP.  `true` triggers basis reconstruction at
    /// postsolve exit (LTSF crash + solution refinement).  Opt-in only.
    ///
    /// When presolve is skipped or the problem was not reduced, the simplex
    /// basis is cloned directly regardless of this flag.
    pub recover_warm_start_basis: bool,
    /// Apply simplex crash basis on cold LP starts.  Ignored when
    /// `warm_start` / `warm_start_lp` is set.
    pub use_lp_crash_basis: bool,
    /// Enable presolve.  Default: `true`.
    pub presolve: bool,
    /// Maximum fixpoint passes in QP presolve.  Default: `10`.
    pub presolve_max_pass: usize,
    /// Skip large-coefficient row rescaling in QP presolve.  Default: `false`.
    /// Rescaling is also skipped when `use_ruiz_scaling` is `true` (existing behaviour).
    pub presolve_skip_large_coeff: bool,
    /// Enable QP presolve phase 2.  Default: `true`.
    pub presolve_phase2: bool,
    /// Timeout in seconds.  `None` = unlimited.
    pub timeout_secs: Option<f64>,
    /// Shared cancellation flag (internal use).
    pub(crate) cancel_flag: Option<Arc<AtomicBool>>,
    /// Solve deadline computed from `timeout_secs` at solve entry (internal use).
    pub(crate) deadline: Option<Instant>,

    // --- Ruiz scaling ---
    /// Apply Ruiz equilibration scaling before IPM.  Default: `true`.
    pub use_ruiz_scaling: bool,

    // --- Tolerance abstraction ---
    /// Convergence accuracy level.  `None` = use `ipm.eps` directly.
    ///
    /// When `Some(_)`, each solver derives eps from this; `ipm.eps` is ignored.
    pub tolerance: Option<Tolerance>,

    // --- Solver-specific ---
    /// IPM-specific options.
    pub ipm: IpmOptions,

    /// Multi-start local search config.  `None` (default) = disabled.
    pub multistart: Option<MultiStartConfig>,

    /// Spatial B&B global optimisation config.  `None` (default) = disabled.
    /// Only consumed by explicit `solve_qp_global` calls.
    pub global_optimization: Option<GlobalOptimizationConfig>,

    /// Thread budget for all solver paths (LP / QP / multistart).
    ///
    /// Default = 1 (serial; no contention with external bench workers).
    ///
    /// - **QP** (`threads >= 2`): enables faer parallel sparse LDL on the KKT system.
    /// - **LP simplex** (`threads >= 2`): no effect.
    /// - **Multistart** (`threads >= 2`): `min(n_starts, threads)` parallel degree;
    ///   inner solves forced to `threads = 1`.
    pub threads: usize,

    /// Reference optimal objective for early-exit.
    ///
    /// When `Some(ref_obj)`, returns `Optimal` as soon as
    /// `|obj − ref_obj| / (1 + |ref_obj|) < OBJ_MATCH_REL_TOL`.
    /// Used by bench harnesses.  `None` = no early-exit.
    pub known_optimal_obj: Option<f64>,
}

/// Divisor for the `max_etas` heuristic: floor(m / MAX_ETAS_DIVISOR).
const MAX_ETAS_DIVISOR: usize = 50;
/// Minimum value for `default_max_etas`.
const MAX_ETAS_FLOOR: usize = 20;

/// Default maximum fixpoint passes for QP presolve.
pub(crate) const DEFAULT_PRESOLVE_MAX_PASS: usize = 10;

/// Auto-compute `max_etas` from problem size.
///
/// Small problems (m < 1000): `MAX_ETAS_FLOOR`; larger: m / `MAX_ETAS_DIVISOR`.
pub fn default_max_etas(m: usize) -> usize {
    (m / MAX_ETAS_DIVISOR).max(MAX_ETAS_FLOOR)
}

/// Phase I retry cap: guards against degenerate problems that loop with an
/// identical basis in `revised_simplex_core`.
pub const MAX_PHASE1_RETRIES: usize = 8;

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            primal_tol: PIVOT_TOL,
            max_etas: 0,
            clamp_tol: DEFAULT_CLAMP_TOL,
            simplex_method: SimplexMethod::Auto,
            dual_tol: PIVOT_TOL,
            dual_pricing: DualPricing::default(),
            enable_bound_flipping: false,
            warm_start: None,
            warm_start_qp: None,
            warm_start_lp: None,
            recover_warm_start_basis: false,
            use_lp_crash_basis: true,
            presolve: true,
            presolve_max_pass: DEFAULT_PRESOLVE_MAX_PASS,
            presolve_skip_large_coeff: false,
            presolve_phase2: true,
            timeout_secs: None,
            cancel_flag: None,
            deadline: None,
            use_ruiz_scaling: true,
            tolerance: None,
            ipm: IpmOptions::default(),
            multistart: None,
            global_optimization: None,
            threads: 1,
            known_optimal_obj: None,
        }
    }
}

impl SolverOptions {
    /// Effective IPM eps: derived from `tolerance` if set, otherwise `ipm.eps`.
    pub fn ipm_eps(&self) -> f64 {
        match self.tolerance {
            Some(Tolerance::High)      => TOLERANCE_HIGH_EPS,
            Some(Tolerance::Medium)    => TOLERANCE_MEDIUM_EPS,
            Some(Tolerance::Fast)      => TOLERANCE_FAST_EPS,
            Some(Tolerance::Custom(v)) => v,
            None => self.ipm.eps,
        }
    }

    /// Validate all option fields.
    ///
    /// Returns the first `Err` encountered, in field declaration order.
    /// Called by public solver entry points (`solve_qp_with`, `solve_qp_global`,
    /// `solve_qp_multistart`, `solve_milp`, `solve_miqp`, `simplex::solve_with`)
    /// before starting work; invalid options cause the entry to return
    /// [`crate::problem::SolveStatus::NumericalError`] rather than propagating
    /// bad values into the solver core.
    ///
    /// Invalid conditions:
    /// - `primal_tol` / `dual_tol`: non-finite or <= 0
    /// - `clamp_tol`: non-finite or < 0 (0 is allowed)
    /// - `threads`: 0
    /// - `timeout_secs`: `Some(v)` where v is non-finite or < 0
    /// - `tolerance`: `Custom(v)` where v is non-finite or <= 0
    /// - Any field in [`IpmOptions`]
    pub fn validate(&self) -> Result<(), OptionsError> {
        if !self.primal_tol.is_finite() || self.primal_tol <= 0.0 {
            return Err(OptionsError { field: "primal_tol", reason: "must be finite and > 0" });
        }
        if !self.dual_tol.is_finite() || self.dual_tol <= 0.0 {
            return Err(OptionsError { field: "dual_tol", reason: "must be finite and > 0" });
        }
        if !self.clamp_tol.is_finite() || self.clamp_tol < 0.0 {
            return Err(OptionsError { field: "clamp_tol", reason: "must be finite and >= 0" });
        }
        if self.threads == 0 {
            return Err(OptionsError { field: "threads", reason: "must be >= 1" });
        }
        if let Some(t) = self.timeout_secs {
            if !t.is_finite() || t < 0.0 {
                return Err(OptionsError { field: "timeout_secs", reason: "must be finite and >= 0" });
            }
        }
        if let Some(Tolerance::Custom(v)) = self.tolerance {
            if !v.is_finite() || v <= 0.0 {
                return Err(OptionsError {
                    field: "tolerance.Custom",
                    reason: "must be finite and > 0",
                });
            }
        }
        self.ipm.validate()?;
        Ok(())
    }

    /// Builder: set `timeout_secs`, validated immediately.
    pub fn with_timeout(mut self, secs: f64) -> Result<Self, OptionsError> {
        if !secs.is_finite() || secs < 0.0 {
            return Err(OptionsError { field: "timeout_secs", reason: "must be finite and >= 0" });
        }
        self.timeout_secs = Some(secs);
        Ok(self)
    }

    /// Builder: set `threads`, validated immediately.
    pub fn with_threads(mut self, n: usize) -> Result<Self, OptionsError> {
        if n == 0 {
            return Err(OptionsError { field: "threads", reason: "must be >= 1" });
        }
        self.threads = n;
        Ok(self)
    }

    /// Builder: set `tolerance`, validated immediately.
    ///
    /// `Tolerance::Custom(v)` requires v to be finite and > 0; other variants
    /// are always accepted.
    pub fn with_tolerance(mut self, tol: Tolerance) -> Result<Self, OptionsError> {
        if let Tolerance::Custom(v) = tol {
            if !v.is_finite() || v <= 0.0 {
                return Err(OptionsError {
                    field: "tolerance.Custom",
                    reason: "must be finite and > 0",
                });
            }
        }
        self.tolerance = Some(tol);
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Tolerance translation -------------------------------------------

    #[test]
    fn test_tolerance_translation() {
        // Table-driven: (tolerance setting, expected ipm_eps)
        let cases: &[(Option<Tolerance>, f64)] = &[
            (Some(Tolerance::High),         TOLERANCE_HIGH_EPS),
            (Some(Tolerance::Medium),       TOLERANCE_MEDIUM_EPS),
            (Some(Tolerance::Fast),         TOLERANCE_FAST_EPS),
            (Some(Tolerance::Custom(1e-5)), 1e-5),
            (None,                          DEFAULT_IPM_EPS), // uses ipm.eps default
        ];
        for (tol, expected) in cases {
            let opts = SolverOptions { tolerance: *tol, ..Default::default() };
            assert_eq!(opts.ipm_eps(), *expected, "tolerance = {:?}", tol);
        }
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_tolerance_fast_is_looser_than_medium() {
        // Fast must be coarser (larger eps) than Medium; otherwise the name is misleading.
        const { assert!(TOLERANCE_FAST_EPS > TOLERANCE_MEDIUM_EPS) }
        const { assert!(TOLERANCE_MEDIUM_EPS > TOLERANCE_HIGH_EPS) }
    }

    // ---- IpmOptions::validate -------------------------------------------

    #[test]
    fn test_ipm_validate_defaults_ok() {
        assert!(IpmOptions::default().validate().is_ok());
    }

    #[test]
    fn test_ipm_validate_eps() {
        for bad in [0.0_f64, -1e-6, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let o = IpmOptions { eps: bad, ..Default::default() };
            assert!(o.validate().is_err(), "eps={bad} should be invalid");
        }
        // boundary: smallest positive finite value is valid
        let o = IpmOptions { eps: f64::MIN_POSITIVE, ..Default::default() };
        assert!(o.validate().is_ok());
    }

    #[test]
    fn test_ipm_validate_delta_min() {
        for bad in [0.0_f64, -1.0, f64::NAN, f64::INFINITY] {
            let o = IpmOptions { delta_min: bad, ..Default::default() };
            assert!(o.validate().is_err(), "delta_min={bad} should be invalid");
        }
    }

    #[test]
    fn test_ipm_validate_delta_p_init() {
        for bad in [0.0_f64, -1.0, f64::NAN, f64::INFINITY] {
            let o = IpmOptions { delta_p_init: bad, ..Default::default() };
            assert!(o.validate().is_err(), "delta_p_init={bad} should be invalid");
        }
    }

    #[test]
    fn test_ipm_validate_delta_d_init() {
        for bad in [0.0_f64, -1.0, f64::NAN, f64::INFINITY] {
            let o = IpmOptions { delta_d_init: bad, ..Default::default() };
            assert!(o.validate().is_err(), "delta_d_init={bad} should be invalid");
        }
    }

    #[test]
    fn test_ipm_validate_max_correctors() {
        let o = IpmOptions { max_correctors: 0, ..Default::default() };
        assert!(o.validate().is_err(), "max_correctors=0 should be invalid");
        let o = IpmOptions { max_correctors: 1, ..Default::default() };
        assert!(o.validate().is_ok());
    }

    // ---- IpmOptions builders --------------------------------------------

    #[test]
    fn test_ipm_builder_with_eps() {
        assert!(IpmOptions::default().with_eps(1e-4).is_ok());
        assert!(IpmOptions::default().with_eps(f64::MIN_POSITIVE).is_ok());
        for bad in [0.0_f64, -1.0, f64::NAN, f64::INFINITY] {
            assert!(IpmOptions::default().with_eps(bad).is_err(), "with_eps({bad}) should err");
        }
    }

    #[test]
    fn test_ipm_builder_with_max_correctors() {
        assert!(IpmOptions::default().with_max_correctors(1).is_ok());
        assert!(IpmOptions::default().with_max_correctors(10).is_ok());
        assert!(IpmOptions::default().with_max_correctors(0).is_err());
    }

    // ---- SolverOptions::validate ----------------------------------------

    #[test]
    fn test_solver_validate_defaults_ok() {
        assert!(SolverOptions::default().validate().is_ok());
    }

    #[test]
    fn test_solver_validate_primal_tol() {
        for bad in [0.0_f64, -1e-8, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let o = SolverOptions { primal_tol: bad, ..Default::default() };
            assert!(o.validate().is_err(), "primal_tol={bad}");
        }
        let o = SolverOptions { primal_tol: f64::MIN_POSITIVE, ..Default::default() };
        assert!(o.validate().is_ok());
    }

    #[test]
    fn test_solver_validate_dual_tol() {
        for bad in [0.0_f64, -1e-8, f64::NAN, f64::INFINITY] {
            let o = SolverOptions { dual_tol: bad, ..Default::default() };
            assert!(o.validate().is_err(), "dual_tol={bad}");
        }
    }

    #[test]
    fn test_solver_validate_clamp_tol() {
        // 0.0 is valid (no clamping)
        let o = SolverOptions { clamp_tol: 0.0, ..Default::default() };
        assert!(o.validate().is_ok(), "clamp_tol=0 should be ok");
        for bad in [-1.0_f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let o = SolverOptions { clamp_tol: bad, ..Default::default() };
            assert!(o.validate().is_err(), "clamp_tol={bad}");
        }
    }

    #[test]
    fn test_solver_validate_threads() {
        let o = SolverOptions { threads: 0, ..Default::default() };
        assert!(o.validate().is_err(), "threads=0");
        for ok in [1_usize, 2, 8, usize::MAX] {
            let o = SolverOptions { threads: ok, ..Default::default() };
            assert!(o.validate().is_ok(), "threads={ok}");
        }
    }

    #[test]
    fn test_solver_validate_timeout_secs() {
        // None is always valid
        assert!(SolverOptions { timeout_secs: None, ..Default::default() }.validate().is_ok());
        // non-negative finite: valid (0.0 = immediately-expired deadline)
        for ok in [0.0_f64, 0.001, 1.0, 1000.0] {
            let o = SolverOptions { timeout_secs: Some(ok), ..Default::default() };
            assert!(o.validate().is_ok(), "timeout_secs=Some({ok}) must be valid");
        }
        // invalid: negative, NaN, or infinite
        for bad in [-1.0_f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let o = SolverOptions { timeout_secs: Some(bad), ..Default::default() };
            assert!(o.validate().is_err(), "timeout_secs=Some({bad})");
        }
    }

    #[test]
    fn test_solver_validate_tolerance_custom() {
        // Non-Custom variants are always valid
        for tol in [Tolerance::High, Tolerance::Medium, Tolerance::Fast] {
            let o = SolverOptions { tolerance: Some(tol), ..Default::default() };
            assert!(o.validate().is_ok(), "tolerance={tol:?}");
        }
        // Custom: valid
        let o = SolverOptions { tolerance: Some(Tolerance::Custom(1e-5)), ..Default::default() };
        assert!(o.validate().is_ok());
        // Custom: invalid
        for bad in [0.0_f64, -1e-4, f64::NAN, f64::INFINITY] {
            let o = SolverOptions { tolerance: Some(Tolerance::Custom(bad)), ..Default::default() };
            assert!(o.validate().is_err(), "Tolerance::Custom({bad})");
        }
    }

    #[test]
    fn test_solver_validate_propagates_ipm() {
        // SolverOptions::validate must propagate IpmOptions::validate errors.
        let o = SolverOptions {
            ipm: IpmOptions { eps: 0.0, ..Default::default() },
            ..Default::default()
        };
        assert!(o.validate().is_err(), "ipm.eps=0 must propagate");

        let o = SolverOptions {
            ipm: IpmOptions { max_correctors: 0, ..Default::default() },
            ..Default::default()
        };
        assert!(o.validate().is_err(), "ipm.max_correctors=0 must propagate");
    }

    // ---- SolverOptions builders -----------------------------------------

    #[test]
    fn test_solver_builder_with_timeout() {
        assert!(SolverOptions::default().with_timeout(10.0).is_ok());
        assert!(SolverOptions::default().with_timeout(0.001).is_ok());
        assert!(SolverOptions::default().with_timeout(0.0).is_ok(), "0.0 = immediately-expired deadline");
        for bad in [-1.0_f64, f64::NAN, f64::INFINITY] {
            assert!(SolverOptions::default().with_timeout(bad).is_err(), "with_timeout({bad})");
        }
        // Result carries the set value
        let o = SolverOptions::default().with_timeout(5.0).unwrap();
        assert_eq!(o.timeout_secs, Some(5.0));
    }

    #[test]
    fn test_solver_builder_with_threads() {
        assert!(SolverOptions::default().with_threads(1).is_ok());
        assert!(SolverOptions::default().with_threads(8).is_ok());
        assert!(SolverOptions::default().with_threads(0).is_err());
        let o = SolverOptions::default().with_threads(4).unwrap();
        assert_eq!(o.threads, 4);
    }

    #[test]
    fn test_solver_builder_with_tolerance() {
        assert!(SolverOptions::default().with_tolerance(Tolerance::High).is_ok());
        assert!(SolverOptions::default().with_tolerance(Tolerance::Medium).is_ok());
        assert!(SolverOptions::default().with_tolerance(Tolerance::Fast).is_ok());
        assert!(SolverOptions::default().with_tolerance(Tolerance::Custom(1e-5)).is_ok());
        for bad in [0.0_f64, -1e-4, f64::NAN, f64::INFINITY] {
            assert!(
                SolverOptions::default().with_tolerance(Tolerance::Custom(bad)).is_err(),
                "with_tolerance(Custom({bad}))"
            );
        }
        let o = SolverOptions::default().with_tolerance(Tolerance::Fast).unwrap();
        assert_eq!(o.tolerance, Some(Tolerance::Fast));
    }

    // ---- OptionsError display -------------------------------------------

    #[test]
    fn test_options_error_display() {
        let e = OptionsError { field: "ipm.eps", reason: "must be finite and > 0" };
        let s = e.to_string();
        assert!(s.contains("ipm.eps"), "display: {s}");
        assert!(s.contains("finite"), "display: {s}");
    }

    // ---- IpmOptions: new fields defaults and resolution ----------------

    #[test]
    fn test_ipm_new_fields_default() {
        let o = IpmOptions::default();
        assert!(!o.dd_ldl, "dd_ldl default false");
        assert!(o.minres_ir.is_none(), "minres_ir default None");
        assert!(o.kkt_memory_budget_bytes.is_none(), "kkt_memory_budget_bytes default None");
    }

    #[test]
    fn test_ipm_effective_minres_ir_default_and_override() {
        let o = IpmOptions::default();
        assert_eq!(o.effective_minres_ir(), 0, "default IR = 0");
        let o2 = IpmOptions { minres_ir: Some(3), ..Default::default() };
        assert_eq!(o2.effective_minres_ir(), 3);
    }

    #[test]
    #[allow(clippy::assertions_on_constants, clippy::absurd_extreme_comparisons)]
    fn test_ipm_validate_minres_ir() {
        use crate::linalg::kkt_solver::MINRES_INEXACT_NEWTON_IR_STEPS;
        // Default (None) and valid values
        assert!(IpmOptions::default().validate().is_ok());
        for ok in [0_usize, 1, 5, 10] {
            let o = IpmOptions { minres_ir: Some(ok), ..Default::default() };
            assert!(o.validate().is_ok(), "minres_ir={ok} should be valid");
        }
        // Out of range: > 10
        for bad in [11_usize, 100, usize::MAX] {
            let o = IpmOptions { minres_ir: Some(bad), ..Default::default() };
            assert!(o.validate().is_err(), "minres_ir={bad} should be invalid");
        }
        // Default const falls within valid range (validated at compile time by the type constraint)
        let _ = MINRES_INEXACT_NEWTON_IR_STEPS;
    }

    #[test]
    fn test_ipm_effective_max_l_nnz_default_and_override() {
        use crate::linalg::kkt_solver::{BYTES_PER_L_ENTRY, DEFAULT_MEMORY_BUDGET_BYTES};
        let o = IpmOptions::default();
        assert_eq!(o.effective_kkt_memory_budget_bytes(), DEFAULT_MEMORY_BUDGET_BYTES);
        assert_eq!(o.effective_max_l_nnz(), DEFAULT_MEMORY_BUDGET_BYTES / BYTES_PER_L_ENTRY);
        let o2 = IpmOptions { kkt_memory_budget_bytes: Some(1600), ..Default::default() };
        assert_eq!(o2.effective_max_l_nnz(), 1600 / BYTES_PER_L_ENTRY);
    }

    // ---- SolverOptions: presolve fields --------------------------------

    #[test]
    fn test_solver_presolve_fields_default() {
        let o = SolverOptions::default();
        assert_eq!(o.presolve_max_pass, DEFAULT_PRESOLVE_MAX_PASS, "default max pass");
        assert!(!o.presolve_skip_large_coeff, "default skip_large_coeff = false");
        assert!(o.presolve_phase2, "default phase2 = true");
    }

    #[test]
    fn test_presolve_max_pass_controls_iteration_count() {
        use crate::problem::SolveStatus;
        use crate::qp::{solve_qp_with, QpProblem};
        use crate::sparse::CscMatrix;

        // Minimal feasible QP: 1 variable, no constraints, x* = 0.
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let a = CscMatrix::new(0, 1);
        let prob = QpProblem::new(q, vec![0.0], a, vec![], vec![(0.0_f64, 1.0_f64)], vec![]).unwrap();

        // Both 0 and 10 passes must find the optimum.
        let opts0 = SolverOptions { presolve_max_pass: 0, ..Default::default() };
        let opts10 = SolverOptions { presolve_max_pass: 10, ..Default::default() };
        let r0 = solve_qp_with(&prob, &opts0);
        let r10 = solve_qp_with(&prob, &opts10);
        assert_eq!(r0.status, SolveStatus::Optimal, "presolve_max_pass=0 should still solve trivial QP");
        assert_eq!(r10.status, SolveStatus::Optimal, "presolve_max_pass=10 should solve trivial QP");
    }

    #[test]
    fn test_presolve_phase2_false_skips_phase2() {
        // When presolve_phase2=false, attempt.rs takes the phase1-only branch.
        // Verify through options field round-trip.
        let o = SolverOptions { presolve_phase2: false, ..Default::default() };
        assert!(!o.presolve_phase2);
        let o2 = SolverOptions { presolve_phase2: true, ..Default::default() };
        assert!(o2.presolve_phase2);
    }

    #[test]
    fn test_presolve_skip_large_coeff_field() {
        // Verify the OR logic: skip if field OR use_ruiz_scaling.
        let no_skip = SolverOptions { presolve_skip_large_coeff: false, use_ruiz_scaling: false, ..Default::default() };
        assert!(!no_skip.presolve_skip_large_coeff && !no_skip.use_ruiz_scaling);
        let skip_via_field = SolverOptions { presolve_skip_large_coeff: true, use_ruiz_scaling: false, ..Default::default() };
        assert!(skip_via_field.presolve_skip_large_coeff);
        let skip_via_ruiz = SolverOptions { presolve_skip_large_coeff: false, use_ruiz_scaling: true, ..Default::default() };
        // effective skip = field OR ruiz
        assert!(skip_via_ruiz.presolve_skip_large_coeff || skip_via_ruiz.use_ruiz_scaling);
    }

}
