//! High-level algebraic modeling API for linear programs.
//!
//! # Example
//! ```
//! use otspot_model::{Model, constraint};
//!
//! let mut model = Model::new("production");
//! let x = model.add_var("x", 0.0, f64::INFINITY);
//! let y = model.add_var("y", 0.0, 10.0);
//! model.add_constraint(constraint!((2.0 * x + 3.0 * y) <= 12.0));
//! model.add_constraint(constraint!((x + y) >= 3.0));
//! model.minimize(x + 2.0 * y);
//! let result = model.solve().unwrap();
//! println!("x = {}", result[x]);
//! ```

use crate::constraint::{Constraint, ConstraintSense};
use crate::expression::Expression;
use crate::quad_expr::{quad_to_csc, QuadExpr};
use crate::variable::{VarKind, Variable};

use crate::variable::VariableDefinition;

use otspot_core::options::Tolerance;
use otspot_core::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot_core::sparse::CscMatrix;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::Index;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

// ---------------------------------------------------------------------------
// Optimization sense
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum OptimizationSense {
    Minimize,
    Maximize,
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// A linear programming model built using the algebraic modeling API.
pub struct Model {
    model_id: u64,
    name: Option<String>,
    variables: Vec<VariableDefinition>,
    constraints: Vec<Constraint>,
    objective: Option<Expression>,
    sense: OptimizationSense,
    /// Quadratic objective Q matrix for QP problems (None = LP mode).
    /// Convention: min 1/2 x^T Q x + c^T x  ("1/2あり" standard).
    quadratic_objective: Option<CscMatrix>,
    /// Error message from the most recent DSL quad conversion failure (NaN/OOB
    /// coefficient that `quad_to_csc` rejected).  Cleared at the **start** of
    /// every `apply_objective` call so a subsequent valid `minimize`/`maximize`
    /// always resets it.  Checked by `validate_objective` as the sole indicator
    /// that the current DSL quad is invalid.
    quad_dsl_error: Option<String>,
    /// Constant term extracted from the objective expression (e.g. the `+4.0`
    /// in `minimize(x*x - 4.0*x + 4.0)`).  Added to every `signed_obj` result
    /// after the maximize sign-flip, independent of `obj_offset`.
    /// Reset on every `minimize`/`maximize` call to avoid double-counting.
    obj_expr_constant: f64,
    invalid_inputs: BTreeMap<&'static str, String>,
    /// Timeout for QP solve in seconds (None = unlimited).
    timeout_secs: Option<f64>,
    /// 収束精度プリセット（None = デフォルト Medium = 1e-6）
    tolerance: Option<Tolerance>,
    /// Presolve 有効/無効（None = SolverOptions::default() に従う = true）
    presolve: Option<bool>,
    /// 並列 thread 上限。None = SolverOptions::default() = 1 (serial)。
    threads: Option<usize>,
    /// 目的関数定数オフセット (QP: 1/2 x^T Q x + c^T x + offset, LP: c^T x + offset)
    obj_offset: f64,
}

impl Model {
    /// Create a new, empty model with the given name.
    pub fn new(name: &str) -> Self {
        Model {
            model_id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            name: Some(name.to_string()),
            variables: Vec::new(),
            constraints: Vec::new(),
            objective: None,
            sense: OptimizationSense::Minimize,
            quadratic_objective: None,
            quad_dsl_error: None,
            obj_expr_constant: 0.0,
            invalid_inputs: BTreeMap::new(),
            timeout_secs: None,
            tolerance: None,
            presolve: None,
            threads: None,
            obj_offset: 0.0,
        }
    }

    /// Set a timeout for QP solve operations.
    pub fn set_timeout(&mut self, secs: f64) {
        if let Err(err) = self.try_set_timeout(secs) {
            self.record_input_error("timeout", err);
        }
    }

    /// Set a timeout for solve operations, returning an error for invalid input.
    pub fn try_set_timeout(&mut self, secs: f64) -> Result<&mut Self, ModelError> {
        validate_timeout(secs)?;
        self.timeout_secs = Some(secs);
        self.invalid_inputs.remove("timeout");
        Ok(self)
    }

    /// 並列 thread 上限を設定する。0 は 1 に補正、default は SolverOptions の 1。
    /// LP / QP / 非凸 multistart すべてに影響する共通設定。
    pub fn set_threads(&mut self, n: usize) -> &mut Self {
        self.threads = Some(n.max(1));
        self
    }

    /// 精度プリセットを設定する。
    pub fn set_tolerance(&mut self, tol: Tolerance) -> &mut Self {
        self.tolerance = Some(tol);
        self
    }

    /// Presolve の有効/無効を設定する（デフォルト: true）。
    pub fn set_presolve(&mut self, flag: bool) -> &mut Self {
        self.presolve = Some(flag);
        self
    }

    /// Add a decision variable to the model.
    ///
    /// Records an error (deferred to `solve()`) if `lb > ub` or either bound is NaN.
    /// Use [`try_add_var`](Self::try_add_var) to get an immediate `Result`.
    pub fn add_var(&mut self, name: &str, lb: f64, ub: f64) -> Variable {
        match validate_bounds(lb, ub) {
            Ok(()) => self.push_var(name, lb, ub, VarKind::Continuous),
            Err(err) => {
                self.record_input_error("variable_bounds", err);
                self.push_var(name, 0.0, 0.0, VarKind::Continuous)
            }
        }
    }

    /// Add a decision variable, returning an error for invalid bounds.
    pub fn try_add_var(&mut self, name: &str, lb: f64, ub: f64) -> Result<Variable, ModelError> {
        validate_bounds(lb, ub)?;
        Ok(self.push_var(name, lb, ub, VarKind::Continuous))
    }

    /// Add an integer decision variable (must take integral values within `[lb, ub]`).
    ///
    /// Records an error (deferred to `solve()`) if `lb > ub` or either bound is NaN.
    /// Use [`try_add_int_var`](Self::try_add_int_var) to get an immediate `Result`.
    ///
    /// Presence of any integer/binary variable routes `solve()` through the
    /// MILP/MIQP branch-and-bound solver instead of the continuous LP/QP path.
    pub fn add_int_var(&mut self, name: &str, lb: f64, ub: f64) -> Variable {
        match validate_bounds(lb, ub) {
            Ok(()) => self.push_var(name, lb, ub, VarKind::Integer),
            Err(err) => {
                self.record_input_error("variable_bounds", err);
                self.push_var(name, 0.0, 0.0, VarKind::Integer)
            }
        }
    }

    /// Add an integer decision variable, returning an error for invalid bounds.
    pub fn try_add_int_var(
        &mut self,
        name: &str,
        lb: f64,
        ub: f64,
    ) -> Result<Variable, ModelError> {
        validate_bounds(lb, ub)?;
        Ok(self.push_var(name, lb, ub, VarKind::Integer))
    }

    /// Add a binary decision variable (integer, fixed to the `{0, 1}` box).
    pub fn add_binary_var(&mut self, name: &str) -> Variable {
        self.push_var(name, 0.0, 1.0, VarKind::Binary)
    }

    /// Return the name of a variable as given to [`add_var`](Self::add_var).
    ///
    /// # Panics
    /// Panics if `var.index` is out of range. Passing a `Variable` from a
    /// different model is **unchecked** and may index into unrelated data.
    /// Use [`try_var_name`](Self::try_var_name) for a checked variant.
    pub fn var_name(&self, var: Variable) -> &str {
        &self.variables[var.index].name
    }

    /// Return the name of a variable, returning an error instead of panicking.
    ///
    /// Returns `Err` if:
    /// - `var` was created by a different model.
    /// - `var.index` is out of range for this model's variable list.
    pub fn try_var_name(&self, var: Variable) -> Result<&str, ModelError> {
        if var.model_id != self.model_id {
            return Err(ModelError::InvalidInput(
                "variable belongs to a different model".to_string(),
            ));
        }
        self.variables
            .get(var.index)
            .map(|v| v.name.as_str())
            .ok_or_else(|| {
                ModelError::InvalidInput(format!(
                    "variable index {} out of range (model has {} variables)",
                    var.index,
                    self.variables.len()
                ))
            })
    }

    fn push_var(&mut self, name: &str, lb: f64, ub: f64, kind: VarKind) -> Variable {
        let index = self.variables.len();
        self.variables.push(VariableDefinition {
            name: name.to_string(),
            lower_bound: lb,
            upper_bound: ub,
            kind,
        });
        Variable {
            index,
            model_id: self.model_id,
        }
    }

    /// Add a constraint to the model.
    pub fn add_constraint(&mut self, c: Constraint) -> &mut Self {
        if let Some(v) = c
            .lhs
            .coefficients
            .keys()
            .find(|v| v.model_id != self.model_id)
        {
            let msg = format!(
                "constraint lhs: var {} belongs to model {}, not this model ({})",
                v.index, v.model_id, self.model_id
            );
            self.record_input_error("constraint_cross_model", ModelError::InvalidInput(msg));
            return self;
        }
        self.constraints.push(c);
        self
    }

    /// Set the objective to minimize the given expression.
    ///
    /// Accepts any linear [`Expression`] or quadratic [`QuadExpr`] (e.g. `x * x`,
    /// `x * y`, `x.pow2()`).  When quadratic terms are present the Q matrix is
    /// built automatically using the 1/2 xᵀQx convention and the model is routed
    /// through the QP solver.
    pub fn minimize(&mut self, obj: impl Into<QuadExpr>) -> &mut Self {
        self.apply_objective(obj.into(), OptimizationSense::Minimize)
    }

    /// Set the objective to maximize the given expression.
    ///
    /// See [`minimize`](Self::minimize) for details on quadratic support.
    /// For maximization the Q matrix is negated internally (Q → -Q).
    pub fn maximize(&mut self, obj: impl Into<QuadExpr>) -> &mut Self {
        self.apply_objective(obj.into(), OptimizationSense::Maximize)
    }

    // -----------------------------------------------------------------------
    // Quad-objective state machine — three private chokepoints.
    //
    // The quad state pair (`quadratic_objective` / `quad_dsl_error`) MUST
    // only be mutated through the three functions below.  The companion
    // `validate_objective` pure-function checks the *current* stored state at
    // solve time so that setter-ordering bugs (P2-a … P2-i) cannot produce
    // silent wrong results.
    // -----------------------------------------------------------------------

    /// Validate the current objective state.  Called once in `solve()` before
    /// building the LP/QP/MIP problem so that the **actual objective being
    /// solved** is always verified, regardless of setter order.
    ///
    /// Checks (pure function — no side effects):
    /// - `quad_dsl_error` is None (no stale DSL quad conversion failure)
    /// - Linear coefficient values are finite
    /// - Linear variable `model_id`s match this model
    /// - `obj_expr_constant` is finite
    /// - Q matrix values (if present) are finite
    fn validate_objective(&self) -> Result<(), ModelError> {
        if let Some(msg) = &self.quad_dsl_error {
            return Err(ModelError::InvalidInput(msg.clone()));
        }
        let Some(obj) = self.objective.as_ref() else {
            return Ok(()); // NoObjective handled separately by caller
        };
        for (&var, &coef) in &obj.coefficients {
            if var.model_id != self.model_id {
                return Err(ModelError::InvalidInput(format!(
                    "objective: linear term (var {}) belongs to model {}, not this model ({})",
                    var.index, var.model_id, self.model_id
                )));
            }
            if !coef.is_finite() {
                return Err(ModelError::InvalidInput(format!(
                    "objective: non-finite linear coefficient for var {}: {coef}",
                    var.index
                )));
            }
        }
        if !self.obj_expr_constant.is_finite() {
            return Err(ModelError::InvalidInput(format!(
                "objective: non-finite constant term: {}",
                self.obj_expr_constant
            )));
        }
        if let Some(q) = &self.quadratic_objective {
            if let Some(&v) = q.values().iter().find(|v| !v.is_finite()) {
                return Err(ModelError::InvalidInput(format!(
                    "objective: non-finite Q coefficient: {v}"
                )));
            }
        }
        Ok(())
    }

    /// Install a Q matrix and clear any stale DSL quad error.
    fn install_quad(&mut self, q: CscMatrix) {
        self.quadratic_objective = Some(q);
        self.quad_dsl_error = None;
    }

    /// Record a DSL quad failure.  Clears Q and records the error for
    /// `validate_objective` to surface at solve time.
    fn fail_dsl_quad(&mut self, msg: String) {
        self.quadratic_objective = None;
        self.quad_dsl_error = Some(msg);
    }

    /// Transition to a pure-linear objective: clear any stored Q matrix.
    fn replace_with_linear_objective(&mut self) {
        self.quadratic_objective = None;
    }

    /// Return an error message if `q` contains any variable from a different
    /// model, checking **both** linear coefficients and quad pairs (P2-d/P2-f).
    /// Returns `None` when all variables belong to this model.
    fn find_foreign_var_error(&self, q: &QuadExpr) -> Option<String> {
        // Linear part: coefficient keys are Variable structs with model_id.
        // The c-vector build reconstructs Variables with self.model_id, so
        // foreign vars would be silently dropped — detect and reject instead.
        if let Some(&v) = q
            .linear
            .coefficients
            .keys()
            .find(|v| v.model_id != self.model_id)
        {
            return Some(format!(
                "linear term (var {}) belongs to model {}, not this model ({})",
                v.index, v.model_id, self.model_id
            ));
        }
        // Quad part
        if let Some(&(va, vb)) = q
            .quad
            .keys()
            .find(|(va, vb)| va.model_id != self.model_id || vb.model_id != self.model_id)
        {
            return Some(format!(
                "quad term ({},{}) belongs to model(s) {}/{}, not this model ({})",
                va.index, vb.index, va.model_id, vb.model_id, self.model_id
            ));
        }
        None
    }

    /// Shared implementation for `minimize`/`maximize`.
    fn apply_objective(&mut self, q: QuadExpr, sense: OptimizationSense) -> &mut Self {
        // Clear any stale DSL quad error from a previous call.  This is the
        // key invariant: every `minimize`/`maximize` call starts with a clean
        // slate so that a subsequent valid quad always replaces an earlier
        // invalid one (P2-i root fix).
        self.quad_dsl_error = None;
        // Validate ALL variables (linear + quad) up front so foreign vars are
        // always rejected, whether the expression is pure-linear, pure-quad, or
        // mixed (P2-d: quad-only; P2-f: linear part was silently dropped).
        if let Some(msg) = self.find_foreign_var_error(&q) {
            self.fail_dsl_quad(msg);
        } else if !q.quad.is_empty() {
            match quad_to_csc(&q.quad, self.variables.len()) {
                Ok(csc) => self.install_quad(csc),
                Err(msg) => self.fail_dsl_quad(msg),
            }
        } else {
            self.replace_with_linear_objective();
        }
        // Fold the objective constant (e.g. `+4.0` in `x*x - 4x + 4.0`) into
        // obj_expr_constant.  Reset on every call so re-minimize never
        // double-counts.
        self.obj_expr_constant = q.linear.constant;
        self.objective = Some(q.linear);
        self.sense = sense;
        self
    }

    /// Set the quadratic objective Q matrix for QP problems.
    ///
    /// **Convention** ("1/2あり"): the objective is min 1/2 x^T Q x + c^T x
    /// 目的関数の定数オフセットを設定する。
    /// `objective_value = (1/2 x^T Q x +) c^T x + offset` として最終結果に加算される。
    pub fn set_obj_offset(&mut self, offset: f64) -> &mut Self {
        self.obj_offset = offset;
        self
    }

    /// Solve the model and return the result.
    ///
    /// # Errors
    /// * `ModelError::NoObjective` if `minimize` or `maximize` was not called.
    /// * `ModelError::SolveError` if the solver returns Infeasible or Unbounded.
    pub fn solve(&mut self) -> Result<ModelResult, ModelError> {
        if let Some(msg) = self.invalid_inputs.values().next() {
            return Err(ModelError::InvalidInput(msg.clone()));
        }

        let obj_expr = self.objective.as_ref().ok_or(ModelError::NoObjective)?;

        // Validate the current objective against the actual stored state.
        // This is the primary defense for all P2-* objective bugs: foreign
        // variables, non-finite coefficients, non-finite constant, non-finite Q.
        // Called here so it covers LP, QP, and MIP build paths with one check.
        self.validate_objective()?;

        let num_vars = self.variables.len();
        let num_constraints = self.constraints.len();

        // --- Build objective vector c ---
        let mid = self.model_id;
        let mut c: Vec<f64> = (0..num_vars)
            .map(|i| {
                obj_expr.coefficient(Variable {
                    index: i,
                    model_id: mid,
                })
            })
            .collect();

        // For maximization, negate c (solver minimizes by default)
        if self.sense == OptimizationSense::Maximize {
            for ci in &mut c {
                *ci = -*ci;
            }
        }

        // --- Build constraint matrix A (triplets) ---
        let mut trip_rows: Vec<usize> = Vec::new();
        let mut trip_cols: Vec<usize> = Vec::new();
        let mut trip_vals: Vec<f64> = Vec::new();
        let mut b: Vec<f64> = Vec::new();
        let mut constraint_types: Vec<ConstraintType> = Vec::new();

        for (i, con) in self.constraints.iter().enumerate() {
            for (&var, &coeff) in &con.lhs.coefficients {
                trip_rows.push(i);
                trip_cols.push(var.index);
                trip_vals.push(coeff);
            }
            // lhs has no constant (normalized), rhs is con.rhs
            b.push(con.rhs);
            constraint_types.push(match con.sense {
                ConstraintSense::Le => ConstraintType::Le,
                ConstraintSense::Ge => ConstraintType::Ge,
                ConstraintSense::Eq => ConstraintType::Eq,
            });
        }

        // Handle zero-constraint case (empty matrix)
        let a = if num_constraints == 0 {
            CscMatrix::new(0, num_vars)
        } else {
            CscMatrix::from_triplets(
                &trip_rows,
                &trip_cols,
                &trip_vals,
                num_constraints,
                num_vars,
            )
            .map_err(|e| ModelError::Internal(e.to_string()))?
        };

        // --- Variable bounds ---
        let bounds: Vec<(f64, f64)> = self
            .variables
            .iter()
            .map(|v| (v.lower_bound, v.upper_bound))
            .collect();

        // --- MIP path: any integer/binary variable routes through branch-and-bound ---
        // (degenerate "no integer var" falls through to the existing LP/QP paths below.)
        let integer_vars: Vec<usize> = self
            .variables
            .iter()
            .enumerate()
            .filter(|(_, v)| v.kind != VarKind::Continuous)
            .map(|(i, _)| i)
            .collect();
        if !integer_vars.is_empty() {
            return self.solve_mip_internal(c, a, b, constraint_types, bounds, integer_vars);
        }

        // --- QP path ---
        if let Some(ref q_orig) = self.quadratic_objective.clone() {
            return self.solve_qp_internal(c, bounds, q_orig.clone(), num_constraints);
        }

        // --- LP path (existing) ---
        let problem = LpProblem::new_general(c, a, b, constraint_types, bounds, self.name.clone())
            .map_err(map_lp_build_err)?;

        let mut lp_opts = otspot_core::options::SolverOptions::default();
        if let Some(t) = self.timeout_secs {
            lp_opts.timeout_secs = Some(t);
        }
        if let Some(tol) = self.tolerance {
            lp_opts.tolerance = Some(tol);
        }
        if let Some(flag) = self.presolve {
            lp_opts.presolve = flag;
        }
        if let Some(n) = self.threads {
            lp_opts.threads = n;
        }
        let solver_result = otspot_core::lp::solve_lp_with(&problem, &lp_opts);

        // SolverResult の dual/rc/slack は extract_dual_info によって
        // 元の制約空間 (Eq/Ge/Le) と変数空間 (bounds 込み) で復元済み。
        let lp_extras = |sr: &otspot_core::problem::SolverResult| {
            let dual = if sr.dual_solution.is_empty() {
                None
            } else {
                Some(sr.dual_solution.clone())
            };
            let rc = if sr.reduced_costs.is_empty() {
                None
            } else {
                Some(sr.reduced_costs.clone())
            };
            let slack = if sr.slack.is_empty() {
                None
            } else {
                Some(sr.slack.clone())
            };
            (dual, rc, slack)
        };

        let signed_obj = |raw: f64| -> f64 {
            let oriented = if self.sense == OptimizationSense::Maximize {
                -raw
            } else {
                raw
            };
            oriented + self.obj_offset + self.obj_expr_constant
        };
        let lp_model_id = self.model_id;
        let build_ok = |sr: otspot_core::problem::SolverResult| {
            let (dual, rc, slack) = lp_extras(&sr);
            let status = sr.status.clone();
            ModelResult {
                model_id: lp_model_id,
                status: status.clone(),
                proof: SolutionProof::from_status(&status),
                objective_value: signed_obj(sr.objective),
                solution: sr.solution,
                dual_solution: dual,
                reduced_costs: rc,
                slack,
                bound_duals: sr.bound_duals,
                stats: sr.stats,
            }
        };

        match solver_result.status {
            SolveStatus::Optimal => Ok(build_ok(solver_result)),
            SolveStatus::MaxIterations => {
                if solver_result.solution.is_empty() {
                    Err(ModelError::SolveError(SolveError::MaxIterations))
                } else {
                    Ok(build_ok(solver_result))
                }
            }
            SolveStatus::SuboptimalSolution => {
                if solver_result.solution.is_empty() {
                    Err(ModelError::Timeout)
                } else {
                    Ok(build_ok(solver_result))
                }
            }
            SolveStatus::Timeout => Err(ModelError::Timeout),
            SolveStatus::LocallyOptimal
            | SolveStatus::NonconvexLocal
            | SolveStatus::NonconvexGlobal => Err(ModelError::Internal(
                "Unexpected nonconvex status on LP path".to_string(),
            )),
            s => Err(classify_status_error(s)
                .unwrap_or_else(|| ModelError::Internal("unhandled LP status".to_string()))),
        }
    }

    /// Build a `QpProblem` from the model (constraint matrix + maximize Q→-Q
    /// negation + offset removal). Shared by the QP and MIQP paths. `c` is already
    /// negated by `solve()` for maximize.
    fn build_qp_problem(
        &self,
        c: Vec<f64>,
        bounds: Vec<(f64, f64)>,
        q_orig: CscMatrix,
    ) -> Result<otspot_core::qp::QpProblem, ModelError> {
        use otspot_core::qp::QpProblem;

        let num_vars = self.variables.len();

        // maximize QP: negate Q (Q→-Q), c is already negated by solve()
        let qp_q = if self.sense == OptimizationSense::Maximize {
            q_orig.scale_values(-1.0)
        } else {
            q_orig
        };

        // --- 制約型変換: Le/Ge/Eq をそのまま QpProblem に渡す ---
        // QP/IPMソルバーは ConstraintType をネイティブに処理する（to_all_le() 廃止済み）。
        let mut qp_trip_rows: Vec<usize> = Vec::new();
        let mut qp_trip_cols: Vec<usize> = Vec::new();
        let mut qp_trip_vals: Vec<f64> = Vec::new();
        let mut qp_b: Vec<f64> = Vec::new();
        let mut qp_constraint_types: Vec<ConstraintType> = Vec::new();
        let mut qp_row = 0usize;

        for con in &self.constraints {
            for (&var, &coeff) in &con.lhs.coefficients {
                qp_trip_rows.push(qp_row);
                qp_trip_cols.push(var.index);
                qp_trip_vals.push(coeff);
            }
            qp_b.push(con.rhs);
            qp_constraint_types.push(match con.sense {
                ConstraintSense::Le => ConstraintType::Le,
                ConstraintSense::Ge => ConstraintType::Ge,
                ConstraintSense::Eq => ConstraintType::Eq,
            });
            qp_row += 1;
        }

        let m_qp = qp_row;
        let qp_a = if m_qp == 0 {
            CscMatrix::new(0, num_vars)
        } else {
            CscMatrix::from_triplets(&qp_trip_rows, &qp_trip_cols, &qp_trip_vals, m_qp, num_vars)
                .map_err(|e| ModelError::Internal(e.to_string()))?
        };

        let mut qp_problem = QpProblem::new(qp_q, c, qp_a, qp_b, bounds, qp_constraint_types)
            .map_err(map_qp_build_err)?;
        // offset は signed_obj で post-solve 加算するため solver には渡さない。
        qp_problem.obj_offset = 0.0;
        Ok(qp_problem)
    }

    /// QP内部求解ロジック（QpProblem構築・求解・結果変換）
    fn solve_qp_internal(
        &self,
        c: Vec<f64>,
        bounds: Vec<(f64, f64)>,
        q_orig: CscMatrix,
        num_model_constraints: usize,
    ) -> Result<ModelResult, ModelError> {
        let qp_problem = self.build_qp_problem(c, bounds, q_orig)?;

        let mut opts = otspot_core::options::SolverOptions::default();
        if let Some(t) = self.timeout_secs {
            opts.timeout_secs = Some(t);
        }
        if let Some(tol) = self.tolerance {
            opts.tolerance = Some(tol);
        }
        if let Some(n) = self.threads {
            opts.threads = n;
        }
        let qp_result = otspot_core::qp::solve_qp_with(&qp_problem, &opts);
        let qp_stats = qp_result.stats.clone();

        // dual_solution: Le=そのまま / Ge=符号反転済み / Eq=μ1-μ2 折り畳み済み。
        let fold_dual = |sol: &[f64]| -> Option<Vec<f64>> {
            if sol.len() == num_model_constraints {
                Some(sol.to_vec())
            } else if !sol.is_empty() && num_model_constraints > 0 {
                let take = num_model_constraints.min(sol.len());
                Some(sol[..take].to_vec())
            } else {
                None
            }
        };

        let signed_obj = |raw: f64| -> f64 {
            let oriented = if self.sense == OptimizationSense::Maximize {
                -raw
            } else {
                raw
            };
            oriented + self.obj_offset + self.obj_expr_constant
        };
        let qp_model_id = self.model_id;
        let build_ok = |status: SolveStatus,
                        raw_obj: f64,
                        sol: Vec<f64>,
                        dual: Option<Vec<f64>>,
                        bd: Vec<f64>| {
            let proof = SolutionProof::from_status(&status);
            ModelResult {
                model_id: qp_model_id,
                status,
                proof,
                objective_value: signed_obj(raw_obj),
                solution: sol,
                dual_solution: dual,
                reduced_costs: None,
                slack: None,
                bound_duals: bd,
                stats: qp_stats.clone(),
            }
        };

        match qp_result.status {
            SolveStatus::Optimal => Ok(build_ok(
                SolveStatus::Optimal,
                qp_result.objective,
                qp_result.solution,
                fold_dual(&qp_result.dual_solution),
                qp_result.bound_duals,
            )),
            SolveStatus::MaxIterations => {
                if qp_result.solution.is_empty() {
                    Err(ModelError::SolveError(SolveError::MaxIterations))
                } else {
                    Ok(build_ok(
                        SolveStatus::MaxIterations,
                        qp_result.objective,
                        qp_result.solution,
                        fold_dual(&qp_result.dual_solution),
                        qp_result.bound_duals,
                    ))
                }
            }
            SolveStatus::SuboptimalSolution => {
                // apply_api_boundary_conversion が通常 Optimal/Timeout に変換済み。予備パス。
                if qp_result.solution.is_empty() {
                    Err(ModelError::Timeout)
                } else {
                    Ok(build_ok(
                        SolveStatus::SuboptimalSolution,
                        qp_result.objective,
                        qp_result.solution,
                        fold_dual(&qp_result.dual_solution),
                        qp_result.bound_duals,
                    ))
                }
            }
            SolveStatus::Timeout => Err(ModelError::Timeout),
            // LocallyOptimal / NonconvexLocal / NonconvexGlobal: global proof なしだが解あり。
            // Model API では caller が status で品質を判断する。NonconvexGlobal は global 証明済。
            status @ (SolveStatus::LocallyOptimal
            | SolveStatus::NonconvexLocal
            | SolveStatus::NonconvexGlobal) => Ok(build_ok(
                status,
                qp_result.objective,
                qp_result.solution,
                fold_dual(&qp_result.dual_solution),
                qp_result.bound_duals,
            )),
            s => Err(classify_status_error(s)
                .unwrap_or_else(|| ModelError::Internal("unhandled QP status".to_string()))),
        }
    }

    /// MILP/MIQP 内部求解: 整数変数があるとき `solve()` から dispatch される。
    ///
    /// 二次目的なし → MILP (各 B&B node で LP relaxation)。二次目的あり → **凸** MIQP
    /// (各 node で QP relaxation)。非凸 (Q 非PSD) は `solve_miqp` が `NonConvex` を返し、
    /// ここで明示エラーに変換する (silent 誤答禁止)。
    fn solve_mip_internal(
        &self,
        c: Vec<f64>,
        a: CscMatrix,
        b: Vec<f64>,
        constraint_types: Vec<ConstraintType>,
        bounds: Vec<(f64, f64)>,
        integer_vars: Vec<usize>,
    ) -> Result<ModelResult, ModelError> {
        let mut opts = otspot_core::options::SolverOptions::default();
        if let Some(t) = self.timeout_secs {
            opts.timeout_secs = Some(t);
        }
        if let Some(tol) = self.tolerance {
            opts.tolerance = Some(tol);
        }
        if let Some(n) = self.threads {
            opts.threads = n;
        }
        let cfg = otspot_core::options::MipConfig::default();

        let result = if let Some(ref q_orig) = self.quadratic_objective.clone() {
            // MIQP: convex QP relaxation per node.
            let qp = self.build_qp_problem(c, bounds, q_orig.clone())?;
            let miqp = otspot_core::mip::MiqpProblem::new(qp, integer_vars.clone())
                .map_err(|e| ModelError::Internal(e.to_string()))?;
            otspot_core::mip::solve_miqp(&miqp, &opts, &cfg)
        } else {
            // MILP: LP relaxation per node.
            if let Some(flag) = self.presolve {
                opts.presolve = flag;
            }
            let lp = LpProblem::new_general(c, a, b, constraint_types, bounds, self.name.clone())
                .map_err(map_lp_build_err)?;
            let milp = otspot_core::mip::MilpProblem::new(lp, integer_vars.clone())
                .map_err(|e| ModelError::Internal(e.to_string()))?;
            otspot_core::mip::solve_milp(&milp, &opts, &cfg)
        };

        self.finish_mip(result, &integer_vars)
    }

    /// Convert a MIP `SolverResult` to a `ModelResult`: apply objective sign /
    /// offset, round integer components, and map the status. Shared by MILP/MIQP.
    fn finish_mip(
        &self,
        result: otspot_core::problem::SolverResult,
        integer_vars: &[usize],
    ) -> Result<ModelResult, ModelError> {
        let signed_obj = |raw: f64| -> f64 {
            let oriented = if self.sense == OptimizationSense::Maximize {
                -raw
            } else {
                raw
            };
            oriented + self.obj_offset + self.obj_expr_constant
        };

        // 整数変数成分を厳密整数に丸める (relaxation 解の 1e-6 級 noise を除去)。
        let round_integers = |mut sol: Vec<f64>| -> Vec<f64> {
            for &j in integer_vars {
                if j < sol.len() {
                    sol[j] = sol[j].round();
                }
            }
            sol
        };

        let mip_model_id = self.model_id;
        let build_ok = |sr: otspot_core::problem::SolverResult| {
            let status = sr.status.clone();
            ModelResult {
                model_id: mip_model_id,
                status: status.clone(),
                proof: SolutionProof::from_status(&status),
                objective_value: signed_obj(sr.objective),
                solution: round_integers(sr.solution),
                dual_solution: None,
                reduced_costs: None,
                slack: None,
                bound_duals: vec![],
                stats: sr.stats,
            }
        };

        match result.status {
            SolveStatus::Optimal => Ok(build_ok(result)),
            SolveStatus::Timeout => {
                // 打ち切りでも incumbent (整数実行可能解) があれば解を返す。
                if result.solution.is_empty() {
                    Err(ModelError::Timeout)
                } else {
                    Ok(build_ok(result))
                }
            }
            SolveStatus::SuboptimalSolution | SolveStatus::MaxIterations => {
                if result.solution.is_empty() {
                    Err(ModelError::SolveError(SolveError::MaxIterations))
                } else {
                    Ok(build_ok(result))
                }
            }
            SolveStatus::LocallyOptimal
            | SolveStatus::NonconvexLocal
            | SolveStatus::NonconvexGlobal => Err(ModelError::Internal(
                "Unexpected nonconvex status on MIP path".to_string(),
            )),
            s => Err(classify_status_error(s)
                .unwrap_or_else(|| ModelError::Internal("unhandled MIP status".to_string()))),
        }
    }

    fn record_input_error(&mut self, key: &'static str, err: ModelError) {
        let msg = match err {
            ModelError::InvalidInput(msg) => msg,
            other => other.to_string(),
        };
        self.invalid_inputs.insert(key, msg);
    }
}

/// Maps `SolveStatus` variants that always produce an error to the corresponding
/// `ModelError`. Returns `None` for statuses that may produce a successful result
/// depending on context (e.g. `MaxIterations` with a non-empty solution).
///
/// Used by all three dispatch paths (LP / QP / MIP) to eliminate duplicated
/// match arms for `Infeasible`, `Unbounded`, `NumericalError`, `NonConvex`, and
/// `NotSupported`.
fn classify_status_error(status: SolveStatus) -> Option<ModelError> {
    match status {
        SolveStatus::Infeasible => Some(ModelError::SolveError(SolveError::Infeasible)),
        SolveStatus::Unbounded => Some(ModelError::SolveError(SolveError::Unbounded)),
        SolveStatus::NumericalError => Some(ModelError::SolveError(SolveError::NumericalError)),
        SolveStatus::NonConvex(msg) => Some(ModelError::NonConvex(msg)),
        SolveStatus::NotSupported(msg) => Some(ModelError::NotSupported(msg)),
        // Ok-capable or context-dependent statuses: returned as `None` so the
        // caller's explicit match arms decide, never as a hard error here.
        SolveStatus::Optimal
        | SolveStatus::LocallyOptimal
        | SolveStatus::MaxIterations
        | SolveStatus::SuboptimalSolution
        | SolveStatus::Timeout
        | SolveStatus::NonconvexLocal
        | SolveStatus::NonconvexGlobal => None,
        // `SolveStatus` is `#[non_exhaustive]` and lives in another crate
        // (otspot-core), so a wildcard is mandatory here — the compile-time
        // exhaustiveness guard the same-crate version had is lost to the crate
        // split. Returning `None` is safe: every caller wraps this in
        // `unwrap_or_else(|| ModelError::Internal(..))`, so an unknown future
        // variant surfaces as `Err`, never a silent success / false Optimal.
        _ => None,
    }
}

fn validate_timeout(secs: f64) -> Result<(), ModelError> {
    if secs.is_finite() && secs >= 0.0 {
        Ok(())
    } else {
        Err(ModelError::InvalidInput(format!(
            "timeout must be finite and non-negative, got {secs}"
        )))
    }
}

/// Map a low-level `LpProblem`/`SolverError` construction failure to a `ModelError`.
///
/// Non-finite coefficients and invalid bounds are user-input errors → `InvalidInput`;
/// dimension/structural failures (which the Model builder controls) stay `Internal`.
fn map_lp_build_err(e: otspot_core::error::SolverError) -> ModelError {
    use otspot_core::error::SolverError;
    match e {
        SolverError::NonFiniteCoefficient { .. } | SolverError::InvalidBounds { .. } => {
            ModelError::InvalidInput(e.to_string())
        }
        _ => ModelError::Internal(e.to_string()),
    }
}

/// Map a low-level `QpProblem` construction failure to a `ModelError` (see [`map_lp_build_err`]).
fn map_qp_build_err(e: otspot_core::qp::QpProblemError) -> ModelError {
    use otspot_core::qp::QpProblemError;
    match e {
        QpProblemError::NonFiniteCoefficient { .. }
        | QpProblemError::InvalidBounds { .. }
        | QpProblemError::TripletIndexOutOfBounds { .. } => ModelError::InvalidInput(e.to_string()),
        QpProblemError::DimensionMismatch(_) => ModelError::Internal(e.to_string()),
        // #[non_exhaustive]: wildcard required for cross-crate matching.
        _ => ModelError::Internal(e.to_string()),
    }
}

fn validate_bounds(lb: f64, ub: f64) -> Result<(), ModelError> {
    if lb.is_nan() || ub.is_nan() {
        return Err(ModelError::InvalidInput(format!(
            "variable bounds must not be NaN: lb={lb}, ub={ub}"
        )));
    }
    if lb > ub {
        return Err(ModelError::InvalidInput(format!(
            "variable lower bound {lb} exceeds upper bound {ub}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ModelResult
// ---------------------------------------------------------------------------

/// What kind of optimality proof backs a successful [`ModelResult`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolutionProof {
    /// A global optimum was proven.
    GlobalOptimal,
    /// A local KKT point was returned without a global proof.
    LocalOptimal,
    /// A feasible incumbent is available, but optimality was not proven.
    FeasibleUnproven,
}

impl SolutionProof {
    fn from_status(status: &SolveStatus) -> Self {
        match status {
            SolveStatus::Optimal | SolveStatus::NonconvexGlobal => SolutionProof::GlobalOptimal,
            SolveStatus::LocallyOptimal => SolutionProof::LocalOptimal,
            SolveStatus::MaxIterations
            | SolveStatus::SuboptimalSolution
            | SolveStatus::Timeout
            | SolveStatus::NonconvexLocal => SolutionProof::FeasibleUnproven,
            // These error statuses never reach build_ok — all three paths (LP, QP, MIP)
            // return Err(...) for them before calling build_ok. The conservative fallback
            // below guards against future regressions; the debug_assert catches them in tests.
            SolveStatus::Infeasible
            | SolveStatus::Unbounded
            | SolveStatus::NumericalError
            | SolveStatus::NonConvex(_)
            | SolveStatus::NotSupported(_) => {
                debug_assert!(
                    false,
                    "from_status called with error status {:?}: this arm is unreachable from build_ok",
                    status
                );
                SolutionProof::FeasibleUnproven
            }
            // #[non_exhaustive]: wildcard required for cross-crate matching.
            _ => SolutionProof::FeasibleUnproven,
        }
    }
}

/// The result of a successful solve.
#[derive(Debug, Clone)]
pub struct ModelResult {
    model_id: u64,
    /// Solver termination status associated with this returned solution.
    ///
    /// Only success-domain variants occur here (`Optimal`, `LocallyOptimal`,
    /// `NonconvexLocal`, `NonconvexGlobal`, `MaxIterations`, `SuboptimalSolution`,
    /// `Timeout`); error variants surface as [`ModelError`] instead. Match on
    /// [`ModelResult::proof`] for the optimality guarantee.
    pub status: SolveStatus,
    /// Optimality proof quality for this returned solution.
    pub proof: SolutionProof,
    /// Optimal objective value.
    pub objective_value: f64,
    /// Primal solution vector (indexed by variable index).
    solution: Vec<f64>,
    /// Dual solution (shadow prices), if available.
    pub dual_solution: Option<Vec<f64>>,
    /// Reduced costs, if available.
    ///
    /// 通常は `c − A^T y`。例外: presolve の bound-tightened-fixed 列が *元の* 上下限に
    /// 着地した場合、`reduced_costs[j]` には bound dual (μ_lb / μ_ub) が暗黙吸収され、
    /// raw `c − A^T y` ではなく at-lb で `max(·, 0)` / at-ub で `min(·, 0)` の clamp 値
    /// となる (presolve/postsolve.rs::BoundAbsorb)。LP path で `bound_duals` は default
    /// 空のため μ を分離取得することはできない (QP path のみ populate)。
    pub reduced_costs: Option<Vec<f64>>,
    /// Constraint slacks, if available.
    pub slack: Option<Vec<f64>>,
    /// Variable bound dual values (QP path).
    /// Layout: `[lb_dual for each var with finite lb, ub_dual for each var with finite ub]`
    /// Empty when not provided by the solver.
    pub bound_duals: Vec<f64>,
    /// Per-solve routing and warm-start statistics (race-free, per-result).
    pub stats: otspot_core::problem::SolveStats,
}

// ---------------------------------------------------------------------------
// Test-only state inspection (state-machine invariant assertions).
// ---------------------------------------------------------------------------
#[cfg(test)]
impl Model {
    /// True if there is a pending DSL-quad error (stale if a later setter
    /// should have cleared it).
    pub(crate) fn has_quad_dsl_error(&self) -> bool {
        self.quad_dsl_error.is_some()
    }

    /// True if a quadratic objective is currently installed (always DSL-owned).
    pub(crate) fn is_quad_via_dsl(&self) -> bool {
        self.quadratic_objective.is_some()
    }

    /// True if any Q matrix is currently stored.
    pub(crate) fn has_quadratic_objective(&self) -> bool {
        self.quadratic_objective.is_some()
    }
}

impl ModelResult {
    /// Get the primal value of a variable.
    ///
    /// # Panics
    /// Panics if the variable index is out of range. Use [`try_value`](Self::try_value)
    /// to handle this case gracefully.
    pub fn value(&self, var: Variable) -> f64 {
        self.solution[var.index]
    }

    /// Get the primal value of a variable, returning an error instead of panicking.
    ///
    /// Returns `Err` if:
    /// - `var` was created by a different model than the one that produced this result.
    /// - `var.index` is out of range for the solution vector.
    pub fn try_value(&self, var: Variable) -> Result<f64, ModelError> {
        if var.model_id != self.model_id {
            return Err(ModelError::InvalidInput(
                "variable belongs to a different model".to_string(),
            ));
        }
        self.solution.get(var.index).copied().ok_or_else(|| {
            ModelError::InvalidInput(format!(
                "variable index {} out of range (solution length {})",
                var.index,
                self.solution.len()
            ))
        })
    }

    /// Get the optimal objective value.
    pub fn objective(&self) -> f64 {
        self.objective_value
    }

    /// Returns true when the solver proved global optimality for this result.
    pub fn has_global_optimality_proof(&self) -> bool {
        self.proof == SolutionProof::GlobalOptimal
    }
}

/// Index a `ModelResult` by `Variable` to get the primal solution value.
///
/// # Example
/// ```rust,no_run
/// # use otspot_model::Model;
/// # let mut model = Model::new("demo");
/// # let x = model.add_var("x", 0.0, f64::INFINITY);
/// # model.minimize(x);
/// # let result = model.solve().unwrap();
/// println!("x = {}", result[x]);
/// ```
impl Index<Variable> for ModelResult {
    type Output = f64;
    fn index(&self, var: Variable) -> &f64 {
        &self.solution[var.index]
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Solver termination status for QP/LP solve operations.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum SolveError {
    /// The problem has no feasible solution.
    Infeasible,
    /// The problem is unbounded.
    Unbounded,
    /// Solver reached the iteration cap before converging (no usable solution).
    MaxIterations,
    /// Solver aborted due to numerical breakdown (no usable solution).
    NumericalError,
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolveError::Infeasible => write!(f, "Problem is infeasible"),
            SolveError::Unbounded => write!(f, "Problem is unbounded"),
            SolveError::MaxIterations => {
                write!(f, "Max iterations reached without optimal solution")
            }
            SolveError::NumericalError => write!(f, "Numerical breakdown during solve"),
        }
    }
}

/// Errors that can occur when building or solving a `Model`.
#[non_exhaustive]
#[derive(Debug)]
pub enum ModelError {
    /// `solve()` was called before `minimize()` or `maximize()`.
    NoObjective,
    /// A modeling API input was invalid.
    InvalidInput(String),
    /// The solver returned a non-optimal status.
    SolveError(SolveError),
    /// Solver timed out before finding an optimal solution.
    Timeout,
    /// Problem has a non-convex (indefinite) objective; global optimality cannot
    /// be guaranteed via IPM. Use `solve_qp_global` for non-convex continuous QP.
    NonConvex(String),
    /// Problem type is not supported by this solver (e.g. QCQP).
    NotSupported(String),
    /// An unexpected internal error (bug or invariant violation).
    Internal(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::NoObjective => write!(
                f,
                "No objective function defined. Call model.minimize() or model.maximize() before solve()."
            ),
            ModelError::InvalidInput(msg) => write!(f, "Invalid model input: {}", msg),
            ModelError::SolveError(e) => write!(f, "Solve failed: {}", e),
            ModelError::Timeout => write!(f, "Solver timed out"),
            ModelError::NonConvex(msg) => write!(f, "Non-convex problem: {}", msg),
            ModelError::NotSupported(msg) => write!(f, "Not supported: {}", msg),
            ModelError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for ModelError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{classify_status_error, Model, ModelError, SolutionProof, SolveError};
    use crate::variable::Variable;
    use otspot_core::problem::SolveStatus;

    // concurrent solver での許容誤差（IPM/IP-PMM 並列実行）
    const EPS: f64 = 2e-3;

    fn assert_close(a: f64, b: f64, name: &str) {
        assert!(
            (a - b).abs() < EPS,
            "{}: expected {:.8}, got {:.8}",
            name,
            b,
            a
        );
    }

    /// Helper: build the classic 2-variable LP:
    ///   min  x + 2y
    ///   s.t. 2x + 3y <= 12
    ///        x + y  >= 3
    ///        x in [0, inf), y in [0, 10]
    /// Optimal: x=3, y=0 → obj=3
    fn basic_model() -> (Model, Variable, Variable) {
        let mut model = Model::new("basic");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, 10.0);
        // Use method API for complex expressions
        model.add_constraint((2.0 * x + 3.0 * y).leq(12.0));
        model.add_constraint((x + y).geq(3.0));
        model.minimize(x + 2.0 * y);
        (model, x, y)
    }

    // -----------------------------------------------------------------------
    // Test 1: Basic LP – 3-variable, 3-constraint problem
    // -----------------------------------------------------------------------
    #[test]
    fn test_basic_lp_3var_3con() {
        // min  x + 2y + 3z
        // s.t. x + y + z >= 6
        //      x + 2y    <= 10
        //      y + z     >= 4
        //      x,y,z in [0, inf)
        let mut model = Model::new("3var");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        let z = model.add_var("z", 0.0, f64::INFINITY);

        // Use method API (complex LHS)
        model.add_constraint((x + y + z).geq(6.0));
        model.add_constraint((x + 2.0 * y).leq(10.0));
        model.add_constraint((y + z).geq(4.0));
        model.minimize(x + 2.0 * y + 3.0 * z);

        let result = model.solve().unwrap();
        // Verify feasibility: x+y+z >= 6, y+z >= 4
        assert!(result[x] + result[y] + result[z] >= 6.0 - 1e-6);
        assert!(result[y] + result[z] >= 4.0 - 1e-6);
        assert!(result[x] >= -1e-9);
        assert!(result[y] >= -1e-9);
        assert!(result[z] >= -1e-9);
        assert!(result.objective_value > 0.0, "objective should be positive");
    }

    // -----------------------------------------------------------------------
    // Test 2: Unbounded problem
    // -----------------------------------------------------------------------
    #[test]
    fn test_unbounded() {
        // min -x  s.t. x >= 0  (objective goes to -inf)
        let mut model = Model::new("unbounded");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        model.minimize(-1.0 * x);

        let err = model.solve().unwrap_err();
        assert!(
            matches!(err, ModelError::SolveError(SolveError::Unbounded)),
            "expected Unbounded, got {:?}",
            err
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: Infeasible problem
    // -----------------------------------------------------------------------
    #[test]
    fn test_infeasible() {
        // x >= 5, x <= 3  (contradictory)
        let mut model = Model::new("infeasible");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        // Single-variable constraint: use constraint! macro
        model.add_constraint(crate::constraint!(x >= 5.0));
        model.add_constraint(crate::constraint!(x <= 3.0));
        model.minimize(x);

        let err = model.solve().unwrap_err();
        assert!(
            matches!(err, ModelError::SolveError(SolveError::Infeasible)),
            "expected Infeasible, got {:?}",
            err
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: Equality constraint
    // -----------------------------------------------------------------------
    #[test]
    fn test_equality_constraint() {
        // min x + y  s.t. x + y == 5, x,y >= 0
        // Optimal: x=5, y=0 (or any split), obj=5
        let mut model = Model::new("eq");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        // Equality with complex LHS: use method API
        model.add_constraint((x + y).eq_constraint(5.0));
        model.minimize(x + y);

        let result = model.solve().unwrap();
        assert!(
            (result.objective_value - 5.0).abs() < 1e-6,
            "obj={} expected 5.0",
            result.objective_value
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: Variable bounds are respected
    // -----------------------------------------------------------------------
    #[test]
    fn test_variable_bounds() {
        // min x  s.t. x in [0, 3]
        // Optimal: x=0 (minimizing)
        let mut model = Model::new("bounds");
        let x = model.add_var("x", 0.0, 3.0);
        model.minimize(x);

        let result = model.solve().unwrap();
        assert!(result[x].abs() < 1e-6, "x should be 0.0, got {}", result[x]);

        // Maximize x in [0, 3] → should hit ub=3
        // Note: add explicit constraint because simplex edge-case (m=0, ub only)
        // does not check variable upper bounds when returning Unbounded.
        let mut model2 = Model::new("bounds_max");
        let x2 = model2.add_var("x", 0.0, 3.0);
        model2.add_constraint(crate::constraint!(x2 <= 3.0));
        model2.maximize(x2);

        let result2 = model2.solve().unwrap();
        assert!(
            (result2[x2] - 3.0).abs() < 1e-6,
            "x should be 3.0, got {}",
            result2[x2]
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: NoObjective error
    // -----------------------------------------------------------------------
    #[test]
    fn test_no_objective_error() {
        let mut model = Model::new("no_obj");
        let _x = model.add_var("x", 0.0, f64::INFINITY);
        let err = model.solve().unwrap_err();
        assert!(matches!(err, ModelError::NoObjective));
    }

    #[test]
    fn solution_proof_mapping_preserves_unproven_statuses() {
        assert_eq!(
            SolutionProof::from_status(&SolveStatus::Optimal),
            SolutionProof::GlobalOptimal
        );
        assert_eq!(
            SolutionProof::from_status(&SolveStatus::NonconvexGlobal),
            SolutionProof::GlobalOptimal
        );
        assert_eq!(
            SolutionProof::from_status(&SolveStatus::LocallyOptimal),
            SolutionProof::LocalOptimal
        );
        assert_eq!(
            SolutionProof::from_status(&SolveStatus::NonconvexLocal),
            SolutionProof::FeasibleUnproven
        );
        assert_eq!(
            SolutionProof::from_status(&SolveStatus::Timeout),
            SolutionProof::FeasibleUnproven
        );
        assert_eq!(
            SolutionProof::from_status(&SolveStatus::SuboptimalSolution),
            SolutionProof::FeasibleUnproven
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: result[x] indexing and result.value(x) agree
    // -----------------------------------------------------------------------
    #[test]
    fn test_result_index_and_value_agree() {
        let (mut model, x, y) = basic_model();
        let result = model.solve().unwrap();
        assert!((result[x] - result.value(x)).abs() < 1e-12);
        assert!((result[y] - result.value(y)).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Test 8: Maximize a simple LP (also tests constraint! macro)
    // -----------------------------------------------------------------------
    #[test]
    fn test_maximize() {
        // max x  s.t. x <= 7, x >= 0
        let mut model = Model::new("max_simple");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        // Single-variable constraint: use constraint! macro to exercise it
        model.add_constraint(crate::constraint!(x <= 7.0));
        model.maximize(x);

        let result = model.solve().unwrap();
        assert!(
            (result[x] - 7.0).abs() < 1e-6,
            "expected x=7, got {}",
            result[x]
        );
        assert!(
            (result.objective() - 7.0).abs() < 1e-6,
            "expected obj=7, got {}",
            result.objective()
        );
    }

    // -----------------------------------------------------------------------
    // Test 9: Model QP basic – Q=2I, c=(-4,-4), no constraints, bounds=[0,∞)
    // -----------------------------------------------------------------------
    #[test]
    fn test_model_qp_basic() {
        // min x^2+y^2 - 4x - 4y  (1/2*[[2,0],[0,2]]*[x,y]^T + [-4,-4]*[x,y])
        // Unconstrained min: x=y=2, obj = 4+4-8-8 = -8
        let mut model = Model::new("qp_basic");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        model.minimize(x * x + y * y + (-4.0) * x + (-4.0) * y);

        let result = model.solve().unwrap();
        assert_close(result[x], 2.0, "T9: x");
        assert_close(result[y], 2.0, "T9: y");
        // obj = 1/2*2*(4+4) - 4*2 - 4*2 = 8 - 16 = -8
        assert_close(result.objective_value, -8.0, "T9: obj");
    }

    // -----------------------------------------------------------------------
    // Test 10: Model QP with Eq constraint – Eq→2行変換の検証
    // -----------------------------------------------------------------------
    #[test]
    fn test_model_qp_equality() {
        // min x^2+y^2  s.t. x+y=1, x,y ∈ (-∞,∞)
        // Expected: x=y=0.5, obj=0.5
        let mut model = Model::new("qp_eq");
        let x = model.add_var("x", f64::NEG_INFINITY, f64::INFINITY);
        let y = model.add_var("y", f64::NEG_INFINITY, f64::INFINITY);
        model.add_constraint((x + y).eq_constraint(1.0));
        model.minimize(x * x + y * y);

        let result = model.solve().unwrap();
        assert_close(result[x], 0.5, "T10: x");
        assert_close(result[y], 0.5, "T10: y");
        assert_close(result.objective_value, 0.5, "T10: obj");
    }

    // -----------------------------------------------------------------------
    // Test 11: Model QP with Ge constraint – Ge→符号反転変換の検証
    // -----------------------------------------------------------------------
    #[test]
    fn test_model_qp_ge_constraint() {
        // min x^2+y^2  s.t. x+y >= 1, x,y ∈ (-∞,∞)
        // Same solution as equality case: x=y=0.5, obj=0.5
        let mut model = Model::new("qp_ge");
        let x = model.add_var("x", f64::NEG_INFINITY, f64::INFINITY);
        let y = model.add_var("y", f64::NEG_INFINITY, f64::INFINITY);
        model.add_constraint((x + y).geq(1.0));
        model.minimize(x * x + y * y);

        let result = model.solve().unwrap();
        let qp_tol = 2e-3;
        assert!(
            (result[x] - 0.5).abs() < qp_tol,
            "T11: x expected 0.5, got {}",
            result[x]
        );
        assert!(
            (result[y] - 0.5).abs() < qp_tol,
            "T11: y expected 0.5, got {}",
            result[y]
        );
        assert!(
            (result.objective_value - 0.5).abs() < qp_tol,
            "T11: obj expected 0.5, got {}",
            result.objective_value
        );
    }

    // -----------------------------------------------------------------------
    // Test 12: Model QP maximize – max -(x^2+y^2) s.t. x+y>=1, x,y>=0
    // -----------------------------------------------------------------------
    #[test]
    fn test_model_qp_maximize() {
        // max -(x^2+y^2), constraint: x+y >= 1, x,y >= 0
        // Expected: x=y=0.5, obj = -(0.25+0.25) = -0.5
        let mut model = Model::new("qp_maximize");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        model.add_constraint((x + y).geq(1.0));
        model.maximize((-1.0) * x * x + (-1.0) * y * y);

        let result = model.solve().unwrap();
        assert_close(result[x], 0.5, "T12: x");
        assert_close(result[y], 0.5, "T12: y");
        assert_close(result.objective_value, -0.5, "T12: obj");
    }

    // -----------------------------------------------------------------------
    // Test 13: Model QP box bounds – bounds=[0,1], T11相当
    // -----------------------------------------------------------------------
    #[test]
    fn test_model_qp_box_bounds() {
        // min x^2+y^2-4x-4y, bounds=[0,1]
        // Unconstrained min: x=y=2 → clipped to ub=1
        // Expected: x=y=1, obj = 1/2*2*(1+1) + (-4-4)*1 = 2-8 = -6
        let mut model = Model::new("qp_box");
        let x = model.add_var("x", 0.0, 1.0);
        let y = model.add_var("y", 0.0, 1.0);
        model.minimize(x * x + y * y + (-4.0) * x + (-4.0) * y);

        let result = model.solve().unwrap();
        assert_close(result[x], 1.0, "T13: x");
        assert_close(result[y], 1.0, "T13: y");
        assert_close(result.objective_value, -6.0, "T13: obj");
    }

    // -----------------------------------------------------------------------
    // Test 14: Model QP timeout – timeout=0.001秒でTimeout返却
    // -----------------------------------------------------------------------
    #[test]
    fn test_model_qp_timeout() {
        // Use a well-defined small QP but set an extremely short timeout.
        let mut model = Model::new("qp_timeout");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        model.minimize(x * x + y * y + (-4.0) * x + (-4.0) * y);
        model.set_timeout(0.000_001); // 1 microsecond → always times out

        let err = model.solve().unwrap_err();
        assert!(
            matches!(err, ModelError::Timeout),
            "expected Timeout, got {:?}",
            err
        );
    }

    // -----------------------------------------------------------------------
    // T8-1: LP with Eq constraint (Q=0 path: solve_as_lp)
    // -----------------------------------------------------------------------
    #[test]
    fn test_model_lp_equality() {
        // min x + 2y  s.t. x + y = 4, x,y >= 0
        // Optimal: x=4, y=0, obj=4
        let mut model = Model::new("lp_eq");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        model.add_constraint((x + y).eq_constraint(4.0));
        model.minimize(x + 2.0 * y);

        let result = model.solve().unwrap();
        assert_close(result.objective_value, 4.0, "T8-1: obj");
        // x+y=4 かつ obj=x+2y=4 → x=4,y=0 が最適
        assert_close(result[x] + result[y], 4.0, "T8-1: x+y=4");
    }

    // -----------------------------------------------------------------------
    // T8-2: Eq制約のdual solution（LeExpansionMap逆変換の検証）
    // -----------------------------------------------------------------------
    #[test]
    fn test_model_eq_dual_solution() {
        // min x^2 + y^2  s.t. x + y = 1, x,y in (-inf, inf)
        // KKT: 2x + λ = 0, 2y + λ = 0 → x=y=-λ/2; x+y=1 → λ=-1, x=y=0.5
        // dual of Eq constraint (shadow price) = λ = -1
        let mut model = Model::new("qp_eq_dual");
        let x = model.add_var("x", f64::NEG_INFINITY, f64::INFINITY);
        let y = model.add_var("y", f64::NEG_INFINITY, f64::INFINITY);
        model.add_constraint((x + y).eq_constraint(1.0));
        model.minimize(x * x + y * y);

        let result = model.solve().unwrap();
        assert_close(result.objective_value, 0.5, "T8-2: obj");
        assert_close(result[x], 0.5, "T8-2: x");
        assert_close(result[y], 0.5, "T8-2: y");

        // dual検証: Eq制約のshadow price = -1
        let dual = result
            .dual_solution
            .as_ref()
            .expect("T8-2: dual_solution is None");
        assert!(
            dual.len() == 1,
            "T8-2: dual length expected 1, got {}",
            dual.len()
        );
        assert!(
            (dual[0] - (-1.0)).abs() < EPS,
            "T8-2: dual[0] expected -1.0, got {}",
            dual[0]
        );
    }

    // -----------------------------------------------------------------------
    // LocalOptimal proof: indefinite-Q QP through Model API (table-driven).
    //
    // Sentinel: replacing from_status with a no-op that always returns
    // GlobalOptimal causes the assert_eq!(proof, LocalOptimal) to FAIL.
    // -----------------------------------------------------------------------
    #[test]
    #[allow(clippy::type_complexity)]
    fn test_model_qp_locally_optimal_proof() {
        // (name, q_diag, bounds, c) — all 2-variable diagonal-Q cases.
        let cases: &[(&str, [f64; 2], (f64, f64), [f64; 2])] = &[
            // Diagonal indefinite Q: eigenvalues -2, +2.
            ("diag(-2,2)", [-2.0, 2.0], (-1.0, 1.0), [0.0, 0.0]),
            // Diagonal indefinite Q: eigenvalues -3, +5 with linear term.
            ("diag(-3,5)", [-3.0, 5.0], (-2.0, 2.0), [-1.0, 0.0]),
        ];

        for &(name, q_diag, (lb, ub), c) in cases {
            let mut model = Model::new(name);
            let x = model.add_var("x0", lb, ub);
            let y = model.add_var("x1", lb, ub);
            // Q = diag(q_diag): DSL term (d/2)*xi*xi per variable
            model.minimize(
                (q_diag[0] / 2.0) * x * x + (q_diag[1] / 2.0) * y * y + c[0] * x + c[1] * y,
            );

            let result = model.solve();
            match result {
                Ok(r) => {
                    assert_eq!(
                        r.status,
                        otspot_core::problem::SolveStatus::LocallyOptimal,
                        "[{name}] expected LocallyOptimal, got {:?}",
                        r.status
                    );
                    assert_eq!(
                        r.proof,
                        SolutionProof::LocalOptimal,
                        "[{name}] expected LocalOptimal proof, got {:?}",
                        r.proof
                    );
                    assert!(
                        !r.has_global_optimality_proof(),
                        "[{name}] has_global_optimality_proof must be false for LocallyOptimal"
                    );
                }
                Err(e) => panic!("[{name}] unexpected Err: {e:?}"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // FeasibleUnproven proof: impossibly-tight tolerance forces SuboptimalSolution
    // on a convex QP that the IPM solves to finite residuals (table-driven).
    //
    // Sentinel: replacing from_status with a no-op returning GlobalOptimal
    // causes the assert_eq!(proof, FeasibleUnproven) to FAIL.
    // -----------------------------------------------------------------------
    #[test]
    #[allow(clippy::type_complexity)]
    fn test_model_qp_feasible_unproven_proof() {
        use otspot_core::options::Tolerance;

        // (name, q_diag, (lb,ub), c)
        let cases: &[(&str, [f64; 2], (f64, f64), [f64; 2])] = &[
            // Convex PSD Q=2I, c=[-4,-4]. IPM converges (residuals ~1e-6) but
            // Custom(1e-200) makes satisfies_eps always false → SuboptimalSolution.
            (
                "convex_2x2_tight_tol",
                [2.0, 2.0],
                (0.0, f64::INFINITY),
                [-4.0, -4.0],
            ),
            // Convex PSD Q=4I, c=[0,-2] with box bounds.
            ("convex_box_tight_tol", [4.0, 4.0], (0.0, 3.0), [0.0, -2.0]),
        ];

        for &(name, q_diag, (lb, ub), c) in cases {
            let mut model = Model::new(name);
            let x = model.add_var("x0", lb, ub);
            let y = model.add_var("x1", lb, ub);
            model.minimize(
                (q_diag[0] / 2.0) * x * x + (q_diag[1] / 2.0) * y * y + c[0] * x + c[1] * y,
            );
            // Impossibly tight tolerance: IPM finds a finite-residual solution
            // but satisfies_eps(1e-200) is always false → SuboptimalSolution.
            model.set_tolerance(Tolerance::Custom(1e-200));

            let result = model.solve();
            match result {
                Ok(r) => {
                    assert_eq!(
                        r.proof,
                        SolutionProof::FeasibleUnproven,
                        "[{name}] expected FeasibleUnproven proof, got {:?} (status={:?})",
                        r.proof,
                        r.status
                    );
                    assert!(
                        !r.has_global_optimality_proof(),
                        "[{name}] has_global_optimality_proof must be false for FeasibleUnproven"
                    );
                    assert!(
                        !r.solution.is_empty(),
                        "[{name}] solution must be non-empty"
                    );
                }
                Err(e) => panic!("[{name}] unexpected Err: {e:?}"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Sentinel: classify_status_error maps NonConvex/NotSupported to typed
    // variants (not Internal). Reverting the mapping to Internal causes FAIL.
    // -----------------------------------------------------------------------
    #[test]
    fn classify_status_error_typed_variants() {
        let cases_nonconvex = ["indefinite Q: eigenvalue < 0", "non-PSD matrix in MIQP"];
        for msg in &cases_nonconvex {
            let status = SolveStatus::NonConvex(msg.to_string());
            let err = classify_status_error(status).expect("NonConvex must map to Some");
            assert!(
                matches!(err, ModelError::NonConvex(_)),
                "NonConvex status must yield ModelError::NonConvex, got {:?}",
                err
            );
        }

        let cases_not_supported = ["QCQP not supported", "constraint type unsupported"];
        for msg in &cases_not_supported {
            let status = SolveStatus::NotSupported(msg.to_string());
            let err = classify_status_error(status).expect("NotSupported must map to Some");
            assert!(
                matches!(err, ModelError::NotSupported(_)),
                "NotSupported status must yield ModelError::NotSupported, got {:?}",
                err
            );
        }
    }

    // -----------------------------------------------------------------------
    // Sentinel: MIQP with indefinite Q returns ModelError::NonConvex.
    // Reverting NonConvex → Internal in finish_mip causes FAIL.
    // Table-driven: multiple indefinite Q shapes.
    // -----------------------------------------------------------------------
    #[test]
    fn miqp_nonconvex_q_returns_nonconvex_error() {
        let cases: &[(&str, [f64; 2])] = &[
            ("diag(-1, 1)", [-1.0, 1.0]),
            ("diag(-2, 3)", [-2.0, 3.0]),
            ("diag(1, -1)", [1.0, -1.0]),
        ];

        for &(name, q_diag) in cases {
            let mut model = Model::new(name);
            let x = model.add_binary_var("x");
            let y = model.add_binary_var("y");
            // Q=diag(q_diag): Q[i][i]=d → (d/2)*xi*xi in DSL
            model.minimize((q_diag[0] / 2.0) * (x * x) + (q_diag[1] / 2.0) * (y * y));

            let err = model
                .solve()
                .expect_err(&format!("[{name}] expected Err(NonConvex), got Ok"));
            assert!(
                matches!(err, ModelError::NonConvex(_)),
                "[{name}] expected ModelError::NonConvex, got {:?}",
                err
            );
        }
    }

    // -----------------------------------------------------------------------
    // Sentinel: validate_bounds rejects lb>ub and NaN.
    // No-op'ing validate_bounds (always Ok) causes these tests to FAIL:
    //   - add_var with lb>ub would not record error → solve() would succeed
    //     (an LP with inverted bounds becomes infeasible but NOT an InvalidInput).
    // -----------------------------------------------------------------------
    #[test]
    fn add_var_lb_gt_ub_defers_error_to_solve() {
        let cases: &[(&str, f64, f64)] = &[
            ("lb=5 > ub=3", 5.0, 3.0),
            ("lb=1.0 > ub=0.999", 1.0, 0.999),
            ("lb=inf > ub=0", f64::INFINITY, 0.0),
        ];
        for &(label, lb, ub) in cases {
            let mut model = Model::new(label);
            let x = model.add_var("x", lb, ub);
            model.minimize(x);
            let err = model
                .solve()
                .expect_err(&format!("[{label}] expected Err, got Ok"));
            assert!(
                matches!(err, ModelError::InvalidInput(_)),
                "[{label}] expected InvalidInput, got {err:?}"
            );
        }
    }

    #[test]
    fn add_var_nan_bounds_defers_error_to_solve() {
        let cases: &[(&str, f64, f64)] = &[
            ("lb=NaN", f64::NAN, 1.0),
            ("ub=NaN", 0.0, f64::NAN),
            ("both=NaN", f64::NAN, f64::NAN),
        ];
        for &(label, lb, ub) in cases {
            let mut model = Model::new(label);
            let x = model.add_var("x", lb, ub);
            model.minimize(x);
            let err = model
                .solve()
                .expect_err(&format!("[{label}] expected Err, got Ok"));
            assert!(
                matches!(err, ModelError::InvalidInput(_)),
                "[{label}] expected InvalidInput, got {err:?}"
            );
        }
    }

    #[test]
    fn add_int_var_lb_gt_ub_defers_error_to_solve() {
        let cases: &[(&str, f64, f64)] =
            &[("int lb=3 > ub=1", 3.0, 1.0), ("int lb=NaN", f64::NAN, 5.0)];
        for &(label, lb, ub) in cases {
            let mut model = Model::new(label);
            let x = model.add_int_var("x", lb, ub);
            model.minimize(x);
            let err = model
                .solve()
                .expect_err(&format!("[{label}] expected Err, got Ok"));
            assert!(
                matches!(err, ModelError::InvalidInput(_)),
                "[{label}] expected InvalidInput, got {err:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // try_add_var / try_add_int_var: immediate Result API.
    // Sentinel: no-op of validate_bounds makes all these Ok → assert!(is_err()) FAILs.
    // -----------------------------------------------------------------------
    #[test]
    fn try_add_var_returns_err_for_invalid_bounds() {
        let cases: &[(&str, f64, f64)] = &[
            ("lb>ub", 2.0, 1.0),
            ("lb=NaN", f64::NAN, 1.0),
            ("ub=NaN", 0.0, f64::NAN),
            ("lb=inf > ub=0", f64::INFINITY, 0.0),
        ];
        for &(label, lb, ub) in cases {
            let mut model = Model::new(label);
            let result = model.try_add_var("x", lb, ub);
            assert!(
                result.is_err(),
                "[{label}] try_add_var should return Err for invalid bounds, got Ok"
            );
        }
    }

    #[test]
    fn try_add_var_returns_ok_for_valid_bounds() {
        let cases: &[(&str, f64, f64)] = &[
            ("lb=ub", 3.0, 3.0),
            ("lb=0 ub=inf", 0.0, f64::INFINITY),
            ("lb=-inf ub=inf", f64::NEG_INFINITY, f64::INFINITY),
            ("lb=-inf ub=0", f64::NEG_INFINITY, 0.0),
        ];
        for &(label, lb, ub) in cases {
            let mut model = Model::new(label);
            assert!(
                model.try_add_var("x", lb, ub).is_ok(),
                "[{label}] try_add_var should return Ok for valid bounds"
            );
        }
    }

    #[test]
    fn try_add_int_var_returns_err_for_invalid_bounds() {
        let cases: &[(&str, f64, f64)] = &[("int lb>ub", 5.0, 2.0), ("int lb=NaN", f64::NAN, 3.0)];
        for &(label, lb, ub) in cases {
            let mut model = Model::new(label);
            assert!(
                model.try_add_int_var("n", lb, ub).is_err(),
                "[{label}] try_add_int_var should return Err"
            );
        }
    }

    // -----------------------------------------------------------------------
    // try_value: safe accessor — wrong model_id and out-of-range both return Err.
    // Sentinel: removing the model_id check makes cross-model test pass/Err → Ok
    //   causing the assert!(result.is_err()) to FAIL.
    // -----------------------------------------------------------------------
    #[test]
    fn try_value_cross_model_returns_err() {
        let mut m1 = Model::new("m1");
        let x1 = m1.add_var("x", 0.0, f64::INFINITY);
        m1.minimize(x1);
        let r1 = m1.solve().unwrap();

        // Variable from a different model — same index (0), different model_id.
        let mut m2 = Model::new("m2");
        let y = m2.add_var("y", 0.0, f64::INFINITY);

        assert!(
            r1.try_value(y).is_err(),
            "try_value with variable from different model must return Err"
        );
        // Correct variable works fine.
        assert!(r1.try_value(x1).is_ok());
    }

    #[test]
    fn try_value_valid_returns_ok() {
        let (mut model, x, y) = basic_model();
        let result = model.solve().unwrap();
        assert!(result.try_value(x).is_ok());
        assert!(result.try_value(y).is_ok());
        assert!((result.try_value(x).unwrap() - result.value(x)).abs() < 1e-12);
    }

    // Out-of-range with a *matching* model_id: a variable added to the same
    // model after solving has an index past the result's solution vector.
    // Sentinel for the `.ok_or_else` branch (the model_id check passes here, so
    // no-op'ing the bounds check — e.g. `self.solution[var.index]` — panics).
    #[test]
    fn try_value_out_of_range_same_model_returns_err() {
        let mut model = Model::new("grow");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        model.minimize(x);
        let result = model.solve().unwrap();

        // Extend the same model: same model_id, index beyond solution length.
        let late = model.add_var("late", 0.0, f64::INFINITY);
        assert_eq!(late.model_id, result.model_id, "same model_id expected");
        assert!(
            late.index >= result.solution.len(),
            "test setup: late var must be out of range"
        );
        assert!(
            result.try_value(late).is_err(),
            "try_value must return Err for an out-of-range index even when model_id matches"
        );
    }

    // -----------------------------------------------------------------------
    // ModelResult: Clone derive
    // -----------------------------------------------------------------------
    #[test]
    fn model_result_clone() {
        let (mut model, x, _y) = basic_model();
        let result = model.solve().unwrap();
        let cloned = result.clone();
        assert!((cloned.objective_value - result.objective_value).abs() < 1e-12);
        assert_eq!(cloned.solution.len(), result.solution.len());
        assert!((cloned[x] - result[x]).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Variable name retention
    // -----------------------------------------------------------------------
    #[test]
    fn var_name_is_retained() {
        let cases = [("alpha", 0.0, 1.0), ("beta_var", 0.0, f64::INFINITY)];
        let mut model = Model::new("named");
        for &(name, lb, ub) in &cases {
            let v = model.add_var(name, lb, ub);
            assert_eq!(model.var_name(v), name, "var_name mismatch for '{name}'");
        }
        let iv = model.add_int_var("gamma_int", 0.0, 10.0);
        assert_eq!(model.var_name(iv), "gamma_int");
    }

    // -----------------------------------------------------------------------
    // try_var_name: checked variant (model_id + index range).
    //
    // Sentinel: replacing try_var_name with a no-op that always returns Ok("")
    // causes the cross-model and out-of-range ERR assertions to FAIL.
    // -----------------------------------------------------------------------
    #[test]
    fn try_var_name_ok_same_model() {
        let cases: &[(&str, f64, f64)] = &[
            ("x", 0.0, f64::INFINITY),
            ("y_var", -1.0, 1.0),
            ("z_int", 0.0, 10.0),
        ];
        let mut model = Model::new("try_var_name_ok");
        let mut vars = Vec::new();
        for &(name, lb, ub) in cases {
            vars.push((model.add_var(name, lb, ub), name));
        }
        for (var, expected_name) in &vars {
            assert_eq!(
                model.try_var_name(*var).unwrap(),
                *expected_name,
                "try_var_name mismatch"
            );
            // P3-4: var_name and try_var_name must agree for same-model vars.
            assert_eq!(
                model.var_name(*var),
                model.try_var_name(*var).unwrap(),
                "var_name and try_var_name must return the same value for '{expected_name}'"
            );
        }
    }

    // P2-2: try_var_name rejects variables from multiple different models.
    // Table-driven: model_b, model_c, model_d all independently reject model_a's var.
    // Sentinel: removing the model_id check makes all these Err → Ok, causing FAIL.
    #[test]
    fn try_var_name_err_cross_model() {
        let mut model_a = Model::new("model_a");
        let x = model_a.add_var("x_in_a", 0.0, 1.0);

        let cases: &[&str] = &["model_b", "model_c", "model_d"];
        for &name in cases {
            let m = Model::new(name);
            let err = m.try_var_name(x).unwrap_err();
            assert!(
                matches!(err, ModelError::InvalidInput(_)),
                "[{name}] expected InvalidInput for cross-model var, got {err:?}"
            );
        }
    }

    // P2-3: try_var_name boundary values — N-1 (last valid), N (exact OOB), usize::MAX.
    // Sentinel: replacing .get(var.index) with direct indexing would panic on N/MAX.
    #[test]
    fn try_var_name_err_index_out_of_range() {
        let mut model = Model::new("range_check");
        let x = model.add_var("x", 0.0, 1.0); // index=0, N=1 variable

        // N-1 = 0: last valid index → Ok.
        assert!(model.try_var_name(x).is_ok(), "index=0 (N-1) must be Ok");

        // N = 1: exact out-of-bound → Err.
        let oob_n = Variable {
            index: 1,
            model_id: x.model_id,
        };
        let err_n = model.try_var_name(oob_n).unwrap_err();
        assert!(
            matches!(err_n, ModelError::InvalidInput(_)),
            "index=N=1: expected InvalidInput, got {err_n:?}"
        );

        // usize::MAX: extreme out-of-bound → Err.
        let oob_max = Variable {
            index: usize::MAX,
            model_id: x.model_id,
        };
        let err_max = model.try_var_name(oob_max).unwrap_err();
        assert!(
            matches!(err_max, ModelError::InvalidInput(_)),
            "index=usize::MAX: expected InvalidInput, got {err_max:?}"
        );
    }

    #[test]
    fn var_name_regression_unchecked_still_works() {
        // Verify existing var_name behaviour is unchanged (regression sentinel).
        let mut model = Model::new("regression");
        let cases: &[(&str, f64, f64)] = &[("a", 0.0, 1.0), ("b", 0.0, 2.0), ("c_int", 0.0, 5.0)];
        let mut pairs = Vec::new();
        for &(name, lb, ub) in cases {
            pairs.push((model.add_var(name, lb, ub), name));
        }
        for (var, expected) in &pairs {
            assert_eq!(model.var_name(*var), *expected);
        }
    }

    // P2-1: var_name is unchecked — cross-model behavior depends on whether the
    // foreign Variable's index is in-range for the receiver model.
    //
    // Case A: index in-range → silently returns the *wrong* model's variable name.
    // Case B: index out-of-range → panics (Rust slice bounds check).
    //
    // These tests document the hazard warned about in the var_name docstring:
    // "unchecked and may index into unrelated data".

    #[test]
    fn var_name_cross_model_in_range_silently_returns_wrong_data() {
        // Both models have exactly one variable (index=0).
        // model_a's var has index=0 with the same integer position as model_b's var.
        // var_name doesn't check model_id, so it reads model_b's slot — wrong data.
        let mut model_a = Model::new("a");
        let x_a = model_a.add_var("x_in_a", 0.0, 1.0); // index=0

        let mut model_b = Model::new("b");
        let _y_b = model_b.add_var("y_in_b", 0.0, 1.0); // index=0

        // var_name is unchecked: x_a.index=0 is in-range for model_b,
        // so it returns model_b's variable name — NOT "x_in_a".
        let returned = model_b.var_name(x_a);
        assert_ne!(
            returned, "x_in_a",
            "unchecked: cross-model var_name returns unrelated data"
        );
        assert_eq!(
            returned, "y_in_b",
            "unchecked: returns model_b's var at the same index"
        );
    }

    // Case B: cross-model var whose index is out-of-range for the receiver panics.
    // Sentinel: using .get() in var_name instead of direct indexing would remove
    // the panic, causing this #[should_panic] test to FAIL.
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn var_name_cross_model_out_of_range_panics() {
        let mut model_a = Model::new("a_2var");
        let _x0 = model_a.add_var("x0", 0.0, 1.0); // index=0
        let x1 = model_a.add_var("x1", 0.0, 1.0); // index=1

        // model_b has only 1 variable (len=1). x1.index=1 is out of range → panic.
        let mut model_b = Model::new("b_1var");
        let _y_b = model_b.add_var("y_in_b", 0.0, 1.0);

        let _ = model_b.var_name(x1);
    }

    // -----------------------------------------------------------------------
    // set_timeout validation (already implemented; table-driven sentinel)
    // No-op'ing validate_timeout makes negative/NaN tests succeed → FAILs.
    // -----------------------------------------------------------------------
    #[test]
    fn set_timeout_invalid_defers_error() {
        let cases: &[(&str, f64)] = &[
            ("negative", -1.0),
            ("NaN", f64::NAN),
            ("neg_inf", f64::NEG_INFINITY),
        ];
        for &(label, secs) in cases {
            let mut model = Model::new(label);
            let x = model.add_var("x", 0.0, f64::INFINITY);
            model.minimize(x);
            model.set_timeout(secs);
            let err = model
                .solve()
                .expect_err(&format!("[{label}] expected Err for invalid timeout"));
            assert!(
                matches!(err, ModelError::InvalidInput(_)),
                "[{label}] expected InvalidInput, got {err:?}"
            );
        }
    }

    #[test]
    fn try_set_timeout_returns_err_for_invalid() {
        let cases: &[(&str, f64)] = &[
            ("negative", -0.001),
            ("NaN", f64::NAN),
            ("inf", f64::INFINITY),
        ];
        for &(label, secs) in cases {
            let mut model = Model::new(label);
            assert!(
                model.try_set_timeout(secs).is_err(),
                "[{label}] try_set_timeout should return Err"
            );
        }
    }

    #[test]
    fn try_set_timeout_ok_for_valid() {
        let valid = [0.0, 0.001, 1.0, 3600.0];
        for &secs in &valid {
            let mut model = Model::new("t");
            assert!(
                model.try_set_timeout(secs).is_ok(),
                "should be Ok for secs={secs}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Model API tests for MILP / MIQP paths.
//
// A lib-target module (kept separate so it runs under the CI `lib-only`
// nextest profile, which excludes `tests/` integration targets). Exercises the
// `Model` high-level API over the MILP/MIQP branch-and-bound solver.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod mip_model_tests {
    use super::{Model, ModelError, SolveError};

    const EPS: f64 = 1e-4;

    #[test]
    fn model_add_int_var_maximize_branches() {
        // max x s.t. 2x <= 3, x integer in [0,5] → x = 1, obj = 1.
        let mut m = Model::new("milp_int");
        let x = m.add_int_var("x", 0.0, 5.0);
        m.add_constraint((2.0 * x).leq(3.0));
        m.maximize(x);
        let r = m.solve().unwrap();
        assert!((r.objective() - 1.0).abs() < EPS, "obj={}", r.objective());
        assert!((r[x] - 1.0).abs() < EPS, "x={}", r[x]);
    }

    #[test]
    fn model_binary_knapsack() {
        let mut m = Model::new("knapsack");
        let a = m.add_binary_var("a");
        let b = m.add_binary_var("b");
        let c = m.add_binary_var("c");
        let d = m.add_binary_var("d");
        m.add_constraint((5.0 * a + 7.0 * b + 4.0 * c + 3.0 * d).leq(14.0));
        m.maximize(8.0 * a + 11.0 * b + 6.0 * c + 4.0 * d);
        let r = m.solve().unwrap();
        assert!((r.objective() - 21.0).abs() < EPS, "obj={}", r.objective());
        assert_eq!(
            (r[a].round(), r[b].round(), r[c].round(), r[d].round()),
            (0.0, 1.0, 1.0, 1.0)
        );
    }

    #[test]
    fn model_integer_infeasible_errors() {
        let mut m = Model::new("infeasible");
        let x = m.add_int_var("x", 0.0, 10.0);
        m.add_constraint((1.0 * x).geq(1.2));
        m.add_constraint((1.0 * x).leq(1.8));
        m.minimize(x);
        let err = m.solve().unwrap_err();
        assert!(
            matches!(err, ModelError::SolveError(SolveError::Infeasible)),
            "got {err:?}"
        );
    }

    #[test]
    fn model_integer_unbounded_errors() {
        let mut m = Model::new("unbounded");
        let x = m.add_int_var("x", 0.0, f64::INFINITY);
        m.maximize(x);
        let err = m.solve().unwrap_err();
        assert!(
            matches!(err, ModelError::SolveError(SolveError::Unbounded)),
            "got {err:?}"
        );
    }

    #[test]
    fn model_convex_miqp_branches_to_integer_optimum() {
        // min x^2 - 5x, x integer in [0,5].
        // Continuous min at x=2.5 (fractional → branch); integer optima x=2 or x=3, obj = -6.
        let mut m = Model::new("convex_miqp");
        let x = m.add_int_var("x", 0.0, 5.0);
        m.minimize(x * x + (-5.0) * x);
        let r = m.solve().unwrap();
        assert!(
            (r.objective() - (-6.0)).abs() < EPS,
            "obj={}",
            r.objective()
        );
        let xr = r[x].round();
        assert!(xr == 2.0 || xr == 3.0, "x must be 2 or 3, got {}", r[x]);
        assert!((r[x] - xr).abs() < EPS, "x must be integral: {}", r[x]);
    }

    #[test]
    fn model_nonconvex_miqp_errors() {
        // indefinite Q (negative curvature) → must return ModelError::NonConvex, not silent wrong.
        let cases: &[(&str, &[f64], &[f64])] = &[
            ("single neg", &[-2.0], &[1.0]),
            ("neg-pos-2var", &[-3.0, 2.0], &[0.0, 1.0]),
        ];
        for &(name, q_diag, c_vec) in cases {
            let n = q_diag.len();
            let mut m = Model::new(name);
            let vars: Vec<_> = (0..n)
                .map(|i| m.add_int_var(&format!("x{i}"), 0.0, 5.0))
                .collect();
            // Build objective: sum (d_i/2)*x_i^2 + c_i*x_i via DSL
            let obj = vars.iter().zip(q_diag).zip(c_vec).fold(
                crate::quad_expr::QuadExpr::from(0.0_f64),
                |acc, ((&v, &d), &c)| acc + (d / 2.0) * (v * v) + c * v,
            );
            m.minimize(obj);
            let err = m.solve().unwrap_err();
            assert!(
                matches!(err, ModelError::NonConvex(_)),
                "[{name}] expected ModelError::NonConvex, got {err:?}"
            );
        }
    }

    #[test]
    fn model_mixed_integer_continuous() {
        // x integer in [0,5], y continuous in [0,5], x + y <= 3.5.
        // max(x + y) → Optimum: x=3 (integer), y=0.5 → obj 3.5.
        let mut m = Model::new("mixed");
        let x = m.add_int_var("x", 0.0, 5.0);
        let y = m.add_var("y", 0.0, 5.0);
        m.add_constraint((x + y).leq(3.5));
        m.maximize(x + y);
        let r = m.solve().unwrap();
        assert!((r.objective() - 3.5).abs() < EPS, "obj={}", r.objective());
        assert!(
            (r[x].round() - r[x]).abs() < EPS,
            "x must be integral, x={}",
            r[x]
        );
    }

    // --- 目的関数定数 fold テスト ---

    // ① 線形 minimize: 定数 +3 が obj_value に含まれること
    #[test]
    fn test_linear_objective_constant_included() {
        // min 2x + 3.0  s.t. x ≥ 1  →  x* = 1, obj* = 2*1 + 3 = 5
        let mut model = Model::new("lin_const");
        let x = model.add_var("x", 1.0, f64::INFINITY);
        model.minimize(2.0 * x + 3.0);
        let result = model.solve().unwrap();
        assert!(
            (result[x] - 1.0).abs() < EPS,
            "x* should be 1, got {}",
            result[x]
        );
        assert!(
            (result.objective_value - 5.0).abs() < EPS,
            "obj* should be 5 (includes constant +3), got {}",
            result.objective_value
        );
    }

    // ② 二次 minimize: min (x-2)² = x²-4x+4, x≥0  →  x*=2, obj*=0
    #[test]
    fn test_quad_objective_constant_included() {
        // minimize(x*x - 4.0*x + 4.0)  s.t. x ≥ 0
        // x* = 2, obj* = 4 - 8 + 4 = 0
        let mut model = Model::new("quad_const");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        model.minimize(x * x + (-4.0) * x + 4.0);
        let result = model.solve().unwrap();
        assert!(
            (result[x] - 2.0).abs() < 1e-4,
            "quad_const: x* should be 2, got {}",
            result[x]
        );
        assert!(
            result.objective_value.abs() < 1e-4,
            "quad_const: obj* should be 0 (constant +4 included), got {}",
            result.objective_value
        );
    }

    // ③ maximize の定数符号: max(-x + 5.0) s.t. x≥0 → x*=0, obj*=5
    #[test]
    fn test_maximize_objective_constant_sign() {
        // max -x + 5.0  s.t. x ≥ 0  →  x* = 0, obj* = 0 + 5 = 5
        let mut model = Model::new("max_const");
        let x = model.add_var("x", 0.0, 10.0);
        model.maximize((-1.0) * x + 5.0);
        let result = model.solve().unwrap();
        assert!(
            (result[x]).abs() < EPS,
            "max_const: x* should be 0, got {}",
            result[x]
        );
        assert!(
            (result.objective_value - 5.0).abs() < EPS,
            "max_const: obj* should be 5 (constant not negated for max), got {}",
            result.objective_value
        );
    }

    // ④ set_obj_offset と DSL 定数の併用: 二重加算しないこと
    #[test]
    fn test_obj_offset_and_dsl_constant_no_double_count() {
        // min x + 3.0  s.t. x ≥ 1, with set_obj_offset(10.0)
        // obj* = 1 + 3 + 10 = 14
        let mut model = Model::new("offset_plus_const");
        let x = model.add_var("x", 1.0, f64::INFINITY);
        model.minimize(1.0 * x + 3.0);
        model.set_obj_offset(10.0);
        let result = model.solve().unwrap();
        assert!(
            (result.objective_value - 14.0).abs() < EPS,
            "offset+const: obj* should be 14 (1+3+10), got {}",
            result.objective_value
        );
    }

    // ⑤ 再 minimize でリセット: 定数が二重加算されないこと
    #[test]
    fn test_reminimize_constant_not_double_counted() {
        // First minimize(x + 3.0), then minimize(x + 7.0): only the last constant counts.
        let mut model = Model::new("reminimize");
        let x = model.add_var("x", 1.0, f64::INFINITY);
        model.minimize(1.0 * x + 3.0); // constant = 3.0
        model.minimize(1.0 * x + 7.0); // constant = 7.0 (replaces 3.0)
        let result = model.solve().unwrap();
        assert!(
            (result.objective_value - 8.0).abs() < EPS,
            "re-minimize: obj* should be 8 (1+7, not 1+3+7=11), got {}",
            result.objective_value
        );
    }

    // ---------------------------------------------------------------------------
    // State-machine tests for P2-a / P2-b objective-setter transitions.
    // ---------------------------------------------------------------------------

    // P2-a: NaN quad (invalid DSL) → pure-linear must clear the stale error.
    #[test]
    fn test_p2a_nan_dsl_quad_then_linear_is_optimal() {
        // minimize(NaN * x*x) sets quad_dsl_error.
        // A subsequent minimize(x) clears it (at the start of apply_objective) → solve succeeds.
        let mut model = Model::new("p2a_nan_then_linear");
        let x = model.add_var("x", 1.0, f64::INFINITY);
        model.minimize(f64::NAN * (x * x)); // invalid DSL quad → error recorded
        model.minimize(1.0 * x); // pure-linear replaces objective
        let result = model.solve();
        assert!(
            result.is_ok(),
            "P2-a: should be Optimal after linear replaces NaN quad, got {result:?}"
        );
        let r = result.unwrap();
        assert!(
            (r[x] - 1.0).abs() < EPS,
            "P2-a: x* should be 1, got {}",
            r[x]
        );
    }

    // P2-a (reverse): valid quad → then invalid (NaN) quad → solve must error.
    #[test]
    fn test_p2a_valid_quad_then_nan_errors() {
        let mut model = Model::new("p2a_valid_then_nan");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        model.minimize(x * x + (-2.0) * x); // valid DSL quad
        model.minimize(f64::NAN * (x * x)); // invalid DSL quad → should error
        let result = model.solve();
        assert!(
            matches!(result, Err(ModelError::InvalidInput(_))),
            "P2-a: NaN quad must yield InvalidInput, got {result:?}"
        );
    }

    // P2-a: maximize with stale DSL error must also clear on replacement.
    #[test]
    fn test_p2a_nan_dsl_then_maximize_linear_is_optimal() {
        let mut model = Model::new("p2a_nan_then_max");
        let x = model.add_var("x", 0.0, 5.0);
        model.minimize(f64::NAN * (x * x)); // stale error
        model.maximize(1.0 * x); // pure-linear maximize replaces
        let result = model.solve();
        assert!(
            result.is_ok(),
            "P2-a: maximize should succeed after NaN DSL cleared, got {result:?}"
        );
        let r = result.unwrap();
        assert!(
            (r[x] - 5.0).abs() < EPS,
            "P2-a: maximize x → x*=5, got {}",
            r[x]
        );
    }

    // State-machine: DSL transition table (DSL paths only × invariants).
    // Each row: transition sequence → expected (has_error, via_dsl, has_q).
    #[test]
    fn test_quad_state_invariants_transition_table() {
        // Helper: build a 1-var model with x in [0,∞) and return (model, x).
        fn mk() -> (Model, crate::variable::Variable) {
            let mut m = Model::new("t");
            let x = m.add_var("x", 0.0, f64::INFINITY);
            (m, x)
        }

        // After DSL success: no error, via_dsl=true, has_q=true.
        {
            let (mut m, x) = mk();
            m.minimize(x * x);
            assert!(!m.has_quad_dsl_error(), "DSL ok: no error");
            assert!(m.is_quad_via_dsl(), "DSL ok: via_dsl=true");
            assert!(m.has_quadratic_objective(), "DSL ok: has_q=true");
        }

        // After DSL fail (NaN): error=true, via_dsl=false, has_q=false.
        {
            let (mut m, x) = mk();
            m.minimize(f64::NAN * (x * x));
            assert!(m.has_quad_dsl_error(), "DSL fail: error recorded");
            assert!(!m.is_quad_via_dsl(), "DSL fail: via_dsl=false");
            assert!(!m.has_quadratic_objective(), "DSL fail: has_q=false");
        }

        // DSL success → DSL fail: error=true, via_dsl=false, has_q=false.
        {
            let (mut m, x) = mk();
            m.minimize(x * x);
            m.minimize(f64::NAN * (x * x));
            assert!(m.has_quad_dsl_error(), "ok→fail: error recorded");
            assert!(!m.is_quad_via_dsl(), "ok→fail: via_dsl=false");
            assert!(!m.has_quadratic_objective(), "ok→fail: has_q=false");
        }

        // DSL success → linear minimize: error=false, via_dsl=false, has_q=false.
        {
            let (mut m, x) = mk();
            m.minimize(x * x);
            m.minimize(1.0 * x);
            assert!(!m.has_quad_dsl_error(), "dsl→linear: no error");
            assert!(!m.is_quad_via_dsl(), "dsl→linear: via_dsl=false");
            assert!(!m.has_quadratic_objective(), "dsl→linear: DSL Q cleared");
        }
    }

    // solve() result after each DSL transition (not just state — actual Optimal/Error).
    #[test]
    fn test_quad_state_solve_outcome_table() {
        let cases: &[(&str, bool)] = &[
            // (description, expect_ok)
            // DSL NaN then linear → Optimal (P2-a regression)
            ("nan_then_linear", true),
            // DSL NaN alone → error
            ("nan_alone", false),
            // DSL ok then NaN → error (P2-a reverse)
            ("ok_then_nan", false),
            // P2-i: DSL NaN then valid DSL quad → Optimal (stale error must not block)
            ("nan_then_valid_quad", true),
            // P2-i: DSL NaN × 2 then valid DSL quad → Optimal
            ("nan_nan_then_valid_quad", true),
        ];

        for &(name, expect_ok) in cases {
            let mut m = Model::new(name);
            let x = m.add_var("x", 0.0, f64::INFINITY);

            match name {
                "nan_then_linear" => {
                    m.minimize(f64::NAN * (x * x) + x);
                    m.minimize(1.0 * x);
                }
                "nan_alone" => {
                    m.minimize(f64::NAN * (x * x) + x);
                }
                "ok_then_nan" => {
                    m.minimize(x * x + (-4.0) * x);
                    m.minimize(f64::NAN * (x * x) + x);
                }
                "nan_then_valid_quad" => {
                    m.minimize(f64::NAN * (x * x));
                    m.minimize(x * x + (-4.0) * x);
                }
                "nan_nan_then_valid_quad" => {
                    m.minimize(f64::NAN * (x * x));
                    m.minimize(f64::NAN * (x * x));
                    m.minimize(x * x + (-4.0) * x);
                }
                _ => unreachable!(),
            }

            let result = m.solve();
            if expect_ok {
                assert!(
                    result.is_ok(),
                    "case '{name}': expected Optimal, got {result:?}"
                );
            } else {
                assert!(
                    matches!(result, Err(ModelError::InvalidInput(_))),
                    "case '{name}': expected InvalidInput, got {result:?}"
                );
            }
        }
    }

    // ---------------------------------------------------------------------------
    // P2-f: linear part of QuadExpr must also reject foreign-model variables
    // ---------------------------------------------------------------------------

    // Mixed: quad from m1 + linear from m2 → InvalidInput.
    #[test]
    fn test_p2f_mixed_quad_local_linear_foreign_rejected() {
        let mut m1 = Model::new("m1");
        let x1 = m1.add_var("x", 0.0, f64::INFINITY);

        let mut m2 = Model::new("m2");
        let y2 = m2.add_var("y", 0.0, f64::INFINITY);

        // x1 is local (quad ok), y2 is foreign (linear must be rejected).
        m1.minimize(x1 * x1 + y2);
        let result = m1.solve();
        assert!(
            matches!(result, Err(ModelError::InvalidInput(_))),
            "P2-f mixed: foreign linear term must give InvalidInput, got {result:?}"
        );
    }

    // Pure-linear path with a foreign variable → must also reject (not drop).
    #[test]
    fn test_p2f_pure_linear_foreign_rejected() {
        let mut m1 = Model::new("m1");
        let _x1 = m1.add_var("x", 0.0, f64::INFINITY);

        let mut m2 = Model::new("m2");
        let y2 = m2.add_var("y", 0.0, f64::INFINITY);

        // minimize(y2) on m1 — y2 is foreign, must not silently drop.
        m1.minimize(1.0 * y2);
        let result = m1.solve();
        assert!(
            matches!(result, Err(ModelError::InvalidInput(_))),
            "P2-f pure-linear: foreign variable must give InvalidInput (not silent drop), got {result:?}"
        );
    }

    // Sanity: mixed with all-local variables must succeed (no false positive).
    #[test]
    fn test_p2f_mixed_all_local_accepted() {
        let mut m = Model::new("m");
        let x = m.add_var("x", 0.0, f64::INFINITY);
        // min x^2 - 4x, x >= 0 → x*=2, obj=-4
        m.minimize(x * x + (-4.0) * x);
        let result = m.solve().unwrap();
        assert!(
            (result[x] - 2.0).abs() < EPS,
            "P2-f no false positive: x*=2, got {}",
            result[x]
        );
    }

    // Cross-term (from m1+m2) + additional linear foreign: both caught.
    #[test]
    fn test_p2f_cross_term_plus_linear_foreign() {
        let mut m1 = Model::new("m1");
        let x1 = m1.add_var("x", 0.0, f64::INFINITY);

        let mut m2 = Model::new("m2");
        let y2 = m2.add_var("y", 0.0, f64::INFINITY);

        // x1 * y2 is a cross-model quad; y2 also appears linear.
        // Both are foreign from m1's perspective.
        m1.minimize(x1 * y2 + 2.0 * y2);
        let result = m1.solve();
        assert!(
            matches!(result, Err(ModelError::InvalidInput(_))),
            "P2-f cross+linear: must give InvalidInput, got {result:?}"
        );
    }

    // maximize path: foreign linear variable also rejected via maximize.
    #[test]
    fn test_p2f_foreign_linear_maximize_rejected() {
        let mut m1 = Model::new("m1");
        let _x1 = m1.add_var("x", 0.0, 5.0);

        let mut m2 = Model::new("m2");
        let y2 = m2.add_var("y", 0.0, 5.0);

        m1.maximize(1.0 * y2);
        let result = m1.solve();
        assert!(
            matches!(result, Err(ModelError::InvalidInput(_))),
            "P2-f maximize foreign linear: must give InvalidInput, got {result:?}"
        );
    }

    // ---------------------------------------------------------------------------
    // P2-h: validate_objective rejects non-finite coefficients and constants
    // ---------------------------------------------------------------------------

    // Non-finite linear coefficient (NaN).
    #[test]
    fn test_p2h_nan_linear_coef_rejected_at_solve() {
        let mut m = Model::new("p2h_nan_coef");
        let x = m.add_var("x", 0.0, f64::INFINITY);
        m.minimize(f64::NAN * x);
        let err = m.solve().unwrap_err();
        assert!(
            matches!(err, ModelError::InvalidInput(_)),
            "P2-h: NaN linear coefficient must give InvalidInput, got {err:?}"
        );
    }

    // Non-finite linear coefficient (Inf).
    #[test]
    fn test_p2h_inf_linear_coef_rejected_at_solve() {
        let mut m = Model::new("p2h_inf_coef");
        let x = m.add_var("x", 0.0, f64::INFINITY);
        m.minimize(f64::INFINITY * x);
        let err = m.solve().unwrap_err();
        assert!(
            matches!(err, ModelError::InvalidInput(_)),
            "P2-h: Inf linear coefficient must give InvalidInput, got {err:?}"
        );
    }

    // Non-finite constant term (NaN via Expression add).
    #[test]
    fn test_p2h_nan_constant_rejected_at_solve() {
        let mut m = Model::new("p2h_nan_const");
        let x = m.add_var("x", 0.0, f64::INFINITY);
        m.minimize(1.0 * x + f64::NAN);
        let err = m.solve().unwrap_err();
        assert!(
            matches!(err, ModelError::InvalidInput(_)),
            "P2-h: NaN constant term must give InvalidInput, got {err:?}"
        );
    }

    // Non-finite constant term (Inf via Expression add).
    #[test]
    fn test_p2h_inf_constant_rejected_at_solve() {
        let mut m = Model::new("p2h_inf_const");
        let x = m.add_var("x", 0.0, f64::INFINITY);
        m.minimize(1.0 * x + f64::INFINITY);
        let err = m.solve().unwrap_err();
        assert!(
            matches!(err, ModelError::InvalidInput(_)),
            "P2-h: Inf constant term must give InvalidInput, got {err:?}"
        );
    }

    // Q-diagonal overflow → Inf stored in CscMatrix → validate_objective 287-293.
    // f64::MAX is finite so quad_to_csc's is_finite() check passes, but 2*MAX
    // overflows to INFINITY in the stored Q.  validate_objective must reject this.
    #[test]
    fn test_p2h_inf_q_coef_via_diagonal_overflow_rejected_at_solve() {
        let mut m = Model::new("p2h_inf_q_diagonal");
        let x = m.add_var("x", 0.0, f64::INFINITY);
        // f64::MAX is finite → quad_to_csc passes the is_finite check, but
        // stores 2.0 * f64::MAX = INFINITY in the diagonal Q entry.
        // validate_objective (lines 287-293) must catch this.
        m.minimize(f64::MAX * (x * x));
        let err = m.solve().unwrap_err();
        assert!(
            matches!(err, ModelError::InvalidInput(_)),
            "P2-h: Inf Q diagonal (2*MAX overflow) must give InvalidInput, got {err:?}"
        );
    }

    // DSL non-finite Q coefficient (NaN): caught by quad_to_csc → fail_dsl_quad
    // → quad_dsl_error fires at line 261, not 287-293. Still InvalidInput.
    #[test]
    fn test_p2h_nan_q_coef_dsl_rejected_at_solve() {
        let mut m = Model::new("p2h_nan_q_dsl");
        let x = m.add_var("x", 0.0, f64::INFINITY);
        // NaN coefficient is caught earlier (quad_to_csc), but the observable
        // result is still InvalidInput at solve time.
        m.minimize(f64::NAN * (x * x));
        let err = m.solve().unwrap_err();
        assert!(
            matches!(err, ModelError::InvalidInput(_)),
            "P2-h: NaN DSL Q coefficient must give InvalidInput, got {err:?}"
        );
    }

    // ---------------------------------------------------------------------------
    // P2-i: stale quad_dsl_error must not block a subsequent valid DSL quad
    // ---------------------------------------------------------------------------

    // Core repro: minimize(NaN*x*x) then minimize(x*x) then solve() → Optimal.
    #[test]
    fn test_p2i_nan_quad_replaced_by_valid_quad_is_optimal() {
        let mut model = Model::new("p2i_nan_then_valid");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        model.minimize(f64::NAN * (x * x)); // DSL fail: quad_dsl_error set
        assert!(
            model.has_quad_dsl_error(),
            "P2-i setup: error must be recorded"
        );

        model.minimize(x * x + (-4.0) * x); // new minimize: clears quad_dsl_error, installs valid Q
        assert!(
            !model.has_quad_dsl_error(),
            "P2-i: valid minimize must clear quad_dsl_error"
        );
        assert!(
            model.has_quadratic_objective(),
            "P2-i: valid Q must be installed"
        );

        // min x² - 4x, x≥0 → x*=2, obj*=-4
        let result = model.solve();
        assert!(
            result.is_ok(),
            "P2-i: valid quad after NaN must be Optimal, got {result:?}"
        );
        let r = result.unwrap();
        assert!((r[x] - 2.0).abs() < EPS, "P2-i: x*=2, got {}", r[x]);
        assert!(
            (r.objective_value - (-4.0)).abs() < EPS,
            "P2-i: obj*=-4, got {}",
            r.objective_value
        );
    }

    // NaN quad alone still gives InvalidInput (regression guard).
    #[test]
    fn test_p2i_nan_quad_alone_is_invalid() {
        let mut model = Model::new("p2i_nan_alone");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        model.minimize(f64::NAN * (x * x));
        let result = model.solve();
        assert!(
            matches!(result, Err(ModelError::InvalidInput(_))),
            "P2-i: NaN quad alone must give InvalidInput, got {result:?}"
        );
    }

    // Table: all setter orderings with ultimate valid/invalid state.
    #[test]
    fn test_p2i_setter_ordering_table() {
        // Each entry: (label, setup closure, expect_optimal)
        type Setup = fn(&mut Model, crate::variable::Variable);
        let cases: &[(&str, Setup, bool)] = &[
            // NaN → valid quad → Optimal (P2-i core)
            (
                "nan→valid_quad",
                |m, x| {
                    m.minimize(f64::NAN * (x * x));
                    m.minimize(x * x + (-4.0) * x);
                },
                true,
            ),
            // NaN → NaN → valid quad → Optimal
            (
                "nan→nan→valid_quad",
                |m, x| {
                    m.minimize(f64::NAN * (x * x));
                    m.minimize(f64::NAN * (x * x));
                    m.minimize(x * x + (-4.0) * x);
                },
                true,
            ),
            // valid quad → NaN → InvalidInput
            (
                "valid→nan",
                |m, x| {
                    m.minimize(x * x + (-4.0) * x);
                    m.minimize(f64::NAN * (x * x));
                },
                false,
            ),
            // NaN → linear → Optimal (P2-a, must still pass)
            (
                "nan→linear",
                |m, x| {
                    m.minimize(f64::NAN * (x * x));
                    m.minimize(1.0 * x); // minimize x, x≥0 → x*=0
                },
                true,
            ),
            // NaN → valid quad → NaN → InvalidInput
            (
                "nan→valid→nan",
                |m, x| {
                    m.minimize(f64::NAN * (x * x));
                    m.minimize(x * x + (-4.0) * x);
                    m.minimize(f64::NAN * (x * x));
                },
                false,
            ),
        ];

        for &(label, setup, expect_optimal) in cases {
            let mut m = Model::new(label);
            let x = m.add_var("x", 0.0, f64::INFINITY);
            setup(&mut m, x);
            let result = m.solve();
            if expect_optimal {
                assert!(
                    result.is_ok(),
                    "P2-i [{label}]: expected Optimal, got {result:?}"
                );
            } else {
                assert!(
                    matches!(result, Err(ModelError::InvalidInput(_))),
                    "P2-i [{label}]: expected InvalidInput, got {result:?}"
                );
            }
        }
    }
}
