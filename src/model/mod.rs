//! High-level algebraic modeling API for linear programs.
//!
//! # Example
//! ```
//! use otspot::model::{Model, constraint};
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

pub mod constraint;
pub mod expression;
pub mod variable;

pub use crate::constraint;
pub use constraint::{Constraint, ConstraintSense};
pub use expression::Expression;
pub use variable::{VarKind, Variable};

use variable::VariableDefinition;

use crate::options::Tolerance;
use crate::problem::{ConstraintType, LpProblem, SolveStatus};
use crate::sparse::CscMatrix;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::Index;

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
    name: Option<String>,
    variables: Vec<VariableDefinition>,
    constraints: Vec<Constraint>,
    objective: Option<Expression>,
    sense: OptimizationSense,
    /// Quadratic objective Q matrix for QP problems (None = LP mode).
    /// Convention: min 1/2 x^T Q x + c^T x  ("1/2あり" standard).
    quadratic_objective: Option<CscMatrix>,
    invalid_inputs: BTreeMap<&'static str, String>,
    /// Timeout for QP solve in seconds (None = unlimited).
    timeout_secs: Option<f64>,
    /// Ruiz スケーリング有効/無効（None = default true）
    use_ruiz_scaling: Option<bool>,
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
            name: Some(name.to_string()),
            variables: Vec::new(),
            constraints: Vec::new(),
            objective: None,
            sense: OptimizationSense::Minimize,
            quadratic_objective: None,
            invalid_inputs: BTreeMap::new(),
            timeout_secs: None,
            use_ruiz_scaling: None,
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

    /// Ruiz equilibration スケーリングの有効/無効を設定する（デフォルト: true）
    pub fn set_use_ruiz_scaling(&mut self, flag: bool) {
        self.use_ruiz_scaling = Some(flag);
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
    /// # Arguments
    /// * `name` - Variable name (for display purposes)
    /// * `lb`   - Lower bound
    /// * `ub`   - Upper bound (use `f64::INFINITY` for unbounded above)
    ///
    /// # Returns
    /// A `Variable` handle that can be used in expressions.
    pub fn add_var(&mut self, name: &str, lb: f64, ub: f64) -> Variable {
        self.add_var_with_kind(name, lb, ub, VarKind::Continuous)
    }

    /// Add an integer decision variable (must take integral values within `[lb, ub]`).
    ///
    /// Presence of any integer/binary variable routes `solve()` through the
    /// MILP/MIQP branch-and-bound solver instead of the continuous LP/QP path.
    pub fn add_int_var(&mut self, name: &str, lb: f64, ub: f64) -> Variable {
        self.add_var_with_kind(name, lb, ub, VarKind::Integer)
    }

    /// Add a binary decision variable (integer, fixed to the `{0, 1}` box).
    pub fn add_binary_var(&mut self, name: &str) -> Variable {
        self.add_var_with_kind(name, 0.0, 1.0, VarKind::Binary)
    }

    fn add_var_with_kind(&mut self, _name: &str, lb: f64, ub: f64, kind: VarKind) -> Variable {
        let index = self.variables.len();
        self.variables.push(VariableDefinition {
            lower_bound: lb,
            upper_bound: ub,
            kind,
        });
        Variable { index }
    }

    /// Add a constraint to the model.
    pub fn add_constraint(&mut self, c: Constraint) -> &mut Self {
        self.constraints.push(c);
        self
    }

    /// Set the objective to minimize the given expression.
    pub fn minimize(&mut self, obj: impl Into<Expression>) -> &mut Self {
        self.objective = Some(obj.into());
        self.sense = OptimizationSense::Minimize;
        self
    }

    /// Set the objective to maximize the given expression.
    pub fn maximize(&mut self, obj: impl Into<Expression>) -> &mut Self {
        self.objective = Some(obj.into());
        self.sense = OptimizationSense::Maximize;
        self
    }

    /// Set the quadratic objective Q matrix for QP problems.
    ///
    /// **Convention** ("1/2あり"): the objective is min 1/2 x^T Q x + c^T x
    /// where c is specified via `minimize()` or `maximize()`.
    ///
    /// If Q is not set, `solve()` runs as a standard LP.
    ///
    /// # Note
    /// For `maximize()` QP, Q must be negative semi-definite (NSD).
    /// Providing a PSD Q with `maximize()` may yield an unbounded problem.
    pub fn set_quadratic_objective(&mut self, q: CscMatrix) -> &mut Self {
        self.quadratic_objective = Some(q);
        self
    }

    /// 対角 Q 行列を `diag` ベクトルから構築して設定する ergonomic helper。
    /// `diag.len()` は変数数と一致する必要がある。
    pub fn set_diagonal_q(&mut self, diag: &[f64]) -> &mut Self {
        if let Err(err) = self.try_set_diagonal_q(diag) {
            self.record_input_error("diagonal_q", err);
        }
        self
    }

    /// Set a diagonal Q matrix, returning an error instead of panicking on invalid input.
    pub fn try_set_diagonal_q(&mut self, diag: &[f64]) -> Result<&mut Self, ModelError> {
        let n = diag.len();
        if n != self.variables.len() {
            return Err(ModelError::InvalidInput(format!(
                "set_diagonal_q: diag length {} != variable count {}",
                n,
                self.variables.len()
            )));
        }
        let idx: Vec<usize> = (0..n).collect();
        let q = CscMatrix::from_triplets(&idx, &idx, diag, n, n)
            .map_err(|e| ModelError::InvalidInput(e.to_string()))?;
        self.invalid_inputs.remove("diagonal_q");
        Ok(self.set_quadratic_objective(q))
    }

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

        let num_vars = self.variables.len();
        let num_constraints = self.constraints.len();

        // --- Build objective vector c ---
        let mut c: Vec<f64> = (0..num_vars)
            .map(|i| obj_expr.coefficient(Variable { index: i }))
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
            return self.solve_qp_internal(c, a, b, bounds, q_orig.clone(), num_constraints);
        }

        // --- LP path (existing) ---
        let problem = LpProblem::new_general(c, a, b, constraint_types, bounds, self.name.clone())
            .map_err(|e| ModelError::Internal(e.to_string()))?;

        let mut lp_opts = crate::options::SolverOptions::default();
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
        let solver_result = crate::lp::solve_lp_with(&problem, &lp_opts);

        // SolverResult の dual/rc/slack は extract_dual_info によって
        // 元の制約空間 (Eq/Ge/Le) と変数空間 (bounds 込み) で復元済み。
        let lp_extras = |sr: &crate::problem::SolverResult| {
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
            oriented + self.obj_offset
        };
        let build_ok = |sr: crate::problem::SolverResult| {
            let (dual, rc, slack) = lp_extras(&sr);
            let status = sr.status.clone();
            ModelResult {
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
            SolveStatus::Infeasible => Err(ModelError::SolveError(SolveError::Infeasible)),
            SolveStatus::Unbounded => Err(ModelError::SolveError(SolveError::Unbounded)),
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
            SolveStatus::NumericalError => Err(ModelError::SolveError(SolveError::NumericalError)),
            SolveStatus::NonConvex(msg) => {
                Err(ModelError::Internal(format!("Non-convex QP: {}", msg)))
            }
            SolveStatus::LocallyOptimal
            | SolveStatus::NonconvexLocal
            | SolveStatus::NonconvexGlobal => Err(ModelError::Internal(
                "Unexpected nonconvex status on LP path".to_string(),
            )),
            SolveStatus::NotSupported(msg) => Err(ModelError::Internal(msg)),
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
    ) -> Result<crate::qp::QpProblem, ModelError> {
        use crate::qp::QpProblem;

        let num_vars = self.variables.len();

        // maximize QP: negate Q (Q→-Q), c is already negated by solve()
        let qp_q = if self.sense == OptimizationSense::Maximize {
            let mut q_neg = q_orig.clone();
            for v in q_neg.values.iter_mut() {
                *v = -*v;
            }
            q_neg
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
            .map_err(|e| ModelError::Internal(e.to_string()))?;
        // offset は signed_obj で post-solve 加算するため solver には渡さない。
        qp_problem.obj_offset = 0.0;
        Ok(qp_problem)
    }

    /// QP内部求解ロジック（QpProblem構築・求解・結果変換）
    fn solve_qp_internal(
        &self,
        c: Vec<f64>,
        _lp_a: CscMatrix,
        _lp_b: Vec<f64>,
        bounds: Vec<(f64, f64)>,
        q_orig: CscMatrix,
        num_model_constraints: usize,
    ) -> Result<ModelResult, ModelError> {
        let qp_problem = self.build_qp_problem(c, bounds, q_orig)?;

        let mut opts = crate::options::SolverOptions::default();
        if let Some(t) = self.timeout_secs {
            opts.timeout_secs = Some(t);
        }
        if let Some(flag) = self.use_ruiz_scaling {
            opts.use_ruiz_scaling = flag;
        }
        if let Some(tol) = self.tolerance {
            opts.tolerance = Some(tol);
        }
        if let Some(n) = self.threads {
            opts.threads = n;
        }
        let qp_result = crate::qp::solve_qp_with(&qp_problem, &opts);
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
            oriented + self.obj_offset
        };
        let build_ok = |status: SolveStatus,
                        raw_obj: f64,
                        sol: Vec<f64>,
                        dual: Option<Vec<f64>>,
                        bd: Vec<f64>| {
            let proof = SolutionProof::from_status(&status);
            ModelResult {
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
                qp_result.status.clone(),
                qp_result.objective,
                qp_result.solution.clone(),
                fold_dual(&qp_result.dual_solution),
                qp_result.bound_duals,
            )),
            SolveStatus::Infeasible => Err(ModelError::SolveError(SolveError::Infeasible)),
            SolveStatus::Unbounded => Err(ModelError::SolveError(SolveError::Unbounded)),
            SolveStatus::MaxIterations => {
                if qp_result.solution.is_empty() {
                    Err(ModelError::SolveError(SolveError::MaxIterations))
                } else {
                    Ok(build_ok(
                        qp_result.status.clone(),
                        qp_result.objective,
                        qp_result.solution.clone(),
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
                        qp_result.status.clone(),
                        qp_result.objective,
                        qp_result.solution.clone(),
                        fold_dual(&qp_result.dual_solution),
                        qp_result.bound_duals,
                    ))
                }
            }
            SolveStatus::Timeout => Err(ModelError::Timeout),
            SolveStatus::NumericalError => Err(ModelError::SolveError(SolveError::NumericalError)),
            SolveStatus::NonConvex(msg) => {
                Err(ModelError::Internal(format!("Non-convex QP: {}", msg)))
            }
            // LocallyOptimal / NonconvexLocal / NonconvexGlobal: 解はあるが (NonconvexGlobal を
            // 除き) global proof なし。Model API 経由では caller が status を観測できない
            // 制約上、Ok(...) で解を返し objective_value を返す (caller は obj quality を別途
            // 検証する責任)。NonconvexGlobal は global proof 済 → 安全に Ok。
            SolveStatus::LocallyOptimal
            | SolveStatus::NonconvexLocal
            | SolveStatus::NonconvexGlobal => Ok(build_ok(
                qp_result.status.clone(),
                qp_result.objective,
                qp_result.solution.clone(),
                fold_dual(&qp_result.dual_solution),
                qp_result.bound_duals,
            )),
            SolveStatus::NotSupported(msg) => Err(ModelError::Internal(msg)),
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
        let mut opts = crate::options::SolverOptions::default();
        if let Some(t) = self.timeout_secs {
            opts.timeout_secs = Some(t);
        }
        if let Some(tol) = self.tolerance {
            opts.tolerance = Some(tol);
        }
        if let Some(n) = self.threads {
            opts.threads = n;
        }
        let cfg = crate::options::MipConfig::default();

        let result = if let Some(ref q_orig) = self.quadratic_objective.clone() {
            // MIQP: convex QP relaxation per node.
            if let Some(flag) = self.use_ruiz_scaling {
                opts.use_ruiz_scaling = flag;
            }
            let qp = self.build_qp_problem(c, bounds, q_orig.clone())?;
            let miqp = crate::mip::MiqpProblem::new(qp, integer_vars.clone())
                .map_err(|e| ModelError::Internal(e.to_string()))?;
            crate::mip::solve_miqp(&miqp, &opts, &cfg)
        } else {
            // MILP: LP relaxation per node.
            if let Some(flag) = self.presolve {
                opts.presolve = flag;
            }
            let lp = LpProblem::new_general(c, a, b, constraint_types, bounds, self.name.clone())
                .map_err(|e| ModelError::Internal(e.to_string()))?;
            let milp = crate::mip::MilpProblem::new(lp, integer_vars.clone())
                .map_err(|e| ModelError::Internal(e.to_string()))?;
            crate::mip::solve_milp(&milp, &opts, &cfg)
        };

        self.finish_mip(result, &integer_vars)
    }

    /// Convert a MIP `SolverResult` to a `ModelResult`: apply objective sign /
    /// offset, round integer components, and map the status. Shared by MILP/MIQP.
    fn finish_mip(
        &self,
        result: crate::problem::SolverResult,
        integer_vars: &[usize],
    ) -> Result<ModelResult, ModelError> {
        let signed_obj = |raw: f64| -> f64 {
            let oriented = if self.sense == OptimizationSense::Maximize {
                -raw
            } else {
                raw
            };
            oriented + self.obj_offset
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

        let build_ok = |sr: crate::problem::SolverResult| {
            let status = sr.status.clone();
            ModelResult {
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
            SolveStatus::Infeasible => Err(ModelError::SolveError(SolveError::Infeasible)),
            SolveStatus::Unbounded => Err(ModelError::SolveError(SolveError::Unbounded)),
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
            SolveStatus::NumericalError => Err(ModelError::SolveError(SolveError::NumericalError)),
            // Non-convex MIQP (non-PSD Q) is out of scope — surface it as an error.
            SolveStatus::NonConvex(msg) => {
                Err(ModelError::Internal(format!("Non-convex MIQP: {}", msg)))
            }
            SolveStatus::LocallyOptimal
            | SolveStatus::NonconvexLocal
            | SolveStatus::NonconvexGlobal => Err(ModelError::Internal(
                "Unexpected nonconvex status on MIP path".to_string(),
            )),
            SolveStatus::NotSupported(msg) => Err(ModelError::Internal(msg)),
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

fn validate_timeout(secs: f64) -> Result<(), ModelError> {
    if secs.is_finite() && secs >= 0.0 {
        Ok(())
    } else {
        Err(ModelError::InvalidInput(format!(
            "timeout must be finite and non-negative, got {secs}"
        )))
    }
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
        }
    }
}

/// The result of a successful solve.
#[derive(Debug)]
pub struct ModelResult {
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
    pub stats: crate::problem::SolveStats,
}

impl ModelResult {
    /// Get the primal value of a variable.
    pub fn value(&self, var: Variable) -> f64 {
        self.solution[var.index]
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
/// # use otspot::model::Model;
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
    /// An internal error (e.g., dimension mismatch in matrix construction).
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
    use super::{Model, ModelError, SolutionProof, SolveError, Variable};
    use crate::problem::SolveStatus;
    use crate::sparse::CscMatrix;

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
        // min 1/2*[[2,0],[0,2]]*[x,y] + [-4,-4]*[x,y] = x^2+y^2 - 4x - 4y
        // Unconstrained min: x=y=2, obj = 4+4-8-8 = -8
        let mut model = Model::new("qp_basic");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        model.set_quadratic_objective(q);
        model.minimize(-4.0 * x + -4.0 * y);

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
        // Q=2I, c=[0,0], Eq: x+y=1
        // Expected: x=y=0.5, obj=0.5
        let mut model = Model::new("qp_eq");
        let x = model.add_var("x", f64::NEG_INFINITY, f64::INFINITY);
        let y = model.add_var("y", f64::NEG_INFINITY, f64::INFINITY);
        model.add_constraint((x + y).eq_constraint(1.0));
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        model.set_quadratic_objective(q);
        model.minimize(0.0 * x + 0.0 * y);

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
        // Q=2I, c=[0,0], Ge: x+y>=1
        // Same solution as equality case: x=y=0.5, obj=0.5
        let mut model = Model::new("qp_ge");
        let x = model.add_var("x", f64::NEG_INFINITY, f64::INFINITY);
        let y = model.add_var("y", f64::NEG_INFINITY, f64::INFINITY);
        model.add_constraint((x + y).geq(1.0));
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        model.set_quadratic_objective(q);
        model.minimize(0.0 * x + 0.0 * y);

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
        // max -(x^2+y^2)
        // Q_orig = [[-2,0],[0,-2]] (NSD, "1/2あり": 1/2*(-2)*(x^2+y^2) = -(x^2+y^2))
        // c_orig = [0, 0]
        // constraint: x+y >= 1, x,y >= 0
        // Expected: x=y=0.5, obj = -(0.25+0.25) = -0.5
        let mut model = Model::new("qp_maximize");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        model.add_constraint((x + y).geq(1.0));
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, -2.0], 2, 2).unwrap();
        model.set_quadratic_objective(q);
        model.maximize(0.0 * x + 0.0 * y);

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
        // min (x-2)^2+(y-2)^2 = 1/2*[[2,0],[0,2]]*[x,y]^T + [-4,-4]*[x,y] + const
        // Q=2I, c=[-4,-4], bounds=[0,1]
        // Unconstrained min: x=y=2 → clipped to ub=1
        // Expected: x=y=1, obj = 1/2*2*(1+1) + (-4-4)*1 = 2-8 = -6
        let mut model = Model::new("qp_box");
        let x = model.add_var("x", 0.0, 1.0);
        let y = model.add_var("y", 0.0, 1.0);
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        model.set_quadratic_objective(q);
        model.minimize(-4.0 * x + -4.0 * y);

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
        // Large QP that should trigger timeout with 0.001s limit.
        // Use a well-defined small problem but set an extremely short timeout.
        let mut model = Model::new("qp_timeout");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        model.set_quadratic_objective(q);
        model.minimize(-4.0 * x + -4.0 * y);
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
        // Lagrangian: x^2+y^2 + λ(x+y-1)
        // KKT: 2x + λ = 0, 2y + λ = 0 → x=y=-λ/2
        // x+y=1 → -λ=1 → λ=-1, x=y=0.5
        // dual of Eq constraint (shadow price) = λ = -1
        let mut model = Model::new("qp_eq_dual");
        let x = model.add_var("x", f64::NEG_INFINITY, f64::INFINITY);
        let y = model.add_var("y", f64::NEG_INFINITY, f64::INFINITY);
        model.add_constraint((x + y).eq_constraint(1.0));
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        model.set_quadratic_objective(q);
        model.minimize(0.0 * x + 0.0 * y);

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
            let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &q_diag, 2, 2).unwrap();
            model.set_quadratic_objective(q);
            model.minimize(c[0] * x + c[1] * y);

            let result = model.solve();
            match result {
                Ok(r) => {
                    assert_eq!(
                        r.status,
                        crate::problem::SolveStatus::LocallyOptimal,
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
    fn test_model_qp_feasible_unproven_proof() {
        use crate::options::Tolerance;

        // (name, q_diag, (lb,ub), c)
        let cases: &[(&str, [f64; 2], (f64, f64), [f64; 2])] = &[
            // Convex PSD Q=2I, c=[-4,-4]. IPM converges (residuals ~1e-6) but
            // Custom(1e-200) makes satisfies_eps always false → SuboptimalSolution.
            ("convex_2x2_tight_tol", [2.0, 2.0], (0.0, f64::INFINITY), [-4.0, -4.0]),
            // Convex PSD Q=4I, c=[0,-2] with box bounds.
            ("convex_box_tight_tol", [4.0, 4.0], (0.0, 3.0), [0.0, -2.0]),
        ];

        for &(name, q_diag, (lb, ub), c) in cases {
            let mut model = Model::new(name);
            let x = model.add_var("x0", lb, ub);
            let y = model.add_var("x1", lb, ub);
            let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &q_diag, 2, 2).unwrap();
            model.set_quadratic_objective(q);
            model.minimize(c[0] * x + c[1] * y);
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
                        r.proof, r.status
                    );
                    assert!(
                        !r.has_global_optimality_proof(),
                        "[{name}] has_global_optimality_proof must be false for FeasibleUnproven"
                    );
                    assert!(!r.solution.is_empty(), "[{name}] solution must be non-empty");
                }
                Err(e) => panic!("[{name}] unexpected Err: {e:?}"),
            }
        }
    }
}
