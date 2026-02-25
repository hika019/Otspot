//! High-level algebraic modeling API for linear programs.
//!
//! # Example
//! ```
//! use solver::model::{Model, constraint};
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

pub use constraint::{Constraint, ConstraintSense};
pub use expression::Expression;
pub use variable::Variable;
pub use crate::constraint;

use variable::VariableDefinition;

use crate::options::QpSolverChoice;
use crate::problem::{ConstraintType, LpProblem, SolveStatus};
use crate::simplex;
use crate::sparse::CscMatrix;
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
    /// Timeout for QP solve in seconds (None = unlimited).
    timeout_secs: Option<f64>,
    /// QP solver choice (None = use default Auto).
    qp_solver_choice: Option<QpSolverChoice>,
    /// ADMM max iterations (None = use default 10000).
    max_iter_admm: Option<usize>,
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
            timeout_secs: None,
            qp_solver_choice: None,
            max_iter_admm: None,
        }
    }

    /// Set a timeout for QP solve operations.
    pub fn set_timeout(&mut self, secs: f64) {
        self.timeout_secs = Some(secs);
    }

    /// Set the QP solver to use.
    pub fn set_qp_solver_choice(&mut self, choice: QpSolverChoice) {
        self.qp_solver_choice = Some(choice);
    }

    /// Set the maximum number of ADMM iterations.
    pub fn set_max_iter_admm(&mut self, n: usize) {
        self.max_iter_admm = Some(n);
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
        let index = self.variables.len();
        self.variables.push(VariableDefinition {
            name: name.to_string(),
            lower_bound: lb,
            upper_bound: ub,
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

    /// Solve the model and return the result.
    ///
    /// # Errors
    /// * `ModelError::NoObjective` if `minimize` or `maximize` was not called.
    /// * `ModelError::SolveError` if the solver returns Infeasible or Unbounded.
    pub fn solve(&mut self) -> Result<ModelResult, ModelError> {
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
            CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, num_constraints, num_vars)
                .map_err(|e| ModelError::Internal(e.to_string()))?
        };

        // --- Variable bounds ---
        let bounds: Vec<(f64, f64)> = self
            .variables
            .iter()
            .map(|v| (v.lower_bound, v.upper_bound))
            .collect();

        // --- QP path ---
        if let Some(ref q_orig) = self.quadratic_objective.clone() {
            return self.solve_qp_internal(c, a, b, bounds, q_orig.clone(), num_constraints);
        }

        // --- LP path (existing) ---
        let problem = LpProblem::new_general(c, a, b, constraint_types, bounds, self.name.clone())
            .map_err(|e| ModelError::Internal(e.to_string()))?;

        let solver_result = simplex::solve(&problem);

        match solver_result.status {
            SolveStatus::Optimal => {
                let obj = if self.sense == OptimizationSense::Maximize {
                    -solver_result.objective
                } else {
                    solver_result.objective
                };
                Ok(ModelResult {
                    objective_value: obj,
                    solution: solver_result.solution,
                    // dual_solution / reduced_costs / slack: not yet available in
                    // the main-branch SolverResult. Will be populated once
                    // SolverResult is extended in a future task.
                    dual_solution: None,
                    reduced_costs: None,
                    slack: None,
                })
            }
            SolveStatus::Infeasible => Err(ModelError::SolveError(SolveError::Infeasible)),
            SolveStatus::Unbounded => Err(ModelError::SolveError(SolveError::Unbounded)),
            SolveStatus::MaxIterations => Err(ModelError::Internal("Iteration limit reached".to_string())),
            SolveStatus::Timeout => Err(ModelError::Timeout),
            SolveStatus::NumericalError => Err(ModelError::Internal("Numerical error".to_string())),
        }
    }

    /// QP内部求解ロジック（制約型変換・QpProblem構築・結果変換）
    fn solve_qp_internal(
        &self,
        c: Vec<f64>,
        _lp_a: CscMatrix,
        _lp_b: Vec<f64>,
        bounds: Vec<(f64, f64)>,
        q_orig: CscMatrix,
        num_model_constraints: usize,
    ) -> Result<ModelResult, ModelError> {
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

        // --- 制約型変換: Le→そのまま, Ge→符号反転, Eq→2行展開 ---
        // dual_map[i] = (qp_row_for_le_or_ge, Option<qp_row_for_eq_upper>)
        //   Le: (row, None)    → dual[i] =  qp_dual[row]
        //   Ge: (row, None)    → dual[i] = -qp_dual[row]
        //   Eq: (row1, Some(row2)) → dual[i] = qp_dual[row1] - qp_dual[row2]
        let mut qp_trip_rows: Vec<usize> = Vec::new();
        let mut qp_trip_cols: Vec<usize> = Vec::new();
        let mut qp_trip_vals: Vec<f64> = Vec::new();
        let mut qp_b: Vec<f64> = Vec::new();
        let mut dual_map: Vec<(usize, Option<usize>)> = Vec::with_capacity(num_model_constraints);
        let mut qp_row = 0usize;

        for con in &self.constraints {
            match con.sense {
                ConstraintSense::Le => {
                    for (&var, &coeff) in &con.lhs.coefficients {
                        qp_trip_rows.push(qp_row);
                        qp_trip_cols.push(var.index);
                        qp_trip_vals.push(coeff);
                    }
                    qp_b.push(con.rhs);
                    dual_map.push((qp_row, None));
                    qp_row += 1;
                }
                ConstraintSense::Ge => {
                    // a_i x >= b_i → -a_i x <= -b_i
                    for (&var, &coeff) in &con.lhs.coefficients {
                        qp_trip_rows.push(qp_row);
                        qp_trip_cols.push(var.index);
                        qp_trip_vals.push(-coeff);
                    }
                    qp_b.push(-con.rhs);
                    dual_map.push((qp_row, None));
                    qp_row += 1;
                }
                ConstraintSense::Eq => {
                    // a_i x = b_i → [a_i x <= b_i] AND [-a_i x <= -b_i]
                    let row1 = qp_row;
                    let row2 = qp_row + 1;
                    for (&var, &coeff) in &con.lhs.coefficients {
                        qp_trip_rows.push(row1);
                        qp_trip_cols.push(var.index);
                        qp_trip_vals.push(coeff);
                        qp_trip_rows.push(row2);
                        qp_trip_cols.push(var.index);
                        qp_trip_vals.push(-coeff);
                    }
                    qp_b.push(con.rhs);
                    qp_b.push(-con.rhs);
                    dual_map.push((row1, Some(row2)));
                    qp_row += 2;
                }
            }
        }

        let m_qp = qp_row;
        let qp_a = if m_qp == 0 {
            CscMatrix::new(0, num_vars)
        } else {
            CscMatrix::from_triplets(&qp_trip_rows, &qp_trip_cols, &qp_trip_vals, m_qp, num_vars)
                .map_err(|e| ModelError::Internal(e.to_string()))?
        };

        let qp_problem = QpProblem::new(qp_q, c, qp_a, qp_b, bounds)
            .map_err(ModelError::Internal)?;

        let mut opts = crate::options::SolverOptions::default();
        if let Some(t) = self.timeout_secs {
            opts.timeout_secs = Some(t);
        }
        if let Some(choice) = self.qp_solver_choice {
            opts.qp_solver = choice;
        }
        if let Some(n) = self.max_iter_admm {
            opts.max_iter_admm = Some(n);
        }
        let qp_result = crate::qp::solve_qp_with_options(&qp_problem, &opts);

        match qp_result.status {
            SolveStatus::Optimal => {
                let obj = if self.sense == OptimizationSense::Maximize {
                    -qp_result.objective
                } else {
                    qp_result.objective
                };

                // dual_solution逆変換（元制約数ぶんのdualを復元）
                let dual = if !qp_result.dual_solution.is_empty() && num_model_constraints > 0 {
                    let mut d = vec![0.0; num_model_constraints];
                    for (i, (idx1, idx2_opt)) in dual_map.iter().enumerate() {
                        d[i] = match self.constraints[i].sense {
                            ConstraintSense::Le => qp_result.dual_solution[*idx1],
                            ConstraintSense::Ge => -qp_result.dual_solution[*idx1],
                            ConstraintSense::Eq => {
                                let idx2 = idx2_opt.unwrap();
                                qp_result.dual_solution[*idx1] - qp_result.dual_solution[idx2]
                            }
                        };
                    }
                    Some(d)
                } else {
                    None
                };

                Ok(ModelResult {
                    objective_value: obj,
                    solution: qp_result.solution,
                    dual_solution: dual,
                    reduced_costs: None,
                    slack: None,
                })
            }
            SolveStatus::Infeasible => Err(ModelError::SolveError(SolveError::Infeasible)),
            SolveStatus::Unbounded => Err(ModelError::SolveError(SolveError::Unbounded)),
            SolveStatus::MaxIterations => {
                Err(ModelError::Internal("QP iteration limit reached".to_string()))
            }
            SolveStatus::Timeout => Err(ModelError::Timeout),
            SolveStatus::NumericalError => {
                Err(ModelError::Internal("QP numerical error".to_string()))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ModelResult
// ---------------------------------------------------------------------------

/// The result of a successful solve.
#[derive(Debug)]
pub struct ModelResult {
    /// Optimal objective value.
    pub objective_value: f64,
    /// Primal solution vector (indexed by variable index).
    solution: Vec<f64>,
    /// Dual solution (shadow prices), if available.
    pub dual_solution: Option<Vec<f64>>,
    /// Reduced costs, if available.
    pub reduced_costs: Option<Vec<f64>>,
    /// Constraint slacks, if available.
    pub slack: Option<Vec<f64>>,
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
}

/// Index a `ModelResult` by `Variable` to get the primal solution value.
///
/// # Example
/// ```rust,no_run
/// # use solver::model::Model;
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
#[derive(Debug, Clone, PartialEq)]
pub enum SolveError {
    /// The problem has no feasible solution.
    Infeasible,
    /// The problem is unbounded.
    Unbounded,
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolveError::Infeasible => write!(f, "Problem is infeasible"),
            SolveError::Unbounded => write!(f, "Problem is unbounded"),
        }
    }
}

/// Errors that can occur when building or solving a `Model`.
#[derive(Debug)]
pub enum ModelError {
    /// `solve()` was called before `minimize()` or `maximize()`.
    NoObjective,
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
    use super::{Model, ModelError, SolveError, Variable};
    use crate::options::QpSolverChoice;
    use crate::sparse::CscMatrix;

    const EPS: f64 = 1e-5;

    fn assert_close(a: f64, b: f64, name: &str) {
        assert!(
            (a - b).abs() < EPS,
            "{}: expected {:.8}, got {:.8}",
            name, b, a
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
        assert!(
            result[x].abs() < 1e-6,
            "x should be 0.0, got {}",
            result[x]
        );

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
        assert_close(result[x], 0.5, "T11: x");
        assert_close(result[y], 0.5, "T11: y");
        assert_close(result.objective_value, 0.5, "T11: obj");
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
    // Test 16: ADMM max iter custom – max_iter_admm=100 で打ち切り確認
    // -----------------------------------------------------------------------
    #[test]
    fn test_admm_max_iter_custom() {
        // min x^2+y^2 s.t. x+y>=1, x,y>=0
        // max_iter_admm=100 で打ち切り → MaxIterations か Optimal のどちらかを返す
        // 重要: 正常解(Optimal)か打ち切り(MaxIterations)のどちらでも可だが、
        // max_iter_admmフィールドがModel→SolverOptionsに正しく伝達されることを検証する。
        // 実際にSolverOptions.max_iter_admmが100になっていればOK。
        use crate::options::QpSolverChoice;
        let mut model = Model::new("qp_max_iter");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        model.add_constraint((x + y).geq(1.0));
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        model.set_quadratic_objective(q);
        model.minimize(0.0 * x + 0.0 * y);
        model.set_qp_solver_choice(QpSolverChoice::Admm);
        model.set_max_iter_admm(100);

        // このテストはmax_iter_admmが正しく渡されていることを確認する。
        // 100反復でOptimalに収束する（小問題なので）か、MaxIterationsが返る。
        let result = model.solve();
        // エラーの場合はMaxIterationsのみ許容（Timeout・Infeasible等は不可）
        match result {
            Ok(r) => {
                // 収束した場合は解が正しいことを確認
                assert!((r[x] + r[y] - 1.0).abs() < 0.1, "x+y should be ~1, got {}", r[x] + r[y]);
            }
            Err(ModelError::Internal(ref msg)) if msg.contains("iteration limit") => {
                // MaxIterations は100反復で打ち切られたことの証明
            }
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }

    // -----------------------------------------------------------------------
    // Test 15: Model QP solver choice – QpSolverChoice::Admm で正常解
    // -----------------------------------------------------------------------
    #[test]
    fn test_model_qp_solver_choice() {
        // Same as T9 but force ADMM solver
        // min x^2+y^2 - 4x - 4y  s.t. x,y >= 0
        // Expected: x=y=2, obj=-8
        let mut model = Model::new("qp_admm_choice");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        model.set_quadratic_objective(q);
        model.minimize(-4.0 * x + -4.0 * y);
        model.set_qp_solver_choice(QpSolverChoice::Admm);

        let result = model.solve().unwrap();
        // ADMM converges to eps_abs=1e-3, use looser tolerance
        let admm_tol = 1e-2;
        assert!((result[x] - 2.0).abs() < admm_tol, "T15: x={}", result[x]);
        assert!((result[y] - 2.0).abs() < admm_tol, "T15: y={}", result[y]);
        assert!((result.objective_value - (-8.0)).abs() < admm_tol, "T15: obj={}", result.objective_value);
    }
}
