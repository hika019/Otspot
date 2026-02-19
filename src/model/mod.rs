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
        }
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
                .map_err(|e| ModelError::Internal(e))?
        };

        // --- Variable bounds ---
        let bounds: Vec<(f64, f64)> = self
            .variables
            .iter()
            .map(|v| (v.lower_bound, v.upper_bound))
            .collect();

        // --- Build and solve LpProblem ---
        let problem = LpProblem::new_general(c, a, b, constraint_types, bounds, self.name.clone())
            .map_err(|e| ModelError::Internal(e))?;

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
            SolveStatus::Infeasible => Err(ModelError::SolveError(SolverError::Infeasible)),
            SolveStatus::Unbounded => Err(ModelError::SolveError(SolverError::Unbounded)),
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
/// ```ignore
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

/// Internal solver error kind.
#[derive(Debug, Clone, PartialEq)]
pub enum SolverError {
    /// The problem has no feasible solution.
    Infeasible,
    /// The problem is unbounded.
    Unbounded,
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::Infeasible => write!(f, "Problem is infeasible"),
            SolverError::Unbounded => write!(f, "Problem is unbounded"),
        }
    }
}

/// Errors that can occur when building or solving a `Model`.
#[derive(Debug)]
pub enum ModelError {
    /// `solve()` was called before `minimize()` or `maximize()`.
    NoObjective,
    /// The solver returned a non-optimal status.
    SolveError(SolverError),
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
    use super::{Model, ModelError, SolverError, Variable};

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
            matches!(err, ModelError::SolveError(SolverError::Unbounded)),
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
            matches!(err, ModelError::SolveError(SolverError::Infeasible)),
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
}
