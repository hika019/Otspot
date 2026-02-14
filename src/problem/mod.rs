//! LP problem definition

use crate::sparse::CscMatrix;
use std::fmt;

/// Type of constraint in an LP problem
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintType {
    /// Less than or equal (<=)
    Le,
    /// Greater than or equal (>=)
    Ge,
    /// Equal (==)
    Eq,
}

/// Status of the solver result
#[derive(Debug, Clone, PartialEq)]
pub enum SolveStatus {
    /// Optimal solution found
    Optimal,
    /// Problem is infeasible
    Infeasible,
    /// Problem is unbounded
    Unbounded,
}

impl fmt::Display for SolveStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolveStatus::Optimal => write!(f, "Optimal"),
            SolveStatus::Infeasible => write!(f, "Infeasible"),
            SolveStatus::Unbounded => write!(f, "Unbounded"),
        }
    }
}

/// Result of solving an LP problem
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Solve status
    pub status: SolveStatus,
    /// Optimal objective value (if optimal)
    pub objective: f64,
    /// Solution vector (if optimal)
    pub solution: Vec<f64>,
}

impl fmt::Display for SolverResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Status: {}, Objective: {}", self.status, self.objective)
    }
}

/// Linear Programming problem: min c^T x  s.t.  Ax <= b,  x >= 0
#[derive(Debug, Clone)]
pub struct LpProblem {
    /// Objective function coefficients (length num_vars)
    pub c: Vec<f64>,
    /// Constraint matrix in CSC format (num_constraints x num_vars)
    pub a: CscMatrix,
    /// Right-hand side of constraints (length num_constraints)
    pub b: Vec<f64>,
    /// Number of decision variables
    pub num_vars: usize,
    /// Number of constraints
    pub num_constraints: usize,
    /// Type of each constraint (length num_constraints)
    pub constraint_types: Vec<ConstraintType>,
    /// Bounds for each variable: (lower, upper) (length num_vars)
    pub bounds: Vec<(f64, f64)>,
    /// Optional problem name
    pub name: Option<String>,
}

impl LpProblem {
    /// Create a new LP problem with validation (backward compatible)
    ///
    /// Creates a standard LP: min c^T x  s.t.  Ax <= b,  x >= 0
    /// All constraints are assumed to be <= and all variables have bounds [0, +inf)
    ///
    /// # Arguments
    /// * `c` - Objective function coefficients
    /// * `a` - Constraint matrix in CSC format
    /// * `b` - Right-hand side of constraints
    ///
    /// # Returns
    /// * `Ok(LpProblem)` if dimensions are valid
    /// * `Err(String)` if validation fails
    pub fn new(c: Vec<f64>, a: CscMatrix, b: Vec<f64>) -> Result<Self, String> {
        let num_vars = c.len();
        let num_constraints = b.len();

        // Set defaults for backward compatibility
        let constraint_types = vec![ConstraintType::Le; num_constraints];
        let bounds = vec![(0.0, f64::INFINITY); num_vars];
        let name = None;

        Self::new_general(c, a, b, constraint_types, bounds, name)
    }

    /// Create a new LP problem with full control over constraint types and bounds
    ///
    /// # Arguments
    /// * `c` - Objective function coefficients
    /// * `a` - Constraint matrix in CSC format
    /// * `b` - Right-hand side of constraints
    /// * `constraint_types` - Type of each constraint (Le, Ge, or Eq)
    /// * `bounds` - Bounds for each variable (lower, upper)
    /// * `name` - Optional problem name
    ///
    /// # Returns
    /// * `Ok(LpProblem)` if dimensions are valid
    /// * `Err(String)` if validation fails
    pub fn new_general(
        c: Vec<f64>,
        a: CscMatrix,
        b: Vec<f64>,
        constraint_types: Vec<ConstraintType>,
        bounds: Vec<(f64, f64)>,
        name: Option<String>,
    ) -> Result<Self, String> {
        // Validate dimensions
        if c.len() != a.ncols {
            return Err(format!(
                "Dimension mismatch: c.len()={} but a.ncols={}",
                c.len(),
                a.ncols
            ));
        }
        if b.len() != a.nrows {
            return Err(format!(
                "Dimension mismatch: b.len()={} but a.nrows={}",
                b.len(),
                a.nrows
            ));
        }
        if constraint_types.len() != b.len() {
            return Err(format!(
                "Dimension mismatch: constraint_types.len()={} but num_constraints={}",
                constraint_types.len(),
                b.len()
            ));
        }
        if bounds.len() != c.len() {
            return Err(format!(
                "Dimension mismatch: bounds.len()={} but num_vars={}",
                bounds.len(),
                c.len()
            ));
        }

        Ok(LpProblem {
            num_vars: c.len(),
            num_constraints: b.len(),
            c,
            a,
            b,
            constraint_types,
            bounds,
            name,
        })
    }
}

impl fmt::Display for LpProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LP: min c^T x, {} vars, {} constraints",
            self.num_vars, self.num_constraints
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lp_problem_new_valid() {
        // 2 variables, 2 constraints
        let c = vec![1.0, 2.0];
        let a = CscMatrix::new(2, 2);
        let b = vec![5.0, 6.0];

        let lp = LpProblem::new(c, a, b).unwrap();
        assert_eq!(lp.num_vars, 2);
        assert_eq!(lp.num_constraints, 2);
    }

    #[test]
    fn test_lp_problem_new_invalid_c_dimension() {
        // c.len() = 3, but a.ncols = 2
        let c = vec![1.0, 2.0, 3.0];
        let a = CscMatrix::new(2, 2);
        let b = vec![5.0, 6.0];

        let result = LpProblem::new(c, a, b);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("c.len()"));
    }

    #[test]
    fn test_lp_problem_new_invalid_b_dimension() {
        // b.len() = 3, but a.nrows = 2
        let c = vec![1.0, 2.0];
        let a = CscMatrix::new(2, 2);
        let b = vec![5.0, 6.0, 7.0];

        let result = LpProblem::new(c, a, b);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("b.len()"));
    }

    #[test]
    fn test_lp_problem_display() {
        let c = vec![1.0, 2.0];
        let a = CscMatrix::new(2, 2);
        let b = vec![5.0, 6.0];
        let lp = LpProblem::new(c, a, b).unwrap();

        let display = format!("{}", lp);
        assert_eq!(display, "LP: min c^T x, 2 vars, 2 constraints");
    }

    #[test]
    fn test_solve_status_display() {
        assert_eq!(format!("{}", SolveStatus::Optimal), "Optimal");
        assert_eq!(format!("{}", SolveStatus::Infeasible), "Infeasible");
        assert_eq!(format!("{}", SolveStatus::Unbounded), "Unbounded");
    }

    #[test]
    fn test_solver_result_display() {
        let result = SolverResult {
            status: SolveStatus::Optimal,
            objective: 42.5,
            solution: vec![1.0, 2.0],
        };
        let display = format!("{}", result);
        assert_eq!(display, "Status: Optimal, Objective: 42.5");
    }
}
