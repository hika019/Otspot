//! Canonical solver-independent optimization problem representation.

use otspot_num::{validate_csc, CscMatrixView, NumericError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableKind {
    Continuous,
    Integer,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Variable {
    pub lower: f64,
    pub upper: f64,
    pub kind: VariableKind,
}

impl Variable {
    pub fn continuous(lower: f64, upper: f64) -> Self {
        Self {
            lower,
            upper,
            kind: VariableKind::Continuous,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Objective<M> {
    /// `None` means a linear objective; otherwise `1/2 x^T Q x`.
    pub quadratic: Option<M>,
    pub linear: Vec<f64>,
    pub offset: f64,
}

#[derive(Debug, Clone)]
pub struct ConstraintSystem<M> {
    /// General row-bounded form: `lower <= A x <= upper`.
    pub matrix: M,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

/// One row-bounded quadratic constraint:
/// `lower[row] <= 1/2 x^T Q x + A[row]^T x <= upper[row]`.
///
/// The affine part and bounds stay in [`ConstraintSystem`]; `linear_row`
/// overlays the quadratic term on that row. This avoids duplicating the sparse
/// affine row and prevents treating a QCQP row as both linear and quadratic.
#[derive(Debug, Clone)]
pub struct QuadraticConstraint<M> {
    pub quadratic: M,
    pub linear_row: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cone {
    Zero(usize),
    Nonnegative(usize),
    SecondOrder(usize),
    RotatedSecondOrder(usize),
}

impl Cone {
    pub fn dimension(self) -> usize {
        match self {
            Self::Zero(n)
            | Self::Nonnegative(n)
            | Self::SecondOrder(n)
            | Self::RotatedSecondOrder(n) => n,
        }
    }
}

/// Affine conic system in canonical form `Gx + s = h`, `s ∈ K`.
#[derive(Debug, Clone)]
pub struct ConicSystem<M> {
    pub matrix: M,
    pub rhs: Vec<f64>,
    pub cones: Vec<Cone>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemClass {
    Lp,
    Qp,
    Qcqp,
    Socp,
    Milp,
    Miqp,
    Miqcp,
    Misocp,
}

#[derive(Debug, Clone)]
pub struct OptimizationProblem<M> {
    pub variables: Vec<Variable>,
    pub objective: Objective<M>,
    pub constraints: ConstraintSystem<M>,
    pub quadratic_constraints: Vec<QuadraticConstraint<M>>,
    pub conic: Option<ConicSystem<M>>,
}

impl<M: CscMatrixView> OptimizationProblem<M> {
    pub fn class(&self) -> ProblemClass {
        let quadratic = self.objective.quadratic.is_some();
        let qcqp = !self.quadratic_constraints.is_empty();
        let conic = self.conic.is_some();
        let integer = self
            .variables
            .iter()
            .any(|v| v.kind != VariableKind::Continuous);
        match (conic, qcqp, quadratic, integer) {
            (true, _, _, false) => ProblemClass::Socp,
            (true, _, _, true) => ProblemClass::Misocp,
            (false, true, _, false) => ProblemClass::Qcqp,
            (false, true, _, true) => ProblemClass::Miqcp,
            (false, false, false, false) => ProblemClass::Lp,
            (false, false, true, false) => ProblemClass::Qp,
            (false, false, false, true) => ProblemClass::Milp,
            (false, false, true, true) => ProblemClass::Miqp,
        }
    }

    /// The single validation boundary for canonical problems.
    pub fn validate(&self) -> Result<(), NumericError> {
        let n = self.variables.len();
        let m = self.constraints.lower.len();

        validate_csc(&self.constraints.matrix)?;
        if self.constraints.matrix.ncols() != n {
            return Err(NumericError::DimensionMismatch {
                field: "constraints.matrix.ncols",
                expected: n,
                got: self.constraints.matrix.ncols(),
            });
        }
        if self.constraints.matrix.nrows() != m {
            return Err(NumericError::DimensionMismatch {
                field: "constraints.matrix.nrows",
                expected: m,
                got: self.constraints.matrix.nrows(),
            });
        }
        if self.constraints.upper.len() != m {
            return Err(NumericError::DimensionMismatch {
                field: "constraints.upper",
                expected: m,
                got: self.constraints.upper.len(),
            });
        }
        if self.objective.linear.len() != n {
            return Err(NumericError::DimensionMismatch {
                field: "objective.linear",
                expected: n,
                got: self.objective.linear.len(),
            });
        }
        if !self.objective.offset.is_finite() {
            return Err(NumericError::NonFinite {
                field: "objective.offset",
                index: 0,
            });
        }
        for (i, value) in self.objective.linear.iter().enumerate() {
            if !value.is_finite() {
                return Err(NumericError::NonFinite {
                    field: "objective.linear",
                    index: i,
                });
            }
        }
        if let Some(q) = &self.objective.quadratic {
            validate_csc(q)?;
            if q.nrows() != n || q.ncols() != n {
                return Err(NumericError::DimensionMismatch {
                    field: "objective.quadratic",
                    expected: n,
                    got: q.nrows().max(q.ncols()),
                });
            }
        }
        for constraint in &self.quadratic_constraints {
            validate_csc(&constraint.quadratic)?;
            if constraint.quadratic.nrows() != n || constraint.quadratic.ncols() != n {
                return Err(NumericError::DimensionMismatch {
                    field: "quadratic_constraints.quadratic",
                    expected: n,
                    got: constraint
                        .quadratic
                        .nrows()
                        .max(constraint.quadratic.ncols()),
                });
            }
            if constraint.linear_row >= m {
                return Err(NumericError::IndexOutOfBounds {
                    context: "quadratic constraint row",
                    index: constraint.linear_row,
                    bound: m,
                });
            }
        }
        if let Some(conic) = &self.conic {
            validate_csc(&conic.matrix)?;
            if conic.matrix.ncols() != n {
                return Err(NumericError::DimensionMismatch {
                    field: "conic.matrix.ncols",
                    expected: n,
                    got: conic.matrix.ncols(),
                });
            }
            if conic.matrix.nrows() != conic.rhs.len() {
                return Err(NumericError::DimensionMismatch {
                    field: "conic.rhs",
                    expected: conic.matrix.nrows(),
                    got: conic.rhs.len(),
                });
            }
            let cone_dimension = conic.cones.iter().map(|cone| cone.dimension()).sum();
            if cone_dimension != conic.rhs.len() {
                return Err(NumericError::DimensionMismatch {
                    field: "conic.cones",
                    expected: conic.rhs.len(),
                    got: cone_dimension,
                });
            }
        }
        for (i, variable) in self.variables.iter().enumerate() {
            if variable.lower.is_nan() || variable.upper.is_nan() || variable.lower > variable.upper
            {
                return Err(NumericError::InvalidBounds {
                    context: "variable",
                    index: i,
                    lower: variable.lower,
                    upper: variable.upper,
                });
            }
            if variable.kind == VariableKind::Binary
                && (variable.lower < 0.0 || variable.upper > 1.0)
            {
                return Err(NumericError::InvalidBounds {
                    context: "binary variable",
                    index: i,
                    lower: variable.lower,
                    upper: variable.upper,
                });
            }
        }
        for (i, (&lower, &upper)) in self
            .constraints
            .lower
            .iter()
            .zip(&self.constraints.upper)
            .enumerate()
        {
            if lower.is_nan() || upper.is_nan() || lower > upper {
                return Err(NumericError::InvalidBounds {
                    context: "linear row",
                    index: i,
                    lower,
                    upper,
                });
            }
        }
        Ok(())
    }
}
