//! Expression type and operator overloads for the modeling API

use std::collections::HashMap;
use std::ops::{Add, Mul, Neg, Sub};

use super::constraint::{Constraint, ConstraintSense};
use super::variable::Variable;

/// A linear expression: sum of (coefficient * variable) + constant.
///
/// Internally stored as a flat `HashMap<Variable, f64>` plus a scalar constant.
/// This representation automatically merges like terms: `x + 2*x` → `{x: 3.0}`.
#[derive(Debug, Clone, Default)]
pub struct Expression {
    pub(crate) coefficients: HashMap<Variable, f64>,
    pub(crate) constant: f64,
}

impl Expression {
    /// Create an expression representing a single scalar constant.
    pub fn from_constant(c: f64) -> Self {
        Expression {
            coefficients: HashMap::new(),
            constant: c,
        }
    }

    /// Retrieve the coefficient for a given variable (0.0 if not present).
    pub(crate) fn coefficient(&self, var: Variable) -> f64 {
        self.coefficients.get(&var).copied().unwrap_or(0.0)
    }

    /// Merge another expression into this one (in-place add).
    fn merge_add(&mut self, rhs: Expression) {
        for (var, coeff) in rhs.coefficients {
            *self.coefficients.entry(var).or_insert(0.0) += coeff;
        }
        self.constant += rhs.constant;
    }

    // --- Constraint builders ---

    fn constraint(self, rhs: impl Into<Expression>, sense: ConstraintSense) -> Constraint {
        let mut lhs = self;
        let mut rhs_expr = rhs.into();
        for (var, coeff) in rhs_expr.coefficients.drain() {
            *lhs.coefficients.entry(var).or_insert(0.0) -= coeff;
        }
        let rhs_val = rhs_expr.constant - lhs.constant;
        lhs.constant = 0.0;
        Constraint {
            lhs,
            rhs: rhs_val,
            sense,
        }
    }

    /// Create a `<=` constraint: `self <= rhs`.
    pub fn leq(self, rhs: impl Into<Expression>) -> Constraint {
        self.constraint(rhs, ConstraintSense::Le)
    }

    /// Create a `>=` constraint: `self >= rhs`.
    pub fn geq(self, rhs: impl Into<Expression>) -> Constraint {
        self.constraint(rhs, ConstraintSense::Ge)
    }

    /// Create an `==` constraint: `self == rhs`.
    pub fn eq_constraint(self, rhs: impl Into<Expression>) -> Constraint {
        self.constraint(rhs, ConstraintSense::Eq)
    }
}

// --- From conversions ---

impl From<Variable> for Expression {
    fn from(var: Variable) -> Self {
        let mut coefficients = HashMap::new();
        coefficients.insert(var, 1.0);
        Expression {
            coefficients,
            constant: 0.0,
        }
    }
}

impl From<f64> for Expression {
    fn from(c: f64) -> Self {
        Expression::from_constant(c)
    }
}

impl From<i32> for Expression {
    fn from(c: i32) -> Self {
        Expression::from_constant(c as f64)
    }
}

// --- Negation ---

impl Neg for Expression {
    type Output = Expression;
    fn neg(mut self) -> Expression {
        for coeff in self.coefficients.values_mut() {
            *coeff = -*coeff;
        }
        self.constant = -self.constant;
        self
    }
}

impl Neg for Variable {
    type Output = Expression;
    fn neg(self) -> Expression {
        -Expression::from(self)
    }
}

// --- Expression + Expression ---

impl Add for Expression {
    type Output = Expression;
    fn add(mut self, rhs: Expression) -> Expression {
        self.merge_add(rhs);
        self
    }
}

// --- Expression - Expression ---

impl Sub for Expression {
    type Output = Expression;
    fn sub(self, rhs: Expression) -> Expression {
        self + (-rhs)
    }
}

// --- f64 * Expression ---

impl Mul<Expression> for f64 {
    type Output = Expression;
    fn mul(self, mut rhs: Expression) -> Expression {
        for coeff in rhs.coefficients.values_mut() {
            *coeff *= self;
        }
        rhs.constant *= self;
        rhs
    }
}

impl Mul<f64> for Expression {
    type Output = Expression;
    fn mul(self, rhs: f64) -> Expression {
        rhs * self
    }
}

// --- f64 * Variable → Expression ---

impl Mul<Variable> for f64 {
    type Output = Expression;
    fn mul(self, rhs: Variable) -> Expression {
        let mut coefficients = HashMap::new();
        coefficients.insert(rhs, self);
        Expression {
            coefficients,
            constant: 0.0,
        }
    }
}

impl Mul<f64> for Variable {
    type Output = Expression;
    fn mul(self, rhs: f64) -> Expression {
        rhs * self
    }
}

// --- Variable + Variable → Expression ---

impl Add<Variable> for Variable {
    type Output = Expression;
    fn add(self, rhs: Variable) -> Expression {
        Expression::from(self) + Expression::from(rhs)
    }
}

impl Sub<Variable> for Variable {
    type Output = Expression;
    fn sub(self, rhs: Variable) -> Expression {
        Expression::from(self) - Expression::from(rhs)
    }
}

// --- Variable + Expression / Expression + Variable ---

impl Add<Expression> for Variable {
    type Output = Expression;
    fn add(self, rhs: Expression) -> Expression {
        Expression::from(self) + rhs
    }
}

impl Add<Variable> for Expression {
    type Output = Expression;
    fn add(mut self, rhs: Variable) -> Expression {
        *self.coefficients.entry(rhs).or_insert(0.0) += 1.0;
        self
    }
}

impl Sub<Expression> for Variable {
    type Output = Expression;
    fn sub(self, rhs: Expression) -> Expression {
        Expression::from(self) - rhs
    }
}

impl Sub<Variable> for Expression {
    type Output = Expression;
    fn sub(mut self, rhs: Variable) -> Expression {
        *self.coefficients.entry(rhs).or_insert(0.0) -= 1.0;
        self
    }
}

// --- f64 + Expression / Expression + f64 ---

impl Add<f64> for Expression {
    type Output = Expression;
    fn add(mut self, rhs: f64) -> Expression {
        self.constant += rhs;
        self
    }
}

impl Add<Expression> for f64 {
    type Output = Expression;
    fn add(self, mut rhs: Expression) -> Expression {
        rhs.constant += self;
        rhs
    }
}

impl Sub<f64> for Expression {
    type Output = Expression;
    fn sub(mut self, rhs: f64) -> Expression {
        self.constant -= rhs;
        self
    }
}

impl Sub<Expression> for f64 {
    type Output = Expression;
    fn sub(self, rhs: Expression) -> Expression {
        self + (-rhs)
    }
}

// --- f64 + Variable / Variable + f64 ---

impl Add<f64> for Variable {
    type Output = Expression;
    fn add(self, rhs: f64) -> Expression {
        Expression::from(self) + rhs
    }
}

impl Add<Variable> for f64 {
    type Output = Expression;
    fn add(self, rhs: Variable) -> Expression {
        self + Expression::from(rhs)
    }
}

impl Sub<f64> for Variable {
    type Output = Expression;
    fn sub(self, rhs: f64) -> Expression {
        Expression::from(self) - rhs
    }
}

impl Sub<Variable> for f64 {
    type Output = Expression;
    fn sub(self, rhs: Variable) -> Expression {
        self - Expression::from(rhs)
    }
}
