//! Constraint types and the `constraint!` macro

use super::expression::Expression;

/// Sense (direction) of a linear constraint.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintSense {
    /// Less-than-or-equal: `lhs <= rhs`
    Le,
    /// Greater-than-or-equal: `lhs >= rhs`
    Ge,
    /// Equality: `lhs == rhs`
    Eq,
}

/// A linear constraint: `lhs sense rhs`.
///
/// Stored in normalized form: `lhs` contains only variable terms (no constant),
/// and `rhs` is a scalar.
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Left-hand side (variable terms only, constant == 0)
    pub(crate) lhs: Expression,
    /// Right-hand side scalar
    pub(crate) rhs: f64,
    /// Constraint direction
    pub(crate) sense: ConstraintSense,
}

/// Build a `Constraint` using natural inequality syntax.
///
/// # Examples
/// ```
/// use solver::model::{Model, constraint};
/// let mut model = Model::new("demo");
/// let x = model.add_var("x", 0.0, f64::INFINITY);
/// let y = model.add_var("y", 0.0, 10.0);
/// model.add_constraint(constraint!((2.0 * x + 3.0 * y) <= 12.0));
/// model.add_constraint(constraint!((x + y) >= 3.0));
/// ```
/// Build a `Constraint` using natural inequality syntax.
///
/// Supported forms:
/// - `constraint!(x <= 5.0)` — single variable on the left
/// - `constraint!((expr) <= rhs)` — parenthesised expression on the left
///
/// For complex LHS expressions, wrap them in parentheses:
/// ```rust,no_run
/// # use solver::model::{Model, constraint};
/// # let mut model = Model::new("demo");
/// # let x = model.add_var("x", 0.0, f64::INFINITY);
/// # let y = model.add_var("y", 0.0, f64::INFINITY);
/// model.add_constraint(constraint!((2.0 * x + 3.0 * y) <= 12.0));
/// ```
/// or use the method API directly:
/// ```rust,no_run
/// # use solver::model::Model;
/// # let mut model = Model::new("demo");
/// # let x = model.add_var("x", 0.0, f64::INFINITY);
/// # let y = model.add_var("y", 0.0, f64::INFINITY);
/// model.add_constraint((2.0 * x + 3.0 * y).leq(12.0));
/// ```
#[macro_export]
macro_rules! constraint {
    // Complex LHS in parentheses
    (($lhs:expr) <= $rhs:expr) => {
        $crate::model::expression::Expression::from($lhs).leq($rhs)
    };
    (($lhs:expr) >= $rhs:expr) => {
        $crate::model::expression::Expression::from($lhs).geq($rhs)
    };
    (($lhs:expr) == $rhs:expr) => {
        $crate::model::expression::Expression::from($lhs).eq_constraint($rhs)
    };
    // Single variable (ident can be followed by <= in macro rules)
    ($lhs:ident <= $rhs:expr) => {
        $crate::model::expression::Expression::from($lhs).leq($rhs)
    };
    ($lhs:ident >= $rhs:expr) => {
        $crate::model::expression::Expression::from($lhs).geq($rhs)
    };
    ($lhs:ident == $rhs:expr) => {
        $crate::model::expression::Expression::from($lhs).eq_constraint($rhs)
    };
}
