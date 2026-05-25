//! Algebraic modeling API for `otspot`.
//!
//! Provides [`Model`], [`Variable`], [`Expression`], [`Constraint`],
//! the [`constraint!`] macro, and associated error/result types.
//!
//! # Example
//! ```
//! use otspot_model::{Model, constraint};
//!
//! let mut model = Model::new("example");
//! let x = model.add_var("x", 0.0, 10.0);
//! let y = model.add_var("y", 0.0, 10.0);
//! model.add_constraint(constraint!((x + y) <= 8.0));
//! model.minimize(2.0 * x + y);
//! let result = model.solve().unwrap();
//! assert!((result[x] + result[y]).abs() < 1e-4);
//! ```

pub mod constraint;
pub mod expression;
pub mod variable;
mod model;

pub use constraint::{Constraint, ConstraintSense};
pub use expression::Expression;
pub use variable::{VarKind, Variable};
pub use model::{Model, ModelError, ModelResult, SolutionProof, SolveError};
