//! Unified problem IR and solver boundary for Otspot.

#![forbid(unsafe_code)]

pub mod context;
pub mod outcome;
pub mod problem;
pub mod solver;

pub use context::SolveContext;
pub use outcome::{Proof, SolveOutcome, SolveStatus};
pub use problem::{
    Cone, ConicSystem, ConstraintSystem, Objective, OptimizationProblem, ProblemClass,
    QuadraticConstraint, Variable, VariableKind,
};
pub use solver::Solver;
