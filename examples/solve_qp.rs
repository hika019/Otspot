//! Minimal quadratic program using the high-level `Model` API.
//!
//! Run: `cargo run --release --example solve_qp`
//!
//!   minimize  x^2 + y^2 - 4x - 4y     (objective is 1/2 xᵀQx + cᵀx, Q = diag(2, 2))
//!   subject to  x + y <= 3
//!               x >= 0, y >= 0
//!
//! The unconstrained optimum is (2, 2); the constraint x + y <= 3 is active at
//! the solution.

use solver::model::{constraint, Model};
use solver::CscMatrix;

fn main() {
    let mut model = Model::new("example_qp");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);

    // Hessian Q under the 1/2 xᵀQx convention: diag(2, 2) gives x^2 + y^2.
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    model.set_quadratic_objective(q);
    model.add_constraint(constraint!((x + y) <= 3.0));
    model.minimize(-4.0 * x + -4.0 * y);

    match model.solve() {
        Ok(result) => {
            println!("objective = {:.6}", result.objective_value);
            println!("x = {:.6}", result[x]);
            println!("y = {:.6}", result[y]);
        }
        Err(e) => eprintln!("solve failed: {e:?}"),
    }
}
