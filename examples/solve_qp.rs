//! Minimal quadratic program using the high-level `Model` API.
//!
//! Run: `cargo run --release --example solve_qp`
//!
//!   minimize  x² + y² - 4x - 4y
//!   subject to  x + y <= 3
//!               x >= 0, y >= 0
//!
//! Optimal solution: x = 1.5, y = 1.5, objective = -7.5 (constraint active).

use otspot::model::{constraint, Model};

fn main() {
    let mut model = Model::new("example_qp");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);

    model.add_constraint(constraint!((x + y) <= 3.0));
    model.minimize(x * x + y * y - 4.0 * x - 4.0 * y);

    match model.solve() {
        Ok(result) => {
            println!("objective = {:.6}", result.objective_value);
            println!("x = {:.6}", result[x]);
            println!("y = {:.6}", result[y]);
        }
        Err(e) => eprintln!("solve failed: {e:?}"),
    }
}
