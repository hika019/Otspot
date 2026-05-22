//! Minimal linear program using the high-level `Model` API.
//!
//! Run: `cargo run --release --example solve_lp`
//!
//!   minimize  x + 2y
//!   subject to  2x + 3y <= 12
//!               x +  y >=  3
//!               x >= 0,  0 <= y <= 10
//!
//! Optimal solution: x = 3, y = 0, objective = 3.

use otspot::model::{constraint, Model};

fn main() {
    let mut model = Model::new("example_lp");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, 10.0);

    model.add_constraint(constraint!((2.0 * x + 3.0 * y) <= 12.0));
    model.add_constraint(constraint!((x + y) >= 3.0));
    model.minimize(x + 2.0 * y);

    match model.solve() {
        Ok(result) => {
            println!("objective = {:.6}", result.objective_value);
            println!("x = {:.6}", result[x]);
            println!("y = {:.6}", result[y]);
        }
        Err(e) => eprintln!("solve failed: {e:?}"),
    }
}
