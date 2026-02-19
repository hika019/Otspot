# solver

A high-performance linear programming (LP) solver written in Rust.

Implements the **Revised Simplex method** with sparse LU decomposition, Ruiz equilibration scaling, and steepest-edge pricing for robust performance on real-world LP instances.

## Features

- **Algebraic modeling API** — express LP problems in natural mathematical notation
- **Revised Simplex** — Phase I/II with sparse LU decomposition and Markowitz threshold pivoting
- **Ruiz equilibration** — row/column scaling pre-processor for better numerical conditioning
- **Steepest-edge pricing** — improved variable selection for faster convergence
- **Dual solution output** — dual variables, reduced costs, and constraint slacks
- **MPS file input** — reads industry-standard MPS format; validated on 23 Netlib instances
- **Configurable options** — tolerance, iteration limit, LU refactorization threshold
- **Benchmarks** — criterion-based benchmarks for scaling, LU factorization, and solve
- **Fuzz testing** — proptest-based randomized testing

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
solver = { path = "path/to/solver" }
```

### Modeling API

The recommended way to define and solve LP problems:

```rust
use solver::model::{Model, constraint};

fn main() {
    // Problem:
    //   minimize    x + 2y
    //   subject to  2x + 3y <= 12
    //               x  +  y >= 3
    //               x in [0, +inf), y in [0, 10]

    let mut model = Model::new("production");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, 10.0);

    model.add_constraint(constraint!((2.0 * x + 3.0 * y) <= 12.0));
    model.add_constraint(constraint!((x + y) >= 3.0));
    model.minimize(x + 2.0 * y);

    let result = model.solve().unwrap();
    println!("objective = {}", result.objective());
    println!("x = {}", result[x]);
    println!("y = {}", result[y]);
}
```

**Output:**
```
objective = 3
x = 3
y = 0
```

### `constraint!` macro

The `constraint!` macro supports natural inequality syntax for single variables and parenthesised expressions:

```rust
// Single variable
model.add_constraint(constraint!(x <= 7.0));
model.add_constraint(constraint!(y >= 0.0));
model.add_constraint(constraint!(x == 5.0));

// Expression on the left-hand side (wrap in parentheses)
model.add_constraint(constraint!((2.0 * x + y) <= 10.0));
```

Alternatively, use the method API directly on expressions:

```rust
model.add_constraint((x + 2.0 * y).leq(8.0));
model.add_constraint((x - y).geq(0.0));
model.add_constraint((x + y).eq_constraint(5.0));
```

### Maximization

```rust
let mut model = Model::new("revenue");
let x = model.add_var("x", 0.0, f64::INFINITY);
let y = model.add_var("y", 0.0, f64::INFINITY);

model.add_constraint(constraint!((x + y) <= 10.0));
model.maximize(3.0 * x + 5.0 * y);

let result = model.solve().unwrap();
println!("max revenue = {}", result.objective());
```

### SolverOptions

Fine-tune the solver behavior:

```rust
use solver::SolverOptions;
use solver::problem::LpProblem;
use solver::simplex;

let opts = SolverOptions {
    primal_tol: 1e-8,          // optimality / feasibility tolerance
    max_iterations: Some(500), // None = auto (100*(m+n)+1000)
    max_etas: 50,              // LU refactorization threshold
    clamp_tol: 1e-14,          // solution micro-value clamp
};

let result = simplex::solve_with(&problem, &opts);
```

### Dual Solution

The low-level `simplex::solve` and `simplex::solve_with` return a `SolverResult` with full dual information:

```rust
use solver::problem::SolverResult;

let result: SolverResult = simplex::solve(&problem);
println!("primal:        {:?}", result.solution);
println!("dual (shadow): {:?}", result.dual_solution);
println!("reduced costs: {:?}", result.reduced_costs);
println!("slacks:        {:?}", result.slack);
```

## Advanced Usage

For performance-critical applications, build the constraint matrix directly in CSC format and call the low-level API:

```rust
use solver::problem::LpProblem;
use solver::sparse::CscMatrix;
use solver::simplex;

// minimize  -x1 - x2
// s.t.       x1 + x2 <= 4
//            x1      <= 3
//                 x2 <= 3
//            x1, x2 >= 0

let c = vec![-1.0, -1.0];

let rows = vec![0, 0, 1, 2];
let cols = vec![0, 1, 0, 1];
let vals = vec![1.0, 1.0, 1.0, 1.0];
let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 2).unwrap();

let b = vec![4.0, 3.0, 3.0];

let problem = LpProblem::new(c, a, b).unwrap();
let result = simplex::solve(&problem);

println!("status:    {}", result.status);   // Optimal
println!("objective: {}", result.objective); // -4
println!("solution:  {:?}", result.solution);// [1.0, 3.0]
```

## MPS Input

Read LP problems from MPS files:

```rust
use std::path::Path;
use solver::io::mps;
use solver::simplex;

let prob = mps::parse_mps_file(Path::new("problem.mps")).expect("MPS parse error");
let result = simplex::solve(&prob);
println!("status: {}", result.status);
```

The solver is validated against 23 Netlib benchmark instances (adlittle, afiro, sc50a, sc50b, kb2, brandy, scorpion, fit1d, share1b, and more).

## Benchmarks

Three criterion-based benchmark suites are included:

```bash
# All benchmarks
cargo bench

# Individual suites
cargo bench --bench scaling_pricing   # Ruiz scaling + steepest-edge pricing
cargo bench --bench lu_bench          # LU factorization throughput
cargo bench --bench solve_bench       # End-to-end LP solve
```

HTML reports are generated in `target/criterion/`.

## Testing

```bash
# Full test suite (unit + Netlib + proptest)
cargo test

# Verbose output
cargo test -- --nocapture

# Netlib integration tests only
cargo test netlib

# Proptest fuzz tests only
cargo test proptest
```

The test suite includes:
- **Unit tests** for all modules
- **23 Netlib instances** for real-world validation
- **3 proptest suites** for randomized fuzz testing
- **Smoke tests** for basic API coverage

## Project Structure

```
src/
├── lib.rs              # Crate entry point
├── model/              # High-level algebraic modeling API
│   ├── mod.rs          # Model, ModelResult, ModelError
│   ├── variable.rs     # Variable handle
│   ├── expression.rs   # Linear expression (+, -, * operators)
│   └── constraint.rs   # Constraint, constraint! macro
├── simplex/            # Revised Simplex solver
│   ├── mod.rs          # solve() / solve_with()
│   └── pricing.rs      # Steepest-edge pricing strategy
├── presolve/           # Pre-processing
│   ├── mod.rs
│   └── scaling.rs      # Ruiz equilibration scaling
├── basis/              # LU decomposition basis management
├── sparse/             # CSC sparse matrix and vector
├── problem/            # LpProblem, SolverResult, SolveStatus
├── options.rs          # SolverOptions
├── tolerances.rs       # Numerical tolerance constants
├── error.rs            # SolverError enum
└── io/
    ├── mod.rs
    └── mps.rs          # MPS file parser
benches/
├── scaling_pricing.rs
├── lu_bench.rs
└── solve_bench.rs
```

## License

Dual-licensed under your choice of:

- [Apache License 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)
