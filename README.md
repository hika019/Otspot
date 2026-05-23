# otspot

**English** | [日本語](README.ja.md)

A **mathematical optimization solver** written in Rust.

otspot implements a **revised simplex method** (sparse LU, Ruiz equilibration, steepest-edge pricing) for LP and an **interior-point method** (Mehrotra predictor–corrector / IP-PMM) for QP, with branch-and-bound on top for mixed-integer problems. It certifies infeasible and unbounded problems and returns full primal/dual information.

## Features

- **Algebraic modeling API** — express problems in natural mathematical notation
- **Revised simplex (LP)** — sparse LU factorization with Markowitz-threshold pivoting
- **Interior-point (QP)** — Mehrotra predictor–corrector / IP-PMM for convex QP
- **Mixed-integer (MILP / convex MIQP)** — baseline branch-and-bound with most-fractional branching over continuous relaxations (LP for MILP, QP for convex MIQP); non-convex MIQP out of scope. Cuts, primal heuristics, SOS constraints, and richer branching strategies are not implemented.
- **Infeasibility / unboundedness certification** — an explicit status, not just a failure
- **Ruiz equilibration** — row/column scaling preconditioner for better conditioning
- **Steepest-edge pricing** — faster convergence via improved entering-variable choice
- **Dual solution output** — dual values, reduced costs, constraint slacks
- **Input formats** — MPS (LP) and QPS / QPLIB (QP)
- **Configurable options** — tolerances, iteration caps, LU refactorization threshold
- **Benchmark + fuzz suites** — criterion microbenchmarks and proptest randomized tests

## Quick start

Requires Rust (edition 2021, stable).

From crates.io:

```toml
[dependencies]
otspot = "0.1"
```

Or as a git dependency:

```toml
[dependencies]
otspot = { git = "https://github.com/hika019/otspot" }
```

Build and run from source:

```bash
git clone https://github.com/hika019/otspot.git
cd otspot
cargo build --release
cargo run --release --example solve_lp   # minimal LP
cargo run --release --example solve_qp   # minimal QP
```

### Modeling API

The recommended way to define and solve an LP:

```rust
use otspot::model::{Model, constraint};

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

### The `constraint!` macro

`constraint!` supports natural inequality syntax for single variables and parenthesized expressions:

```rust
// Single variable
model.add_constraint(constraint!(x <= 7.0));
model.add_constraint(constraint!(y >= 0.0));
model.add_constraint(constraint!(x == 5.0));

// Expression on the left-hand side (wrap in parentheses)
model.add_constraint(constraint!((2.0 * x + y) <= 10.0));
```

Or use the method API directly on expressions:

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

### Tolerance and options

Set the convergence tolerance (eps) and other common knobs directly on the model. `Tolerance`
has presets `High` (1e-8), `Medium` (1e-6, default) and `Fast`, plus `Custom(f64)`:

```rust
use otspot::Tolerance;

model.set_tolerance(Tolerance::Custom(1e-8)); // eps = 1e-8
model.set_presolve(true);                      // presolve on/off (default: on)
model.set_timeout(60.0);                       // wall-clock limit, seconds
```

For fine-grained control, the low-level `solve_with` takes a full `SolverOptions`:

```rust
use otspot::SolverOptions;
use otspot::solve_with;

let opts = SolverOptions {
    primal_tol: 1e-8,   // LP simplex optimality / feasibility tolerance
    max_etas: 50,       // LU refactorization threshold (0 = auto)
    clamp_tol: 1e-14,   // solution micro-value clamp
    ..Default::default()
};
let result = solve_with(&problem, &opts); // problem: &LpProblem
```

### Dual solution

Alongside the primal solution, `model.solve()` returns dual values, reduced costs and constraint slacks (each `Option<Vec<f64>>`):

```rust
let result = model.solve().unwrap();
println!("dual (shadow): {:?}", result.dual_solution);
println!("reduced costs: {:?}", result.reduced_costs);
println!("slacks:        {:?}", result.slack);
```

### Quadratic programming (QP)

QP uses the same modeling API as LP — just add a quadratic objective. Set it with the
`set_diagonal_q` shorthand (or `set_quadratic_objective` for a full `CscMatrix`). The objective
follows the "1/2" convention, minimize ½·xᵀQx + cᵀx, where the linear part c comes from
`minimize` / `maximize`.

```rust
use otspot::model::{constraint, Model};

// min  x² + y²        (= ½·xᵀQx with Q = diag(2, 2))
// s.t. x + y >= 1
fn main() {
    let mut model = Model::new("qp");
    let x = model.add_var("x", f64::NEG_INFINITY, f64::INFINITY);
    let y = model.add_var("y", f64::NEG_INFINITY, f64::INFINITY);

    model.add_constraint(constraint!((x + y) >= 1.0));
    model.set_diagonal_q(&[2.0, 2.0]); // Q = diag(2, 2)
    model.minimize(0.0 * x + 0.0 * y); // linear part c = 0

    let result = model.solve().unwrap();
    println!("objective = {:.4}", result.objective());
    println!("x = {:.4}, y = {:.4}", result[x], result[y]);
}
```

**Output:**
```
objective = 0.5000
x = 0.5000, y = 0.5000
```

For direct matrix input a low-level `qp::solve_qp` / `QpProblem` API (taking Q, c, A, b, bounds as
arrays) is also available; `solve_qp_warm` carries the previous active set across SQP iterations.

## Advanced

For performance-critical applications, build the constraint matrix directly in CSC format and call the low-level API:

```rust
use otspot::problem::LpProblem;
use otspot::sparse::CscMatrix;
use otspot::solve;

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
let result = solve(&problem);

println!("status:    {}", result.status);    // Optimal
println!("objective: {}", result.objective);  // -4
println!("solution:  {:?}", result.solution); // [1.0, 3.0]
```

## MPS input

Read an LP from an MPS file:

```rust
use std::path::Path;
use otspot::io::mps;
use otspot::solve;

let prob = mps::parse_mps_file(Path::new("problem.mps")).expect("MPS parse error");
let result = solve(&prob);
println!("status: {}", result.status);
```

The parser is validated against the Netlib LP set and the Maros–Mészáros QP set (see [Performance](#performance)).

## Performance

Measured on standard public benchmark sets, `timeout = 1000s`, 6-way parallel, at two tolerances.
**Optimal** = optimum verified against the known objective. **Valid** = a feasible solution meeting
the solver's optimality criteria but with no external reference to verify against (counted separately
from *optimal*). For infeasible / unbounded sets the metric is a correct certificate.

| Problem type | Set | # | @ 1e-6 | @ 1e-8 |
|---|---|---:|---|---|
| Feasible LP | Netlib | 109 | 109 optimal | 106 optimal |
| Convex QP | Maros–Mészáros | 138 | 129 optimal, 7 valid | 124 optimal, 4 valid |
| Infeasible LP | Netlib | 29 | 29 certified | 29 certified |
| Unbounded LP | synthetic | 12 | 12 certified | 12 certified |

The remaining `1e-6` QP misses are the `LISWET` family (`LISWET9`/`LISWET12`), projections onto the
cone of convex sequences:
their constraint normal matrix is the discrete biharmonic operator (cond ≈ n⁴ ≈ 1e15 at
n = 10⁴) and the optimal duals are huge (|y|∞ ≈ 1e5–1e6), making the objective hypersensitive
to primal residuals that f64 interior-point arithmetic cannot reduce below ~1e-6. otspot returns
`SuboptimalSolution`/`Timeout` on these. At the tighter `1e-8`, additional ill-conditioned
instances land just above the residual threshold; this is an accuracy-floor effect.

Benchmark data is gitignored and reproducible; see [Benchmark data](#benchmark-data).

## Benchmarks (criterion)

Four criterion microbenchmark suites are included:

```bash
# All benchmarks
cargo bench

# Individual suites
cargo bench --bench scaling_pricing   # Ruiz scaling + steepest-edge pricing
cargo bench --bench lu_bench          # LU factorization throughput
cargo bench --bench solve_bench       # end-to-end LP solve
cargo bench --bench qp_bench          # QP solve
```

HTML reports are generated in `target/criterion/`.

## Tests

```bash
# Full test suite (unit + Netlib + proptest) — requires benchmark data
cargo nextest run --release

# Unit / bin tests only (no benchmark data needed)
cargo nextest run --release --profile lib-only

# Doc tests
cargo test --doc --release
```

Many `tests/*.rs` require `data/lp_problems_*/`, `data/qplib/`, etc. and assert their
presence (`assert!(path.exists())`), so they **panic** when data is missing — by design,
following the project rule of "panic, don't SKIP" so that no verification gap goes
unnoticed. On a machine without the data, use `--profile lib-only` to run only the
unit and bin tests.

To fetch benchmark data, see [Development (Docker)](#development-docker) or run
`bash scripts/download_all_bench_data.sh`.

The suite includes:
- **Unit tests** across all modules
- **Netlib integration** for real-world validation
- **proptest** randomized fuzz tests
- **Smoke tests** for basic API coverage

## Development (Docker)

`Dockerfile.dev` is a development container that reproduces the test/bench environment
on other machines (Rust 1.83 + python3 with numpy/scipy/cvxpy/clarabel + cargo-nextest
+ a precompiled Netlib `emps` decoder).

```bash
# 1. Build (~5-10 min the first time)
docker build -f Dockerfile.dev -t otspot-dev .

# 2. Interactive development (source shared with the host, saves apply instantly)
docker run -it --rm -v "$PWD":/workspace -w /workspace otspot-dev bash

# 3. One-shot test
docker run --rm -v "$PWD":/workspace -w /workspace otspot-dev \
  cargo nextest run --release
```

### Bare-host run (without Docker)

The data-generation scripts (`scripts/gen_*.py`) require these Python packages:

```bash
pip install numpy scipy cvxpy clarabel
```

`cvxpy` / `clarabel` are needed only by the `osqp_bench` family generators
(`osqp_bench`, `osqp_bench_*`, `qp_dense_a`). The LP suites (`lp_problems*`) need
only `curl` + `emps`.

`scripts/download_all_bench_data.sh` checks for numpy/scipy when entering QP mode
and, if absent, prints the Docker / pip instructions and exits.

### Benchmark data

`data/` is gitignored, so after cloning you need to generate the benchmark data yourself:

```bash
# Everything (Netlib LP + synthetic QP)
bash scripts/download_all_bench_data.sh

# LP only (Netlib fetch + synthetic, deterministic)
bash scripts/download_all_bench_data.sh --lp

# Check what is present
bash scripts/download_all_bench_data.sh --check
```

Synthetic sets use fixed seeds, so any environment reproduces identical output. The
Maros–Mészáros and QPLIB sets have no download script and must be placed manually
(URL hints are in `download_all_bench_data.sh`); run `--check` to see what is present.

## Project structure

```
src/
├── lib.rs              # crate entry point / public API re-exports
├── model/              # high-level algebraic modeling API (Model, constraint! macro)
├── lp.rs               # LP solve entry
├── simplex/            # revised simplex (primal / dual)
├── qp/                 # QP solve (interior-point IPM / IP-PMM, postsolve)
├── mip/                # mixed-integer (MILP / MIQP) branch-and-bound
├── presolve/           # presolve (Ruiz scaling, postsolve)
├── linalg/             # linear algebra (LU, LDLᵀ)
├── basis/              # basis management
├── sparse/             # CSC sparse matrix / sparse vector
├── problem/            # LpProblem / QpProblem, SolverResult, SolveStatus
├── screening.rs        # problem screening
├── options.rs          # SolverOptions
├── tolerances.rs       # numerical tolerance constants
├── error.rs            # SolverError
├── io/                 # input parsers (mps / qps / qplib)
└── bin/                # CLI tools (qp_runner, qp_diag, qps_benchmark, ...)
examples/               # usage examples (solve_lp, solve_qp)
benches/                # criterion benchmarks (lu_bench, qp_bench, solve_bench, scaling_pricing)
```

## License

[GNU Affero General Public License v3.0 (AGPL-3.0-only)](LICENSE)
