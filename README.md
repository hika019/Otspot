# otspot

**English** | [日本語](README.ja.md)

A **mathematical optimization solver** written in Rust.

LP: revised simplex (sparse LU, Ruiz equilibration, steepest-edge pricing).
QP: interior-point (Mehrotra predictor–corrector / IP-PMM) + spatial branch-and-bound for non-convex QP (α-BB / McCormick).
MILP / convex MIQP: branch-and-bound.
`Optimal` is **proof-carrying** (full KKT verification); infeasible and unbounded problems are certified.

## Features

- **Algebraic modeling API** — natural math notation including quadratic objectives (`x * x`, `x * y`)
- **Revised simplex (LP)** — sparse LU, Markowitz-threshold pivoting, steepest-edge pricing
- **Interior-point (QP)** — Mehrotra predictor–corrector / IP-PMM for convex QP
- **Non-convex QP (global)** — spatial B&B (α-BB / McCormick); global optimum carries bound-gap certificate, local-only reported as `NonconvexLocal`
- **Mixed-integer (MILP / convex MIQP)** — branch-and-bound with GMI/MIR/cover/clique/implied-bound cuts, reliability branching, RINS, conflict analysis
- **Sensitivity analysis (LP)** — RHS and objective coefficient ranging
- **Proof-carrying optimality** — `Optimal` requires full KKT certificate; unprovable solutions are downgraded
- **Infeasibility / unboundedness certification**
- **Dual solution output** — dual values, reduced costs, slacks
- **Input formats** — MPS (LP), QPS / QPLIB (QP)

## Quick start

Requires Rust (edition 2021, stable).

```toml
[dependencies]
otspot = "0.7"
```

```bash
git clone https://github.com/hika019/otspot.git
cd otspot
cargo run --release --example solve_lp   # minimal LP
cargo run --release --example solve_qp   # minimal QP
```

### LP

```rust
use otspot::model::{constraint, Model};

fn main() {
    // minimize  x + 2y   s.t.  2x + 3y <= 12,  x + y >= 3,  x >= 0, y in [0,10]
    let mut model = Model::new("example");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, 10.0);
    model.add_constraint(constraint!((2.0 * x + 3.0 * y) <= 12.0));
    model.add_constraint(constraint!((x + y) >= 3.0));
    model.minimize(x + 2.0 * y);

    let result = model.solve().unwrap();
    println!("obj={} x={} y={}", result.objective_value, result[x], result[y]);
}
```

`constraint!` also accepts single-variable forms (`constraint!(x <= 7.0)`) and the expression
method API (`.leq()`, `.geq()`, `.eq_constraint()`). Use `model.maximize(...)` for maximization.

Tolerance / options:

```rust
use otspot::Tolerance;
model.set_tolerance(Tolerance::High); // 1e-8; Medium (1e-6, default), Fast, Custom(f64)
model.set_timeout(60.0);
```

### QP

```rust
use otspot::model::{constraint, Model};

fn main() {
    // minimize  x² + y²   s.t.  x + y >= 1
    let mut model = Model::new("qp");
    let x = model.add_var("x", f64::NEG_INFINITY, f64::INFINITY);
    let y = model.add_var("y", f64::NEG_INFINITY, f64::INFINITY);
    model.add_constraint(constraint!((x + y) >= 1.0));
    model.minimize(x * x + y * y);

    let result = model.solve().unwrap();
    println!("obj={:.4} x={:.4} y={:.4}", result.objective_value, result[x], result[y]);
}
```

### Low-level API

```rust
use otspot::{problem::LpProblem, sparse::CscMatrix, solve};

let c = vec![-1.0, -1.0];
let rows = vec![0usize, 0, 1, 2];
let cols = vec![0usize, 1, 0, 1];
let vals = vec![1.0, 1.0, 1.0, 1.0];
let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 2).unwrap();
let problem = LpProblem::new(c, a, vec![4.0, 3.0, 3.0]).unwrap();
let result = solve(&problem);
println!("{} {}", result.status, result.objective); // Optimal -4
```

### MPS input

```rust
use otspot::{io::mps, solve};
let prob = mps::parse_mps_file("problem.mps".as_ref()).unwrap();
let result = solve(&prob);
```

## Performance

Solve-rate benchmark on standard public sets via the `otspot-dev` benchmark harness
(shell scripts — **not** `cargo bench`), `timeout = 1000s`:

| Problem type | Set | # | @1e-6 | @1e-8 |
|---|---|---:|---|---|
| Feasible LP | Netlib | 109 | 108 optimal, 1 suboptimal | 107 optimal, 1 primal-feasibility fail, 1 suboptimal |
| Convex QP | Maros–Mészáros | 138 | 121 optimal, 12 suboptimal, 1 mismatch, 4 no-ref | 93 optimal, 42 suboptimal, 1 timeout, 2 no-ref |
| MILP | MIPLIB 2017 small | 20 | 5 optimal, 13 timeout, 2 error | 5 optimal, 13 timeout, 2 error |
| Infeasible LP | Netlib | 29 | 29 certified | 29 certified |
| Unbounded LP | synthetic | 12 | 12 certified | 12 certified |

**Optimal** = verified against known objective (proof-carrying KKT). `timeout = 1000s`, `jobs = 6`.

LP: @1e-6 is 108/109 optimal, 0 timeout; the sole miss is `cycle` (SuboptimalSolution). @1e-8 is 107/109 optimal, 0 timeout; misses are `greenbea` (PFEAS_FAIL) and `cycle` (SuboptimalSolution).

QP: @1e-6 is 121/138 optimal, 0 timeout. Misses are 12 SuboptimalSolution, 1 OBJ_MISMATCH (`LISWET7`), and 4 solved-but-unverified cases with no published reference. @1e-8 is 93/138 optimal, with 42 SuboptimalSolution, 1 TIMEOUT (`POWELL20`), and 2 solved-but-unverified cases.

MILP: @1e-6 and @1e-8 both prove 5/20 optimal (`flugpl`, `gr4x6`, `gt2`, `khb05250`, `p0201`). Both runs also report 13 TIMEOUT and 2 ERROR inside `TOTAL`; the ERROR cases are `noswot` and `timtab1` (`no_output_exit=101`).

Reproduce (data is gitignored; see [Benchmark data](#benchmark-data)):

```bash
for eps in 1e-6 1e-8; do
  bash scripts/run_lp_bench.sh --suite standard --eps "$eps" --jobs 6 --timeout 1000
  bash scripts/run_lp_bench.sh --suite infeas --eps "$eps" --jobs 6 --timeout 1000
  bash scripts/bench_parallel.sh --data-dir data/lp_problems_unbounded --eps "$eps" --jobs 6 \
       --timeout 1000 --output "/tmp/lp_unbounded_${eps}.txt"
  bash scripts/bench_parallel.sh --data-dir data/maros_meszaros --eps "$eps" --jobs 6 \
       --timeout 1000 --output "/tmp/qp_maros_${eps}.txt"
  bash scripts/bench_parallel.sh --data-dir data/miplib_small --eps "$eps" --jobs 6 \
       --timeout 1000 --output "/tmp/miplib_small_${eps}.txt"  # exits non-zero on ERROR/external timeout
done
```

## Tests

```bash
cargo nextest run --release --test-threads 3          # full suite (requires data/)
cargo nextest run --release --profile lib-only       # lib + bin tests (kind=lib + kind=bin), no integration data needed
cargo test --doc --release
```

Integration tests assert `data/` presence and panic when missing — use `--profile lib-only` on machines without data.

## Development (Docker)

```bash
docker build -f Dockerfile.dev -t otspot-dev .
docker run -it --rm -v "$PWD":/workspace -w /workspace otspot-dev bash
```

### Benchmark data

```bash
bash scripts/download_all_bench_data.sh          # Netlib LP + Maros-Meszaros + QPLIB + synthetic QP
bash scripts/download_all_bench_data.sh --lp     # LP only
bash scripts/download_all_bench_data.sh --check  # check what is present
```

QP data generation (synthetic suites) requires `numpy scipy cvxpy clarabel` (`pip install`).

## Project structure

Cargo workspace:

```
otspot/          # facade crate — re-exports core / io / model
otspot-core/     # solver engine (simplex, IPM, B&B, presolve, linalg, sparse)
otspot-io/       # file parsers (MPS, QPS, QPLIB)
otspot-model/    # algebraic modeling API (Model, Variable, constraint! macro)
otspot-dev/      # dev-only binaries (qps_benchmark, qp_runner, …; not published)
examples/        # solve_lp, solve_qp
tests/           # integration tests
scripts/         # data-generation scripts
```

## License

[GNU Affero General Public License v3.0 (AGPL-3.0-only)](LICENSE)
