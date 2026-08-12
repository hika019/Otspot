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

### Python

The `otspot-py` crate (PyO3/maturin) exposes the same `Model` API to Python, symbol-for-symbol:

```python
import otspot

model = otspot.Model("example")
x = model.add_var("x", 0.0, float("inf"))
y = model.add_var("y", 0.0, 10.0)
model.add_constraint((2.0 * x + 3.0 * y).leq(12.0))
model.add_constraint((x + y).geq(3.0))
model.minimize(x + 2.0 * y)

result = model.solve()
print(result.status, result.objective_value, result.value(x), result[y])
```

See [`otspot-py/README.md`](otspot-py/README.md) for installation, the full API
(quadratic objectives, integer/binary variables, error handling), and caveats
(operator overloading differs from Rust's `constraint!` macro, thread-safety,
GIL release during `solve()`, pickling).

## Performance

Solve-rate benchmark on standard public sets via the `otspot-dev` benchmark harness
(shell scripts — **not** `cargo bench`). Measured 2026-08-08 at commits `e4895ff3`–`e2f184d3`
(the diff between these commits is docs-only; solver code is identical):
`jobs = 6`, `timeout = 1000s`, `eps ∈ {1e-6, 1e-8}` (SOCP uses `jobs = 1`; see below).

| Problem type | Set | # | @1e-6 | @1e-8 |
|---|---|---:|---|---|
| Feasible LP | Netlib | 109 | 109 optimal | 108 optimal, 1 suboptimal |
| Convex QP | Maros–Mészáros | 138 | 121 optimal, 1 suboptimal, 10 stalled, 2 mismatch, 4 no-ref | 97 optimal, 4 suboptimal, 34 stalled, 1 timeout, 2 no-ref |
| QCQP | QPLIB | 41 | 12 optimal, 3 suboptimal, 7 stalled, 3 timeout, 11 not-supported, 5 skip | 10 optimal, 4 suboptimal, 7 stalled, 4 timeout, 11 not-supported, 5 skip |
| MILP | MIPLIB 2017 small | 20 | 7 optimal, 13 timeout, 0 error | 7 optimal, 13 timeout, 0 error |
| Infeasible LP | Netlib | 29 | 29 certified | 29 certified |
| Unbounded LP | synthetic | 12 | 12 certified | 12 certified |

**Optimal** = KKT-verified against a known objective (proof-carrying). **Stalled** = the IPM made no further progress before its iteration/time budget and claims no solution. **Suboptimal** = solved but fails the strict optimality certificate. **No-ref** = solved with no published objective to check against.

LP: the only miss is `greenbea` (Suboptimal at @1e-8).

QP: named misses are `UBH1` (Suboptimal at both eps), `LISWET7` (objective mismatch at @1e-6, Stalled at @1e-8), and `POWELL20` (Timeout at @1e-8).

QCQP (QPLIB, single-shot IPM via `bench_qplib` — does not exercise the `--global` spatial B&B path): `not-supported` cases need a finite bound on every variable for the McCormick relaxation and have at least one unbounded variable; `skip` cases have integer variables or unsupported constraint types. Both categories are eps-independent.

MILP: optimal at both eps are `dcmulti`, `flugpl`, `gr4x6`, `gt2`, `khb05250`, `markshare_4_0`, `p0201`. `noswot` and `timtab1` time out at both eps.

Reproduce (data is gitignored; see [Benchmark data](#benchmark-data)):

```bash
for eps in 1e-6 1e-8; do
  bash scripts/run_lp_bench.sh --suite standard --eps "$eps" --jobs 6 --timeout 1000
  bash scripts/run_lp_bench.sh --suite infeas --eps "$eps" --jobs 6 --timeout 1000
  bash scripts/bench_parallel.sh --data-dir data/lp_problems_unbounded --eps "$eps" --jobs 6 \
       --timeout 1000 --output "/tmp/lp_unbounded_${eps}.txt"
  bash scripts/bench_parallel.sh --data-dir data/maros_meszaros --eps "$eps" --jobs 6 \
       --timeout 1000 --output "/tmp/qp_maros_${eps}.txt"
  bash scripts/bench_parallel.sh --data-dir data/qplib --eps "$eps" --jobs 6 \
       --timeout 1000 --output "/tmp/qplib_${eps}.txt"
  bash scripts/bench_parallel.sh --data-dir data/miplib_small --eps "$eps" --jobs 6 \
       --timeout 1000 --output "/tmp/miplib_small_${eps}.txt"  # exits non-zero on ERROR/external timeout
done
```

### SOCP

Otspot is run against Hans Mittelmann's [Large Second-Order Cone benchmark](https://plato.asu.edu/ftp/socp.html) (18 CBLIB instances), which publishes runtimes for MOSEK, ECOS and COPT under a 1-hour limit — those three columns below are Mittelmann's published values, not measured here, and CBLIB/Mittelmann publish no objective values so Otspot's results are not cross-checked against the commercial optima (`Optimal` here means KKT convergence at 1e-6 only). Otspot's own runs use **`jobs = 1`** — the large instances need up to ~18 GB RSS each, so running them in parallel exhausts memory — on a memory-constrained 19 GB VM (8 vCPU), not Mittelmann's 64 GB machine, so its absolute seconds are directional.

All 18 rows below are this run's own measurement, taken 2026-08-08–09 at `eps = 1e-6`, `timeout = 1000s`, under a 14 GB memory cap.

| Problem | nnz | Otspot | MOSEK | ECOS | COPT |
|---|---:|---|---:|---:|---:|
| chainsing-50000-1 | 0.9M | **5.8** | 3 | f | 3 |
| chainsing-50000-2 | 0.75M | **7.0** | 4 | f | 3 |
| chainsing-50000-3 | 0.6M | **4.9** | 3 | f | 2 |
| beam7 | 15M | 434.5 | 17 | 206 | 18 |
| db-plate-yield-line | 1.5M | timeout (>1000s) | 6 | f | 5 |
| db-joint-soerensen | 6M | timeout (>1000s) | 29 | f | 46 |
| firL2L1alph | 10M | timeout (>1000s) | 6 | 202 | 5 |
| firL1Linfeps | 9.9M | timeout (>1000s) | 26 | 2531 | 13 |
| firL2Linfeps | 19M | timeout (>1000s) | 25 | 687 | 14 |
| firL1 | 40M | timeout (>1000s) | 16 | 1305 | 9 |
| firL1Linfalph | 80M | timeout (>1000s) | 56 | 2847 | 23 |
| firL2L1eps | 40M | timeout (>1000s) | 14 | 797 | 9 |
| firL2a | 50M | timeout (>1000s) | 3 | 945 | 4 |
| firLinf | 80M | timeout (>1000s) | 95 | 3479 | 27 |
| wbNRL | 39M | timeout (>1000s) | 9 | 1333 | 7 |
| dsNRL | 67M | timeout (>1000s) | 56 | f | 27 |
| beam30 | 64M | timeout (>1000s) | 99 | 2465 | 84 |
| firL2Linfalph | 122M | OOM (>14 GB cap) | 27 | f | 25 |
| **solved** | | **4/18** | 18/18 | 11/18 | 18/18 |

Otspot solves the three `chainsing-50000` instances (rotated cones, ~1M nonzeros) in 5–7s, where ECOS fails all three; the large dense-Jacobian `fir`/`db`/`beam30`/`dsNRL` instances (6–122M nonzeros) time out or exhaust memory. Supported cone types are `F`/`L±`/`L=`/`Q`/`QR` plus MISOCP via branch-and-bound; `EXP` and PSD cones are rejected as unsupported.

SOCP has no `bench_parallel.sh` harness yet (that script is `.mps`/`.qps`/`.qplib` only); run the `solve_cbf` example directly with an external timeout, one result file per problem. **Run strictly sequentially (jobs=1)** per the memory requirement above. Capture `timeout`'s real exit code directly (do **not** pipe `solve_cbf` through `grep` before reading `$?` — a pipeline's `$?` is grep's status, which masks the 124 timeout code):

```bash
cargo build --release --example solve_cbf
out=/tmp/socp18; mkdir -p "$out"
# Mittelmann Large-SOCP 18, in data/cblib_full/; CBLIB stems vary per problem —
# fir*/dsNRL/wbNRL carry a 2013_ prefix, beam7/beam30/chainsing-*/db-* do not
# (see plato.asu.edu/ftp/socp.html) — and some are still gzip-compressed.
for n in beam7 beam30 chainsing-50000-1 chainsing-50000-2 chainsing-50000-3 \
         db-joint-soerensen db-plate-yield-line dsNRL firL1 firL1Linfalph \
         firL1Linfeps firL2L1alph firL2L1eps firL2Linfalph firL2Linfeps \
         firL2a firLinf wbNRL; do
  case "$n" in fir*|dsNRL|wbNRL) stem="2013_$n" ;; *) stem="$n" ;; esac
  f="data/cblib_full/$stem.cbf"
  if [ ! -f "$f" ]; then
    # No 0-byte artifact on a fresh clone: require the .gz, decompress via a
    # temp so a failed gunzip never leaves a broken .cbf to be reused.
    [ -f "$f.gz" ] || { echo "$n,Missing,,," >> "$out/$n.csv"; continue; }
    gunzip -c "$f.gz" > "$f.tmp" && mv "$f.tmp" "$f" \
      || { echo "$n,DecompressFailed,,," >> "$out/$n.csv"; continue; }
  fi
  timeout 1000 ./target/release/examples/solve_cbf --eps 1e-6 "$f" > "$out/$n.csv" 2>/dev/null
  rc=$?   # no pipe: 0 = status is in the CSV; else the run died before writing it
  case "$rc" in
    0) : ;;
    124) echo "$n,Timeout,,,1000.0" >> "$out/$n.csv" ;;       # timeout 1000
    137|134) echo "$n,OOM,,," >> "$out/$n.csv" ;;             # 14 GB limit
    *) echo "$n,Error$rc,,," >> "$out/$n.csv" ;;
  esac
done
grep -hv '^problem,' "$out"/*.csv   # per-problem status,objective,iters,time
```

## Tests

```bash
cargo nextest run --release --test-threads 6          # full suite (requires data/)
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
otspot-py/       # Python bindings (PyO3/maturin); not in default-members
otspot-dev/      # dev-only binaries (qps_benchmark, qp_runner, …; not published)
examples/        # solve_lp, solve_qp
tests/           # integration tests
scripts/         # data-generation scripts
```

## License

[GNU Affero General Public License v3.0 (AGPL-3.0-only)](LICENSE)
