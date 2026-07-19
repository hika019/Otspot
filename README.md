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
| Feasible LP | Netlib | 109 | 109 optimal | 108 optimal, 1 suboptimal |
| Convex QP | Maros–Mészáros | 138 | 121 optimal, 1 suboptimal, 9 stalled, 2 maxiter, 1 mismatch, 4 no-ref | 93 optimal, 42 suboptimal, 1 timeout, 2 no-ref |
| QCQP | QPLIB | 41 | 11 optimal, 3 suboptimal, 8 stalled, 3 timeout, 11 not-supported, 5 skip | 8 optimal, 4 suboptimal, 9 stalled, 4 timeout, 11 not-supported, 5 skip |
| MILP | MIPLIB 2017 small | 20 | 5 optimal, 15 timeout, 0 error | 5 optimal, 15 timeout, 0 error |
| SOCP | Mittelmann Large-SOCP | 18 | 4 optimal @1000s (6 @3600s), rest timeout, 1 OOM | n/a (1e-6 only) |
| Infeasible LP | Netlib | 29 | 29 certified | 29 certified |
| Unbounded LP | synthetic | 12 | 12 certified | 12 certified |

**Optimal** = verified against known objective (proof-carrying KKT). **Stalled** = the IPM made no further progress before its iteration/time budget and reports no solution claim (an honest non-convergent status; earlier taxonomy versions folded this into `SuboptimalSolution`). LP/QP/QCQP/MILP rows: `timeout = 1000s`, `jobs = 6`. The SOCP row follows Mittelmann's benchmark instead — `jobs = 1` (sequential; large instances need up to ~18 GB RSS each) with per-problem timeouts noted below; see the SOCP notes for its distinct methodology.

LP: @1e-6 is 109/109 optimal, 0 timeout. @1e-8 is 108/109 optimal, 0 timeout; the sole miss is `greenbea` (SuboptimalSolution after failing the stricter primal proof gate) — this column predates the taxonomy split above and has not been re-measured.

QP: @1e-6 is 121/138 optimal, 0 timeout. Misses are 1 SuboptimalSolution (`UBH1`), 9 Stalled (non-converged, no solution claimed), 2 MaxIterations, 1 OBJ_MISMATCH (`LISWET7`), and 4 solved-but-unverified cases with no published reference. @1e-8 is 93/138 optimal, with 42 SuboptimalSolution, 1 TIMEOUT (`POWELL20`), and 2 solved-but-unverified cases (this column predates the taxonomy split above and has not been re-measured).

QCQP (QPLIB, single-shot IPM via `bench_qplib` — this suite run does not exercise the `--global` spatial B&B path): @1e-6 is 11/41 optimal, 3 TIMEOUT. Non-passing cases are 3 SuboptimalSolution, 8 Stalled (non-converged IPM iterate, no solution claimed — the honest replacement for what the pre-refactor taxonomy folded into SuboptimalSolution), 11 NOT_SUPPORTED (the non-convex McCormick relaxation requires finite bounds on every variable; these instances have an unbounded one), and 5 SKIP (parse-time out of scope: integer variables or unsupported constraint types). @1e-8 is 8/41 optimal, with 4 SuboptimalSolution and 9 Stalled (4 TIMEOUT; NOT_SUPPORTED/SKIP are eps-independent, unchanged). Loosening to @1e-4 recovers more: 15/41 optimal, 2 SuboptimalSolution, 5 Stalled, 3 TIMEOUT.

MILP: @1e-6 and @1e-8 both prove 5/20 optimal (`flugpl`, `gr4x6`, `gt2`, `khb05250`, `p0201`). Both runs report 15 TIMEOUT and 0 ERROR inside `TOTAL`; `noswot` and `timtab1` now time out instead of panicking in tree-cut separation.

SOCP: Otspot is run against Hans Mittelmann's [Large Second-Order Cone benchmark](https://plato.asu.edu/ftp/socp.html) (18 CBLIB instances, 29 Jun 2026), which carries published runtimes for MOSEK, ECOS, KNITRO, COPT and cuOpt under a 1-hour limit. This replaces an earlier ad-hoc 22-instance self-baseline; the commercial/OSS runtimes below are an *external* yardstick, not Otspot's own numbers. **Otspot's times are on a memory-constrained 19 GB QEMU VM (8 vCPU), not Mittelmann's Intel i7-11700K / 64 GB**, so absolute seconds are directional and the 64 GB headroom is why `firL2Linfalph` (122M nonzeros, 2.76 GB input) runs out of memory here at the 18 GB cap rather than solving. Otspot returns `Optimal` = proof-carrying KKT convergence at 1e-6; CBLIB/Mittelmann publish no objective values, so these are not cross-checked against the commercial optima.

Otspot solves **4/18 within its own 1000s default**; extending to the benchmark's 3600s limit adds `firL2L1alph` (1064s) and `firL2Linfeps` (1841s) → **6/18** (the other timeouts were not exhaustively re-run to 3600s, so the true 3600s count could be higher). ECOS — the closest OSS SOCP peer — solves 11/18; commercial MOSEK and COPT solve all 18, with Mittelmann's published shifted geometric mean 1.35 and ~1 respectively (a dimensionless figure normalized so the fastest solver = 1, on his hardware — their raw per-problem seconds are in the table below).

Runtimes in seconds (Otspot measured; MOSEK/ECOS/COPT are Mittelmann's published values; `f` = that solver failed; `nnz` = nonzeros):

| Problem | nnz | Otspot | MOSEK | ECOS | COPT |
|---|---:|---|---:|---:|---:|
| chainsing-50000-1 | 0.9M | **6** | 3 | f | 3 |
| chainsing-50000-2 | 0.75M | **8** | 4 | f | 3 |
| chainsing-50000-3 | 0.6M | **6** | 3 | f | 2 |
| beam7 | 15M | 481 | 17 | 206 | 18 |
| firL2L1alph | 10M | 1064 | 6 | 202 | 5 |
| firL2Linfeps | 19M | 1841 | 25 | 687 | 14 |
| firL1Linfeps | 9.9M | timeout (>3600s) | 26 | 2531 | 13 |
| firL1 | 40M | timeout (>1000s) | 16 | 1305 | 9 |
| firL1Linfalph | 80M | timeout (>1000s) | 56 | 2847 | 23 |
| firL2L1eps | 40M | timeout (>1000s) | 14 | 797 | 9 |
| firL2a | 50M | timeout (>1000s) | 3 | 945 | 4 |
| firLinf | 80M | timeout (>1000s) | 95 | 3479 | 27 |
| wbNRL | 39M | timeout (>1000s) | 9 | 1333 | 7 |
| dsNRL | 67M | timeout (>1000s) | 56 | f | 27 |
| beam30 | 64M | timeout (>1000s) | 99 | 2465 | 84 |
| db-joint-soerensen | 6M | timeout (>1000s) | 29 | f | 46 |
| db-plate-yield-line | 1.5M | timeout (>1000s) | 6 | f | 5 |
| firL2Linfalph | 122M | OOM (>18 GB) | 27 | f | 25 |
| **solved** | | **6/18** | 18/18 | 11/18 | 18/18 |

Where Otspot wins: all three `chainsing-50000` instances (50k rotated cones, ~1M nonzeros) solve in 6–8s while ECOS fails all three (MOSEK/COPT take ~3s). Where Otspot loses: the large dense-Jacobian `fir`/`db` instances (10–122M nonzeros) time out or exhaust memory — its conic IPM does not yet scale to systems that MOSEK/COPT dispatch in seconds and ECOS (on the 11 it handles) in minutes. Otspot is a developing OSS SOCP solver: competitive on structured sparse cone problems, not yet on large dense ones. (The solver also supports cone types `F`/`L±`/`L=`/`Q`/`QR` and MISOCP via branch-and-bound; `EXP` and PSD cones are rejected as unsupported.)

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

SOCP has no `bench_parallel.sh` harness yet (that script is `.mps`/`.qps`/`.qplib` only); run the `solve_cbf` example directly with an external timeout, one result file per problem. **Run strictly sequentially (jobs=1)** — the large `fir` instances each need up to ~18 GB RSS, so parallel runs collectively exhaust memory and abort. Capture `timeout`'s real exit code directly (do **not** pipe `solve_cbf` through `grep` before reading `$?` — a pipeline's `$?` is grep's status, which masks the 124 timeout code):

```bash
cargo build --release --example solve_cbf
out=/tmp/socp18; mkdir -p "$out"
# Mittelmann Large-SOCP 18; CBLIB download stems vary (most 2013_<name>, but
# beam7/beam30/chainsing-* have no 2013_ prefix) — adjust per plato.asu.edu/ftp/socp.html.
for n in beam7 beam30 chainsing-50000-1 chainsing-50000-2 chainsing-50000-3 \
         db-joint-soerensen db-plate-yield-line dsNRL firL1 firL1Linfalph \
         firL1Linfeps firL2L1alph firL2L1eps firL2Linfalph firL2Linfeps \
         firL2a firLinf wbNRL; do
  f="data/cblib/$n.cbf"
  timeout 3600 ./target/release/examples/solve_cbf --eps 1e-6 "$f" > "$out/$n.csv" 2>/dev/null
  rc=$?   # no pipe: 124 = timeout, 137/134 = OOM, 0 = see status in the CSV
  [ "$rc" = 124 ] && echo "$n,Timeout,,,3600.0" >> "$out/$n.csv"
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
otspot-dev/      # dev-only binaries (qps_benchmark, qp_runner, …; not published)
examples/        # solve_lp, solve_qp
tests/           # integration tests
scripts/         # data-generation scripts
```

## License

[GNU Affero General Public License v3.0 (AGPL-3.0-only)](LICENSE)
