//! CBLIB SOCP/MISOCP bench runner: solves `.cbf` files and prints one CSV
//! row per problem (`problem,status,objective,iterations,time_sec`).
//!
//! Run: `cargo run --release --example solve_cbf -- [--eps <value>] <file-or-dir.cbf> [...]`
//!
//! `--eps <value>` overrides the convergence tolerance for the whole run
//! (default [`DEFAULT_BENCH_TOL`], `1e-6`); e.g. `--eps 1e-8` for a tighter
//! pass over the same CBLIB set.
//!
//! `otspot_core::conic::{ConicOptions,BbOptions}` both carry a `deadline:
//! Option<Instant>` (checked once per IPM iteration / B&B node), but this
//! runner leaves it `None` and does not set one per file: a single
//! `Instant::now() + per_file_budget` deadline shared across an entire
//! directory's worth of files would apply to only the first file, and a
//! fresh deadline per file needs a per-file loop restructure this bench
//! runner doesn't do. Until that lands, bound wall-clock time per problem by
//! wrapping each *file* invocation in an external timeout instead, e.g.:
//!
//! ```sh
//! for f in data/cblib_socp/*.cbf; do
//!     timeout 180 cargo run --release --example solve_cbf -- "$f"
//! done
//! ```
//!
//! Objective values are reported in the CBF file's original sense (handles
//! `OBJSENSE MAX` sign flip and the `OBJBCOORD` constant via
//! [`CbfProblem::true_objective`]).
//!
//! The `iterations` column is interior-point iterations for a continuous
//! SOCP, or branch-and-bound node count for a MISOCP (the two solvers count
//! different things; there is no shared "iteration" unit across them).

use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

use otspot_core::conic::{solve_misocp, solve_socp, BbOptions, ConicOptions};
use otspot_io::cbf::{parse_cbf, CbfError, CbfProblem};

/// Default convergence tolerance for this bench run (task requirement: eps
/// ~= 1e-6), used when `--eps` is not given. Overridable at runtime via
/// `--eps <value>` (see [`parse_eps_flag`]).
const DEFAULT_BENCH_TOL: f64 = 1e-6;

/// Collects `.cbf` files from a mix of file and directory arguments,
/// expanding directories (non-recursively) and sorting their contents for a
/// deterministic run order.
fn collect_cbf_files(args: &[String]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for arg in args {
        let path = Path::new(arg);
        if path.is_dir() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
                .unwrap_or_else(|e| panic!("read_dir {}: {e}", path.display()))
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("cbf"))
                .collect();
            entries.sort();
            files.extend(entries);
        } else {
            files.push(path.to_path_buf());
        }
    }
    files
}

/// CSV-escapes a status/message field: commas and newlines cannot appear
/// verbatim in a CSV cell (some `SolveStatus` variants carry a free-text
/// payload, e.g. `NotSupported(msg)`).
fn csv_field(s: &str) -> String {
    s.replace(',', ";").replace('\n', " ")
}

/// Pulls a `--eps <value>` flag out of the raw CLI args, returning the
/// resolved tolerance (default [`DEFAULT_BENCH_TOL`] if absent) and the
/// remaining args unchanged and in order (the `.cbf` file/directory
/// positionals `collect_cbf_files` expects). Exits the process with a
/// stderr message and status 2 if `--eps` is given without a value or with
/// a value that does not parse as `f64`.
fn parse_eps_flag(args: Vec<String>) -> (f64, Vec<String>) {
    let mut eps = DEFAULT_BENCH_TOL;
    let mut rest = Vec::with_capacity(args.len());
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        if arg == "--eps" {
            let value = iter.next().unwrap_or_else(|| {
                eprintln!("--eps requires a value, e.g. --eps 1e-8");
                std::process::exit(2);
            });
            eps = value.parse::<f64>().unwrap_or_else(|e| {
                eprintln!("invalid --eps value {value:?}: {e}");
                std::process::exit(2);
            });
        } else {
            rest.push(arg);
        }
    }
    (eps, rest)
}

fn main() {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    let (eps, args) = parse_eps_flag(raw_args);
    if args.is_empty() {
        eprintln!("usage: solve_cbf [--eps <value>] <file-or-dir.cbf> [file-or-dir.cbf ...]");
        std::process::exit(2);
    }
    let files = collect_cbf_files(&args);
    if files.is_empty() {
        eprintln!("no .cbf files found in: {}", args.join(", "));
        std::process::exit(2);
    }

    let conic_opts = ConicOptions {
        tol: eps,
        ..ConicOptions::default()
    };
    let bb_opts = BbOptions {
        int_tol: eps,
        gap_tol: eps,
        ..BbOptions::default()
    };

    println!("problem,status,objective,iterations,time_sec");
    for path in &files {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let start = Instant::now();
        match parse_cbf(path) {
            Err(CbfError::Unsupported(msg)) => {
                let elapsed = start.elapsed().as_secs_f64();
                println!("{name},Unsupported,,,{elapsed:.3}");
                eprintln!("[solve_cbf] {name}: unsupported: {msg}");
            }
            Err(e) => {
                let elapsed = start.elapsed().as_secs_f64();
                println!("{name},ParseError,,,{elapsed:.3}");
                eprintln!("[solve_cbf] {name}: parse error: {e}");
            }
            Ok(cbf @ CbfProblem::Socp { .. }) => {
                let CbfProblem::Socp { ref problem, .. } = cbf else {
                    unreachable!()
                };
                let res = solve_socp(problem, &conic_opts);
                let elapsed = start.elapsed().as_secs_f64();
                let objective = cbf.true_objective(res.objective);
                println!(
                    "{name},{},{objective:.10e},{},{elapsed:.3}",
                    csv_field(&res.status.to_string()),
                    res.iterations
                );
            }
            Ok(cbf @ CbfProblem::Misocp { .. }) => {
                let CbfProblem::Misocp { ref problem, .. } = cbf else {
                    unreachable!()
                };
                let res = solve_misocp(problem, &conic_opts, &bb_opts);
                let elapsed = start.elapsed().as_secs_f64();
                let objective = cbf.true_objective(res.objective);
                println!(
                    "{name},{},{objective:.10e},{},{elapsed:.3}",
                    csv_field(&res.status.to_string()),
                    res.nodes
                );
            }
        }
    }
}
