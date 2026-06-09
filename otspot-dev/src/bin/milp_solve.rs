//! Solve a MILP from an MPS file with otspot's branch-and-bound MILP solver.
//!
//! Reads integer variables from INTORG/INTEND markers and BV/LI/UI bounds
//! (see `io::mps::parse_milp`). Prints a small key:value report parseable by the
//! MILP-vs-HiGHS comparison harness.
//!
//! Usage:
//!   `cargo run --release --bin milp_solve -- <file.mps> [--timeout <secs>] [--eps <tol>]`

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use otspot_core::options::{MipConfig, SolverOptions, Tolerance};
use otspot_core::solve_milp_with_stats;
use otspot_io::mps::parse_milp_file;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: milp_solve <file.mps> [--timeout <secs>] [--eps <tol>]");
        return ExitCode::from(2);
    }

    let mut path: Option<String> = None;
    let mut timeout_secs = 100.0_f64;
    let mut eps = 1e-6_f64;
    let mut cuts = false;
    let mut cut_rounds = 0usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--timeout" => {
                i += 1;
                timeout_secs = args[i].parse().expect("--timeout value");
            }
            "--eps" => {
                i += 1;
                eps = args[i].parse().expect("--eps value");
            }
            "--cuts" => cuts = true,
            "--cut-rounds" => {
                i += 1;
                cut_rounds = args[i].parse().expect("--cut-rounds value");
                cuts = true;
            }
            other => path = Some(other.to_string()),
        }
        i += 1;
    }

    let path = match path {
        Some(p) => p,
        None => {
            eprintln!("error: no MPS file given");
            return ExitCode::from(2);
        }
    };

    let milp = match parse_milp_file(Path::new(&path)) {
        Ok(m) => m,
        Err(e) => {
            println!("file: {path}");
            println!("status: PARSE_ERROR");
            println!("error: {e}");
            return ExitCode::from(1);
        }
    };

    let n_vars = milp.num_vars();
    let n_cons = milp.lp.num_constraints;
    let n_int = milp.integer_vars.len();

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(timeout_secs);
    opts.tolerance = Some(Tolerance::Custom(eps));
    let cfg = MipConfig {
        gap_tol: eps,
        integer_feas_tol: eps,
        cuts,
        max_cut_rounds: cut_rounds,
        ..Default::default()
    };

    let start = Instant::now();
    let (res, stats) = solve_milp_with_stats(&milp, &opts, &cfg);
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("file: {path}");
    println!("n_vars: {n_vars}");
    println!("n_cons: {n_cons}");
    println!("n_int: {n_int}");
    println!("status: {:?}", res.status);
    if res.objective.is_finite() {
        println!("objective: {:.9}", res.objective);
    } else {
        println!("objective: {}", res.objective);
    }
    println!("wall_ms: {wall_ms:.3}");
    println!("root_lp_bound: {}", stats.root_lp_bound);
    println!("nodes: {}", stats.nodes_processed);
    println!("incumbent_updates: {}", stats.incumbent_updates);
    println!("max_depth: {}", stats.max_depth_seen);
    println!("pruned: {}", stats.pruned);

    ExitCode::SUCCESS
}
