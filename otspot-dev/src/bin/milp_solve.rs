//! Solve a MILP from an MPS file with otspot's branch-and-bound MILP solver.
//!
//! Reads integer variables from INTORG/INTEND markers and BV/LI/UI bounds
//! (see `io::mps::parse_milp`). Prints a small key:value report parseable by the
//! MILP-vs-HiGHS comparison harness.
//!
//! Usage:
//!   `cargo run --release --bin milp_solve -- <file.mps> [--timeout <secs>] [--eps <tol>] [--no-cuts] [--cut-rounds N]`

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
        eprintln!("usage: milp_solve <file.mps> [--timeout <secs>] [--eps <tol>] [--no-cuts] [--cut-rounds N]");
        return ExitCode::from(2);
    }

    let mut path: Option<String> = None;
    let mut timeout_secs = 100.0_f64;
    let mut eps = 1e-6_f64;
    let mut cuts = true;
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
            "--no-cuts" => cuts = false,
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
    let mut cfg = MipConfig::default();
    cfg.gap_tol = eps;
    cfg.integer_feas_tol = eps;
    cfg.cuts = cuts;
    cfg.max_cut_rounds = cut_rounds;

    let profile = otspot_core::diag::lp_scale_profile_enabled();
    if profile {
        otspot_core::diag::reset_lp_scale_profile();
        otspot_core::diag::reset_simplex_fallback_profile();
    }

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
    println!("fp_incumbent_found: {}", stats.fp_incumbent_found);
    println!("max_depth: {}", stats.max_depth_seen);
    println!("pruned: {}", stats.pruned);
    println!("propagation_pruned: {}", stats.propagation_pruned);
    println!(
        "conflict_clauses_learned: {}",
        stats.conflict_clauses_learned
    );
    println!("conflict_pruned: {}", stats.conflict_pruned);
    println!("rc_vars_fixed: {}", stats.rc_vars_fixed);
    println!("rens_calls: {}", stats.rens_calls);
    println!("rens_improvements: {}", stats.rens_improvements);
    println!("rins_calls: {}", stats.rins_calls);
    println!("rins_improvements: {}", stats.rins_improvements);
    println!("local_branching_calls: {}", stats.local_branching_calls);
    println!(
        "local_branching_improvements: {}",
        stats.local_branching_improvements
    );
    println!("tree_cut_rounds: {}", stats.tree_cut_rounds);
    println!("lp_presolve_us: {}", stats.lp_presolve_us_total);
    println!("lp_solve_us: {}", stats.lp_solve_us_total);
    println!("lp_postsolve_us: {}", stats.lp_postsolve_us_total);
    if profile {
        println!("lp_solve_us_root: {}", stats.lp_solve_us_root);
        println!("lp_solve_us_desc: {}", stats.lp_solve_us_desc);
        println!("lp_scale_us_root: {}", stats.lp_scale_us_root);
        println!("lp_scale_us_desc: {}", stats.lp_scale_us_desc);
        println!("lp_scale_calls_root: {}", stats.lp_scale_calls_root);
        println!("lp_scale_calls_desc: {}", stats.lp_scale_calls_desc);
        println!("fp_us: {}", stats.fp_us);
        println!("root_cut_us: {}", stats.root_cut_us);
        println!("node_propagation_us: {}", stats.node_propagation_us);
        println!("strong_branch_calls: {}", stats.strong_branch_calls);
        println!(
            "strong_branch_candidates: {}",
            stats.strong_branch_candidates
        );
        println!("strong_branch_lp_solves: {}", stats.strong_branch_lp_solves);
        println!("strong_branch_us: {}", stats.strong_branch_us);
        println!(
            "fallback_ub_violation_out_of_scope: {}",
            stats.fallback_ub_violation_out_of_scope
        );
        println!(
            "fallback_phase1_bound_violation: {}",
            stats.fallback_phase1_bound_violation
        );
        println!(
            "fallback_crash_infeasible: {}",
            stats.fallback_crash_infeasible
        );
    }
    if stats.nodes_processed > 0 {
        let n = stats.nodes_processed as f64;
        println!(
            "per_node_us: presolve={:.1} solve={:.1} postsolve={:.1}",
            stats.lp_presolve_us_total as f64 / n,
            stats.lp_solve_us_total as f64 / n,
            stats.lp_postsolve_us_total as f64 / n,
        );
    }

    ExitCode::SUCCESS
}
