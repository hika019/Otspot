//! Diagnostic LP runner for simplex path profiling.

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use otspot_core::options::{SolverOptions, Tolerance};
use otspot_core::solve_lp_with;
use otspot_io::mps::parse_mps_file;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: lp_solve_diag <file.mps|file.qps> [--timeout <secs>] [--eps <tol>]");
        return ExitCode::from(2);
    }

    let mut path: Option<String> = None;
    let mut timeout_secs = 100.0_f64;
    let mut eps = 1e-6_f64;
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
            other => path = Some(other.to_string()),
        }
        i += 1;
    }

    let path = match path {
        Some(p) => p,
        None => {
            eprintln!("error: no LP file given");
            return ExitCode::from(2);
        }
    };

    let lp = match parse_mps_file(Path::new(&path)) {
        Ok(p) => p,
        Err(e) => {
            println!("file: {path}");
            println!("status: PARSE_ERROR");
            println!("error: {e}");
            return ExitCode::from(1);
        }
    };

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(timeout_secs);
    opts.tolerance = Some(Tolerance::Custom(eps));

    let profile = otspot_core::diag::lp_scale_profile_enabled();
    if profile {
        otspot_core::diag::reset_lp_scale_profile();
        otspot_core::diag::reset_simplex_fallback_profile();
    }

    let start = Instant::now();
    let res = solve_lp_with(&lp, &opts);
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("file: {path}");
    println!("n_vars: {}", lp.num_vars);
    println!("n_cons: {}", lp.num_constraints);
    println!("status: {:?}", res.status);
    if res.objective.is_finite() {
        println!("objective: {:.9}", res.objective);
    } else {
        println!("objective: {}", res.objective);
    }
    println!("wall_ms: {wall_ms:.3}");
    if let Some(tb) = res.timing_breakdown {
        println!("lp_presolve_us: {}", tb.presolve_us);
        println!("lp_solve_us: {}", tb.solve_us);
        println!("lp_postsolve_us: {}", tb.postsolve_us);
    }
    if profile {
        let scale = otspot_core::diag::lp_scale_profile_snapshot();
        let fallback = otspot_core::diag::simplex_fallback_profile_snapshot();
        println!("lp_scale_us: {}", scale.scale_us);
        println!("lp_scale_calls: {}", scale.calls);
        println!(
            "fallback_ub_violation_out_of_scope: {}",
            fallback.ub_violation_out_of_scope
        );
        println!(
            "fallback_phase1_bound_violation: {}",
            fallback.phase1_bound_violation
        );
        println!("fallback_crash_infeasible: {}", fallback.crash_infeasible);
    }

    ExitCode::SUCCESS
}
