//! MIP speed benchmark — MILP/MIQP scaling study.
//!
//! Usage: cargo run --release --bin mip_speed_bench -- [--timeout <secs>] [--out <csv>]
//!
//! Builds synthetic MILP (knapsack/assignment-style) and convex MIQP
//! (PSD Q = LLᵀ + ridge) problems with a deterministic LCG (see `kernels`), so
//! runs are reproducible without external data files. Each CSV row is one
//! distinct problem instance.
//!
//! Output CSV columns:
//!   problem_type, n, m, int_vars, int_ratio, density, seed,
//!   status, objective, nodes_processed, incumbent_updates,
//!   max_depth_seen, pruned, wall_ms, timeout_hit

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use otspot_core::{
    options::{MipBranching, MipConfig, SolverOptions},
    solve_milp_with_stats, solve_miqp_with_stats, MilpProblem, MiqpProblem,
};
use std::{
    fs::File,
    io::{BufWriter, Write as IoWrite},
    time::Instant,
};

mod kernels;
use kernels::{gen_assignment_milp, gen_convex_miqp, gen_knapsack_milp, knapsack_weights_capacity};

// ---------------------------------------------------------------------------
// Result row
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Row {
    problem_type: &'static str,
    n: usize,
    m: usize,
    n_int: usize,
    int_ratio: f64,
    density: f64,
    seed: u64,
    status: String,
    objective: String,
    nodes_processed: usize,
    incumbent_updates: usize,
    max_depth_seen: usize,
    pruned: usize,
    wall_ms: f64,
    timeout_hit: bool,
    // per-node cost breakdown
    relax_total_ms: f64,
    relax_root_ms: f64,
    relax_desc_ms: f64,
    relax_optimal_ms: f64,
    relax_infeasible_ms: f64,
    lp_presolve_us: u64,
    lp_solve_us: u64,
    lp_postsolve_us: u64,
    bounds_bytes_per_node: usize,
}

impl Row {
    fn csv_header() -> &'static str {
        "problem_type,n,m,n_int,int_ratio,density,seed,\
         status,objective,nodes_processed,incumbent_updates,\
         max_depth_seen,pruned,wall_ms,timeout_hit,\
         relax_total_ms,relax_root_ms,relax_desc_ms,\
         relax_optimal_ms,relax_infeasible_ms,\
         lp_presolve_us,lp_solve_us,lp_postsolve_us,\
         bounds_bytes_per_node"
    }

    fn csv_line(&self) -> String {
        format!(
            "{},{},{},{},{:.2},{:.2},{},{},{},{},{},{},{},{:.1},{},\
             {:.3},{:.3},{:.3},{:.3},{:.3},{},{},{},{}",
            self.problem_type,
            self.n,
            self.m,
            self.n_int,
            self.int_ratio,
            self.density,
            self.seed,
            self.status,
            self.objective,
            self.nodes_processed,
            self.incumbent_updates,
            self.max_depth_seen,
            self.pruned,
            self.wall_ms,
            self.timeout_hit,
            self.relax_total_ms,
            self.relax_root_ms,
            self.relax_desc_ms,
            self.relax_optimal_ms,
            self.relax_infeasible_ms,
            self.lp_presolve_us,
            self.lp_solve_us,
            self.lp_postsolve_us,
            self.bounds_bytes_per_node,
        )
    }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

fn run_milp(problem: &MilpProblem, opts: &SolverOptions, cfg: &MipConfig, timeout_secs: f64) -> Row {
    let n = problem.num_vars();
    let m = problem.lp.num_constraints;
    let n_int = problem.integer_vars.len();

    let start = Instant::now();
    let (result, stats) = solve_milp_with_stats(problem, opts, cfg);
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    use otspot_core::problem::SolveStatus::*;
    let timeout_hit = matches!(result.status, Timeout) || wall_ms / 1000.0 >= timeout_secs * 0.99;

    Row {
        problem_type: "MILP",
        n,
        m,
        n_int,
        int_ratio: n_int as f64 / n as f64,
        density: if m > 0 { m as f64 / n as f64 } else { 0.0 },
        seed: 0,
        status: format!("{:?}", result.status),
        objective: if result.objective.is_finite() {
            format!("{:.6}", result.objective)
        } else {
            format!("{}", result.objective)
        },
        nodes_processed: stats.nodes_processed,
        incumbent_updates: stats.incumbent_updates,
        max_depth_seen: stats.max_depth_seen,
        pruned: stats.pruned,
        wall_ms,
        timeout_hit,
        relax_total_ms: stats.relaxation_time_total_ms,
        relax_root_ms: stats.relaxation_time_root_ms,
        relax_desc_ms: stats.relaxation_time_desc_ms,
        relax_optimal_ms: stats.relaxation_time_optimal_ms,
        relax_infeasible_ms: stats.relaxation_time_infeasible_ms,
        lp_presolve_us: stats.lp_presolve_us_total,
        lp_solve_us: stats.lp_solve_us_total,
        lp_postsolve_us: stats.lp_postsolve_us_total,
        bounds_bytes_per_node: stats.approx_bounds_bytes_per_node,
    }
}

fn run_miqp(problem: &MiqpProblem, opts: &SolverOptions, cfg: &MipConfig, timeout_secs: f64) -> Row {
    let n = problem.num_vars();
    let m = problem.qp.num_constraints;
    let n_int = problem.integer_vars.len();

    let start = Instant::now();
    let (result, stats) = solve_miqp_with_stats(problem, opts, cfg);
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    use otspot_core::problem::SolveStatus::*;
    let timeout_hit = matches!(result.status, Timeout) || wall_ms / 1000.0 >= timeout_secs * 0.99;

    Row {
        problem_type: "MIQP",
        n,
        m,
        n_int,
        int_ratio: n_int as f64 / n as f64,
        density: if m > 0 { m as f64 / n as f64 } else { 0.0 },
        seed: 0,
        status: format!("{:?}", result.status),
        objective: if result.objective.is_finite() {
            format!("{:.6}", result.objective)
        } else {
            format!("{}", result.objective)
        },
        nodes_processed: stats.nodes_processed,
        incumbent_updates: stats.incumbent_updates,
        max_depth_seen: stats.max_depth_seen,
        pruned: stats.pruned,
        wall_ms,
        timeout_hit,
        relax_total_ms: stats.relaxation_time_total_ms,
        relax_root_ms: stats.relaxation_time_root_ms,
        relax_desc_ms: stats.relaxation_time_desc_ms,
        relax_optimal_ms: stats.relaxation_time_optimal_ms,
        relax_infeasible_ms: stats.relaxation_time_infeasible_ms,
        lp_presolve_us: stats.lp_presolve_us_total,
        lp_solve_us: stats.lp_solve_us_total,
        lp_postsolve_us: stats.lp_postsolve_us_total,
        bounds_bytes_per_node: stats.approx_bounds_bytes_per_node,
    }
}

/// Brute-force the all-integer knapsack over {0,1}^n for the inline sanity check.
fn brute_force_knapsack(n_int: usize, c: &[f64], weights: &[f64], capacity: f64) -> Option<f64> {
    if n_int > 20 {
        return None; // too slow
    }
    let mut best = None::<f64>;
    for mask in 0u32..(1u32 << n_int) {
        let mut w_sum = 0.0_f64;
        let mut obj = 0.0_f64;
        for j in 0..n_int {
            if mask & (1 << j) != 0 {
                w_sum += weights[j];
                obj += c[j];
            }
        }
        if w_sum <= capacity + 1e-9 {
            best = Some(best.map_or(obj, |b: f64| b.min(obj)));
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // 60s/case: a scaling study wants the per-size explosion threshold across
    // many instances, not convergence proofs, so the per-case budget sits well
    // below the 1000s production bench to keep the full sweep tractable.
    let mut timeout_secs = 60.0_f64;
    let mut out_path = "reports/mip_speed_bench.csv".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--timeout" => {
                i += 1;
                timeout_secs = args[i].parse().expect("--timeout value");
            }
            "--out" => {
                i += 1;
                out_path = args[i].clone();
            }
            _ => {}
        }
        i += 1;
    }

    let sizes: &[usize] = &[10, 20, 40, 80, 160];
    let int_ratios: &[f64] = &[0.5, 1.0]; // 50% int, all-int
    let densities: &[f64] = &[0.3, 0.6]; // constraint density (assignment / MIQP)
    let seeds: &[u64] = &[42, 137, 999]; // multiple data patterns

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(timeout_secs);
    let cfg = MipConfig {
        gap_tol: 1e-6,
        branching: MipBranching::MostFractional,
        ..Default::default()
    };

    // Knapsack is single-constraint → density-independent (one run per
    // (n, int_ratio, seed)); assignment and MIQP run per density. Every CSV
    // row is therefore a distinct problem — no duplicate solves.
    let knapsack_cases = sizes.len() * int_ratios.len() * seeds.len();
    let density_cases = sizes.len() * int_ratios.len() * densities.len() * seeds.len() * 2;
    let total = knapsack_cases + density_cases;
    println!("MIP speed bench: {} cases, timeout={:.0}s each", total, timeout_secs);
    println!("Output: {}", out_path);

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output dir");
        }
    }

    let file = File::create(&out_path).expect("create CSV");
    let mut w = BufWriter::new(file);
    writeln!(w, "{}", Row::csv_header()).unwrap();

    let mut done = 0usize;
    let mut correctness_mismatches = 0usize;
    let progress = |w: &mut BufWriter<File>, r: &Row, done: &mut usize| {
        writeln!(w, "{}", r.csv_line()).unwrap();
        *done += 1;
        if *done % 10 == 0 {
            print!("\r  {}/{} done...", *done, total);
            std::io::stdout().flush().unwrap();
        }
    };

    for &n in sizes {
        for &int_ratio in int_ratios {
            // --- MILP knapsack (density-independent) ---
            for &seed in seeds {
                let prob = gen_knapsack_milp(n, int_ratio, seed);
                let n_int = prob.integer_vars.len();
                // Brute-force only when ALL variables are integer (a mixed
                // knapsack needs an LP sub-solve for the continuous part).
                let do_bf_check = n <= 15 && n_int == n;
                let mut r = run_milp(&prob, &opts, &cfg, timeout_secs);
                r.seed = seed;
                if do_bf_check && !r.timeout_hit && r.status.contains("Optimal") {
                    let (weights, cap) = knapsack_weights_capacity(n, seed);
                    if let Some(bf_obj) = brute_force_knapsack(n, &prob.lp.c, &weights, cap) {
                        let solver_obj: f64 = r.objective.parse().unwrap_or(f64::INFINITY);
                        if (solver_obj - bf_obj).abs() > 1e-3 {
                            eprintln!(
                                "MISMATCH knapsack n={} seed={}: solver={:.4} bf={:.4}",
                                n, seed, solver_obj, bf_obj
                            );
                            correctness_mismatches += 1;
                        }
                    }
                }
                progress(&mut w, &r, &mut done);
            }

            // --- Assignment MILP + convex MIQP (density-controlled) ---
            for &density in densities {
                for &seed in seeds {
                    let prob = gen_assignment_milp(n, int_ratio, density, seed);
                    let mut r = run_milp(&prob, &opts, &cfg, timeout_secs);
                    r.seed = seed;
                    r.problem_type = "MILP_assign";
                    progress(&mut w, &r, &mut done);

                    let prob = gen_convex_miqp(n, int_ratio, density, seed);
                    let mut r = run_miqp(&prob, &opts, &cfg, timeout_secs);
                    r.seed = seed;
                    progress(&mut w, &r, &mut done);
                }
            }
        }
    }
    println!("\r  {}/{} done.        ", done, total);

    w.flush().unwrap();
    println!("CSV written to: {}", out_path);

    if correctness_mismatches > 0 {
        eprintln!(
            "\nWARNING: {} correctness mismatch(es) detected! \
             Review solver behavior for small instances.",
            correctness_mismatches
        );
        std::process::exit(1);
    }
}
