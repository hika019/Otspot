//! LP coverage screening binary.
//!
//! Runs solve() on all available Netlib LP problems and reports failures.
//! Purpose: expose LP simplex bugs (RANGES / FR / OBJSENSE MAX / negative bounds, etc.)
//! that the regular LP unit test suite does not cover.
//!
//! Usage: `cargo run --release --bin lp_screen -- [--dir <dir>] [--csv <csv>] [--tol <rel_tol>] [--timeout <secs>]`
//!
//! Baseline CSV values are Netlib official values (MINOS 5.3).
//! This solver adds obj_offset (N-row RHS) to the reported objective value.
//! Comparison: exp_adjusted = netlib_ref + problem.obj_offset.

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::SolveStatus;
use solver::qp::solve_qp_with;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

const DEFAULT_TIMEOUT_SEC: f64 = 20.0;
const DEFAULT_PROBLEMS_DIR: &str = "data/lp_problems";
const DEFAULT_BASELINE_CSV: &str = "data/baseline_objectives/netlib_lp.csv";
const DEFAULT_REL_TOL: f64 = 1e-3;

#[derive(Debug)]
#[allow(dead_code)]
enum Verdict {
    ObjMismatch { got: f64, expected: f64, rel_err: f64 },
    BadStatus { status: SolveStatus, expected_optimal: f64 },
    Timeout,
    Slow { secs: f64 },
}

fn load_baseline(csv_path: &str) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    let content = fs::read_to_string(csv_path)
        .unwrap_or_else(|e| panic!("Failed to read baseline {}: {}", csv_path, e));
    for line in content.lines() {
        if line.starts_with('#') || line.is_empty() || line.starts_with("problem_name") {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() >= 2 {
            if let Ok(v) = cols[1].parse::<f64>() {
                map.insert(cols[0].to_string(), v);
            }
        }
    }
    map
}

fn parse_args() -> (String, String, f64, f64) {
    let args: Vec<String> = std::env::args().collect();
    let mut dir = DEFAULT_PROBLEMS_DIR.to_string();
    let mut csv = DEFAULT_BASELINE_CSV.to_string();
    let mut rel_tol = DEFAULT_REL_TOL;
    let mut timeout = DEFAULT_TIMEOUT_SEC;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" if i + 1 < args.len() => { dir = args[i + 1].clone(); i += 2; }
            "--csv" if i + 1 < args.len() => { csv = args[i + 1].clone(); i += 2; }
            "--tol" if i + 1 < args.len() => {
                rel_tol = args[i + 1].parse().unwrap_or(DEFAULT_REL_TOL); i += 2;
            }
            "--timeout" if i + 1 < args.len() => {
                timeout = args[i + 1].parse().unwrap_or(DEFAULT_TIMEOUT_SEC); i += 2;
            }
            _ => { i += 1; }
        }
    }
    (dir, csv, rel_tol, timeout)
}

fn main() {
    let (problems_dir, baseline_csv, rel_tol, timeout_sec) = parse_args();

    let dir = Path::new(&problems_dir);
    if !dir.exists() {
        eprintln!("{} not found — run scripts/netlib_lp_download.sh first", problems_dir);
        std::process::exit(1);
    }

    let baseline = load_baseline(&baseline_csv);

    let mut entries: Vec<_> = fs::read_dir(dir)
        .expect("read_dir failed")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|s| s == "QPS" || s == "qps").unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.path());

    let mut bugs: Vec<(String, Verdict, f64)> = Vec::new();
    let mut pass = 0;
    let mut total_time = 0.0;

    for entry in &entries {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();

        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(timeout_sec);

        let problem = match parse_qps(&path) {
            Ok(p) => p,
            Err(e) => {
                bugs.push((
                    name.clone(),
                    Verdict::BadStatus { status: SolveStatus::NumericalError, expected_optimal: 0.0 },
                    0.0,
                ));
                eprintln!("[parse_fail] {}: {:?}", name, e);
                continue;
            }
        };

        let start = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solve_qp_with(&problem, &opts)
        }));
        let elapsed = start.elapsed().as_secs_f64();
        total_time += elapsed;

        match result {
            Err(_) => {
                bugs.push((
                    name.clone(),
                    Verdict::BadStatus {
                        status: SolveStatus::NumericalError,
                        expected_optimal: *baseline.get(&name).unwrap_or(&0.0),
                    },
                    elapsed,
                ));
                eprintln!("[PANIC] {}: panicked during solve", name);
            }
            Ok(r) => {
                let expected = baseline.get(&name).copied();
                match (r.status, expected) {
                    (SolveStatus::Optimal, Some(exp)) => {
                        let exp_adjusted = exp + problem.obj_offset;
                        let denom = exp_adjusted.abs().max(1.0);
                        let rel_err = (r.objective - exp_adjusted).abs() / denom;
                        if rel_err > rel_tol {
                            bugs.push((
                                name.clone(),
                                Verdict::ObjMismatch { got: r.objective, expected: exp_adjusted, rel_err },
                                elapsed,
                            ));
                            if problem.obj_offset != 0.0 {
                                eprintln!(
                                    "[OBJ_MISMATCH] {}: got={:.6e} netlib_ref={:.6e} obj_offset={:.6e} exp_adj={:.6e} rel={:.2e} time={:.2}s",
                                    name, r.objective, exp, problem.obj_offset, exp_adjusted, rel_err, elapsed
                                );
                            } else {
                                eprintln!(
                                    "[OBJ_MISMATCH] {}: got={:.6e} expected={:.6e} rel={:.2e} time={:.2}s",
                                    name, r.objective, exp_adjusted, rel_err, elapsed
                                );
                            }
                        } else {
                            pass += 1;
                            if problem.num_vars < 200 && elapsed > 30.0 {
                                bugs.push((name.clone(), Verdict::Slow { secs: elapsed }, elapsed));
                                eprintln!("[SLOW] {}: small problem took {:.2}s", name, elapsed);
                            } else if problem.obj_offset != 0.0 {
                                eprintln!(
                                    "[OK] {}: obj={:.6e} (netlib_ref={:.6e} + obj_offset={:.6e}) time={:.2}s",
                                    name, r.objective, exp, problem.obj_offset, elapsed
                                );
                            } else {
                                eprintln!("[OK] {}: obj={:.6e} time={:.2}s", name, r.objective, elapsed);
                            }
                        }
                    }
                    (SolveStatus::Optimal, None) => {
                        pass += 1;
                        eprintln!("[OK_NO_REF] {}: obj={:.6e} time={:.2}s", name, r.objective, elapsed);
                    }
                    (SolveStatus::Timeout, _) => {
                        bugs.push((name.clone(), Verdict::Timeout, elapsed));
                        eprintln!("[TIMEOUT] {}: time={:.2}s", name, elapsed);
                    }
                    (status, exp) => {
                        let s_dbg = format!("{:?}", status);
                        bugs.push((
                            name.clone(),
                            Verdict::BadStatus { status, expected_optimal: exp.unwrap_or(0.0) },
                            elapsed,
                        ));
                        eprintln!(
                            "[BAD_STATUS] {}: status={} obj={:.6e} time={:.2}s",
                            name, s_dbg, r.objective, elapsed
                        );
                    }
                }
            }
        }
    }

    eprintln!("\n=== LP COVERAGE SCREEN SUMMARY ===");
    eprintln!("Total problems: {}", entries.len());
    eprintln!("PASS: {}", pass);
    eprintln!("BUGS: {}", bugs.len());
    eprintln!("Total wall time: {:.2}s", total_time);
    if !bugs.is_empty() {
        eprintln!("\n=== BUG LIST ===");
        for (name, verdict, time) in &bugs {
            eprintln!("  {} [{:.2}s]: {:?}", name, time, verdict);
        }
        std::process::exit(1);
    }
}
