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

use otspot_core::options::SolverOptions;
use otspot_dev::screening::{is_bug, load_baseline, screen_single, DEFAULT_REL_TOL, DEFAULT_TIMEOUT_SEC};
use std::fs;
use std::path::Path;

const DEFAULT_PROBLEMS_DIR: &str = "data/lp_problems";
const DEFAULT_BASELINE_CSV: &str = "data/baseline_objectives/netlib_lp.csv";

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

    let mut bugs = 0usize;
    let mut pass = 0usize;
    let mut total_time = 0.0f64;
    let mut bug_list: Vec<String> = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();

        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(timeout_sec);

        let screen = screen_single(&path, &name, &opts, &baseline, rel_tol);
        total_time += screen.elapsed_secs;

        if is_bug(&screen.verdict) {
            bugs += 1;
            bug_list.push(format!("  {} [{:.2}s]: {:?}", screen.name, screen.elapsed_secs, screen.verdict));
        } else {
            pass += 1;
        }
    }

    eprintln!("\n=== LP COVERAGE SCREEN SUMMARY ===");
    eprintln!("Total problems: {}", entries.len());
    eprintln!("PASS: {}", pass);
    eprintln!("BUGS: {}", bugs);
    eprintln!("Total wall time: {:.2}s", total_time);
    if !bug_list.is_empty() {
        eprintln!("\n=== BUG LIST ===");
        for line in &bug_list {
            eprintln!("{}", line);
        }
        std::process::exit(1);
    }
}
