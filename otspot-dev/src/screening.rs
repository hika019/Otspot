// LP coverage screening utilities.
// Moved from otspot-io (where it was pub) to otspot-dev (publish = false),
// so it no longer appears in the public otspot-io API surface.

use otspot_io::qps::parse_qps;
use otspot_core::options::SolverOptions;
use otspot_core::problem::SolveStatus;
use otspot_core::qp::solve_qp_with;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

pub const DEFAULT_TIMEOUT_SEC: f64 = 20.0;
pub const DEFAULT_REL_TOL: f64 = 1e-3;

/// Classification of a single problem's solve result.
#[derive(Debug)]
pub enum ScreenVerdict {
    /// Optimal and objective matches baseline within tolerance.
    Optimal,
    /// Optimal status but objective deviates from baseline.
    ObjMismatch { got: f64, expected: f64, rel_err: f64 },
    /// Non-optimal status when Optimal was expected.
    BadStatus { status: SolveStatus, expected_optimal: f64 },
    /// Solver returned Timeout.
    Timeout,
    /// Small problem solved correctly but too slowly.
    Slow { secs: f64 },
    /// QPS parse failed.
    ParseError,
    /// Solver panicked.
    Panic,
}

/// Result of screening one LP.
pub struct ScreenEntry {
    pub name: String,
    pub verdict: ScreenVerdict,
    pub elapsed_secs: f64,
}

/// Load the baseline objective CSV. Panics if the file cannot be read.
pub fn load_baseline(csv_path: &str) -> HashMap<String, f64> {
    crate::bench_utils::load_baseline_objectives(std::path::Path::new(csv_path), true)
}

/// Screen a single LP file: parse, solve, classify verdict.
pub fn screen_single(
    path: &Path,
    name: &str,
    opts: &SolverOptions,
    baseline: &HashMap<String, f64>,
    rel_tol: f64,
) -> ScreenEntry {
    let problem = match parse_qps(path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[parse_fail] {}: {:?}", name, e);
            return ScreenEntry {
                name: name.to_string(),
                verdict: ScreenVerdict::ParseError,
                elapsed_secs: 0.0,
            };
        }
    };

    let start = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        solve_qp_with(&problem, opts)
    }));
    let elapsed = start.elapsed().as_secs_f64();

    let verdict = match result {
        Err(_) => {
            eprintln!("[PANIC] {}: panicked during solve", name);
            ScreenVerdict::Panic
        }
        Ok(r) => {
            let expected = baseline.get(name).copied();
            match (r.status, expected) {
                (SolveStatus::Optimal, Some(exp)) => {
                    let exp_adj = exp + problem.obj_offset;
                    let denom = exp_adj.abs().max(1.0);
                    let rel_err = (r.objective - exp_adj).abs() / denom;
                    if rel_err > rel_tol {
                        if problem.obj_offset != 0.0 {
                            eprintln!(
                                "[OBJ_MISMATCH] {}: got={:.6e} netlib_ref={:.6e} obj_offset={:.6e} exp_adj={:.6e} rel={:.2e} {:.2}s",
                                name, r.objective, exp, problem.obj_offset, exp_adj, rel_err, elapsed
                            );
                        } else {
                            eprintln!(
                                "[OBJ_MISMATCH] {}: got={:.6e} expected={:.6e} rel={:.2e} {:.2}s",
                                name, r.objective, exp_adj, rel_err, elapsed
                            );
                        }
                        ScreenVerdict::ObjMismatch { got: r.objective, expected: exp_adj, rel_err }
                    } else if problem.num_vars < 200 && elapsed > 30.0 {
                        eprintln!("[SLOW] {}: small problem took {:.2}s", name, elapsed);
                        ScreenVerdict::Slow { secs: elapsed }
                    } else {
                        if problem.obj_offset != 0.0 {
                            eprintln!(
                                "[OK] {}: obj={:.6e} (netlib_ref={:.6e} + obj_offset={:.6e}) {:.2}s",
                                name, r.objective, exp, problem.obj_offset, elapsed
                            );
                        } else {
                            eprintln!("[OK] {}: obj={:.6e} {:.2}s", name, r.objective, elapsed);
                        }
                        ScreenVerdict::Optimal
                    }
                }
                (SolveStatus::Optimal, None) => {
                    eprintln!("[OK_NO_REF] {}: obj={:.6e} {:.2}s", name, r.objective, elapsed);
                    ScreenVerdict::Optimal
                }
                (SolveStatus::Timeout, _) => {
                    eprintln!("[TIMEOUT] {}: {:.2}s", name, elapsed);
                    ScreenVerdict::Timeout
                }
                (status, exp) => {
                    eprintln!("[BAD_STATUS] {}: {:?} exp={:?} {:.2}s", name, status, exp, elapsed);
                    ScreenVerdict::BadStatus {
                        status,
                        expected_optimal: exp.unwrap_or(0.0),
                    }
                }
            }
        }
    };

    ScreenEntry { name: name.to_string(), verdict, elapsed_secs: elapsed }
}

/// Returns `true` if the verdict represents a failure (not Optimal / Slow).
pub fn is_bug(v: &ScreenVerdict) -> bool {
    !matches!(v, ScreenVerdict::Optimal)
}
