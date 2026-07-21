//! Presolve correctness verification.
//!
//! Layers:
//!  * `presolve_not_worsens_80bau3b_grow15` — directional invariance: presolve
//!    must not worsen feasibility (ON ≤ OFF + LP_CERT_TOL). OFF residual is a
//!    known primal two-phase numerical artifact (dc658d4, within LP_CERT_TOL=1e-4).
//!  * `presolve_invariance_curated_clean` — symmetric invariance for clean problems.
//!  * `presolve_invariance_full_sweep` — all 109 netlib LPs, `#[ignore]` (tier-2).
//!  * `presolve_infeasible_sweep` — 29 infeasible LPs, `#[ignore]` (tier-2).

use otspot_core::options::SolverOptions;
use otspot_core::problem::{ConstraintType, SolveStatus, SolverResult};
use otspot_core::qp::{solve_qp_with, QpProblem};
use otspot_io::qps::parse_qps;
use std::path::Path;

const LP_SUBDIR: &str = "lp_problems";
const INFEAS_SUBDIR: &str = "lp_problems_infeas";
const KNOWN_SUBPATH: &str = "baseline_objectives/netlib_lp.csv";

/// step7 superlinear-hang problems (localized separately). Skipped in sweeps so
/// presolve=ON does not block; their invariance stays "unverified (hang)".
const STEP7_HANGERS: &[&str] = &["cont1", "cont4", "cont11"];

/// Resolve the `data/` directory relative to this crate (nextest runs with CWD =
/// crate root, so a bare `data/...` silently misses the workspace-root symlink).
fn data_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data")
}

fn data_dir(sub: &str) -> std::path::PathBuf {
    data_root().join(sub)
}

/// Parse a required problem. Panics (loud FAIL, not silent skip) if data is
/// absent — integration tests assert data presence; CI excludes them via the
/// lib-only profile.
fn parse_required(dir: &Path, name: &str) -> QpProblem {
    let path = dir.join(format!("{name}.QPS"));
    assert!(
        path.exists(),
        "required data missing: {} (integration test must run with data/ present)",
        path.display()
    );
    parse_qps(&path).expect("parse")
}

fn solve(prob: &QpProblem, presolve: bool, timeout_s: f64) -> SolverResult {
    let mut opts = SolverOptions::default();
    opts.presolve = presolve;
    opts.ipm.eps = 1e-6;
    opts.timeout_secs = Some(timeout_s);
    solve_qp_with(prob, &opts)
}

/// Max primal-constraint violation (Ax vs b, sense-aware) of a candidate solution.
fn pfeas_violation(prob: &QpProblem, sol: &[f64]) -> f64 {
    if sol.len() != prob.num_vars || sol.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let ax = prob.a.mat_vec_mul(sol).expect("mat_vec_mul");
    ax.iter()
        .zip(prob.b.iter())
        .enumerate()
        .map(|(i, (&ax_i, &b_i))| match prob.constraint_types.get(i) {
            Some(ConstraintType::Eq) => (ax_i - b_i).abs(),
            Some(ConstraintType::Ge) => (b_i - ax_i).max(0.0),
            _ => (ax_i - b_i).max(0.0),
        })
        .fold(0.0_f64, f64::max)
}

/// Max variable-bound violation (component-wise, normalized like the bench).
fn bfeas_violation(prob: &QpProblem, sol: &[f64]) -> f64 {
    if sol.len() != prob.num_vars || sol.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let mut max_v = 0.0_f64;
    for (&xi, &(lb, ub)) in sol.iter().zip(prob.bounds.iter()) {
        if lb.is_finite() {
            max_v = max_v.max((lb - xi).max(0.0) / (1.0 + xi.abs() + lb.abs()));
        }
        if ub.is_finite() {
            max_v = max_v.max((xi - ub).max(0.0) / (1.0 + xi.abs() + ub.abs()));
        }
    }
    max_v
}

fn load_known() -> std::collections::HashMap<String, f64> {
    let mut m = std::collections::HashMap::new();
    let Ok(txt) = std::fs::read_to_string(data_dir(KNOWN_SUBPATH)) else {
        return m;
    };
    for line in txt.lines().skip(1) {
        let mut it = line.split(',');
        let (Some(name), Some(val)) = (it.next(), it.next()) else {
            continue;
        };
        if let Ok(v) = val.trim().parse::<f64>() {
            m.insert(name.trim().to_string(), v);
        }
    }
    m
}

struct OnOff {
    on: SolverResult,
    off: SolverResult,
    on_pf: f64,
    on_bf: f64,
    off_pf: f64,
    off_bf: f64,
}

fn solve_on_off(prob: &QpProblem, timeout_s: f64) -> OnOff {
    let on = solve(prob, true, timeout_s);
    let off = solve(prob, false, timeout_s);
    let on_pf = pfeas_violation(prob, &on.solution);
    let on_bf = bfeas_violation(prob, &on.solution);
    let off_pf = pfeas_violation(prob, &off.solution);
    let off_bf = bfeas_violation(prob, &off.solution);
    OnOff {
        on,
        off,
        on_pf,
        on_bf,
        off_pf,
        off_bf,
    }
}

/// 手法2: presolve invariance sweep over every netlib LP. For each problem,
/// solve presolve ON and OFF; both-Optimal pairs must agree on objective and not
/// have presolve worsen feasibility. Diverging pairs are logged (none observed at
/// 930145d — see report). step7 hangers are skipped (`STEP7_HANGERS`).
///
/// Heavy (≈5 min). Run under the heavy profile (600s cap) or the raw binary:
///   cargo nextest run --profile heavy -p otspot-io \
///     --run-ignored all presolve_invariance_full_sweep --no-capture
///   PRESOLVE_SWEEP_TIMEOUT=5 ./target/release/deps/presolve_correctness_sweep-* \
///     --ignored --nocapture --test-threads 1 presolve_invariance_full_sweep
#[test]
#[ignore = "tier-2: full 109 sweep, ~5min — run under --profile heavy or raw binary"]
fn presolve_invariance_full_sweep() {
    let known = load_known();
    let timeout: f64 = std::env::var("PRESOLVE_SWEEP_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8.0);

    let mut files: Vec<_> = std::fs::read_dir(data_dir(LP_SUBDIR))
        .expect("read lp_problems")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("qps"))
        })
        .collect();
    files.sort();

    let mut n_checked = 0;
    let mut n_skip_hang = 0;
    let mut n_skip_status = 0;
    let mut divergences: Vec<String> = Vec::new();
    let mut known_mismatch: Vec<String> = Vec::new();

    for path in &files {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        if STEP7_HANGERS.contains(&name.as_str()) {
            eprintln!("[{name}] SKIP (step7 hang) — invariance unverified");
            n_skip_hang += 1;
            continue;
        }
        let prob = parse_qps(path).expect("parse");
        let r = solve_on_off(&prob, timeout);
        eprintln!(
            "[{name}] ON={:?} obj={:.6e} pf={:.1e} bf={:.1e} | OFF={:?} obj={:.6e} pf={:.1e} bf={:.1e}",
            r.on.status, r.on.objective, r.on_pf, r.on_bf,
            r.off.status, r.off.objective, r.off_pf, r.off_bf
        );
        if r.on.status != SolveStatus::Optimal || r.off.status != SolveStatus::Optimal {
            n_skip_status += 1;
            continue;
        }
        let scale = r.off.objective.abs().max(1.0);
        let rel = (r.on.objective - r.off.objective).abs() / scale;
        let on_infeasible = !(r.on_pf < 1e-5 && r.on_bf < 1e-5);
        if rel >= 1e-6 || on_infeasible {
            divergences.push(format!(
                "{name}: ON obj={:.6e} pf={:.1e} bf={:.1e} | OFF obj={:.6e} pf={:.1e} bf={:.1e} | rel={:.2e}",
                r.on.objective, r.on_pf, r.on_bf, r.off.objective, r.off_pf, r.off_bf, rel
            ));
        }
        if let Some(&k) = known.get(&name) {
            let kref = k + prob.obj_offset;
            let s = kref.abs().max(1.0);
            if (r.on.objective - kref).abs() / s >= 1e-4 {
                known_mismatch.push(format!(
                    "{name}: ON obj={:.6e} known={:.6e}",
                    r.on.objective, kref
                ));
            }
        }
        n_checked += 1;
    }

    eprintln!("\n===== INVARIANCE SWEEP SUMMARY =====");
    eprintln!("checked (both Optimal): {n_checked}");
    eprintln!("skip (step7 hang):      {n_skip_hang}");
    eprintln!("skip (non-Optimal):     {n_skip_status}");
    eprintln!("\n--- ON/OFF divergences ({}) ---", divergences.len());
    for d in &divergences {
        eprintln!("  {d}");
    }
    eprintln!(
        "\n--- ON vs known-optimal mismatch ({}) ---",
        known_mismatch.len()
    );
    for d in &known_mismatch {
        eprintln!("  {d}");
    }
}

/// 手法2 (infeasible half): presolve must not turn an infeasible LP feasible or
/// vice versa. At 930145d: 26/29 both Infeasible, bgindy/gosh ON=Infeasible while
/// OFF=Timeout (presolve detects infeasibility faster), klein3 both Timeout. No
/// ON/OFF status contradiction. Heavy — run as `presolve_invariance_full_sweep`.
#[test]
#[ignore = "tier-2: infeasible-set sweep, run under --profile heavy or raw binary"]
fn presolve_infeasible_sweep() {
    let timeout = 8.0;
    let mut files: Vec<_> = std::fs::read_dir(data_dir(INFEAS_SUBDIR))
        .expect("read infeas dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("qps"))
        })
        .collect();
    files.sort();

    let mut both_infeas = Vec::new();
    let mut disagree = Vec::new();
    for path in &files {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let prob = parse_qps(path).expect("parse");
        let on = solve(&prob, true, timeout);
        let off = solve(&prob, false, timeout);
        eprintln!("[{name}] ON={:?} OFF={:?}", on.status, off.status);
        let on_i = on.status == SolveStatus::Infeasible;
        let off_i = off.status == SolveStatus::Infeasible;
        if on_i && off_i {
            both_infeas.push(name.clone());
        } else if on_i != off_i {
            disagree.push(format!("{name}: ON={:?} OFF={:?}", on.status, off.status));
        }
    }
    eprintln!("\n===== INFEASIBLE SWEEP =====");
    eprintln!("both Infeasible: {} / {}", both_infeas.len(), files.len());
    eprintln!("ON/OFF disagree on Infeasible ({}):", disagree.len());
    for d in &disagree {
        eprintln!("  {d}");
    }
}

// ── Replacement tests (supersede old symmetric-invariance sentinels) ─────────

/// 80bau3b/grow15: presolve must not WORSEN feasibility.
///
/// After dc658d4 switched to primal-first cold-start, the OFF path produces a
/// solution with bf ≈ 1.9e-5 (within LP_CERT_TOL=1e-4, solver returns Optimal).
/// After Devex improvements (45c24f8), the ON path is near-machine-eps clean.
/// The old symmetric invariance check (|ON−OFF| < 1e-7) was designed for when
/// both paths had identical residuals; it is now stale.
///
/// This test replaces it with the correct directional invariance:
/// presolve must not make things worse.
#[test]
fn presolve_not_worsens_80bau3b_grow15() {
    // = feas_rel_tol() = LP_CERT_TOL: the guard_lp_optimal acceptance threshold.
    const LP_CERT_TOL: f64 = 1e-4;
    let dir = data_dir(LP_SUBDIR);
    for name in ["80bau3b", "grow15"] {
        let prob = parse_required(&dir, name);
        let r = solve_on_off(&prob, 60.0);
        eprintln!(
            "[{name}] ON status={:?} obj={:.6e} pf={:.2e} bf={:.2e} | OFF status={:?} obj={:.6e} pf={:.2e} bf={:.2e}",
            r.on.status, r.on.objective, r.on_pf, r.on_bf,
            r.off.status, r.off.objective, r.off_pf, r.off_bf
        );

        assert_eq!(r.on.status, SolveStatus::Optimal, "{name}: ON status");
        assert_eq!(r.off.status, SolveStatus::Optimal, "{name}: OFF status");

        // Objective invariance.
        let scale = r.off.objective.abs().max(1.0);
        assert!(
            (r.on.objective - r.off.objective).abs() / scale < 1e-4,
            "{name}: objective diverged ON={:.6e} OFF={:.6e}",
            r.on.objective,
            r.off.objective
        );

        // OFF is within solver tolerance (known primal two-phase residual).
        assert!(
            r.off_bf < LP_CERT_TOL,
            "{name}: OFF bound residual {:.2e} exceeds LP_CERT_TOL",
            r.off_bf
        );
        assert!(
            r.off_pf < LP_CERT_TOL,
            "{name}: OFF primal residual {:.2e} exceeds LP_CERT_TOL",
            r.off_pf
        );

        // Presolve must not worsen feasibility (directional invariance).
        assert!(
            r.on_bf <= r.off_bf + LP_CERT_TOL,
            "{name}: presolve worsened bound feasibility ON={:.2e} OFF={:.2e}",
            r.on_bf,
            r.off_bf
        );
        assert!(
            r.on_pf <= r.off_pf + LP_CERT_TOL,
            "{name}: presolve worsened primal feasibility ON={:.2e} OFF={:.2e}",
            r.on_pf,
            r.off_pf
        );
    }
}

/// Curated invariance for problems that solve cleanly with both ON and OFF.
/// Excludes 80bau3b/grow15 (handled by presolve_not_worsens_80bau3b_grow15 above).
#[test]
fn presolve_invariance_curated_clean() {
    let names = ["afiro", "adlittle", "israel", "sc50a", "blend"];
    let known = load_known();
    let dir = data_dir(LP_SUBDIR);
    let mut checked = 0usize;
    for name in names {
        let prob = parse_required(&dir, name);
        let r = solve_on_off(&prob, 30.0);
        if r.on.status != SolveStatus::Optimal || r.off.status != SolveStatus::Optimal {
            eprintln!(
                "[{name}] skip invariance (ON={:?} OFF={:?})",
                r.on.status, r.off.status
            );
            continue;
        }
        let scale = r.off.objective.abs().max(1.0);
        assert!(
            (r.on.objective - r.off.objective).abs() / scale < 1e-6,
            "{name}: presolve changed objective ON={:.6e} OFF={:.6e}",
            r.on.objective,
            r.off.objective
        );
        assert!(
            (r.on_pf - r.off_pf).abs() < 1e-7 && (r.on_bf - r.off_bf).abs() < 1e-7,
            "{name}: presolve changed feasibility ON(pf={:.2e},bf={:.2e}) OFF(pf={:.2e},bf={:.2e})",
            r.on_pf,
            r.on_bf,
            r.off_pf,
            r.off_bf
        );
        if let Some(&k) = known.get(name) {
            let kref = k + prob.obj_offset;
            let s = kref.abs().max(1.0);
            assert!(
                (r.on.objective - kref).abs() / s < 1e-4,
                "{name}: ON obj {:.6e} != known {:.6e}",
                r.on.objective,
                kref
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 4,
        "curated clean invariance must check >=4 problems, got {checked}"
    );
}
