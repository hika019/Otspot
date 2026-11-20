//! Presolve correctness verification.
//!
//! Layers:
//!  * `presolve_not_worsens_80bau3b_grow15` — directional invariance: presolve
//!    must not worsen feasibility (ON ≤ OFF + LP_CERT_TOL). OFF residual is a
//!    known primal two-phase numerical artifact (dc658d4, within LP_CERT_TOL=1e-4).
//!  * `presolve_invariance_curated_clean_known_with_constant` — symmetric invariance for clean problems.
//!  * `presolve_invariance_full_sweep_known_with_constant` — all 109 netlib LPs, `#[ignore]` (tier-2).
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
///     --run-ignored all presolve_invariance_full_sweep_known_with_constant --no-capture
///   PRESOLVE_SWEEP_TIMEOUT=5 ./target/release/deps/presolve_correctness_sweep-* \
///     --ignored --nocapture --test-threads 1 presolve_invariance_full_sweep_known_with_constant
#[test]
#[ignore = "tier-2: full 109 sweep, ~5min — run under --profile heavy or raw binary"]
fn presolve_invariance_full_sweep_known_with_constant() {
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
            r.on.status,
            r.on.objective,
            r.on_pf,
            r.on_bf,
            r.off.status,
            r.off.objective,
            r.off_pf,
            r.off_bf
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
            let s = k.abs().max(1.0);
            if (r.on.objective - k).abs() / s >= 1e-4 {
                known_mismatch.push(format!(
                    "{name}: ON obj={:.6e} known={:.6e}",
                    r.on.objective, k
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
/// ON/OFF status contradiction. Heavy — run as `presolve_invariance_full_sweep_known_with_constant`.
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
            r.on.status,
            r.on.objective,
            r.on_pf,
            r.on_bf,
            r.off.status,
            r.off.objective,
            r.off_pf,
            r.off_bf
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
fn presolve_invariance_curated_clean_known_with_constant() {
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
            let s = k.abs().max(1.0);
            assert!(
                (r.on.objective - k).abs() / s < 1e-4,
                "{name}: ON obj {:.6e} != known {:.6e}",
                r.on.objective,
                k
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 4,
        "curated clean invariance must check >=4 problems, got {checked}"
    );
}

// ---------------------------------------------------------------------------
// Step 12 (dependent Eq rows) family.
// ---------------------------------------------------------------------------

use otspot_core::presolve::{run_presolve_with_flags, PresolveFlags};
use otspot_core::problem::LpProblem;

/// QpProblem (Q = 0 only) -> LpProblem, mirroring `qp/lp_dispatch.rs`.
/// Returns `None` for any nonzero quadratic part (never happens for the LP
/// suites; the guard keeps the test honest if one ever lands in the dir).
fn lp_from_qp(prob: &QpProblem) -> Option<LpProblem> {
    if prob.q.values().iter().any(|v| *v != 0.0) || !prob.quadratic_constraints.is_empty() {
        return None;
    }
    LpProblem::new_general(
        prob.c.clone(),
        prob.a.clone(),
        prob.b.clone(),
        prob.constraint_types.clone(),
        prob.bounds.clone(),
        None,
    )
    .ok()
}

/// Reduced row counts with step12 disabled vs enabled: `(off, on)`.
/// `on < off` is direct evidence that rank discovery removed at least one
/// linearly dependent Eq row on this instance.
fn dependent_eq_row_delta(prob: &QpProblem) -> Option<(usize, usize)> {
    let lp = lp_from_qp(prob)?;
    let max_pass = SolverOptions::default().presolve_max_pass;
    let on = run_presolve_with_flags(&lp, None, max_pass, None, PresolveFlags::default()).ok()?;
    let off = run_presolve_with_flags(
        &lp,
        None,
        max_pass,
        None,
        PresolveFlags {
            enable_dependent_eq: false,
            ..PresolveFlags::default()
        },
    )
    .ok()?;
    Some((
        off.reduced_problem.num_constraints,
        on.reduced_problem.num_constraints,
    ))
}

/// Detection-sweep instances whose solve budget is genuinely near the limit:
/// their FULL deterministic solve (simplex iterations + postsolve dual
/// certificate) straddles `PRESOLVE_SWEEP_TIMEOUT`, so their outcome is drawn
/// from the budget-cut family, and which member a run lands on is a function
/// of runner speed/load only — the arithmetic is run-invariant. Measured on
/// this box for degen3 with the sweep's own solve() (budget ladder probe,
/// 2026-11-20): 8/8.5/9s → ON=Timeout (iterations cut at 17520/18189/18830,
/// before the optimum basis); 9.5/10/10.5s → ON=SuboptimalSolution (all
/// 19141 iterations run, objective = optimum, primal feasible, but the
/// deadline cuts the postsolve dual certificate: postsolve_dfeas ≈ 2.15
/// instead of 2.4e-14, so the result is honestly demoted per
/// `guard_lp_incumbent_claim` semantics); 60s → ON=Optimal, OFF=Optimal
/// (ON wall ≈ 11.0s, OFF wall ≈ 12.3s idle). CI heavy at b8a57857 landed ON
/// in the SuboptimalSolution member with OFF=Timeout at the 8s budget; the
/// green b5ed5f65 run differs from b8a57857 only by a log crate dependabot
/// bump, which establishes this as a budget borderline, not a code change.
/// For exactly these instances the sweep tolerates ANY pair drawn from the
/// budget-cut family {Optimal, SuboptimalSolution, Timeout}, not only
/// both-Timeout: a mixed Optimal/Timeout or SuboptimalSolution/Optimal pair
/// here is the same load-split artifact this const's doc always described
/// ("they split between Optimal and Timeout with machine load"), and the old
/// both-Timeout-only allowance under-implemented it (CI runs 44516245649 /
/// 44648487407 failed on such pairs). What remains checked even for
/// borderline instances: every PROVEN optimum (either side reporting
/// Optimal) is compared against the ON incumbent and the known Netlib
/// value, and ON feasibility is checked whenever ON returns a point. Any
/// other instance producing a non-Optimal outcome is a hard error. The
/// tolerated set is a named allowlist so the borderline intent is stated,
/// not hidden.
const DEP_EQ_BORDERLINE_BUDGET: &[&str] = &["cre-b", "cre-d", "degen3"];

/// Per-instance outcome of the step12 ON/OFF invariance decision.
enum DepEqOutcome {
    /// Full strict invariance held (both Optimal, all cross-checks passed).
    StrictOk,
    /// Allowlisted instance whose pair is a budget-cut family outcome with
    /// every proven value still matching.
    BorderlineTolerated,
    /// Contract violation; the string is the violation message.
    Violation(String),
}

/// Observed ON/OFF pair for one instance, as inputs to the pure decision.
#[derive(Clone, Copy)]
struct DepEqPairObs<'a> {
    on_status: &'a SolveStatus,
    off_status: &'a SolveStatus,
    on_obj: f64,
    off_obj: f64,
    on_pf: f64,
    on_bf: f64,
    known: Option<f64>,
}

/// Pure per-instance invariance decision, split out of the detection sweep so
/// the budget-borderline tolerance is testable without solving (sentinel
/// `dep_eq_pair_tolerates_budget_cut_family`).
///
/// Non-allowlisted instances keep the original strict contract:
/// ON == OFF == Optimal plus objective match, ON feasibility, and known
/// optimum match. Allowlisted (`DEP_EQ_BORDERLINE_BUDGET`) instances accept
/// any pair from the budget-cut family, but proven values still decide: an
/// Optimal side's objective is the oracle for the ON incumbent, and the
/// known Netlib optimum is checked against ON whenever ON returned a finite
/// point and at least one side proved Optimal. A Timeout side's objective is
/// +inf by construction (`Timeout` early-returns) and never participates in
/// a comparison.
fn classify_degen_eq_pair(borderline: bool, o: &DepEqPairObs<'_>) -> DepEqOutcome {
    let DepEqPairObs {
        on_status,
        off_status,
        on_obj,
        off_obj,
        on_pf,
        on_bf,
        known,
    } = *o;
    // Max primal/bound violation a returned incumbent may carry (identical to
    // the strict-branch threshold the sweep has always used for Optimal
    // solutions; a budget-cut SuboptimalSolution is guard-verified primal
    // feasible, so anything worse is not a budget artifact).
    const ON_FEAS_MAX: f64 = 1e-5;
    // Relative objective tolerance between ON and a PROVEN optimum (OFF or
    // the known Netlib constant), same value the strict branch has always
    // used for both-Optimal pairs.
    const PROVEN_OBJ_REL: f64 = 1e-6;
    // Relative tolerance of a proven objective vs the known Netlib optimum
    // (baseline-certificate contract value, unchanged from the strict branch).
    const KNOWN_OBJ_REL: f64 = 1e-4;
    let optimal = SolveStatus::Optimal;
    let timeout = SolveStatus::Timeout;

    if !borderline && (on_status != off_status || on_status != &optimal) {
        return DepEqOutcome::Violation(format!(
            "status ON={on_status:?} OFF={off_status:?} (require ON == OFF == Optimal \
             unless a documented budget-cut borderline pair)"
        ));
    }
    // ON must not present an infeasible point as an incumbent. A Timeout side
    // carries no solution (pf/bf are NaN by construction) and is not judged.
    if on_status != &timeout && (on_pf >= ON_FEAS_MAX || on_bf >= ON_FEAS_MAX) {
        return DepEqOutcome::Violation(format!(
            "ON solution infeasible pf={on_pf:.1e} bf={on_bf:.1e}"
        ));
    }
    // Compare the ON incumbent only against a PROVEN optimum (OFF == Optimal;
    // a Timeout objective is +inf and a SuboptimalSolution objective is just
    // another unproven incumbent — mid-trajectory incumbents legitimately
    // differ, see the const doc ladder where OFF incumbents ran -9.60e2…-9.83e2
    // while the optimum was -9.87294e2).
    if off_status == &optimal && on_obj.is_finite() {
        let rel = (on_obj - off_obj).abs() / off_obj.abs().max(1.0);
        if rel >= PROVEN_OBJ_REL {
            return DepEqOutcome::Violation(format!(
                "ON obj={on_obj:.6e} | OFF obj={off_obj:.6e} | rel={rel:.2e}"
            ));
        }
    }
    if (on_status == &optimal || off_status == &optimal) && on_obj.is_finite() {
        if let Some(k) = known {
            let s = k.abs().max(1.0);
            if (on_obj - k).abs() / s >= KNOWN_OBJ_REL {
                return DepEqOutcome::Violation(format!("ON obj={on_obj:.6e} known={k:.6e}"));
            }
        }
    }
    if borderline && (on_status != &optimal || off_status != &optimal) {
        DepEqOutcome::BorderlineTolerated
    } else {
        DepEqOutcome::StrictOk
    }
}

/// Sweep for step12's footprint: for every LP in `lp_problems`, compare the
/// presolve-reduced row count with `enable_dependent_eq` off vs on, list the
/// rank-deficient instances, then re-solve each detected instance ON vs OFF
/// and require strict invariance: the ON and OFF statuses must be EQUAL, and
/// for the both-Optimal pairs (every instance that reaches Optimal on either
/// side) both objectives, ON feasibility, and the known Netlib optimum must
/// match. The only exception is the `DEP_EQ_BORDERLINE_BUDGET` allowlist:
/// those instances' deterministic solve time itself straddles the budget, so
/// their outcomes come from the budget-cut family (Timeout when the deadline
/// cuts the iterations, SuboptimalSolution when it cuts the postsolve dual
/// certificate, Optimal when it finishes) and the split is load-dependent;
/// any pair from that family is tolerated for listed instances while every
/// PROVEN value (an Optimal side's objective vs the ON incumbent and the
/// known optimum, plus ON feasibility whenever ON returns a point) stays
/// checked. The observed borderline list is printed in the failure message so
/// CI logs show it. Any other instance producing a non-Optimal outcome is an
/// error: this sweep guards a rank-based row removal, so unlike the sibling
/// sweeps it takes no credit for presolve making a borderline instance time
/// out on instances that are NOT budget-borderline.
///
/// Revert-fail sentinel: the `assert!(!detected.is_empty())` at the end fails
/// the moment step12 stops removing rows (reverted or short-circuited),
/// because these Netlib instances genuinely contain dependent Eq rows.
///
/// Heavy (≈ presolve-only sweep over 109 files + solves of the detected few).
/// Run under the heavy profile or the raw binary, like `presolve_invariance_full_sweep_known_with_constant`.
#[test]
#[ignore = "tier-2: dependent-Eq detection sweep over lp_problems, run under --profile heavy or raw binary"]
fn presolve_dependent_eq_detection_sweep() {
    let timeout: f64 = std::env::var("PRESOLVE_SWEEP_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8.0);
    let known = load_known();

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

    let mut detected: Vec<(String, usize, usize)> = Vec::new();
    for path in &files {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        if STEP7_HANGERS.contains(&name.as_str()) {
            continue;
        }
        let prob = parse_qps(path).expect("parse");
        let Some((m_off, m_on)) = dependent_eq_row_delta(&prob) else {
            continue;
        };
        if m_on < m_off {
            eprintln!(
                "[{name}] rank-deficient: rows off={m_off} on={m_on} (-{})",
                m_off - m_on
            );
            detected.push((name, m_off, m_on));
        }
    }

    let mut errors: Vec<String> = Vec::new();
    let mut borderline: Vec<String> = Vec::new();
    for (name, m_off, m_on) in &detected {
        let prob = parse_required(&data_dir(LP_SUBDIR), name);
        let r = solve_on_off(&prob, timeout);
        eprintln!(
            "[{name}] ON={:?} obj={:.6e} pf={:.1e} bf={:.1e} | OFF={:?} obj={:.6e} pf={:.1e} bf={:.1e}",
            r.on.status,
            r.on.objective,
            r.on_pf,
            r.on_bf,
            r.off.status,
            r.off.objective,
            r.off_pf,
            r.off_bf,
        );
        let obs = DepEqPairObs {
            on_status: &r.on.status,
            off_status: &r.off.status,
            on_obj: r.on.objective,
            off_obj: r.off.objective,
            on_pf: r.on_pf,
            on_bf: r.on_bf,
            known: known.get(name.as_str()).copied(),
        };
        let outcome =
            classify_degen_eq_pair(DEP_EQ_BORDERLINE_BUDGET.contains(&name.as_str()), &obs);
        match outcome {
            DepEqOutcome::StrictOk => {}
            DepEqOutcome::BorderlineTolerated => borderline.push(name.clone()),
            DepEqOutcome::Violation(msg) => errors.push(format!("{name}: {msg}")),
        }
        let _ = (*m_off, *m_on);
    }

    eprintln!("detected rank-deficient instances: {}", detected.len());
    assert!(
        !detected.is_empty(),
        "step12 must detect dependent Eq rows on at least one lp_problems \
         instance; empty detection means the rank pass became a no-op"
    );
    assert!(
        errors.is_empty(),
        "step12-touched solves must be invariant (ON == OFF status; both-Optimal \
         objective match, feasible, match known; non-Optimal pairs only for \
         DEP_EQ_BORDERLINE_BUDGET instances). borderline both-Timeout \
         observed this run: [{}] | violations: {}",
        borderline.join(", "),
        errors.join("\n")
    );
}

/// Sentinel for the budget-cut tolerance in `classify_degen_eq_pair`
/// (reverting the tolerance to the old strict / both-Timeout-only logic
/// fails the `tolerated` assertions here; weakening the checks further —
/// e.g. comparing Timeout's +inf objective or skipping feasibility on a
/// SuboptimalSolution incumbent — fails the `violates` assertions).
///
/// The tolerated fixtures use the exact values observed in the two failing
/// CI runs for degen3 (known optimum -9.87294e2 from
/// `baseline_objectives/netlib_lp.csv`):
///  * run 44648487407 (b8a57857): ON=SuboptimalSolution obj=-9.872940e2
///    (pf=6.4e-14), OFF=Timeout obj=-9.725670e2 — mixed pair, old code
///    failed it although ON's incumbent IS the optimum;
///  * run 44516245649 (PR #87 merge): ON=SuboptimalSolution obj=-9.872940e2,
///    OFF=Optimal obj=-9.872940e2 — same story with OFF unbothered.
///
/// Both are budget-cut family outcomes proven by the budget-ladder probe
/// documented on `DEP_EQ_BORDERLINE_BUDGET` (identical iterations/objective,
/// only the postsolve dual certificate differs by where the deadline falls).
#[test]
fn dep_eq_pair_tolerates_budget_cut_family() {
    let opt = SolveStatus::Optimal;
    let sub = SolveStatus::SuboptimalSolution;
    let tout = SolveStatus::Timeout;
    let known = Some(-9.87294e2_f64);
    // Measured CI values (const doc above): optimum, mid-run OFF incumbent,
    // clean ON feasibility residuals from the guard-verified incumbent.
    let best = -9.872940e2_f64;
    let mid_incumbent = -9.725670e2_f64;
    let (pf_ok, bf_ok) = (6.4e-14_f64, 2.5e-14_f64);
    let (pf_bad, bf_bad) = (1.0e-3_f64, 0.0_f64);
    // NaN residuals as the real sweep computes them for a Timeout side.
    let nan = f64::NAN;

    let classified = |borderline: bool,
                      on_s: &SolveStatus,
                      off_s: &SolveStatus,
                      on_o: f64,
                      off_o: f64,
                      pf: f64,
                      bf: f64|
     -> bool {
        matches!(
            classify_degen_eq_pair(
                borderline,
                &DepEqPairObs {
                    on_status: on_s,
                    off_status: off_s,
                    on_obj: on_o,
                    off_obj: off_o,
                    on_pf: pf,
                    on_bf: bf,
                    known,
                }
            ),
            DepEqOutcome::Violation(_)
        )
    };

    // Tolerated: the exact b8a57857 failure pair on an allowlisted instance.
    assert!(!classified(
        true,
        &sub,
        &tout,
        best,
        mid_incumbent,
        pf_ok,
        bf_ok
    ));
    // Tolerated: the exact PR #87 failure pair (ON incumbent matches the
    // proven OFF optimum).
    assert!(!classified(true, &sub, &opt, best, best, pf_ok, bf_ok));
    // Tolerated: mirror split (ON finished, OFF cut) and both-Timeout (the
    // previously-only-allowed shape stays allowed).
    assert!(!classified(
        true,
        &opt,
        &tout,
        best,
        f64::INFINITY,
        pf_ok,
        bf_ok
    ));
    assert!(!classified(
        true,
        &tout,
        &tout,
        f64::INFINITY,
        mid_incumbent,
        nan,
        nan
    ));
    // Still strict-ok on genuine agreement, allowlisted or not.
    assert!(matches!(
        classify_degen_eq_pair(
            false,
            &DepEqPairObs {
                on_status: &opt,
                off_status: &opt,
                on_obj: best,
                off_obj: best,
                on_pf: pf_ok,
                on_bf: bf_ok,
                known,
            }
        ),
        DepEqOutcome::StrictOk
    ));

    // Violations must stay violations:
    // 1. Non-allowlisted instance with a budget-cut pair (the whole point of
    //    the allowlist: no credit for arbitrary instances).
    assert!(classified(
        false,
        &sub,
        &tout,
        best,
        mid_incumbent,
        pf_ok,
        bf_ok
    ));
    assert!(classified(
        false,
        &tout,
        &tout,
        f64::INFINITY,
        mid_incumbent,
        nan,
        nan
    ));
    // 2. Allowlisted, but the ON incumbent contradicts the proven optimum
    //    (a genuinely wrong point is never a budget artifact).
    assert!(classified(true, &sub, &opt, -9.5e2, best, pf_ok, bf_ok));
    // 3. Allowlisted, but the ON incumbent is infeasible (guard semantics:
    //    SuboptimalSolution must be a feasible point).
    assert!(classified(
        true,
        &sub,
        &tout,
        best,
        mid_incumbent,
        pf_bad,
        bf_bad
    ));
    // 4. Allowlisted, ON claims Optimal but disagrees with the known value.
    assert!(classified(
        true,
        &opt,
        &tout,
        -9.0e2,
        f64::INFINITY,
        pf_ok,
        bf_ok
    ));
}

/// Curated tier-1 subset of `presolve_dependent_eq_detection_sweep`: the
/// instances where the flag comparison measurably drops rows (populated from
/// the detection sweep output on this machine). Each is re-solved ON vs OFF;
/// objective agreement, feasibility including the removed dependent rows, and
/// the known-optimal constant are all required.
#[test]
fn presolve_dependent_eq_curated_known() {
    const CURATED: &[&str] = &[
        // Detected by `presolve_dependent_eq_detection_sweep` on this machine
        // (rows off->on): degen2 380->378, scorpion 206->183, pds-02
        // 2177->2166, sierra 997->987, ken-07 947->898. Small/fast enough
        // for the tier-1 budget; cre-*/ken-11 stay in the tier-2 sweep.
        "degen2", "scorpion", "pds-02", "sierra", "ken-07",
    ];
    assert!(!CURATED.is_empty());
    let known = load_known();
    let mut errors: Vec<String> = Vec::new();
    for name in CURATED {
        let prob = parse_required(&data_dir(LP_SUBDIR), name);
        let (m_off, m_on) = dependent_eq_row_delta(&prob)
            .unwrap_or_else(|| panic!("{name}: must be an LP convertible to LpProblem"));
        assert!(
            m_on < m_off,
            "{name}: curated rank-deficient instance must still lose rows to \
             step12 (off={m_off} on={m_on})"
        );
        let r = solve_on_off(&prob, 8.0);
        // Lenient status convention as in the tier-2 sweep: only ON being
        // worse than OFF is a presolve fault.
        if r.on.status != SolveStatus::Optimal {
            errors.push(format!(
                "{name}: status ON={:?} OFF={:?}",
                r.on.status, r.off.status
            ));
            continue;
        }
        if r.on_pf >= 1e-5 || r.on_bf >= 1e-5 {
            errors.push(format!(
                "{name}: ON solution infeasible pf={:.1e} bf={:.1e}",
                r.on_pf, r.on_bf
            ));
        }
        if r.off.status == SolveStatus::Optimal {
            let rel = (r.on.objective - r.off.objective).abs() / r.off.objective.abs().max(1.0);
            if rel >= 1e-6 {
                errors.push(format!(
                    "{name}: ON obj={:.6e} | OFF obj={:.6e} | rel={:.2e}",
                    r.on.objective, r.off.objective, rel
                ));
            }
        }
        if let Some(&k) = known.get(*name) {
            let s = k.abs().max(1.0);
            if (r.on.objective - k).abs() / s >= 1e-4 {
                errors.push(format!(
                    "{name}: ON obj={:.6e} known={:.6e}",
                    r.on.objective, k
                ));
            }
        }
    }
    assert!(
        errors.is_empty(),
        "curated dependent-Eq invariance broken: {errors:?}"
    );
}
