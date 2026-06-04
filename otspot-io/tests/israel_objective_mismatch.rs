//! Diagnostic tests for the israel OBJ_MISMATCH bug.
//!
//! Root cause (a4200da): the Big-M Phase I path in
//! `otspot-core/src/simplex/dual_advanced/phase1.rs` recomputes the objective
//! as `obj_orig = c·solution` from the UN-shifted solution (already the full
//! original objective) and then adds `sf.obj_offset` (= Σ c_j·lb_j) a second
//! time. The error is dormant when all lower bounds are zero (obj_offset = 0);
//! presolve bound-tightening raises some lower bounds, exposing it. The
//! returned *solution* is correct — only the reported `objective` is wrong.
//!
//! These tests use only the public API (parse_qps + solve_qp_with).

use otspot_core::options::SolverOptions;
use otspot_core::problem::{ConstraintType, SolveStatus};
use otspot_core::qp::{solve_qp_with, QpProblem};
use otspot_io::qps;
use std::path::Path;

struct Expect {
    name: &'static str,
    vars: usize,
    cons: usize,
    le: usize,
    eq: usize,
    known_opt: f64,
}

// Ground truth from the raw .QPS ROWS/COLUMNS sections. The parser converts
// G (>=) rows into Le by negation, so `le` counts L plus G rows.
const PROBLEMS: &[Expect] = &[
    Expect { name: "israel",   vars: 142, cons: 174, le: 174, eq: 0,  known_opt: -8.9664482186e+05 },
    Expect { name: "afiro",    vars: 32,  cons: 27,  le: 19,  eq: 8,  known_opt: -4.6475314286e+02 },
    Expect { name: "adlittle", vars: 97,  cons: 56,  le: 41,  eq: 15, known_opt:  2.2549496316e+05 },
];

fn parse(name: &str) -> Option<QpProblem> {
    let path = format!("data/lp_problems/{name}.QPS");
    let p = Path::new(&path);
    if !p.exists() {
        eprintln!("{name}.QPS not found, skipping");
        return None;
    }
    Some(qps::parse_qps(p).expect("parse"))
}

/// Parse fidelity: parsed model dimensions and constraint senses must match the
/// raw QPS files. (Parse is faithful for israel — the bug is downstream.)
#[test]
fn parse_fidelity_matches_raw_qps() {
    for e in PROBLEMS {
        let Some(prob) = parse(e.name) else { continue };
        let le = prob.constraint_types.iter().filter(|c| matches!(c, ConstraintType::Le)).count();
        let eq = prob.constraint_types.iter().filter(|c| matches!(c, ConstraintType::Eq)).count();
        assert_eq!(prob.num_vars, e.vars, "{}: num_vars", e.name);
        assert_eq!(prob.num_constraints, e.cons, "{}: num_constraints", e.name);
        assert_eq!(le, e.le, "{}: Le count", e.name);
        assert_eq!(eq, e.eq, "{}: Eq count", e.name);
    }
}

fn solve(prob: &QpProblem, presolve: bool) -> otspot_core::problem::SolverResult {
    let mut opts = SolverOptions::default();
    opts.presolve = presolve;
    opts.ipm.eps = 1e-6;
    opts.timeout_secs = Some(60.0);
    solve_qp_with(prob, &opts)
}

fn cx(prob: &QpProblem, sol: &[f64]) -> f64 {
    prob.c.iter().zip(sol).map(|(c, x)| c * x).sum::<f64>() + prob.obj_offset
}

/// Presolve must not change the optimal objective value.
///
/// `solve(presolve=true).objective == solve(presolve=false).objective`.
/// israel FAILS at a4200da: presolve=true reports -9.786e5 (vs correct
/// -8.966e5), an 82000 (9.1%) error, because presolve bound-tightening raises
/// 7 lower bounds and the Big-M path double-counts Σ c_j·lb_j. afiro/adlittle
/// are controls that should already pass.
/// Expected after the phase1.rs fix: all problems pass.
#[test]
fn presolve_objective_invariance() {
    for e in PROBLEMS {
        let Some(prob) = parse(e.name) else { continue };
        let on = solve(&prob, true);
        let off = solve(&prob, false);
        assert_eq!(on.status, SolveStatus::Optimal, "{}: presolve=on status", e.name);
        assert_eq!(off.status, SolveStatus::Optimal, "{}: presolve=off status", e.name);
        let scale = e.known_opt.abs().max(1.0);
        assert!(
            (on.objective - off.objective).abs() / scale < 1e-6,
            "{}: presolve must not change objective — on={:.6e} off={:.6e} (Δ={:.3e})",
            e.name,
            on.objective,
            off.objective,
            on.objective - off.objective
        );
        // The objective must also be consistent with the returned solution.
        assert!(
            (on.objective - cx(&prob, &on.solution)).abs() / scale < 1e-6,
            "{}: reported objective {:.6e} must equal c·x {:.6e} of the returned solution",
            e.name,
            on.objective,
            cx(&prob, &on.solution)
        );
    }
}
