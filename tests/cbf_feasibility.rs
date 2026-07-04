//! Sanity check: solutions returned for real CBLIB instances satisfy the
//! original conic constraints, independently recomputed from `A`/`G`/`h`
//! rather than trusting the solver's reported slack/residuals.
//!
//! Data is gitignored; each test skips gracefully when its file is absent.

use otspot_core::conic::{solve_misocp, solve_socp, BbOptions, ConicOptions, ConicProblem};
use otspot_core::problem::SolveStatus;
use otspot_io::cbf::{parse_cbf, CbfProblem};
use std::path::{Path, PathBuf};

/// Convergence tolerance for the solves under test (bench standard eps).
const TOL: f64 = 1e-6;
/// Feasibility slack: solver converges to TOL, so constraint residuals up to
/// a small multiple of it are conforming (rounding/conditioning headroom).
const FEAS_SLACK: f64 = 10.0 * TOL;

fn cblib_path(name: &str) -> PathBuf {
    Path::new("data/cblib_socp").join(name)
}

/// Asserts `A x == b` and `h - G x in K`, with the slack recomputed
/// independently of the solver's reported `s`.
fn assert_conic_feasible(label: &str, problem: &ConicProblem, x: &[f64]) {
    let ax = problem.a.mat_vec_mul(x).expect("A x");
    for (i, (&axi, &bi)) in ax.iter().zip(problem.b.iter()).enumerate() {
        assert!(
            (axi - bi).abs() < FEAS_SLACK,
            "{label}: equality row {i} violated: (Ax)_i={axi}, b_i={bi}"
        );
    }

    let gx = problem.g.mat_vec_mul(x).expect("G x");
    let s: Vec<f64> = problem
        .h
        .iter()
        .zip(gx.iter())
        .map(|(&h, &g)| h - g)
        .collect();

    for (i, &si) in s[..problem.cone.l].iter().enumerate() {
        assert!(
            si >= -FEAS_SLACK,
            "{label}: orthant row {i} violated: s_i={si}"
        );
    }

    let mut cursor = problem.cone.l;
    for (block_idx, &dim) in problem.cone.soc.iter().enumerate() {
        let block = &s[cursor..cursor + dim];
        let t = block[0];
        let norm_u: f64 = block[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            t >= norm_u - FEAS_SLACK,
            "{label}: SOC block {block_idx} violated: t={t}, ||u||={norm_u}"
        );
        cursor += dim;
    }
}

/// Solves one CBLIB MISOCP to optimality and verifies the returned point
/// against the original constraints (conic feasibility + integrality).
fn check_misocp(file: &str) {
    let path = cblib_path(file);
    if !path.exists() {
        eprintln!("[cbf-feasibility] skip: data missing: {}", path.display());
        return;
    }
    let cbf = parse_cbf(&path).unwrap_or_else(|e| panic!("parse {file}: {e}"));
    let CbfProblem::Misocp { problem, .. } = cbf else {
        panic!("{file}: expected a MISOCP");
    };
    let opts = ConicOptions {
        tol: TOL,
        ..ConicOptions::default()
    };
    let bb = BbOptions::default();
    let res = solve_misocp(&problem, &opts, &bb);
    assert_eq!(
        res.status,
        SolveStatus::Optimal,
        "{file}: status {:?}",
        res.status
    );
    assert_conic_feasible(file, &problem.base, &res.x);

    for (k, &j) in problem.integers.iter().enumerate() {
        let v = res.x[j];
        assert!(
            (v - v.round()).abs() < FEAS_SLACK,
            "{file}: integer var {k} (x[{j}]={v}) not integral"
        );
        assert!(
            problem.int_lb[k] - FEAS_SLACK <= v && v <= problem.int_ub[k] + FEAS_SLACK,
            "{file}: integer var {k} (x[{j}]={v}) outside [{}, {}]",
            problem.int_lb[k],
            problem.int_ub[k]
        );
    }
}

#[test]
fn classical_20_0_solution_satisfies_original_constraints() {
    check_misocp("classical_20_0.cbf");
}

#[test]
fn classical_30_0_solution_satisfies_original_constraints() {
    check_misocp("classical_30_0.cbf");
}

/// Continuous `solve_socp` path on real data: the four continuous CBLIB
/// instances all exceed the test-time budget, so the root relaxation of
/// classical_20_0 stands in for the direct SOCP entry point.
#[test]
fn classical_20_0_root_relaxation_is_feasible() {
    let path = cblib_path("classical_20_0.cbf");
    if !path.exists() {
        eprintln!("[cbf-feasibility] skip: data missing: {}", path.display());
        return;
    }
    let cbf = parse_cbf(&path).expect("parse classical_20_0.cbf");
    let CbfProblem::Misocp { problem, .. } = cbf else {
        panic!("classical_20_0.cbf: expected a MISOCP");
    };
    let opts = ConicOptions {
        tol: TOL,
        ..ConicOptions::default()
    };
    let res = solve_socp(&problem.base, &opts);
    assert_eq!(
        res.status,
        SolveStatus::Optimal,
        "classical_20_0 root: status {:?}",
        res.status
    );
    assert_conic_feasible("classical_20_0 root", &problem.base, &res.x);
}
