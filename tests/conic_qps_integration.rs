//! Integration: parse a real QPS file and solve the convex QP through the conic
//! (QCQP→SOCP) bridge, cross-checking the objective against the native QP solver.
//!
//! Data is gitignored; the test skips gracefully when the file is absent.

use otspot_core::conic::{solve_qp_problem_as_qcqp, ConicOptions};
use otspot_core::options::SolverOptions;
use otspot_core::problem::SolveStatus;
use otspot_core::qp::solve_qp_with;
use otspot_io::qps::parse_qps;
use std::path::Path;

fn try_load(name: &str) -> Option<otspot_core::qp::QpProblem> {
    let p = format!("data/maros_meszaros/{name}");
    let path = Path::new(&p);
    if !path.exists() {
        eprintln!("[conic-qps] skip: data missing: {p}");
        return None;
    }
    parse_qps(path).ok()
}

#[test]
fn qps_convex_qp_via_conic_bridge_matches_native_qp() {
    // Small convex Maros–Mészáros instances.
    for name in ["HS21.QPS", "QPTEST.QPS", "TAME.QPS"] {
        let Some(qp) = try_load(name) else {
            continue;
        };
        let native = solve_qp_with(&qp, &SolverOptions::default());
        if native.status != SolveStatus::Optimal {
            // Only cross-check instances the native solver proves optimal.
            continue;
        }
        let conic = solve_qp_problem_as_qcqp(&qp, &ConicOptions::default());
        assert_eq!(
            conic.status,
            SolveStatus::Optimal,
            "{name}: conic bridge status {:?}",
            conic.status
        );
        // `solve_qp_problem_as_qcqp` reports only 1/2 x^T Q x + c^T x;
        // the public QP route adds the parsed objective offset afterwards.
        let conic_objective = conic.objective + qp.obj_offset;
        let rel = (conic_objective - native.objective).abs() / (1.0 + native.objective.abs());
        assert!(
            rel < 1e-4,
            "{name}: conic {} vs native {} (rel {rel:.2e})",
            conic_objective,
            native.objective
        );
    }
}
