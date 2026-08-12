//! LISWET7/11 OBJ_MISMATCH triage (bug-frontier 2026-08-02): determine whether the
//! ~50% objective gap against the baseline CSV is a baseline data error or a
//! genuine solver bug, using Clarabel (independent third-party solver) on the
//! exact same parsed `QpProblem` (same Q/c/A/b, so a QPS-parsing convention bug
//! would affect both solvers identically and NOT explain a Clarabel/ours gap).
#![allow(clippy::field_reassign_with_default)]

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::qp::solve_qp_with;
use otspot::{QpProblem, SolveStatus};

use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver};

#[path = "helpers/clarabel_utils.rs"]
mod clarabel_helper;
use clarabel_helper::build_clarabel;

fn solve_clarabel_tol(prob: &QpProblem, tol: f64, max_iter: u32) -> (f64, String) {
    let (p, q, a, b, cones) = build_clarabel(prob);
    let mut s = DefaultSettings::default();
    s.verbose = false;
    s.tol_gap_abs = tol;
    s.tol_gap_rel = tol;
    s.tol_feas = tol;
    s.max_iter = max_iter;
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, s).expect("clarabel setup");
    solver.solve();
    let obj_internal = {
        let x = &solver.solution.x;
        let qx = prob.q.mat_vec_mul(x).expect("Qx");
        0.5 * qx.iter().zip(x.iter()).map(|(&q, &x)| q * x).sum::<f64>()
            + prob
                .c
                .iter()
                .zip(x.iter())
                .map(|(&c, &x)| c * x)
                .sum::<f64>()
    };
    (obj_internal, format!("{:?}", solver.info.status))
}

fn check_obj_agrees_with_clarabel(name: &str, baseline_known: f64) {
    let path = std::path::PathBuf::from(format!("data/maros_meszaros/{}.QPS", name));
    assert!(path.exists(), "{:?} not found", path);
    let prob = parse_qps(&path).expect("parse");

    let (clarabel_obj, clarabel_status) = solve_clarabel_tol(&prob, 1e-12, 100_000);

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);
    let ours = solve_qp_with(&prob, &opts);

    eprintln!(
        "\n===== {name} =====\nours:     status={:?} obj={:.10e}\nclarabel: status={clarabel_status} obj={:.10e}\nbaseline_csv: {baseline_known:.10e}\nours/clarabel ratio: {:.6}\nclarabel/baseline ratio: {:.6}",
        ours.status, ours.objective, clarabel_obj,
        ours.objective / clarabel_obj,
        clarabel_obj / baseline_known,
    );

    assert_eq!(
        ours.status,
        SolveStatus::Optimal,
        "{name}: ours must report Optimal, got {:?} (obj={:.10e})",
        ours.status,
        ours.objective,
    );

    // Load-bearing assertion: ours and an independent third-party solver (same
    // parsed Q/c/A/b, so no QPS-convention confound) must agree to within 1%.
    // This is the fact this test exists to establish either way.
    let rel_diff = (ours.objective - clarabel_obj).abs() / clarabel_obj.abs().max(1.0);
    eprintln!("ours vs clarabel rel_diff = {rel_diff:.6e}");
    assert!(
        rel_diff < 1e-2,
        "{name}: ours (obj={:.10e}) vs clarabel (status={clarabel_status}, obj={:.10e}) \
         rel_diff={rel_diff:.6e} exceeds 1% tolerance",
        ours.objective,
        clarabel_obj,
    );
}

#[test]
fn liswet7_obj_independent_oracle() {
    // baseline CSV claims -5.01e3ish for LISWET7.
    check_obj_agrees_with_clarabel("LISWET7", -5.01e3);
}

#[test]
fn liswet11_obj_independent_oracle() {
    check_obj_agrees_with_clarabel("LISWET11", -5.02e3);
}
