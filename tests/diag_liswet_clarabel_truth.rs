//! LISWET-family 12 問の Clarabel 真値取得 diag (baseline 再記録用)。
//! Clarabel (tol=1e-12, max_iter=100k) を独立 reference として objective を出力。

use otspot::io::qps::parse_qps;
use clarabel::solver::{DefaultSolver, DefaultSettings, IPSolver, SolverStatus};

#[path = "helpers/clarabel_utils.rs"]
mod clarabel_helper;
use clarabel_helper::{build_clarabel, compute_internal_obj};

const STRICT_TOL: f64 = 1e-12;
const STRICT_MAX_ITER: u32 = 100_000;

/// Clarabel strict 解で LISWET-family 12 問の真値出力。
/// Clarabel が convergence しなかったら fail。
#[test]
#[ignore = "diag: Clarabel strict tol=1e-12 / max_iter=100k で LISWET-family 12 問の真値取得"]
fn diag_liswet_family_clarabel_truth() {
    let names = [
        "LISWET1", "LISWET2", "LISWET3", "LISWET4", "LISWET5", "LISWET6",
        "LISWET7", "LISWET8", "LISWET9", "LISWET10", "LISWET11", "LISWET12",
    ];

    let mut results: Vec<(String, String, f64, f64, u32)> = Vec::new();
    let mut failed: Vec<String> = Vec::new();

    for name in &names {
        let path = std::path::PathBuf::from(format!("data/maros_meszaros/{}.QPS", name));
        assert!(path.exists(), "{:?} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行", path);
        let prob = parse_qps(&path).expect("parse failed");
        let (p, q, a, b, cones) = build_clarabel(&prob);

        let mut settings = DefaultSettings::default();
        settings.verbose = false;
        settings.tol_gap_abs = STRICT_TOL;
        settings.tol_gap_rel = STRICT_TOL;
        settings.tol_feas = STRICT_TOL;
        settings.max_iter = STRICT_MAX_ITER;

        let mut solver = match DefaultSolver::new(&p, &q, &a, &b, &cones, settings) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[{}] Clarabel setup error: {:?}", name, e);
                failed.push(format!("{} (setup)", name));
                continue;
            }
        };
        solver.solve();

        let status = format!("{:?}", solver.info.status);
        let cost = solver.info.cost_primal;
        let iters = solver.info.iterations;
        let internal = compute_internal_obj(&prob, &solver.solution.x);

        eprintln!(
            "[{}] status={} iters={} cost_primal={:.10e} internal(via Q,c)={:.10e} obj_offset={:.6e}",
            name, status, iters, cost, internal, prob.obj_offset
        );

        results.push((name.to_string(), status.clone(), cost, internal, iters));

        if !matches!(
            solver.info.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ) {
            failed.push(format!("{} ({})", name, status));
        }
    }

    eprintln!("\n========= SUMMARY (LISWET-family Clarabel strict ground truth) =========");
    eprintln!(
        "{:10} {:18} {:>16} {:>16} {:>8}",
        "problem", "status", "cost_primal", "internal_obj", "iters"
    );
    for (name, status, cost, internal, iters) in &results {
        eprintln!(
            "{:10} {:18} {:16.6e} {:16.6e} {:>8}",
            name, status, cost, internal, iters
        );
    }

    assert!(
        failed.is_empty(),
        "Clarabel strict did not converge on: {:?}",
        failed
    );
}
