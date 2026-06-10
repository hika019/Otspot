//! Diagnostic: attribute QP postsolve wall time to the 5 stage buckets.
//! Env-gated (OTSPOT_DIAG_LISWET=1) so it never runs in normal CI.
//! Timeout via OTSPOT_DIAG_TIMEOUT (secs, default 120).

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::qp::solve_qp_with;

fn run(name: &str, timeout: f64) {
    let path = std::path::PathBuf::from(format!("data/maros_meszaros/{}.QPS", name));
    assert!(path.exists(), "{:?} not found", path);
    let prob = parse_qps(&path).expect("parse");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(timeout);
    let t0 = std::time::Instant::now();
    let res = solve_qp_with(&prob, &opts);
    let wall = t0.elapsed().as_secs_f64();
    let tb = res.timing_breakdown.expect("timing");
    let ax = prob
        .a
        .mat_vec_mul(&res.solution)
        .unwrap_or_else(|_| vec![0.0; prob.num_constraints]);
    let mut maxviol = 0.0_f64;
    for (i, ct) in prob.constraint_types.iter().enumerate() {
        let raw = ax[i] - prob.b[i];
        let viol = match ct {
            otspot::problem::ConstraintType::Le => raw.max(0.0),
            otspot::problem::ConstraintType::Ge => (-raw).max(0.0),
            otspot::problem::ConstraintType::Eq => raw.abs(),
            _ => 0.0,
        };
        maxviol = maxviol.max(viol);
    }
    for (x, &(lb, ub)) in res.solution.iter().zip(prob.bounds.iter()) {
        if lb.is_finite() {
            maxviol = maxviol.max((lb - x).max(0.0));
        }
        if ub.is_finite() {
            maxviol = maxviol.max((x - ub).max(0.0));
        }
    }
    eprintln!(
        "\n===== {} (n={} m={}) wall={:.1}s status={:?} iters={} obj={:.12e} maxviol={:.3e} =====",
        name,
        prob.num_vars,
        prob.num_constraints,
        wall,
        res.status,
        res.iterations,
        res.objective,
        maxviol
    );
    let us = |x: u64| x as f64 / 1e6;
    eprintln!("ipm_factorize = {:.2}s", us(tb.ipm_factorize_us));
    eprintln!("ipm_solve     = {:.2}s", us(tb.ipm_solve_us));
    eprintln!("postsolve_map      = {:.2}s", us(tb.postsolve_map_us));
    eprintln!(
        "postsolve_lsq      = {:.2}s (refine_postsolve_dual_lsq)",
        us(tb.postsolve_lsq_us)
    );
    eprintln!(
        "postsolve_recovery = {:.2}s (refine_postsolve_recovery)",
        us(tb.postsolve_recovery_us)
    );
    eprintln!(
        "postsolve_refine   = {:.2}s (refine_post_processing stage1+2)",
        us(tb.postsolve_refine_us)
    );
    eprintln!(
        "postsolve_krylov   = {:.2}s (refine_krylov_and_projection)",
        us(tb.postsolve_krylov_ir_us)
    );
    eprintln!("postsolve_total    = {:.2}s", us(tb.postsolve_us));
}

#[test]
#[ignore = "diag: env-gated OTSPOT_DIAG_LISWET=1"]
fn diag_qp_postsolve_timing() {
    if std::env::var("OTSPOT_DIAG_LISWET").is_err() {
        eprintln!("skip: set OTSPOT_DIAG_LISWET=1");
        return;
    }
    let timeout: f64 = std::env::var("OTSPOT_DIAG_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120.0);
    let probs = std::env::var("OTSPOT_DIAG_PROBS").unwrap_or_else(|_| "LISWET12".to_string());
    for name in probs.split(',') {
        run(name.trim(), timeout);
    }
}
