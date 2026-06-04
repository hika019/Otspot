mod api;
mod bound_duals;
mod concurrent;
mod dispatch;
mod dual_recovery;
mod dual_refit;
mod emptycol_skip;
mod micro_q_dispatch;
mod pfeas;
mod postsolve;
mod presolve;
mod psd_nonconvex;
mod qcqp_guard;
mod smoke;
mod status_dfeas;
mod validate_wiring;

const EPS: f64 = 1e-2;

fn assert_close(a: f64, b: f64, eps: f64, name: &str) {
    assert!(
        (a - b).abs() < eps,
        "{}: expected {:.8}, got {:.8} (diff={:.2e})",
        name,
        b,
        a,
        (a - b).abs()
    );
}
