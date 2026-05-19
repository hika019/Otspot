mod api;
mod bound_duals;
mod concurrent;
mod dual_recovery;
mod dual_refit;
mod emptycol_skip;
mod pfeas;
mod postsolve;
mod presolve;
mod psd_nonconvex;
mod smoke;
mod status_dfeas;

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
