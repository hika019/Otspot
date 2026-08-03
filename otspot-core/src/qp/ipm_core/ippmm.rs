//! IP-PMM (Pougkakiotis & Gondzio 2021, DOI 10.1007/s10589-020-00240-9)
//!
//! Augmented KKT (quasi-definite, upper-tri CSC):
//!   K = [(Q + ρI),  Aᵀ ]
//!       [A,        -D  ]   D = Σ + δI, Σ = diag(s/y)
//!
//! PMM update rule (Algorithm PEU §5.1.4):
//!   r = |μ_k − μ_{k+1}| / μ_k (実 μ)
//!   primal_improved = 0.95·prev_nr_p > nr_p  →  y_ref=y, δ *= (1−r),  else δ *= (1−r/3)
//!   dual_improved   = 0.95·prev_nr_d > nr_d  →  x_ref=x, ρ *= (1−r),  else ρ *= (1−r/3)

mod factorize;
mod init;
mod iter;
mod state;
mod warm_start;

pub(crate) use iter::solve_ippmm_inner;

pub(crate) fn probe_schur_decision(
    problem: &crate::qp::problem::QpProblem,
    options: &crate::options::SolverOptions,
) -> bool {
    use crate::qp::ipm_core::kkt::build_extended_constraints;
    use otspot_num::linalg::parallelism::with_solver_pool;
    use otspot_num::linalg::timeout::TimeoutCtx;
    let timeout_ctx = TimeoutCtx::new(
        options.deadline,
        options.timeout_secs,
        options.cancel_flag.clone(),
    );
    let (a_ext, _, m_ext, _, _, _) = build_extended_constraints(problem);
    // Same confinement as `solve_ippmm_inner`: this probe factorizes the KKT
    // system too, so it must respect the same thread budget. The `par` comes
    // from the confinement, never computed alongside it — see
    // `with_solver_pool`.
    with_solver_pool(options.threads, |par| {
        factorize::auto_schur_enabled(problem, &a_ext, m_ext, options, &timeout_ctx, par)
    })
}

#[cfg(test)]
mod tests;
