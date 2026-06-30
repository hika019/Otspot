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
    use crate::linalg::parallelism::solver_par_from_threads;
    use crate::linalg::timeout::TimeoutCtx;
    use crate::qp::ipm_core::kkt::build_extended_constraints;
    let timeout_ctx = TimeoutCtx::from_options(options);
    let par = solver_par_from_threads(options.threads);
    let (a_ext, _, m_ext, _, _, _) = build_extended_constraints(problem);
    factorize::auto_schur_enabled(problem, &a_ext, m_ext, options, &timeout_ctx, par)
}

#[cfg(test)]
mod tests;
