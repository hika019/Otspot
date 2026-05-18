//! Q=0 退化ケースを LP entry に転送する (#36)。
//!
//! 旧実装は `simplex::solve_with` を直接呼んでいたが、`crate::lp` を
//! 経由することで LP-specific 経路を全て LP module に集約し、
//! telemetry counter (`lp::telemetry::lp_forwarded_from_qp_calls`) で
//! QP→LP forward を識別できるようにする。

use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolverResult};

use super::QpProblem;

pub(crate) fn solve_as_lp_pub(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    let opts_with_deadline;
    let options: &SolverOptions = if options.deadline.is_none() {
        if let Some(secs) = options.timeout_secs {
            opts_with_deadline = {
                let mut o = options.clone();
                o.deadline = Some(std::time::Instant::now() + std::time::Duration::from_secs_f64(secs));
                o.timeout_secs = None;
                o
            };
            &opts_with_deadline
        } else {
            options
        }
    } else {
        options
    };

    let lp = match LpProblem::new_general(
        problem.c.clone(),
        problem.a.clone(),
        problem.b.clone(),
        problem.constraint_types.clone(),
        problem.bounds.clone(),
        None,
    ) {
        Ok(lp) => lp,
        Err(_) => return SolverResult::infeasible(),
    };

    let mut result = crate::lp::solve_lp_forwarded_from_qp(&lp, options);
    result.objective += problem.obj_offset;
    result
}
