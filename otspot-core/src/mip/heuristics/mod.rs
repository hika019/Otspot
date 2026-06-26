pub(crate) mod feasibility_pump;
pub(crate) mod local_branching;
pub(crate) mod rens;
pub(crate) mod rins;

use crate::mip::{MilpProblem, MipConfig};
use crate::options::SolverOptions;
use crate::problem::SolverResult;

#[cfg(not(test))]
pub(crate) fn solve_sub_milp(
    problem: &MilpProblem,
    options: &SolverOptions,
    cfg: &MipConfig,
) -> SolverResult {
    crate::mip::solve_milp(problem, options, cfg)
}

#[cfg(test)]
thread_local! {
    static SUB_MIP_CONFIGS: std::cell::RefCell<Vec<MipConfig>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn solve_sub_milp(
    problem: &MilpProblem,
    options: &SolverOptions,
    cfg: &MipConfig,
) -> SolverResult {
    SUB_MIP_CONFIGS.with(|configs| configs.borrow_mut().push(cfg.clone()));
    crate::mip::solve_milp(problem, options, cfg)
}

#[cfg(test)]
pub(crate) fn clear_recorded_sub_mip_configs() {
    SUB_MIP_CONFIGS.with(|configs| configs.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn take_recorded_sub_mip_configs() -> Vec<MipConfig> {
    SUB_MIP_CONFIGS.with(|configs| std::mem::take(&mut *configs.borrow_mut()))
}
