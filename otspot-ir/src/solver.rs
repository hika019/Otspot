//! Solver boundary.

use otspot_num::CscMatrixView;

use crate::{OptimizationProblem, SolveContext, SolveOutcome};

pub trait Solver<M: CscMatrixView> {
    type Options;

    fn solve(
        &self,
        problem: &OptimizationProblem<M>,
        options: &Self::Options,
        context: &SolveContext,
    ) -> SolveOutcome;
}
