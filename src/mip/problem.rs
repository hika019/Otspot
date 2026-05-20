//! MILP / MIQP problem definitions (#14).
//!
//! A MILP/MIQP is a continuous LP/QP **relaxation** plus a set of variables
//! constrained to integer values. We deliberately wrap the existing
//! [`LpProblem`] / `QpProblem` rather than adding an integrality field to them:
//! the continuous solvers stay untouched (zero regression surface), and each
//! branch-and-bound node solves the relaxation by swapping `bounds` — the same
//! mechanism the spatial QP B&B already uses for box subproblems.

use crate::problem::LpProblem;

/// Construction error for [`MilpProblem`].
#[non_exhaustive]
#[derive(Debug, PartialEq, Eq)]
pub enum MipProblemError {
    /// An integer-variable index is out of range for the relaxation.
    InvalidIntegerVar { index: usize, num_vars: usize },
}

impl std::fmt::Display for MipProblemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MipProblemError::InvalidIntegerVar { index, num_vars } => write!(
                f,
                "integer variable index {} out of range (num_vars = {})",
                index, num_vars
            ),
        }
    }
}

impl std::error::Error for MipProblemError {}

/// Mixed-Integer Linear Program: minimize `c^T x` over the [`LpProblem`]
/// feasible region with `x[j]` integral for every `j` in `integer_vars`.
#[derive(Debug, Clone)]
pub struct MilpProblem {
    /// Continuous LP relaxation (objective, constraints, bounds).
    pub lp: LpProblem,
    /// Sorted, de-duplicated indices of variables required to be integral.
    pub integer_vars: Vec<usize>,
}

impl MilpProblem {
    /// Build a MILP from an LP relaxation and the integer-variable indices.
    ///
    /// `integer_vars` is sorted and de-duplicated. An empty set is permitted and
    /// means the problem is a plain LP (the solver falls back to the LP path).
    pub fn new(lp: LpProblem, integer_vars: Vec<usize>) -> Result<Self, MipProblemError> {
        let n = lp.num_vars;
        let mut iv = integer_vars;
        iv.sort_unstable();
        iv.dedup();
        if let Some(&j) = iv.last() {
            if j >= n {
                return Err(MipProblemError::InvalidIntegerVar { index: j, num_vars: n });
            }
        }
        Ok(Self { lp, integer_vars: iv })
    }

    /// Number of decision variables.
    pub fn num_vars(&self) -> usize {
        self.lp.num_vars
    }

    /// Boolean mask of length `num_vars`; `true` where the variable is integral.
    pub(crate) fn integer_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; self.lp.num_vars];
        for &j in &self.integer_vars {
            mask[j] = true;
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    fn lp_2var() -> LpProblem {
        // trivial 2-var LP, bounds [0,5]^2, one <= constraint
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![5.0],
            vec![crate::problem::ConstraintType::Le],
            vec![(0.0, 5.0), (0.0, 5.0)],
            None,
        )
        .unwrap()
    }

    #[test]
    fn new_sorts_and_dedups_integer_vars() {
        let m = MilpProblem::new(lp_2var(), vec![1, 0, 1]).unwrap();
        assert_eq!(m.integer_vars, vec![0, 1]);
    }

    #[test]
    fn new_rejects_out_of_range_index() {
        let err = MilpProblem::new(lp_2var(), vec![0, 2]).unwrap_err();
        assert_eq!(err, MipProblemError::InvalidIntegerVar { index: 2, num_vars: 2 });
    }

    #[test]
    fn empty_integer_vars_allowed() {
        let m = MilpProblem::new(lp_2var(), vec![]).unwrap();
        assert!(m.integer_vars.is_empty());
    }

    #[test]
    fn integer_mask_marks_only_integer_vars() {
        let m = MilpProblem::new(lp_2var(), vec![1]).unwrap();
        assert_eq!(m.integer_mask(), vec![false, true]);
    }
}
