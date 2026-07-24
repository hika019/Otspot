//! Adapter layer from legacy `otspot-core` types to the redesigned crates.
//!
//! This module is intentionally thin: algorithms must not be reimplemented
//! here.  It exists so migration can proceed one solver at a time while all
//! existing public APIs remain operational.

use otspot_ir::{
    Cone, ConicSystem, ConstraintSystem, Objective, OptimizationProblem, QuadraticConstraint,
    Variable, VariableKind,
};
use otspot_num::NumericError;

use crate::conic::{qcqp_matrix_to_csc, ConicProblem, MisocpProblem};
use crate::mip::{MilpProblem, MiqpProblem};
use crate::problem::{ConstraintType, LpProblem};
use crate::qp::QpProblem;
use crate::sparse::CscMatrix;

fn row_bounds(rhs: &[f64], constraint_types: &[ConstraintType]) -> (Vec<f64>, Vec<f64>) {
    let mut lower = Vec::with_capacity(rhs.len());
    let mut upper = Vec::with_capacity(rhs.len());
    for (&b, kind) in rhs.iter().zip(constraint_types) {
        match kind {
            ConstraintType::Le => {
                lower.push(f64::NEG_INFINITY);
                upper.push(b);
            }
            ConstraintType::Ge => {
                lower.push(b);
                upper.push(f64::INFINITY);
            }
            ConstraintType::Eq => {
                lower.push(b);
                upper.push(b);
            }
        }
    }
    (lower, upper)
}

fn continuous_variables(bounds: &[(f64, f64)]) -> Vec<Variable> {
    bounds
        .iter()
        .map(|&(lower, upper)| Variable::continuous(lower, upper))
        .collect()
}

/// Convert a legacy LP into the canonical IR without changing coefficients.
pub fn lp_to_ir(problem: &LpProblem) -> OptimizationProblem<CscMatrix> {
    let (lower, upper) = row_bounds(&problem.b, &problem.constraint_types);
    OptimizationProblem {
        variables: continuous_variables(&problem.bounds),
        objective: Objective {
            quadratic: None,
            linear: problem.c.clone(),
            offset: problem.obj_offset,
        },
        constraints: ConstraintSystem {
            matrix: (*problem.a).clone(),
            lower,
            upper,
        },
        quadratic_constraints: Vec::new(),
        conic: None,
    }
}

/// Convert a legacy QP/QCQP into the canonical IR.
pub fn qp_to_ir(problem: &QpProblem) -> OptimizationProblem<CscMatrix> {
    let (lower, upper) = row_bounds(&problem.b, &problem.constraint_types);
    let quadratic_constraints = if problem.quadratic_constraints.is_empty() {
        Vec::new()
    } else {
        problem
            .quadratic_constraints
            .iter()
            .enumerate()
            .map(|(row, quadratic)| QuadraticConstraint {
                quadratic: qcqp_matrix_to_csc(quadratic),
                linear_row: row,
            })
            .collect()
    };

    OptimizationProblem {
        variables: continuous_variables(&problem.bounds),
        objective: Objective {
            quadratic: (!problem.is_zero_q()).then(|| problem.q.clone()),
            linear: problem.c.clone(),
            offset: problem.obj_offset,
        },
        constraints: ConstraintSystem {
            matrix: problem.a.clone(),
            lower,
            upper,
        },
        quadratic_constraints,
        conic: None,
    }
}

/// Mark variables as integer/binary in an already converted canonical problem.
pub fn apply_integrality(
    problem: &mut OptimizationProblem<CscMatrix>,
    integer_indices: &[usize],
) -> Result<(), NumericError> {
    let bound = problem.variables.len();
    for &index in integer_indices {
        let variable = problem
            .variables
            .get_mut(index)
            .ok_or(NumericError::IndexOutOfBounds {
                context: "integer variable",
                index,
                bound,
            })?;
        variable.kind = if variable.lower >= 0.0 && variable.upper <= 1.0 {
            VariableKind::Binary
        } else {
            VariableKind::Integer
        };
    }
    Ok(())
}

pub fn milp_to_ir(problem: &MilpProblem) -> Result<OptimizationProblem<CscMatrix>, NumericError> {
    let mut ir = lp_to_ir(&problem.lp);
    apply_integrality(&mut ir, &problem.integer_vars)?;
    Ok(ir)
}

pub fn miqp_to_ir(problem: &MiqpProblem) -> Result<OptimizationProblem<CscMatrix>, NumericError> {
    let mut ir = qp_to_ir(&problem.qp);
    apply_integrality(&mut ir, &problem.integer_vars)?;
    Ok(ir)
}

pub fn conic_to_ir(problem: &ConicProblem) -> OptimizationProblem<CscMatrix> {
    let cones = std::iter::once(Cone::Nonnegative(problem.cone.l))
        .filter(|cone| cone.dimension() > 0)
        .chain(problem.cone.soc.iter().copied().map(Cone::SecondOrder))
        .collect();
    OptimizationProblem {
        variables: vec![Variable::continuous(f64::NEG_INFINITY, f64::INFINITY); problem.n()],
        objective: Objective {
            quadratic: None,
            linear: problem.c.clone(),
            offset: 0.0,
        },
        constraints: ConstraintSystem {
            matrix: problem.a.clone(),
            lower: problem.b.clone(),
            upper: problem.b.clone(),
        },
        quadratic_constraints: Vec::new(),
        conic: (!problem.h.is_empty()).then(|| ConicSystem {
            matrix: problem.g.clone(),
            rhs: problem.h.clone(),
            cones,
        }),
    }
}

pub fn misocp_to_ir(
    problem: &MisocpProblem,
) -> Result<OptimizationProblem<CscMatrix>, NumericError> {
    if problem.integers.len() != problem.int_lb.len() {
        return Err(NumericError::DimensionMismatch {
            field: "misocp.int_lb",
            expected: problem.integers.len(),
            got: problem.int_lb.len(),
        });
    }
    if problem.integers.len() != problem.int_ub.len() {
        return Err(NumericError::DimensionMismatch {
            field: "misocp.int_ub",
            expected: problem.integers.len(),
            got: problem.int_ub.len(),
        });
    }
    let mut ir = conic_to_ir(&problem.base);
    for ((&index, &lower), &upper) in problem
        .integers
        .iter()
        .zip(&problem.int_lb)
        .zip(&problem.int_ub)
    {
        let bound = ir.variables.len();
        let variable = ir
            .variables
            .get_mut(index)
            .ok_or(NumericError::IndexOutOfBounds {
                context: "misocp integer variable",
                index,
                bound,
            })?;
        variable.lower = lower;
        variable.upper = upper;
    }
    apply_integrality(&mut ir, &problem.integers)?;
    Ok(ir)
}

#[cfg(test)]
mod tests {
    use otspot_ir::{Cone, ProblemClass};

    use super::*;
    use crate::conic::ConeSpec;
    use crate::qp::QcqpMatrix;

    #[test]
    fn lp_adapter_preserves_rows_bounds_and_offset() {
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 2.0], 2, 1).unwrap();
        let mut lp = LpProblem::new_general(
            vec![3.0],
            a,
            vec![4.0, 5.0],
            vec![ConstraintType::Le, ConstraintType::Eq],
            vec![(-1.0, 2.0)],
            None,
        )
        .unwrap();
        lp.obj_offset = 7.0;

        let ir = lp_to_ir(&lp);
        ir.validate().unwrap();
        assert_eq!(ir.class(), ProblemClass::Lp);
        assert_eq!(ir.constraints.lower, vec![f64::NEG_INFINITY, 5.0]);
        assert_eq!(ir.constraints.upper, vec![4.0, 5.0]);
        assert_eq!(ir.objective.offset, 7.0);
    }

    #[test]
    fn qp_adapter_classifies_integrality_without_new_problem_types() {
        let q = CscMatrix::identity(2);
        let a = CscMatrix::new(0, 2);
        let qp = QpProblem::new(
            q,
            vec![0.0; 2],
            a,
            Vec::new(),
            vec![(0.0, 1.0), (-2.0, 2.0)],
            Vec::new(),
        )
        .unwrap();
        let mut ir = qp_to_ir(&qp);
        assert_eq!(ir.class(), ProblemClass::Qp);
        apply_integrality(&mut ir, &[0, 1]).unwrap();
        ir.validate().unwrap();
        assert_eq!(ir.variables[0].kind, VariableKind::Binary);
        assert_eq!(ir.variables[1].kind, VariableKind::Integer);
        assert_eq!(ir.class(), ProblemClass::Miqp);
    }

    #[test]
    fn qcqp_adapter_overlays_quadratic_term_on_existing_linear_row() {
        let q = CscMatrix::identity(1);
        let a = CscMatrix::identity(1);
        let mut qp = QpProblem::new(
            q,
            vec![0.0],
            a,
            vec![1.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            vec![ConstraintType::Le],
        )
        .unwrap();
        qp.quadratic_constraints = vec![QcqpMatrix {
            n: 1,
            triplets: vec![(0, 0, 2.0)],
        }];

        let ir = qp_to_ir(&qp);
        ir.validate().unwrap();
        assert_eq!(ir.class(), ProblemClass::Qcqp);
        assert_eq!(ir.constraints.matrix.nrows(), 1);
        assert_eq!(ir.quadratic_constraints[0].linear_row, 0);
    }

    #[test]
    fn conic_and_misocp_adapters_share_the_same_problem_type() {
        let base = ConicProblem {
            c: vec![1.0, 0.0],
            a: CscMatrix::new(0, 2),
            b: Vec::new(),
            g: CscMatrix::identity(2),
            h: vec![1.0, 0.0],
            cone: ConeSpec { l: 0, soc: vec![2] },
        };
        let continuous = conic_to_ir(&base);
        continuous.validate().unwrap();
        assert_eq!(continuous.class(), ProblemClass::Socp);
        assert_eq!(
            continuous.conic.as_ref().unwrap().cones,
            vec![Cone::SecondOrder(2)]
        );

        let mixed = MisocpProblem {
            base,
            integers: vec![0],
            int_lb: vec![0.0],
            int_ub: vec![1.0],
        };
        let mixed_ir = misocp_to_ir(&mixed).unwrap();
        mixed_ir.validate().unwrap();
        assert_eq!(mixed_ir.class(), ProblemClass::Misocp);
        assert_eq!(mixed_ir.variables[0].kind, VariableKind::Binary);
    }
}
