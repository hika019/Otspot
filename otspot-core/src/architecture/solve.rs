//! Canonical IR → legacy solver dispatch during the migration.

use otspot_ir::{
    Cone, OptimizationProblem, ProblemClass, SolveContext, SolveOutcome, SolveStatus as IrStatus,
    VariableKind,
};
use otspot_num::{CscMatrixView, NumericError};

use crate::conic::{
    solve_misocp, solve_socp, BbOptions, ConeSpec, ConicOptions, ConicProblem, MisocpProblem,
};
use crate::mip::{solve_milp, solve_miqp, MilpProblem, MiqpProblem};
use crate::options::{MipConfig, SolverOptions};
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::qp::{solve_qp_with, QcqpMatrix, QpProblem};
use crate::sparse::CscMatrix;

fn status(status: &SolveStatus) -> IrStatus {
    match status {
        SolveStatus::Optimal | SolveStatus::NonconvexGlobal => IrStatus::Optimal,
        SolveStatus::SuboptimalSolution
        | SolveStatus::LocallyOptimal
        | SolveStatus::NonconvexLocal => IrStatus::FeasiblePoint,
        SolveStatus::FeasiblePoint => IrStatus::FeasiblePoint,
        SolveStatus::Infeasible => IrStatus::Infeasible,
        SolveStatus::Unbounded => IrStatus::Unbounded,
        SolveStatus::Stalled => IrStatus::Stalled,
        SolveStatus::MaxIterations => IrStatus::IterationLimit,
        SolveStatus::Timeout => IrStatus::Timeout,
        SolveStatus::NumericalError | SolveStatus::NonConvex(_) => IrStatus::NumericalFailure,
        SolveStatus::NotSupported(_) => IrStatus::NotSupported,
    }
}

fn outcome(result: SolverResult) -> SolveOutcome {
    SolveOutcome {
        status: status(&result.status),
        objective: result.objective.is_finite().then_some(result.objective),
        primal: result.solution,
        dual: result.dual_solution,
        proof: None,
        iterations: result.iterations,
    }
}

fn unsupported() -> SolveOutcome {
    SolveOutcome::unsupported()
}

fn row_senses<M: CscMatrixView>(
    problem: &OptimizationProblem<M>,
) -> Result<(Vec<f64>, Vec<ConstraintType>), NumericError> {
    let mut rhs = Vec::with_capacity(problem.constraints.lower.len());
    let mut senses = Vec::with_capacity(problem.constraints.lower.len());
    for (row, (&lower, &upper)) in problem
        .constraints
        .lower
        .iter()
        .zip(&problem.constraints.upper)
        .enumerate()
    {
        if lower == upper && lower.is_finite() {
            rhs.push(lower);
            senses.push(ConstraintType::Eq);
        } else if lower == f64::NEG_INFINITY && upper.is_finite() {
            rhs.push(upper);
            senses.push(ConstraintType::Le);
        } else if upper == f64::INFINITY && lower.is_finite() {
            rhs.push(lower);
            senses.push(ConstraintType::Ge);
        } else {
            return Err(NumericError::InvalidBounds {
                context: "legacy row representation",
                index: row,
                lower,
                upper,
            });
        }
    }
    Ok((rhs, senses))
}

fn qp_from_ir(problem: &OptimizationProblem<CscMatrix>) -> Result<QpProblem, NumericError> {
    let (rhs, senses) = row_senses(problem)?;
    let q = problem
        .objective
        .quadratic
        .clone()
        .unwrap_or_else(|| CscMatrix::new(problem.variables.len(), problem.variables.len()));
    let bounds = problem
        .variables
        .iter()
        .map(|variable| (variable.lower, variable.upper))
        .collect();
    let mut qp = QpProblem::new(
        q,
        problem.objective.linear.clone(),
        problem.constraints.matrix.clone(),
        rhs,
        bounds,
        senses,
    )
    .map_err(|_| NumericError::InvalidSparseStructure {
        message: "canonical IR could not be converted to QpProblem",
    })?;
    qp.obj_offset = problem.objective.offset;
    if !problem.quadratic_constraints.is_empty() {
        let mut quadratic =
            vec![QcqpMatrix::new(problem.variables.len()); problem.constraints.lower.len()];
        for constraint in &problem.quadratic_constraints {
            let mut triplets = Vec::with_capacity(constraint.quadratic.nnz());
            for column in 0..constraint.quadratic.ncols() {
                let (rows, values) = constraint.quadratic.column(column);
                triplets.extend(
                    rows.iter()
                        .zip(values)
                        .map(|(&row, &value)| (row, column, value)),
                );
            }
            quadratic[constraint.linear_row] = QcqpMatrix {
                n: problem.variables.len(),
                triplets,
            };
        }
        qp.quadratic_constraints = quadratic;
    }
    Ok(qp)
}

fn conic_from_ir(problem: &OptimizationProblem<CscMatrix>) -> Result<ConicProblem, NumericError> {
    let conic = problem
        .conic
        .as_ref()
        .ok_or(NumericError::InvalidSparseStructure {
            message: "missing conic system",
        })?;
    if problem.objective.quadratic.is_some() || !problem.quadratic_constraints.is_empty() {
        return Err(NumericError::InvalidSparseStructure {
            message: "quadratic terms must be bridged before conic dispatch",
        });
    }
    let (rhs, senses) = row_senses(problem)?;
    if senses.iter().any(|sense| *sense != ConstraintType::Eq) {
        return Err(NumericError::InvalidSparseStructure {
            message: "conic equality system contains a non-equality row",
        });
    }
    let mut l = 0;
    let mut soc = Vec::new();
    for cone in &conic.cones {
        match *cone {
            Cone::Nonnegative(dim) => l += dim,
            Cone::SecondOrder(dim) => soc.push(dim),
            Cone::Zero(_) | Cone::RotatedSecondOrder(_) => {
                return Err(NumericError::InvalidSparseStructure {
                    message: "cone kind is not supported by the legacy conic solver",
                })
            }
        }
    }
    Ok(ConicProblem {
        c: problem.objective.linear.clone(),
        a: problem.constraints.matrix.clone(),
        b: rhs,
        g: conic.matrix.clone(),
        h: conic.rhs.clone(),
        cone: ConeSpec { l, soc },
    })
}

/// Solve a canonical IR problem through the current production algorithms.
pub fn solve_ir(
    problem: &OptimizationProblem<CscMatrix>,
    options: &SolverOptions,
    context: &SolveContext,
) -> SolveOutcome {
    if problem.validate().is_err() {
        return unsupported();
    }
    let mut options = options.clone();
    options.deadline = context.deadline().or(options.deadline);
    options.cancel_flag = context.cancel_flag().or(options.cancel_flag);

    match problem.class() {
        ProblemClass::Lp => match qp_from_ir(problem) {
            Ok(qp) => outcome(solve_qp_with(&qp, &options)),
            Err(_) => unsupported(),
        },
        ProblemClass::Qp | ProblemClass::Qcqp => match qp_from_ir(problem) {
            Ok(qp) => outcome(solve_qp_with(&qp, &options)),
            Err(_) => unsupported(),
        },
        ProblemClass::Milp | ProblemClass::Miqp | ProblemClass::Miqcp => {
            let Ok(qp) = qp_from_ir(problem) else {
                return unsupported();
            };
            let integers = problem
                .variables
                .iter()
                .enumerate()
                .filter_map(|(index, variable)| {
                    (variable.kind != VariableKind::Continuous).then_some(index)
                })
                .collect();
            if problem.class() == ProblemClass::Milp {
                let lp = LpProblem {
                    c: qp.c,
                    a: std::sync::Arc::new(qp.a),
                    b: qp.b,
                    num_vars: qp.num_vars,
                    num_constraints: qp.num_constraints,
                    constraint_types: qp.constraint_types,
                    bounds: qp.bounds,
                    name: None,
                    obj_offset: qp.obj_offset,
                };
                match MilpProblem::new(lp, integers) {
                    Ok(milp) => outcome(solve_milp(&milp, &options, &MipConfig::default())),
                    Err(_) => unsupported(),
                }
            } else {
                match MiqpProblem::new(qp, integers) {
                    Ok(miqp) => outcome(solve_miqp(&miqp, &options, &MipConfig::default())),
                    Err(_) => unsupported(),
                }
            }
        }
        ProblemClass::Socp | ProblemClass::Misocp => {
            let Ok(base) = conic_from_ir(problem) else {
                return unsupported();
            };
            let conic_options = ConicOptions {
                deadline: context.deadline(),
                cancel_flag: context.cancel_flag(),
                ..Default::default()
            };
            if problem.class() == ProblemClass::Socp {
                let result = solve_socp(&base, &conic_options);
                SolveOutcome {
                    status: status(&result.status),
                    objective: result.objective.is_finite().then_some(result.objective),
                    primal: result.x,
                    dual: [result.y, result.z].concat(),
                    proof: None,
                    iterations: result.iterations,
                }
            } else {
                let mut integers = Vec::new();
                let mut lower = Vec::new();
                let mut upper = Vec::new();
                for (index, variable) in problem.variables.iter().enumerate() {
                    if variable.kind != VariableKind::Continuous {
                        integers.push(index);
                        lower.push(variable.lower);
                        upper.push(variable.upper);
                    }
                }
                let result = solve_misocp(
                    &MisocpProblem {
                        base,
                        integers,
                        int_lb: lower,
                        int_ub: upper,
                    },
                    &conic_options,
                    &BbOptions::default(),
                );
                SolveOutcome {
                    status: status(&result.status),
                    objective: result.objective.is_finite().then_some(result.objective),
                    primal: result.x,
                    dual: Vec::new(),
                    proof: None,
                    iterations: result.nodes,
                }
            }
        }
    }
}
