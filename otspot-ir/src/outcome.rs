//! Solver-independent termination and proof types.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveStatus {
    Optimal,
    FeasiblePoint,
    Infeasible,
    Unbounded,
    Stalled,
    IterationLimit,
    Timeout,
    NumericalFailure,
    NotSupported,
}

#[derive(Debug, Clone)]
pub enum Proof {
    OptimalKkt {
        primal_residual: f64,
        dual_residual: f64,
        complementarity: f64,
    },
    Farkas {
        residual: f64,
    },
    UnboundedRay {
        residual: f64,
    },
    BoundGap {
        lower: f64,
        upper: f64,
    },
}

#[derive(Debug, Clone)]
pub struct SolveOutcome {
    pub status: SolveStatus,
    pub objective: Option<f64>,
    pub primal: Vec<f64>,
    pub dual: Vec<f64>,
    pub proof: Option<Proof>,
    pub iterations: usize,
}

impl SolveOutcome {
    pub fn unsupported() -> Self {
        Self {
            status: SolveStatus::NotSupported,
            objective: None,
            primal: Vec::new(),
            dual: Vec::new(),
            proof: None,
            iterations: 0,
        }
    }
}
