use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use otspot as ots;
use ots::model::{Constraint, Expression, Model, ModelError, ModelResult, SolveError, Variable};
use ots::sparse::CscMatrix;
use ots::Tolerance;

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Python exceptions
// ---------------------------------------------------------------------------

pyo3::create_exception!(otspot, OtspotError, PyRuntimeError, "Base error for otspot.");
pyo3::create_exception!(otspot, InfeasibleError, OtspotError, "Problem is infeasible.");
pyo3::create_exception!(otspot, UnboundedError, OtspotError, "Problem is unbounded.");
pyo3::create_exception!(
    otspot,
    MaxIterationsError,
    OtspotError,
    "Solver reached max iterations."
);
pyo3::create_exception!(
    otspot,
    NumericalSolveError,
    OtspotError,
    "Numerical breakdown during solve."
);
pyo3::create_exception!(otspot, SolveTimeoutError, OtspotError, "Solver timed out.");

fn model_error_to_py(e: ModelError) -> PyErr {
    match e {
        ModelError::SolveError(SolveError::Infeasible) => {
            InfeasibleError::new_err("Problem is infeasible")
        }
        ModelError::SolveError(SolveError::Unbounded) => {
            UnboundedError::new_err("Problem is unbounded")
        }
        ModelError::SolveError(SolveError::MaxIterations) => {
            MaxIterationsError::new_err("Max iterations reached without convergence")
        }
        ModelError::SolveError(SolveError::NumericalError) => {
            NumericalSolveError::new_err("Numerical breakdown during solve")
        }
        ModelError::Timeout => SolveTimeoutError::new_err("Solver timed out"),
        ModelError::NoObjective => PyValueError::new_err(
            "No objective set. Call minimize() or maximize() before solve().",
        ),
        _ => OtspotError::new_err(e.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Internal expression representation
//
// Expression::constant and Expression::coefficients are pub(crate) so we
// cannot read them outside the otspot crate. Instead, PyExpression owns
// its own HashMap<Variable, f64> + constant and converts to otspot::Expression
// only when passing into the solver.
// ---------------------------------------------------------------------------

/// Owner of linear expression data that lives outside the otspot crate.
#[derive(Clone, Default)]
struct ExprData {
    coeffs: HashMap<Variable, f64>,
    constant: f64,
}

impl ExprData {
    fn from_var(v: Variable) -> Self {
        let mut coeffs = HashMap::new();
        coeffs.insert(v, 1.0);
        ExprData { coeffs, constant: 0.0 }
    }

    fn from_const(c: f64) -> Self {
        ExprData { coeffs: HashMap::new(), constant: c }
    }

    /// Build the Rust `Expression` for passing into the solver.
    fn to_rust(&self) -> Expression {
        let mut result = Expression::from_constant(self.constant);
        for (&var, &coeff) in &self.coeffs {
            result = result + (Expression::from(var) * coeff);
        }
        result
    }

    fn add(&self, rhs: &ExprData) -> ExprData {
        let mut coeffs = self.coeffs.clone();
        for (&var, &c) in &rhs.coeffs {
            *coeffs.entry(var).or_insert(0.0) += c;
        }
        ExprData { coeffs, constant: self.constant + rhs.constant }
    }

    fn sub(&self, rhs: &ExprData) -> ExprData {
        let mut coeffs = self.coeffs.clone();
        for (&var, &c) in &rhs.coeffs {
            *coeffs.entry(var).or_insert(0.0) -= c;
        }
        ExprData { coeffs, constant: self.constant - rhs.constant }
    }

    fn scale(&self, s: f64) -> ExprData {
        let coeffs = self.coeffs.iter().map(|(&v, &c)| (v, c * s)).collect();
        ExprData { coeffs, constant: self.constant * s }
    }
}

// ---------------------------------------------------------------------------
// Helper: extract any Python object as an ExprData
// ---------------------------------------------------------------------------

fn extract_expr(ob: &Bound<'_, PyAny>) -> PyResult<ExprData> {
    if let Ok(v) = ob.extract::<PyRef<PyVariable>>() {
        return Ok(ExprData::from_var(v.inner));
    }
    if let Ok(e) = ob.extract::<PyRef<PyExpression>>() {
        return Ok(e.data.clone());
    }
    if let Ok(f) = ob.extract::<f64>() {
        return Ok(ExprData::from_const(f));
    }
    if let Ok(i) = ob.extract::<i64>() {
        return Ok(ExprData::from_const(i as f64));
    }
    Err(PyValueError::new_err(format!(
        "Cannot convert {} to Expression; expected Variable, Expression, int, or float",
        ob.get_type().name()?
    )))
}

fn extract_scalar(ob: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(f) = ob.extract::<f64>() {
        return Ok(f);
    }
    if let Ok(i) = ob.extract::<i64>() {
        return Ok(i as f64);
    }
    Err(PyValueError::new_err(format!(
        "Expected a numeric scalar (int or float), got {}",
        ob.get_type().name()?
    )))
}

fn make_constraint(lhs: ExprData, rhs: ExprData, op: CompareOp) -> PyResult<PyConstraint> {
    let inner: Constraint = match op {
        CompareOp::Le => lhs.to_rust().leq(rhs.to_rust()),
        CompareOp::Ge => lhs.to_rust().geq(rhs.to_rust()),
        CompareOp::Eq => lhs.to_rust().eq_constraint(rhs.to_rust()),
        _ => unreachable!(),
    };
    Ok(PyConstraint { inner })
}

// ---------------------------------------------------------------------------
// PyVariable
// ---------------------------------------------------------------------------

#[pyclass(name = "Variable")]
#[derive(Clone)]
struct PyVariable {
    inner: Variable,
    /// Variable index mirrored here for hashing/repr (Variable::index is pub(crate)).
    idx: usize,
}

#[pymethods]
impl PyVariable {
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        Ok(PyExpression { data: ExprData::from_var(self.inner).add(&extract_expr(other)?) })
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        Ok(PyExpression { data: extract_expr(other)?.add(&ExprData::from_var(self.inner)) })
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        Ok(PyExpression { data: ExprData::from_var(self.inner).sub(&extract_expr(other)?) })
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        Ok(PyExpression { data: extract_expr(other)?.sub(&ExprData::from_var(self.inner)) })
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        Ok(PyExpression { data: ExprData::from_var(self.inner).scale(extract_scalar(other)?) })
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        self.__mul__(other)
    }

    fn __neg__(&self) -> PyExpression {
        PyExpression { data: ExprData::from_var(self.inner).scale(-1.0) }
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<PyObject> {
        let py = other.py();
        let lhs = ExprData::from_var(self.inner);
        let rhs = extract_expr(other)?;
        match op {
            CompareOp::Le | CompareOp::Ge | CompareOp::Eq => {
                Ok(Py::new(py, make_constraint(lhs, rhs, op)?)?.into_any())
            }
            _ => Ok(py.NotImplemented()),
        }
    }

    fn __hash__(&self) -> usize {
        self.idx
    }

    fn __repr__(&self) -> String {
        format!("Variable({})", self.idx)
    }
}

// ---------------------------------------------------------------------------
// PyExpression
// ---------------------------------------------------------------------------

#[pyclass(name = "Expression")]
#[derive(Clone)]
struct PyExpression {
    data: ExprData,
}

#[pymethods]
impl PyExpression {
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        Ok(PyExpression { data: self.data.add(&extract_expr(other)?) })
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        Ok(PyExpression { data: extract_expr(other)?.add(&self.data) })
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        Ok(PyExpression { data: self.data.sub(&extract_expr(other)?) })
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        Ok(PyExpression { data: extract_expr(other)?.sub(&self.data) })
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        Ok(PyExpression { data: self.data.scale(extract_scalar(other)?) })
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpression> {
        self.__mul__(other)
    }

    fn __neg__(&self) -> PyExpression {
        PyExpression { data: self.data.scale(-1.0) }
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<PyObject> {
        let py = other.py();
        let rhs = extract_expr(other)?;
        match op {
            CompareOp::Le | CompareOp::Ge | CompareOp::Eq => {
                Ok(Py::new(py, make_constraint(self.data.clone(), rhs, op)?)?.into_any())
            }
            _ => Ok(py.NotImplemented()),
        }
    }
}

// ---------------------------------------------------------------------------
// PyConstraint
// ---------------------------------------------------------------------------

#[pyclass(name = "Constraint")]
struct PyConstraint {
    inner: Constraint,
}

// ---------------------------------------------------------------------------
// PyModelResult
// ---------------------------------------------------------------------------

#[pyclass(name = "ModelResult")]
struct PyModelResult {
    inner: ModelResult,
}

#[pymethods]
impl PyModelResult {
    #[getter]
    fn objective(&self) -> f64 {
        self.inner.objective_value
    }

    fn value(&self, var: PyRef<PyVariable>) -> f64 {
        self.inner.value(var.inner)
    }

    fn __getitem__(&self, var: PyRef<PyVariable>) -> f64 {
        self.inner.value(var.inner)
    }

    #[getter]
    fn dual_solution(&self) -> Option<Vec<f64>> {
        self.inner.dual_solution.clone()
    }

    #[getter]
    fn reduced_costs(&self) -> Option<Vec<f64>> {
        self.inner.reduced_costs.clone()
    }

    #[getter]
    fn slack(&self) -> Option<Vec<f64>> {
        self.inner.slack.clone()
    }

    fn __repr__(&self) -> String {
        format!("ModelResult(objective={})", self.inner.objective_value)
    }
}

// ---------------------------------------------------------------------------
// PyModel
// ---------------------------------------------------------------------------

#[pyclass(name = "Model")]
struct PyModel {
    inner: Model,
    num_vars: usize,
}

#[pymethods]
impl PyModel {
    #[new]
    fn new(name: &str) -> Self {
        PyModel { inner: Model::new(name), num_vars: 0 }
    }

    /// Add a continuous variable. `lb` defaults to -∞, `ub` to +∞.
    #[pyo3(signature = (name, lb=None, ub=None))]
    fn add_var(&mut self, name: &str, lb: Option<f64>, ub: Option<f64>) -> PyVariable {
        let lb = lb.unwrap_or(f64::NEG_INFINITY);
        let ub = ub.unwrap_or(f64::INFINITY);
        let var = self.inner.add_var(name, lb, ub);
        let idx = self.num_vars;
        self.num_vars += 1;
        PyVariable { inner: var, idx }
    }

    /// Add an integer variable. `lb` defaults to -∞, `ub` to +∞.
    #[pyo3(signature = (name, lb=None, ub=None))]
    fn add_int_var(&mut self, name: &str, lb: Option<f64>, ub: Option<f64>) -> PyVariable {
        let lb = lb.unwrap_or(f64::NEG_INFINITY);
        let ub = ub.unwrap_or(f64::INFINITY);
        let var = self.inner.add_int_var(name, lb, ub);
        let idx = self.num_vars;
        self.num_vars += 1;
        PyVariable { inner: var, idx }
    }

    /// Add a binary variable (integer restricted to {0, 1}).
    fn add_binary_var(&mut self, name: &str) -> PyVariable {
        let var = self.inner.add_binary_var(name);
        let idx = self.num_vars;
        self.num_vars += 1;
        PyVariable { inner: var, idx }
    }

    fn add_constraint(&mut self, c: PyRef<PyConstraint>) -> PyResult<()> {
        self.inner.add_constraint(c.inner.clone());
        Ok(())
    }

    fn minimize(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let data = extract_expr(obj)?;
        // Constant terms in the objective do not affect the optimal solution
        // but shift the reported objective value. Use set_obj_offset so the
        // solver accounts for them correctly.
        self.inner.minimize(data.to_rust());
        self.inner.set_obj_offset(data.constant);
        Ok(())
    }

    fn maximize(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        let data = extract_expr(obj)?;
        self.inner.maximize(data.to_rust());
        self.inner.set_obj_offset(data.constant);
        Ok(())
    }

    /// Set a diagonal Q matrix from a Python list of floats.
    fn set_diagonal_q(&mut self, diag: Vec<f64>) -> PyResult<()> {
        if diag.len() != self.num_vars {
            return Err(PyValueError::new_err(format!(
                "set_diagonal_q: diag length {} != variable count {}",
                diag.len(),
                self.num_vars
            )));
        }
        self.inner.set_diagonal_q(&diag);
        Ok(())
    }

    /// Set a sparse Q matrix from (row, col, value) triplets.
    /// `n` must equal the number of variables. Example: `[(0,0,2.0),(1,1,2.0)]`.
    fn set_quadratic_objective(
        &mut self,
        triplets: &Bound<'_, PyAny>,
        n: usize,
    ) -> PyResult<()> {
        if n != self.num_vars {
            return Err(PyValueError::new_err(format!(
                "set_quadratic_objective: n={} != variable count {}",
                n, self.num_vars
            )));
        }
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for item in triplets.try_iter()? {
            let item = item?;
            let tup = item.downcast::<PyTuple>()?;
            if tup.len() != 3 {
                return Err(PyValueError::new_err(
                    "Each triplet must be (row: int, col: int, value: float)",
                ));
            }
            rows.push(tup.get_item(0)?.extract::<usize>()?);
            cols.push(tup.get_item(1)?.extract::<usize>()?);
            vals.push(tup.get_item(2)?.extract::<f64>()?);
        }
        let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n)
            .map_err(|e| PyValueError::new_err(format!("Invalid Q matrix: {e}")))?;
        self.inner.set_quadratic_objective(q);
        Ok(())
    }

    fn set_timeout(&mut self, secs: f64) {
        self.inner.set_timeout(secs);
    }

    fn set_threads(&mut self, n: usize) {
        self.inner.set_threads(n);
    }

    fn set_presolve(&mut self, flag: bool) {
        self.inner.set_presolve(flag);
    }

    /// Set solver tolerance. Maps to High (<=1e-8), Medium (<=1e-6), or Custom.
    fn set_tolerance(&mut self, eps: f64) -> PyResult<()> {
        if eps <= 0.0 {
            return Err(PyValueError::new_err("eps must be positive"));
        }
        let tol = if eps <= 1e-8 {
            Tolerance::High
        } else if eps <= 1e-6 {
            Tolerance::Medium
        } else {
            Tolerance::Custom(eps)
        };
        self.inner.set_tolerance(tol);
        Ok(())
    }

    fn solve(&mut self) -> PyResult<PyModelResult> {
        self.inner
            .solve()
            .map(|r| PyModelResult { inner: r })
            .map_err(model_error_to_py)
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Python bindings for the otspot optimization solver.
#[pymodule]
#[pyo3(name = "otspot")]
fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVariable>()?;
    m.add_class::<PyExpression>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyModelResult>()?;
    m.add("OtspotError", m.py().get_type::<OtspotError>())?;
    m.add("InfeasibleError", m.py().get_type::<InfeasibleError>())?;
    m.add("UnboundedError", m.py().get_type::<UnboundedError>())?;
    m.add("MaxIterationsError", m.py().get_type::<MaxIterationsError>())?;
    m.add("NumericalSolveError", m.py().get_type::<NumericalSolveError>())?;
    m.add("SolveTimeoutError", m.py().get_type::<SolveTimeoutError>())?;
    Ok(())
}
