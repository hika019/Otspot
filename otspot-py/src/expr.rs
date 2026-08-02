//! `Expression` (linear) and `QuadExpr` (linear-or-quadratic objective)
//! bindings, plus the shared operand coercion used by all three arithmetic
//! classes (`Variable`, `Expression`, `QuadExpr`).
//!
//! Every arithmetic method below is a thin call-through to the real
//! `otspot_model` operator overload (`self.0 <op> rhs`), never a
//! reimplementation — Python and Rust execute the identical code path.

use otspot_model::{Expression, QuadExpr, Variable};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::IntoPyObjectExt;

use crate::constraint::PyConstraint;
use crate::variable::PyVariable;

/// One operand of a binary arithmetic expression, resolved from a Python
/// object by trying each bound class/`float` in turn.
pub(crate) enum Operand {
    F(f64),
    V(Variable),
    E(Expression),
    Q(QuadExpr),
}

/// Resolves `obj`'s concrete operand kind. Uses `downcast` (not `extract`)
/// for the three pyclass checks: `extract::<PyRef<T>>()` on a type mismatch
/// allocates and returns a Python `TypeError` that this function immediately
/// discards, and every arithmetic dunder tries up to three of these before
/// falling through to the `f64` case — `downcast` reports a type mismatch as
/// a plain Rust `Err` (no Python exception object), same outcome, no
/// allocation on the common multi-operand-type paths.
pub(crate) fn coerce(obj: &Bound<'_, PyAny>) -> Option<Operand> {
    if let Ok(q) = obj.cast::<PyQuadExpr>() {
        return Some(Operand::Q(q.borrow().0.clone()));
    }
    if let Ok(e) = obj.cast::<PyExpression>() {
        return Some(Operand::E(e.borrow().0.clone()));
    }
    if let Ok(v) = obj.cast::<PyVariable>() {
        return Some(Operand::V(v.borrow().0));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Some(Operand::F(f));
    }
    None
}

fn not_implemented(py: Python<'_>) -> Py<PyAny> {
    py.NotImplemented()
}

// ---------------------------------------------------------------------------
// Expression
// ---------------------------------------------------------------------------

/// A linear expression: mirrors `otspot_model::Expression`.
#[pyclass(module = "otspot", name = "Expression", from_py_object)]
#[derive(Clone)]
pub struct PyExpression(pub(crate) Expression);

#[pymethods]
impl PyExpression {
    fn __add__(&self, py: Python<'_>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let lhs = self.0.clone();
        match coerce(rhs) {
            Some(Operand::F(f)) => PyExpression(lhs + f).into_py_any(py),
            Some(Operand::V(v)) => PyExpression(lhs + v).into_py_any(py),
            Some(Operand::E(e)) => PyExpression(lhs + e).into_py_any(py),
            Some(Operand::Q(q)) => PyQuadExpr(lhs + q).into_py_any(py),
            None => Ok(not_implemented(py)),
        }
    }

    fn __radd__(&self, py: Python<'_>, lhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match coerce(lhs) {
            Some(Operand::F(f)) => PyExpression(f + self.0.clone()).into_py_any(py),
            _ => Ok(not_implemented(py)),
        }
    }

    fn __sub__(&self, py: Python<'_>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let lhs = self.0.clone();
        match coerce(rhs) {
            Some(Operand::F(f)) => PyExpression(lhs - f).into_py_any(py),
            Some(Operand::V(v)) => PyExpression(lhs - v).into_py_any(py),
            Some(Operand::E(e)) => PyExpression(lhs - e).into_py_any(py),
            Some(Operand::Q(q)) => PyQuadExpr(lhs - q).into_py_any(py),
            None => Ok(not_implemented(py)),
        }
    }

    fn __rsub__(&self, py: Python<'_>, lhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match coerce(lhs) {
            Some(Operand::F(f)) => PyExpression(f - self.0.clone()).into_py_any(py),
            _ => Ok(not_implemented(py)),
        }
    }

    fn __mul__(&self, py: Python<'_>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let lhs = self.0.clone();
        match coerce(rhs) {
            Some(Operand::F(f)) => PyExpression(lhs * f).into_py_any(py),
            Some(Operand::V(v)) => PyQuadExpr(lhs * v).into_py_any(py),
            _ => Ok(not_implemented(py)),
        }
    }

    fn __rmul__(&self, py: Python<'_>, lhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match coerce(lhs) {
            Some(Operand::F(f)) => PyExpression(f * self.0.clone()).into_py_any(py),
            _ => Ok(not_implemented(py)),
        }
    }

    /// `self += rhs`. `__add__` above always clones `self.0` (Python's `+`
    /// must not mutate either operand), so `obj = obj + term` in a loop is
    /// O(n) per step / O(n^2) total (measured: ~0.17s at n=16000, vs. ~0.007s
    /// at n=2000 -- a 24x slowdown for an 8x larger n). `+=` has no such
    /// obligation: Python's augmented-assignment protocol expects `__iadd__`
    /// to mutate in place when it can, so this uses `mem::take` to move the
    /// current value out (leaving a cheap `Default` placeholder) and adds
    /// into it directly, with no clone. Prefer `total += term` over
    /// `total = total + term` in accumulation loops (see otspot-py/README.md
    /// and otspot.pyi).
    ///
    /// `rhs: QuadExpr` is not accepted here (`self` stays an `Expression`
    /// Python object; PyO3's `__iadd__` mutates in place and cannot rebind
    /// `self` to a different type the way `__add__`'s ordinary return value
    /// can) -- raises `TypeError` telling the caller to use `self = self +
    /// rhs` for that specific combination instead.
    fn __iadd__(&mut self, rhs: &Bound<'_, PyAny>) -> PyResult<()> {
        match coerce(rhs) {
            Some(Operand::F(f)) => {
                let owned = std::mem::take(&mut self.0);
                self.0 = owned + f;
                Ok(())
            }
            Some(Operand::V(v)) => {
                let owned = std::mem::take(&mut self.0);
                self.0 = owned + v;
                Ok(())
            }
            Some(Operand::E(e)) => {
                let owned = std::mem::take(&mut self.0);
                self.0 = owned + e;
                Ok(())
            }
            Some(Operand::Q(_)) => Err(PyTypeError::new_err(
                "Expression += QuadExpr is not supported in place (it would change the \
                 object's type); use `expr = expr + quad_expr` instead",
            )),
            None => Err(PyTypeError::new_err(format!(
                "unsupported operand type(s) for +=: 'Expression' and {}",
                rhs.get_type().name()?
            ))),
        }
    }

    fn __neg__(&self) -> PyExpression {
        PyExpression(-self.0.clone())
    }

    /// `self <= rhs` — matches `Expression::leq`.
    fn leq(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
        build_constraint(self.0.clone(), rhs, Expression::leq)
    }

    /// `self >= rhs` — matches `Expression::geq`.
    fn geq(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
        build_constraint(self.0.clone(), rhs, Expression::geq)
    }

    /// `self == rhs` — matches `Expression::eq_constraint`.
    fn eq_constraint(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
        build_constraint(self.0.clone(), rhs, Expression::eq_constraint)
    }

    fn __repr__(&self) -> String {
        "Expression(...)".to_string()
    }
}

fn expression_rhs(rhs: &Bound<'_, PyAny>) -> PyResult<Expression> {
    match coerce(rhs) {
        Some(Operand::F(f)) => Ok(Expression::from(f)),
        Some(Operand::V(v)) => Ok(Expression::from(v)),
        Some(Operand::E(e)) => Ok(e),
        _ => Err(PyTypeError::new_err(
            "constraint right-hand side must be a Variable, Expression, or number",
        )),
    }
}

fn build_constraint(
    lhs: Expression,
    rhs: &Bound<'_, PyAny>,
    f: impl FnOnce(Expression, Expression) -> otspot_model::Constraint,
) -> PyResult<PyConstraint> {
    let rhs = expression_rhs(rhs)?;
    Ok(PyConstraint(f(lhs, rhs)))
}

// ---------------------------------------------------------------------------
// QuadExpr
// ---------------------------------------------------------------------------

/// A linear-or-quadratic objective expression: mirrors `otspot_model::QuadExpr`.
#[pyclass(module = "otspot", name = "QuadExpr", from_py_object)]
#[derive(Clone)]
pub struct PyQuadExpr(pub(crate) QuadExpr);

#[pymethods]
impl PyQuadExpr {
    fn is_linear(&self) -> bool {
        self.0.is_linear()
    }

    fn __add__(&self, py: Python<'_>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let lhs = self.0.clone();
        match coerce(rhs) {
            Some(Operand::F(f)) => PyQuadExpr(lhs + f).into_py_any(py),
            Some(Operand::V(v)) => PyQuadExpr(lhs + v).into_py_any(py),
            Some(Operand::E(e)) => PyQuadExpr(lhs + e).into_py_any(py),
            Some(Operand::Q(q)) => PyQuadExpr(lhs + q).into_py_any(py),
            None => Ok(not_implemented(py)),
        }
    }

    fn __radd__(&self, py: Python<'_>, lhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match coerce(lhs) {
            Some(Operand::F(f)) => PyQuadExpr(f + self.0.clone()).into_py_any(py),
            _ => Ok(not_implemented(py)),
        }
    }

    fn __sub__(&self, py: Python<'_>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let lhs = self.0.clone();
        match coerce(rhs) {
            Some(Operand::F(f)) => PyQuadExpr(lhs - f).into_py_any(py),
            Some(Operand::V(v)) => PyQuadExpr(lhs - v).into_py_any(py),
            Some(Operand::E(e)) => PyQuadExpr(lhs - e).into_py_any(py),
            Some(Operand::Q(q)) => PyQuadExpr(lhs - q).into_py_any(py),
            None => Ok(not_implemented(py)),
        }
    }

    fn __rsub__(&self, py: Python<'_>, lhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match coerce(lhs) {
            Some(Operand::F(f)) => PyQuadExpr(f - self.0.clone()).into_py_any(py),
            _ => Ok(not_implemented(py)),
        }
    }

    fn __mul__(&self, py: Python<'_>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match coerce(rhs) {
            Some(Operand::F(f)) => PyQuadExpr(self.0.clone() * f).into_py_any(py),
            _ => Ok(not_implemented(py)),
        }
    }

    fn __rmul__(&self, py: Python<'_>, lhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match coerce(lhs) {
            Some(Operand::F(f)) => PyQuadExpr(f * self.0.clone()).into_py_any(py),
            _ => Ok(not_implemented(py)),
        }
    }

    /// `self += rhs`. See `PyExpression::__iadd__`'s doc comment: same
    /// mem::take-based clone avoidance, no type-changing case here (unlike
    /// Expression, QuadExpr is already the "widest" of the two -- adding a
    /// Variable/Expression/QuadExpr/float always stays a QuadExpr).
    fn __iadd__(&mut self, rhs: &Bound<'_, PyAny>) -> PyResult<()> {
        match coerce(rhs) {
            Some(Operand::F(f)) => {
                let owned = std::mem::take(&mut self.0);
                self.0 = owned + f;
                Ok(())
            }
            Some(Operand::V(v)) => {
                let owned = std::mem::take(&mut self.0);
                self.0 = owned + v;
                Ok(())
            }
            Some(Operand::E(e)) => {
                let owned = std::mem::take(&mut self.0);
                self.0 = owned + e;
                Ok(())
            }
            Some(Operand::Q(q)) => {
                let owned = std::mem::take(&mut self.0);
                self.0 = owned + q;
                Ok(())
            }
            None => Err(PyTypeError::new_err(format!(
                "unsupported operand type(s) for +=: 'QuadExpr' and {}",
                rhs.get_type().name()?
            ))),
        }
    }

    fn __neg__(&self) -> PyQuadExpr {
        PyQuadExpr(-self.0.clone())
    }

    fn __repr__(&self) -> String {
        format!("QuadExpr(is_linear={})", self.0.is_linear())
    }
}

pub(crate) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyExpression>()?;
    m.add_class::<PyQuadExpr>()?;
    Ok(())
}
