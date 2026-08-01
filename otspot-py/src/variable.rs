//! `Variable` binding: mirrors `otspot_model::Variable`.
//!
//! Arithmetic methods call straight through to the real Rust operator
//! overloads in `otspot_model::expression` / `otspot_model::quad_expr`.
//! `leq`/`geq`/`eq_constraint` replace Rust's `constraint!` macro (Python has
//! no macros): they convert `self` via `Expression::from` and call the real
//! `Expression::leq`/`geq`/`eq_constraint` — the exact code path the macro
//! itself expands to.

use otspot_model::{Expression, Variable};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::IntoPyObjectExt;

use crate::constraint::PyConstraint;
use crate::expr::{coerce, Operand};

#[pyclass(module = "otspot", name = "Variable", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyVariable(pub(crate) Variable);

fn not_implemented(py: Python<'_>) -> Py<PyAny> {
    py.NotImplemented()
}

#[pymethods]
impl PyVariable {
    fn pow2(&self) -> crate::expr::PyQuadExpr {
        crate::expr::PyQuadExpr(self.0.pow2())
    }

    fn __add__(&self, py: Python<'_>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let lhs = self.0;
        match coerce(rhs) {
            Some(Operand::F(f)) => crate::expr::PyExpression(lhs + f).into_py_any(py),
            Some(Operand::V(v)) => crate::expr::PyExpression(lhs + v).into_py_any(py),
            Some(Operand::E(e)) => crate::expr::PyExpression(lhs + e).into_py_any(py),
            Some(Operand::Q(q)) => crate::expr::PyQuadExpr(lhs + q).into_py_any(py),
            None => Ok(not_implemented(py)),
        }
    }

    fn __radd__(&self, py: Python<'_>, lhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match coerce(lhs) {
            Some(Operand::F(f)) => crate::expr::PyExpression(f + self.0).into_py_any(py),
            _ => Ok(not_implemented(py)),
        }
    }

    fn __sub__(&self, py: Python<'_>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let lhs = self.0;
        match coerce(rhs) {
            Some(Operand::F(f)) => crate::expr::PyExpression(lhs - f).into_py_any(py),
            Some(Operand::V(v)) => crate::expr::PyExpression(lhs - v).into_py_any(py),
            Some(Operand::E(e)) => crate::expr::PyExpression(lhs - e).into_py_any(py),
            Some(Operand::Q(q)) => crate::expr::PyQuadExpr(lhs - q).into_py_any(py),
            None => Ok(not_implemented(py)),
        }
    }

    fn __rsub__(&self, py: Python<'_>, lhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match coerce(lhs) {
            Some(Operand::F(f)) => crate::expr::PyExpression(f - self.0).into_py_any(py),
            _ => Ok(not_implemented(py)),
        }
    }

    fn __mul__(&self, py: Python<'_>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let lhs = self.0;
        match coerce(rhs) {
            Some(Operand::F(f)) => crate::expr::PyExpression(lhs * f).into_py_any(py),
            Some(Operand::V(v)) => crate::expr::PyQuadExpr(lhs * v).into_py_any(py),
            Some(Operand::E(e)) => crate::expr::PyQuadExpr(lhs * e).into_py_any(py),
            Some(Operand::Q(_)) | None => Ok(not_implemented(py)),
        }
    }

    fn __rmul__(&self, py: Python<'_>, lhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match coerce(lhs) {
            Some(Operand::F(f)) => crate::expr::PyExpression(f * self.0).into_py_any(py),
            _ => Ok(not_implemented(py)),
        }
    }

    fn __neg__(&self) -> crate::expr::PyExpression {
        crate::expr::PyExpression(-self.0)
    }

    /// `self <= rhs` via `Expression::from(self).leq(rhs)`.
    fn leq(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
        Ok(PyConstraint(Expression::from(self.0).leq(rhs_expr(rhs)?)))
    }

    /// `self >= rhs` via `Expression::from(self).geq(rhs)`.
    fn geq(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
        Ok(PyConstraint(Expression::from(self.0).geq(rhs_expr(rhs)?)))
    }

    /// `self == rhs` via `Expression::from(self).eq_constraint(rhs)`.
    fn eq_constraint(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyConstraint> {
        Ok(PyConstraint(
            Expression::from(self.0).eq_constraint(rhs_expr(rhs)?),
        ))
    }

    fn __repr__(&self) -> String {
        "Variable(...)".to_string()
    }
}

fn rhs_expr(rhs: &Bound<'_, PyAny>) -> PyResult<Expression> {
    match coerce(rhs) {
        Some(Operand::F(f)) => Ok(Expression::from(f)),
        Some(Operand::V(v)) => Ok(Expression::from(v)),
        Some(Operand::E(e)) => Ok(e),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "constraint right-hand side must be a Variable, Expression, or number",
        )),
    }
}

pub(crate) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyVariable>()?;
    Ok(())
}
