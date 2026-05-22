"""
Smoke tests for otspot Python bindings.

Each test verifies the solution against a known analytical optimum, NOT
against the Rust implementation (to avoid "implementation as reference" trap).
"""

import math
import pytest
import otspot


ABS_TOL = 1e-4  # absolute tolerance for solution comparison


# ---------------------------------------------------------------------------
# QP: min x^2 + y^2 - 4x - 4y  s.t.  x+y <= 3, x,y >= 0
#   Analytical: constraint x+y=3 active at optimum, by symmetry x=y=1.5
#   Objective = 1.5^2 + 1.5^2 - 4*1.5 - 4*1.5 = -7.5
# ---------------------------------------------------------------------------

def test_qp_basic():
    m = otspot.Model("qp")
    x = m.add_var("x", lb=0)
    y = m.add_var("y", lb=0)

    m.set_diagonal_q([2, 2])
    m.add_constraint(x + y <= 3)
    m.minimize(-4 * x + -4 * y)

    r = m.solve()
    assert abs(r.objective - (-7.5)) < ABS_TOL
    assert abs(r[x] - 1.5) < ABS_TOL
    assert abs(r[y] - 1.5) < ABS_TOL


def test_qp_value_method():
    """ModelResult.value(var) is equivalent to r[var]."""
    m = otspot.Model("qp2")
    x = m.add_var("x", lb=0)
    y = m.add_var("y", lb=0)
    m.set_diagonal_q([2, 2])
    m.add_constraint(x + y <= 3)
    m.minimize(-4 * x + -4 * y)
    r = m.solve()
    assert abs(r.value(x) - r[x]) < 1e-15
    assert abs(r.value(y) - r[y]) < 1e-15


def test_qp_triplet_api():
    """set_quadratic_objective via triplets gives the same result as set_diagonal_q."""
    m1 = otspot.Model("qp_diag")
    x1 = m1.add_var("x", lb=0)
    y1 = m1.add_var("y", lb=0)
    m1.set_diagonal_q([2, 2])
    m1.add_constraint(x1 + y1 <= 3)
    m1.minimize(-4 * x1 + -4 * y1)
    r1 = m1.solve()

    m2 = otspot.Model("qp_triplet")
    x2 = m2.add_var("x", lb=0)
    y2 = m2.add_var("y", lb=0)
    m2.set_quadratic_objective([(0, 0, 2.0), (1, 1, 2.0)], 2)
    m2.add_constraint(x2 + y2 <= 3)
    m2.minimize(-4 * x2 + -4 * y2)
    r2 = m2.solve()

    assert abs(r1.objective - r2.objective) < ABS_TOL
    assert abs(r1[x1] - r2[x2]) < ABS_TOL
    assert abs(r1[y1] - r2[y2]) < ABS_TOL


# ---------------------------------------------------------------------------
# LP: min x + 2y  s.t.  2x+3y<=12, x+y>=3, x>=0, 0<=y<=10
#   Analytical: at corner x=3, y=0 → objective = 3
# ---------------------------------------------------------------------------

def test_lp_basic():
    m = otspot.Model("lp")
    x = m.add_var("x", lb=0)
    y = m.add_var("y", lb=0, ub=10)

    m.add_constraint(2 * x + 3 * y <= 12)
    m.add_constraint(x + y >= 3)
    m.minimize(x + 2 * y)

    r = m.solve()
    assert abs(r.objective - 3.0) < ABS_TOL
    assert abs(r[x] - 3.0) < ABS_TOL
    assert abs(r[y] - 0.0) < ABS_TOL


def test_lp_maximize():
    """LP maximization: max x  s.t.  x<=5, x>=0 → x=5."""
    m = otspot.Model("max_lp")
    x = m.add_var("x", lb=0, ub=10)
    m.add_constraint(x <= 5)
    m.maximize(x)
    r = m.solve()
    assert abs(r.objective - 5.0) < ABS_TOL
    assert abs(r[x] - 5.0) < ABS_TOL


def test_lp_equality_constraint():
    """LP with equality constraint: min x+y  s.t.  x+y==4, x>=0, y>=0."""
    m = otspot.Model("eq_lp")
    x = m.add_var("x", lb=0)
    y = m.add_var("y", lb=0)
    m.add_constraint((x + y) == 4)
    m.minimize(x + y)
    r = m.solve()
    assert abs(r.objective - 4.0) < ABS_TOL
    assert abs(r[x] + r[y] - 4.0) < ABS_TOL


# ---------------------------------------------------------------------------
# MILP: min -x - 2y  s.t.  x+y<=3.5, x,y in {0,1,2,...}  (binary approx)
#   With integer variables: x and y in integers bounded by [0,3]
#   Analytical: x=1, y=2 gives x+y=3<=3.5, obj=-1-4=-5
#   (x=2,y=2 gives x+y=4>3.5 — infeasible; x=1,y=2 is optimal)
# ---------------------------------------------------------------------------

def test_milp_basic():
    m = otspot.Model("milp")
    x = m.add_int_var("x", lb=0, ub=3)
    y = m.add_int_var("y", lb=0, ub=3)

    m.add_constraint(x + y <= 3.5)
    m.minimize(-x + -2 * y)

    r = m.solve()
    # Analytical: x=1, y=2 or x=0, y=3 — check obj=-5 or obj=-6
    # x=0, y=3: x+y=3<=3.5 ✓, obj=0-6=-6 → better
    # x=1, y=2: obj=-1-4=-5 → worse
    # So optimal is x=0, y=3, objective=-6
    assert abs(r.objective - (-6.0)) < ABS_TOL
    assert r[x] == pytest.approx(0.0, abs=ABS_TOL)
    assert r[y] == pytest.approx(3.0, abs=ABS_TOL)


def test_milp_binary():
    """Binary variables: max x + 2y  s.t.  x+y<=1.5, x,y in {0,1}."""
    m = otspot.Model("milp_bin")
    x = m.add_binary_var("x")
    y = m.add_binary_var("y")
    # x+y <= 1.5 → at most one of them can be 1 (if both=1 → 2 > 1.5)
    m.add_constraint(x + y <= 1.5)
    m.maximize(x + 2 * y)
    r = m.solve()
    # Optimal: y=1, x=0 → x+y=1<=1.5, obj=0+2=2
    assert abs(r.objective - 2.0) < ABS_TOL
    assert r[x] == pytest.approx(0.0, abs=ABS_TOL)
    assert r[y] == pytest.approx(1.0, abs=ABS_TOL)


# ---------------------------------------------------------------------------
# DSL operator tests
# ---------------------------------------------------------------------------

def test_dsl_radd():
    """5 + x should work via __radd__."""
    m = otspot.Model("dsl")
    x = m.add_var("x", lb=0, ub=10)
    m.minimize(5 + x)
    m.add_constraint(x >= 3)
    r = m.solve()
    # min 5+x s.t. x>=3  → x=3, obj=8
    assert abs(r.objective - 8.0) < ABS_TOL


def test_dsl_rsub():
    """10 - x should work via __rsub__."""
    m = otspot.Model("dsl2")
    x = m.add_var("x", lb=0, ub=10)
    m.maximize(10 - x)
    m.add_constraint(x >= 2)
    r = m.solve()
    # max 10-x s.t. x>=2  → x=2, obj=8
    assert abs(r.objective - 8.0) < ABS_TOL


def test_dsl_neg():
    """Negation of variable."""
    m = otspot.Model("neg")
    x = m.add_var("x", lb=0, ub=5)
    m.minimize(-x)
    r = m.solve()
    # min -x s.t. 0<=x<=5 → x=5, obj=-5
    assert abs(r.objective - (-5.0)) < ABS_TOL
    assert abs(r[x] - 5.0) < ABS_TOL


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_infeasible():
    m = otspot.Model("infeasible")
    x = m.add_var("x", lb=0, ub=1)
    m.add_constraint(x >= 2)  # x<=1 but x>=2 — infeasible
    m.minimize(x)
    with pytest.raises(otspot.InfeasibleError):
        m.solve()


def test_no_objective():
    m = otspot.Model("no_obj")
    x = m.add_var("x", lb=0)
    with pytest.raises(Exception):
        m.solve()


def test_set_diagonal_q_length_mismatch():
    m = otspot.Model("qp_err")
    m.add_var("x", lb=0)
    with pytest.raises(ValueError):
        m.set_diagonal_q([2.0, 2.0])  # 2 values but only 1 variable


def test_set_quadratic_objective_n_mismatch():
    m = otspot.Model("qp_err2")
    m.add_var("x", lb=0)  # only 1 variable
    with pytest.raises(ValueError):
        m.set_quadratic_objective([(0, 0, 2.0)], 2)  # n=2 but 1 variable


# ---------------------------------------------------------------------------
# Constant-term handling (DSL → constraint normalization / objective offset)
# These exercise the Python-side ExprData re-implementation against analytical
# optima, NOT against the Rust path.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build_constraint, expected_x", [
    # x + 1 <= 3  →  x <= 2
    (lambda m, x: x + 1 <= 3, 2.0),
    # 2x + 3 <= x + 7  →  x <= 4   (constants on BOTH sides)
    (lambda m, x: 2 * x + 3 <= x + 7, 4.0),
    # 10 - x <= 7  →  x >= 3 ... but maximizing x within ub=10 unaffected;
    # use a binding upper form: x - 2 <= 5 → x <= 7
    (lambda m, x: x - 2 <= 5, 7.0),
])
def test_constraint_constant_move(build_constraint, expected_x):
    """Constants in a constraint expression must be transposed to the RHS."""
    m = otspot.Model("cc")
    x = m.add_var("x", lb=0, ub=10)
    m.add_constraint(build_constraint(m, x))
    m.maximize(x)
    r = m.solve()
    assert abs(r[x] - expected_x) < ABS_TOL


@pytest.mark.parametrize("build_obj, lb, expected_obj", [
    # x + x = 2x ; min over x>=3 → 6
    (lambda x: x + x, 3.0, 6.0),
    # 2x - x = x ; min over x>=5 → 5
    (lambda x: 2 * x - x, 5.0, 5.0),
    # 3x - x - x = x ; min over x>=4 → 4
    (lambda x: 3 * x - x - x, 4.0, 4.0),
])
def test_coefficient_aggregation(build_obj, lb, expected_obj):
    """Repeated variables must aggregate their coefficients."""
    m = otspot.Model("agg")
    x = m.add_var("x", lb=0)
    m.add_constraint(x >= lb)
    m.minimize(build_obj(x))
    r = m.solve()
    assert abs(r.objective - expected_obj) < ABS_TOL


def test_objective_constant_offset_min():
    """A constant in a minimize objective shifts the reported value (+offset)."""
    m = otspot.Model("off_min")
    x = m.add_var("x", lb=0)
    m.add_constraint(x >= 3)
    m.minimize(x + 10)  # min x+10 s.t. x>=3 → 3+10 = 13
    r = m.solve()
    assert abs(r.objective - 13.0) < ABS_TOL
    assert abs(r[x] - 3.0) < ABS_TOL


def test_objective_constant_offset_max():
    """For maximize, the offset is added (not negated) to the true optimum."""
    m = otspot.Model("off_max")
    x = m.add_var("x", lb=0, ub=10)
    m.add_constraint(x <= 4)
    m.maximize(x + 5)  # max x+5 s.t. x<=4 → 4+5 = 9
    r = m.solve()
    assert abs(r.objective - 9.0) < ABS_TOL


def test_objective_constant_offset_qp():
    """Offset is applied post-solve on the QP path too."""
    m = otspot.Model("off_qp")
    x = m.add_var("x", lb=0)
    y = m.add_var("y", lb=0)
    m.set_diagonal_q([2, 2])
    m.add_constraint(x + y <= 3)
    m.minimize(-4 * x + -4 * y + 100)  # base optimum -7.5, +100 → 92.5
    r = m.solve()
    assert abs(r.objective - 92.5) < ABS_TOL


def test_repeated_minimize_resets_offset():
    """A second minimize() must replace, not accumulate, the offset."""
    m = otspot.Model("reset")
    x = m.add_var("x", lb=0)
    m.add_constraint(x >= 3)
    m.minimize(x + 5)
    m.minimize(x + 2)  # offset must now be 2, not 7
    r = m.solve()
    assert abs(r.objective - 5.0) < ABS_TOL  # 3 + 2


def test_default_bounds_free_variable():
    """Omitting lb/ub gives ±inf: a free variable bounded only by a constraint."""
    m = otspot.Model("free")
    x = m.add_var("x")  # (-inf, +inf)
    m.add_constraint(x >= -5)
    m.minimize(x)  # min x s.t. x>=-5 → -5
    r = m.solve()
    assert abs(r[x] - (-5.0)) < ABS_TOL


# ---------------------------------------------------------------------------
# Off-diagonal QP: min x² + xy + y²  s.t.  x + y == 1, x,y free
#   Q full symmetric = [[2,1],[1,2]] (½xᵀQx = x²+xy+y²)
#   KKT → x = y = 0.5, obj = 0.25+0.25+0.25 = 0.75
# ---------------------------------------------------------------------------

def test_qp_off_diagonal():
    m = otspot.Model("qp_offdiag")
    x = m.add_var("x")
    y = m.add_var("y")
    m.add_constraint((x + y) == 1)
    m.set_quadratic_objective([(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)], 2)
    m.minimize(0 * x + 0 * y)
    r = m.solve()
    assert abs(r[x] - 0.5) < 2e-3
    assert abs(r[y] - 0.5) < 2e-3
    assert abs(r.objective - 0.75) < 2e-3


# ---------------------------------------------------------------------------
# Operator coverage: int/float scalar mix across +, -, *, unary -, l/r forms
# ---------------------------------------------------------------------------

def test_operator_mix_int_float():
    """min (3 + 2.0*x - 1) = 2x + 2 ; s.t. x>=4 → 10."""
    m = otspot.Model("opmix")
    x = m.add_var("x", lb=0, ub=10)
    expr = 3 + 2.0 * x - 1  # __radd__ (int), __mul__ (float), __sub__ (int)
    m.add_constraint(x >= 4)
    m.minimize(expr)
    r = m.solve()
    assert abs(r.objective - 10.0) < ABS_TOL


def test_variable_times_variable_errors():
    """x * y is not a supported scalar product → ValueError (no silent quadratic)."""
    m = otspot.Model("mulerr")
    x = m.add_var("x")
    y = m.add_var("y")
    with pytest.raises(ValueError):
        _ = x * y


# ---------------------------------------------------------------------------
# Unbounded / Timeout exception mapping
# ---------------------------------------------------------------------------

def test_unbounded():
    m = otspot.Model("unb")
    x = m.add_var("x", lb=0)  # x in [0, inf)
    m.minimize(-x)  # → -inf
    with pytest.raises(otspot.UnboundedError):
        m.solve()


def test_timeout():
    m = otspot.Model("to")
    x = m.add_var("x", lb=0)
    y = m.add_var("y", lb=0)
    m.set_diagonal_q([2, 2])
    m.minimize(-4 * x + -4 * y)
    m.set_timeout(1e-6)  # 1 microsecond → always times out
    with pytest.raises(otspot.SolveTimeoutError):
        m.solve()


def test_exception_hierarchy():
    """All solver exceptions derive from OtspotError."""
    for exc in (
        otspot.InfeasibleError,
        otspot.UnboundedError,
        otspot.MaxIterationsError,
        otspot.NumericalSolveError,
        otspot.SolveTimeoutError,
    ):
        assert issubclass(exc, otspot.OtspotError)
