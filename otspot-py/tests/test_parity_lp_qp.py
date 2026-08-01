"""Behavior parity: the same small LP/QP problems, solved through the Python
`otspot` API, must match the same independently hand-computed oracle values
as the Rust side (see the `oracle` module in tests/api_manifest_rust.rs --
the constants below are re-derived independently here, not copied from a
Rust computation, per CLAUDE.md's independent-oracle requirement).

Does not depend on data/ (absent in worktrees, per repo convention).
"""

import otspot

# IPM `Tolerance::Medium` (eps=1e-6) bounds KKT residuals, not raw
# variable-value error directly; matches the tolerance used on the Rust side.
TOL = 1e-4


def test_lp_oracle():
    """min x + 2y  s.t. 2x + 3y <= 12, x + y >= 3, x in [0, inf), y in [0, 10].

    Hand solution: c_x=1 < c_y=2, so push y to 0 and satisfy x+y>=3 with
    x=3 (tight); 2*3+3*0=6<=12 has slack. Any y>0 raises the objective
    faster than it could relax the x lower bound, so (x,y)=(3,0) is optimal
    with objective 3.
    """
    model = otspot.Model("lp_oracle")
    x = model.add_var("x", 0.0, float("inf"))
    y = model.add_var("y", 0.0, 10.0)
    assert model.var_name(x) == "x"
    assert model.var_name(y) == "y"

    model.add_constraint((2.0 * x + 3.0 * y).leq(12.0))
    model.add_constraint((x + y).geq(3.0))
    model.minimize(x + 2.0 * y)

    result = model.solve()
    assert isinstance(result.status, otspot.SolveStatus.Optimal)
    assert result.proof == otspot.SolutionProof.GlobalOptimal
    assert result.has_global_optimality_proof()
    assert abs(result.objective() - 3.0) < TOL
    assert abs(result.value(x) - 3.0) < TOL
    assert abs(result[y] - 0.0) < TOL


def test_qp_oracle():
    """min x^2 + y^2 - 2x - 4y + 5  s.t. x + y <= 3, x >= 0, y >= 0.

    Hand solution: complete the square, x^2-2x+y^2-4y+5 = (x-1)^2+(y-2)^2.
    The unconstrained minimizer (1, 2) satisfies x+y=3<=3, x>=0, y>=0, so it
    is feasible; a feasible unconstrained minimizer of a strictly convex
    function is automatically the constrained global minimizer too.
    Objective value at (1, 2) is exactly 0.
    """
    model = otspot.Model("qp_oracle")
    x = model.add_var("x", 0.0, float("inf"))
    y = model.add_var("y", 0.0, float("inf"))

    model.add_constraint((x + y).leq(3.0))
    obj = x.pow2() + y.pow2() - 2.0 * x - 4.0 * y + 5.0
    assert not obj.is_linear()
    model.minimize(obj)
    model.set_timeout(30.0)

    result = model.solve()
    assert isinstance(result.status, otspot.SolveStatus.Optimal)
    assert abs(result.objective() - 0.0) < TOL
    assert abs(result.value(x) - 1.0) < TOL
    assert abs(result.value(y) - 2.0) < TOL


def test_infeasible_lp_raises_solve_failed_error():
    model = otspot.Model("infeasible")
    x = model.add_var("x", 0.0, 1.0)
    model.add_constraint(x.geq(5.0))
    model.minimize(x)
    try:
        model.solve()
        raise AssertionError("expected SolveFailedError")
    except otspot.SolveFailedError:
        pass


def test_missing_objective_raises_no_objective_error():
    model = otspot.Model("no_objective")
    model.add_var("x", 0.0, 1.0)
    try:
        model.solve()
        raise AssertionError("expected NoObjectiveError")
    except otspot.NoObjectiveError:
        pass


def test_maximize_lp():
    """max x + y  s.t. x + y <= 8, x in [0,10], y in [0,10].

    Hand solution: gradient of x+y is (1,1), constraint boundary x+y=8 is
    the whole feasible frontier in that direction; objective is exactly 8
    anywhere on it, so this checks the maximize sign-flip path rather than
    a single unique point.
    """
    model = otspot.Model("max_lp")
    x = model.add_var("x", 0.0, 10.0)
    y = model.add_var("y", 0.0, 10.0)
    model.add_constraint((x + y).leq(8.0))
    model.maximize(x + y)
    result = model.solve()
    assert isinstance(result.status, otspot.SolveStatus.Optimal)
    assert abs(result.objective() - 8.0) < TOL
