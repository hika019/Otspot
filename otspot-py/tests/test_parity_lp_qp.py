"""Behavior parity: the same small LP/QP/MILP problems, solved through the
Python `otspot` API, must match the same independently hand-computed oracle
values as the Rust side (see the `oracle` module in
tests/api_manifest_rust.rs -- the constants below are re-derived
independently here, not copied from a Rust computation, per CLAUDE.md's
independent-oracle requirement).

Does not depend on data/ (absent in worktrees, per repo convention).
"""

import threading

import pytest

import otspot

# IPM `Tolerance::Medium` (eps=1e-6) bounds KKT residuals, not raw
# variable-value error directly; matches the tolerance used on the Rust side.
TOL = 1e-4


def test_lp_oracle():
    """min x + 2y  s.t. -x <= 0, 2x + 3y <= 12, x + y >= 3, x in [0, inf), y in [0, 10].

    Hand solution: c_x=1 < c_y=2, so push y to 0 and satisfy x+y>=3 with
    x=3 (tight); 2*3+3*0=6<=12 and -3<=0 both have slack. Any y>0 raises the
    objective faster than it could relax the x lower bound, so (x,y)=(3,0)
    is optimal with objective 3.
    """
    model = otspot.Model("lp_oracle")
    x = model.add_var("x", 0.0, float("inf"))
    y = model.add_var("y", 0.0, 10.0)
    assert model.var_name(x) == "x"
    assert model.var_name(y) == "y"
    assert model.var_kind(x) == otspot.VarKind.Continuous

    # Row 0: exercises __neg__ (redundant vs. x's own lb=0, never binding).
    model.add_constraint((-x).leq(0.0))
    model.add_constraint((2.0 * x + 3.0 * y).leq(12.0))
    model.add_constraint((x + y).geq(3.0))
    model.minimize(x + 2.0 * y)
    model.set_tolerance(otspot.Tolerance.Medium())
    model.set_presolve(True)
    model.set_threads(1)
    model.set_obj_offset(0.0)

    result = model.solve()
    assert isinstance(result.status, otspot.SolveStatus.Optimal)
    assert result.proof == otspot.SolutionProof.GlobalOptimal
    assert result.has_global_optimality_proof()
    assert abs(result.objective() - 3.0) < TOL
    assert abs(result.value(x) - 3.0) < TOL
    assert abs(result[y] - 0.0) < TOL

    # Sign-convention-independent checks: exact slack values, and
    # complementary-slackness zero-duals on the two non-binding rows.
    slack = result.slack
    assert slack is not None and len(slack) == 3
    assert abs(slack[1] - 6.0) < TOL  # row 1 (2x+3y<=12): 12 - 6 = 6
    assert abs(slack[2] - 0.0) < TOL  # row 2 (x+y>=3): tight

    dual = result.dual_solution
    assert dual is not None and len(dual) == 3
    assert abs(dual[0]) < TOL, "row 0 (-x<=0) is slack"
    assert abs(dual[1]) < TOL, "row 1 (2x+3y<=12) is slack"

    rc = result.reduced_costs
    assert rc is not None and len(rc) == 2
    assert abs(rc[0]) < TOL, "x is not at a bound"

    assert result.bound_duals == [], "LP path leaves bound_duals empty by design"


def test_qp_oracle():
    """min x^2 + y^2  s.t. x + y >= 2, x >= 0, y >= 0.

    Hand solution: the unconstrained minimizer of x^2+y^2 is the origin,
    which violates x+y>=2. The constrained minimum over the half-plane
    x+y>=2 is the perpendicular projection of the origin onto the line
    x+y=2, i.e. (1, 1) (elementary geometry). x=1,y=1 satisfies x,y>=0.
    Objective = 1+1 = 2. This constraint is load-bearing (removing it
    changes the optimum to (0, 0)), unlike an `x+y<=3` formulation where the
    unconstrained optimum already satisfies the bound.
    """
    model = otspot.Model("qp_oracle")
    x = model.add_var("x", 0.0, float("inf"))
    y = model.add_var("y", 0.0, float("inf"))

    model.add_constraint((x + y).geq(2.0))
    # `0.0 + ...` exercises QuadExpr.__radd__.
    obj = 0.0 + x.pow2() + y.pow2()
    assert not obj.is_linear()
    model.minimize(obj)
    model.set_timeout(30.0)

    result = model.solve()
    assert isinstance(result.status, otspot.SolveStatus.Optimal)
    assert abs(result.objective() - 2.0) < TOL
    assert abs(result.value(x) - 1.0) < TOL
    assert abs(result.value(y) - 1.0) < TOL

    dual = result.dual_solution
    assert dual is not None and len(dual) == 1
    assert abs(dual[0]) > 1e-3, f"x+y>=2 must be load-bearing, got {dual[0]}"
    assert result.bound_duals != [], "QP path must populate bound_duals, unlike LP"


def test_milp_oracle():
    """min 2b + z  s.t. b + z >= 2.5, b in {0,1}, z integer in [0, 10].

    Hand solution: b=0 forces z>=2.5, i.e. z>=3 (integer ceiling),
    objective 3. b=1 forces z>=1.5, i.e. z>=2, objective 2+2=4. The minimum
    over both branches is 3, at (b=0, z=3).
    """
    model = otspot.Model("milp_oracle")
    b = model.add_binary_var("b")
    z = model.add_int_var("z", 0.0, 10.0)
    assert model.var_kind(b) == otspot.VarKind.Binary
    assert model.var_kind(z) == otspot.VarKind.Integer

    model.add_constraint((b + z).geq(2.5))
    model.minimize(2.0 * b + z)

    result = model.solve()
    assert isinstance(result.status, otspot.SolveStatus.Optimal)
    assert abs(result.objective() - 3.0) < TOL
    assert abs(result.value(b) - 0.0) < TOL
    assert abs(result.value(z) - 3.0) < TOL


def test_maximize_oracle():
    """max x + y  s.t. x + y <= 8, x,y in [0, 10]. Multiple optima; the
    objective value (8) is the unique, checkable quantity."""
    model = otspot.Model("maximize_oracle")
    x = model.add_var("x", 0.0, 10.0)
    y = model.add_var("y", 0.0, 10.0)
    model.add_constraint((x + y).leq(8.0))
    # `0.0 + x` exercises Variable.__radd__.
    model.maximize(0.0 + x + y)
    result = model.solve()
    assert isinstance(result.status, otspot.SolveStatus.Optimal)
    assert abs(result.objective() - 8.0) < TOL


def test_eq_constraint_oracle():
    """min x  s.t. 5.0 - x == 0, x in [0, 10]. Forces x=5 exactly."""
    model = otspot.Model("eq_constraint_oracle")
    x = model.add_var("x", 0.0, 10.0)
    # `5.0 - x` exercises Variable.__rsub__.
    model.add_constraint((5.0 - x).eq_constraint(0.0))
    model.minimize(x)
    result = model.solve()
    assert isinstance(result.status, otspot.SolveStatus.Optimal)
    assert abs(result.objective() - 5.0) < TOL
    assert abs(result.value(x) - 5.0) < TOL


def test_infeasible_lp_raises_solve_failed_error():
    model = otspot.Model("infeasible")
    x = model.add_var("x", 0.0, 1.0)
    model.add_constraint(x.geq(5.0))
    model.minimize(x)
    with pytest.raises(otspot.SolveFailedError) as exc_info:
        model.solve()
    assert exc_info.value.error == otspot.SolveError.Infeasible


def test_unbounded_lp_raises_solve_failed_error():
    """min -x  s.t. x >= 0 (objective goes to -inf)."""
    model = otspot.Model("unbounded")
    x = model.add_var("x", 0.0, float("inf"))
    model.minimize(-1.0 * x)
    with pytest.raises(otspot.SolveFailedError) as exc_info:
        model.solve()
    assert exc_info.value.error == otspot.SolveError.Unbounded


def test_missing_objective_raises_no_objective_error():
    model = otspot.Model("no_objective")
    model.add_var("x", 0.0, 1.0)
    with pytest.raises(otspot.NoObjectiveError):
        model.solve()


def test_zero_timeout_raises_timeout_error():
    """`set_timeout(0.0)` reliably (verified empirically) yields a Timeout
    even for a trivial LP, since the deadline is already expired by the
    first check."""
    model = otspot.Model("timeout")
    x = model.add_var("x", 0.0, float("inf"))
    y = model.add_var("y", 0.0, 10.0)
    model.add_constraint((2.0 * x + 3.0 * y).leq(12.0))
    model.add_constraint((x + y).geq(3.0))
    model.minimize(x + 2.0 * y)
    model.set_timeout(0.0)
    with pytest.raises(otspot.TimeoutError):
        model.solve()


def test_miqp_indefinite_q_raises_nonconvex_error():
    """Indefinite-Q MIQP (mirrors otspot-model's own
    `miqp_nonconvex_q_returns_nonconvex_error` sentinel). A *continuous*
    indefinite QP is not a reliable trigger: IPM inertia correction can
    converge it to a LocallyOptimal KKT point instead (verified empirically
    while designing this test), not NonConvex."""
    model = otspot.Model("nonconvex")
    x = model.add_binary_var("x")
    y = model.add_binary_var("y")
    model.minimize((-0.5) * x.pow2() + 0.5 * y.pow2())
    with pytest.raises(otspot.NonConvexError):
        model.solve()


def test_cross_model_value_raises_invalid_input_error():
    """A `Variable` from a different `Model` passed into `ModelResult.value`
    must raise, not silently return that other model's own value at the
    coincidentally-matching index (the otspot-model bug this PR fixes;
    otspot-model/src/model.rs has the corresponding Rust-level sentinel)."""
    model_a = otspot.Model("a")
    x_a = model_a.add_var("x_in_a", 0.0, 1.0)  # index 0
    model_a.minimize(x_a)
    model_a.solve()

    model_b = otspot.Model("b")
    y_b = model_b.add_var("y_in_b", 2.0, 2.0)  # index 0, fixed at 2.0
    model_b.minimize(y_b)
    result_b = model_b.solve()
    assert abs(result_b.value(y_b) - 2.0) < TOL

    with pytest.raises(otspot.InvalidInputError):
        result_b.value(x_a)
    with pytest.raises(otspot.InvalidInputError):
        result_b[x_a]


def test_cross_model_var_name_and_var_kind_raise_invalid_input_error():
    model_a = otspot.Model("a")
    x_a = model_a.add_binary_var("x_in_a")

    model_b = otspot.Model("b")
    with pytest.raises(otspot.InvalidInputError):
        model_b.var_name(x_a)
    with pytest.raises(otspot.InvalidInputError):
        model_b.var_kind(x_a)


def test_solve_releases_the_gil():
    """`Model.solve` must run under `Python::detach` so other Python threads
    can make progress during a long solve (otherwise a `timeout_secs`-bounded
    solve -- unbounded by default -- would also freeze `KeyboardInterrupt`
    delivery for its whole duration).

    Design: a moderately sized LP (~0.4s to solve) plus a background thread
    that increments a counter in a tight loop with no blocking calls. If the
    GIL is held throughout `solve()`, the background thread cannot execute
    any Python bytecode during that window (measured empirically while
    writing this test: ~1.2e5 increments, a fixed cost from thread startup
    latency, independent of solve duration -- confirmed by re-running a 3.5x
    longer solve and seeing the same ~1.2e5 count). With the GIL released,
    the same setup reaches ~4.5e6 increments. The threshold below sits
    comfortably between the two regimes (4x above the held-GIL ceiling, 9x
    below the typical released count): an order-of-magnitude check, not a
    tight timing race.
    """
    n = 150
    model = otspot.Model("gil_release_probe")
    variables = [model.add_var(f"x{i}", 0.0, 10.0) for i in range(n)]
    obj = variables[0]
    for i in range(1, n):
        obj = obj + float(i % 7 + 1) * variables[i]
    model.minimize(obj)
    for i in range(n - 1):
        model.add_constraint((variables[i] + variables[i + 1]).geq(float((i % 5) + 1)))

    counter = {"n": 0}
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            counter["n"] += 1

    thread = threading.Thread(target=spin)
    thread.start()
    try:
        result = model.solve()
    finally:
        stop.set()
        thread.join()

    assert isinstance(result.status, otspot.SolveStatus.Optimal)
    assert counter["n"] > 500_000, (
        f"background thread only progressed {counter['n']} increments during "
        "solve() -- GIL appears to be held throughout, not released"
    )
