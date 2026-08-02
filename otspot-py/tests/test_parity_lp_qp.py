"""Behavior parity: the same small LP/QP/MILP problems, solved through the
Python `otspot` API, must match the same independently hand-computed oracle
values as the Rust side (see the `oracle` module in
tests/api_manifest_rust.rs -- the constants below are re-derived
independently here, not copied from a Rust computation, per CLAUDE.md's
independent-oracle requirement).

Does not depend on data/ (absent in worktrees, per repo convention).
"""

import copy
import pickle
import threading
import time

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
    # `.objective()` and the `.objective_value` field must agree (the
    # manifest lists both; this also verifies `.objective()` is genuinely a
    # passthrough rather than an independently-computed value).
    assert result.objective() == result.objective_value
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

    # Exercises the field (manifested) without pinning the LP path's current
    # choice not to populate it as a cross-language contract: that's an
    # implementation detail of *this* solver's LP route, not something a
    # Rust<->Python parity test should encode as ground truth (the QP oracle
    # below separately verifies the field is populated when it is expected
    # to be, on the route where it is).
    assert isinstance(result.bound_duals, list)


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


def test_iadd_matches_add_and_preserves_identity():
    """`+=` (`__iadd__`) must produce the same expression `+` does. Solved on
    a small oracle (same `max x+y s.t. x+y<=8` problem as test_maximize_oracle)
    so this is a real correctness check, not just structural equality."""
    model = otspot.Model("iadd_oracle")
    x = model.add_var("x", 0.0, 10.0)
    y = model.add_var("y", 0.0, 10.0)
    model.add_constraint((x + y).leq(8.0))
    obj = x + 0.0  # force Expression (not Variable) before accumulating
    identity = id(obj)
    obj += y
    assert id(obj) == identity, "__iadd__ must mutate in place, not rebind to a new object"
    model.maximize(obj)
    result = model.solve()
    assert isinstance(result.status, otspot.SolveStatus.Optimal)
    assert abs(result.objective() - 8.0) < TOL

    # QuadExpr += also mutates in place.
    quad = x.pow2()
    quad_identity = id(quad)
    quad += y.pow2()
    assert id(quad) == quad_identity
    assert not quad.is_linear()

    # Expression += QuadExpr can't mutate self into a different Python type.
    expr = x + 0.0
    with pytest.raises(TypeError):
        expr += x.pow2()


def test_iadd_avoids_add_quadratic_blowup():
    """The actual point of `__iadd__` existing: `+` clones the growing
    expression on every call (Python's `+` must not mutate either operand),
    so `total = total + term` in a loop is O(n) per step / O(n^2) total;
    `+=` mutates in place via `mem::take`, no clone. No `Model.solve()` here
    (this is a pure Python-object timing check, not a solver correctness
    check -- solving a 4000-variable LP would make this test itself slow for
    no benefit). Measured while designing this fix: `+` took 0.0035s at
    n=2000 and 0.1743s at n=16000 (50x slower for an 8x larger n, i.e.
    non-linear); `+=` took 0.0035s and 0.0300s (8.6x, i.e. roughly linear).
    This test uses a smaller n (`+`'s absolute cost is real but not the
    point here) and a generous ratio threshold, since the exact multiplier
    is build-profile- and machine-dependent -- what must hold everywhere is
    that `+=` scales more favorably than `+` as n grows.
    """
    n = 6000
    model = otspot.Model("iadd_scaling_probe")
    variables = [model.add_var(f"x{i}", 0.0, 10.0) for i in range(n)]

    def time_accumulation(op) -> float:
        t0 = time.perf_counter()
        total = variables[0] + 0.0
        for i in range(1, n):
            total = op(total, variables[i])
        return time.perf_counter() - t0

    def do_add(a, b):
        return a + b

    def do_iadd(a, b):
        a += b
        return a

    add_time = time_accumulation(do_add)
    iadd_time = time_accumulation(do_iadd)
    assert iadd_time < add_time * 0.5, (
        f"+= ({iadd_time:.4f}s) should be meaningfully faster than + "
        f"({add_time:.4f}s) at n={n} -- the O(n) clone-per-call cost of + "
        "should dominate over += 's O(1)-per-call mem::take"
    )


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


def _build_gil_probe_model(n: int = 1200) -> otspot.Model:
    """A single-model-thread solve of this size takes ~0.12s in a release
    build and ~21s in debug (measured empirically: this LP's build+solve
    path is 130-170x slower in debug and grows worse than linearly with
    `n`). `n` is a compromise: large enough for a stable release-mode ratio
    signal (n=800 gave a consistent ~0.51 ratio across repeated trials, but
    one earlier one-off measurement under incidental system load read 0.89
    -- too close to the 0.75 threshold below for comfort; n=1200 keeps the
    same ~0.50-0.52 ratio with roughly double the absolute wall-clock
    margin against that kind of transient noise), while keeping the full
    test (2 serial + 2 concurrent solves) around ~65s even in debug,
    comfortably inside the 3-minute-per-test budget."""
    model = otspot.Model("gil_release_probe")
    variables = [model.add_var(f"x{i}", 0.0, 10.0) for i in range(n)]
    obj: otspot.Variable | otspot.Expression | otspot.QuadExpr = variables[0]
    for i in range(1, n):
        obj = obj + float(i % 7 + 1) * variables[i]
    model.minimize(obj)
    for i in range(n - 1):
        model.add_constraint((variables[i] + variables[i + 1]).geq(float((i % 5) + 1)))
    return model


def test_solve_releases_the_gil():
    """`Model.solve` must run under `Python::detach` so other Python threads
    can make progress during a long solve (otherwise a `timeout_secs`-bounded
    solve -- unbounded by default -- would also freeze `KeyboardInterrupt`
    delivery for its whole duration).

    Design: wall-clock ratio, not a counter-increment race. An earlier
    version of this test counted background-thread increments during a
    single ~0.4s (debug-build-calibrated) solve; it failed deterministically
    against a `--release` wheel (the same solve takes 5-6ms in release, far
    too short for the counter to distinguish "GIL held" from "GIL released"
    -- measured 4/4 CI failures). This version instead solves two
    *independent* `Model`s (never the same instance across threads: PyO3
    pyclasses use runtime borrow checking, so concurrent `&mut self` access
    to one shared `Model` raises "already borrowed", which is why each
    thread gets its own) serially and concurrently, and compares wall time.
    If the GIL is held throughout each `solve()`, the second thread cannot
    even *start* its own solve's Rust work until the first thread's call
    returns to Python, so concurrent time is roughly what serial is. With
    the GIL released, both run on separate cores.

    Threshold rationale (0.75, not changed lightly -- the margin is real but
    not huge): repeated measurements put the *released* regime's ratio at
    0.50-0.73 and the *held* (`Python::detach` reverted) regime's ratio at
    0.90-1.03 across build profiles and machine load; 0.75 sits in the gap
    between those ranges with headroom on both sides, but is closer to the
    released ceiling (0.73) than to the held floor (0.90) is comfortable.

    Requires >=2 available CPU cores to distinguish the two regimes at all:
    on a single-vCPU host, two threads cannot run Rust work in parallel
    regardless of whether the GIL is released, so the released regime would
    also read ratio ~1.0 and this test could fail even with a correct
    `Python::detach`. Not a concern for the CI runners this test targets
    (>=2 cores), but worth knowing if running locally in a constrained
    container/VM.
    """
    m1, m2 = _build_gil_probe_model(), _build_gil_probe_model()
    t0 = time.perf_counter()
    r1 = m1.solve()
    r2 = m2.solve()
    serial = time.perf_counter() - t0

    m3, m4 = _build_gil_probe_model(), _build_gil_probe_model()
    results: dict[str, otspot.ModelResult] = {}

    def run(key: str, model: otspot.Model) -> None:
        results[key] = model.solve()

    thread_a = threading.Thread(target=run, args=("a", m3))
    thread_b = threading.Thread(target=run, args=("b", m4))
    t0 = time.perf_counter()
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()
    concurrent = time.perf_counter() - t0

    for result in (r1, r2, results["a"], results["b"]):
        assert isinstance(result.status, otspot.SolveStatus.Optimal)

    assert concurrent < serial * 0.75, (
        f"concurrent={concurrent:.3f}s not comfortably faster than serial={serial:.3f}s "
        f"(ratio {concurrent / serial:.3f}) -- GIL appears to be held throughout "
        "solve(), preventing real concurrency"
    )


def test_solve_failed_error_error_attribute_survives_pickle():
    """`SolveFailedError.error` (a `PySolveError` set via `setattr`, see
    errors.rs) must survive `pickle`/`copy.deepcopy` for `multiprocessing`
    error propagation to work (a worker process's exception is pickled to
    send back to the parent). Regression: `PySolveError` initially had no
    `__reduce__`, so `pickle.dumps` on the exception raised `TypeError:
    cannot pickle 'otspot.SolveError' object`."""
    model = otspot.Model("infeasible_pickle_check")
    x = model.add_var("x", 0.0, 1.0)
    model.add_constraint(x.geq(5.0))
    model.minimize(x)
    with pytest.raises(otspot.SolveFailedError) as exc_info:
        model.solve()
    original = exc_info.value

    restored = pickle.loads(pickle.dumps(original))
    assert isinstance(restored, otspot.SolveFailedError)
    assert restored.error == otspot.SolveError.Infeasible

    deep_copied = copy.deepcopy(original)
    assert deep_copied.error == otspot.SolveError.Infeasible


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        (otspot.VarKind.Continuous, otspot.VarKind),
        (otspot.SolutionProof.GlobalOptimal, otspot.SolutionProof),
        (otspot.SolveError.Infeasible, otspot.SolveError),
        (otspot.SolveStatus.Optimal(), otspot.SolveStatus),
        (otspot.SolveStatus.NonConvex("indefinite Q"), otspot.SolveStatus),
        (otspot.Tolerance.Medium(), otspot.Tolerance),
        (otspot.Tolerance.Custom(1e-7), otspot.Tolerance),
    ],
)
def test_enum_values_survive_pickle_round_trip(value, expected_type):
    restored = pickle.loads(pickle.dumps(value))
    assert isinstance(restored, expected_type)
    assert restored == value or (
        # Complex-enum variants (SolveStatus/Tolerance) are not `eq`-comparable
        # (no `#[pyclass(eq)]`); compare via `__reduce__`'s own payload instead.
        type(restored) is type(value) and restored.__reduce__()[1] == value.__reduce__()[1]
    )
