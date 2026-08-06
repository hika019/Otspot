"""Behavior parity: the same small LP/QP/MILP problems, solved through the
Python `otspot` API, must match the same independently hand-computed oracle
values as the Rust side (see the `oracle` module in
tests/api_manifest_rust.rs -- the constants below are re-derived
independently here, not copied from a Rust computation, per CLAUDE.md's
independent-oracle requirement).

Does not depend on data/ (absent in worktrees, per repo convention).
"""

import copy
import os
import pickle
import signal
import sys
import threading
import time
from pathlib import Path

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


def test_iadd_self_alias_doubles_instead_of_panicking():
    """`expr += expr` (the identical Python object on both sides of `+=`) used
    to panic with `PanicException: Already mutably borrowed: PyBorrowError`
    (Codex PR #31 review, P2): `__iadd__(&mut self, rhs)` had PyO3 eagerly
    `.borrow_mut()` self as the receiver, and `coerce(rhs)`'s `.borrow()` on
    the *same* cell (`rhs` being self) then panicked. `PanicException`
    subclasses `BaseException`, not `Exception`, so ordinary `except
    Exception` does not catch it -- the same failure class `var_name`/
    `var_kind` were deliberately written to avoid.

    Fixed in `otspot-py/src/expr.rs`'s `__iadd__` for both `Expression` and
    `QuadExpr`: takes `slf: &Bound<'_, Self>` instead of `&mut self`, so PyO3
    does not borrow anything up front, and defers `slf.borrow_mut()` until
    *after* `coerce(rhs)` (the match scrutinee) has already run and dropped
    its own borrow of `rhs`. `coerce`/self and `slf.borrow_mut()` are then
    never borrowed at the same time even when `rhs` *is* `slf`, so no
    explicit identity check is needed: `coerce` clones out `self`'s current
    value first, then `self` is moved out (`mem::take`) and added to that
    clone -- `self + self`, i.e. `2 * self`. Independent oracle: `expr +=
    expr` on `x` (coefficient 1) must double it to 2, verified by solving
    `max 2x s.t. x<=5` -> 10 (not `max x s.t. x<=5` -> 5, which a no-op or a
    silently-discarded `+=` would give). Same check for QuadExpr via `y**2`
    doubled to `2*y**2`, minimized at y=3 (via `y>=3`) -> 18.
    """
    model = otspot.Model("iadd_self_alias_expr")
    x = model.add_var("x", 0.0, 5.0)
    expr = x + 0.0
    expr += expr
    model.maximize(expr)
    result = model.solve()
    assert isinstance(result.status, otspot.SolveStatus.Optimal)
    assert abs(result.objective() - 10.0) < TOL

    model2 = otspot.Model("iadd_self_alias_quad")
    y = model2.add_var("y", 0.0, 10.0)
    quad = y.pow2()
    quad += quad
    model2.add_constraint(y.geq(3.0))
    model2.minimize(quad)
    result2 = model2.solve()
    assert isinstance(result2.status, otspot.SolveStatus.Optimal)
    assert abs(result2.objective() - 18.0) < 1e-3

    # except Exception (not except BaseException) must actually catch any
    # remaining failure mode here -- this is the property that made the
    # panic a real production hazard, not just an unhandled-exception nuisance.
    try:
        other_expr = x + 0.0
        other_expr += other_expr
    except Exception:
        pass


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


def _cgroup_cpu_quota(cgroup_root: Path = Path("/sys/fs/cgroup")) -> float | None:
    """Effective CPU budget from the CFS bandwidth controller (cgroup v2
    `cpu.max`, falling back to cgroup v1 `cpu.cfs_quota_us`/
    `cpu.cfs_period_us`), or `None` if unconstrained/unreadable.

    `os.sched_getaffinity` reports which cores the process *may run on*, not
    a fractional throughput cap: a container started with e.g. Docker's
    `--cpus=1` on a multi-core host still reports every host core in its
    affinity set (cpuset is untouched) while CFS bandwidth control throttles
    it to one core's worth of wall-clock CPU time. Affinity alone would pass
    the `>= 2` gate below on such a host and then fail on GIL-release timing
    that assumes genuine parallelism.

    `cgroup_root` defaults to the real mount point; tests override it with a
    `tmp_path`-backed fake layout rather than monkeypatching `Path` itself.
    """
    v2 = cgroup_root / "cpu.max"
    if v2.exists():
        try:
            max_str, period_str = v2.read_text().split()
            return None if max_str == "max" else int(max_str) / int(period_str)
        except (OSError, ValueError, ZeroDivisionError):
            return None

    quota_path = cgroup_root / "cpu" / "cpu.cfs_quota_us"
    period_path = cgroup_root / "cpu" / "cpu.cfs_period_us"
    if quota_path.exists() and period_path.exists():
        try:
            quota = int(quota_path.read_text())
            period = int(period_path.read_text())
            return None if quota <= 0 else quota / period
        except (OSError, ValueError, ZeroDivisionError):
            return None

    return None


def _available_cpu_count() -> int:
    """`os.sched_getaffinity` (Linux-only: honors taskset/cpuset CPU
    restrictions, unlike `os.cpu_count()`, which reports the host's total
    core count even inside a constrained container) with an
    `os.cpu_count()` fallback for platforms where `sched_getaffinity`
    doesn't exist (e.g. macOS), further capped by the cgroup CFS quota (see
    `_cgroup_cpu_quota`) since affinity alone misses a fractional/whole
    `--cpus` throughput limit that leaves the affinity set untouched."""
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity_count = os.cpu_count() or 1

    quota = _cgroup_cpu_quota()
    if quota is None:
        return affinity_count
    # floor, never below 1: a quota below 1 whole CPU still schedules, just
    # never concurrently, so `max(1, ...)` -- not `0` -- is the honest floor.
    return min(affinity_count, max(1, int(quota)))


def test_cgroup_cpu_quota_v2_max_means_unconstrained(tmp_path):
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert _cgroup_cpu_quota(tmp_path) is None


@pytest.mark.parametrize(
    ("contents", "expected"),
    [("100000 100000\n", 1.0), ("50000 100000\n", 0.5), ("150000 100000\n", 1.5)],
)
def test_cgroup_cpu_quota_v2_parses_quota_over_period(tmp_path, contents, expected):
    (tmp_path / "cpu.max").write_text(contents)
    assert _cgroup_cpu_quota(tmp_path) == pytest.approx(expected)


def test_cgroup_cpu_quota_v1_fallback_when_no_v2_file(tmp_path):
    v1_dir = tmp_path / "cpu"
    v1_dir.mkdir()
    (v1_dir / "cpu.cfs_quota_us").write_text("200000\n")
    (v1_dir / "cpu.cfs_period_us").write_text("100000\n")
    assert _cgroup_cpu_quota(tmp_path) == pytest.approx(2.0)


def test_cgroup_cpu_quota_v1_quota_minus_one_means_unconstrained(tmp_path):
    v1_dir = tmp_path / "cpu"
    v1_dir.mkdir()
    (v1_dir / "cpu.cfs_quota_us").write_text("-1\n")
    (v1_dir / "cpu.cfs_period_us").write_text("100000\n")
    assert _cgroup_cpu_quota(tmp_path) is None


def test_cgroup_cpu_quota_none_when_neither_file_exists(tmp_path):
    assert _cgroup_cpu_quota(tmp_path) is None


def test_available_cpu_count_capped_by_quota_even_with_wide_affinity(monkeypatch):
    """The exact regression this fixes: a host reporting >= 2 cores of
    affinity but a cgroup quota of 1 CPU (the Docker `--cpus=1` case in the
    module docstring) must report 1, not the affinity count -- a quota=1 CI
    runner would otherwise pass `test_solve_releases_the_gil`'s `< 2` skip
    gate and then fail on GIL-release timing that assumes real parallelism.

    Sentinel: `_available_cpu_count` returning bare `affinity_count` (no
    quota capping at all, i.e. the pre-fix behavior) makes this assert 4
    instead of 1 -- confirmed by reverting and re-running.
    """
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(4)), raising=False)
    monkeypatch.setattr(sys.modules[__name__], "_cgroup_cpu_quota", lambda: 1.0)
    assert _available_cpu_count() == 1


def test_available_cpu_count_unaffected_when_quota_unconstrained(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(4)), raising=False)
    monkeypatch.setattr(sys.modules[__name__], "_cgroup_cpu_quota", lambda: None)
    assert _available_cpu_count() == 4


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
    also read ratio ~1.0 -- not a real failure of `Python::detach`, but a
    physical impossibility of measuring its effect this way on that host.
    Skipped rather than asserted around, below.
    """
    if _available_cpu_count() < 2:
        pytest.skip(
            "test_solve_releases_the_gil needs >=2 available CPU cores to "
            "distinguish GIL-released from GIL-held (both read ratio ~1.0 "
            "on a single core regardless of Python::detach)"
        )
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


@pytest.mark.skipif(
    not hasattr(signal, "SIGINT"), reason="SIGINT is not available on this platform"
)
def test_sigint_interrupts_long_solve():
    """Ctrl-C during a `timeout_secs`-unbounded `solve()` must raise
    `KeyboardInterrupt` promptly, not only after the solve finishes on its
    own -- the actual bug this test guards against (Codex PR #31 review, P1).

    Root cause was two layers, not one:
    1. `Model.solve()` used to run the entire blocking Rust computation under
       `Python::detach` on the *same* thread that would otherwise process
       the signal. Releasing the GIL there lets *other* Python threads make
       progress, but does nothing for `KeyboardInterrupt` on a script's own
       main thread, since `Python::check_signals`/the default SIGINT handler
       only fire when the interpreter actually gets to run Python bytecode
       -- which a `detach`-for-the-whole-solve design never gave it the
       chance to do. Fixed in `otspot-py/src/model.rs`: `solve()` now runs
       the computation on a worker thread (`std::thread::scope`) while the
       calling thread polls `Python::check_signals` with an
       exponentially-backed-off interval, setting `Model.set_cancel_flag`'s
       flag on a caught signal.
    2. `otspot-core`'s Farkas-infeasibility-certificate verification
       (`dual_advanced::phase1::farkas_infeasibility_certified` and
       `primal::extract_farkas_certificate`) each ran a `for row in
       art_rows` loop -- one BTRAN solve + an O(n) certificate check per
       still-basic artificial row -- that checked neither `deadline` nor
       `cancel_flag` internally, only at entry. Cancelling early (the exact
       scenario Ctrl-C produces: Phase I bails on its first iteration,
       leaving nearly every artificial basic, i.e. `art_rows.len()` close to
       `m`) went unnoticed until that whole O(m) loop ran to completion on
       its own. Fixed in `otspot-core/src/simplex/dual_advanced/phase1.rs`
       and `otspot-core/src/simplex/primal/mod.rs` by checking
       `options.external_stop_requested()` inside the loop.

       This is a *debug-build-specific* fix, not something this Python-level
       test can demonstrate on its own: the O(m) Farkas loop is cheap enough
       at `opt-level=3` (this test's own build, and any `--release` wheel)
       that even *without* the fix, SIGINT here was honored in ~0.130s --
       fast, not the multi-second stall this fixes in a `dev`-profile
       build. The dedicated sentinel for fix #2
       (`otspot-core::lp::tests::farkas_certificate_probe_loop_honors_cancel_flag_preset`,
       n=6000) reverts the fix and re-runs it *under `cargo test`'s own
       `opt-level=3` profile* specifically so it reproduces there (1.9s
       reverted vs 0.05s fixed); this test's job is only to prove fix #1
       (the worker-thread/poll-loop redesign) end to end.

    n=2000 (not `_build_gil_probe_model`'s default 1200) with a 0.1s send
    delay: chosen for margin, not just correctness. At n=1200 this test's
    natural (uninterrupted) release-build completion time is only ~0.145s --
    a mere ~31ms/24% past the 0.1s send delay, thin enough that a faster
    machine than this one could plausibly finish the solve *before* the
    signal is even processed, failing the `pytest.raises(KeyboardInterrupt)`
    below and (worse) leaving the not-yet-delivered SIGINT to surface as a
    stray `KeyboardInterrupt` in whatever runs next -- guarded against
    directly (not just by margin) via `cancel_send` below. n=2000 measured
    ~0.337s natural completion (release) against the same 0.1s delay --
    ~237ms/237% margin -- while `KeyboardInterrupt` itself still arrives in
    ~0.108-0.111s (12/12 trials, release) since the poll loop's response
    latency is governed by its backoff ramp-up, not by problem size. Debug
    build: ~0.457-0.472s (5/5 trials) to `KeyboardInterrupt`, still
    orders of magnitude under n=2000's natural debug completion time
    (~59s, extrapolated from `_build_gil_probe_model`'s own docstring
    scaling data), so no analogous margin concern there. The 2s bound below
    comfortably covers both build-profile regimes with headroom; the
    `KeyboardInterrupt` assertion (not just the timing bound) remains the
    primary correctness check.
    """
    model = _build_gil_probe_model(2000)
    # Set (from a `finally`, before `sender.join()`) the moment `solve()`
    # returns *for any reason* -- the expected `KeyboardInterrupt`, some
    # other exception, or (the exact regression this guards against) solving
    # to completion without raising at all before the sender's 0.1s delay
    # elapses. Without this, an early finish leaves `sender` still asleep;
    # by the time it wakes and fires `os.kill`, this test's `with`/`finally`
    # scope has already exited, so the SIGINT lands on whatever pytest runs
    # next (teardown, or an unrelated subsequent test) instead of here.
    cancel_send = threading.Event()

    def send_sigint() -> None:
        time.sleep(0.1)
        if not cancel_send.is_set():
            os.kill(os.getpid(), signal.SIGINT)

    sender = threading.Thread(target=send_sigint, daemon=True)
    t0 = time.perf_counter()
    sender.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            model.solve()
        elapsed = time.perf_counter() - t0
    finally:
        cancel_send.set()
        sender.join()

    assert elapsed < 2.0, (
        f"KeyboardInterrupt took {elapsed:.3f}s to arrive after SIGINT was sent "
        "at ~0.1s -- solve() is not honoring Ctrl-C promptly"
    )


@pytest.mark.skipif(
    not hasattr(signal, "SIGINT"), reason="SIGINT is not available on this platform"
)
def test_solve_join_after_sigint_releases_the_gil():
    """`test_sigint_interrupts_long_solve` above proves `KeyboardInterrupt`
    arrives promptly; it does not prove *other Python threads keep running*
    while `solve()` waits for that to happen. The two are different claims:
    the poll loop's own `py.detach` around its condvar wait (verified by
    `test_solve_releases_the_gil`, unchanged by this fix) only covers time
    *between* polls -- it says nothing about the `handle.join()` that runs
    right after a caught SIGINT, which previously ran with the GIL still
    held (Codex PR #31 review, found via lead's direct read: existing tests
    only ever looked at the normal-completion side of GIL release). Held
    there, every other Python thread freezes until the cancelled worker
    reaches its own next internal cancel-flag check point -- a real,
    measurable stretch for a still-converging solve, not a negligible detail.

    Supersedes an earlier version of this test (deleted, Codex PR #31
    follow-up review round 2, P1) that sampled its "before" counter value
    from the *sending* thread right before `os.kill()`. That anchor is too
    early: SIGINT can land at any point in the poll loop's own cycle, and
    the pending signal is not actually processed until the next
    `Python::check_signals` call, up to `SIGNAL_POLL_INTERVAL_MAX` (10ms)
    later -- during which the loop's ordinary, unrelated `py.detach` around
    `Condvar::wait_timeout` may *already* be releasing the GIL, letting the
    counter advance regardless of whether the post-signal `handle.join()`
    is fixed or reverted. That confound was measured directly: with the
    join bug reverted, `os.kill`-anchored sampling showed a bimodal
    ~3.6ms/~13.4ms split in signal-to-interrupt latency across trials
    (~10ms apart, exactly `SIGNAL_POLL_INTERVAL_MAX`), and the old test
    falsely passed (missed the reverted bug) in 8/10 pytest runs and 4/10
    bare-script runs.

    Fixed by anchoring the "before" sample inside a custom `SIGINT` handler
    installed via `signal.signal`, which runs synchronously *inside*
    `Python::check_signals`'s own signal dispatch -- i.e. at the exact
    instant `solve()`'s poll loop notices the pending signal, not some
    unknown and possibly much earlier time before it. Between that instant
    and `KeyboardInterrupt` reaching this test, `solve()` does only
    `cancel.store(...)` (negligible) and `py.detach(|| handle.join())`, so
    `progressed` below isolates (almost) exactly the join. Reverting the
    join fix therefore leaves the counting thread unable to acquire the GIL
    for that whole stretch, collapsing `progressed` to (near) 0
    deterministically, rather than depending on where in the poll cycle the
    signal happened to land.
    """
    model = _build_gil_probe_model(2000)
    counter = {"n": 0}
    stop_counting = threading.Event()

    def count_forever() -> None:
        while not stop_counting.is_set():
            counter["n"] += 1

    counter_thread = threading.Thread(target=count_forever, daemon=True)
    counter_thread.start()

    cancel_send = threading.Event()
    count_at_signal: dict[str, int] = {}
    original_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum: int, frame: object) -> None:
        count_at_signal["n"] = counter["n"]
        raise KeyboardInterrupt

    def send_sigint() -> None:
        time.sleep(0.1)
        if not cancel_send.is_set():
            os.kill(os.getpid(), signal.SIGINT)

    signal.signal(signal.SIGINT, handle_sigint)
    sender = threading.Thread(target=send_sigint, daemon=True)
    sender.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            model.solve()
        count_after_interrupt = counter["n"]
    finally:
        cancel_send.set()
        signal.signal(signal.SIGINT, original_handler)
        sender.join()
        stop_counting.set()
        counter_thread.join()

    assert "n" in count_at_signal, (
        "custom SIGINT handler never ran -- solve() never observed the signal"
    )
    progressed = count_after_interrupt - count_at_signal["n"]
    assert progressed > 5000, (
        f"background counting thread advanced by only {progressed} between "
        "check_signals() noticing SIGINT and KeyboardInterrupt being caught "
        "-- expected tens of thousands of increments if the GIL was "
        "genuinely released throughout the post-signal handle.join()"
    )


def _build_tiny_lp() -> otspot.Model:
    model = otspot.Model("tiny")
    x = model.add_var("x", 0.0, 10.0)
    y = model.add_var("y", 0.0, 10.0)
    model.add_constraint((x + y).geq(3.0))
    model.minimize(x + 2.0 * y)
    return model


def test_solve_poll_backoff_does_not_floor_small_solve_latency():
    """`Model.solve()`'s `Python::check_signals` poll loop used to sleep a
    flat 10ms before its very first `is_finished`/`check_signals` check,
    putting a ~10ms latency *floor* under every `solve()` call no matter how
    fast the underlying LP actually was (Codex PR #31 review, P2): a
    trivial 2-variable LP that solves in ~0.2ms on its own measured 10.2ms
    end to end through `solve()` -- a ~60x slowdown -- reproducibly across
    200 repeated calls.

    Fixed with an adaptive backoff (`SIGNAL_POLL_INTERVAL_INITIAL` doubling
    via `SIGNAL_POLL_BACKOFF_FACTOR` up to `SIGNAL_POLL_INTERVAL_MAX`, see
    `otspot-py/src/model.rs`): the first poll is far shorter than the old
    flat interval, so a solve that finishes before the loop has ramped up
    to a coarse polling cadence is observed almost immediately, while a
    long solve still settles into the same coarse cadence (and the same
    `KeyboardInterrupt` latency) as before.

    Sentinel: median latency across 200 repeated tiny solves measured
    ~0.96ms with the backoff alone, improving further to ~0.68ms once
    `solve()` also switched its poll wait from a plain `thread::sleep` to
    `Condvar::wait_timeout` (see `test_solve_condvar_wakes_immediately_
    on_completion` below for the fix that specifically targets) -- a
    regression back to a flat 10ms floor would push the median to ~10ms+.

    7ms bound, not 5ms (Codex PR #31 audit P3: a fixed millisecond ceiling
    against a wall-clock measurement is inherently machine-dependent) --
    widened for headroom over a CI wheel build's thread-spawn/GIL-detach
    overhead, while still sitting clearly below the ~10ms a regression back
    to a flat poll interval would produce. Not rewritten as a ratio or a
    deterministic poll-count check: this is a 200-trial *median*, already
    resistant to the occasional-outlier-scheduling-hiccup failure mode a
    single wall-clock measurement (like the since-fixed `qp_phase2.rs` one)
    is vulnerable to, and `solve()`'s poll cadence is otspot-py-internal --
    no counter is exposed across the FFI boundary to assert against instead.
    """
    n = 200
    times: list[float] = []
    for _ in range(n):
        model = _build_tiny_lp()
        t0 = time.perf_counter()
        result = model.solve()
        times.append(time.perf_counter() - t0)
        assert isinstance(result.status, otspot.SolveStatus.Optimal)

    times.sort()
    median = times[n // 2]
    assert median < 0.007, (
        f"median solve() latency over {n} trivial solves was {median * 1000:.3f}ms "
        "-- expected sub-millisecond-scale, not a coarse fixed poll-interval floor "
        "(regression back to a flat SIGNAL_POLL_INTERVAL would read ~10ms here)"
    )


def _build_medium_lp(n: int = 20) -> otspot.Model:
    """Chain LP, small enough that the natural solve time (~10-20ms, this
    machine) straddles several `SIGNAL_POLL_INTERVAL_INITIAL` backoff
    doublings without reaching `SIGNAL_POLL_INTERVAL_MAX` -- exactly the
    "worker finished mid-ramp-up" case `Condvar::wait_timeout` targets. Not
    `_build_tiny_lp` (finishes before the first poll matters) or
    `_build_gil_probe_model` (finishes well after backoff has already
    reached its cap, where a plain-sleep design's *last* interval is the
    same 10ms `Condvar` would also cap out at)."""
    model = otspot.Model("medium")
    variables = [model.add_var(f"x{i}", 0.0, 10.0) for i in range(n)]
    obj: otspot.Variable | otspot.Expression | otspot.QuadExpr = variables[0]
    for i in range(1, n):
        obj = obj + float(i % 7 + 1) * variables[i]
    model.minimize(obj)
    for i in range(n - 1):
        model.add_constraint((variables[i] + variables[i + 1]).geq(float((i % 5) + 1)))
    return model


def test_solve_condvar_wakes_immediately_on_completion():
    """`Model.solve()`'s poll loop used to `thread::sleep(poll_interval)`
    unconditionally between polls: a worker that finished *during* that
    sleep still wasn't noticed until the sleep ran out, adding up to a full
    poll interval (up to `SIGNAL_POLL_INTERVAL_MAX` = 10ms once backoff has
    ramped up) of pure dead latency on top of an already-finished solve --
    worst for solves whose natural runtime lands mid-ramp-up, neither so
    fast the first short poll catches them nor so slow the eventual 10ms
    cadence is negligible next to the total (Codex PR #31 review, P2
    follow-up). Fixed by waiting on a `Condvar` the worker thread notifies
    the instant it finishes (`Condvar::wait_timeout`, see `otspot-py/src/
    model.rs`), so completion is observed immediately regardless of which
    poll interval is currently in effect.

    Sentinel: `_build_medium_lp()` (n=20) measured median 23.5ms end to end
    with a plain-`thread::sleep` poll wait vs. 14.0ms with `Condvar::
    wait_timeout` (10 trials each, this machine, debug build) -- the ~9.5ms
    difference is this fix's effect, not solve-time variance (both regimes
    solve the identical problem). Confirmed by reverting to `thread::sleep`
    and re-measuring (23.5ms, over the bound below).

    21ms bound, not 20ms (Codex PR #31 audit P3, same rationale as the 5ms
    -> 7ms widening above): ~1.5x the fixed design's 14.0ms, giving more
    headroom for a CI wheel build's machine variance, while staying below
    the reverted design's 23.5ms so a regression back to plain
    `thread::sleep` still fails here. Not rewritten as a deterministic
    poll-count check for the same reason as above: `solve()`'s wait
    primitive is otspot-py-internal, and this 10-trial median is already
    more contention-resistant than a single wall-clock sample.
    """
    n = 10
    times: list[float] = []
    for _ in range(n):
        model = _build_medium_lp()
        t0 = time.perf_counter()
        result = model.solve()
        times.append(time.perf_counter() - t0)
        assert isinstance(result.status, otspot.SolveStatus.Optimal)

    times.sort()
    median = times[n // 2]
    assert median < 0.021, (
        f"median solve() latency over {n} medium (n=20) solves was "
        f"{median * 1000:.3f}ms -- expected close to the solve's own natural "
        "time (~14ms, this machine), not inflated by a full dead poll "
        "interval on top (regression back to plain thread::sleep would read "
        "~23.5ms here)"
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
        # `restored == value` now covers every type here (SolveStatus/Tolerance
        # gained `#[pyclass(eq)]`); this branch is kept as a fallback so a
        # future complex-enum addition without `eq` still passes via
        # `__reduce__`'s own payload instead of failing this test outright.
        type(restored) is type(value) and restored.__reduce__()[1] == value.__reduce__()[1]
    )


def test_solve_status_and_tolerance_compare_by_value_not_identity():
    """`SolveStatus`/`Tolerance` are PyO3 "complex enums" (payload-carrying
    variants become subclasses); before `#[pyclass(eq)]` neither had a value
    comparison, so `==` fell back to Python's default object-identity check.
    `ModelResult.status` in particular builds a *fresh* wrapper on every
    property access (see `result.rs`'s getter), so `result.status ==
    otspot.SolveStatus.Optimal()` was always `False` even for a genuinely
    Optimal result (Codex PR #31 review).
    """
    model = otspot.Model("solve_status_eq")
    x = model.add_var("x", 0.0, 10.0)
    model.minimize(x)
    result = model.solve()
    assert isinstance(result.status, otspot.SolveStatus.Optimal)

    # Two independent `.status` accesses are distinct Python objects (a fresh
    # wrapper each time) but must still compare equal by value.
    first_access = result.status
    second_access = result.status
    assert first_access is not second_access, (
        "sanity: ModelResult.status must build a fresh wrapper per access -- "
        "otherwise this test cannot distinguish value equality from identity"
    )
    assert first_access == second_access
    assert result.status == otspot.SolveStatus.Optimal()

    # Cross-variant and payload (NonConvex) comparisons.
    assert otspot.SolveStatus.Optimal() != otspot.SolveStatus.Infeasible()
    assert otspot.SolveStatus.NonConvex("indefinite Q") == otspot.SolveStatus.NonConvex(
        "indefinite Q"
    )
    assert otspot.SolveStatus.NonConvex("a") != otspot.SolveStatus.NonConvex("b")

    assert otspot.Tolerance.Medium() == otspot.Tolerance.Medium()
    assert otspot.Tolerance.Medium() != otspot.Tolerance.Fast()
    assert otspot.Tolerance.Custom(1e-7) == otspot.Tolerance.Custom(1e-7)
    assert otspot.Tolerance.Custom(1e-7) != otspot.Tolerance.Custom(1e-6)


def test_solve_status_is_hashable_tolerance_is_not():
    """Adding `#[pyclass(eq)]` to `SolveStatus` (previous fix, see the test
    above) silently made it unhashable: CPython nulls `tp_hash` whenever
    `tp_richcompare` (`__eq__`) is set unless a hash is explicitly supplied,
    and PyO3 only emits `__hash__` under `#[pyclass(hash)]` -- confirmed by
    the reviewer via a wheel A/B (`hash()` worked before the `eq`-only fix,
    raised `TypeError` after). `SolveStatus` now also declares `hash`, so
    it is usable as a `set` member / `dict` key again, hashing by variant
    *and* payload (not just the tag). `Tolerance` cannot follow suit
    (`Custom(f64)`, and `f64` has no `Hash` impl) and stays unhashable by
    design for every variant, not just `Custom`.
    """
    status_set = {otspot.SolveStatus.Optimal(), otspot.SolveStatus.Optimal()}
    assert len(status_set) == 1, "equal SolveStatus values must hash equal (set dedup)"

    status_dict = {otspot.SolveStatus.Infeasible(): "reason"}
    assert status_dict[otspot.SolveStatus.Infeasible()] == "reason"

    payload_set = {
        otspot.SolveStatus.NonConvex("msg"),
        otspot.SolveStatus.NonConvex("msg"),
        otspot.SolveStatus.NonConvex("other"),
    }
    assert len(payload_set) == 2, (
        "the payload must participate in the hash, not just the variant tag "
        "-- otherwise NonConvex('msg') and NonConvex('other') would collide "
        "into the same set entry"
    )

    with pytest.raises(TypeError):
        hash(otspot.Tolerance.Medium())
    with pytest.raises(TypeError):
        hash(otspot.Tolerance.Custom(1e-6))
