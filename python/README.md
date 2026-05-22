# otspot Python bindings

Python bindings for the [otspot](https://github.com/hika019/otspot) optimization solver (LP / QP / MILP / MIQP).

## Install

```bash
pip install maturin
cd python/
python -m venv .venv && source .venv/bin/activate
maturin develop
```

## Quick start

### QP — minimize x² + y² − 4x − 4y, s.t. x + y ≤ 3, x,y ≥ 0

```python
import otspot

m = otspot.Model("qp")
x = m.add_var("x", lb=0)
y = m.add_var("y", lb=0)

m.set_diagonal_q([2, 2])          # Q = diag(2, 2); objective = ½xᵀQx + cᵀx
m.add_constraint(x + y <= 3)
m.minimize(-4*x + -4*y)

r = m.solve()
print(r.objective, r[x], r[y])    # -7.5  1.5  1.5
```

### LP — minimize x + 2y, s.t. 2x+3y ≤ 12, x+y ≥ 3, x ≥ 0, y ∈ [0,10]

```python
m = otspot.Model("lp")
x = m.add_var("x", lb=0)
y = m.add_var("y", lb=0, ub=10)

m.add_constraint(2*x + 3*y <= 12)
m.add_constraint(x + y >= 3)
m.minimize(x + 2*y)

r = m.solve()
print(r.objective, r[x], r[y])    # 3.0  3.0  0.0
```

### MILP — integer variables

```python
m = otspot.Model("milp")
x = m.add_int_var("x", lb=0, ub=5)
y = m.add_binary_var("y")         # integer in {0, 1}

m.add_constraint(x + y <= 3.5)
m.minimize(-x - 2*y)

r = m.solve()
print(r.objective, r[x], r[y])
```

## API reference

| Method | Description |
|--------|-------------|
| `Model(name)` | Create a new model |
| `add_var(name, lb, ub)` | Add a continuous variable (default bounds: −∞, +∞) |
| `add_int_var(name, lb, ub)` | Add an integer variable |
| `add_binary_var(name)` | Add a binary variable ({0,1}) |
| `add_constraint(c)` | Add a `Constraint` object |
| `minimize(expr)` | Set minimization objective |
| `maximize(expr)` | Set maximization objective |
| `set_diagonal_q(list)` | Set diagonal Q matrix for QP |
| `set_quadratic_objective(triplets, n)` | Set sparse Q from (i,j,v) list |
| `set_timeout(secs)` | Solver timeout in seconds |
| `set_threads(n)` | Parallel thread count |
| `set_presolve(flag)` | Enable/disable presolve (LP) |
| `set_tolerance(eps)` | Solver convergence tolerance |
| `solve()` | Returns `ModelResult` or raises an exception |

### Expression DSL

Variables and expressions support: `+`, `-`, `*` (scalar), unary `-`, and comparison operators `<=`, `>=`, `==` (which return `Constraint` objects).

```python
c1 = x + y <= 3          # Constraint
c2 = 2*x - y >= 1
c3 = (x + y) == 4        # equality constraint
```

### Exceptions

| Exception | Cause |
|-----------|-------|
| `InfeasibleError` | No feasible solution |
| `UnboundedError` | Problem is unbounded |
| `MaxIterationsError` | Solver hit iteration limit |
| `NumericalSolveError` | Numerical breakdown |
| `SolveTimeoutError` | Timeout reached |

All inherit from `OtspotError`.
