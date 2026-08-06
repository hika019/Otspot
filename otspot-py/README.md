# otspot (Python bindings)

Python bindings for [otspot](https://github.com/hika019/otspot) — LP (Revised
Simplex) and QP (Interior Point / Mehrotra predictor-corrector) — built with
[PyO3](https://pyo3.rs)/[maturin](https://www.maturin.rs).

The Python API mirrors the Rust `otspot_model` algebraic modeling API
symbol-for-symbol (same class/method/enum-variant names); see
[`api_manifest.json`](./api_manifest.json) for the tracked Rust<->Python
mapping and what is intentionally out of scope.

## Install

From a wheel (once published):

```bash
pip install otspot
```

From source (this directory):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop  # or: maturin build --release && pip install ../target/wheels/*.whl
# maturin build's default --out resolves through the workspace root (this crate
# is a workspace member, not its own crate root), landing wheels in
# ../target/wheels/, not ./target/wheels/ -- verified by running the command
# above from this directory. Pass --out dist to override the location.
```

### Building from source: prerequisites

Building requires a `python3` interpreter with its shared library
discoverable by [pyo3-build-config](https://docs.rs/pyo3-build-config).
`maturin build`/`develop` link this cdylib without `-lpython3.<minor>`
(feature `extension-module`, on by default — the *host* interpreter that
`dlopen`s the module supplies those symbols at import time). Running
`cargo test -p otspot-py --no-default-features` instead (a normal linked
Rust test binary, not a Python extension) does need to link
`-lpython3.<minor>` directly, which on some Debian/Ubuntu systems requires
the `python3-dev`/`libpython3.<minor>-dev` package. If that package is not
installed but a self-contained interpreter is available elsewhere (its
`sysconfig` `LIBPL` points at the actual `.so`, while `LIBDIR` does not), set:

```bash
RUSTFLAGS="-L $(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBPL"))')" \
  cargo test -p otspot-py --no-default-features
```

CI (`actions/setup-python`) ships a self-contained interpreter and does not
need this workaround.

## Usage

```python
import otspot

model = otspot.Model("production")
x = model.add_var("x", 0.0, float("inf"))
y = model.add_var("y", 0.0, 10.0)
model.add_constraint((2.0 * x + 3.0 * y).leq(12.0))
model.add_constraint((x + y).geq(3.0))
model.minimize(x + 2.0 * y)

result = model.solve()
print(result.status, result.objective_value, result.value(x), result[y])
```

Quadratic objectives use `Variable.pow2()` or `Variable * Variable`:

```python
model.minimize(x.pow2() + y.pow2())
```

Integer/binary variables route through the MILP/MIQP branch-and-bound solver
automatically:

```python
b = model.add_binary_var("b")
z = model.add_int_var("z", 0.0, 10.0)
```

### `==`/`<=`/`>=` are not overloaded on `Variable`/`Expression`

Unlike some other Python modeling libraries, `x <= 5.0` does **not** build a
constraint — Python has no equivalent of Rust's `constraint!` macro, so
constraint-building is an explicit method call instead:

```python
model.add_constraint(x.leq(5.0))     # not: model.add_constraint(x <= 5.0)
model.add_constraint((x + y).geq(3.0))
model.add_constraint(x.eq_constraint(5.0))
```

`x == 5.0` in particular does **not** raise — it silently evaluates to
`False` (default Python object-identity comparison), since `==` is not
overloaded. Arithmetic operators (`+`, `-`, `*`, unary `-`) *are* real
operator overloads and behave as expected.

### Errors

`Model.solve()` raises a subclass of `otspot.OtspotError` on failure instead
of returning a Rust-style `Result`:

```python
try:
    result = model.solve()
except otspot.SolveFailedError as e:
    print(e.error)  # otspot.SolveError.Infeasible / .Unbounded / ...
except otspot.NoObjectiveError:
    ...
```

See `api_manifest.json`'s `model_error_exceptions` for the full
`ModelError` variant -> exception class mapping.

`SolveFailedError.error` (and every `VarKind`/`SolutionProof`/`SolveError`/
`SolveStatus`/`Tolerance` value) supports `pickle`/`copy.deepcopy`, so
propagating a caught exception across a `multiprocessing` process boundary
works as expected.

### Threading and the GIL

`Model.solve()` runs with the GIL released (`Python::detach`): other Python
threads make progress during a solve, and a long or unbounded
(`timeout_secs` defaults to no limit) solve does not freeze
`KeyboardInterrupt` delivery for the rest of the process. Independent
`Model` instances can therefore solve concurrently on separate threads.

A single `Model` instance is **not** safe to call concurrently from multiple
threads: PyO3 pyclasses use runtime borrow checking, so e.g. two threads
both calling `.solve()` (or any other `&mut self` method) on the *same*
`Model` raise a "already borrowed"-style `RuntimeError` rather than racing
silently. Give each thread (or process) its own `Model`.

## Testing

```bash
maturin develop
pip install pytest
pytest tests/
```

`tests/test_api_manifest.py` and `tests/test_parity_lp_qp.py` are the
Python-side half of the API/behavior parity guarantee described in
`api_manifest.json`; `../tests/api_manifest_rust.rs` is the Rust-side half.
