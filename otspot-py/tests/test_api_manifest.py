"""Python-side half of the API parity guarantee (see api_manifest.json and
tests/api_manifest_rust.rs for the Rust-side half).

Every check here compares the *built* `otspot` extension module against
api_manifest.json via introspection, in both directions:
  - manifest entry -> must exist in the Python module (nothing promised but
    missing)
  - Python module's public surface -> must be represented in the manifest
    (nothing exposed but undocumented)
"""

import json
from pathlib import Path

import pytest

import otspot

MANIFEST = json.loads((Path(__file__).parent.parent / "api_manifest.json").read_text())

# `object` is the one baseline that can never be silently widened by a future
# change to this crate's own classes (unlike deriving the baseline from one
# of our own pyclasses, e.g. `Constraint`: adding a method to `Constraint`
# would have silently loosened the check for every *other* class too, since
# they all diffed against the same live, mutable reference).
STANDARD_ATTRS = frozenset(dir(object))
# Verified empirically (not `dir(object)`): every PyO3 `#[pyclass]` carries
# `__module__` beyond plain `object`, regardless of what methods it defines.
PYO3_CLASS_EXTRAS = frozenset({"__module__"})
# `#[pyclass(eq, eq_int)]` (VarKind/SolutionProof/SolveError) additionally
# attaches `__int__` for the C-like int cast.
PYO3_EQ_INT_EXTRAS = PYO3_CLASS_EXTRAS | {"__int__"}

CLASS_NAMES_WITH_METHODS = [
    "Model",
    "Variable",
    "Expression",
    "QuadExpr",
    "ModelResult",
    "Constraint",
]


def _split_python_field(field: str) -> tuple[str, list[str]]:
    """'Variable.__add__/__radd__' -> ('Variable', ['__add__', '__radd__'])."""
    cls_name, _, rest = field.partition(".")
    return cls_name, rest.split("/")


def _manifested_method_names(cls_name: str) -> set[str]:
    names: set[str] = set()
    for entry in MANIFEST["methods"].get(cls_name, []):
        _, methods = _split_python_field(entry["python"])
        names.update(methods)
    names.discard("__init__")  # constructor: always present, not a "method" in dir()
    return names


# ---------------------------------------------------------------------------
# manifest -> Python (nothing promised but missing)
# ---------------------------------------------------------------------------


def test_manifest_types_exist_in_python():
    for entry in MANIFEST["types"]:
        py_name = entry["python"].removeprefix("otspot.")
        assert hasattr(otspot, py_name), (
            f"manifest promises otspot.{py_name} (rust: {entry['rust']}) but it is missing"
        )


def test_manifest_exceptions_exist_in_python():
    assert hasattr(otspot, "OtspotError")
    assert issubclass(otspot.OtspotError, Exception)
    for entry in MANIFEST["model_error_exceptions"]["entries"]:
        py_name = entry["python"].removeprefix("otspot.")
        cls = getattr(otspot, py_name, None)
        assert cls is not None, f"manifest promises otspot.{py_name} but it is missing"
        assert issubclass(cls, otspot.OtspotError), (
            f"otspot.{py_name} must subclass otspot.OtspotError (maps {entry['rust_variant']})"
        )


@pytest.mark.parametrize("cls_name", CLASS_NAMES_WITH_METHODS)
def test_manifest_methods_exist_in_python(cls_name):
    cls = getattr(otspot, cls_name)
    for name in _manifested_method_names(cls_name):
        assert hasattr(cls, name), (
            f"manifest promises {cls_name}.{name} but it is missing"
        )


def test_manifest_variants_exist_in_python():
    for enum_name, variants in MANIFEST["variants"].items():
        cls = getattr(otspot, enum_name)
        for variant in variants:
            assert hasattr(cls, variant), f"manifest promises {enum_name}.{variant} but it is missing"


def test_manifest_solve_failed_error_has_error_attribute():
    """`SolveFailedError.error` is not a `dir()`-discoverable class attribute
    (it is set per-instance via `setattr` in errors.rs, not a `#[pyo3(get)]`
    field), so it needs its own instance-level check rather than folding into
    `test_python_class_has_no_undocumented_public_methods` below."""
    model = otspot.Model("infeasible_for_manifest_check")
    x = model.add_var("x", 0.0, 1.0)
    model.add_constraint(x.geq(5.0))
    model.minimize(x)
    with pytest.raises(otspot.SolveFailedError) as exc_info:
        model.solve()
    assert exc_info.value.error == otspot.SolveError.Infeasible


# ---------------------------------------------------------------------------
# Python -> manifest (nothing exposed but undocumented)
# ---------------------------------------------------------------------------


def test_python_module_has_no_undocumented_public_types():
    manifested = {e["python"].removeprefix("otspot.") for e in MANIFEST["types"]}
    manifested |= {
        e["python"].removeprefix("otspot.") for e in MANIFEST["model_error_exceptions"]["entries"]
    }
    manifested.add("OtspotError")

    module_public = {
        n
        for n in dir(otspot)
        if not n.startswith("_") and n != "otspot"  # self-reference some builds expose
    }
    module_public.discard("__version__")

    undocumented = module_public - manifested
    assert not undocumented, f"otspot module exposes undocumented public symbols: {sorted(undocumented)}"


@pytest.mark.parametrize("cls_name", CLASS_NAMES_WITH_METHODS)
def test_python_class_has_no_undocumented_public_methods(cls_name):
    """Exact-set comparison: every non-standard attribute on `cls` must be
    exactly the set the manifest declares for it (catches both directions at
    once for method-level parity). Adding an undocumented method to *any*
    class here is caught independently of every other class, since the
    baseline (`object`) can never itself grow such a method."""
    cls = getattr(otspot, cls_name)
    extra = set(dir(cls)) - STANDARD_ATTRS - PYO3_CLASS_EXTRAS
    manifested = _manifested_method_names(cls_name)
    assert extra == manifested, (
        f"{cls_name}: dir() extras {sorted(extra - manifested)} not in manifest, "
        f"manifest entries {sorted(manifested - extra)} not found on the class"
    )


@pytest.mark.parametrize("enum_name", list(MANIFEST["variants"].keys()))
def test_python_enum_has_no_undocumented_variants(enum_name):
    cls = getattr(otspot, enum_name)
    extra = set(dir(cls)) - STANDARD_ATTRS - PYO3_EQ_INT_EXTRAS
    manifested = set(MANIFEST["variants"][enum_name])
    manifested |= set(MANIFEST.get("python_only_variants", {}).get(enum_name, []))
    assert extra == manifested, (
        f"{enum_name}: dir() extras {sorted(extra - manifested)} not in manifest, "
        f"manifest entries {sorted(manifested - extra)} not found on the class"
    )


@pytest.mark.parametrize(
    "cls_name",
    [k for k in MANIFEST.get("baseline_shadowed_dunders", {}) if not k.startswith("_")],
)
def test_baseline_shadowed_dunders_are_actually_overridden(cls_name):
    """`__repr__`/`__eq__` etc. are already part of `dir(object)`, so
    `test_python_class_has_no_undocumented_public_methods` /
    `test_python_enum_has_no_undocumented_variants` (which only look at
    attributes *beyond* the `object` baseline) cannot see whether a class
    still has its custom override or has silently fallen back to the
    default (e.g. `Variable.__repr__` reverting to `object`'s identity
    repr). Verify each manifested override directly by identity."""
    cls = getattr(otspot, cls_name)
    for dunder in MANIFEST["baseline_shadowed_dunders"][cls_name]:
        assert hasattr(cls, dunder)
        assert getattr(cls, dunder) is not getattr(object, dunder), (
            f"{cls_name}.{dunder} is object's default implementation -- "
            "the custom override appears to have been removed"
        )


def test_constraint_sense_is_not_bound():
    """Sentinel for api_manifest.json's out_of_scope entry: if a future
    change accidentally reintroduces a `ConstraintSense` binding without
    updating the manifest, this catches it (`test_python_module_has_no_
    undocumented_public_types` would too, but this pins the specific,
    previously-real name so its removal is not silently forgotten)."""
    assert not hasattr(otspot, "ConstraintSense")
    assert not hasattr(otspot, "NotSupportedError")
