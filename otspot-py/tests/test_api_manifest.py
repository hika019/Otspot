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

# `Constraint` has zero manifested methods of its own (see api_manifest.json),
# so its `dir()` is exactly what PyO3/CPython attach to every pyclass with no
# custom methods -- the "standard attribute" baseline every other class is
# diffed against below, computed live instead of hardcoded so it tracks
# whatever PyO3/CPython version is actually running the test.
STANDARD_ATTRS = frozenset(dir(otspot.Constraint))

CLASS_NAMES_WITH_METHODS = ["Model", "Variable", "Expression", "QuadExpr", "ModelResult"]


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
    once for method-level parity)."""
    cls = getattr(otspot, cls_name)
    extra = set(dir(cls)) - STANDARD_ATTRS
    manifested = _manifested_method_names(cls_name)
    assert extra == manifested, (
        f"{cls_name}: dir() extras {sorted(extra - manifested)} not in manifest, "
        f"manifest entries {sorted(manifested - extra)} not found on the class"
    )


@pytest.mark.parametrize("enum_name", list(MANIFEST["variants"].keys()))
def test_python_enum_has_no_undocumented_variants(enum_name):
    cls = getattr(otspot, enum_name)
    # `#[pyclass(eq, eq_int)]` (VarKind/ConstraintSense/SolutionProof/SolveError)
    # attaches `__int__` for the C-like int cast; it is a structural PyO3
    # feature, not a variant, so it is allowed alongside the STANDARD_ATTRS
    # baseline (which was computed from a plain, non-enum pyclass).
    extra = set(dir(cls)) - STANDARD_ATTRS - {"__int__"}
    manifested = set(MANIFEST["variants"][enum_name])
    assert extra == manifested, (
        f"{enum_name}: dir() extras {sorted(extra - manifested)} not in manifest, "
        f"manifest entries {sorted(manifested - extra)} not found on the class"
    )
