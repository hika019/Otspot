import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "encapsulation", ROOT / "scripts/check_csc_encapsulation.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class EncapsulationGateTests(unittest.TestCase):
    def test_counts_fields_but_not_accessors_or_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "otspot-core/src/x.rs"
            path.parent.mkdir(parents=True)
            path.write_text(
                "fn f(a: &M) { let _ = a.col_ptr[0]; let _ = a.values(); }\n"
                "mod tests { fn t(a: &M) { let _ = a.row_ind[0]; } }\n"
            )
            self.assertEqual(gate.counts(root), {"otspot-core/src/x.rs": 1})

    def test_new_file_access_has_zero_baseline_allowance(self):
        failures = gate.violations({"otspot-core/src/new.rs": 1}, {})
        self.assertEqual(failures, ["otspot-core/src/new.rs: 1 > baseline 0"])

    def test_reduction_is_allowed_but_growth_fails(self):
        baseline = {"a.rs": 3}
        self.assertEqual(gate.violations({"a.rs": 2}, baseline), [])
        self.assertTrue(gate.violations({"a.rs": 4}, baseline))


if __name__ == "__main__":
    unittest.main()
