import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gate", ROOT / "scripts/check_function_size.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class FunctionSizeTests(unittest.TestCase):
    def test_scanner_measures_function_and_ignores_test_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "otspot-core/src/x.rs"
            path.parent.mkdir(parents=True)
            path.write_text("fn production() {\n  if true {\n  }\n}\nmod tests {\nfn huge() {\n" + "x();\n" * 300 + "}\n}\n")
            rows = gate.functions(root)
            self.assertEqual(list(rows.values()), [4])

    def test_long_function_is_reported_by_ratchet_logic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "otspot-num/src/x.rs"
            path.parent.mkdir(parents=True)
            path.write_text("fn giant() {\n" + "work();\n" * 230 + "}\n")
            rows = gate.functions(root)
            self.assertGreater(next(iter(rows.values())), gate.LIMIT)


if __name__ == "__main__":
    unittest.main()
