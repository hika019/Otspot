import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("arch", ROOT / "scripts/check_architecture.py")
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)


class ArchitectureGateTests(unittest.TestCase):
    def fixture(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        for crate, deps in {"otspot-num": "", "otspot-ir": 'otspot-num = { path = "../otspot-num" }', "otspot-core": ""}.items():
            (root / crate / "src").mkdir(parents=True)
            (root / crate / "Cargo.toml").write_text(f"[package]\nname='{crate}'\nversion='0.1.0'\n[dependencies]\n{deps}\n")
        owners = {
            "otspot-num/src/sparse/csc.rs": "pub struct CscMatrix;",
            "otspot-num/src/sparse/vec.rs": "pub struct SparseVec;",
            "otspot-num/src/sparse/view.rs": "pub trait CscMatrixView {}",
            "otspot-num/src/kkt.rs": "pub trait KktBackend {}",
            "otspot-ir/src/problem.rs": "pub struct OptimizationProblem;",
            "otspot-ir/src/solver.rs": "pub trait Solver {}",
        }
        for name, text in owners.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        (root / "otspot-num/src/lib.rs").write_text("")
        (root / "otspot-ir/src/lib.rs").write_text("")
        (root / "otspot-core/src/linalg.rs").write_text("pub use otspot_num::linalg::*;")
        (root / "otspot-core/src/sparse").mkdir()
        (root / "otspot-core/src/sparse/mod.rs").write_text("pub use otspot_num::sparse::*;")
        return tmp, root

    def test_valid_shape_passes(self):
        tmp, root = self.fixture()
        self.addCleanup(tmp.cleanup)
        self.assertEqual(arch.check(root), [])

    def test_reverse_dependency_fails(self):
        tmp, root = self.fixture()
        self.addCleanup(tmp.cleanup)
        (root / "otspot-num/Cargo.toml").write_text("[package]\nname='otspot-num'\nversion='0.1.0'\n[dependencies]\notspot-core={path='../otspot-core'}\n")
        self.assertTrue(any("forbidden dependencies" in x for x in arch.check(root)))

    def test_duplicate_owner_fails(self):
        tmp, root = self.fixture()
        self.addCleanup(tmp.cleanup)
        (root / "otspot-core/src/duplicate.rs").write_text("pub struct CscMatrix;")
        self.assertTrue(any("owner must be" in x for x in arch.check(root)))

    def test_legacy_implementation_directory_fails(self):
        tmp, root = self.fixture()
        self.addCleanup(tmp.cleanup)
        (root / "otspot-core/src/linalg").mkdir()
        self.assertTrue(any("legacy implementation" in x for x in arch.check(root)))


if __name__ == "__main__":
    unittest.main()
