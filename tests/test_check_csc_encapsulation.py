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

    def test_struct_literal_construction_is_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "otspot-core/src/build.rs"
            path.parent.mkdir(parents=True)
            path.write_text(
                "fn build() -> CscMatrix {\n"
                "    Some(CscMatrix {\n"
                "        col_ptr: vec![0],\n"
                "        row_ind: vec![],\n"
                "        values: vec![],\n"
                "        nrows: 0,\n"
                "        ncols: 0,\n"
                "    })\n"
                "}\n"
                "fn build_vec() -> SparseVec {\n"
                "    SparseVec { indices: vec![], values: vec![], len: 0 }\n"
                "}\n"
            )
            # 2 struct-literal constructions (CscMatrix + SparseVec); the
            # `-> CscMatrix {` return-type signature must NOT be counted.
            self.assertEqual(gate.counts(root), {"otspot-core/src/build.rs": 2})

    def test_struct_definition_and_impl_header_are_not_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "otspot-core/src/shape.rs"
            path.parent.mkdir(parents=True)
            path.write_text(
                "pub struct CscMatrix {\n"
                "    pub col_ptr: Vec<usize>,\n"
                "}\n"
                "impl CscMatrix {\n"
                "    fn nrows(&self) -> usize { 0 }\n"
                "}\n"
                "fn helper(m: &CscMatrix) -> CscMatrix {\n"
                "    m.clone()\n"
                "}\n"
            )
            self.assertEqual(gate.counts(root), {})

    def test_multiline_struct_literal_construction_is_counted(self):
        # rustfmt always keeps `Type {` joined on one physical line, but the
        # scanner must not silently miss a hand-edited (non-fmt-clean) file
        # that splits the brace onto the next line — `cargo fmt --check`
        # catches the formatting violation, but this gate must not depend on
        # that alone to see the construction.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "otspot-core/src/wrap.rs"
            path.parent.mkdir(parents=True)
            path.write_text(
                "fn build() -> CscMatrix {\n"
                "    let out = CscMatrix\n"
                "    {\n"
                "        col_ptr: vec![0],\n"
                "        row_ind: vec![],\n"
                "        values: vec![],\n"
                "        nrows: 0,\n"
                "        ncols: 0,\n"
                "    };\n"
                "    out\n"
                "}\n"
            )
            self.assertEqual(gate.counts(root), {"otspot-core/src/wrap.rs": 1})

    def test_multiline_return_type_signature_is_not_construction(self):
        # A wrapped `-> CscMatrix` return-type signature with the body brace
        # on the next line must not be misread as construction.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "otspot-core/src/sig.rs"
            path.parent.mkdir(parents=True)
            path.write_text(
                "fn helper(m: &CscMatrix)\n"
                "    -> CscMatrix\n"
                "{\n"
                "    m.clone()\n"
                "}\n"
            )
            self.assertEqual(gate.counts(root), {})

    def test_type_definition_directory_is_excluded_from_construction_scan(self):
        # Defensive: even if ROOTS ever grows to cover the type's own crate,
        # its defining module must stay exempt (`Self { .. }` there is the
        # legitimate constructor implementation, not a bypass).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "otspot-num/src/sparse/csc.rs"
            path.parent.mkdir(parents=True)
            path.write_text(
                "fn build() -> CscMatrix {\n"
                "    CscMatrix { col_ptr: vec![0], row_ind: vec![], values: vec![], nrows: 0, ncols: 0 }\n"
                "}\n"
            )
            original_roots = gate.ROOTS
            gate.ROOTS = original_roots + ("otspot-num/src",)
            try:
                self.assertEqual(gate.counts(root), {})
            finally:
                gate.ROOTS = original_roots


if __name__ == "__main__":
    unittest.main()
