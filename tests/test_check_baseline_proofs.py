"""Self-tests for scripts/check_baseline_proofs.py (offline proof verifier).

Each detection layer has both a pass lineage and a fail lineage built from a
synthetic fixture (a real scalar QP, a real checker-saved witness). These fail
lineages are the sentinel set: a verifier mutation that ignores tampering must
turn one of them red (see /tmp/revert_proof_stage3b.sh for the measured
revert-proof runs).
"""
import contextlib
import gzip
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import baseline_certificate as checker
import highs_baseline_truth as truth  # noqa: E402  (shared sys.modules instance)
import check_baseline_proofs as verifier

# min 0.5*x^2 - 3*x over free x: exact optimum -9/2 at x = 3.
FIXTURE_QPS = """NAME          FIXTUREQP
ROWS
 N  OBJ
COLUMNS
    X         OBJ       -3.0
RHS
BOUNDS
 FR BND       X
QUADOBJ
    X         X         1.0
ENDATA
"""
# regenerate 用の独立生成スクリプト。liswet_exact_certificate.py と同じ CLI
# (--evidence-dir + 位置 QPS) を模擬し、同一問題から同一証跡を書く。
GENERATOR = """import hashlib, sys
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, {scripts!r})
import baseline_certificate as checker
import highs_baseline_truth as truth
argv = sys.argv[1:]
evidence = argv[argv.index("--evidence-dir") + 1]
qps = Path(argv[-1])
enc = checker.ExactModel(truth.load_qps(qps)).check([Fraction(3)], [Fraction(3)], [])
checker.save_evidence(Path(evidence) / (qps.stem + ".proof.json.gz"), enc, "-4.5",
                      hashlib.sha256(qps.read_bytes()).hexdigest(),
                      proposal={{"method": "synthetic"}})
"""
FAILING_GENERATOR = "import sys; sys.exit(3)\n"


class ProofVerifierTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "repo"
        (self.root / "proofs" / "liswet").mkdir(parents=True)
        data = self.root / "data"
        data.mkdir()
        self.csv = self.root / "baseline.csv"
        self.data_dir = data

        # A1: committed non-LISWET proof. L1: manifest-backed claim that the
        # working generator can regenerate. U1: labelled pre-contract value.
        qps = data / "A1.QPS"
        qps.write_text(FIXTURE_QPS)
        self.input_hash = hashlib.sha256(qps.read_bytes()).hexdigest()
        enc = checker.ExactModel(truth.load_qps(qps)).check(
            [checker.rational(3)], [checker.rational(3)], [])
        proof = self.root / "proofs" / "A1.proof.json.gz"
        checker.save_evidence(proof, enc, "-4.5", self.input_hash,
                              proposal={"method": "synthetic"})
        self.a1_stamp = verifier.normalized_hash(self.payload(proof))
        self.gen = Path(tmp.name) / "gen.py"
        self.gen.write_text(GENERATOR.format(scripts=str(ROOT / "scripts")))
        (data / "L1.QPS").write_text(FIXTURE_QPS)
        self.manifest = self.root / "proofs" / "liswet" / "manifest.json"
        self.write_manifest(args=[sys.executable, str(self.gen), "--evidence-dir",
                                  "{evidence_dir}", str(data / "L1.QPS")])
        self.csv.write_text(
            "problem_name,optimal_obj,source\n"
            f"A1,-4.5,{self.claim('proofs/A1.proof.json.gz', self.a1_stamp, checker.checker_digest())}\n"
            f"L1,-4.5,{self.claim(verifier.LISWET_MANIFEST, self.a1_stamp, '0' * 64)}\n"
            "U1,-4.5,claim=pre_contract_numeric_reference_not_certified;"
            "revalidation_psd_unknown_UNCERTIFIED\n")

    def claim(self, proof, stamp, checker_sha):
        return (f"{verifier.ENCLOSURE_TAG};proof={proof};proof_sha256={stamp};"
                f"input_sha256={self.input_hash};checker_sha256={checker_sha}")

    def write_manifest(self, args, stamp=None):
        self.manifest.write_text(json.dumps({"rows": {"L1": {
            "gaps": [1], "args": args, "command": " ".join(args),
            "proof_sha256": stamp or self.a1_stamp, "reference": "-4.5"}}}))

    @staticmethod
    def payload(path):
        return json.loads(gzip.decompress(Path(path).read_bytes()).decode())

    def edit_row(self, name, value=None, source=None):
        lines = self.csv.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.startswith(name + ","):
                parts = line.split(",")
                parts[1] = value if value is not None else parts[1]
                parts[2] = source if source is not None else parts[2]
                lines[i] = ",".join(parts)
        self.csv.write_text("\n".join(lines) + "\n")

    def run_verifier(self, *args, data_dir=None):
        buf = io.StringIO()
        argv = ["--csv", str(self.csv), "--root", str(self.root),
                "--data-dir", str(data_dir if data_dir is not None else self.data_dir)]
        with contextlib.redirect_stdout(buf):
            code = verifier.main(argv + list(args))
        return code, buf.getvalue()

    # ---- pass lineages ----------------------------------------------------
    def test_clean_fixture_passes_with_explicit_liswet_skip(self):
        code, out = self.run_verifier()
        self.assertEqual(code, 0, out)
        self.assertIn("S: 3 pass, 0 fail", out)
        self.assertIn("A: 1 pass, 0 fail, 1 skip [LISWET proof not committed", out)
        self.assertIn("E: 1 pass, 0 fail, 1 skip", out)
        # D は情報専用: 旧ツリー stamp があっても exit code を汚さない。
        self.assertIn("1 against an older tree", out)

    def test_regenerate_closes_liswet_layers(self):
        code, out = self.run_verifier("--regenerate")
        self.assertEqual(code, 0, out)
        self.assertIn("A: 2 pass, 0 fail, 0 skip", out)
        self.assertIn("E: 2 pass, 0 fail, 0 skip", out)

    def test_missing_data_dir_skips_layers_explicitly(self):
        code, out = self.run_verifier(data_dir=self.root / "absent")
        self.assertEqual(code, 0, out)
        self.assertIn("2 skip [LISWET proof not committed; use --regenerate; QPS data not available]", out)
        self.assertIn("QPS data or checker deps not available]", out)

    # ---- fail lineages (each pins one detection path) ----------------------
    def test_forged_bounds_with_rehashed_proof_caught_by_replay(self):
        # 最悪ケース: 攻撃者は lower を 1e-9 だけ広げ (B の予算内)、正規化
        # ハッシュも CSV stamp も正規の手順で再計算してくる。ハッシュ系は
        # 全部緑のまま通るが、証跡内 bounds が独立再計算と食い違う。
        proof = self.root / "proofs" / "A1.proof.json.gz"
        payload = self.payload(proof)
        from fractions import Fraction
        payload["lower"] = str(Fraction(payload["lower"]) - Fraction(1, 10**9))
        proof.write_bytes(gzip.compress(json.dumps(payload).encode()))
        self.edit_row("A1", source=self.claim("proofs/A1.proof.json.gz",
                                              verifier.normalized_hash(payload),
                                              checker.checker_digest()))
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("A: 1 pass", out)
        self.assertIn("E: 0 pass, 1 fail", out)
        self.assertIn("replay rejected", out)

    def test_tampered_payload_without_rehash_caught_by_hash(self):
        proof = self.root / "proofs" / "A1.proof.json.gz"
        payload = self.payload(proof)
        payload["lower"] = "-9"
        proof.write_bytes(gzip.compress(json.dumps(payload).encode()))
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("canonical proof hash != CSV proof_sha256", out)

    def test_csv_stamp_mismatch_caught(self):
        self.edit_row("A1", source=self.claim("proofs/A1.proof.json.gz",
                                              "a" * 64, checker.checker_digest()))
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("A: 0 pass, 1 fail", out)

    def test_value_outside_bounds_caught_by_layer_b(self):
        # 証跡とハッシュは完全なまま、CSV の主張値だけを bounds 外へ。
        self.edit_row("A1", value="100.0")
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("A: 1 pass", out)
        self.assertIn("B: 0 pass, 1 fail", out)

    def test_missing_proof_file_caught(self):
        (self.root / "proofs" / "A1.proof.json.gz").unlink()
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("proof file missing", out)

    def test_proof_checker_version_checked(self):
        proof = self.root / "proofs" / "A1.proof.json.gz"
        payload = self.payload(proof)
        payload["checker"] = "totally-different-era"
        proof.write_bytes(gzip.compress(json.dumps(payload).encode()))
        self.edit_row("A1", source=self.claim("proofs/A1.proof.json.gz",
                                              verifier.normalized_hash(payload),
                                              checker.checker_digest()))
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("proof checker version", out)

    def test_missing_manifest_caught(self):
        self.manifest.unlink()
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("manifest missing for LISWET claims", out)

    def test_manifest_hash_disagreement_caught(self):
        self.write_manifest(args=["true"], stamp="b" * 64)
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("manifest expected hash != CSV proof_sha256", out)

    def test_manifest_reference_drift_caught(self):
        text = self.manifest.read_text().replace('"reference": "-4.5"', '"reference": "-9"')
        self.manifest.write_text(text)
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("manifest reference", out)

    def test_regeneration_failure_caught(self):
        dead = Path(self.gen).with_name("dead.py")
        dead.write_text(FAILING_GENERATOR)
        self.write_manifest(args=[sys.executable, str(dead), "--evidence-dir",
                                  "{evidence_dir}", str(self.data_dir / "L1.QPS")])
        code, out = self.run_verifier("--regenerate")
        self.assertEqual(code, 1, out)
        self.assertIn("regeneration failed", out)

    def test_uncertified_row_needs_claim_marker(self):
        self.edit_row("U1", source="revalidation_psd_unknown_UNCERTIFIED")
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("uncertified row lacks the claim marker", out)

    def test_malformed_claim_keys_caught(self):
        row = ("A1,-4.5,exact_enclosure_v1;proof=proofs/A1.proof.json.gz;"
               f"input_sha256={self.input_hash};checker_sha256={'0' * 64}")
        lines = [row if line.startswith("A1,") else line
                 for line in self.csv.read_text().splitlines()]
        self.csv.write_text("\n".join(lines) + "\n")
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("neither a claim nor _UNCERTIFIED", out)

    def test_non_hex_digest_caught(self):
        self.edit_row("A1", source=self.claim("proofs/A1.proof.json.gz",
                                              "zz" * 32, checker.checker_digest()))
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("non-hex64 claim digests", out)

    def test_proof_path_escape_caught(self):
        self.edit_row("A1", source=self.claim("../outside.proof.json.gz",
                                              self.a1_stamp, checker.checker_digest()))
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("proof path escapes proofs/", out)

    def test_tampered_input_file_caught_by_layer_c(self):
        (self.data_dir / "A1.QPS").write_text(FIXTURE_QPS + "X\n")
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("C: 0 pass, 1 fail", out)
        self.assertIn("QPS sha256 != CSV input_sha256", out)


if __name__ == "__main__":
    unittest.main()
