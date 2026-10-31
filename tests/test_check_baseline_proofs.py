"""Self-tests for scripts/check_baseline_proofs.py (offline proof verifier).

Every detection path from the adversarial review (mu1-mu10, T3c) is pinned by
a boundary fixture here, not by the ephemeral mutation script: a verifier
mutation that ignores tampering must turn one of these red. The regenerate
happy path for real LISWET proofs cannot run hermetically (the LISWET script
needs its banded exact structure), so it is exercised by the CI data job and
locally; the fixtures pin everything the audit job can see.
"""
import contextlib
import gzip
import hashlib
import io
import json
import lzma
import sys
import tempfile
import unittest
from fractions import Fraction
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
# mu3 probe (measured): 100000000.01 rounds UP to the next binary64; with the
# bounds below the DECIMAL reading sits exactly on REFERENCE_TOLERANCE (pass,
# fail is strict >) while the BINARY64 reading exceeds it (fail). A verifier
# that only checks the decimal interpretation survives nothing here.
MU3_VALUE = "100000000.01"
MU3_BOUND = "99999990009999899/1000000000"


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

        # A1: committed non-LISWET proof. L1: manifest-backed claim.
        # U1: labelled pre-contract value.
        pins = self.root / "scripts"
        pins.mkdir()
        self.pin_file = pins / "maros_meszaros_qps.sha256"
        self.pin_stems("A1", "L1", "U1")
        qps = data / "A1.QPS"
        qps.write_text(FIXTURE_QPS)
        self.input_hash = hashlib.sha256(qps.read_bytes()).hexdigest()
        enc = checker.ExactModel(truth.load_qps(qps)).check(
            [checker.rational(3)], [checker.rational(3)], [])
        proof = self.root / "proofs" / "A1.proof.json.gz"
        checker.save_evidence(proof, enc, "-4.5", self.input_hash,
                              proposal={"method": "synthetic"})
        self.a1_stamp = verifier.normalized_hash(self.payload(proof))
        (data / "L1.QPS").write_text(FIXTURE_QPS)
        self.manifest = self.root / "proofs" / "liswet" / "manifest.json"
        self.write_manifest_rows({"L1": self.l1_entry()})
        self.csv.write_text(
            "problem_name,optimal_obj,source\n"
            f"A1,-4.5,{self.claim('proofs/A1.proof.json.gz', self.a1_stamp, checker.checker_digest())}\n"
            f"L1,-4.5,{self.claim(verifier.LISWET_MANIFEST, self.a1_stamp, '0' * 64)}\n"
            "U1,-4.5,claim=pre_contract_numeric_reference_not_certified;"
            "revalidation_psd_unknown_UNCERTIFIED\n")

    # ---- fixture helpers ----------------------------------------------------
    def pin_stems(self, *names):
        self.pin_file.write_text("".join(f"{'0' * 64}  {n}.QPS\n" for n in names))

    def append_row(self, line):
        text = self.csv.read_text().rstrip("\n")
        self.csv.write_text(text + "\n" + line + "\n")

    def claim(self, proof, stamp, checker_sha):
        return (f"{verifier.ENCLOSURE_TAG};proof={proof};proof_sha256={stamp};"
                f"input_sha256={self.input_hash};checker_sha256={checker_sha}")

    def l1_entry(self, **overrides):
        entry = {"problem": "L1", "gaps": [], "qps": "L1.QPS",
                 "proof_sha256": self.a1_stamp, "reference": "-4.5"}
        entry.update(overrides)
        return entry

    def write_manifest_rows(self, rows):
        self.manifest.write_text(json.dumps({"policy": "fixture", "rows": rows}))

    @staticmethod
    def payload(path):
        return json.loads(gzip.decompress(Path(path).read_bytes()).decode())

    def forge_payload(self, mut, name="A1", proof_rel=None, value=None):
        """Rewrite a proof payload, optionally re-stamping the CSV row (a
        capable attacker recomputes the canonical hash too)."""
        proof = self.root / "proofs" / (name + ".proof.json.gz")
        payload = self.payload(proof)
        mut(payload)
        proof.write_bytes(gzip.compress(json.dumps(payload).encode()))
        stamp = verifier.normalized_hash(payload)
        rel = proof_rel or f"proofs/{name}.proof.json.gz"
        self.edit_row(name, value=value,
                      source=self.claim(rel, stamp, checker.checker_digest()))
        return stamp

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

    # ---- pass lineages ------------------------------------------------------
    def test_clean_fixture_passes_and_advertises_the_data_job(self):
        code, out = self.run_verifier()
        self.assertEqual(code, 0, out)
        self.assertIn("S: 3 pass, 0 fail", out)
        self.assertIn("A: 1 pass, 0 fail, 1 skip [LISWET proof not committed", out)
        self.assertIn("E: 1 pass, 0 fail, 1 skip", out)
        # audit と data job の関係は出力で表明される (review P0 の code-level 表明)。
        self.assertIn("note: C/E skips here are only acceptable in the hermetic", out)
        # D は情報専用: 旧ツリー stamp があっても exit code を汚さない。
        self.assertIn("1 against an older tree", out)

    def test_missing_data_dir_skips_layers_explicitly(self):
        code, out = self.run_verifier(data_dir=self.root / "absent")
        self.assertEqual(code, 0, out)
        self.assertIn("2 skip [LISWET proof not committed; use --regenerate; QPS data not available]", out)
        self.assertIn("QPS data or checker deps not available]", out)

    def test_require_layers_fails_on_skips_and_passes_without_them(self):
        code, out = self.run_verifier("--require-layers", "S")
        self.assertEqual(code, 0, out)
        code, out = self.run_verifier("--require-layers", "C")
        self.assertEqual(code, 1, out)
        self.assertIn("R C: required layer reported 1 skip(s)", out)
        code, out = self.run_verifier("--require-layers", "D")
        self.assertEqual(code, 1, out)
        self.assertIn("informational and cannot be required", out)

    # ---- manifest structure (mu1/mu2, P1-2 RCE closure) ----------------------
    def test_manifest_non_object_json_fails_closed(self):
        self.manifest.write_text("[1, 2]")
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("manifest must be an object with a rows object", out)

    def test_manifest_entry_missing_skips_without_crash(self):
        self.write_manifest_rows({})
        code, out = self.run_verifier("--regenerate")
        self.assertEqual(code, 1, out)
        self.assertIn("no manifest entry", out)
        self.assertIn("manifest entry missing or malformed", out)

    def test_manifest_argv_keys_rejected(self):
        # P1-2: free-form argv arrays were the RCE surface; the schema has no
        # place for a command any more.
        self.write_manifest_rows({"L1": {**self.l1_entry(), "command": "touch pwned"}})
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("manifest entry keys must be exactly", out)
        self.assertFalse((self.root / "pwned").exists())

    def test_manifest_qps_mismatch_and_template_rejected(self):
        for bad in ("/etc/passwd", "L1{evidence_dir}.QPS", "Other.QPS"):
            self.write_manifest_rows({"L1": self.l1_entry(qps=bad)})
            code, out = self.run_verifier()
            self.assertEqual(code, 1, f"{bad}: {out}")
            self.assertIn("manifest entry malformed", out)

    def test_manifest_extra_row_rejected(self):
        self.write_manifest_rows({"L1": self.l1_entry(),
                                  "L2": self.l1_entry(problem="L2", qps="L2.QPS")})
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("manifest rows not referenced by CSV", out)

    def test_manifest_reference_and_hash_agreement(self):
        self.write_manifest_rows({"L1": self.l1_entry(reference="-9")})
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("manifest reference", out)
        self.write_manifest_rows({"L1": self.l1_entry(proof_sha256="b" * 64)})
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("manifest expected hash != CSV proof_sha256", out)

    def test_gaps_structures_rejected(self):  # mu7
        for bad in ([0], [-1], [1.5], ["1"], [True], [2, 2], {"g": 1}):
            self.write_manifest_rows({"L1": self.l1_entry(gaps=bad)})
            code, out = self.run_verifier()
            self.assertEqual(code, 1, f"{bad!r}: {out}")
            self.assertIn("manifest entry malformed", out)

    def test_fake_gaps_regenerate_fails_closed(self):
        # gaps は構造検証を通っても、実再生成が失敗すれば赤 (偽 gaps 防御)。
        self.write_manifest_rows({"L1": self.l1_entry(gaps=[999])})
        code, out = self.run_verifier("--regenerate")
        self.assertEqual(code, 1, out)
        self.assertIn("regeneration failed", out)

    # ---- proof tampering (mu3-mu5, mu9, mu10, T3c) ---------------------------
    def test_forged_bounds_with_rehashed_proof_caught_by_replay(self):
        # 攻撃者は lower を 1e-9 広げ (B 予算内) 正規化ハッシュも再計算する。
        self.forge_payload(lambda p: p.__setitem__(
            "lower", str(Fraction(p["lower"]) - Fraction(1, 10**9))))
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

    def test_malformed_claim_keys_caught(self):
        row = ("A1,-4.5,exact_enclosure_v1;proof=proofs/A1.proof.json.gz;"
               f"input_sha256={self.input_hash};checker_sha256={'0' * 64}")
        lines = [row if line.startswith("A1,") else line
                 for line in self.csv.read_text().splitlines()]
        self.csv.write_text("\n".join(lines) + "\n")
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("neither a claim nor _UNCERTIFIED", out)

    def test_extra_claim_key_rejected(self):  # mu6
        self.edit_row("A1", source=self.claim("proofs/A1.proof.json.gz",
                                              self.a1_stamp, checker.checker_digest())
                      + ";evil=1")
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("neither a claim nor _UNCERTIFIED", out)

    def test_non_hex_digest_caught(self):
        self.edit_row("A1", source=self.claim("proofs/A1.proof.json.gz",
                                              "zz" * 32, checker.checker_digest()))
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("non-hex64 claim digests", out)

    def test_proof_path_escape_and_sibling_caught(self):  # mu8
        for bad in ("../outside.proof.json.gz", "other/A1.proof.json.gz"):
            self.edit_row("A1", source=self.claim(bad, self.a1_stamp,
                                                   checker.checker_digest()))
            code, out = self.run_verifier()
            self.assertEqual(code, 1, f"{bad}: {out}")
            self.assertIn("proof path escapes proofs/", out)

    def test_binary64_reading_caught(self):  # mu3
        # decimal 解釈はちょうど tol 上 (pass)、binary64 解釈は超える (fail)。
        # bounds_error が両方の読みを検証していることの境界 pin。
        self.forge_payload(lambda p: p.update({"lower": MU3_BOUND, "upper": MU3_BOUND}),
                           value=MU3_VALUE)
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("B: 0 pass, 1 fail", out)

    def test_budget_ordering_boundaries(self):  # mu4
        # ×100 緩和では通らない magnitude 境界: 1.1e-7 絶対リテラルで赤、
        # 9.9e-8 で緑 (bounds=[0,0]、value が誤差そのもの)。緑側も bounds を
        # 偽装しているので replay は落ちる — B 層単体の境界を固定する。
        self.forge_payload(lambda p: p.update({"lower": "0", "upper": "0"}),
                           value="1.1e-7")
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("value outside bounds budget", out)
        self.forge_payload(lambda p: p.update({"lower": "0", "upper": "0"}),
                           value="9.9e-8")
        code, out = self.run_verifier()
        self.assertIn("B: 1 pass, 0 fail", out)

    def test_upper_endpoint_budget_checked(self):  # mu10
        # lower だけ見れば green になる非対称 bounds (|r|=1 なので誤差は
        # 距離/2)。upper 端の距離 2.2e-7 が予算超で赤。
        self.forge_payload(lambda p: p.update(
            {"lower": "19999999/20000000", "upper": "50000011/50000000"}), value="1.0")
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("B: 0 pass, 1 fail", out)
        # lower 距離 5e-8 / upper 距離 8e-8 なら両端 green。
        self.forge_payload(lambda p: p.update(
            {"lower": "19999999/20000000", "upper": "50000004/50000000"}), value="1.0")
        code, out = self.run_verifier()
        self.assertIn("B: 1 pass, 0 fail", out)

    def test_payload_input_hash_cross_checked(self):  # mu5
        # 攻撃者が CSV の proof_sha256 まで再計算しても (A green)、proof 内の
        # input_sha256 と CSV input_sha256 の突合 (C) で必ず赤くなる。
        self.forge_payload(lambda p: p.__setitem__("input_sha256", "c" * 64))
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("A: 1 pass", out)
        self.assertIn("proof input hash != CSV input_sha256", out)

    def test_forged_reference_caught_by_replay(self):  # mu9
        self.forge_payload(lambda p: p.__setitem__("reference", "999"))
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("A: 1 pass", out)
        self.assertIn("B: 1 pass", out)
        self.assertIn("replay rejected", out)

    def test_forged_minimal_proof_hits_replay(self):  # T3c 回帰
        # 攻撃レビュー T3c: 簡略区間 + 空 witness の偽 proof を、全ハッシュを
        # 自己整合させてコミットしても、data ありでは replay が赤くする。
        name = "F1"
        qps = self.data_dir / "F1.QPS"
        qps.write_text(FIXTURE_QPS)
        h = hashlib.sha256(qps.read_bytes()).hexdigest()
        payload = {"checker": checker.CHECKER_VERSION, "input_sha256": h,
                   "model_sha256": "0" * 64, "reference": "-4.5",
                   "lower": "-9/2", "upper": "-9/2", "relative_error": "0",
                   "primal": [], "linearization": [], "dual": [],
                   "checker_sha256": checker.checker_digest(), "proposal": None}
        proof = self.root / "proofs" / "F1.proof.json.gz"
        proof.write_bytes(gzip.compress(json.dumps(payload).encode()))
        self.pin_stems("A1", "L1", "U1", "F1")
        lines = self.csv.read_text().splitlines()
        lines.append(f"{name},-4.5,"
                     + f"{verifier.ENCLOSURE_TAG};proof=proofs/{name}.proof.json.gz;"
                     + f"proof_sha256={verifier.normalized_hash(payload)};"
                     + f"input_sha256={h};checker_sha256={checker.checker_digest()}")
        self.csv.write_text("\n".join(lines) + "\n")
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        # A/B/C は通る (ハッシュ自己整合) — 検出は replay 依存であることを固定。
        self.assertIn("A: 2 pass", out)
        self.assertIn("B: 2 pass", out)
        self.assertIn("C: 2 pass", out)
        self.assertIn("E: 1 pass, 1 fail", out)
        self.assertIn("replay rejected", out)

    def test_non_object_proof_json_caught(self):  # P3
        proof = self.root / "proofs" / "A1.proof.json.gz"
        proof.write_bytes(gzip.compress(b"[1, 2]"))
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("proof unreadable", out)

    def test_missing_proof_file_caught(self):
        (self.root / "proofs" / "A1.proof.json.gz").unlink()
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("proof file missing", out)

    def test_proof_checker_version_checked(self):
        self.forge_payload(lambda p: p.__setitem__("checker", "totally-different-era"))
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("proof checker version", out)

    # ---- codec dispatch (committed entities may be lzma) --------------------
    def test_lzma_codec_roundtrip_keeps_hash(self):
        gz = self.root / "proofs" / "A1.proof.json.gz"
        lz = self.root / "proofs" / "A1.proof.json.lzma"
        lz.write_bytes(lzma.compress(gzip.decompress(gz.read_bytes()), preset=6))
        gz.unlink()
        self.edit_row("A1", source=self.claim("proofs/A1.proof.json.lzma",
                                              self.a1_stamp, checker.checker_digest()))
        code, out = self.run_verifier()
        self.assertEqual(code, 0, out)
        self.assertIn("A: 1 pass", out)
        self.assertIn("E: 1 pass", out)

    def test_undecompressible_lzma_caught(self):
        lz = self.root / "proofs" / "A1.proof.json.lzma"
        lz.write_bytes(b"not-an-xz-stream-at-all")
        (self.root / "proofs" / "A1.proof.json.gz").unlink()
        self.edit_row("A1", source=self.claim("proofs/A1.proof.json.lzma",
                                              self.a1_stamp, checker.checker_digest()))
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("proof unreadable", out)

    def test_unknown_codec_refused_not_guessed(self):
        gz = self.root / "proofs" / "A1.proof.json.gz"
        (self.root / "proofs" / "A1.proof.json.zip").write_bytes(gz.read_bytes())
        gz.unlink()
        self.edit_row("A1", source=self.claim("proofs/A1.proof.json.zip",
                                              self.a1_stamp, checker.checker_digest()))
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("unsupported proof codec", out)

    # ---- CSV structure (P2) ---------------------------------------------------
    def test_uncertified_row_needs_claim_marker(self):
        self.edit_row("U1", source="revalidation_psd_unknown_UNCERTIFIED")
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("uncertified row lacks the claim marker", out)

    def test_duplicate_problem_name_caught(self):
        dup = ('A1,-4.5,' + self.claim("proofs/A1.proof.json.gz", self.a1_stamp,
                                        checker.checker_digest()))
        text = self.csv.read_text().rstrip("\n")
        self.csv.write_text(text + "\n" + dup + "\n")
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("duplicate problem_name", out)

    def test_header_count_claims_pin_down_the_rows(self):
        good = ("# Certified (2 of 3)\n# Uncertified (1 of 3)\n"
                "#   Breakdown: 1 revalidation_psd_unknown (fixture)\n")
        text = self.csv.read_text()
        self.csv.write_text(good + text)
        code, out = self.run_verifier()
        self.assertEqual(code, 0, out)
        bad = ("# Certified (9 of 3)\n# Uncertified (1 of 3)\n"
               "#   Breakdown: 2 revalidation_psd_unknown (fixture)\n")
        self.csv.write_text(bad + text)
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("header:", out)

    def test_tampered_input_file_caught_by_layer_c(self):
        (self.data_dir / "A1.QPS").write_text(FIXTURE_QPS + "X\n")
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("C: 0 pass, 1 fail", out)
        self.assertIn("QPS sha256 != CSV input_sha256", out)

    # ---- re-attack P1: problem_name re-anchoring (回帰攻撃 発見1) -------------
    def test_unsafe_problem_names_rejected(self):
        # 各危険名は独立の CSV で検証 (前の行の検出メッセージで後の変種が
        # 緑を偽装できないようにする)。
        base = self.csv.read_text()
        for bad in (".", "..", "../A1", "-flag", "sub/A1", "A B"):
            self.csv.write_text(base)
            self.append_row(f"{bad},-4.5," + self.claim("proofs/A1.proof.json.gz",
                                                        self.a1_stamp,
                                                        checker.checker_digest()))
            code, out = self.run_verifier()
            self.assertEqual(code, 1, f"{bad!r}: {out}")
            self.assertIn(f"unsafe problem_name {bad!r}", out)

    def test_lead_reproduction_traversal_row_forgery_closed(self):
        # lead 再現手順そのもの: ../maros_meszaros/BOYD2 相当 — data_dir の外
        # (root/A1.QPS) を指す行で、hash・proof・header すべて自己整合させても
        # 「OK: 2 certified claims checked」の緑偽装になる (修正前は実際 rc=0、
        # /tmp/pr51-reattack/before_attack1.log 実測)。
        (self.root / "A1.QPS").write_text(FIXTURE_QPS)  # data_dir/../A1.QPS
        self.append_row("../A1,-4.5," + self.claim("proofs/A1.proof.json.gz",
                                                   self.a1_stamp,
                                                   checker.checker_digest()))
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertNotIn("OK: 2 certified claims", out)
        self.assertIn("unsafe problem_name", out)
        self.assertIn("outside pinned QPS set", out)

    def test_pinned_stem_set_agreement(self):
        self.pin_stems("A1", "L1")  # U1 の行はあるが pin に無い
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("outside pinned QPS set", out)
        self.pin_stems("A1", "L1", "U1", "Z9")  # pin にあって行に無い
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("without CSV row", out)
        self.pin_file.unlink()
        code, out = self.run_verifier()
        self.assertEqual(code, 1, out)
        self.assertIn("missing (cannot pin the problem set)", out)

    # ---- re-attack P2: silent inventory zeroing (回帰攻撃 発見2) --------------
    def flip_all_uncertified(self):
        lines = []
        for line in self.csv.read_text().splitlines():
            if line.startswith(("A1,", "L1,")):
                line = line.split(",")[0] + (",1e99,claim=pre_contract_numeric_"
                                            "reference_not_certified;revalidation_psd_unknown_UNCERTIFIED")
            lines.append(line)
        self.csv.write_text("\n".join(lines) + "\n")

    def test_zero_inventory_red_even_with_full_layers(self):
        # 全行 UNCERTIFIED 反転 + ガベ値: 層は一度も走らないのに従来は緑だった
        # (修正前実測 /tmp/pr51-reattack/before_attack2.log)。
        self.flip_all_uncertified()
        code, out = self.run_verifier("--require-layers", "S,A,B,C,E",
                                      "--regenerate")
        self.assertEqual(code, 1, out)
        self.assertIn("required layer produced 0 passing checks", out)
        code, out = self.run_verifier("--min-certified", "1")
        self.assertEqual(code, 1, out)
        self.assertIn("certified claims < required minimum 1", out)

    def test_min_certified_boundary(self):
        # fixture の認証在庫は 2 (A1 + L1)。
        code, out = self.run_verifier("--min-certified", "2")
        self.assertEqual(code, 0, out)
        code, out = self.run_verifier("--min-certified", "3")
        self.assertEqual(code, 1, out)
        self.assertIn("certified claims < required minimum 3", out)

    def test_require_layer_zero_pass_red(self):
        # pass=0 かつ skip=0 の層 (認証行ゼロで誰も走らない) を require したら赤。
        self.flip_all_uncertified()
        code, out = self.run_verifier("--require-layers", "C")
        self.assertEqual(code, 1, out)
        self.assertIn("R C: required layer produced 0 passing checks", out)


if __name__ == "__main__":
    unittest.main()
