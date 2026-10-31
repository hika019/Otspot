#!/usr/bin/env python3
"""Offline verifier for the committed Maros-Meszaros proof claims (PR #51 review P1-1).

The CSV source column is only honest if somebody can check it without rerunning
the full certification pipeline. This script is that somebody. It resolves every
claim in a baseline CSV and checks it in independent layers:

  S  shape (always, hermetic): row count, safe problem_name file stems, CSV
     rows set-equal to the pinned QPS set (scripts/maros_meszaros_qps.sha256),
     no duplicate problem_name, header count claims (when present) match the
     actual 88/50/138 breakdown, certified sources carry exactly the four
     claim keys with 64-hex digests and proof paths under proofs/, uncertified
     rows carry the pre-contract claim marker, and the LISWET manifest is
     structurally exact (per-row problem/qps/gaps/proof_sha256/reference,
     set-equal to the CSV's LISWET claims).
  A  hash (hermetic for committed entities): SHA-256 over the canonical
     expanded proof JSON equals the CSV proof_sha256; the proof's contract
     version matches. LISWET proofs are not committed; layer A covers them
     only under --regenerate.
  B  value (runs whenever a payload resolved): the CSV optimal_obj lies inside
     the proof bounds within REFERENCE_TOLERANCE, evaluated over BOTH the
     decimal and the binary64 interpretation of the value text.
  C  input (data-gated): the CSV input_sha256 equals SHA-256 of data/*.QPS and
     of the hash stored inside the proof.
  E  replay (data-gated): checker.replay_evidence recomputes the enclosure
     from the witness; a forged proof (tight bounds, empty or bogus witness)
     cannot survive this even when every hash is self-consistent.
  D  checker drift (informational, never fails the run): rows whose
     checker_sha256 stamp differs from the current checker_digest() were
     verified against an older tree. This layer is a report, not a guard:
     the real guards against forged proofs are the checker's own adversarial
     tests plus layer E; a code change calls for revalidation, not re-stamping.

CI placement matters for reading the SKIP lines: the audit job has no QPS
data, so there layers C/E skip by design (audit runs --require-layers S).
A separate CI job downloads the QPS subset and MUST run
--regenerate --require-layers S,A,B,C,E: that job is where the LISWET claims
are actually verified and where forged proofs hit layer E. If that job is
removed or stops being required, the audit job's green is NOT a proof of
certificate integrity — the skip note printed by this tool says so.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import lzma
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_certificate as checker  # noqa: E402
import revalidate_baselines as reval  # noqa: E402  (stdlib-only at import time)

# Kept in sync with revalidate_baselines.output_source (which hardcodes the
# tag); this verifier is deliberately outside CHECKER_FILES so adding it did
# not move the checker digest.
ENCLOSURE_TAG = "exact_enclosure_v1"
PROOF_SUBDIR = "proofs"
LISWET_MANIFEST = "proofs/liswet/manifest.json"
UNCERTIFIED_CLAIM = "claim=pre_contract_numeric_reference_not_certified"
# Undecodable proof entities are a fail, never a silent skip.
PROOF_ERRORS = (OSError, ValueError, EOFError, lzma.LZMAError)
# Structured manifest schema: free-form argv arrays were an RCE surface under
# --regenerate (review P1-2). The verifier builds argv itself and runs exactly
# one script; a manifest entry can only choose gaps for one problem file.
MANIFEST_KEYS = {"problem", "gaps", "qps", "proof_sha256", "reference"}
REGEN_SCRIPT = "scripts/liswet_exact_certificate.py"


def normalized_hash(payload: dict) -> str:
    """The contract hash: canonical JSON of the expanded proof content."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def expand_proof(path: Path) -> dict:
    """Decode a committed proof entity. The storage codec is the file
    extension; proof_sha256 is defined over the expanded content, so any
    lossless codec is legal. An unknown codec is refused, never guessed."""
    raw = path.read_bytes()
    if path.name.endswith(".lzma"):
        expanded = lzma.decompress(raw)
    elif path.name.endswith(".gz"):
        expanded = gzip.decompress(raw)
    else:
        raise ValueError(f"unsupported proof codec extension: {path.name}")
    payload = json.loads(expanded.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("proof JSON is not an object")
    return payload


def _is_hex64(text) -> bool:
    return isinstance(text, str) and len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def read_text_and_rows(csv_path: Path):
    text = csv_path.read_text()
    lines = [line for line in text.splitlines()
             if not line.startswith("#") and line.strip()]
    return text, list(csv.DictReader(lines))


def parse_claim(source: str):
    """Return the claim keys of a certified source, or None if malformed."""
    parts = source.split(";")
    if parts[0] != ENCLOSURE_TAG:
        return None
    keys: dict[str, str] = {}
    for item in parts[1:]:
        key, sep, value = item.partition("=")
        if not sep or key in keys or not value:
            return None
        keys[key] = value
    if set(keys) != set(reval.CLAIM_KEYS):
        return None
    return keys


def bounds_error(payload: dict, value: str) -> Fraction:
    """Independent budget recomputation over decimal AND binary64 readings."""
    lo, up = Fraction(payload["lower"]), Fraction(payload["upper"])
    worst = Fraction(0)
    for r in (Fraction(value), Fraction(float(value))):
        worst = max(worst, max(abs(r - lo), abs(r - up)) / (1 + abs(r)))
    return worst


def decimal_error(payload: dict, value: str) -> Fraction:
    lo, up = Fraction(payload["lower"]), Fraction(payload["upper"])
    r = Fraction(value)
    return max(abs(r - lo), abs(r - up)) / (1 + abs(r))


class Layers:
    def __init__(self) -> None:
        self.names = ("S", "A", "B", "C", "E")
        self.count = {name: {"pass": 0, "fail": 0, "skip": 0} for name in self.names}
        self.skips: dict[str, set[str]] = {name: set() for name in self.names}
        self.failures: list[str] = []

    def record(self, layer: str, result: str, name: str, reason: str = "") -> None:
        self.count[layer][result] += 1
        if result == "skip":
            self.skips[layer].add(reason)
        elif result == "fail":
            self.failures.append(f"{layer} {name}: {reason}")


def load_manifest(path: Path, problems: set[str], layers: Layers) -> dict:
    """Parse the LISWET manifest fail-closed: exact key sets, exact types.

    Any structural deviation (missing file, non-object JSON, missing rows
    mapping, unknown/missing entry keys, template remnants, bad gaps, qps that
    is not <problem>.QPS) is a FAIL, never a guess.
    """
    def fail(name: str, reason: str):
        layers.failures.append(f"S {name}: {reason}")

    if not path.is_file():
        fail(LISWET_MANIFEST, "manifest missing for LISWET claims")
        return {}
    try:
        doc = json.loads(path.read_text())
    except PROOF_ERRORS as exc:
        fail(LISWET_MANIFEST, f"manifest unreadable: {exc}")
        return {}
    if not isinstance(doc, dict) or not isinstance(doc.get("rows"), dict):
        fail(LISWET_MANIFEST, "manifest must be an object with a rows object")
        return {}
    entries = doc["rows"]
    rows = {}
    for name in sorted(problems):
        entry = entries.get(name)
        if entry is None:
            fail(name, "no manifest entry")
            continue
        if not isinstance(entry, dict) or set(entry) != MANIFEST_KEYS:
            keys = sorted(entry) if isinstance(entry, dict) else type(entry).__name__
            fail(name, f"manifest entry keys must be exactly {sorted(MANIFEST_KEYS)}, got {keys}")
            continue
        ok = (entry["problem"] == name
              and isinstance(entry["qps"], str) and entry["qps"] == f"{name}.QPS"
              and _is_hex64(entry["proof_sha256"])
              and isinstance(entry["reference"], str) and bool(entry["reference"])
              and isinstance(entry["gaps"], list)
              and all(isinstance(g, int) and not isinstance(g, bool) and g >= 1
                      for g in entry["gaps"])
              and len(set(entry["gaps"])) == len(entry["gaps"]))
        if not ok:
            fail(name, "manifest entry malformed (problem/qps/gaps/hash/reference)")
            continue
        rows[name] = entry
    extra = sorted(set(map(str, entries)) - set(problems))
    if extra:
        fail(LISWET_MANIFEST, f"manifest rows not referenced by CSV: {extra}")
    return rows


def regenerate(entry: dict, root: Path, data_dir: Path):
    """Re-run the fixed LISWET script with gaps from the manifest.

    argv is built here, not read from the manifest: the executable is
    sys.executable, the script is REGEN_SCRIPT, and the only manifest-chosen
    data are the gap indices and the <problem>.QPS file inside data_dir.
    """
    with tempfile.TemporaryDirectory(prefix="proofcheck-") as tmp:
        argv = [sys.executable, str(root / REGEN_SCRIPT), "--csv",
                "--gaps", ",".join(str(g) for g in sorted(entry["gaps"])),
                "--evidence-dir", tmp, str(data_dir / entry["qps"])]
        try:
            subprocess.run(argv, cwd=root, check=True, capture_output=True, timeout=3600)
        except (subprocess.SubprocessError, OSError) as exc:
            return None, f"regeneration failed: {exc}"
        proof = Path(tmp) / (Path(entry["qps"]).stem + ".proof.json.gz")
        if not proof.is_file():
            return None, "regeneration produced no proof file"
        try:
            return expand_proof(proof), ""
        except PROOF_ERRORS as exc:
            return None, f"regenerated proof unreadable: {exc}"


def check_stem_set(root: Path, names: set[str], layers: Layers) -> None:
    """The CSV must cover exactly the pinned QPS set (review re-attack P1).

    scripts/maros_meszaros_qps.sha256 is the download-side truth of which
    problems exist. Without this pin, adding a row named ../maros_meszaros/X
    (or ../lp_problems/X) re-anchors C/E and the LISWET argv onto a body
    outside the certified inventory while every hash stays self-consistent.
    Extra rows, dropped rows and renames are all red; a missing pin file is
    fail-closed, never an implicit skip.
    """
    sha = root / "scripts" / "maros_meszaros_qps.sha256"
    if not sha.is_file():
        layers.failures.append(
            f"S pinned: {sha.name} missing (cannot pin the problem set)")
        return
    stems = set()
    for line in sha.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            stems.add(Path(line.split()[-1]).stem)
    extra, missing = sorted(names - stems), sorted(stems - names)
    if extra:
        layers.failures.append(f"S pinned: CSV rows outside pinned QPS set: {extra[:5]}")
    if missing:
        layers.failures.append(f"S pinned: pinned QPS files without CSV row: {missing[:5]}")


def check_header_counts(text: str, rows: list[dict], certified: int,
                        uncertified: int, reasons: dict, layers: Layers) -> None:
    """Header claims (Certified (N of M) / Uncertified (K of M) / breakdown)
    must agree with the rows. A header without claims is legal (fixtures)."""
    def fail(reason: str):
        layers.failures.append(f"S header: {reason}")
    m = re.search(r"Certified \((\d+) of (\d+)\)", text)
    if m:
        if int(m.group(1)) != certified or int(m.group(2)) != len(rows):
            fail(f"'Certified ({m.group(1)} of {m.group(2)})' != {certified} claims / {len(rows)} rows")
    m = re.search(r"Uncertified \((\d+) of (\d+)\)", text)
    if m:
        if int(m.group(1)) != uncertified or int(m.group(2)) != len(rows):
            fail(f"'Uncertified ({m.group(1)} of {m.group(2)})' != {uncertified} / {len(rows)}")
    for count, reason in re.findall(r"(\d+) revalidation_([a-z_]+) \(", text):
        if reasons.get(reason, 0) != int(count):
            fail(f"breakdown claims {count} {reason}, rows have {reasons.get(reason, 0)}")


def check(csv_path: Path, root: Path, data_dir: Path, regenerate_liswet: bool,
          require_layers: set[str], min_certified: int) -> int:
    layers = Layers()
    text, rows = read_text_and_rows(csv_path)
    proofs_dir = csv_path.parent / PROOF_SUBDIR

    certified: dict[str, tuple[dict, dict]] = {}
    liswet_rows: set[str] = set()
    uncertified = 0
    reasons: dict[str, int] = {}
    seen: set[str] = set()
    row_names: set[str] = set()
    for index, row in enumerate(rows):
        name, source = row.get("problem_name"), row.get("source")
        if not name or not isinstance(name, str) or not isinstance(source, str):
            layers.failures.append(f"S row{index}: missing problem_name or source")
            continue
        # Names reach filesystem paths (data_dir/<name>.QPS, proofs/<name>...)
        # and one argv slot, so they must be plain file stems. SAFE_NAME alone
        # admits ".." and "-flag", hence the extra checks (reuses the writer's
        # guard in revalidate_baselines); dot-only names are rejected because
        # SAFE_NAME admits them and they are pure directory references.
        if (not reval.SAFE_NAME.fullmatch(name) or Path(name).name != name
                or name.startswith("-") or set(name) == {"."}):
            layers.failures.append(f"S row{index}: unsafe problem_name {name!r}")
            row_names.add(name)
            continue
        row_names.add(name)
        if name in seen:
            # last-wins in dict readers would hide a second claim for one row.
            layers.failures.append(f"S {name}: duplicate problem_name")
            continue
        seen.add(name)
        if source.endswith("_UNCERTIFIED"):
            # A retained pre-contract value is only honest while it says so.
            if source.startswith(UNCERTIFIED_CLAIM + ";"):
                uncertified += 1
                reason = source.split("revalidation_")[-1].split("_UNCERTIFIED")[0]
                reasons[reason] = reasons.get(reason, 0) + 1
                layers.count["S"]["pass"] += 1
            else:
                layers.failures.append(f"S {name}: uncertified row lacks the claim marker")
            continue
        keys = parse_claim(source)
        if keys is None:
            layers.failures.append(f"S {name}: source is neither a claim nor _UNCERTIFIED")
            continue
        bad = [key for key in ("proof_sha256", "input_sha256", "checker_sha256")
               if not _is_hex64(keys[key])]
        proof = keys["proof"]
        if bad:
            layers.failures.append(f"S {name}: non-hex64 claim digests: {bad}")
            continue
        if proof.startswith("/") or ".." in Path(proof).parts \
                or not proof.startswith(PROOF_SUBDIR + "/"):
            layers.failures.append(f"S {name}: proof path escapes {PROOF_SUBDIR}/: {proof}")
            continue
        certified[name] = (row, keys)
        if proof == LISWET_MANIFEST:
            liswet_rows.add(name)
        layers.count["S"]["pass"] += 1

    manifest_rows = load_manifest(proofs_dir / "liswet" / "manifest.json",
                                  liswet_rows, layers) if liswet_rows else {}
    check_header_counts(text, rows, len(certified), uncertified, reasons, layers)
    check_stem_set(root, row_names, layers)

    data_present = data_dir.is_dir()
    truth = None
    try:
        import highs_baseline_truth as truth  # noqa: F811
    except ImportError:
        pass

    for name, (row, keys) in sorted(certified.items()):
        value = row["optimal_obj"]
        if name in liswet_rows:
            entry = manifest_rows.get(name)
            if entry is None:
                # load_manifest already recorded the S failure.
                for layer in ("A", "B", "C", "E"):
                    layers.record(layer, "skip", name, "manifest entry missing or malformed")
                continue
            if entry["reference"] != value:
                layers.failures.append(
                    f"S {name}: manifest reference {entry['reference']} != CSV {value}")
            if entry["proof_sha256"] != keys["proof_sha256"]:
                layers.failures.append(
                    f"S {name}: manifest expected hash != CSV proof_sha256")
            if not regenerate_liswet:
                for layer in ("A", "B", "C", "E"):
                    layers.record(layer, "skip", name,
                                  "LISWET proof not committed; use --regenerate")
                continue
            payload, why = regenerate(entry, root, data_dir)
            if payload is None:
                layers.record("A", "fail", name, why)
                for layer in ("B", "C", "E"):
                    layers.record(layer, "skip", name, "regeneration failed")
                continue
        else:
            proof_path = csv_path.parent / keys["proof"]
            if not proof_path.is_file():
                layers.record("A", "fail", name, f"proof file missing: {keys['proof']}")
                for layer in ("B", "C", "E"):
                    layers.record(layer, "skip", name, "proof file missing")
                continue
            try:
                payload = expand_proof(proof_path)
            except PROOF_ERRORS as exc:
                layers.record("A", "fail", name, f"proof unreadable: {exc}")
                for layer in ("B", "C", "E"):
                    layers.record(layer, "skip", name, "proof unreadable")
                continue

        if payload.get("checker") != checker.CHECKER_VERSION:
            layers.record("A", "fail", name, f"proof checker version {payload.get('checker')!r}")
        elif normalized_hash(payload) != keys["proof_sha256"]:
            layers.record("A", "fail", name, "canonical proof hash != CSV proof_sha256")
        else:
            layers.record("A", "pass", name)

        try:
            err = bounds_error(payload, value)
        except (KeyError, ValueError, TypeError) as exc:
            layers.record("B", "fail", name, f"bounds unreadable: {exc}")
        else:
            if err > checker.REFERENCE_TOLERANCE:
                layers.record("B", "fail", name, f"value outside bounds budget ({float(err):.3e})")
            else:
                layers.record("B", "pass", name)

        qps = data_dir / (name + ".QPS")
        if not data_present or not qps.is_file():
            layers.record("C", "skip", name, "QPS data not available")
        elif hashlib.sha256(qps.read_bytes()).hexdigest() != keys["input_sha256"]:
            layers.record("C", "fail", name, "QPS sha256 != CSV input_sha256")
        elif payload.get("input_sha256") != keys["input_sha256"]:
            layers.record("C", "fail", name, "proof input hash != CSV input_sha256")
        else:
            layers.record("C", "pass", name)

        if truth is None or not data_present or not qps.is_file():
            layers.record("E", "skip", name, "QPS data or checker deps not available")
        else:
            try:
                checker.replay_evidence(truth.load_qps(qps), payload, keys["input_sha256"])
                layers.record("E", "pass", name)
            except Exception as exc:  # any rejection by the checker is a proof failure
                layers.record("E", "fail", name, f"replay rejected: {exc}")

    for layer in sorted(require_layers):
        stat = layers.count[layer]
        if stat["skip"]:
            layers.failures.append(
                f"R {layer}: required layer reported {stat['skip']} skip(s) "
                f"[{'; '.join(sorted(layers.skips[layer]))}]")
        if stat["pass"] == 0:
            # A required layer that never ran a single check (empty or fully
            # uncertified CSV) is not "passing", it is an absent guard.
            layers.failures.append(
                f"R {layer}: required layer produced 0 passing checks")
    if len(certified) < min_certified:
        layers.failures.append(
            f"R min-certified: {len(certified)} certified claims < required "
            f"minimum {min_certified} (inventory shrink must be a reviewed decision)")

    stamped_current = sum(1 for _, keys in certified.values()
                          if keys["checker_sha256"] == checker.checker_digest())

    for layer in layers.names:
        stat = layers.count[layer]
        line = f"{layer}: {stat['pass']} pass, {stat['fail']} fail, {stat['skip']} skip"
        if stat["skip"]:
            line += " [" + "; ".join(sorted(layers.skips[layer])) + "]"
        print(line)
    print(f"D (info): {stamped_current}/{len(certified)} claims stamped with the current "
          f"checker digest, {len(certified) - stamped_current} against an older tree "
          "(rerun revalidation to refresh)")
    if layers.count["C"]["skip"] or layers.count["E"]["skip"]:
        print("note: C/E skips here are only acceptable in the hermetic audit job "
              "(--require-layers S). Certificate integrity requires the data job "
              "run: --regenerate --require-layers S,A,B,C,E.")
    for line in layers.failures[:40]:
        print("FAIL", line)
    if layers.failures:
        return 1
    print(f"OK: {len(certified)} certified claims checked, {uncertified} uncertified rows labelled")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path,
                    default=ROOT / "data" / "baseline_objectives" / "maros_meszaros.csv")
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="repo root; LISWET regeneration runs <root>/" + REGEN_SCRIPT)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="QPS directory for the input-hash and replay layers "
                         "(default: <root>/data/maros_meszaros)")
    ap.add_argument("--regenerate", action="store_true",
                    help="regenerate uncommitted LISWET proofs from the manifest "
                         "(needs the QPS data and the cert-suite pip deps)")
    ap.add_argument("--require-layers", default="",
                    help="comma list of layers that must run with zero skips "
                         "and at least one passing check "
                         "(audit job: S; data job: S,A,B,C,E)")
    ap.add_argument("--min-certified", type=int, default=0,
                    help="fail if the CSV carries fewer certified claims than "
                         "this floor (CI jobs pass 88)")
    args = ap.parse_args(argv)
    if args.min_certified < 0:
        print("FATAL: --min-certified must be >= 0")
        return 1
    require = {part.strip().upper() for part in args.require_layers.split(",") if part.strip()}
    unknown = require - {"S", "A", "B", "C", "E", "D"}
    if "D" in require:
        print("FATAL: layer D is informational and cannot be required")
        return 1
    if unknown:
        print(f"FATAL: unknown --require-layers entries: {sorted(unknown)}")
        return 1
    data_dir = args.data_dir or args.root / "data" / "maros_meszaros"
    try:
        return check(args.csv, args.root, data_dir, args.regenerate, require,
                     args.min_certified)
    except Exception as exc:  # fail closed with a diagnosis, never a traceback
        print(f"FATAL: verifier aborted fail-closed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
