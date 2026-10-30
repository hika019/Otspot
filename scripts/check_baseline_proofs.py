#!/usr/bin/env python3
"""Offline verifier for the committed Maros-Meszaros proof claims (PR #51 review P1-1).

The CSV source column is only honest if somebody can check it without rerunning
the full certification pipeline. This script is that somebody. It resolves every
claim in a baseline CSV and checks it in independent layers:

  S  shape (always, hermetic): certified sources carry exactly the four claim
     keys with 64-hex digests, and proof paths stay under the proofs/
     directory; LISWET claims resolve to proofs/liswet/manifest.json, which
     must exist and carry per-row gaps, regeneration args, and an expected hash
     equal to the CSV stamp; uncertified rows must end in _UNCERTIFIED and
     begin with the pre-contract claim marker defined by the contract.
  A  hash (hermetic when the proof entity is committed): SHA-256 over the
     canonical expanded proof JSON equals the CSV proof_sha256. The canonical
     form is json.dumps(payload, sort_keys=True, separators=(',',':'),
     ensure_ascii=False).encode('utf-8'), so gzip timestamps cannot move it
     and the hash survives lossless recompression. LISWET proofs are not
     committed; layer A covers them only under --regenerate.
  B  value (runs whenever a payload resolved): the CSV optimal_obj lies inside
     the proof bounds within REFERENCE_TOLERANCE (recomputed here, not read
     from the proof's own relative_error field).
  C  input (data-gated): the CSV input_sha256 equals SHA-256 of data/*.QPS and
     of the hash stored inside the proof.
  E  replay (data-gated): checker.replay_evidence recomputes the enclosure
     from the witness; bounds forged into a proof cannot survive this even if
     every hash is self-consistent.
  D  checker drift (informational, never fails the run): rows whose
     checker_sha256 stamp differs from the current checker_digest() were
     verified against an older tree; rerun revalidation, not this verifier,
     when that happens.

Layers C and E print an explicit SKIP reason when data (or checker deps) are
absent, so a passing run is never ambiguous about what was actually checked.
Any FAIL exits 1; SKIP is reported per layer and never counted as pass.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import lzma
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
    return json.loads(expanded.decode("utf-8"))


def _is_hex64(text: str) -> bool:
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def read_rows(csv_path: Path) -> list[dict]:
    lines = [line for line in csv_path.read_text().splitlines()
             if not line.startswith("#") and line.strip()]
    return list(csv.DictReader(lines))


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
    """Independent budget recomputation: matches reference_error semantics."""
    lo, up = Fraction(payload["lower"]), Fraction(payload["upper"])
    worst = Fraction(0)
    for r in (Fraction(value), Fraction(float(value))):
        worst = max(worst, max(abs(r - lo), abs(r - up)) / (1 + abs(r)))
    return worst


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
    if not path.is_file():
        layers.record("S", "fail", LISWET_MANIFEST, "manifest missing for LISWET claims")
        return {}
    try:
        entries = json.loads(path.read_text())["rows"]
    except (ValueError, KeyError) as exc:
        layers.record("S", "fail", LISWET_MANIFEST, f"manifest unreadable: {exc}")
        return {}
    rows = {}
    for name in sorted(problems):
        entry = entries.get(name)
        if entry is None:
            layers.record("S", "fail", name, "no manifest entry")
            continue
        malformed = not (
            isinstance(entry.get("gaps"), list)
            and all(isinstance(g, int) and not isinstance(g, bool) and g >= 1
                    for g in entry["gaps"])
            and isinstance(entry.get("args"), list) and bool(entry["args"])
            and _is_hex64(str(entry.get("proof_sha256", "")))
            and bool(entry.get("reference")))
        if malformed:
            layers.record("S", "fail", name, "manifest entry malformed")
            continue
        rows[name] = entry
    return rows


def regenerate(entry: dict, root: Path):
    """Run the manifest command into a throwaway dir and expand its proof."""
    stem = Path(str(entry["args"][-1])).stem
    with tempfile.TemporaryDirectory(prefix="proofcheck-") as tmp:
        argv = [part.replace("{evidence_dir}", tmp) for part in entry["args"]]
        try:
            subprocess.run(argv, cwd=root, check=True, capture_output=True, timeout=3600)
        except (subprocess.SubprocessError, OSError) as exc:
            return None, f"regeneration failed: {exc}"
        proof = Path(tmp) / (stem + ".proof.json.gz")
        if not proof.is_file():
            return None, "regeneration produced no proof file"
        try:
            return expand_proof(proof), ""
        except PROOF_ERRORS as exc:
            return None, f"regenerated proof unreadable: {exc}"


def check(csv_path: Path, root: Path, data_dir: Path, regenerate_liswet: bool) -> int:
    layers = Layers()
    rows = read_rows(csv_path)
    proofs_dir = csv_path.parent / PROOF_SUBDIR

    certified: dict[str, tuple[dict, dict]] = {}
    liswet_rows: set[str] = set()
    uncertified = 0
    for row in rows:
        name, source = row["problem_name"], row["source"]
        if source.endswith("_UNCERTIFIED"):
            # A retained pre-contract value is only honest while it says so.
            if source.startswith(UNCERTIFIED_CLAIM + ";"):
                uncertified += 1
                layers.count["S"]["pass"] += 1
            else:
                layers.record("S", "fail", name, "uncertified row lacks the claim marker")
            continue
        keys = parse_claim(source)
        if keys is None:
            layers.record("S", "fail", name, "source is neither a claim nor _UNCERTIFIED")
            continue
        bad = [key for key in ("proof_sha256", "input_sha256", "checker_sha256")
               if not _is_hex64(keys[key])]
        proof = keys["proof"]
        if bad:
            layers.record("S", "fail", name, f"non-hex64 claim digests: {bad}")
            continue
        if proof.startswith("/") or ".." in Path(proof).parts \
                or not proof.startswith(PROOF_SUBDIR + "/"):
            layers.record("S", "fail", name, f"proof path escapes {PROOF_SUBDIR}/: {proof}")
            continue
        certified[name] = (row, keys)
        if proof == LISWET_MANIFEST:
            liswet_rows.add(name)
        layers.count["S"]["pass"] += 1

    manifest_rows = load_manifest(proofs_dir / "liswet" / "manifest.json",
                                  liswet_rows, layers) if liswet_rows else {}

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
            if entry is not None:
                if entry["reference"] != value:
                    layers.record("S", "fail", name,
                                  f"manifest reference {entry['reference']} != CSV {value}")
                if entry["proof_sha256"] != keys["proof_sha256"]:
                    layers.record("S", "fail", name,
                                  "manifest expected hash != CSV proof_sha256")
            if not regenerate_liswet:
                for layer in ("A", "B", "C", "E"):
                    layers.record(layer, "skip", name,
                                  "LISWET proof not committed; use --regenerate")
                continue
            payload, why = regenerate(entry, root)
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
        except (KeyError, ValueError) as exc:
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
                    help="repo root used to run LISWET regeneration commands")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="QPS directory for the input-hash and replay layers "
                         "(default: <root>/data/maros_meszaros)")
    ap.add_argument("--regenerate", action="store_true",
                    help="regenerate uncommitted LISWET proofs from the manifest "
                         "(needs the QPS data and the cert-suite pip deps)")
    args = ap.parse_args(argv)
    data_dir = args.data_dir or args.root / "data" / "maros_meszaros"
    return check(args.csv, args.root, data_dir, args.regenerate)


if __name__ == "__main__":
    raise SystemExit(main())
