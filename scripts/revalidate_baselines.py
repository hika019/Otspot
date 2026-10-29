#!/usr/bin/env python3
"""Recheck every CSV row and retain replayable rational evidence for accepted rows."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_certificate import (LOADED_CHECKER_DIGEST, ExactModel, checker_digest,
                                  reference_error, REFERENCE_TOLERANCE, save_evidence)


class CheckerCodeChanged(RuntimeError):
    """The checker code differs from the digest pinned at the start of the run."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_gaps(paths):
    records = {}
    for path in paths:
        text = Path(path).read_text()
        try:
            items = json.loads(text)
            if isinstance(items, dict):
                items = [items]
        except json.JSONDecodeError:
            items = [json.loads(line) for line in text.splitlines() if line.strip()]
        for item in items:
            records[item["problem"]] = item["gaps"]
    return records


def evaluate_row(config, row, expected_digest):
    import numpy as np
    import highs_baseline_truth as truth
    import liswet_exact_certificate as liswet

    name = row["problem_name"]
    path = Path(config["data_dir"]) / (name + ".QPS")
    input_hash = sha256(path)
    prob = truth.load_qps(path)
    model = ExactModel(prob)
    result = dict(problem_name=name, old_value=row["optimal_obj"], old_source=row["source"],
                  reference=row["optimal_obj"], accepted=False, input_sha256=input_hash,
                  checker_sha256=expected_digest, reason="no_verified_enclosure")
    verdict = truth._psd_ok(prob["Q"])
    if verdict is not True:
        result["reason"] = "nonconvex" if verdict is False else "psd_unknown"
        return result
    proposal = None
    if name.startswith("LISWET"):
        cert = liswet.certify(path, gaps=config["gaps"].get(name),
                             time_limit=config["timeout"])
        enclosure = cert.enclosure
        reference = liswet.format_rational_csv(cert.objective)
        proposal = dict(method="liswet_banded", gaps=list(cert.gaps))
    else:
        candidate_path = Path(config["candidate_dir"]) / (name + ".npz")
        metadata_path = candidate_path.with_suffix(".json")
        if not candidate_path.exists() or not metadata_path.exists():
            result["reason"] = "missing_candidate"
            return result
        proposal = json.loads(metadata_path.read_text())
        if proposal.get("input_sha256") != input_hash:
            result["reason"] = "candidate_input_hash_mismatch"
            return result
        with np.load(candidate_path, allow_pickle=False) as vectors:
            x, y = vectors["x"], vectors["y"]
        if x.shape != (model.n,) or y.shape != (model.m,) or not (
                np.isfinite(x).all() and np.isfinite(y).all()):
            result["reason"] = "invalid_candidate"
            return result
        cert = truth.best_certificate(prob, x, y)
        enclosure = cert.enclosure
        reference = truth.format_csv_value(cert.p)
        result["method"] = cert.method
    if enclosure is None:
        return result
    if reference_error(enclosure, reference) > REFERENCE_TOLERANCE:
        result["reason"] = "enclosure_too_wide"
        return result
    if sha256(path) != input_hash:
        raise ValueError("input changed during verification")
    # Keep a published reference only when this run proves that exact string too.
    if reference_error(enclosure, row["optimal_obj"]) <= REFERENCE_TOLERANCE:
        reference = row["optimal_obj"]
    if checker_digest() != expected_digest:
        raise CheckerCodeChanged("checker code changed during verification")
    compact = save_evidence(Path(config["output_dir"]) / (name + ".proof.json.gz"),
                            enclosure, reference, input_hash, proposal, expected_digest)
    result.update(compact)
    result.update(accepted=True, reason="certified", reference=reference)
    return result


def worker(config_path, name):
    config = json.loads(Path(config_path).read_text())
    row = next(row for row in config["rows"] if row["problem_name"] == name)
    output = Path(config["output_dir"])
    started = time.monotonic()
    result = None
    try:
        expected = config["checker_sha256"]
        if LOADED_CHECKER_DIGEST != expected or checker_digest() != expected:
            raise CheckerCodeChanged("imported code differs from the digest pinned at run start")
        result = evaluate_row(config, row, expected)
        if checker_digest() != expected:
            raise CheckerCodeChanged("checker code changed during verification")
    except Exception as exc:
        reason = "checker_code_changed" if isinstance(exc, CheckerCodeChanged) else "error"
        result = dict(problem_name=name, old_value=row["optimal_obj"],
                      old_source=row["source"], reference=row["optimal_obj"],
                      accepted=False, reason=reason, error=f"{type(exc).__name__}: {exc}")
    result["elapsed_seconds"] = time.monotonic() - started
    # An unaccepted row must not leave a proof file behind as if it were verified.
    if not result["accepted"]:
        (output / (name + ".proof.json.gz")).unlink(missing_ok=True)
    write_json(output / (name + ".result.json"), result)


def run_row(config_path, row, output, timeout):
    name = row["problem_name"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.monotonic()
    reason = None
    with (output / (name + ".log")).open("w") as log:
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                      "--worker", str(config_path), name],
                                     stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout)
            if process.returncode:
                reason = "worker_failed"
        except subprocess.TimeoutExpired:
            reason = "verification_timeout"
    if reason is None:
        result = json.loads((output / (name + ".result.json")).read_text())
    else:
        # Same contract as the worker: no accepted row, no adjacent proof file.
        (output / (name + ".proof.json.gz")).unlink(missing_ok=True)
        result = dict(problem_name=name, old_value=row["optimal_obj"], old_source=row["source"],
                      reference=row["optimal_obj"], accepted=False, reason=reason,
                      elapsed_seconds=time.monotonic() - started)
        write_json(output / (name + ".result.json"), result)
    print(f"{name}: {result['reason']} ({result['elapsed_seconds']:.3f}s)", flush=True)
    return result


def output_source(result):
    # Repeated failed runs never retain the old source's certification claim.
    if result["accepted"]:
        return "exact_enclosure_v1"
    return f"revalidation_{result['reason']}_UNCERTIFIED"


def invalidate(result, reason):
    return dict(problem_name=result["problem_name"], old_value=result["old_value"],
                old_source=result["old_source"], reference=result["old_value"],
                accepted=False, reason=reason)


def finalize(results, output, start_digest):
    """Write the run outputs under a code freeze: digest drift voids every row."""
    frozen = checker_digest() == start_digest
    if not frozen:
        for result in results:
            if result.get("proof"):
                (output / result["proof"]).unlink(missing_ok=True)
        results = [invalidate(result, "checker_code_changed") for result in results]
    with (output / "baseline.csv").open("w") as handle:
        handle.write("# Exact binary64 model enclosures; evidence in adjacent *.proof.json.gz.\n")
        writer = csv.writer(handle)
        writer.writerow(["problem_name", "optimal_obj", "source"])
        for result in results:
            writer.writerow([result["problem_name"], result["reference"], output_source(result)])
    summary = dict(total=len(results), certified=sum(r["accepted"] for r in results),
                   results=results, checker_sha256=start_digest, code_frozen=frozen)
    write_json(output / "summary.json", summary)
    print(f"{summary['certified']}/{summary['total']} certified", flush=True)
    return 0 if frozen else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", nargs=2, metavar=("CONFIG", "NAME"))
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--candidate-dir", type=Path)
    ap.add_argument("--gaps-json", action="append", default=[])
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--timeout", type=float, default=1000)
    args = ap.parse_args(argv)
    if args.worker:
        worker(*args.worker)
        return 0
    if not all((args.csv, args.data_dir, args.candidate_dir, args.output_dir, args.jobs)):
        ap.error("--csv, --data-dir, --candidate-dir, --output-dir, --jobs are required")
    if args.jobs < 1 or args.timeout <= 0:
        ap.error("jobs and timeout must be positive")
    rows = read_rows(args.csv)
    names = [r["problem_name"] for r in rows]
    if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        ap.error("duplicate or invalid problem name")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = dict(rows=rows, data_dir=str(args.data_dir.resolve()),
                  candidate_dir=str(args.candidate_dir.resolve()), output_dir=str(output),
                  gaps=read_gaps(args.gaps_json), timeout=args.timeout,
                  checker_sha256=checker_digest())
    config_path = output / "run.json"
    write_json(config_path, config)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda row: run_row(config_path, row, output, args.timeout), rows))
    return finalize(results, output, config["checker_sha256"])


if __name__ == "__main__":
    sys.exit(main())
