#!/usr/bin/env python3
"""Recheck every CSV row and retain replayable rational evidence for accepted rows."""
import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


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


def checker_digest():
    root = Path(__file__).parent
    files = ("baseline_certificate.py", "highs_baseline_truth.py",
             "liswet_exact_certificate.py", "revalidate_baselines.py")
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in files)).hexdigest()


def evaluate_row(config, row):
    import numpy as np
    import highs_baseline_truth as truth
    import liswet_exact_certificate as liswet
    from baseline_certificate import ExactModel, evidence, reference_error, REFERENCE_TOLERANCE

    name = row["problem_name"]
    path = Path(config["data_dir"]) / (name + ".QPS")
    input_hash = sha256(path)
    prob = truth.load_qps(path)
    model = ExactModel(prob)
    result = dict(problem_name=name, old_value=row["optimal_obj"], old_source=row["source"],
                  reference=row["optimal_obj"], accepted=False, input_sha256=input_hash,
                  checker_sha256=checker_digest(), reason="no_verified_enclosure")
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
    proof = evidence(enclosure, reference, input_hash)
    proof.update(checker_sha256=result["checker_sha256"], proposal=proposal)
    proof_path = Path(config["output_dir"]) / (name + ".proof.json.gz")
    with gzip.open(proof_path, "wt") as handle:
        json.dump(proof, handle, allow_nan=False)
    result.update(accepted=True, reason="certified", reference=reference,
                  lower=proof["lower"], upper=proof["upper"],
                  relative_error=proof["relative_error"],
                  proof=proof_path.name, proof_sha256=sha256(proof_path))
    return result


def worker(config_path, name):
    config = json.loads(Path(config_path).read_text())
    row = next(row for row in config["rows"] if row["problem_name"] == name)
    started = time.monotonic()
    try:
        result = evaluate_row(config, row)
    except Exception as exc:
        result = dict(problem_name=name, old_value=row["optimal_obj"],
                      old_source=row["source"], reference=row["optimal_obj"],
                      accepted=False, reason="error", error=f"{type(exc).__name__}: {exc}")
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(Path(config["output_dir"]) / (name + ".result.json"), result)


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
                  gaps=read_gaps(args.gaps_json), timeout=args.timeout)
    config_path = output / "run.json"
    write_json(config_path, config)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda row: run_row(config_path, row, output, args.timeout), rows))
    with (output / "baseline.csv").open("w") as handle:
        handle.write("# Exact binary64 model enclosures; evidence in adjacent *.proof.json.gz.\n")
        writer = csv.writer(handle)
        writer.writerow(["problem_name", "optimal_obj", "source"])
        for result in results:
            writer.writerow([result["problem_name"], result["reference"], output_source(result)])
    summary = dict(total=len(results), certified=sum(r["accepted"] for r in results), results=results,
                   checker_sha256=checker_digest())
    write_json(output / "summary.json", summary)
    print(f"{summary['certified']}/{summary['total']} certified", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
