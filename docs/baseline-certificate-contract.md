# Baseline reference certificates

The baseline generator must establish an objective reference independently of
the solver under test. Certification targets relative error at most `1e-7`,
one tenth of the benchmark's default `1e-6` tolerance. The number of accepted
problems is a coverage result, not a correctness criterion.

## Model

The shared QPS parser defines one canonical model with binary64 coefficients,
including coefficient accumulation, selected named vectors, bounds, and the
objective constant. A variable with no BOUNDS entry defaults to `x >= 0`
(MPS convention). The checker interprets these binary64 values as exact
rational numbers. Every candidate generator, including the LISWET specialization,
must refer to this same model. Decimal-token and binary64 certificates must not
be mixed as certificates of an identical model.

Two thresholds are part of the model contract: candidate primal points enter
certification only at relative violation at most `1e-7` (`PRIMAL_INF_TOL`), and
a retained CSV claim requires serialized reference error at most `1e-7`
(`REFERENCE_TOLERANCE`, pinned by `tests/test_certificate_contract_adversarial.py`).

## Proof

For a minimization problem, the checker establishes an interval `[L, U]`:

1. Prove that the symmetric Hessian is positive semidefinite. A tolerance on
   negative eigenvalues is not a proof. An exhausted proof budget is unknown.
2. Check every original row and variable bound for a rational primal point.
   Its exactly recomputed objective is an upper bound `U`.
3. Independently recompute reduced costs from a dual witness and linearization
   point, check multiplier and bound signs, and evaluate a rational weak-duality
   lower bound `L`.
4. Require `L <= U`. Exact KKT equality is a special case, not a requirement.
5. For the serialized CSV reference `r`, require
   `max(abs(r - L), abs(r - U)) / (1 + abs(r)) <= 1e-7`, accounting for both the
   decimal representation and its binary64 interpretation by benchmark readers.

The primal point and dual linearization point may differ. Solver status,
small residuals, active-set guesses, multiplier-weighted violations, and
producer-supplied reduced costs are not substitutes for these checks.

## Candidate search and provenance

Candidate generation may use numerical solvers, polishing, or structured exact
recovery. Residual and sensitivity scores only order unverified candidates.
Verified enclosures take priority, and successful termination depends on their
certified error. A failed recovery attempt cannot make an unverified value valid.

Each retained certification claim must be traceable to the input hash, checker
version, bounds, and reference checked. Revalidation compares every row with its
previous value. Failure or unknown status must not silently preserve an old
certification label; neither implies that the previous numerical value is wrong.
An accepted row carries that provenance in its `source` column, format
`exact_enclosure_v1;proof=<path relative to the CSV's directory>;proof_sha256=
<hex>;input_sha256=<hex>;checker_sha256=<hex>`, with proofs stored in a `proofs/`
subdirectory beside the CSV. A retained accepted row missing any of these fields
is a run error; the writer fails closed instead of downgrading silently.
Uncertified rows are labelled, not dropped; see Evidence storage below.

## Evidence storage and the offline verifier

Committed proof entities live in a `proofs/` subdirectory beside the CSV. Exact
rational witnesses are megabyte-scale for the LISWET family, so the 12 LISWET
proofs are not committed: their `proof=` resolves to
`proofs/liswet/manifest.json`, whose per-row entries are structured data only —
`problem`, `gaps`, `qps` (which must equal `<problem>.QPS`), `proof_sha256`,
`reference`, with any other key a verifier fail. An earlier schema carried a
free-form regeneration command; under `--regenerate` that was arbitrary-argv
execution of committed data, so it was removed: the verifier builds argv itself
(`sys.executable` + `scripts/liswet_exact_certificate.py` + manifest gaps + the
QPS file under `--data-dir`) and executes nothing else. A certified row's
`proof=` therefore resolves either to a committed proof entity (stored as
`.lzma`; the storage codec is free because `proof_sha256` is defined over
expanded content, and the checker's native writer produces gzip) or to that
manifest, and never leaves `proofs/`.

`proof_sha256` is defined over the canonical expanded content: SHA-256 of
`json.dumps(payload, sort_keys=True, separators=(',',':'), ensure_ascii=False).encode('utf-8')`.
It is stable under gzip timestamps and member headers and survives lossless
recompression, so the storage codec can change without reissuing the CSV.
Regeneration must be deterministic under this hash: every LISWET proof was
generated twice during the 2026-10-31 reissue and the canonical hashes were
compared (12/12 identical while the gzip bytes differed).

`scripts/check_baseline_proofs.py` is the verifier the format demands. Layer S
(shape: safe `problem_name` stems (writer-side `SAFE_NAME` plus no path
separators and no leading `-`, because names reach file paths and one argv
slot), set agreement between CSV rows and the pinned QPS stems in
`scripts/maros_meszaros_qps.sha256` — extra rows, dropped rows and renames are
all red —, duplicate names, header count claims, claim keys, proof paths,
exact manifest schema and set agreement with the CSV's LISWET rows) and layer
B (committed bounds against the CSV value, over both decimal and binary64
readings) run hermetically anywhere; layer A (canonical hash) additionally
covers the LISWET rows under `--regenerate`; layer C (input file hash against
the CSV and against the hash stored inside the proof) and layer E
(`replay_evidence` recomputation of the enclosure from the witness) run where
the QPS data is present. Skips are reported per layer and never counted as
checks; `--require-layers L,...` fails the run if a required layer skipped
anything or produced zero passing checks (a guard that never ran is not a
guard), and `--min-certified N` fails it if fewer than N certified claims
exist. CI placement: the audit job has no QPS data and runs
`--require-layers S --min-certified 88` — its C/E skips are by design. The
`baseline proof verification` job in `audit.yml` downloads the ci-subset,
cross-checks CSV rows against data files with `check_data_coverage.py`, and
must run `--regenerate --require-layers S,A,B,C,E --min-certified 88`: that
job is where the 12 LISWET claims are actually content-checked and where
forged proofs hit layer E, so if it is ever removed or unrequired, the audit
job's green is not a certificate integrity claim. The floor 88 is intentionally
hardcoded in both CI invocations and double-managed beside the 88/138 figures
in the README: inventory only grows through revalidation effort, and any change
that lowers the floor must pass PR review. Layer E is what a hash-only review
cannot defeat: bounds forged into a proof fail independent recomputation even
when every hash is self-consistent (the audit job alone cannot see this; see
the layer-C attack reproduced in `tests/test_check_baseline_proofs.py`).

Two digest honesty notes. The `checker_sha256` stamped in a committed row
records the checker tree the claim was last reissued and verified against;
proof payloads separately embed the digest of the code that generated them.
For the 76 non-LISWET proofs reissued on 2026-10-31 these differ (the frozen
run predates the framing-v2 digest scheme; verification semantics did not
change). The reissue stamps the current digest and the payload keeps the
generation-time value; the verifier reports such drift as information (layer D)
without failing, because stale provenance calls for revalidation, not for
deleting proofs. Layer D is a report, not a guard: the real defenses against
forgery are the checker's adversarial test suite plus verifier layer E, and
neither depends on D. A payload whose content hash or contract version
disagrees with the claim still fails layer A.

Uncertified rows keep their value for benchmark regression but must say what
they are:
`claim=pre_contract_numeric_reference_not_certified;revalidation_<reason>_UNCERTIFIED`.
That marker is verified (layer S), not decorative: an uncertified row without
it fails the verifier.

## Test coverage honesty

Known gaps a reader must not mistake for coverage:

1. Rejecting a false `Optimal` via aggregated row violations in `prove_optimal`
   is pinned only by two function-level unit tests (`certificate.rs`:
   `prove_optimal_rejects_accumulated_row_violation`,
   `prove_optimal_aggregate_feasibility_applies_at_lp_cert_tol`). No
   integration scenario through `solve_qp_with` exercises that path under the
   current data convention, so the integration route is uncovered.
2. The Python checker parser maps "negative UP bound with no LO" to
   `lo = -inf` (HiGHS/CPLEX convention) while the Rust QPS parser keeps the
   `x >= 0` default for that input. This is a known deviation from the "one
   canonical model" statement above; it is latent — a sweep of all 188
   Maros-Meszaros QPS files found no occurrence, and no certified value
   depends on it.
3. Revert demonstrations for the Python-side sentinels (R1-R9b, R10-R13, the
   stage-3b verifier mutations, and the stage-3c structured-manifest / layer
   requirement mutations) live only in the ephemeral temporary area
   outside this repository
   (`/tmp/revert_proof.sh`, `/tmp/revert_proof_p21b.sh`,
   `/tmp/revert_proof_stage3b.sh`, `/tmp/revert_proof_stage3c.sh` and their
   `/tmp/revert_case*.log`). The verifier detection paths these mutations
   targeted are pinned permanently by boundary fixtures in
   `tests/test_check_baseline_proofs.py`; a permanent CI job replaying the
   mutation scripts themselves is a separate change.
4. Cache-generation precision: `ci.yml` uses bench-data cache keys up to `v3`
   and `test-heavy.yml` a separate keyspace up to `v4`. A blanket "CI cache
   v4" is imprecise; `v4` names only the heavy-side generation.
5. The storage policy's cost: in default verification mode (no QPS data, no
   `--regenerate`) the 12 LISWET claims contribute skips, not checks, to
   layers A/B/C/E. A manifest agreement pins the expected hash, the gaps, and
   the QPS file name, not the proof content; content verification of those
   rows requires `--regenerate` together with the data and cert-suite
   dependencies — which the `baseline proof verification` CI job runs on every
   push and PR (locally measured 2m01s with the ci-subset warm). Committed
   non-LISWET proofs are fully checked up to replay wherever layer C/E can
   run.
