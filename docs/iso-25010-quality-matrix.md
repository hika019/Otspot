# ISO/IEC 25010 product quality matrix for v0.7

Scope: otspot Rust workspace (`otspot`, `otspot-core`, `otspot-io`,
`otspot-model`, `otspot-dev`) as a solver library, CLI/benchmark harness, and
published crates. This is a repository gate map, not a certification claim.

| ISO/IEC 25010 characteristic | v0.7 repo evidence | Release gate before v0.7 | Remaining gap |
| --- | --- | --- | --- |
| Functional suitability | Solver API contracts, LP/QP/MIP regressions, Netlib/QPLIB/MPS/QPS parser fixtures, Clarabel cross-checks, proptest fuzzing. | Pre-merge (`scripts/pre-merge-audit.sh`): `cargo nextest run --release --features parallel --test-threads 6`; CI (`ci.yml` job `test-integration`): `cargo nextest run --release --features parallel --no-fail-fast --test-threads 3`; `cargo test --doc --release` (pre-merge; the CI `docs` job runs `cargo test --doc` against the debug lib); `python3 tests/test_check_data_coverage.py` (pre-merge). | External oracle coverage is still uneven for MIP and nonconvex QP; keep expanding checked benchmark manifests when new regressions are found. |
| Performance efficiency | Timing sentinels for postsolve, simplex stalls, bench timeout handling, nextest slow-timeout limits, memory regression tests. | Targeted sentinel selectors over the release suite (`-E` subset probes layered on the required stage commands below, not standalone gate steps): full-suite subset `cargo nextest run --release --features parallel -E 'binary(memory_regression) | binary(diag_bench_timeout_honored) | binary(diag_dfl001_postsolve_speedup)' --test-threads 6`; ignored heavy subset probe on top of the required heavy stage (local `[profile.heavy]` default 6 threads; CI `test-heavy.yml` must-pass runs the full heavy command, no `-E`, with `--test-threads 3`): `cargo nextest run --release --features parallel --profile heavy --run-ignored all --no-fail-fast -E 'binary(diag_lp_simplex_stall_sentinel)' --test-threads 6`; the required stages themselves: `scripts/pre-merge-audit.sh`, `.github/workflows/test-heavy.yml`. | Wall-clock sentinels are machine-sensitive; ignored heavy surveillance is a required stage of `scripts/pre-merge-audit.sh` (a `set -e` failure aborts the audit) and the must-pass job of `.github/workflows/test-heavy.yml`; `scripts/check_test_data_requirements.py` gates its data before the stage runs. |
| Compatibility | Public API diff and published package checks cover downstream crate compatibility; feature checks cover default/no-default/all-features builds. | `.github/workflows/ci.yml` jobs `public-api`, `package`, and `compatibility`. | No SemVer policy document beyond public API diff; add one if external consumers require stricter API lifecycle guarantees. |
| Usability | README examples, doctests, model API tests, solver-wide API contract tests, CLI parse fixtures. | `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`; `cargo test --doc --release` (pre-merge) and `cargo test --doc` (CI docs job, debug lib); API-surface selectors over the release suite (`-E` subset probes, not standalone CI steps): `cargo nextest run --release -E 'test(model_api) | binary(api_correctness) | binary(solver_wide_api_contract)'`. | Error-message ergonomics are regression-tested only indirectly; add targeted diagnostics tests when user-facing failures are triaged. |
| Reliability | Deadline/timeout tests, numerical regression fixtures, infeasible/unbounded certificates, no silent Timeout-to-Optimal promotion, heavy convergence surveillance. | Pre-merge (`scripts/pre-merge-audit.sh`, `set -e`): `cargo nextest run --release --features parallel --test-threads 6` plus the required heavy stage `cargo nextest run --release --features parallel --profile heavy --run-ignored all --no-fail-fast --test-threads 6`; CI runs the same heavy command with `--test-threads 3` (`.github/workflows/test-heavy.yml`); `.config/nextest.toml` slow-timeout. | Heavy convergence tests require large data and long runtimes (~50 min), so they run last in `scripts/pre-merge-audit.sh` after the cheap gates, but they are required under `set -e`, not informational. |
| Security | Rust dependency audit, no network/download bypass for Netlib decoder, no TODO/HACK markers, package excludes private data and internal tests. | `.github/workflows/audit.yml` `cargo audit`; `scripts/pre-merge-audit.sh`; `cargo package --workspace --no-verify`. | This is not a hardened service boundary; no fuzzing of maliciously large binary inputs beyond parser/resource regressions. |
| Maintainability | Clippy deny-warnings, rustdoc warnings, file-size gate, comment ratio/block gates, memo-comment grep gate, data coverage sentinel. | `cargo clippy --workspace --all-targets --all-features -- -D warnings`; `bash scripts/check_file_size.sh`; `bash scripts/check_comment_block_size.sh`; `bash scripts/check_comment_ratio.sh`; `bash scripts/lib/check_memo_grep.sh`. | Architecture review remains human judgment; gates only catch size/comment/dependency drift. |
| Portability | Rust 1.95 pinned toolchain, Linux full CI, lightweight cargo checks on Ubuntu/macOS/Windows. | `.github/workflows/ci.yml` job `portability` (ubuntu/macos/windows): `cargo check --workspace --all-targets --all-features`; the no-default-features variant runs in the `compatibility` job: `cargo check --workspace --all-targets --no-default-features`. | Runtime solver behavior is primarily validated on Linux; add platform-specific runtime tests if Windows/macOS users report solver differences. |

## v0.7 lightweight checklist

Run this before release branch merge:

```bash
bash scripts/pre-merge-audit.sh
```

The script already runs `cargo test --doc --release`, this
`python3 tests/test_iso_25010_quality_matrix.py`, and the required heavy stage
as its stages, so listing them again here only risks drift.

Do not add ISO process artifacts unless they create a concrete repository gate,
test, or release decision.
