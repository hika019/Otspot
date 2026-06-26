# ISO/IEC 25010 product quality matrix for v0.7

Scope: otspot Rust workspace (`otspot`, `otspot-core`, `otspot-io`,
`otspot-model`, `otspot-dev`) as a solver library, CLI/benchmark harness, and
published crates. This is a repository gate map, not a certification claim.

| ISO/IEC 25010 characteristic | v0.7 repo evidence | Release gate before v0.7 | Remaining gap |
| --- | --- | --- | --- |
| Functional suitability | Solver API contracts, LP/QP/MIP regressions, Netlib/QPLIB/MPS/QPS parser fixtures, Clarabel cross-checks, proptest fuzzing. | `cargo nextest run --release --features parallel --no-fail-fast`; `cargo test --doc`; `python3 tests/test_check_data_coverage.py`. | External oracle coverage is still uneven for MIP and nonconvex QP; keep expanding checked benchmark manifests when new regressions are found. |
| Performance efficiency | Timing sentinels for postsolve, simplex stalls, bench timeout handling, nextest slow-timeout limits, memory regression tests. | Default release gate: `cargo nextest run --release --features parallel -E 'binary(memory_regression) | binary(diag_bench_timeout_honored) | binary(diag_dfl001_postsolve_speedup)' --test-threads 3`; ignored heavy surveillance: `cargo nextest run --release --features parallel --profile heavy --run-ignored all -E 'binary(diag_lp_simplex_stall_sentinel)' --test-threads 3`; `.github/workflows/test-heavy.yml`. | Wall-clock sentinels are machine-sensitive; ignored heavy surveillance is informational and must not be counted as the release gate. |
| Compatibility | Public API diff and published package checks cover downstream crate compatibility; feature checks cover default/no-default/all-features builds. | `.github/workflows/ci.yml` jobs `public-api`, `package`, and `compatibility`. | No SemVer policy document beyond public API diff; add one if external consumers require stricter API lifecycle guarantees. |
| Usability | README examples, doctests, model API tests, solver-wide API contract tests, CLI parse fixtures. | `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`; `cargo test --doc`; `cargo nextest run --release -E 'test(model_api) | binary(api_correctness) | binary(solver_wide_api_contract)'`. | Error-message ergonomics are regression-tested only indirectly; add targeted diagnostics tests when user-facing failures are triaged. |
| Reliability | Deadline/timeout tests, numerical regression fixtures, infeasible/unbounded certificates, no silent Timeout-to-Optimal promotion, heavy convergence surveillance. | `cargo nextest run --release --features parallel --no-fail-fast`; `.config/nextest.toml` slow-timeout; `.github/workflows/test-heavy.yml`. | Heavy convergence tests are non-gating surveillance because they require large data and long runtimes. |
| Security | Rust dependency audit, no network/download bypass for Netlib decoder, no TODO/HACK markers, package excludes private data and internal tests. | `.github/workflows/audit.yml` `cargo audit`; `scripts/pre-merge-audit.sh`; `cargo package --workspace --no-verify`. | This is not a hardened service boundary; no fuzzing of maliciously large binary inputs beyond parser/resource regressions. |
| Maintainability | Clippy deny-warnings, rustdoc warnings, file-size gate, comment ratio/block gates, memo-comment grep gate, data coverage sentinel. | `cargo clippy --workspace --all-targets --all-features -- -D warnings`; `bash scripts/check_file_size.sh`; `bash scripts/check_comment_block_size.sh`; `bash scripts/check_comment_ratio.sh`; `bash scripts/lib/check_memo_grep.sh`. | Architecture review remains human judgment; gates only catch size/comment/dependency drift. |
| Portability | Rust 1.95 pinned toolchain, Linux full CI, lightweight cargo checks on Ubuntu/macOS/Windows. | `.github/workflows/ci.yml` job `portability`; `cargo check --workspace --all-targets --no-default-features`; `cargo check --workspace --all-targets --all-features`. | Runtime solver behavior is primarily validated on Linux; add platform-specific runtime tests if Windows/macOS users report solver differences. |

## v0.7 lightweight checklist

Run this before release branch merge:

```bash
bash scripts/pre-merge-audit.sh
cargo test --doc
python3 tests/test_iso_25010_quality_matrix.py
```

Do not add ISO process artifacts unless they create a concrete repository gate,
test, or release decision.
