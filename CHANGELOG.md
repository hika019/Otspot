# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-05-26

### Breaking Changes

- **Workspace split**: internal implementation moved to sub-crates (`otspot-core`, `otspot-io`, `otspot-model`). Public `otspot::` API paths are preserved via re-exports in the facade crate.
- **Removed from public API**: `otspot::bench_utils`, `otspot::screening`, presolve function root re-exports, `bound_flip` — these were dev/internal utilities and are now in `otspot-dev` (not published).

### Added

- `ModelResult.status` + `ModelResult.proof` + `SolutionProof` — typed solution status with proof obligations; all status paths covered by tests.
- `Model::try_add_var` / `Model::try_value` — fallible variants that return `ModelError` instead of panicking.
- `SolverOptions` / `IpmOptions` validation + builder ergonomics.
- `Tolerance::Fast` preset (1e-4) for rapid prototyping.
- `ModelError::NonConvex` / `ModelError::NotSupported` — typed error variants replacing string-based errors.
- `CscMatrix` read-only encapsulation — internal invariants enforced at construction; mutation only via controlled API.
- MPS/QPS streaming parser — replaces `read_to_string` buffering with `BufRead` for lower peak memory on large instances.
- Multistart serial fallback — `ThreadPool` panic in rayon fallback is caught and execution continues single-threaded.
- Solver entry `options` validation — `SolverOptions` validated before dispatching to LP/QP backends.

### Known Issues / Scope

- `core::io` module still retained as test-only source (`qps.rs`); full removal tracked in #29.
- Guard hook complete removal deferred to #15.

## [0.1.1] - 2026-05-23

- LP の実行不可判定バグを修正
- LP / QP の求解安定性・汎用性を改善
- MILP の MPS 読み込みに対応
- Python バインディングを追加

## [0.1.0]

- 初版（LP: Revised Simplex / QP: Interior Point）
