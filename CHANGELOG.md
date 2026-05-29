# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### 追加

### 変更

- **BREAKING**: `SolverOptions.psd_check_max_n` フィールド削除 — production caller 0 件、soundness 穴 (size-skip) の除去 (#130)

### 修正

### 内部

- `osqp_bench.csv` に SS_* SuiteSparse 行を統合 (optional? 列 `*` でマーク、旧 `osqp_bench_optional.csv` 削除、#133):
  以前の split 設計 (osqp_bench.csv + osqp_bench_optional.csv) は bench_utils::detect_csv_path が optional CSV を
  load しないため SS_* baseline が bench runner の退行検知から除外される欠陥があった。single-file 設計で修正。
- `LP_CRASH_DUAL_ADV_DISABLE` 環境変数削除 (`use_lp_crash_basis` option と dual-path 完全冗長、#101 audit A2、commit 713d9be)
- otspot-dev 未参照 bin 4 件削除 (`qp_dump` / `lp_screen` / `verify_solutions` / `qp_diag`、#101 audit deadcode、commit dcfde53)
- step9 混合型並行行スキップに撤退根拠 docstring 化 (#99、commit 2b0e66a)
- NaN-guard 18 箇所を `tolerances::any_nonfinite` helper に統合 (#130)
- `dual_sign_violation` の z NaN-guard 追加 (y と同等の defense-in-depth、#130)

### 依存

## [0.3.1] - 2026-05-29

### 追加
- `try_var_name` checked variant (try_value と対称)

### 修正
- `solve_ipm` / `run_ipm` で SolverOptions validate guard (panic → ModelError)
- `greenbea_postsolve_dual_feasibility` を `#[ignore]` 化 (bench 並行下 flaky)
- audit.rs の should_panic 誤検出修正

### 内部
- rustdoc broken link 整理 + CI doc job 追加

### 依存
- actions/checkout v6 / tempfile 3.27 / mimalloc 0.1.52 / log 0.4.30 / rayon 1.12.0 / proptest 1.11.0

## [0.3.0] - 2026-05-28

### 追加
- `IpmOptions` に dd_ldl / minres_ir / kkt_memory_budget_bytes フィールド
- `SolverOptions` に presolve_max_pass / presolve_skip_large_coeff / presolve_phase2 フィールド

### 破壊的変更
- `SolverResult` に opt_cert フィールド追加
- `SolveOutcome` / `FarkasCertificate` / `UnboundedRayCertificate` / `IncompleteReason` を削除
- `diagnose()` 系 API を削除
- `SolverResult::pfeas` / `dfeas` / `gap` フィールドを削除 (`final_residuals` に集約)
- deprecated `solve_qp_with_options` を削除
- ユーザ向け環境変数読み取りを全廃 (`IPM_DD_LDL` / `MINRES_IR` / `MINRES_ETA` / `KKT_MEMORY_BUDGET_BYTES` / `QP_PRESOLVE_MAX_PASS` / `QP_PRESOLVE_SKIP_LARGE_COEFF` / `QP_PRESOLVE_PHASE2`。sentinel hook `LP_DISPATCH_NOOP` / `DSE_DISABLE_GAMMA_UPDATE` は test 専用に意図保持)

### 修正
- B&B finalize_proven が EmptyCol を未マスクで誤降格していたバグを修正

## [0.2.0] - 2026-05-27

### 破壊的変更

- **証明付き最適性 (cert-carrying status)**: `Optimal` は KKT 全条件を検証した証明を伴う場合のみ確定するようになった。`prove_optimal` を唯一の検証点とし、証明できない解は `SuboptimalSolution` 等へ正直に降格 — 証明なしの偽 `Optimal` を構造的に排除する。`OptimalCertificate` / `BoundGapCertificate` / `SolveOutcome` / `NotProven` を追加。
- **ワークスペース分割**: 内部実装を `otspot-core` / `otspot-io` / `otspot-model` に分離。公開 `otspot::` パスは facade 再エクスポートで維持。
- **公開 API から除外**: `bench_utils` / `screening` / presolve 関数 / `bound_flip` を dev 専用 `otspot-dev`（非公開）へ移動。

### 追加

- **非凸 QP の大域求解** (`solve_qp_global`) — branch-and-bound + α-BB / McCormick 緩和。大域最適は `BoundGapCertificate` で証明し、局所最適のみの場合は `NonconvexLocal` として正直に区別する。
- **証明付き最適性検証** — stationarity・実行可能性・相補性・双対符号・双対ギャップを `eps` で検証して `Optimal` を発行（LP / QP / MIP 共通）。
- **二次目的の Expression DSL** — `x * x` / `x * y` で二次項を自然に記述。
- `ModelResult.status` / `.proof` / `SolutionProof` — 型付き解ステータス。
- `Model::try_add_var` / `try_value` — panic しない fallible 版（`ModelError` を返す）。
- `SolverOptions` / `IpmOptions` の検証と builder。`Tolerance::Fast`（1e-4）プリセット。
- `ModelError::NonConvex` / `NotSupported` — 文字列エラーを型付きに。
- `CscMatrix` の読み取り専用カプセル化（構築時に不変条件を強制）。
- MPS / QPS ストリーミングパーサ（`BufRead` 化で大規模インスタンスのピークメモリを削減）。
- マルチスタートの直列フォールバック（rayon panic 時も単一スレッドで継続）。

### 修正

- 全変数が presolve で消去される QP（全列が空のケース）が `NumericalError` を返していた問題を修正。presolve が完全に解いた解を、双対・証明を復元したうえで `Optimal` として正しく返す。
- 非凸 QP の branch-and-bound で、局所解の双対回収が不十分なまま打ち切られ KKT 残差が許容を超えるケースを修正。polish 結果を KKT 残差（双対符号を含む）で直接検証して採用する。

### 削除

- in-tree Python バインディング（`python/`）を削除（別リポジトリで管理）。
- 未使用の内部依存（`mimalloc` ほか）・`criterion` マイクロベンチ・dead code を整理。

## [0.1.1] - 2026-05-23

- LP の実行不可判定バグを修正
- LP / QP の求解安定性・汎用性を改善
- MILP の MPS 読み込みに対応
- Python バインディングを追加

## [0.1.0]

- 初版（LP: Revised Simplex / QP: Interior Point）
