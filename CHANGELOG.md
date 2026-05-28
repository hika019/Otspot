# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### 追加

- **`IpmOptions` — KKT/MINRES 設定フィールド追加** (再現性改善):
  - `dd_ldl: bool` (default `false`) — TwoFloat (double-double, ~106-bit) LDL を使用。旧 `IPM_DD_LDL=1` 環境変数を廃止。
  - `minres_ir: Option<usize>` (default `None` → 0, max 10) — MINRES 反復精密化ラウンド数。旧 `MINRES_IR` 環境変数を廃止。
  - `kkt_memory_budget_bytes: Option<usize>` (default `None` → 4 GiB) — KKT LDL 因子化メモリ上限。旧 `KKT_MEMORY_BUDGET_BYTES` 環境変数を廃止。
- **`SolverOptions` — QP presolve 設定フィールド追加**:
  - `presolve_max_pass: usize` (default `10`) — QP presolve 固定点反復上限。旧 `QP_PRESOLVE_MAX_PASS` 環境変数を廃止。
  - `presolve_skip_large_coeff: bool` (default `false`) — 大係数行スケーリングをスキップ。旧 `QP_PRESOLVE_SKIP_LARGE_COEFF` 環境変数を廃止。
  - `presolve_phase2: bool` (default `true`) — QP presolve phase 2 を有効化。旧 `QP_PRESOLVE_PHASE2=0` 環境変数を廃止。
- **`KktConfig` 構造体** (`linalg::kkt_solver`) — KKT 因子化設定を型付きで伝搬する内部型。

### 非互換性 (旧環境変数)

以下の環境変数は読み取られなくなった。`SolverOptions`/`IpmOptions` フィールドで同等の設定が可能:
`IPM_DD_LDL`, `MINRES_IR`, `KKT_MEMORY_BUDGET_BYTES`,
`QP_PRESOLVE_MAX_PASS`, `QP_PRESOLVE_SKIP_LARGE_COEFF`, `QP_PRESOLVE_PHASE2`.

`MINRES_ETA` は IPM main ループで Eisenstat-Walker 更新 (`set_iterative_tol`) により常に上書きされていたため、公開 Option 化は行わず廃止のみ。

`QP_PRESOLVE_SKIP` はテスト限定 (`#[cfg(test)]`) で読み取りを継続。本番ビルドでは常に `false` を返す。

### 破壊的変更

- **`SolverResult::opt_cert: Option<OptimalCertificate>` 追加**: B&B incumbent の KKT 証明書フィールド (`prove_optimal` が発行、降格時は `None`)。`SolverResult` はソルバ出力型のため通常は受け取るだけだが、フィールド全列挙の struct-literal で構築している場合は `..Default::default()` を併用すること。

### 削除

- **`solve_qp_with_options` 削除**: 0.1.0 から deprecated だった `solve_qp_with_options` を完全削除。代わりに `solve_qp_with` を使用すること。

### 修正

- **B&B `finalize_proven` の EmptyCol false-demote 修正**: presolve が消去した EmptyCol 変数 (Q 列空・A 列空) を `eliminated_cols` マスクなしで stationarity 検査すると c[j]≠0 の spurious 残差が生じ valid な proven 解が NonconvexLocal に誤降格していた。構造的マスク (`structural_empty_col_mask`) を導入して attempt.rs と同方式で解決。

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
