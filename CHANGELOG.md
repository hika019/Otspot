# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-05-26

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

## [0.1.1] - 2026-05-23

- LP の実行不可判定バグを修正
- LP / QP の求解安定性・汎用性を改善
- MILP の MPS 読み込みに対応
- Python バインディングを追加

## [0.1.0]

- 初版（LP: Revised Simplex / QP: Interior Point）
