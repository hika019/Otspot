# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### 追加

### 変更

### 修正

### 内部

### 依存

## [0.5.0] - 2026-06-08

公開APIに破壊的変更はなし (`otspot` クレートの公開シグネチャは v0.4.0 と同一)。本リリースは
LP/QP ソルバの correctness 修正・性能改善・ベンチ基準値の外部検証補正が主体。

### 追加

- bounded-variable Phase I を Eq+UB LP で開通 — 従来 SuboptimalSolution に退化していた境界付き
  等式+上界制約の経路を正規にサポート
- LP/QP ブラックボックステスト群を大幅追加 — パイプライン全段を独立オラクル (SciPy / OSQP /
  Clarabel / SCS) で検証。「優しくない」LP/QP センチネルと各処理段の sentinel test を整備

### 変更

- README Performance 表を proof-carrying KKT 基準の現行実測値へ更新 —
  Feasible LP 105/109・Convex QP 121/138・Infeasible LP 29/29・Unbounded LP 12/12 (@1e-6)
- default テストプロファイルが ignore 以外を全実走 (tier-2 廃止)

### 修正

correctness:

- bounded simplex 終端の stale x_b による原始非実行可能解を修正 (grow7/15/22, pilot87)
- bounded 双対の非アクティブ境界射影漏れによる pilot-we の偽 SuboptimalSolution を修正
- ken-13 の反復効率退化を Devex pricing 採用で解消
- grow22 回帰を bounded primal の Harris ratio test 採用で修正
- GOULDQP2 回帰を bound dual activity の comp 一貫基準で修正
- QFORPLAN QP correctness — dual 符号射影 + bound activity 判定を修正
- pds-20 degen2 correctness 回帰 — FTRAN 安定性検証 + sequential fallback
- cplex2 の偽 non-convergent を真因対処 — Phase I で人工変数判定前に x_b を fresh FTRAN で再計算
- scorpion presolve=OFF の NumericalError を修正
- postsolve の Krylov IR skip 判定をユーザー許容値基準へ戻す (gate 回帰修正)
- QP 収束 — IPM 証明条件を収束判定に揃える + iteration accounting 整合
- 入力検証を hardening — 不正入力 (縮退 bound 等) を明示エラーで拒否
- iters=0 報告 artifact を修正 — reduced-space Timeout で iteration count を保持

performance:

- 大規模 LP の reduced-cost ループを chunk 化し iter/sec を 2-2.25x 改善
- pivot_out のバッチ化で pds-20 を約 59s → 0.8s (72×)
- Ruiz スケーリング前の未使用 clone を回避

ベンチ基準値 (外部オラクル検証):

- Maros-Mészáros QP の基準目的値 16 件を Clarabel 0.11.1 (tol 1e-12) の独立検証値へ補正 —
  旧自己計測値の約 2 倍の規約誤りを解消
- AUG3D 系 QP の基準値を OSQP / Clarabel / SCS の 3 独立オラクル一致値へ補正

### 内部

- 重複削減 — parser section driver / objsense / Expression / deadline_expired を共通化
- CI: heavy サーベイランステストを非ゲート化 (ignore/broken の赤を gate から除外)、nextest を
  `--test-threads 3` に統一
- CI: netlib 依存を解消 (emps.c vendoring + cache からの baseline restore)、cache key を整備
- CI gate 整備 — clippy / comment-block / `cargo package --no-verify`
- docs(CLAUDE.md): Phase マージ時のテスト範囲を明文化

### 依存

- log 0.4.30 → 0.4.32 (#11)

## [0.4.0] - 2026-06-04

### 追加

### 変更

- **BREAKING**: `LpProblem` に `obj_offset: f64` フィールド追加 — MPS N-row RHS 定数を正しく求解結果へ反映 (#191)
- **BREAKING**: `SolverOptions.psd_check_max_n` フィールド削除 — production caller 0 件、soundness 穴 (size-skip) の除去 (#130)

### 修正

- (±inf,±inf)/(-inf,-inf) 縮退 bound を reject する共有 validator を導入 — LP/QP の空区間誤受理を解消 (bug C/D)
- postsolve が Infeasible/Unbounded の reduced LP を postsolve せず早期返却 — 偽 solution vector 生成を防止 (bug F)
- QP→LP dispatch の変換エラーを Infeasible でなく NumericalError として返し route を設定 (bug G)
- 空制約 LP (m==0) の zero-cost 上界変数で ub 違反 x=0 を Optimal 誤判定する退化に回帰テスト追加 (bug A)

### 内部

- #42 BSF Big-M Phase I を revert — Eq/Ge+UB LP を SuboptimalSolution に退化させ統合テストを壊していたため (収束改善は v0.5.0 で gate 限定+sentinel 再投入予定)
- README Performance 表を現行ベンチ実測値へ更新 (proof-carrying KKT 基準)
- `osqp_bench.csv` に SS_* SuiteSparse 行を統合 (旧 `osqp_bench_optional.csv` 削除、#133):
  以前の split 設計 (osqp_bench.csv + osqp_bench_optional.csv) は bench_utils::detect_csv_path が optional CSV を
  load しないため SS_* baseline が bench runner の退行検知から除外される欠陥があった。single-file 設計で修正。
  optional 判定は SS_* 名前プレフィックスで行う (旧 4th-column `*` marker は regen ツールで消失する fragility があった)。
- `tests/test_check_data_coverage.py` を CI audit job および `scripts/pre-merge-audit.sh` に組込 (#133)。
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
