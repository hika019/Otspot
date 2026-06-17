# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.5.1] - 2026-06-17

公開API破壊的変更なし (`cargo public-api diff` = 変更なし)。大規模 LP の pricing 高速化 (pds-20 を初求解)、MIP の探索効率・正当性修正、QP postsolve 安定化が主体。

### 追加

- MIP: Gomory Mixed-Integer (GMI) カット生成を追加
- LP: 大規模 LP 向け cyclic partial pricing を primal bounded core に導入 (pricing コスト約 60% 削減、pds-20 を初めて求解)
- LP: anti-degeneracy (primal core の Bland anti-cycling + RHS 摂動、既定 OFF)

### 変更

- LP: DualPricing 既定を Dual Steepest Edge (Forrest-Goldfarb) へ変更し dual simplex の反復を削減
- LP: primal_simplex_aug の冗長 FTRAN を排除し per-iter コストを削減
- LP: bounded Eq+UB Phase I の汎用化 / Ge 制約経路の開通

### 修正

- MIP: feasibility pump の偽 incumbent (1e12 等) を修正 (gen-ip002 真因・mas76)
- MIP: ノード LP の crash basis を無効化し、crash-infeasible 起因の探索木膨張を根治
- QP: postsolve の deadline spin を抑止し、saddle-IR best 選別の指標を統一
- LP: postsolve dual の検証空白を正規 assert 化 (pilot-ja / perold の stale ignore 解消)
- テスト: ken-13 deadline ガードが deadline 内に停止した SuboptimalSolution を受理するよう是正

### 内部

- LP: bound_contrib 重複計算を O(n²)→O(n) に削減
- MIP: ノード LP の timing 計装 (lp_solve_us) を復元
- `#[ignore]` テストの棚卸し (隠れバグ無しを実測確定、broken 誤ラベル訂正、tier-2 を default profile へ)
- comment-hygiene gate / Audit CI の修復

## [0.5.0] - 2026-06-08

公開API破壊的変更なし (`cargo public-api diff` = 変更なし)。LP/QP correctness 修正・性能改善・ベンチ基準値の外部検証補正が主体。

### 追加

- bounded-variable Phase I を Eq+UB LP で開通 (従来 SuboptimalSolution に退化していた経路を正規サポート)
- LP/QP ブラックボックステストを独立オラクル (SciPy / OSQP / Clarabel / SCS) で大幅拡張

### 変更

- README Performance 表を現行ベンチ実測値へ更新 (proof-carrying KKT 基準: Feasible LP 105/109・Convex QP 121/138・Infeasible 29/29 @1e-6)
- default テストプロファイルが ignore 以外を全実走 (tier-2 廃止)

### 修正

- bounded simplex の退化を複数修正 — 終端 stale x_b の原始非実行可能解 (grow7/15/22, pilot87)、非アクティブ境界双対の射影漏れ (pilot-we)、反復効率 (ken-13 Devex)、Harris ratio test (grow22)
- QP correctness — QFORPLAN (dual 符号射影 + bound activity)、GOULDQP2 (bound dual の comp 一貫基準)、IPM 証明条件を収束判定へ整合
- pds-20 / cplex2 / scorpion / osa-60 の correctness 回帰を真因修正 (FTRAN 安定性 / Phase I fresh FTRAN / 人工変数 cleanup)
- 大規模 LP の性能改善 — reduced-cost ループ chunk 化 (iter/sec 2-2.25x)、pivot_out バッチ化 (pds-20 約 59s→0.8s)
- Maros-Mészáros (16 件) / AUG3D 系 QP の基準目的値を外部オラクル (Clarabel / OSQP / SCS) 検証値へ補正
- その他 — 入力検証 hardening、postsolve Krylov IR gate 回帰、KKT 残差の双対長不足検出、Timeout 目的値の復元解再計算、iters=0 報告 artifact

### 内部

- 重複削減 (parser / objsense / Expression / deadline_expired 共通化)
- CI: heavy 非ゲート化 + `--test-threads 3`、netlib 依存解消 (emps.c vendoring + cache)、gate 整備 (clippy / comment-block / package)

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
