# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

- LP/QP→LP の未証明解を `Optimal` として返す経路と、MIP tree cut で微小な bound 逆転が panic する問題を修正

## [0.7.0] - 2026-06-25

MIP のカット・分岐強化、LP 感度分析の追加、QP の正当性修正。

- BREAKING: `LpProblem::a` の型を `CscMatrix` から `Arc<CscMatrix>` に変更し、B&B のサブ問題複製コストを削減
- MIP のカットを Cover / Clique / Implied Bound まで拡張し、分岐設定も調整して探索を高速化
- LP 感度分析 `compute_sensitivity` を追加し、退化した最適基底や有限上界を持つケースの算出も修正
- QP の LISWET7 で KKT 証明を満たさない解が偽 `Optimal` になる問題を修正
- ベンチマーク: LP 108/109 optimal + 1 SuboptimalSolution、QP 121/138 @1e-6、MILP 5/20 optimal + 15 TIMEOUT + 0 ERROR @1e-6

## [0.6.0] - 2026-06-24

MIP B&B の標準機能一式 + LP presolve/postsolve 改善 + リファクタリング。

- MIP に pseudocost branching、hybrid node selection、bound propagation、reduced cost fixing、RINS、conflict analysis、MIR cuts を追加
- LP の singleton Le/Ge と forcing row presolve を拡張し、postsolve を stack replay ベースに統一
- LP postsolve の双対誤判定と相補性違反、MIP の GMI カット数値不安定と pseudocost 計算不具合を修正
- BREAKING: `SolverOptions` などの公開 config/stats struct を `#[non_exhaustive]` 化し、`MipConfig::default().cuts` を `true` に変更
- deadline/timeout 共通化や巨大モジュール分割など、内部の大きな整理を実施

## [0.5.2] - 2026-06-19

公開API破壊的変更なし。wide LP (n/m比 1.0–2.2, 10k変数以上) を IPM 経由で解く dispatch 機構を追加し、simplex 単独では timeout していた大規模 LP を新規求解。

- wide LP を IPM 経由で解く dispatch を追加し、dfl001 / ken-18 を新規求解、ken-13 を大幅高速化
- IPM から simplex への crossover と、不要な postsolve を避ける `lp_crossover_will_certify` を追加
- threads=1 sentinel や dfl001 postsolve など、IPM dispatch 経路のテストを補強
- ベンチマーク: standard 108/109 PASS、hard 27 PASS + 25 TIMEOUT + 1 SUBOPT、extra 2 PASS + 2 TIMEOUT、infeas 29/29 PASS

## [0.5.1] - 2026-06-17

公開API破壊的変更なし (`cargo public-api diff` = 変更なし)。大規模 LP の pricing 高速化 (pds-20 を初求解)、MIP の探索効率・正当性修正、QP postsolve 安定化が主体。

- MIP に Gomory Mixed-Integer (GMI) カットを追加し、feasibility pump や crash basis 起因の探索不具合を修正
- LP の cyclic partial pricing と anti-degeneracy を導入し、DualPricing 既定も Dual Steepest Edge に変更して大規模問題を高速化
- bounded Eq+UB Phase I や Ge 制約経路を広げ、postsolve dual の検証も強化
- QP postsolve の deadline spin を抑止し、saddle-IR の選別基準を整理
- 内部では O(n^2) 重複計算の削減、ノード LP 計装復元、ignore テスト棚卸し、Audit CI 修復を実施

## [0.5.0] - 2026-06-08

公開API破壊的変更なし (`cargo public-api diff` = 変更なし)。LP/QP correctness 修正・性能改善・ベンチ基準値の外部検証補正が主体。

- Eq+UB LP 向け bounded-variable Phase I を開通し、LP/QP ブラックボックステストを独立オラクルで大幅拡張
- bounded simplex の退化、QP の dual 符号や相補性、pds-20 などの correctness 回帰をまとめて修正
- 大規模 LP の reduced-cost ループや `pivot_out` を高速化し、README のベンチ基準値も現行実測に更新
- Maros-Meszaros / AUG3D 系 QP の基準目的値を外部オラクルで補正し、入力検証や KKT 残差検査も強化
- CI と内部共通化を整理し、`log` を 0.4.32 へ更新

## [0.4.0] - 2026-06-04

- BREAKING: `LpProblem` に `obj_offset: f64` を追加し、`SolverOptions.psd_check_max_n` を削除
- 縮退 bound の誤受理、Infeasible/Unbounded reduced LP の postsolve、QP→LP dispatch の誤分類を修正
- 空制約 LP の上界違反を `Optimal` と誤判定する退化に回帰テストを追加
- BSF Big-M Phase I を revert し、README のベンチ表、OSQP ベンチ CSV、CI audit、NaN guard など周辺整備も実施

## [0.3.1] - 2026-05-29

- `try_var_name` checked variant を追加
- `solve_ipm` / `run_ipm` に `SolverOptions` の validate guard を追加し、panic を `ModelError` に変更
- flaky な postsolve テストと `audit.rs` の `should_panic` 誤検出を修正し、rustdoc と CI doc job も整理
- GitHub Actions と主要依存を更新

## [0.3.0] - 2026-05-28

- `IpmOptions` と `SolverOptions` に presolve / IR / メモリ予算まわりの設定を追加
- BREAKING: `SolverResult` に `opt_cert` を追加し、`diagnose()` 系 API や旧 certificate / residual フィールド、旧 QP API を整理
- ユーザ向けの環境変数設定を廃止し、B&B `finalize_proven` の EmptyCol 誤降格バグを修正

## [0.2.0] - 2026-05-27

- BREAKING: KKT 証明付きでのみ `Optimal` を返す cert-carrying status に刷新し、関連する証明 API と結果型を整理
- 内部実装を `otspot-core` / `otspot-io` / `otspot-model` に分割し、公開 API から開発用ユーティリティを分離
- 非凸 QP の大域求解 `solve_qp_global`、二次目的の Expression DSL、fallible API、設定 builder、MPS/QPS ストリーミングパーサを追加
- presolve で全列が消える QP や非凸 QP branch-and-bound の証明回収不備を修正
- in-tree Python バインディングと未使用依存・dead code を削除

## [0.1.1] - 2026-05-23

- LP の実行不可判定バグを修正
- LP / QP の求解安定性・汎用性を改善
- MILP の MPS 読み込みに対応
- Python バインディングを追加

## [0.1.0]

- 初版（LP: Revised Simplex / QP: Interior Point）
