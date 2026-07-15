# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.7.2] - 2026-07-16

conic/QCQP の正当性修正と SOCP IPM の収束改善、大規模 LP の presolve 性能改善、MPS/QPS パーサの形式判定の作り直し。

- BREAKING: `mps::parse_mps_reader` / `mps::parse_milp_reader` / `qps::parse_qps_reader` の型境界を `R: BufRead` から `R: BufRead + Seek` に変更。固定桁 MPS と判明したファイルは先頭から読み直す必要があり、`Seek` はそれを全行バッファ無しで行うための条件 (MPS は GiB 級になりうるため入力の全行保持は不可)。`File` / `Cursor` は `Seek` 済みでそのまま渡せる。stdin やパイプなど seek 不可の入力は、呼び出し側で `Cursor::new(buf)` に読み切ってから渡す
- MPS/QPS の形式判定を行単位のヒューリスティクスからファイル単位の決定に作り直し、固定桁 MPS (名前に空白を含む Netlib forplan 等) の誤読と、未宣言の行名/列名の黙殺 (silent data loss) を修正。ROWS の行名重複、`OBJSENSE` のヘッダ行記法・`MAXIMIZE`/`MINIMIZE` 綴り、コメント欄 (62-72 桁) とシーケンス番号欄 (73-80 桁) も正しく扱う
- BREAKING: BOUNDS で値を取る型 (`UP`/`LO`/`FX`/`UI`/`LI`) の余剰トークン (`UP BND x1 5.0 10.0` 等) を reject するようにした (MPS/QPS 共通)。v0.7.1 の MPS 自由形式リーダはこれを黙殺しており (QPS は元から reject していた)、MPS 利用者にとっては従来 `Ok` だった入力が `Err` になりうる非互換な挙動変更。固定桁形式でも field 5/6 (BOUNDS が定義しない領域) の内容を reject するよう追加した — 自由形式側の reject だけでは、grid に整列した (=現実の MPS の大多数を占める) ファイルは format fallback で固定桁として再読され、その黙殺により `Ok` に戻ってしまい実効性がなかった。field 4 は `FR`/`MI`/`BV`/`PL` の冗長な値欄として引き続き許容する (`leo1`/`leo2`)
- 非凸 QCQP が凸 SOCP として誤って「証明付き Optimal」と報告される問題を修正: Cholesky がゼロピボット列の非ゼロ off-diagonal を捨てて不定値を PSD と誤判定していた。PSD 判定の許容をスケール相対化
- QCQP の McCormick global fallback で実行可能性許容が最適性ギャップ許容と混同され、制約に違反する点を `Optimal` と報告する問題を修正
- SOCP IPM にデータ駆動の初期点と Mehrotra 相補均衡化を導入し CBLIB conic 問題の収束を改善。B&B 緩和ノードでは均衡化を無効化して MIQCP の退化を回避
- presolve が冗長な含意上界を大量に生成して標準形の行数を爆発させ、大規模 LP を大幅に遅くする問題を修正
- 公開 `QpProblem` フィールドへ不整合な二次制約を代入した際の添字範囲外パニックを、中央検証で防止
- conic/QP の残差・目的値・実行可能性判定で、次元不一致を 0 ベクトルに化けさせて処理を続行する経路 (最悪ケースで実行不能点を `Optimal` と誤報告しうる) を排除し、不変条件を明示的な検証に置き換え
- Model DSL の tolerance 伝播、他モデル変数の混入検出、CBF の非有限定数拒否を修正
- ベンチマーク: LP @1e-6 109/109 optimal・@1e-8 108/109、QP Maros 121/138 @1e-6・93/138 @1e-8、MILP 5/20 optimal + 15 TIMEOUT、SOCP は Mittelmann Large-SOCP 18問で他ソルバ (MOSEK/ECOS/COPT) と比較し 4/18 @1000s (6/18 @3600s)

## [0.7.1] - 2026-07-14

v0.7.0 向けのパッチリリース相当。公開APIの型・シグネチャ変更なし。

- Model API: LP 最大化で `dual_solution` / `reduced_costs` が内部最小化の符号のまま返る問題を修正
- Model API: `var_name()` に別モデルの変数や範囲外 index が渡された場合、誤った名前を返さず明示的に panic するよう修正
- Model API: QP 双対ベクトル長や MIP 解ベクトルの不整合を prefix 切り詰め・丸めで隠さずエラー化
- IO: MPS/QPS の RHS/RANGES 値ペア末尾欠落や BOUNDS の不正数値をエラー化しつつ、FORPLAN など固定幅MPSの空白入り名前は正しく受理
- IO: QPS の固定幅/自由形式の判定を列整合ベースに厳密化し、空白入りの列名・bound-set 名・数値を含む行名（例 `A   22 1`, `B ND`, `C 1`）を BOUNDS/RHS/RANGES で正しく復元しつつ、自由形式の余分トークンは誤検出せず拒否（RHS/RANGES の第2ペアや空白入り set-name も含む）
- Core: LP 証明、postsolve、感度分析、MIP カット生成、reduced-cost fixing、QP bound dual / duality gap で欠損ベクトルを 0 埋め・skip せず拒否
- Core: キャンセル済み QP で空解を KKT 評価に渡して panic する経路を修正
- Core: presolve 済み LP のキャンセル/Timeout で双対が空でも `Timeout` を保持し、`NumericalError` に誤変換しない
- Dependencies: `crossbeam-epoch` を 0.9.20 に更新し、RUSTSEC-2026-0204 を解消

## [0.7.0] - 2026-06-29

MIP のカット・分岐強化、LP 感度分析の追加、QP/LP の正当性修正。

- BREAKING: `LpProblem::a` の型を `CscMatrix` から `Arc<CscMatrix>` に変更し、B&B のサブ問題複製コストを削減
- MIP のカットを Cover / Clique / Implied Bound まで拡張し、分岐設定も調整して探索を高速化
- LP 感度分析 `compute_sensitivity` を追加し、退化した最適基底や有限上界を持つケースの算出も修正
- QP の LISWET7 で KKT 証明を満たさない解が偽 `Optimal` になる問題を修正
- LP/QP→LP の未証明解を `Optimal` として返す経路と、MIP tree cut で微小な bound 逆転が panic する問題を修正
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
