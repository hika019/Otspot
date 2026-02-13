# 数理最適化ソルバー 要件定義書

作成日: 2026-02-12
担当: ashigaru5 (subtask_011d)
入力: oss_solvers.md, commercial_solvers.md, research_trends.md, benchmarking.md, winning_strategy.md

---

## 1. 対象問題種別のスコープ

### 1.1 LP（線形計画）: 必須

**事実に基づく根拠:**
- LP はあらゆる最適化問題の基盤。MIPソルバーの内部でもLP緩和を繰り返し解く（出典: commercial_solvers.md, Gurobi MIP Primer）
- HiGHS が OSS 最速の地位を確立したのは LP/MIP 特化戦略による（出典: oss_solvers.md, Mittelmann ベンチマーク）
- OSS 最速の HiGHS でも商用最速（COPT）比で LP 20 倍遅い（出典: commercial_solvers.md, HiGHS Discussion #1683）
- PDHG アルゴリズムが 2024-2025 年に商用・OSS 双方で急速普及。GPU 対応 PDHG が HiGHS, Gurobi, Xpress で実装済み（出典: research_trends.md, GAMS Blog 2025）

**推奨:** LP は第一優先の必須スコープ。Simplex 法と Interior Point 法（PDHG 含む）の両方を実装すべき。

### 1.2 MIP（混合整数計画）: 必須

**事実に基づく根拠:**
- MIP は産業応用の中核。生産計画、輸送最適化、スケジューリングなど実問題の大半が MIP に帰着する（出典: commercial_solvers.md, Gurobi/CPLEX の主要対応問題）
- 商用ソルバー（Gurobi/CPLEX）と OSS（HiGHS/SCIP）の性能差が最も大きい領域（20-30 倍）であり、最大の差別化機会（出典: commercial_solvers.md §6）
- ML によるカット選択（Cut Ranking）が産業用ソルバーで 12.42% 高速化を達成（出典: research_trends.md, Pattern Recognition 論文）
- NVIDIA cuOpt の GPU ヒューリスティクスが HiGHS と統合で最適性ギャップ 28%→21% に改善（出典: research_trends.md, cuOpt Blog）

**推奨:** MIP は LP と並ぶ第一優先の必須スコープ。Branch-and-cut を基本とし、ML 統合による差別化を最初から視野に入れる。

### 1.3 QP（二次計画）: Phase 2 でスコープイン（推奨）

**事実に基づく根拠:**
- HiGHS は LP/MIP/QP をカバーし、OSS 最速の地位を確立した（出典: oss_solvers.md）
- Gurobi は QP/MIQP/QCP/MIQCP まで対応（出典: commercial_solvers.md §2）
- ポートフォリオ最適化（金融工学）など QP の産業需要は大きい（出典: commercial_solvers.md §4, Mosek 分析）

**足軽の分析:**
初期フェーズでは LP/MIP に集中すべき。QP は LP ソルバーの延長線上に自然に実装できるため、コア性能が確立した Phase 2 で追加するのが最適。最初から QP を入れるとリソースが分散し、LP/MIP の競争力確立が遅れるリスクがある。

### 1.4 SOCP/SDP: スコープアウト（推奨）

**事実に基づく根拠:**
- SOCP/SDP は Mosek の独壇場。内点法最強で大規模 LP でも Gurobi を超える実績あり（出典: commercial_solvers.md §4）
- SCIP は MINLP/CP まで包括カバーするが、メモリ使用量が OSS 最大（172-261MB）であり、汎用性と引き換えに性能を犠牲にしている（出典: winning_strategy.md §1.2）
- HiGHS の成功は LP/MIP/QP への「一点突破」戦略による（出典: winning_strategy.md §4.1）

**足軽の分析:**
SOCP/SDP は Mosek という強力な専門ソルバーが存在し、ここで戦うのは開発リソースの無駄遣い。SCIP モデル（包括カバレッジ）ではなく HiGHS モデル（一点突破）を採用すべき。SOCP/SDP のニーズがある場合は外部ソルバー連携で対応する。

### スコープ総括

| 問題種別 | 判定 | Phase | 根拠 |
|---------|------|-------|------|
| LP | 必須 | Phase 1 | 全ての基盤。商用比 20 倍差で改善余地大 |
| MIP | 必須 | Phase 1 | 産業需要の中核。ML/GPU で差別化可能 |
| QP | 推奨 | Phase 2 | LP の延長。金融需要大。ただし初期は集中のため後回し |
| SOCP/SDP | アウト | - | Mosek の独壇場。戦うべきでない |
| MINLP/CP | アウト | - | SCIP の領域。リソース分散を避ける |

---

## 2. 並列化戦略

### 2.1 スレッド並列（共有メモリ）

**事実に基づく根拠:**
- Gurobi はデフォルトで全 CPU コアを使用。Primal/dual simplex、barrier 法を並列実行（出典: commercial_solvers.md §2）
- CPLEX はノード並列処理で困難なモデルほど効果大。Strong branching も並列化済み（出典: commercial_solvers.md §3）
- HiGHS は基本シングルスレッド。一部コンポーネントのみマルチコア活用可能（出典: winning_strategy.md §1.1）
- 小問題（60 秒未満）では並列化が逆効果（Mosek のガイドライン）（出典: commercial_solvers.md §4）

**足軽の分析・推奨:**
スレッド並列は**必須**。HiGHS の最大の弱点がシングルスレッド中心設計であり、ここが最大の差別化チャンス。ただし闇雲に並列化せず、以下の優先順位で実装すべき:
1. MIP の Branch-and-bound ノード探索並列化（最大効果）
2. LP の行列演算並列化（内点法・simplex のコア部分）
3. Presolve/Cut 生成の並列化

### 2.2 GPU 活用

**事実に基づく根拠:**
- NVIDIA cuOpt がオープンソース化（2025 COIN-OR Cup 受賞）。GPU 上で LP/MIP を解く（出典: research_trends.md §3）
- GPU 対応 PDHG が HiGHS, Gurobi, Xpress で実装済み（出典: research_trends.md §1）
- Xpress の GPU 大規模 LP で最大 50 倍高速化（出典: commercial_solvers.md §5）
- cuOpt + HiGHS 統合で MIP 最適性ギャップ 28%→21%（出典: research_trends.md §3）
- cuOpt の GPU ヒューリスティクスで MILP 最大 8.6 倍高速化（出典: research_trends.md §3, SimpleRose 統合）

**足軽の分析・推奨:**
GPU 活用は**Phase 2 以降で積極投資**すべき差別化ポイント。理由:
1. cuOpt のオープンソース化により、GPU ソルバー技術が無料で利用可能になった
2. 商用ソルバーでも GPU 対応はまだ発展途上（Gurobi は Linux 限定、Xpress はベータ版）
3. 大規模 LP での 50 倍高速化は劇的な差別化になりうる

ただし Phase 1 では CPU 並列化に集中し、GPU は PDHG の GPU カーネル実装から段階的に導入すべき。最初から GPU に全振りすると、GPU なし環境のユーザーを失う。

### 2.3 分散並列

**事実に基づく根拠:**
- CPLEX は分散並列で Coordinated concurrent approach と B&C tree 並列化の 2 アプローチを実装（出典: commercial_solvers.md §3）
- 2025 年の CPLEX 研究: Dynamic task decomposition（Scheduler + Workers 役割分担）（出典: commercial_solvers.md §3, CP 2025 論文）
- Gurobi は単一ワークステーション/マルチコアクラスタ/クラウドでシームレスにスケール（出典: commercial_solvers.md §2）

**足軽の分析・推奨:**
分散並列は**Phase 3 以降**。理由:
1. 実装複雑度が極めて高い（CPLEX でさえ「数十人年」の投資）
2. クラウド環境依存が強く、ローカルユーザーへの恩恵が薄い
3. スレッド並列と GPU で十分な差別化が可能

ただし、アーキテクチャ設計時点では分散対応を意識した疎結合設計にすべき（後から入れられるようにする）。

### 並列化戦略の総括

| 戦略 | Phase | 優先度 | 期待効果 |
|------|-------|--------|---------|
| スレッド並列（MIP ノード探索） | Phase 1 | 最高 | HiGHS 超えの最短経路 |
| スレッド並列（LP 行列演算） | Phase 1 | 高 | LP 性能の底上げ |
| GPU PDHG | Phase 2 | 高 | 大規模 LP で 50 倍高速化の可能性 |
| GPU ヒューリスティクス | Phase 2 | 中 | MIP で 8.6 倍高速化の可能性 |
| 分散並列 | Phase 3 | 低 | クラウド対応、企業向け |

---

## 3. 実装言語

### 候補比較

**事実に基づく根拠:**

| 言語 | 性能 | 安全性 | エコシステム | 開発速度 | 採用実績 |
|------|------|--------|-------------|---------|---------|
| **Rust** | C++ 同等 | メモリ安全（コンパイル時保証） | 成長中。Mosek が Rust API 提供（出典: commercial_solvers.md §4）。russcip が活発開発中（出典: oss_solvers.md §2） | C++ より遅い（学習曲線） | ソルバー領域では少ない |
| **C++** | 最高 | 手動管理（バグ源） | 最も成熟。HiGHS/SCIP/CBC/Gurobi/CPLEX 全て C++（出典: oss_solvers.md, commercial_solvers.md） | 標準的 | ソルバー領域のデファクト |
| **C + Python** | C は最高 | C は手動管理 | Python バインディング容易 | C は低速 | GLPK が C 実装（出典: oss_solvers.md §5） |

**足軽の分析・推奨: Rust**

根拠:
1. **メモリ安全性**: ソルバーは大規模行列操作が中心。C++ のメモリバグ（use-after-free, buffer overflow）はデバッグが極めて困難。Rust のコンパイル時保証はソルバー開発で特に価値が高い
2. **並列安全性**: Rust の所有権システムはデータ競合をコンパイル時に防止。並列化を第一級に扱う本プロジェクトにとって決定的な利点
3. **性能**: Rust は C++ と同等の性能を達成可能（ゼロコスト抽象化）
4. **エコシステム**: russcip（2025 年 12 月更新）、Mosek Rust API の存在は、ソルバー分野での Rust 採用が始まっていることを示す
5. **差別化**: 全主要ソルバーが C++ の中、Rust 実装は明確な技術的差別化。安全性・並列性をアピールできる
6. **FFI**: Rust→C の FFI は成熟しており、Python バインディング（PyO3）も高品質

リスク:
- Rust のソルバー領域での実績が少ない（ただし russcip 等で増加中）
- 初期の開発速度は C++ より遅い可能性（学習曲線）
- 数値計算ライブラリの成熟度が C++ に劣る（ただし nalgebra, ndarray 等が成長中）

リスク緩和策:
- コア数値カーネルは最初から BLAS/LAPACK（C/Fortran）を FFI 経由で利用
- Rust の unsafe ブロックは最小限に留め、安全なインターフェースでラップ

---

## 4. API 設計方針

### 4.1 Python バインディング: 必須

**事実に基づく根拠:**
- 全主要ソルバーが Python API を提供（出典: oss_solvers.md, commercial_solvers.md 全体）
- highspy（HiGHS）は pybind11 使用、PyPI で配布（出典: oss_solvers.md §1）
- PySCIPOpt は conda-forge で 645.1K DL（出典: oss_solvers.md §2）
- Gurobi/CPLEX/Mosek/Xpress 全てが numpy 対応 Python API を提供（出典: commercial_solvers.md）
- Rust→Python は PyO3 + maturin で高品質なバインディングが可能

### 4.2 API 設計: 独自 API + Gurobi 互換レイヤー（推奨）

**事実に基づく根拠:**
- Gurobi は最も広く使われる商用ソルバー API（学術ライセンス無料・制限なし）（出典: commercial_solvers.md §2）
- OR-Tools は複数ソルバーへのラッパーとして機能し、ソルバー切り替えが容易（出典: oss_solvers.md §4）
- HiGHS の成功要因の一つはシンプルな API（出典: winning_strategy.md §4.1）
- SCIP はプラグインアーキテクチャで拡張性を実現（出典: oss_solvers.md §2）

**足軽の分析・推奨:**

**Phase 1: 独自 API（シンプル優先）**
- HiGHS スタイルのシンプルな API を設計
- モデル構築 → 求解 → 結果取得の基本フローを最短コードで実現
- numpy 配列の直接入力対応

**Phase 2: Gurobi 互換レイヤー**
- 既存 Gurobi ユーザーの移行コストを最小化
- `from solver import gurobi_compat` のような互換モジュール提供
- 完全互換は不要。頻出パターン（変数追加、制約追加、目的関数設定、求解）をカバー

**理由:**
1. 独自 API を先にすることで、設計が Gurobi の制約に縛られない
2. Gurobi 互換は後付けでも十分実現可能
3. 「Gurobi からの移行が簡単」は採用促進の強力な訴求

### 4.3 その他の言語バインディング

| 言語 | Phase | 理由 |
|------|-------|------|
| C API | Phase 1 | FFI 基盤。他言語バインディングの土台 |
| Python | Phase 1 | 最大ユーザーベース |
| Julia | Phase 2 | JuMP エコシステム（学術での影響力大） |
| JavaScript/WASM | Phase 3 | ブラウザ実行。教育・デモ用 |

---

## 5. OSS ライセンス選定

### 推奨: Apache-2.0

**事実に基づく根拠:**
- OSS プロジェクトの 60% 以上が MIT, GPL, Apache 2.0 のいずれか（出典: winning_strategy.md §3.4）
- HiGHS の成功は MIT ライセンスが大きい（SciPy, MathWorks 等への組み込み容易性）（出典: winning_strategy.md §3.4）
- SCIP は Apache 2.0 に移行（SCIP 10.0）（出典: oss_solvers.md §2）
- GPL は派生物も GPL 化必要で商用組み込みに要注意（出典: oss_solvers.md §ライセンス選定のポイント）
- Apache 2.0 は MIT と同等の自由度 + 特許保護（出典: winning_strategy.md §3.4）

**足軽の分析:**

| ライセンス | 商用組込 | 特許保護 | 採用しやすさ | ソルバー実績 |
|-----------|---------|---------|------------|------------|
| MIT | 自由 | なし | 最高 | HiGHS |
| Apache-2.0 | 自由 | あり | 最高 | SCIP 10.0, OR-Tools |
| GPL | 派生物 GPL | なし | 低い（商用敬遠） | GLPK |
| MPL-2.0 | ファイル単位 | なし | 中 | 実績少 |

**Apache-2.0 を推奨する理由:**
1. MIT と同等の自由度（商用組み込み自由）
2. **特許保護条項**が追加。特許トロールへの防御になる
3. SCIP が Apache 2.0 に移行した事実は、ソルバー分野でのトレンドを示す
4. Rust エコシステムでは Apache-2.0 / MIT デュアルライセンスが慣例
5. コントリビューター由来の特許リスクを低減

MIT でなく Apache-2.0 を推す理由は**特許保護**の一点。ソルバーアルゴリズムは特許訴訟リスクがあり（商用ソルバーの特許戦略）、Apache-2.0 の特許グラントは防衛的価値が高い。

---

## 6. 差別化戦略（天下を取るための戦略）

### 6.1 何で勝つか

**事実の整理:**
- 商用 vs OSS の性能差: MIP で 20-30 倍、LP で 10-20 倍（出典: commercial_solvers.md §6）
- 小～中規模問題では HiGHS が Gurobi 同等性能を達成済み（出典: winning_strategy.md §2.2）
- HiGHS の弱点: シングルスレッド中心、大規模問題で 60-100 倍遅い場合あり（出典: winning_strategy.md §1.1）
- ML 統合: Cut Ranking で 12.42% 高速化、汎化性能が課題（出典: research_trends.md §2）
- GPU: cuOpt オープンソース化、大規模 LP 50 倍高速化、MILP 8.6 倍高速化（出典: research_trends.md §3）
- 10 年前に解けなかった問題が数秒で解ける時代（出典: research_trends.md §5, Math Programming サーベイ）

**足軽の分析: 3 つの差別化軸**

#### 軸 1: 並列化ファースト設計（最重要）

HiGHS がシングルスレッド中心である事実は、**設計レベルで並列化を組み込んだソルバー**が差別化できることを意味する。後付け並列化（HiGHS のアプローチ）ではなく、データ構造・アルゴリズム選択の段階から並列実行を前提とした設計にすべき。

具体的には:
- Lockfree データ構造による行列操作
- MIP ノード探索の work-stealing スケジューラ
- Rust の所有権モデルによるデータ競合防止
- Phase 2 で GPU PDHG、Phase 3 で分散対応

#### 軸 2: ML ネイティブ統合

既存ソルバーの ML 活用は「後付け」（外部 ML モデルを API 経由で呼ぶ）。本プロジェクトでは ML をソルバーの内部コンポーネントとして設計すべき。

具体的には:
- Cut Ranking の内蔵（学習済みモデルを同梱）
- 分枝変数選択の ML ポリシー
- ハイパーパラメータ自動チューニング（問題構造の自動認識）
- ただし ML なしでも十分動作するフォールバック設計（ML は加速器、必須依存にしない）

#### 軸 3: Rust による安全性・開発者体験

全主要ソルバーが C++ の中、Rust 実装は以下のメッセージを発信できる:
- 「メモリ安全なソルバー」（セキュリティ意識の高い企業への訴求）
- 「並列安全なソルバー」（データ競合のないマルチスレッド）
- 「モダンな開発体験」（cargo によるビルド・テスト・ベンチマーク統合）

### 6.2 何を捨てるか

**明確に捨てるもの:**
1. **SOCP/SDP/MINLP 対応**: Mosek/SCIP の領域。リソース分散を避ける
2. **全問題クラスでの商用超え**: 大規模 MIP で Gurobi に完勝する必要はない。「中規模で同等、大規模で 1 桁差以内」が現実的目標
3. **商用サポート/SLA**: OSS として技術で勝負。商用サポートは将来の法人化フェーズで検討
4. **Windows ネイティブ最適化**: Linux/macOS 優先。Windows は動作保証するが性能最適化は後回し
5. **レガシー API 互換**: AMPL/GAMS 互換は不要。Python ファーストで十分

### 6.3 勝利条件の定義

| 指標 | Phase 1 目標 | Phase 2 目標 | 長期目標 |
|------|-------------|-------------|---------|
| LP 性能（Mittelmann） | HiGHS 同等 | HiGHS 超え | 商用比 5 倍差以内 |
| MIP 性能（MIPLIB 2017） | HiGHS 同等 | HiGHS 超え（並列） | 商用比 10 倍差以内 |
| Python API 品質 | 基本機能 | numpy 完全対応 | Gurobi 互換レイヤー |
| エコシステム統合 | PyPI 公開 | SciPy/OR-Tools 連携 | デフォルトソルバー採用 |
| コミュニティ | GitHub 公開 | 100+ stars | 外部コントリビューター |

### 6.4 ベンチマーク戦略

**事実に基づく実践手順（出典: benchmarking.md）:**
1. MIPLIB 2017 Benchmark Set（240 インスタンス）を標準テストセットとして使用
2. Netlib（90 問）で LP 専用ベンチマーク
3. Shifted geometric mean（s=10）で性能比較
4. Performance profiles で可視化
5. 比較対象: HiGHS, SCIP（OSS 同士の公正比較）
6. ハードウェア・パラメータ・テストセット選定理由を全て公開
7. ベンチマークスクリプトとログを GitHub で公開（再現性確保）

---

## 要件定義サマリ

| 項目 | 決定 | 根拠の核心 |
|------|------|----------|
| LP | 必須（Phase 1） | 全ての基盤。PDHG で差別化可能 |
| MIP | 必須（Phase 1） | 産業需要の中核。ML/GPU で差別化 |
| QP | Phase 2 | LP の延長。金融需要大だが初期は集中 |
| SOCP/SDP | アウト | Mosek の独壇場。戦わない |
| 並列化 | スレッド並列 Phase 1、GPU Phase 2、分散 Phase 3 | HiGHS の最大弱点を突く |
| 実装言語 | Rust | メモリ安全 + 並列安全 + 性能 |
| API | 独自 + Gurobi 互換（後付け） | シンプルさ優先、移行容易性 |
| ライセンス | Apache-2.0 | MIT 同等の自由度 + 特許保護 |
| 差別化 | 並列ファースト + ML ネイティブ + Rust 安全性 | 既存 OSS が弱い 3 領域で勝負 |

---

**記載方針**: 事実（ソース付き）と足軽の意見/分析を明確に分離して記載した。
