# 線形ソルバー調査アブストラクト

作成日: 2026-02-12
担当: ashigaru1
任務ID: subtask_027e2

---

## 概要

本レポートは線形ソルバー（LP: Linear Programming、MIP: Mixed-Integer Programming）の現状を包括的に調査したものである。オープンソース6種（HiGHS, SCIP, CBC, OR-Tools, GLPK, Ipopt）、商用4種（Gurobi, CPLEX, Mosek, Xpress）を対象とし、性能、ライセンス、並列化、Python統合の4軸で比較。ベンチマーク手法（MIPLIB 2017、Netlib、Mittelmannベンチマーク）と評価指標（Shifted Geometric Mean等）を解説し、2024-2026年の研究動向（PDHG普及、ML統合、GPU加速）を分析した。線形分野における天下取り戦略として、一点突破型（LP/MIP特化）vs包括型（全問題種別）の比較と推奨アプローチを提示する。

---

## 主要知見（事実・ソース付き）

### OSS線形ソルバーの現状と主要プレイヤー

#### 性能ランキング（LP/MIP）

- **HiGHS**: Mittelmannベンチマークでオープンソース線形最適化ソフトウェアの中で最高の性能。ただしGurobiとは約1桁（10倍）の性能差。 [ソース: HiGHS Official Site, HiGHS GitHub Discussion #1683]
- **SCIP**: Mittelmannベンチマークでは HiGHS と Gurobi の中間、CBC より高速。学術開発ソルバーの中で最速クラスのMIP/MINLPソルバー。 [ソース: HiGHS GitHub Discussion #1683, SCIP Official Site]
- **CBC**: 長年の人気選択肢だが、現在 HiGHS と比較して著しく性能劣位。 [ソース: COIN-OR CBC GitHub]
- **OR-Tools**: 複数ソルバーのラッパーとして機能。CP-SAT に受賞歴あり。 [ソース: OR-Tools Official Site]
- **GLPK**: 性能は中程度。並列化研究では12コアで最大21.9倍、GPU実装で最大63倍高速化実績あり。 [ソース: ResearchGate, arXiv]

#### ライセンス

- **MIT（HiGHS）**: 最も制約少ない。商用利用も自由。 [ソース: HiGHS Official Site]
- **Apache 2.0（SCIP v10, OR-Tools）**: MIT同等の自由度。 [ソース: SCIP Suite 10.0 Paper, OR-Tools GitHub]
- **EPL（CBC, Ipopt）**: 弱いコピーレフト。実用上問題少ない。 [ソース: COIN-OR CBC License]
- **GPL v3（GLPK）**: 派生物もGPL化必要。商用製品への組み込みは要注意。 [ソース: GLPK.jl JuMP]

#### 並列化対応

- **HiGHS**: dual simplex ソルバーでマルチスレッディング対応。GPU対応PDHG実装が追加（2025年）。 [ソース: HiGHS Documentation, GAMS Blog 2025]
- **SCIP**: UGフレームワークによる並列化（共有・分散メモリ両対応）、FiberSCIPで決定論的並列化。 [ソース: SCIP Suite 10.0 Paper]
- **CBC**: 明示的な並列化情報なし（主にシリアル実装と推測）。
- **GLPK**: 研究段階で並列化実績あり（スレッド並列、GPU実装）。 [ソース: ResearchGate, arXiv]

#### Python統合

- **OR-Tools**: Google製、Python 3.13/3.14対応、成熟。 [ソース: ortools PyPI]
- **PySCIPOpt**: 活発開発、SCIPリリース毎に更新、プラグイン可能。 [ソース: PySCIPOpt GitHub]
- **highspy**: 軽量、配列アクセス注意点あるが実用レベル。 [ソース: highspy PyPI]
- **CyLP（CBC）**: 基本的なインターフェース。 [ソース: CyLP GitHub]

---

### 商用ソルバーの優位性と技術的核心

#### 性能差の実態

- **MIP**: 商用（CPLEX/Xpress/Gurobi）はOSS（HiGHS/CBC/SCIP）より約20-30倍高速（一部ケース）。Mittelmannベンチマークでは約1桁差。 [ソース: HiGHS Discussion #1683]
- **LP**: 最良OSS（HiGHS）は最良商用（COPT）より20倍遅い。 [ソース: HiGHS Discussion #1683]
- **小規模問題**: OSS内（SCIP/HiGHS/GLPK）は同程度の性能。 [ソース: Wageningen Study]

#### 技術的コア要素

1. **Presolve（前処理）**: 分枝限定法開始前に問題サイズ削減・定式化強化。最重要要素の一つ。 [ソース: Gurobi MIP Primer]
2. **Cutting Planes（切除平面法）**: Gomory混合整数カット（2.52倍高速化）、MIR（1.83倍高速化）等。適応的戦略で問題特性に応じて動的適用。 [ソース: Gurobi Help]
3. **並列化**: Gurobiはデフォルトで全CPUコア使用。CPLEXは並列B&C、分散処理、動的タスク分解（2025年最新研究）。 [ソース: Gurobi Parallel Optimization, CP 2025 Parallel MIP]
4. **GPU対応**: Gurobi（Linux、NVIDIA cuOpt使用）、Xpress（Windows/Linux、バージョン9.8からベータ版）がPDHG実装。大規模LPで最大50倍高速化。 [ソース: Gurobi GPU Solver, FICO Xpress Wikipedia]

#### 商用ソルバー個別特徴

- **Gurobi**: MIP/QP/QCPで最速級。GPU対応（Linux限定）。学術ライセンス無料（制限なし）。2024年8月にMittelmannベンチマーク撤退。 [ソース: solver.com, Gurobi Academic Program, Mittelmann Benchmark]
- **CPLEX**: 並列MIP処理成熟（ノード並列、Strong branching並列、分散処理）。学術ライセンス無料（1年更新）。 [ソース: IBM CPLEX Features]
- **Mosek**: 錐最適化（SOCP/SDP）特化、内点法最強。大規模LPでGurobi超え（研究データあり）。実行不可能性検出が信頼性高い。 [ソース: arXiv MINLP Study, MOSEK Wikipedia]
- **Xpress**: MIP/MINLP性能が近年大幅向上（14-68%高速化）。GPU対応がWindows/Linuxで利用可。 [ソース: FICO Xpress MIP Performance]

#### 性能差の理由

- 商用MIPは「person-decades」規模の開発投資（高額ライセンス収益で資金調達）。並列ツリー探索、各種問題クラス対応トリック。 [ソース: HiGHS Discussion #1683]

---

### ベンチマーク手法と主要結果

#### 標準テストセット

- **MIPLIB 2017**: 第6版、初期プール5,721インスタンスから240インスタンスがBenchmark Set。混合整数最適化ソルバーの性能比較標準。2024年にMIPLIB 2024への投稿受付開始。 [ソース: MIPLIB 2017, MIPLIB 2024 Submission]
- **Netlib**: LP専用。実生活のLP問題コレクション。90問（最大N=13,525変数、M=3,000制約）。 [ソース: The NETLIB LP Test Problem Set, ALGLIB Benchmark]
- **Mittelmannベンチマーク**: Arizona State UniversityのHans D. Mittelmann教授が維持。2024年8月にGurobi、2024年12月にMindOptが撤退。 [ソース: plato.asu.edu/bench.html]

#### 評価指標

- **Shifted Geometric Mean（s=10）**: 計算MIPの標準指標。非常に大きな外れ値にも非常に小さな外れ値にも妥協しない。最速ソルバーを1.0にスケールして比較。 [ソース: arXiv:2302.01164v1, Mittelmann Plots]
- **Performance Profiles**: 複数ソルバーの性能指標の分布関数。Dolan & Moré (2002)が広めた（Google Scholar約5,000引用）。視覚的に直感的。 [ソース: Benchmarking Optimization Software with Performance Profiles]

#### ベンチマーク結果の解釈

- 同一ハードウェア環境、デフォルトパラメータ、透明なテストセット選定が公平性の前提。 [ソース: MIP Solvers Unleashed, ResearchGate]

---

### 研究動向（線形関連、2024-2026年）

#### PDHGアルゴリズムの普及

- 2024年にCardinal Optimization社のCOPTがPDHG実装を先駆的に導入。2025年にCOPT 7.2、Gurobi 13、HiGHS 1.10、Knitro 15、Xpress 46が導入・更新。GPU対応PDHGもHiGHS、Gurobi、Xpressで利用可能に。 [ソース: GAMS Blog 2025]

#### 機械学習の統合

- **カット選択へのML適用**: Cut Ranking（複数インスタンス学習）が産業用ソルバーに実装済み。大規模MIP問題で平均12.42%の高速化を達成。 [ソース: Learning to Select Cuts, GAMS Blog 2025]
- **理論基盤**: 2025年ICLR論文で、スコアリング関数が区分多項式構造を持つ場合のサンプル複雑度境界を確立。 [ソース: Generalization Guarantees - arXiv]
- **課題**: 汎化性能（異なるタスク・問題クラスへの適用）が重要課題。 [ソース: Survey on MIP via ML]

#### GPU加速（NVIDIA cuOpt）

- **cuOpt概要**: LP、MIP、車両経路問題をGPU上で解くソルバー。2025 COIN-OR Cup受賞。オープンソース化。 [ソース: NVIDIA cuOpt Blog, COIN-OR Cup Award]
- **性能**: HiGHS単独で28%ギャップ → cuOpt統合で21%に改善（MIPLIB、5分制限）。MILP最大8.6倍高速化。 [ソース: HiGHS and cuOpt Blog, SimpleRose cuOpt Integration]
- **ハイブリッド手法**: GPUで原始ヒューリスティック実行、CPUで双対境界改善。 [ソース: cuOpt Technical Blog]

#### OSSソルバーのアップデート

- **HiGHS 1.12**: 新しい内点法ソルバーHiPO導入（マルチスレッド活用）。 [ソース: GAMS Blog 2025]
- **SCIP 10**: Benders分解フレームワーク、IIS（既約実行不可能部分系）計算機能、CONOPTをNLPソルバーとして統合。 [ソース: GAMS Blog 2025]

---

## 分析・意見（足軽ashigaru1の見解）

### LP/MIP領域の競争状況評価

#### OSS vs 商用の現実

- **性能差**: 商用は20-30倍高速（大規模MIP）、10-20倍高速（LP）。ただし小規模問題では差が小さい。
- **OSS最速はHiGHS**: MIT ライセンス、SciPy/MathWorks/PyPSA等にデフォルトソルバーとして組み込まれ、エコシステム統合に成功。性能でOSS内最速だが、商用との差は依然大きい。
- **SCIPの位置づけ**: 包括的カバレッジ（MILP/MINLP/CP）と拡張性で学術・研究に強い。ただし外部LPソルバー依存による通信コストが性能ボトルネック。

#### 商用ソルバーの強み

- **技術的優位**: 長年の開発投資（person-decades規模）による並列化、ヒューリスティクス、カット生成の洗練度。Presolve + Cutting Planes が最重要。
- **最近の動向**: PDHG普及、GPU対応拡大、ML統合（Gurobiのヒューリスティック改良等）。ただし2024年に主要商用ソルバーがMittelmannベンチマークから撤退し、第三者評価が困難に。

#### Mosekの独自性

- 線形以外（錐最適化）が強みだが、大規模LP問題でGurobi超え（研究データあり）は注目に値する。内点法特化が線形問題でも有効なケースがある証左。

### 天下取り推奨戦略（線形分野）

#### Phase 1: 一点突破で橋頭堡確保（HiGHSモデル）

- **対象**: LP/MIPに特化。まず小～中規模問題でGurobi並み性能を目指す（HiGHSが達成済み）。
- **ライセンス**: MIT推奨。商用・学術双方で自由に使用可能。SciPy/Pandas/NumPy等への組み込みを最優先。
- **API設計**: シンプルさ最優先（学習コスト最小化）。HiGHSの成功はAPI統合の容易性が大きい。
- **技術的差別化**:
  1. **並列化/GPU活用**: 商用ソルバーがまだ弱い（Linux限定、ベータ版等）領域。cuOptのオープンソース化を活用。
  2. **特殊構造最適化**: ネットワークフロー、スケジューリング等の実問題構造を活用した高速化。
  3. **ML統合**: ハイパーパラメータ自動調整、カット選択学習（Cut Ranking等）。汎化性能向上が鍵。

#### Phase 2: エコシステム拡大

- **採用促進**: 成功事例・ユースケース蓄積。Pythonエコシステムへのデフォルト組み込み。
- **コミュニティ形成**: PyTorchモデル（企業バックアップ + OSSコミュニティ）またはPostgreSQLモデル（複数企業が利益を持つ構造）。定期イベント、明確なコントリビューションガイドライン。
- **企業スポンサー**: 複数企業が利益を持つエコシステム（単一企業依存を避ける）。

#### Phase 3: 段階的拡張（慎重に）

- 性能確立後、周辺問題クラス（QP, SOCP等）に拡大。ただしコア性能（LP/MIP）を犠牲にしない範囲で。SCIPのような包括型は学術・研究には強いが、産業界採用ではシンプル・特化型（HiGHS）が有利。

### リスクと不確実性

#### 技術的リスク

- **性能差の壁**: 大規模MIPで商用との20-30倍差を縮めるには「person-decades」規模の投資が必要。現実的目標は「1桁差まで縮める」（小～中規模で同等、大規模で十分使えるレベル）。
- **汎化性能**: ML統合において、異なる問題クラスへの汎化が未解決の重要課題。特定問題に過学習すると実用性が下がる。
- **並列化の限界**: Mosekの報告（小問題では並列化が逆効果）のように、全ての問題で並列化が有効とは限らない。問題サイズ・構造に応じた適応的並列化が必要。

#### エコシステムリスク

- **ライセンス選択の影響**: GPL選択は学術には問題ないが、産業界統合ではMIT/Apacheが圧倒的に有利。ただしGPL（SCIP）でも学術・研究分野では成功している。
- **商用ソルバーの動向**: 商用がMittelmannベンチマーク撤退した理由（Gurobiの行動により公平性議論）は、第三者評価の難しさを示す。OSS側は透明性を武器にできる。
- **GPU環境コスト**: GPU加速はハードウェア投資必要。クラウドGPUの低価格化が前提だが、中小企業にはコスト障壁。

#### 市場リスク

- **商用ソルバーの学術ライセンス**: Gurobi/CPLEX/Xpress全てが無料学術ライセンス提供。学術市場ではOSSの価格優位性がない。差別化は「自由度」「透明性」「拡張性」。
- **採用障壁**: 既存ユーザー（商用ソルバー利用企業）の移行コスト。性能以外の要素（サポート、SLA、実績）も重要。

#### 推奨リスク対応

1. **現実的な目標設定**: 全領域で商用超えは不可能。「小～中規模で同等、特定領域で商用超え」を目指す。
2. **透明性の確保**: ベンチマーク手順、生データ、再現スクリプトをGitHub公開。第三者検証可能性を担保。
3. **段階的拡張**: 最初から全問題種別をカバーせず、LP/MIPで橋頭堡確保後に周辺拡大。
4. **コミュニティ投資**: 長期的な成功はコミュニティの厚みに依存（PostgreSQLの30年の成功が証明）。

---

## 総括

線形ソルバー分野は「商用の圧倒的優位 vs OSSの急速な追い上げ」が現状である。HiGHSの成功（OSS最速、エコシステム統合）が示すのは、「一点突破 + シンプルAPI + Permissiveライセンス」戦略の有効性である。2024-2026年の技術トレンド（PDHG、ML統合、GPU加速）は商用・OSS双方で進行しており、今後5-10年でOSS性能が商用に更に接近する可能性がある。ただし大規模MIPでの性能差は依然大きく、「全領域で商用超え」は非現実的。現実的目標は「小～中規模で同等、特定領域（並列化、特殊構造）で商用超え、大規模で十分使えるレベル」であり、エコシステム統合とコミュニティ形成が長期的成功の鍵となる。
