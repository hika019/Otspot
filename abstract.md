# 数理最適化ソルバー調査　アブストラクト

調査実施日: 2026-02-12
調査担当: 足軽2番、足軽3番
統括: 家老

---

## 1. 概要

本調査は、数理最適化ソルバーの現状と今後の開発戦略を明らかにするため、オープンソース（OSS）ソルバー、商用ソルバー、最新研究動向（2024-2026年）、ベンチマーク手法、差別化戦略の5つの観点から包括的に実施した。調査対象は、HiGHS、SCIP、CBC、OR-Tools、GLPK、Ipopt等のOSSソルバー、Gurobi、CPLEX、Mosek、FICO Xpress等の商用ソルバー、並びにPDHG、GPU加速、機械学習統合等の最新技術トレンドである。目的は、実務で使える高性能ソルバーの技術要件を明確化し、既存ソルバーとの差別化ポイントを特定することにある。

---

## 2. 主要知見（事実・ソース付き）

### 2.1 OSSソルバーの現状と主要プレイヤー

**HiGHS（最速OSS）**
- LP/MIP/QP対応、MITライセンス。Mittelmannベンチマークで世界中のOSS線形最適化ソフトウェアの中で最高性能 [出典: HiGHS公式サイト]
- Gurobiとは約1桁の性能差（Gurobiの方が高速）。小～中規模問題ではGurobiと同等性能 [出典: HiGHS Discussion #1683]
- デフォルトソルバーとして採用実績: SciPy 1.6.0以降、MathWorks Optimization Toolbox、NAGライブラリ [出典: HiGHS Wikipedia]
- 2025年にPDHGアルゴリズム導入（v1.10）、新内点法HiPO導入（v1.12） [出典: GAMS Blog 2025]

**SCIP（包括的カバレッジ）**
- MILP/MINLP/CP対応、Apache 2.0またはLGPL。最新リリースSCIP 10.0.1（2026年2月3日） [出典: SCIP GitHub Releases]
- MittelmannベンチマークでHiGHSとGurobiの中間性能 [出典: HiGHS Discussion #1683]
- フレームワークとしての拡張性: SCIP-Jack（Steiner tree solver）、SCIP-SDP（半正定値計画）等 [出典: SCIP Official Site]

**その他OSS**
- CBC: 旧世代MIPソルバー、HiGHSに劣後 [出典: HiGHS Discussion #1683]
- OR-Tools: Google製、複数ソルバーのラッパー、CP-SAT受賞歴あり [出典: OR-Tools Official Site]
- GLPK: LP/MIP対応、GPL v3、並列化は研究段階 [出典: GLPK GNU Project]
- Ipopt: NLP特化、並列線形ソルバー経由で並列化 [出典: Ipopt Documentation]

### 2.2 商用ソルバーの優位性と技術的核心

**性能差の実態**
- MIP: 商用（CPLEX/Xpress/Gurobi）はOSS（HiGHS/CBC/SCIP）より20-30倍高速（一部ケース） [出典: OSS vs Commercial Analysis]
- LP: 最良OSS（HiGHS）は最良商用（COPT）より20倍遅い [出典: HiGHS Discussion #1683]
- 大規模問題では性能差が1桁超に拡大 [出典: Gurobi Open Source vs Gurobi]

**技術的コア（商用の強み）**
1. **Cutting Planes（切除平面法）**: Gurobiは対称性検出を含む最新実装。Gomory混合整数カットで2.52倍、MIRで1.83倍高速化 [出典: Gurobi Help - Cutting Planes]
2. **Presolve（前処理）**: CPLEXはProbingなど高コストだが強力な削減手法を実装 [出典: Presolve and Cutting Planes - ZIB]
3. **並列処理**: Gurobiはデフォルトで全CPUコア使用、分散コンピューティング対応 [出典: Gurobi Parallel Optimization]
4. **GPU対応**: Gurobi 13、Xpress 46でPDHG GPU実装。大規模LPで最大50倍高速化（Xpress） [出典: FICO Xpress Solver]

**機能的アドバンテージ**
- CPLEX: 並列MIP（ノード並列、Strong branching並列、分散処理）、動的タスク分解（2025年研究） [出典: CP 2025 Parallel MIP]
- Mosek: 錐最適化（SOCP/SDP）最強、内点法で大規模LPでGurobi超え [出典: arXiv MINLP Study]
- Xpress: MINLP性能68%高速化、100秒超モデルで5.3倍高速 [出典: FICO Xpress MIP Performance]

**ベンチマーク撤退問題**
- 2024年8月Gurobi、2024年12月MindOptがMittelmannベンチマークから撤退。第三者評価が困難に [出典: Mittelmann Benchmark]

### 2.3 最新研究動向のハイライト（2024-2026年）

**PDHGアルゴリズムの急速普及**
- 2024年COPT先駆導入、2025年にGurobi 13、HiGHS 1.10、Knitro 15、Xpress 46が続々導入 [出典: GAMS Blog 2025]
- GPU対応版がHiGHS、Gurobi、Xpressで利用可能 [出典: GAMS Blog 2025]

**機械学習統合の実用化**
- Cut Ranking: 産業用ソルバーに実装、平均12.42%高速化（精度劣化なし） [出典: Learning to Select Cuts - arXiv]
- 分枝変数選択、ノード選択、カット平面生成への適用研究が爆発的増加 [出典: GAMS Blog 2025]
- 理論的基盤確立: 2025年ICLR論文でサンプル複雑度境界を確立 [出典: Generalization Guarantees - ICLR 2025]

**NVIDIA cuOptの登場**
- GPU加速MIP/LP/VRPソルバー、2025年オープンソース化、COIN-OR Cup受賞 [出典: COIN-OR Cup Award]
- HiGHSとの統合で最適性ギャップ28%→21%改善 [出典: HiGHS and cuOpt Blog]
- MILP最大8.6倍高速化（高精度維持） [出典: SimpleRose cuOpt Integration]

**性能向上の実態**
- 「10年前には到達不可能だった問題を数秒で解ける」（Mathematical Programming誌サーベイ論文） [出典: Last Fifty Years of Integer LP]

### 2.4 ベンチマーク手法の標準

**標準テストセット**
- MIPLIB 2017: 初期プール5,721インスタンスから240インスタンスをBenchmark Setとして選定（最新公開版） [出典: MIPLIB 2017]
- MIPLIB 2024: 2024年投稿受付開始、移行進行中 [出典: MIPLIB 2024 Submission]
- Netlib: LP専用、90問テストセット（最大13,525変数、3,000制約） [出典: Benchmarking ALGLIB]

**評価指標**
- 主要メトリクス: 解答時間、最適性ギャップ、ノード数、メモリ使用量 [出典: Gurobi MIP Models Documentation]
- Shifted Geometric Mean: 計算MIPの標準性能指標。シフト値s=10が実務標準（Mittelmannベンチマーク） [出典: Visualizations of Mittelmann benchmarks]
- Performance Profiles: Dolan & Moré (2002)による多様なインスタンス集合での包括的比較手法、Google Scholarで約5,000件引用 [出典: A Note on Performance Profiles]

**比較方法論**
- ハードウェア統一、時間制限設定（通常1-2時間/インスタンス）、デフォルトパラメータ使用が原則 [出典: Benchmarks for Current Linear and Mixed Integer Optimization Solvers]
- 透明性確保: テストセット選定理由、ハードウェアスペック、パラメータ設定の全公開 [出典: Benchmarks - ResearchGate]

### 2.5 差別化・天下取りの要点

**OSSの成功モデル: HiGHS**
- 一点突破戦略: LP/MIP特化、MITライセンス、シンプルAPI [出典: HiGHS Wikipedia]
- エコシステム統合: SciPy、MathWorks、NAGへのデフォルト組み込み [出典: HiGHS Documentation]
- 技術的差別化: 小～中規模でGurobi同等、GPU対応PDHG導入 [出典: HiGHS Discussion #1683, GAMS Blog 2025]

**OSSの成功モデル: SCIP**
- 包括的カバレッジ: MILP/MINLP/CP対応、フレームワーク提供 [出典: SCIP Official Site]
- 学術・研究特化: 拡張プロジェクト（SCIP-Jack, SCIP-SDP）の生成 [出典: Why SCIP? - PySCIPOpt]

**商用ソルバーの弱点（攻め口）**
- 並列化（GPU活用）が未成熟: 2025年にようやくPDHG GPU対応開始 [出典: GAMS Blog 2025]
- 小規模問題では性能差縮小: HiGHSがGurobi同等 [出典: HiGHS Discussion #1683]
- ライセンス制約・価格不透明: 全商用ソルバーが価格非公開（カスタム見積） [出典: 商用調査レポート]

**技術的ブレークスルー候補**
- GPU並列化: cuOptのオープンソース化が証明（8.6倍高速化） [出典: SimpleRose cuOpt Integration]
- ML統合: Cut Ranking産業実装で12.42%高速化 [出典: Learning to Select Cuts - arXiv]
- クラウドネイティブ設計: 分散環境最適化 [出典: 天下取り戦略レポート]

**コミュニティ戦略の成功例**
- PyTorch: Facebook支援 + 900人以上OSSコントリビューター、明確なコントリビューションガイドライン [出典: CircleCI PyTorch Case Study]
- PostgreSQL: 30年の歴史、複数企業が利益を持つエコシステム、単一企業非依存 [出典: FOSDEM 2026]
- HiGHS: MITライセンス + 主要ライブラリ統合 + 7言語バインディング [出典: HiGHS Wikipedia]

**ライセンス戦略の影響**
- MITライセンス: 最大限の配布、プロプライエタリ組み込み可能 → HiGHSの成功要因 [出典: Understanding Open Source Licenses - credativ]
- Apache 2.0: 企業プロジェクト向け特許保護付き → SCIP 10.0で採用 [出典: SCIP Suite 10.0 Paper]
- GPL: 最大限のオープン性だがビジネス統合に制約 → GLPK採用 [出典: GLPK.jl JuMP]

---

## 3. 分析・意見（家老/足軽の見解）

### 3.1 最も重要な発見

**【足軽の見解】技術パラダイムの転換点に到達**

2024-2026年は数理最適化ソルバー史において画期的な転換期である。3つの革命が同時進行している:

1. **アルゴリズム革命**: PDHGの全面普及（商用・OSS双方）。従来の内点法・単体法に次ぐ第三の柱が確立
2. **計算資源革命**: cuOptオープンソース化によるGPU最適化の民主化。8.6倍高速化は実務に直結
3. **知能化革命**: ML統合の実用化（Cut Ranking 12.42%高速化）。「研究段階」から「産業実装」へ移行

これらは独立ではなく相互強化している。GPU並列化が大規模ML訓練を可能にし、学習済みモデルがカット選択を最適化し、分枝限定木探索が効率化される好循環が生まれている。

**【家老の見解】OSSが商用に迫る現実的シナリオが見えた**

HiGHSの成功（小～中規模でGurobi同等）は、「OSS = 商用の劣化版」という固定観念を打破した。成功要因は明確:
- 一点突破（LP/MIP特化）
- MITライセンスによる広範採用
- エコシステム統合（SciPy等へのデフォルト組み込み）

この戦略は再現可能である。ただし、大規模問題では依然1桁の性能差がある（Gurobi等との比較）。この差を埋めるには、商用が未成熟な領域（GPU並列化、ML統合、クラウド分散処理）への集中投資が必須。

### 3.2 矛盾する情報や不確実な点

**【矛盾点1】ベンチマークの信頼性低下**

Mittelmannベンチマークから主要商用ソルバー（Gurobi、CPLEX、Xpress、MindOpt）が2024年に相次いで撤退した。公式理由は「公平性への懸念」だが、これにより第三者による性能比較が困難になった。現在の性能差データ（1桁差等）は撤退前の古いデータに依存しており、2026年時点の正確な差は不明。

**【矛盾点2】ML統合の汎化性能**

Cut Rankingは特定の産業用問題で12.42%高速化を達成したが、汎化性能（異なる問題クラスへの適用）については明示的なデータがない。2025年ICLR論文で理論的基盤は確立されたが、実務での広範な適用可能性は未検証。

**【不確実点1】GPU加速の費用対効果**

cuOptはMILP 8.6倍高速化を達成したが、NVIDIA H100 GPU（推奨ハードウェア）の調達コストは非公開。中小企業での採用障壁（GPU環境構築コスト）が実際にどの程度かは不明。クラウドGPU低価格化の進展次第で状況が変わる。

**【不確実点2】MIPLIB 2024への移行時期**

MIPLIB 2024への投稿受付は2024年に開始されたが、公式リリース時期は不明。現在の標準はMIPLIB 2017（240インスタンス）だが、移行期における混乱（どちらで比較すべきか）が予想される。

### 3.3 要件定義に向けた示唆

**【家老の示唆】3段階の開発戦略を推奨**

**Phase 1: 橋頭堡確保（1-2年目）**
- **ターゲット**: LP/MIP特化、中規模問題でHiGHS同等性能
- **技術**: PDHG実装（既に成熟）、基本的なカット生成（Gomory, MIR）
- **ライセンス**: MIT（広範採用最優先）
- **API設計**: PythonファーストでNumPy/Pandas統合、シンプルなインターフェース
- **ベンチマーク**: MIPLIB 2017 Benchmark Set（240インスタンス）で検証
- **目標**: Netlib 90問で95%成功率、MIPLIB中規模問題で解答時間shifted geometric mean 2.0以下（HiGHS比）

**Phase 2: 差別化（2-3年目）**
- **GPU並列化**: cuOptとの統合またはPDHG GPU実装（8倍高速化目標）
- **ML統合**: Cut Ranking実装（10%高速化目標）
- **エコシステム統合**: SciPy/Pandas/OR-Toolsへのプラグイン提供
- **コミュニティ形成**: GitHub Star 1,000以上、月次コントリビューター10人以上
- **目標**: 特定領域（例: ネットワークフロー、スケジューリング）で商用ソルバー同等性能

**Phase 3: 市場確立（3-5年目）**
- **クラウドネイティブ**: 分散環境での並列処理最適化
- **問題クラス拡張**: QP、SOCP等への段階的拡大（ただしLP/MIP性能は維持）
- **企業スポンサー獲得**: PostgreSQLモデル（複数企業が利益を持つ構造）
- **目標**: 商用ソルバー学術ライセンスの代替として認知される

**【足軽の技術的示唆】実装優先順位**

1. **最優先（Phase 1必須）**:
   - PDHG実装（既に成熟、HiGHS/Gurobi/Xpress実績あり）
   - Gomory混合整数カット、MIRカット（Gurobiで実証済み高速化）
   - 基本的なプリソルバ（変数削減、制約強化）
   - Python API（NumPy配列対応）

2. **高優先（Phase 2推奨）**:
   - GPU PDHG（cuOpt統合またはCUDA直接実装）
   - Cut Ranking ML実装（PyTorch/TensorFlow活用）
   - 並列分枝限定（ノード並列）

3. **中優先（Phase 3検討）**:
   - 分散並列MIP（CPLEX動的タスク分解を参考）
   - 錐最適化（SOCP/SDP）

4. **低優先（様子見）**:
   - 非線形（Ipopt/Mosekとの競合回避）
   - 制約プログラミング（SCIP/OR-Toolsが強い）

**【家老の市場戦略示唆】**

商用ソルバーとの直接競合は避け、「棲み分け」を意識すべき:
- **狙う市場**: 小～中規模問題、予算制約のあるスタートアップ・中小企業、学術研究（学術ライセンス更新の手間回避）
- **避ける市場**: 大規模MIP（20-30倍差は埋められない）、ミッションクリティカル（商用サポート必須）
- **差別化軸**: 「無料」「軽量」「統合容易」「GPU対応」

**【足軽のライセンス・コミュニティ示唆】**

- **ライセンス**: MIT一択（HiGHSの成功モデル）。Apache 2.0は特許保護が必要な場合のみ検討
- **コントリビューションガイドライン**: 初日から整備（PyTorchモデル）
- **定期イベント**: 月次オンラインミートアップ + 年1回対面ワークショップ
- **企業スポンサー戦略**: 初期は単一企業バックアップ（開発資金確保）、成熟後に複数企業体制へ移行（PostgreSQLモデル）

---

**調査完了日**: 2026-02-12
**調査チーム**: 足軽2番（商用・天下取り戦略）、足軽3番（OSS・研究動向・ベンチマーク）
**統括**: 家老（全体分析・戦略策定）
