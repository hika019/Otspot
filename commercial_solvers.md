# 商用ソルバー調査レポート

調査対象: Gurobi, CPLEX (IBM), Mosek, FICO Xpress
調査日: 2026-02-12
調査担当: ashigaru2

---

## 1. 総合比較表

| ソルバー | 対応問題種別 | 性能評価 | 学術ライセンス | 商用価格帯 | 強み |
|---------|------------|---------|--------------|-----------|------|
| **Gurobi** | LP/MIP/QP/MIQP/QCP/MIQCP/SOCP/MISOCP/非凸QP | MIP/QP/QCPで最速級 | 無料（full-featured） | サブスク制（要問合せ） | GPU対応（PDHG）、並列処理、カット生成 |
| **CPLEX** | LP/MIP/QP/MIQP/QCP/MIQCP/SOCP | MIP最速級の一角 | 無料（IBM Academic Initiative） | サブスク制（要問合せ） | 並列B&C、分散処理、強力なプリソルバ |
| **Mosek** | LP/MIP/QP/MIQP/SOCP/SDP/指数錐/累乗錐 | 錐最適化で最強、大規模LPで優位 | 学術プログラム有 | 要問合せ | 内点法、錐最適化、SDP |
| **Xpress** | LP/MIP/QP/MIQP/非線形/制約プログラミング | MIP高速 | 無料（FICO Academic Partner） | カスタム見積 | GPU対応（PDHG）、MIP 14-68%高速化 |

**ベンチマーク状況（2024-2025）:**
主要商用ソルバー（Gurobi/CPLEX/Xpress）はMittelmannベンチマークから撤退済。公式比較データが減少傾向。

---

## 2. Gurobi

### 事実

#### 対応問題種別
- LP（線形計画）、MIP（混合整数線形計画）
- QP（二次計画）、MIQP（混合整数二次計画）
- QCP（二次制約付き計画）、MIQCP（混合整数二次制約付き計画）
- SOCP（二次錐計画）、MISOCP（混合整数二次錐計画）
- 凸・非凸QP、双線形、非線形問題
出典: [AMPL Gurobi](https://ampl.com/products/solvers/linear-solvers/gurobi/), [Gurobi Optimizer](https://www.gurobi.com/solutions/gurobi-optimizer/)

#### 性能
- MIPソルバー中で最速、QP/QCPソルバー中でも最速と評価（過去データ）
出典: [solver.com - Gurobi](https://www.solver.com/gurobi-solver-engine)
- **重要**: 2024年8月、Gurobiは公式Mittelmannベンチマークから撤退。結果削除済
出典: [Mittelmann Benchmark](https://plato.asu.edu/bench.html)

#### ライセンス・価格
- 学術ライセンス: 無料（full-featured、モデルサイズ制限なし）。大学の教員・学生・スタッフ対象
出典: [Gurobi Academic Program](https://www.gurobi.com/academia/academic-program-and-licenses/)
- 商用: サブスクリプション制（月額・年額）、柔軟なライセンス形態（ローカル/クラウド/コンテナ対応）
出典: [AMPL Gurobi](https://ampl.com/products/solvers/linear-solvers/gurobi/)
- 無料試用版あり
出典: [Gurobi Benchmarks](https://www.gurobi.com/resources/use-benchmarks-to-find-the-best-solver-for-your-needs/)

#### 技術的強み

**カット生成:**
- Gomory混合整数カット（2.52倍高速化）、MIR（混合整数丸め、1.83倍高速化）
- Flow cover, Knapsack cover, Clique, Implied bound, Flow path, GUB coverカット
- 適応的戦略: 問題特性に応じてGomory/lift-and-project/implied boundカットを動的適用
出典: [Gurobi Help - Cutting Planes](https://support.gurobi.com/hc/en-us/community/posts/360060841412-Cutting-planes), [MIR Medium](https://medium.com/@minkyunglee_5476/integer-programming-the-cutting-plane-algorithm-26bbabf04815)

**プリソルバ:**
- MIPに特に重要。Probingなど高コストだが強力な削減手法を実装
- カットは前処理済みモデルに適用
出典: [Presolve and Cutting Planes](https://co-at-work.zib.de/berlin2009/downloads/2009-09-25/2009-09-25-1400-BB-MIP-4.pdf)

**分枝限定法:**
- 1990年代中盤以降、カット生成とbranch-and-boundを組み合わせたbranch-and-cutが主流
- 数値不安定性克服手法も実装
出典: [Cutting-plane method - Wikipedia](https://en.wikipedia.org/wiki/Cutting-plane_method)

**並列化:**
- デフォルトで全CPUコアを使用。マルチスレッド対応
- Primal/dual simplex、barrier法を並列実行（1つ収束後も他が終了待ちする設計）
出典: [Gurobi Parallel Optimization](https://www.gurobi.com/features/gurobi-optimizer-delivers-parallel-optimization/), [Google Groups - Parallel Default](https://groups.google.com/g/gurobi/c/7Q0s080QDi8)
- 分散コンピューティング: 単一ワークステーション/マルチコアクラスタ/クラウドでシームレスにスケール
出典: [Gurobi Distributed Optimization](https://cdn.gurobi.com/wp-content/uploads/webinar-parallel-and-distributed-optimization-english.pdf)

**GPU対応（2025年）:**
- GPU対応PDHG（Primal-Dual Hybrid Gradient）実装
- 要件: NVIDIA H100 GPU推奨、CUDA 12.9、Linux 64bit（Windowsは未対応）、NVIDIA cuOptエンジン使用
- 巨大LPに最適（従来困難だったスケールに対応）
出典: [Gurobi GPU Support](https://support.gurobi.com/hc/en-us/articles/360012237852-Does-Gurobi-support-GPUs), [Gurobi GPU Solver](https://www.gurobi.com/resources/introducing-gurobis-first-gpu-accelerated-solver/)
- 2025 GTC: NVIDIA CEOがcuOptのPDLP（Primal-Dual LP）をオープンソース化発表
出典: [GPU-Accelerated Unit Commitment](https://arxiv.org/html/2512.06715v1)

#### API・バインディング
- Python, C, C++, Java, .NET, MATLAB, R対応
- 公式ドキュメント完備
出典: [Gurobi Documentation](https://docs.gurobi.com/projects/optimizer/en/current/concepts/parameters/guidelines.html)

### 分析

**強み:**
- MIP/QP/QCPで最速級の実績（ベンチマーク撤退前の評価）
- GPU対応（Linux限定）により巨大LP問題に新たな可能性
- 学術ライセンスが制限なしで使いやすい
- カット生成の種類が豊富で適応的戦略を持つ

**弱み:**
- ベンチマーク撤退により第三者評価が困難に
- GPU対応がLinux限定（Windows未対応）
- 商用価格非公開（見積要）

**適用場面:**
- 大規模MIP/QP問題で最高性能を求める場合
- GPU活用できる環境（Linux）で巨大LP問題を扱う場合
- 学術研究（無料ライセンス活用）

---

## 3. CPLEX (IBM)

### 事実

#### 対応問題種別
- LP, MIP, QP, MIQP, QCP, MIQCP, SOCP
- Simplex法、Barrier法、Branch-and-bound、Cutting plane、Presolve技術
出典: [IBM CPLEX Optimizer](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer), [AMPL CPLEX](https://ampl.com/products/solvers/linear-solvers/cplex/)

#### 性能
- MIP分散並列アルゴリズム、高性能LP/MIPソルバー
- 高度なプリソルブ、並列処理、ロバストなbranch-and-cut、実行可能解発見ヒューリスティクス
出典: [IBM CPLEX Features](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer)

#### ライセンス・価格
- 学術ライセンス: 無料（1年、IBM Academic Initiative経由）、機能・モデルサイズ制限なし
出典: [IBM Academic Initiative](https://www.ibm.com/products/ilog-cplex-optimization-studio), [AMPL Academia](https://ampl.com/academia/supported-discounted-licenses-for-academia/)
- 商用: サブスクリプション制（月額・年額）、開発用途限定、変数・制約は無制限
出典: [IBM CPLEX Pricing](https://www.ibm.com/products/ilog-cplex-optimization-studio/pricing), [AMPL Buy CPLEX](https://ampl.com/buy-cplex/)
- 無料版: 1,000変数・1,000制約まで（No-cost edition）
出典: [AMPL CPLEX](https://ampl.com/products/solvers/linear-solvers/cplex/)

#### 技術的強み

**カット生成・プリソルバ:**
- Simplex/Barrier法、Branch-and-bound、Cutting plane、Presolve技術を統合
出典: [IBM CPLEX Features](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer)

**並列化・分散処理:**
- **スレッド並列**: グローバルThreadsパラメータ>1でノード並列処理。CPLEXがノードプール自動管理、スレッド終了時に次ノード割当
出典: [CPLEX Parallel MIP](https://www.tu-chemnitz.de/mathematik/discrete/manuals/cplex/doc/userman/html/moreUsing29.html)
- **Strong branching並列化**: StrongThreadLim>1で変数選択計算を複数プロセッサで並列実行
出典: [CPLEX Parallel MIP](http://www-eio.upc.edu/lceio/manuals/cplex75/doc/usermanccpp/html/moreUsing23.html)
- **分散並列**: Coordinated concurrent approach、Branch-and-cut tree並列化の2アプローチ。有望な性能
出典: [LLNL Distributed CPLEX](https://www.osti.gov/servlets/purl/1165747), [ResearchGate PUBB2](https://www.researchgate.net/publication/220767909_Effectiveness_of_parallelizing_the_ILOG-CPLEX_mixed_integer_optimizer_in_the_PUBB2_framework)
- **2025年最新研究**: Dynamic task decomposition。Scheduler（タスクツリー管理）＋Workers（MIPソルバー起動）の役割分担。ルートWorkerは元問題、一般Workerは動的割当タスクを解く
出典: [CP 2025 Parallel MIP](https://drops.dagstuhl.de/storage/00lipics/lipics-vol340-cp2025/LIPIcs.CP.2025.26/LIPIcs.CP.2025.26.pdf)

**性能実績:**
- 並列MIPで顕著な高速化。特に多数ノード処理する困難なモデルで効果大
出典: [CPLEX Parallel Performance](https://www.tu-chemnitz.de/mathematik/discrete/manuals/cplex/doc/userman/html/moreUsing29.html)

#### API・バインディング
- **Concert Technology**: C++, C#, Java用インターフェース
- **Python API**: CベースAPI上に構築（C互換）
出典: [CPLEX API Overview](https://www.ibm.com/docs/el/icos/20.1.0?topic=cplex-python-reference-manual), [CPLEX Getting Started](https://home.engineering.iastate.edu/~jdm/ee458/CPLEX-UsersManual2015.pdf)
- **Docplex**: 管理されたOOP API（C APIラッパー）
- **公式ドキュメント**: Python/Java/C++ APIリファレンスマニュアル
出典: [IBM CPLEX Python API](https://www.ibm.com/docs/el/icos/20.1.0?topic=cplex-python-reference-manual), [Python Pool CPLEX](https://www.pythonpool.com/cplex-python/)

### 分析

**強み:**
- 並列MIP処理が成熟（ノード並列、Strong branching並列、分散処理）
- 2025年最新研究（動的タスク分解）で更なる性能向上期待
- 学術ライセンスが制限なし（1年更新）
- IBM製品エコシステムとの統合性
- Concert Technologyによる多言語対応

**弱み:**
- 商用価格非公開
- GPU対応の記載なし（調査範囲内）
- 無料版は1,000変数制限（学術外で試用しづらい）

**適用場面:**
- 大規模MIPで並列処理・分散処理が必要な場合
- IBM製品スタック内での利用
- 学術研究（1年ライセンス更新運用可能な場合）

---

## 4. Mosek

### 事実

#### 対応問題種別
- LP, MIP, QP, MIQP, SOCP（二次錐計画）
- SDP（半正定値計画）
- 指数錐（exponential cone）、累乗錐（power cone）※バージョン9で追加
- 凸非線形最適化
出典: [MOSEK Wikipedia](https://en.wikipedia.org/wiki/MOSEK), [GAMS MOSEK](https://www.gams.com/latest/docs/S_MOSEK.html), [MathWorks MOSEK](https://www.mathworks.com/products/connections/product_detail/mosek.html)

#### 性能
- **内点法が最強**: 連続LP/QP/錐問題で最先端の内点最適化器
出典: [solver.com MOSEK](https://www.solver.com/mosek-solver-engine), [GAMS MOSEK](https://www.gams.com/latest/docs/S_MOSEK.html)
- **大規模問題で優位**: MosekとGurobi比較研究で、Mosekが大規模問題でGurobiを上回り、実行可能性でより正確な解
出典: [arXiv MINLP Study](https://arxiv.org/pdf/2303.04216)
- **同次モデル（homogeneous model）**: 原始・双対の実行不可能性を確実に検出可能。複数論文で実証
出典: [MOSEK Wikipedia](https://en.wikipedia.org/wiki/MOSEK)

#### ライセンス・価格
- 学術プログラムあり（詳細は要確認）
出典: [AMPL Academia](https://ampl.com/academia/supported-discounted-licenses-for-academia/)
- 商用: 要問合せ
出典: [AMPL MOSEK](https://ampl.com/products/solvers/linear-solvers/mosek/)

#### 技術的強み

**内点法アルゴリズム:**
- 同次内点法（homogeneous interior-point）。先進的な並列化線形代数、密な列を効率処理
出典: [Springer MOSEK Interior Point](https://link.springer.com/chapter/10.1007/978-1-4757-3216-0_8), [GAMS MOSEK](https://www.gams.com/latest/docs/S_MOSEK.html)
- 実行不可能性検出が信頼性高い
出典: [ResearchGate Homogeneous Algorithm](https://www.researchgate.net/publication/243774586_The_Mosek_Interior_Point_Optimizer_for_Linear_Programming_An_Implementation_of_the_Homogeneous_Algorithm)

**並列化:**
- 内点法・混合整数最適化器が並列化済
- 線形代数計算などの大タスクを並列化
出典: [MOSEK Rmosek Guidelines](https://docs.mosek.com/latest/rmosek/guidelines-rmosek.html), [MOSEK Python Guidelines](https://docs.mosek.com/latest/pythonapi/guidelines-optimizer.html)
- スレッド数: デフォルト自動選択、MSK_IPAR_NUM_THREADSパラメータで最大数設定可
出典: [MOSEK MATLAB Threading](https://docs.mosek.com/8.1/toolbox/solving-parallel.html)
- **性能ガイドライン**: 並列化効果は問題・ハードウェア依存。小問題（60秒未満）では並列化は逆効果（オーバーヘッド）。アルゴリズムの全部分を並列化できないため、CPU使用率が1コア分だけの時もある
出典: [MOSEK Guidelines](https://docs.mosek.com/latest/rmosek/guidelines-rmosek.html)

#### API・バインディング
- C/C++, Java, Python, MATLAB, .NET, Julia, Rust, R対応
出典: [MOSEK Interfaces](https://docs.mosek.com/latest/intro/interfaces.html)
- **Optimizer API**: 行列指向の最適化インターフェース。C Optimizer APIがコア（最適化アルゴリズム含む、C互換言語から利用可）、全APIがその上に構築
出典: [MOSEK Interfaces](https://docs.mosek.com/latest/intro/interfaces.html)
- **Fusion API**: オブジェクト指向の錐最適化表現用API
出典: [MOSEK Fusion Python](https://docs.mosek.com/latest/pythonfusion/index.html), [MOSEK Fusion C++](https://docs.mosek.com/latest/cxxfusion/index.html)
- Python: numpy配列受付
出典: [MOSEK Interfaces](https://docs.mosek.com/latest/intro/interfaces.html)
- ドキュメント: mosek.com/documentationでHTML/PDF版利用可（2025年版）
出典: [MOSEK Documentation](https://www.mosek.com/documentation/)

#### 産業応用
- テクノロジー、金融、エネルギー、林業業界で広く採用
出典: [MOSEK ApS](https://www.mosek.com/)

### 分析

**強み:**
- 錐最適化（SOCP/SDP）に特化、内点法最強
- 大規模LP問題でGurobi超え（研究データあり）
- 実行不可能性検出が信頼性高い（同次モデル）
- 指数錐・累乗錐対応（他ソルバーにない機能）
- 金融・エネルギー分野に強い

**弱み:**
- MIP性能は他商用ソルバーに劣る可能性（内点法特化のため）
- 小問題では並列化が逆効果（60秒未満）
- GPU対応の記載なし
- 学術ライセンス詳細不明

**適用場面:**
- 錐最適化（SOCP/SDP）が必要な問題
- ポートフォリオ最適化など金融工学
- 大規模LP問題（内点法が有利な場合）
- 実行可能性判定が重要な問題

---

## 5. FICO Xpress

### 事実

#### 対応問題種別
- LP, MIP, QP, MIQP, 非線形、混合整数非線形（MINLP）
- 制約プログラミング（Constraint Programming）
出典: [FICO Xpress Wikipedia](https://en.wikipedia.org/wiki/FICO_Xpress), [FICO Xpress Overview](https://www.fico.com/fico-xpress-optimization/docs/latest/overview.html), [MathWorks Xpress](https://www.mathworks.com/products/connections/product_detail/fico-xpress-optimization.html)

#### 性能
- **MIP性能向上**: 2020年比で最大5.7倍高速化。全体14%高速、100秒超モデルで24%高速
出典: [FICO Xpress MIP Performance](https://www.fico.com/blogs/blogs/experience-faster-mixed-integer-programming-optimization-with-xpress)
- **MINLP性能**: 全体68%高速、100秒超モデルで5.3倍高速
出典: [FICO Xpress MIP Performance](https://www.fico.com/blogs/blogs/experience-faster-mixed-integer-programming-optimization-with-xpress)
- 高速、ロバスト、柔軟性を重視。先進的プリソルブ、カット生成、並列処理、高度なヒューリスティクス
出典: [FICO Xpress Solver](https://www.fico.com/en/latest-thinking/solution-sheet/fico-xpress-solver), [Artelys Xpress](https://www.artelys.com/solvers/xpress/)

#### ライセンス・価格
- 学術ライセンス: 無料（FICO Academic Partner Program経由）
出典: [AMPL Academia](https://ampl.com/academia/supported-discounted-licenses-for-academia/)
- 商用: カスタム見積。ニーズに応じた価格設定
出典: [Capterra Xpress Pricing](https://www.capterra.com/p/162284/FICO-Xpress-Optimization/), [AMPL Xpress](https://ampl.com/products/solvers/linear-solvers/xpress/)

#### 技術的強み

**アルゴリズム:**
- **LP/QP**: Primal simplex, Dual simplex, Barrier interior-point法、Primal-dual hybrid gradient（PDHG）
- **MIP/非凸問題**: Branch-and-bound + Cutting-plane
出典: [FICO Xpress Wikipedia](https://en.wikipedia.org/wiki/FICO_Xpress)

**並列化:**
- マルチスレッド並列処理を標準搭載。複数CPUコア活用、数十コアまでスケール良好
出典: [FICO Xpress Solver](https://www.fico.com/en/latest-thinking/solution-sheet/fico-xpress-solver)

**GPU対応:**
- GPU対応PDHG実装（バージョン9.8からベータ版）
- 対応OS: Linux（x86_64, ARM64）、Windows（x86_64）
- 使用法: BARALG=4（hybrid gradient solver起動）+ BARHGGPU=1（GPU使用）。CUDAライブラリとGPU存在を自動チェック
出典: [FICO Xpress Wikipedia](https://en.wikipedia.org/wiki/FICO_Xpress)
- **GPU性能**: 大規模LPで最大50倍高速化
出典: [FICO Xpress Solver](https://www.fico.com/en/latest-thinking/solution-sheet/fico-xpress-solver)

#### API・バインディング
- **BCL（Builder Component Library）**: C, C++, Java, .NET Framework用
- **独立インターフェース**: Python, MATLAB
出典: [FICO Xpress Wikipedia](https://en.wikipedia.org/wiki/FICO_Xpress)
- **Python**: xpressモジュール。NumPy対応、問題作成・解決・クエリ機能
出典: [PyPI Xpress](https://pypi.org/project/xpress/), [FICO Python Reference](https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/python/HTML/GUID-616C323F-05D8-3460-B0D7-80F77DA7D046.html)
- **Java**: バージョン9.4で新オブジェクト指向.NET/Javaインターフェース導入。Java 8以上必要
出典: [FICO Xpress Wikipedia](https://en.wikipedia.org/wiki/FICO_Xpress), [FICO Release Notes](https://www.fico.com/fico-xpress-optimization/docs/latest/relnotes/GUID-85032F3B-84B8-42A1-A4D4-A0A24FF0A648.html)
- **C++**: C++ 17以上必要
出典: [FICO Xpress Wikipedia](https://en.wikipedia.org/wiki/FICO_Xpress)

### 分析

**強み:**
- MIP/MINLP性能が近年大幅向上（14-68%高速化）
- GPU対応がWindows/Linuxで利用可（Linux+ARM64も対応）
- GPU大規模LP性能が50倍高速化（Gurobiと並ぶGPU対応）
- 非線形・制約プログラミング対応（多様な問題に適用可）
- NumPy対応でPythonユーザーに便利

**弱み:**
- Mittelmannベンチマーク非参加（第三者評価困難）
- GPU対応がベータ版（安定性要確認）
- 商用価格カスタム見積（透明性低い）
- 学術ライセンス詳細不明

**適用場面:**
- 大規模LPでGPU活用したい場合（Windows環境可）
- 非線形最適化・制約プログラミングが必要
- NumPy中心のPythonワークフロー

---

## 6. OSSソルバーとの性能差

### 事実

#### 全体的な性能差
- **MIP**: 商用（CPLEX/Xpress/Gurobi）はOSS（HiGHS/CBC/SCIP）より約2桁高速（一部ケース）。Mittelmannベンチマークでは約1桁差
出典: [HiGHS Discussion #1683](https://github.com/ERGO-Code/HiGHS/discussions/1683)
- **具体的数値**: CBCが最速OSSだが、Gurobi/CPLEXは20-30倍高速。小規模モデルでGUROBIが最速、次CPLEX、SCIPが最遅
出典: [OSS vs Commercial Analysis](http://www.tdp.cat/issues11/tdp.a114a12.pdf)

#### LP性能差
- **HiGHS vs COPT**: Mittelmannベンチマークで、最良OSS（HiGHS）は最良商用（COPT）より20倍遅い
出典: [HiGHS Discussion #1683](https://github.com/ERGO-Code/HiGHS/discussions/1683)
- **OSS内比較**: 小問題でSCIP/HiGHS/GLPKは同程度の性能
出典: [Wageningen Study](https://edepot.wur.nl/638173)

#### 性能差の理由
- 商用MIPは「person-decades」規模の開発投資（高額ライセンス収益で資金調達）
- 並列ツリー探索、各種問題クラス対応トリック
出典: [HiGHS Discussion #1683](https://github.com/ERGO-Code/HiGHS/discussions/1683)

### 分析

**商用の優位性:**
- MIPで20-30倍、LPで10-20倍の性能差（問題依存）
- 小規模問題では差が小さいが、大規模・困難な問題ほど差が拡大
- 長年の開発投資による並列化・ヒューリスティクス・カット生成の洗練度

**OSSの意義:**
- 無料・オープン、ライセンス制約なし
- 小規模問題や予算制約下では十分実用的
- HiGHSなど近年性能向上中

**使い分け:**
- 学術研究: 商用学術ライセンス（無料・高性能）
- 商用・大規模: 商用ソルバー（20-30倍高速）
- 商用・小規模/予算制約: OSS（HiGHS/CBC）で十分

---

## 7. 総合分析

### ソルバー選定の判断基準

| 用途 | 推奨ソルバー | 理由 |
|-----|------------|------|
| 大規模MIP（学術） | Gurobi/CPLEX | 無料学術ライセンス、最高性能 |
| 大規模MIP（商用） | Gurobi/CPLEX/Xpress | 20-30倍高速、並列・分散処理成熟 |
| 錐最適化（SOCP/SDP） | Mosek | 内点法最強、大規模LPで優位 |
| 巨大LP（GPU活用） | Gurobi（Linux）/Xpress（Win/Linux） | GPU最大50倍高速 |
| 非線形・制約プログラミング | Xpress | MINLP 5.3倍高速、CP対応 |
| 小規模問題（商用） | HiGHS/CBC（OSS） | 性能差小、無料 |
| 金融工学 | Mosek | 錐最適化、ポートフォリオ最適化実績 |

### 技術トレンド（2025-2026）

1. **GPU対応の拡大**: Gurobi（Linux限定）、Xpress（Win/Linux）がPDHG実装。NVIDIA cuOptオープンソース化で今後拡大予想
2. **ベンチマーク撤退**: Gurobi/CPLEX/Xpress/MindOptが公式ベンチマーク離脱。第三者評価が困難に
3. **並列化の深化**: CPLEX動的タスク分解（2025研究）など、並列戦略が高度化
4. **錐の拡張**: Mosekが指数錐・累乗錐対応（v9）。問題表現力向上

### 商用価格の透明性

- 全ソルバーとも商用価格非公開（カスタム見積）
- 学術ライセンスは全て無料（Gurobi: 制限なし、CPLEX: 1年更新、Mosek/Xpress: 詳細要確認）
- 価格交渉力のない中小企業には不透明

### OSSの戦略的意義

- 商用依存リスク回避（価格変更、ベンダーロックイン）
- 小規模問題では実用的性能
- HiGHS等の進化で性能差縮小傾向

---

## 8. 情報源

### Gurobi
- [AMPL - Gurobi Solver](https://ampl.com/products/solvers/linear-solvers/gurobi/)
- [Gurobi - Optimizer](https://www.gurobi.com/solutions/gurobi-optimizer/)
- [Gurobi - Benchmarks PDF](https://www.gurobi.com/pdfs/benchmarks.pdf)
- [Gurobi - Academic Program](https://www.gurobi.com/academia/academic-program-and-licenses/)
- [Gurobi Help - Cutting Planes](https://support.gurobi.com/hc/en-us/community/posts/360060841412-Cutting-planes)
- [Gurobi - Parallel Optimization](https://www.gurobi.com/features/gurobi-optimizer-delivers-parallel-optimization/)
- [Gurobi - GPU Support](https://support.gurobi.com/hc/en-us/articles/360012237852-Does-Gurobi-support-GPUs)
- [Gurobi - GPU Solver Introduction](https://www.gurobi.com/resources/introducing-gurobis-first-gpu-accelerated-solver/)
- [arXiv - GPU-Accelerated Unit Commitment](https://arxiv.org/html/2512.06715v1)

### CPLEX
- [IBM - CPLEX Optimizer](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer)
- [IBM - CPLEX Pricing](https://www.ibm.com/products/ilog-cplex-optimization-studio/pricing)
- [IBM Docs - CPLEX Python API](https://www.ibm.com/docs/el/icos/20.1.0?topic=cplex-python-reference-manual)
- [AMPL - CPLEX Solver](https://ampl.com/products/solvers/linear-solvers/cplex/)
- [CPLEX - Parallel MIP Optimizer (TU Chemnitz)](https://www.tu-chemnitz.de/mathematik/discrete/manuals/cplex/doc/userman/html/moreUsing29.html)
- [CPLEX - Parallel MIP Optimizer (UPC)](http://www-eio.upc.edu/lceio/manuals/cplex75/doc/usermanccpp/html/moreUsing23.html)
- [LLNL - Distributed CPLEX](https://www.osti.gov/servlets/purl/1165747)
- [CP 2025 - Parallel MIP with Dynamic Task Decomposition](https://drops.dagstuhl.de/storage/00lipics/lipics-vol340-cp2025/LIPIcs.CP.2025.26/LIPIcs.CP.2025.26.pdf)

### Mosek
- [MOSEK - Wikipedia](https://en.wikipedia.org/wiki/MOSEK)
- [MOSEK - Official Site](https://www.mosek.com/)
- [MOSEK - Documentation](https://www.mosek.com/documentation/)
- [MOSEK - Interfaces](https://docs.mosek.com/latest/intro/interfaces.html)
- [MOSEK - Python Fusion API](https://docs.mosek.com/latest/pythonfusion/index.html)
- [MOSEK - C++ Fusion API](https://docs.mosek.com/latest/cxxfusion/index.html)
- [Springer - MOSEK Interior Point Optimizer](https://link.springer.com/chapter/10.1007/978-1-4757-3216-0_8)
- [GAMS - MOSEK](https://www.gams.com/latest/docs/S_MOSEK.html)
- [arXiv - MINLP Solvers Study](https://arxiv.org/pdf/2303.04216)

### FICO Xpress
- [FICO - Xpress Optimization](https://www.fico.com/en/products/fico-xpress-optimization)
- [FICO - Xpress Solver](https://www.fico.com/en/latest-thinking/solution-sheet/fico-xpress-solver)
- [FICO - Xpress Overview](https://www.fico.com/fico-xpress-optimization/docs/latest/overview.html)
- [FICO - Xpress MIP Performance](https://www.fico.com/blogs/blogs/experience-faster-mixed-integer-programming-optimization-with-xpress)
- [FICO - Xpress Python Reference](https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/python/HTML/GUID-616C323F-05D8-3460-B0D7-80F77DA7D046.html)
- [FICO - Xpress Wikipedia](https://en.wikipedia.org/wiki/FICO_Xpress)
- [PyPI - xpress](https://pypi.org/project/xpress/)

### ベンチマーク・比較
- [Mittelmann Benchmark - Decision Tree](https://plato.asu.edu/bench.html)
- [Mittelmann Plots - Interactive Visualization](https://mattmilten.github.io/mittelmann-plots/)
- [HiGHS Discussion #1683 - OSS vs Commercial](https://github.com/ERGO-Code/HiGHS/discussions/1683)
- [Analysis - Commercial vs Free OSS Solvers](http://www.tdp.cat/issues11/tdp.a114a12.pdf)
- [Gurobi - Open Source vs Gurobi](https://www.gurobi.com/resources/open-source-solvers-vs-gurobi-key-considerations/)

### 学術ライセンス
- [AMPL - Academia](https://ampl.com/academia/supported-discounted-licenses-for-academia/)
- [Gurobi - Academic Licenses](https://www.gurobi.com/academia/academic-program-and-licenses/)
- [GAMS - Academic Programs by Solver](https://support.gams.com/solver:academic_programs_by_solver_partners)

### その他
- [Cutting-plane method - Wikipedia](https://en.wikipedia.org/wiki/Cutting-plane_method)
- [Medium - Gomory Cutting Plane](https://medium.com/@minkyunglee_5476/integer-programming-the-cutting-plane-algorithm-26bbabf04815)
- [ZIB - Presolve and Cutting Planes](https://co-at-work.zib.de/berlin2009/downloads/2009-09-25/2009-09-25-1400-BB-MIP-4.pdf)

---

**調査完了日**: 2026-02-12
**調査担当**: ashigaru2 (Sonnet 4.5)
