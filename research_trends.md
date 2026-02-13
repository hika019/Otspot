# 数理最適化ソルバー最新研究動向調査（2024-2026年）

調査実施日: 2026-02-12
調査担当: 足軽三番

## 概要

本レポートは2024年から2026年にかけての数理最適化ソルバーの最新研究動向を調査したものである。主要な学術ジャーナル（INFORMS、Mathematical Programming等）、商用・OSSソルバーのリリースノート、学術会議の成果を網羅的に調査した。

調査対象領域:
1. ソルバー技術の最新進歩（プリソルバ、カット生成、分枝戦略、新アルゴリズム）
2. 機械学習のソルバーへの活用
3. 並列化・分散処理・GPU加速の動向
4. 新興OSSプロジェクトと主要ソルバーのアップデート

---

## 1. ソルバー技術の最新進歩

### 事実

#### プリソルバの改良

- プリソルバは制約の削減と定式化の強化を目的とした問題縮約手法の集合であり、分枝限定法の実行前に適用される ([Gurobi MIP Primer](https://www.gurobi.com/resources/mixed-integer-programming-mip-a-primer-on-the-basics/))
- BARON（非凸最適化ソルバー）では、プリソルバ、凸性識別、緩和・分離・削減戦略、メモリ管理、限界計算まで幅広い領域で改善が進んでいる ([FICO Xpress Global](https://optimization-online.org/2025/07/solving-minlps-to-global-optimality-with-fico-xpress-global/))

#### カット生成の革新

- カット平面法は整数計画における計算性能向上の最も重要な貢献要素として広く認められている ([Gurobi MIP Primer](https://www.gurobi.com/resources/mixed-integer-programming-mip-a-primer-on-the-basics/))
- データ駆動アプローチが登場：パラメータ化されたカット平面群から最適なカットを選択し、分枝限定木のサイズ削減を目指す研究が進展 ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))
- Cut Ranking: 産業用ソルバーに実装され、製品計画問題のオンラインA/Bテストで平均12.42%の高速化を達成（精度劣化なし） ([Learning to Select Cuts](https://www.sciencedirect.com/science/article/abs/pii/S0031320321005331))

#### PDHG（Primal-Dual Hybrid Gradient）アルゴリズムの普及

- 2024年にCardinal Optimization社のCOPTがPDHG実装を先駆的に導入 ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))
- 2025年には以下のソルバーがPDHGアルゴリズムを導入・更新:
  - COPT 7.2
  - Gurobi 13
  - HiGHS 1.10
  - Knitro 15
  - Xpress 46
- GPU対応PDHGもHiGHS、Gurobi、Xpressで利用可能に ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))

#### 分枝限定法の改良

- Gurobiは新バージョンでヒューリスティックを改良し、実行可能解（incumbent）が不要になった。ユーザーがMipStartオプションで提供した「良質だが実行不可能な初期解」からもヒューリスティックが動作可能に ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))

### 分析

2024-2026年の技術進歩で特筆すべきは**PDHGアルゴリズムの急速な普及**である。内点法の一種であるPDHGは、大規模線形計画・凸最適化に有効とされてきたが、商用・OSSの双方で実装が進んだことで、実務レベルでの選択肢が大きく広がった。特にGPU対応版の登場により、並列計算環境での性能向上が期待できる。

また、カット生成におけるデータ駆動アプローチ（Cut Ranking等）の産業実装が進んでいることは、機械学習とソルバー技術の融合が「研究段階」から「実用段階」に移行しつつある証左である。12%の高速化という数値は、大規模計画問題において非常に大きなインパクトを持つ。

---

## 2. 機械学習のソルバーへの活用

### 事実

#### 全体動向

- 近年、分枝限定アルゴリズムの主要タスク（primal heuristics、分枝、カット平面、ノード選択、ソルバー設定）すべてにおいて、機械学習を活用する研究開発が爆発的に増加 ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))

#### カット選択へのML適用

- **Cut Ranking**: 複数インスタンス学習（Multiple Instance Learning）を用いたデータ駆動型カット選択手法。インスタンス固有のカット特徴量でスコアリング関数を訓練 ([Learning to Select Cuts - Pattern Recognition](https://www.sciencedirect.com/science/article/abs/pii/S0031320321005331))
- 産業用ソルバーに実装済み：大規模MIP問題で平均12.42%の高速化を達成 ([Learning to Select Cuts](https://arxiv.org/pdf/2105.13645))

#### 分枝変数選択・ノード選択へのML適用

- 分枝変数選択、ノード選択、カット平面生成をカバーする研究が進行中 ([ML Augmented B&B](https://link.springer.com/article/10.1007/s10107-024-02130-y))
- 2024年の研究では分枝変数選択とソルバー強化に関する成果が多数報告 ([Survey on MIP via ML](https://www.researchgate.net/publication/365434388_A_Survey_for_Solving_Mixed_Integer_Programming_via_Machine_Learning))

#### 理論的基盤の確立（2025年）

- 分枝限定法の効率は、ノード選択・カット選択・分枝変数選択のヒューリスティック方策に大きく依存 ([Generalization Guarantees - ICLR 2025](https://openreview.net/pdf?id=6yENDA7J4G))
- 従来のソルバーは手動調整されたパラメータのヒューリスティックを使用してきたが、近年は機械学習でデータから直接方策を学習するアプローチが増加 ([Generalization Guarantees - arXiv](https://arxiv.org/abs/2505.11636))
- 2025年の論文では、スコアリング関数が区分多項式構造を持つ場合の分枝限定方策学習に対する厳密なサンプル複雑度境界を確立 ([Generalization Guarantees - arXiv HTML](https://arxiv.org/html/2505.11636))

#### 汎化性能の課題

- 学習モデルの異なるタスクや問題クラスへの汎化性能が依然として重要な課題 ([Survey on MIP via ML](https://www.researchgate.net/publication/365434388_A_Survey_for_Solving_Mixed_Integer_Programming_via_Machine_Learning))

### 分析

機械学習とソルバーの統合は**研究段階から実用段階への転換期**を迎えている。特にCut Rankingが産業用ソルバーで実装され、A/Bテストで実証された事実は重要である。これは「学術的に面白い」だけでなく「実務で使える」段階に到達したことを示す。

一方で、2025年のICLR論文に見られる理論研究（サンプル複雑度の解析）も進展しており、「なぜMLがソルバーで機能するのか」の理論的理解が深まっている。これは今後の研究・開発の方向性を示す重要な基盤となる。

課題は**汎化性能**である。特定の問題インスタンスで訓練されたモデルが、異なる問題クラスでどの程度性能を発揮するかは未解決の重要テーマである。実務での展開には、この汎化性能の向上が鍵となる。

---

## 3. 並列化・分散処理・GPU加速の動向

### 事実

#### NVIDIA cuOptの登場と進化

- **cuOpt概要**: NVIDIA cuOptは線形計画（LP）、混合整数計画（MIP）、車両経路問題（VRP）をGPU上で解くソルバー。多様なベンチマークで競争力のある性能を達成 ([NVIDIA cuOpt Blog](https://developer.nvidia.com/blog/learn-how-nvidia-cuopt-accelerates-mixed-integer-optimization-using-primal-heuristics))
- **オープンソース化**: NVIDIAは独自のPDHGとGPU加速MIPアルゴリズムをオープンソースソルバーcuOptとして公開 ([GAMS cuOpt Blog](https://www.gams.com/blog/2025/09/gpu-accelerated-optimization-with-gams-and-nvidia-cuopt/))
- **2025 COIN-OR Cup受賞**: cuOptチームは高品質な産業グレードの最適化コードをオープンソース化した功績により、2025 COIN-OR Cupを受賞 ([COIN-OR Cup Award](https://www.coin-or.org/2025/10/26/2025-coin-or-cup-award-nvidia-cuopt/))

#### cuOptのMIP解法アプローチ

- **ハイブリッド手法**: MIP問題に対してGPUで原始ヒューリスティック（primal heuristics）を実行し、CPUで双対境界（dual bound）を改善するハイブリッド手法を採用 ([cuOpt Technical Blog](https://developer.nvidia.com/blog/learn-how-nvidia-cuopt-accelerates-mixed-integer-optimization-using-primal-heuristics))
- **高度なヒューリスティック**: GPU最適化されたFeasibility Pump（FP）を一次PDLPソルバーおよびドメイン伝播と組み合わせ、大規模MIPインスタンスで大幅な高速化と解の品質向上を達成。MIPLIBベンチマークの未解決問題も解決 ([cuOpt Technical Blog](https://developer.nvidia.com/blog/learn-how-nvidia-cuopt-accelerates-mixed-integer-optimization-using-primal-heuristics))

#### 性能評価

- **HiGHSとcuOptの統合**: HiGHS単独でMIPLIBベンチマークを5分制限で実行した場合、最適性からのギャップは28%。H100 GPU上のcuOptと統合すると、ギャップが21%に改善 ([HiGHS and cuOpt Blog](https://blogs.ed.ac.uk/mathematics/2025/03/18/highs-and-nvidia-cuopt-driving-open-source-innovation-in-optimization/))
- **MILP高速化**: cuOptのヒューリスティック整数実行可能解により、Roseソルバーは不要な計算を枝刈り可能となり、MILPを最大8.6倍高速化（高精度と証明可能な最適性を維持） ([SimpleRose cuOpt Integration](https://simplerose.com/blog/how-simplerose-and-nvidia-cuopt-solve-lp-and-milp-problems-faster/))

#### LP高速化

- 大規模線形計画問題に対するcuOptの性能向上についての技術解説も公開 ([Accelerate Large LP](https://developer.nvidia.com/blog/accelerate-large-linear-programming-problems-with-nvidia-cuopt/))

### 分析

GPU加速ソルバーの登場は**計算パラダイムの転換点**である。特にcuOptのオープンソース化は、これまで一部の研究機関や大企業に限られていたGPU最適化技術を、広く実務で利用可能にする画期的な動きである。

HiGHSとの統合で最適性ギャップが28%から21%に改善した事実は、既存OSSソルバーとの相乗効果を示している。GPUソルバーは「既存技術の置き換え」ではなく「既存ソルバーの補完」として機能することで、実務での採用障壁を下げている。

また、MILPの8.6倍高速化という数値は、特に時間制約の厳しい実務問題（輸送計画、生産スケジューリング等）で大きなインパクトを持つ。ただし、GPU環境が必須であるため、コスト面での評価も必要である。

---

## 4. 新興OSSプロジェクトと主要ソルバーのアップデート

### 事実

#### HiGHSの大型アップデート

- **HiGHS 1.12**: 新しい内点法ソルバーHiPOを導入。既存のIPXソルバーを補完する形で、マルチスレッド活用と予測可能な実行時間を実現。一部問題ではメモリ使用量増加と精度低下のトレードオフあり ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))
- **HiGHS 1.10**: PDHGアルゴリズムの導入・更新 ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))
- **cuOptとの統合**: GPU加速により性能向上（前述） ([HiGHS and cuOpt Blog](https://blogs.ed.ac.uk/mathematics/2025/03/18/highs-and-nvidia-cuopt-driving-open-source-innovation-in-optimization/))

#### SCIP/SoPlex Optimization Suite 10

- **GAMS 53でリリース**: SoPlex 8とSCIP 10がGAMSユーザーに利用可能に ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))
- **Benders分解フレームワーク**: GAMS 53でBenders分解フレームワークがGAMSユーザーに提供開始 ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))
- **NLPソルバー統合**: SCIPでCONOPTをNLPソルバーとして使用可能に ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))
- **IIS計算**: 実行不可能問題に対する既約実行不可能部分系（Irreducible Infeasible Subsystem）の計算機能を提供 ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))

#### Gurobiの機能強化

- **Gurobi 13**: PDHGアルゴリズム導入、GPU対応PDHG実装 ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))
- **ヒューリスティック改良**: MipStartオプションで提供された良質だが実行不可能な解からもヒューリスティックが機能するように改良（前述） ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))

#### その他の商用ソルバー

- **COPT 7.2**: PDHGアルゴリズム更新 ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))
- **Knitro 15**: PDHGアルゴリズム導入 ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))
- **Xpress 46**: PDHGアルゴリズム導入、GPU対応PDHG実装 ([GAMS Blog 2025](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/))

#### NVIDIA cuOptのオープンソース化（再掲）

- 前述の通り、cuOptのオープンソース化は2025年の最重要イベントの一つ ([COIN-OR Cup Award](https://www.coin-or.org/2025/10/26/2025-coin-or-cup-award-nvidia-cuopt/))

### 分析

2024-2026年のOSS動向で最も注目すべきは**商用・OSSの境界線の曖昧化**である。従来、GPU加速や最先端アルゴリズムは商用ソルバーの専売特許であったが、cuOptのオープンソース化、HiGHSの急速な機能拡充により、OSSでも実務レベルの性能が達成可能になりつつある。

SCIP 10のBenders分解フレームワークやIIS計算機能の追加は、実務での問題診断能力を大きく向上させる。特に「なぜ実行不可能なのか」を特定するIIS機能は、モデル構築段階での試行錯誤を劇的に効率化する。

Gurobiのヒューリスティック改良（MipStartの柔軟化）は、実務での「良い初期解を持っているが厳密には実行不可能」というケースへの対応力を示す。これは産業界のフィードバックが商用ソルバーの開発に反映された好例である。

全体として、商用・OSS双方で「実務での使いやすさ」を重視した機能拡充が進んでいる。

---

## 5. 主要ジャーナル・学会動向

### 事実

#### Mathematical Programming（Springer）

- **概要**: Mathematical Optimization Societyの公式ジャーナル。線形・非線形・整数・錐・確率・組合せ最適化を含む数理最適化のあらゆる側面を扱う ([Mathematical Programming Journal](https://link.springer.com/journal/10107))
- **最近の出版**: 2025年5月号まで発行。Global Optimization特集号、IPCO 2023特集号を含む ([Mathematical Programming Volumes](https://link.springer.com/journal/10107/volumes-and-issues))
- **重要論文（2024年後半）**: 混合整数線形計画（MILP）の厳密解法の最近の進歩に関するサーベイ論文が発表。現代の分枝カット法、分枝カットプライス法の設計、Dantzig-Wolfe分解やBenders分解の主要な発展を解説。現代のソルバーは10年前には到達不可能だった問題を数秒で解けるようになったと報告 ([Last Fifty Years of Integer LP](https://www.sciencedirect.com/science/article/pii/S0377221724008877))

#### INFORMS Journals

- **Mathematics of Operations Research (MOOR)**: 2025年第50巻第2号（5月）、第1号（2月）発行。連続・離散・確率最適化、数理計画、動的計画の数学的・計算的基盤を扱う ([MOOR Current Issue](https://pubsonline.informs.org/toc/moor/current))
- **Operations Research**: 2025年年次報告書を発行。編集長による現状報告と四半期統計を含む ([Operations Research Journal](https://pubsonline.informs.org/journal/opre))
- **INFORMS Journal on Computing**: 2025年11-12月号発行。最適化と計算の境界を拡張する高品質論文を掲載 ([IJOC Journal](https://pubsonline.informs.org/journal/ijoc))

#### IPCO（Integer Programming and Combinatorial Optimization）

- **IPCO 2025**: 第26回会議が2025年6月にボルチモア（メリーランド州）で開催。109件の投稿から33論文を採択。整数計画と組合せ最適化の理論・計算・応用の最近の発展に焦点 ([IPCO 2025 Proceedings](https://link.springer.com/book/10.1007/978-3-031-93112-3))

### 分析

学術界では「10年前は解けなかった問題が今は数秒で解ける」という劇的な性能向上が報告されている。これは前述のPDHG、ML統合、GPU加速といった技術進歩の総合的な成果である。

IPCOの採択率（33/109 ≒ 30%）は高水準を維持しており、整数計画分野の研究活動が活発であることを示す。また、INFORMS各誌が2025年も継続的に発行されていることから、実務と理論の双方で活発な研究交流が続いている。

---

## 6. 総合考察

### 技術トレンドの統合的理解

2024-2026年の数理最適化ソルバー分野は、以下の3つの大きな潮流が同時進行している:

1. **アルゴリズム進化**: PDHGの普及、カット生成の高度化、分枝戦略の洗練
2. **計算資源の革新**: GPU加速（cuOpt等）、並列・分散処理の実用化
3. **知能化**: 機械学習によるヒューリスティック方策の自動最適化

これらは独立ではなく、相互に強化し合っている。例えば、GPU加速により大規模な機械学習モデルの訓練が可能になり、その学習済みモデルがカット選択を最適化し、結果として分枝限定木の探索が効率化される。

### 実務への影響

- **計算時間の劇的短縮**: 10年前には数時間かかった問題が数秒〜数分で解ける
- **問題規模の拡大**: より大規模・複雑な実問題への適用が可能に
- **導入障壁の低下**: OSSソルバー（HiGHS、SCIP、cuOpt）の高性能化により、商用ソルバーなしでも実務適用可能に

### 今後の展望

- **ML統合の深化**: 汎化性能の向上により、業界・問題クラス横断での適用が進む
- **GPU環境の普及**: クラウドGPUの低価格化により、中小企業でもGPUソルバーが利用可能に
- **ハイブリッド手法の標準化**: CPU/GPU、商用/OSS、古典アルゴリズム/MLを組み合わせた最適な構成が模索される

---

## 情報源一覧

### 主要ソルバー・技術解説

- [The Year 2025 for GAMS Solvers](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/)
- [MILP Solvers: Methods and Advances](https://www.emergentmind.com/topics/mixed-integer-linear-programming-milp-solvers)
- [Mixed-Integer Programming (MIP/MILP) – A Primer on the Basics - Gurobi](https://www.gurobi.com/resources/mixed-integer-programming-mip-a-primer-on-the-basics/)
- [Solving MINLPs to global optimality with FICO Xpress Global](https://optimization-online.org/2025/07/solving-minlps-to-global-optimality-with-fico-xpress-global/)

### 機械学習統合

- [Machine learning augmented branch and bound for mixed integer linear programming | Mathematical Programming](https://link.springer.com/article/10.1007/s10107-024-02130-y)
- [Learning to Select Cuts for Efficient Mixed-Integer Programming (arXiv)](https://arxiv.org/pdf/2105.13645)
- [Learning to select cuts for efficient mixed-integer programming - Pattern Recognition](https://www.sciencedirect.com/science/article/abs/pii/S0031320321005331)
- [Generalization Guarantees for Learning Branch-and-Cut Policies in Integer Programming - ICLR 2025](https://openreview.net/pdf?id=6yENDA7J4G)
- [Generalization Guarantees for Learning Branch-and-Cut Policies in Integer Programming - arXiv](https://arxiv.org/abs/2505.11636)
- [A Survey for Solving Mixed Integer Programming via Machine Learning](https://www.researchgate.net/publication/365434388_A_Survey_for_Solving_Mixed_Integer_Programming_via_Machine_Learning)

### GPU加速・並列処理

- [GPU-Accelerated Optimization with GAMS and NVIDIA cuOpt](https://www.gams.com/blog/2025/09/gpu-accelerated-optimization-with-gams-and-nvidia-cuopt/)
- [HiGHS and NVIDIA cuOpt: Driving open-source innovation in optimization](https://blogs.ed.ac.uk/mathematics/2025/03/18/highs-and-nvidia-cuopt-driving-open-source-innovation-in-optimization/)
- [Learn How NVIDIA cuOpt Accelerates Mixed Integer Optimization using Primal Heuristics](https://developer.nvidia.com/blog/learn-how-nvidia-cuopt-accelerates-mixed-integer-optimization-using-primal-heuristics)
- [Accelerate Large Linear Programming Problems with NVIDIA cuOpt](https://developer.nvidia.com/blog/accelerate-large-linear-programming-problems-with-nvidia-cuopt/)
- [2025 COIN-OR Cup Award: NVIDIA cuOpt](https://www.coin-or.org/2025/10/26/2025-coin-or-cup-award-nvidia-cuopt/)
- [Accelerating Optimization: How SimpleRose and NVIDIA cuOpt Solve LP and MILP Problems Faster](https://simplerose.com/blog/how-simplerose-and-nvidia-cuopt-solve-lp-and-milp-problems-faster/)

### 学術ジャーナル・学会

- [Mathematical Programming Journal | Springer Nature](https://link.springer.com/journal/10107)
- [Last fifty years of integer linear programming: A focus on recent practical advances](https://www.sciencedirect.com/science/article/pii/S0377221724008877)
- [IPCO 2025 Proceedings](https://link.springer.com/book/10.1007/978-3-031-93112-3)
- [Mathematics of Operations Research - Current Issue](https://pubsonline.informs.org/toc/moor/current)
- [Operations Research | INFORMS](https://pubsonline.informs.org/journal/opre)
- [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc)

---

**調査完了日**: 2026-02-12
**報告者**: 足軽三番（Ashigaru 3）
