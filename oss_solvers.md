# OSS Solver Survey

## 事実（Facts with Sources）

### 1. HiGHS

#### 対応問題種別
- Linear Programming (LP), Mixed-Integer Programming (MIP), Quadratic Programming (QP)
- 大規模なsparse問題に対応
- Source: [HiGHS Official Site](https://highs.dev/), [HiGHS Wikipedia](https://en.wikipedia.org/wiki/HiGHS_optimization_solver)

#### ライセンス
- MIT License（オープンソース、ライセンス料不要）
- Source: [HiGHS Official Site](https://highs.dev/)

#### 性能ベンチマーク
- Mittelmannベンチマークにおいて、世界中のオープンソース線形最適化ソフトウェアの中で最高の性能
- Gurobiとの比較では約1桁の性能差（Gurobiの方が高速）
- Source: [HiGHS Official Site](https://highs.dev/), [HiGHS GitHub Discussion #1683](https://github.com/ERGO-Code/HiGHS/discussions/1683)
- Mittelmann benchmarks: [plato.asu.edu/bench.html](https://plato.asu.edu/bench.html), [Interactive Plots](https://mattmilten.github.io/mittelmann-plots/)

#### 並列化対応
- シリアル・パラレル両対応
- dual simplexソルバーでマルチスレッディング対応
- hyper-sparsityの活用
- GPU対応PDHG実装が追加
- Source: [HiGHS Documentation](https://ergo-code.github.io/HiGHS/dev/), [HiGHS Parallel Documentation](https://ergo-code.github.io/HiGHS/dev/parallel/)

#### 開発活発さ
- 2026年6月1日にエジンバラでHiGHS Workshop 3開催予定
- GitHub: [ERGO-Code/HiGHS](https://github.com/ERGO-Code/HiGHS)
- Source: [HiGHS Official Site](https://highs.dev/)

#### アーキテクチャ
- 実装言語: C++11
- 第三者依存なし
- 主要アルゴリズム:
  - Primal and dual revised simplex (Qi Huangfu, Julian Hall開発)
  - Interior point solver for LP (Lukas Schork開発)
  - Active set QP solver (Michael Feldmeier開発)
  - MIP solver (Leona Gottwald開発)
- 中心となるHighsクラスがエントリーポイント
- Source: [HiGHS GitHub](https://github.com/ERGO-Code/HiGHS), [HiGHS Architecture (DeepWiki)](https://deepwiki.com/ERGO-Code/HiGHS/1-overview-of-highs)

#### Python/その他バインディング
- highspy: pybind11を使用したPythonバインディング
- PyPiで利用可能
- numpy依存（自動インストール）
- C, C#, FORTRAN, Julia, Pythonインターフェース提供
- 注意: 配列値へのエントリごとのアクセスは遅い（リスト化推奨）
- Source: [highspy PyPI](https://pypi.org/project/highspy/1.7.1.dev1/), [HiGHS Python Documentation](https://ergo-code.github.io/HiGHS/dev/interfaces/python/)

---

### 2. SCIP

#### 対応問題種別
- Mixed-Integer Linear Programming (MILP)
- Mixed-Integer Nonlinear Programming (MINLP) - convex/nonconvex両対応
- Constraint Programming (CP)
- Branch-cut-and-priceフレームワーク
- Source: [SCIP Official Site](https://www.scipopt.org/), [SCIP Suite 10.0 Paper](https://arxiv.org/html/2511.18580v1)

#### ライセンス
- Apache 2.0 または GNU Lesser General Public License (LGPL)
- SCIP 10.0, SoPlex 8.0, PaPILO 3.0, GCG 4.0がこのライセンス
- Source: [SCIP Suite 10.0 Paper](https://arxiv.org/html/2511.18580v1)

#### 性能ベンチマーク
- Mittelmannベンチマーク: HiGHSとGurobiの中間（CBCより高速）
- CBCと比較して約1桁未満の性能差
- Source: [HiGHS GitHub Discussion #1683](https://github.com/ERGO-Code/HiGHS/discussions/1683)

#### 並列化対応
- UGフレームワークによる並列化（共有メモリ・分散メモリ両対応）
- FiberSCIP: 共有メモリ環境での決定論的並列化
- GCG: 価格問題の並列化対応
- Source: [SCIP Suite 10.0 Paper](https://arxiv.org/html/2511.18580v1), [FiberSCIP Paper](https://pubsonline.informs.org/doi/10.1287/ijoc.2017.0762)

#### 開発活発さ
- 最新リリース: SCIP 10.0.1 (2026年2月3日)
- SCIP Optimization Suite 10.0 (2025年11月リリース)
- 多数のコントリビュータ（Christopher Hojny, Mathieu Besançonなど）
- 関連プロジェクトも活発: SCIPpp (2025年12月18日更新), russcip (2025年12月16日更新)
- Source: [SCIP GitHub Releases](https://github.com/scipopt/scip/releases), [scipopt GitHub](https://github.com/scipopt)

#### アーキテクチャ
- 実装言語: C++
- Constraint Integer Programmingフレームワーク
- プラグインアーキテクチャ
- Source: [SCIP Official Site](https://www.scipopt.org/)

#### Python/その他バインディング
- PySCIPOpt: Python→SCIP Optimization Suiteインターフェース
- Python内で新たなSCIPプラグインを完全記述可能
- SCIP C APIの拡張カバレッジを優先
- 各SCIPリリース時に最低でもPySCIPOptをリリース
- conda-forge経由で約645.1K ダウンロード（v5.6.0時点）
- v8.0.3以降Apache 2.0ライセンス
- 2026年1月時点でアクティブなコミット活動
- Source: [PySCIPOpt GitHub](https://github.com/scipopt/PySCIPOpt), [PySCIPOpt PyPI](https://pypi.org/project/PySCIPOpt/), [PySCIPOpt conda-forge](https://anaconda.org/conda-forge/pyscipopt)

---

### 3. COIN-OR CBC

#### 対応問題種別
- Mixed-Integer Linear Programming (MIP)
- Large-scale Linear Programming (LP)
- Source: [COIN-OR CBC GitHub](https://github.com/coin-or/Cbc)

#### ライセンス
- Eclipse Public License 2.0
- オープンソース、産業・学術利用両対応
- Source: [COIN-OR CBC License](https://github.com/coin-or/Cbc/blob/master/LICENSE), [COIN-OR CBC GitHub](https://github.com/coin-or/Cbc)

#### 性能ベンチマーク
- 長年のオープンソースMIPソルバーの人気選択肢だったが、現在HiGHSと比較して著しく性能劣位
- 2007年時点のベンチマーク: GAMS 22.5、COIN-ORウェブサイトで利用可能
- Source: [Benchmarks of GAMS solvers](https://www.coin-or.org/GAMSlinks/benchmarks/), [COIN-OR CBC User Guide](https://www.coin-or.org/Cbc/cbcuserguide.html)

#### 並列化対応
- 明示的な並列化情報なし（主にシリアル実装と推測）

#### 開発活発さ
- 最新コミット: 2026年1月6日
- 968コミット、134コントリビュータ、159 issues
- プロジェクトマネージャー: John Forrest, Ted Ralphs, Stefan Vigerske, Haroldo Gambini Santos, その他CBCチーム
- Source: [COIN-OR CBC GitHub](https://github.com/coin-or/Cbc), [COIN-OR CBC Releases](https://github.com/coin-or/Cbc/releases)

#### アーキテクチャ
- 実装言語: C++
- Branch-and-cutアルゴリズム
- 呼び出し可能ライブラリまたはスタンドアロン実行可能形式
- 多様なモデリングシステム、パッケージ経由で利用可能
- Source: [COIN-OR CBC GitHub](https://github.com/coin-or/Cbc)

#### Python/その他バインディング
- CyLP: CLP, CBC, CGLへのPythonインターフェース（LP/MIP解決用）
- Source: [CyLP GitHub](https://github.com/coin-or/CyLP)

---

### 4. Google OR-Tools

#### 対応問題種別
- Linear Programming (LP)
- Mixed-Integer Programming (MIP)
- Constraint Programming (CP)
- Vehicle Routing Problems (VRP)
- 関連する最適化問題全般
- Source: [OR-Tools Official Site](https://developers.google.com/optimization/), [OR-Tools Wikipedia](https://en.wikipedia.org/wiki/OR-Tools)

#### ライセンス
- Apache License 2.0
- Source: [OR-Tools GitHub](https://github.com/google/or-tools)

#### 性能ベンチマーク
- 複数ソルバーのラッパーとして機能（Gurobi, CPLEX, SCIP, GLPK, GLOP, CP-SAT等）
- CP-SAT: 受賞歴あり
- 具体的なベンチマーク数値は見当たらず（ラッパー性質のため）
- Source: [OR-Tools Official Site](https://developers.google.com/optimization/)

#### 並列化対応
- 使用するソルバーに依存
- 情報なし

#### 開発活発さ
- 2026年1月時点: Python 3.13サポート追加、GIL-less Python 3.14対応改善
- Google内で2010年から本番利用
- Source: [OR-Tools Official Site](https://developers.google.com/optimization/)

#### アーキテクチャ
- 実装言語: C++
- MPSolver, ModelBuilder: 商用・オープンソースソルバーへのラッパー
- 内蔵ソルバー: GLOP (LP), CP-SAT (CP)
- Source: [OR-Tools Official Site](https://developers.google.com/optimization/)

#### Python/その他バインディング
- Python, C#, Javaラッパー提供
- PyPI経由でインストール可能（ortools）
- Python 3.13, 3.14対応
- Source: [ortools PyPI](https://pypi.org/project/ortools/), [OR-Tools Python Getting Started](https://developers.google.com/optimization/introduction/python)

---

### 5. GLPK

#### 対応問題種別
- Linear Programming (LP)
- Mixed-Integer Programming (MIP)
- 大規模問題対応
- Source: [GLPK GNU Project](https://www.gnu.org/software/glpk/), [GLPK Wikipedia](https://en.wikipedia.org/wiki/GNU_Linear_Programming_Kit)

#### ライセンス
- GNU General Public License (GPL v3)
- Source: [GLPK.jl JuMP](https://jump.dev/JuMP.jl/stable/packages/GLPK/)

#### 性能ベンチマーク
- 具体的な2026年ベンチマーク情報なし
- 並列化研究での性能改善実績あり（後述）

#### 並列化対応
- 研究段階で並列化実績あり:
  - スレッド並列GLPK実装: 12コアIntel Xeonで9.6倍高速化（1e5 LPs、サイズ100）
  - Cache-aware + OpenMP: 12コアAMD Opteronで中央値21.9倍高速化
  - GPU実装: 50k LPs（サイズ100）で最大18.3倍、約400万LPs（サイズ5）で63倍高速化
- Source: [Mixed-Precision Parallel LP Solver (ResearchGate)](https://www.researchgate.net/publication/221306535_Mixed-Precision_Parallel_Linear_Programming_Solver), [Solving Batched LPs on GPU (arXiv)](https://arxiv.org/abs/1609.08114)

#### 開発活発さ
- 2026年の具体的開発情報なし
- 長期にわたる安定リリース実績

#### アーキテクチャ
- 実装言語: C
- 呼び出し可能ライブラリとして構成
- Source: [GLPK GNU Project](https://www.gnu.org/software/glpk/)

#### Python/その他バインディング
- 複数のPythonラッパー存在（PuLP経由、Sage経由等）
- Julia: GLPK.jl
- Source: [GLPK.jl JuMP](https://jump.dev/JuMP.jl/stable/packages/GLPK/), [GLPK Sage](https://doc.sagemath.org/html/en/reference/spkg/glpk.html)

---

### 6. Ipopt（追加の有力OSS）

#### 対応問題種別
- Nonlinear Programming (NLP)
- 大規模非線形最適化問題
- 非凸関数も対応可能（ただし2階連続微分可能であること）
- Source: [Ipopt Documentation](https://coin-or.github.io/Ipopt/), [Ipopt Wikipedia](https://en.wikipedia.org/wiki/IPOPT)

#### ライセンス
- Eclipse Public License (EPL)
- COIN-ORからオープンソース提供
- Source: [Ipopt Documentation](https://coin-or.github.io/Ipopt/)

#### 性能ベンチマーク
- 性能は選択する線形ソルバーに大きく依存
- 具体的なベンチマーク情報なし

#### 並列化対応
- 並列線形ソルバー経由で並列化対応:
  - MKL Pardiso
  - HSL MA86, HSL MA97
  - SPRAL (Sparse Parallel Robust Algorithms Library)
  - MUMPS (MUltifrontal Massively Parallel sparse direct Solver)
- 最適化アルゴリズム自体ではなく、内部線形ソルバーによる並列化
- Source: [Ipopt Documentation](https://coin-or.github.io/Ipopt/), [Exploring Benefits of Linear Solver Parallelism (arXiv)](https://arxiv.org/abs/1909.08104)

#### 開発活発さ
- 2026年時点の具体的情報なし
- COIN-ORプロジェクトとして継続メンテナンス

#### アーキテクチャ
- 実装言語: C++
- Primal-dual interior point method
- Filter methodベースのline search
- Sparse symmetric indefinite linear system solverに強く依存
- Source: [Ipopt Documentation](https://coin-or.github.io/Ipopt/), [GAMS IPOPT](https://www.gams.com/latest/docs/S_IPOPT.html)

#### Python/その他バインディング
- C, C++, Fortran, Java, R, Python, その他多数の言語から呼び出し可能
- Source: [Ipopt Documentation](https://coin-or.github.io/Ipopt/)

---

### 7. その他の有力OSSソルバー

#### Ceres Solver
- 大規模・複雑な最適化問題向けオープンソースC++ライブラリ
- 2010年からGoogle本番利用
- 非線形最適化に特化
- Source: [Ceres Solver Official Site](http://ceres-solver.org/)

#### GCG (Generic Column Generation)
- Mixed-Integer Programs (MIPs)用汎用分解ソルバー
- Dantzig-Wolfe reformationの自動実行
- Full branch-price-and-cutアルゴリズム
- Source: [Baeldung - Best Open-Source MIP Solver](https://www.baeldung.com/cs/best-open-source-mixed-integer-optimization-solver)

#### PuLP
- Pythonでの最適化モデリング用ライブラリ
- 複数のオープンソース・商用ソルバーへのインターフェース
- Source: [PuLP Documentation](https://coin-or.github.io/pulp/)

---

## 比較表

| ソルバー | 問題種別 | ライセンス | 並列化 | 主要言語 | Python品質 | 性能（相対） |
|---------|---------|-----------|--------|---------|-----------|-------------|
| **HiGHS** | LP/MIP/QP | MIT | マルチスレッド+GPU | C++11 | highspy（良好、注意点あり） | ⭐⭐⭐⭐⭐（OSS最速） |
| **SCIP** | MILP/MINLP/CP | Apache 2.0/LGPL | UG並列・FiberSCIP決定論的 | C++ | PySCIPOpt（高品質、活発開発） | ⭐⭐⭐⭐（商用に次ぐ） |
| **CBC** | MIP/LP | EPL 2.0 | シリアル主体 | C++ | CyLP | ⭐⭐（旧世代、HiGHSに劣後） |
| **OR-Tools** | LP/MIP/CP/VRP | Apache 2.0 | ソルバー依存 | C++ | ortools（高品質、充実） | ⭐⭐⭐⭐（ラッパー、CP-SAT受賞） |
| **GLPK** | LP/MIP | GPL v3 | 研究段階 | C | 複数ラッパー | ⭐⭐⭐（標準的） |
| **Ipopt** | NLP | EPL | 線形ソルバー経由 | C++ | 対応 | ⭐⭐⭐⭐（NLP特化） |

---

## 分析（Analysis / 足軽の意見）

### 性能評価の総括

1. **LP/MIPの王者はHiGHS**: Mittelmannベンチマークで明確。オープンソースの中では最速。商用（Gurobi等）との差は約1桁だが、無料で使える価値は大きい。

2. **MIP複雑問題ならSCIP**: MINLPや制約プログラミングにも対応。並列化も本格的。開発が活発で2026年2月に10.0.1リリース。

3. **CBCは過去の遺産**: 性能でHiGHSに大きく劣後。新規採用の理由は薄い。

4. **OR-Toolsは統合フレームワーク**: 複数ソルバーのラッパーとして柔軟性高い。VRP等の特殊問題にも対応。CP-SATの受賞歴は魅力。

5. **GLPKは教育・小規模向け**: GPL制約あり。性能は中程度だが、並列化研究での改善可能性は示されている。

6. **非線形ならIpopt**: LP/MIPソルバーとは別次元。非線形最適化の定番。

### ライセンス選定のポイント

- **MIT（HiGHS）**: 最も制約少ない。商用利用も自由。
- **Apache 2.0（SCIP, OR-Tools）**: MIT同等の自由度。
- **EPL（CBC, Ipopt）**: 弱いコピーレフト。実用上問題少ない。
- **GPL（GLPK）**: 派生物もGPL化必要。商用製品への組み込みは要注意。

### 並列化の現実

- **HiGHS, SCIP**: 本格並列化対応。HiGHSはGPUも。
- **Ipopt**: 線形ソルバー次第（Pardiso, MUMPS等）。
- **CBC, GLPK**: 並列化弱い。GLPKは研究段階で可能性あり。

### Python統合の品質

- **OR-Tools**: Google製、Python 3.13/3.14対応、成熟。
- **PySCIPOpt**: 活発開発、SCIPリリース毎に更新、プラグイン可能。
- **highspy**: 軽量、注意点（配列アクセス）あるが実用レベル。
- **CyLP（CBC）**: 基本的なインターフェース。

### 推奨選定基準

| 用途 | 推奨ソルバー | 理由 |
|-----|-------------|------|
| LP高速化最重視 | HiGHS | OSS最速、MIT、並列・GPU対応 |
| MINLP・CP必要 | SCIP | 幅広い問題種別、活発開発 |
| 統合環境・VRP | OR-Tools | 柔軟性、Google品質、VRP対応 |
| 非線形最適化 | Ipopt | NLP定番、並列線形ソルバー対応 |
| 教育・GPL OK | GLPK | シンプル、歴史ある安定性 |

### 2026年のトレンド

- **HiGHSの台頭**: OSSコミュニティでのデファクトスタンダード化が進行中。
- **SCIPの進化**: 10.0系でMINLP性能向上、並列化強化。
- **GPU対応の広がり**: HiGHSでPDHG実装。今後の発展期待。
- **Python統合の充実**: 全主要ソルバーで高品質Pythonバインディング提供。
