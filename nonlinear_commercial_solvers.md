# 非線形商用ソルバー調査レポート

調査対象: KNITRO, BARON, MOSEK, GAMS, Gurobi (非線形), CPLEX (非線形), NAG, SNOPT, MINOS, CONOPT, その他
調査日: 2026-02-12
調査担当: ashigaru2
プロジェクト: solver (Phase 1 市場調査)

---

## 1. 総合比較表

| ソルバー | 対応NL問題種別 | アルゴリズム | 強み | 学術ライセンス | 商用価格帯 |
|---------|--------------|------------|------|--------------|-----------|
| **KNITRO** | NLP/MINLP特化 | 内点法/SQP/AL | MINLP最速級、26%高速化 | 無料（教育用、年更新） | 要問合せ |
| **BARON** | グローバルMINLP | Branch-and-reduce | 決定論的大域最適保証 | 学術割引（CMU/UIUC/UGA無料） | 月額/年額/永久 |
| **MOSEK** | SOCP/SDP/錐最適化 | 内点法（同次） | 錐最適化最強、指数錐対応 | 学術プログラム有 | 要問合せ |
| **Gurobi** | QP/QCQP/双線形 | 内点法/B&B | 非凸QP大域最適化、v12大幅改善 | 無料（full-featured） | サブスク制 |
| **CPLEX** | QP/QCQP/SOCP | 内点法/active-set/B&B | 非凸QP対応、並列処理成熟 | 無料（1年更新） | サブスク制 |
| **XPRESS** | NLP/MINLP/QP | 内点法/PDHG/B&B | MINLP 5.3倍高速、GPU対応 | 無料（学術） | カスタム見積 |
| **NAG** | NLP（制約付き） | 内点法/active-set SQP | IPOPT統合、第一次法対応 | 要確認 | ライセンス料 |
| **SNOPT** | 大規模疎NLP | SQP（準ニュートン） | 高価な関数評価に強い | 要確認 | ライセンス料 |
| **MINOS** | 大規模疎NLP | GRG/Simplex | 自動スケーリング、warm-start | 要確認 | ライセンス料 |
| **CONOPT** | 一般NLP | GRG+SLP/SQP | 実行可能パス法、難非線形対応 | GAMS経由 | GAMS経由 |

**大域最適性の保証:**
- BARON: 決定論的大域最適保証（有限範囲条件下）
- Gurobi: 非凸QP/QCQP大域最適化（v9.0以降）
- その他: 局所最適解（大域最適性保証なし）

---

## 2. KNITRO (Artelys)

### 事実

#### 概要
Artelys Knitroは大規模非線形数理最適化問題を解く商用ソフトウェア。世界中の数百サイトで採用。
出典: [Artelys Knitro - Wikipedia](https://en.wikipedia.org/wiki/Artelys_Knitro), [Knitro - Artelys](https://www.artelys.com/solvers/knitro/)

#### 対応問題種別
- NLP（非線形計画）、MINLP（混合整数非線形計画）に特化
- 凸・非凸問題対応
出典: [Knitro - Artelys](https://www.artelys.com/solvers/knitro/)

#### アルゴリズム
最新技術を実装:
- **内点法 (Interior-point method)**: バリア法
- **active-set法**: 制約の活性集合を追跡
- **拡大ラグランジアン法 (Augmented Lagrangian)**: v15.0で新実装、退化問題に有効
出典: [Algorithms — Artelys Knitro 15.1 User's Manual](https://www.artelys.com/app/docs/knitro/2_userGuide/algorithms.html), [The Year 2025 for GAMS Solvers](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/)

**MINLP専用アルゴリズム**:
- **MISQP**: 高価な関数評価向け、緩和不能整数変数対応
- **NLPBB**: 並列分枝限定法（v13で導入、4スレッドで5倍高速化）
出典: [Mixed-integer nonlinear programming — Artelys Knitro 15.1 User's Manual](https://www.artelys.com/app/docs/knitro/2_userGuide/minlp.html), [Artelys Knitro 13 solves MINLP 5 times faster](https://www.artelys.com/news/artelys-knitro-13-solves-minlp-problems-5-times-faster/)

#### 性能
- **v14**: 凸MINLPで26%高速化（最新presolve、カット、並列ヒューリスティクス）
出典: [Artelys Knitro 14 new release](https://www.artelys.com/news/artelys-knitro-14-new-release-nonlinear-optimization-solver/)
- **v13.1**: MINLPさらに高速化
出典: [Artelys Knitro 13.1 solves MINLP faster](https://www.artelys.com/news/artelys-knitro-13-1-solves-minlp-problems-even-faster/)
- **ベンチマーク結果（2023）**: KNITRO (interior-point/D) は第2位の収束率、平均CPU時間がBARONより2桁高速
出典: [Comparative Analysis of Nonlinear Programming Solvers](https://www.mdpi.com/2504-446X/7/8/487)

#### 並列化
- MINLP並列分枝限定法（v13以降）
- 4スレッドで5倍高速化（困難なMINLPインスタンス）
出典: [Artelys Knitro 13](https://www.artelys.com/news/artelys-knitro-13-solves-minlp-problems-5-times-faster/)

#### ライセンス・価格
- **教育ライセンス**: 無料、full-featured、12ヶ月有効（年更新可）。学位授与機関の教員対象
出典: [knitro program - EN - Artelys](https://www.artelys.com/solvers/knitro/programs/), [Knitro Solver - AMPL](https://ampl.com/products/solvers/nonlinear-solvers/knitro/)
- **試用版**: 1ヶ月無料、6ヶ月（300制約制限）
出典: [Knitro Solver - AMPL](https://ampl.com/products/solvers/nonlinear-solvers/knitro/)
- **商用**: 要問合せ
出典: [Artelys Knitro Solver Engine License](https://www.solver.com/knitro-solver-engine-software)

#### API・統合
対応モデリング言語: AIMMS, AMPL, GAMS, MATLAB
出典: [Knitro — AIMMS Documentation](https://documentation.aimms.com/platform/solvers/knitro.html)

### 分析

**強み:**
- NLP/MINLP専門特化、最速級の性能実績
- 3つのアルゴリズム（内点法/active-set/AL）で問題特性に柔軟対応
- v13-v14で大幅性能改善（5倍→26%追加）継続中
- 教育ライセンスが制限なし無料（年更新）

**弱み:**
- 大域最適化非保証（局所最適解のみ）
- 商用価格非公開
- GPU対応の記載なし

**適用場面:**
- 大規模NLP/MINLP問題（凸・非凸）
- 関数評価が高価な問題（MISQP）
- 学術研究（無料教育ライセンス活用）

---

## 3. BARON (The Optimization Firm)

### 事実

#### 概要
BARON (Branch-And-Reduce Optimization Navigator) は決定論的大域最適化を保証するMINLPソルバー。30年近い学術研究の成果（INFORMS Computing Society Prize、Beale-Orchard-Hays Prize受賞）。
出典: [BARON Solver](https://www.minlp.com/baron-solver), [BARON - Wikipedia](https://en.wikipedia.org/wiki/BARON)

#### 対応問題種別
- LP, NLP, MIP, MINLP
出典: [BARON Solver](https://www.minlp.com/baron-solver)

#### アルゴリズム
**Branch-and-reduce法**:
- 分枝限定法 (branch-and-bound) + 制約伝播 + 区間解析 + 双対性技術
- 凸緩和 (convex relaxations) + 領域削減 (domain reduction) で大域最適不能領域を系統的排除
出典: [BARON - GAMS](https://www.gams.com/latest/docs/S_BARON.html), [BARON Solver](https://www.minlp.com/baron-solver)

#### 性能
- **2024年比較**: BARON は局所・大域ソルバーの両方を上回る性能
出典: [Solving continuous and discrete nonlinear programs with BARON](https://link.springer.com/article/10.1007/s10589-024-00633-0)
- **2025年アップデート**: 9回の更新。presolve、凸性識別、緩和、分離、削減戦略、メモリ管理、marginals計算の改善
出典: [The Year 2025 for GAMS Solvers](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/)
- **実績**: 数十万変数・制約の問題を解決（問題構造が良好な場合）
出典: [BARON Solver](https://www.minlp.com/baron-solver)

#### 大域最適性保証
- 有限な変数上下限が存在する場合、公正に一般的な仮定下で**決定論的大域最適解**を保証
- 従来NLP/MINLPアルゴリズムが凸性仮定下でのみ収束保証するのに対し、BARONは非凸問題でも大域最適性を保証する商用初の最適化ソフトウェア
出典: [BARON Solver](https://www.minlp.com/baron-solver), [BARON - Wikipedia](https://en.wikipedia.org/wiki/BARON)

#### 並列化
- 整数変数を含むMINLP解決時に並列コンピューティング対応
- 下限計算ステップでCPLEX/CBCを並列モードで自動実行、複数CPUコア活用
出典: [BARON - GAMS](https://www.gams.com/latest/docs/S_BARON.html)

#### ライセンス・価格
- **学術ライセンス**: 学術割引価格。Carnegie Mellon University、University of Illinois at Urbana-Champaign、University System of Georgia所属者は**無料**
出典: [BARON Licenses](https://www.minlp.com/baron-licenses)
- **ライセンス形態**: 永久/年額/月額、シングル/マルチユーザー
出典: [BARON Licenses](https://www.minlp.com/baron-licenses)
- **価格**: 要問合せ (sales@minlp.com)
出典: [BARON Solver - AMPL](https://ampl.com/products/solvers/global-solvers/baron/)

#### サポート
- 全ライセンス: 全機能アクセス、問題サイズ無制限、無制限同時使用、全システム/プラットフォーム対応
- MATLAB, Pyomo, JuMP, YALMIP, BARON独自モデリング言語で任意のOS/プロセッサで使用可
出典: [BARON Licenses](https://www.minlp.com/baron-licenses)

#### 顧客基盤
- Fortune 500企業、国立研究所、大学など1,000超の顧客
- エネルギー、金融、医療、材料、テクノロジー分野
出典: [The Leader in Global Optimization](https://minlp.com/)

### 分析

**強み:**
- **唯一の決定論的大域最適化保証**（商用MINLP）
- 30年の学術研究、INFORMS賞受賞の信頼性
- 2025年も継続的アップデート（9回）
- 並列化対応、大規模問題実績
- 特定大学（CMU等）は無料

**弱み:**
- 局所ソルバー（KNITRO等）より計算時間が桁違い（大域最適性のトレードオフ）
- 商用価格非公開（カスタム見積）
- GPU対応の記載なし

**適用場面:**
- 大域最適解が必須の問題（投資、エネルギー最適化）
- 非凸MINLP問題
- 決定論的保証が必要な産業用途（金融、医療、材料）
- 特定大学の研究（無料）

---

## 4. GAMS (General Algebraic Modeling System)

### 事実

#### 概要
GAMSは数理最適化のための高水準モデリングシステム。線形・非線形・混合整数問題のモデリングと解決に設計。
出典: [GAMS - ScienceDirect Topics](https://www.sciencedirect.com/topics/engineering/general-algebraic-modeling-system)

#### 位置づけ
- **モデリングシステム** = ソルバーではなく、複数の商用/OSSソルバーを統合するプラットフォーム
- 1行のコード変更でソルバー切替、性能比較が可能
出典: [Model and Solve Statements - GAMS](https://www.gams.com/latest/docs/UG_ModelSolve.html)

#### 非線形ソルバーサポート

**局所NLPソルバー:**
- **CONOPT**: 最も効果的な組込みNLPソルバー。非線形制約に適合。実行可能パス法（GRG）ベース
出典: [Good NLP Formulations - GAMS](https://www.gams.com/latest/docs/UG_NLP_GoodFormulations.html)

**大域NLPソルバー:**
- 局所最適解を発見（大域最適性コメント不可）
- 大域ソルバー: 大域最適解発見+証明可能
出典: [Good NLP Formulations - GAMS](https://www.gams.com/latest/docs/UG_NLP_GoodFormulations.html)

**MINLPソルバー:**
- **SBB**: 標準分枝限定法 + GAMS既存NLPソルバー組合せ
出典: [SBB - GAMS](https://www.gams.com/latest/docs/S_SBB.html)
- **BARON**: 決定論的大域最適化（Branch-and-reduce）
出典: [BARON - GAMS](https://www.gams.com/latest/docs/S_BARON.html)
- **DICOPT**: NLP/MIPサブ問題を繰返し解決
出典: [DICOPT - GAMS](https://www.gams.com/latest/docs/S_DICOPT.html)
- **AlphaECP**: 拡張カット平面法（ECP）、擬似凸MINLP大域最適保証
出典: [AlphaECP - GAMS](https://www.gams.com/latest/docs/S_ALPHAECP.html)

#### ソルバー比較機能
- モデルタイプ（線形・非線形）切替容易
- ソルバー性能比較が1行コード/オプション設定で実行可能
出典: [Model and Solve Statements - GAMS](https://www.gams.com/latest/docs/UG_ModelSolve.html)

### 分析

**強み:**
- 複数NLP/MINLPソルバーを統合プラットフォームで管理
- ソルバー切替・比較が容易（開発効率向上）
- BARON等の大域最適化ソルバーへのアクセス
- 2024年CONOPT買収で開発継続保証

**弱み:**
- GAMS自体はソルバーでなく、個別ソルバーライセンスが必要
- 統合プラットフォーム依存によるオーバーヘッド

**適用場面:**
- 複数ソルバーを試行・比較したい研究開発
- 問題種別が混在するプロジェクト（LP/NLP/MINLP切替）
- モデリング効率重視の産業応用

---

## 5. NAG Library (Numerical Algorithms Group)

### 事実

#### 概要
NAG Library for Python は1,900超の関数を含む数値計算・データサイエンスライブラリ。最適化サブモジュール (`library.opt`) がNLP機能を提供。
出典: [library.opt Submodule — NAG Library for Python 31.1.0.0](https://support.nag.com/numeric/py/nagdoc_latest/naginterfaces.library.opt.html)

#### 対応問題種別
- 制約付き大規模NLP（Nonlinear Programming）
- active-set SQP法または内点法ベース
出典: [library.opt Submodule — NAG Library for Python 31.1.0.0](https://support.nag.com/numeric/py/nagdoc_latest/naginterfaces.library.opt.html)

#### 主要NLPソルバー

**1. handle_solve_ipopt**
- IPOPT（Interior Point OPTimizer）ベースの内点法ソルバー
- 大規模制約付きNLP対応
出典: [naginterfaces.library.opt.handle_solve_ipopt](https://support.nag.com/numeric/py/nagdoc_latest/naginterfaces.library.opt.handle_solve_ipopt.html), [Interior Point Method for Nonlinear Optimization - NAG](https://support.nag.com/industryarticles/nlp_ipm_mk26_v1.1.2.pdf)

**2. handle_solve_ssqp**
- active-set Sequential Quadratic Programming (SQP)法
- 準ニュートン近似（limited-memory quasi-Newton）でラグランジアンのヘシアン近似
- 大規模NLP対応
出典: [naginterfaces.library.opt.handle_solve_ssqp](https://support.nag.com/numeric/py/nagdoc_latest/naginterfaces.library.opt.handle_solve_ssqp.html)

**3. handle_solve_bounds_foas**
- 大規模境界制約付き非線形最適化
- 第一次法（first-order method）、超低メモリ要件
- active-set法 + 非単調射影勾配法 (NPG) + 非線形共役勾配法 (CG/LCG)
出典: [naginterfaces.library.opt.handle_solve_bounds_foas](https://support.nag.com/numeric/py/nagdoc_latest/naginterfaces.library.opt.handle_solve_bounds_foas.html)

**4. handle_solve_nldf**
- 制約付き一般非線形データフィッティング問題
- 各種損失関数・正則化関数対応
出典: [naginterfaces.library.opt.handle_solve_nldf](https://support.nag.com/numeric/py/nagdoc_latest/naginterfaces.library.opt.handle_solve_nldf.html)

#### API
- Python, C, C#, Java, MATLAB, R対応
出典: [GitHub - NAGPythonExamples](https://github.com/numericalalgorithmsgroup/NAGPythonExamples)

### 分析

**強み:**
- IPOPT統合（実績あるOSS内点法ソルバー活用）
- 複数アルゴリズム選択肢（内点法/SQP/第一次法）
- 第一次法による超低メモリソルバー（大規模問題）
- 1,900超の数値計算関数ライブラリと統合

**弱み:**
- MINLP非対応（NLPのみ）
- 大域最適化非保証
- ライセンス・価格情報が明示的でない
- KNITROやBARONと比較して知名度が低い

**適用場面:**
- IPOPT商用サポート付き利用
- 数値計算ライブラリと統合した最適化（科学技術計算）
- Pythonベースのデータフィッティング問題

---

## 6. MOSEK（非線形・錐最適化観点）

### 事実

#### 概要
MOSEK は線形・混合整数線形・二次・混合整数二次・二次制約付き・錐・凸非線形数理最適化問題を解くソフトウェアパッケージ。
出典: [MOSEK - Wikipedia](https://en.wikipedia.org/wiki/MOSEK)

#### 非線形対応問題種別
- **二次計画（QP/MIQP）**
- **二次錐計画（SOCP）**
- **半正定値計画（SDP）**
- **指数錐（exponential cone）**、**累乗錐（power cone）** ※v9で追加
- **凸非線形最適化**
出典: [MOSEK - Wikipedia](https://en.wikipedia.org/wiki/MOSEK), [MOSEK - GAMS](https://www.gams.com/latest/docs/S_MOSEK.html)

#### アルゴリズム
**同次内点法（homogeneous interior-point method）**:
- 連続LP/QP/錐問題で最先端
- 原始・双対の実行不可能性を確実に検出（複数論文実証）
出典: [MOSEK - Wikipedia](https://en.wikipedia.org/wiki/MOSEK), [Springer - MOSEK Interior Point Optimizer](https://link.springer.com/chapter/10.1007/978-1-4757-3216-0_8)

#### 非線形錐制約対応（2026年最新）
MOSEK v11（2026年2月リリース）は以下の錐を扱う:
- 二次錐（second-order cone）
- 累乗錐（power cone）: 累乗、幾何平均、積、p-ノルム
- 指数錐（exponential cone）: 指数、対数、log-sum-exp、エントロピー、相対エントロピー、幾何計画（geometric programming）
- 半正定値錐（semidefinite cone）
出典: [MOSEK Optimization Suite Release 11.1.5](https://docs.mosek.com/11.1/intro.pdf), [3 Conic quadratic optimization — MOSEK Modeling Cookbook 3.4.0](https://docs.mosek.com/modeling-cookbook/cqo.html)

#### 性能
- **大規模問題で優位**: MINLP研究でMOSEKが大規模問題でGurobi上回り、実行可能性でより正確な解
出典: [arXiv - MINLP Study](https://arxiv.org/pdf/2303.04216)
- **内点法最強**: 連続LP/QP/錐問題で最先端の内点最適化器
出典: [solver.com - MOSEK](https://www.solver.com/mosek-solver-engine), [GAMS - MOSEK](https://www.gams.com/latest/docs/S_MOSEK.html)

#### ライセンス・API
- 学術プログラムあり（詳細要確認）
- C/C++, Java, Python, MATLAB, .NET, Julia, Rust, R対応
出典: [MOSEK Interfaces](https://docs.mosek.com/latest/intro/interfaces.html)

### 分析（非線形観点）

**強み:**
- **錐最適化の圧倒的専門性**（SOCP/SDP/指数錐/累乗錐）
- 幾何計画、エントロピー最適化など特殊非線形問題を錐定式化で効率的に解決
- 内点法最強、実行不可能性検出の信頼性高い
- 金融工学（ポートフォリオ最適化）、エネルギー分野に強い

**弱み:**
- 一般的なNLPには不向き（錐定式化可能な問題に限定）
- MINLP性能はKNITRO/BARON劣る可能性
- 小問題では並列化が逆効果（60秒未満）

**適用場面:**
- SOCP/SDP問題（ポートフォリオ最適化、ロバスト最適化）
- 幾何計画、エントロピー最適化
- 大規模LP（内点法が有利）
- 実行可能性判定が重要な問題

---

## 7. Gurobi（非線形観点）

### 事実

#### 非線形対応範囲
- **QP（二次計画）/MIQP（混合整数二次計画）**
- **QCP（二次制約付き計画）/MIQCP（混合整数二次制約付き計画）**
- **SOCP（二次錐計画）**
- **双線形（bilinear）制約**
- **一般非凸二次制約・目的関数**
- **多変量合成非線形関数制約**（v12以降）
出典: [What types of models can Gurobi solve?](https://support.gurobi.com/hc/en-us/articles/360013156432-What-types-of-models-can-Gurobi-solve), [Gurobi Optimizer](https://www.gurobi.com/solutions/gurobi-optimizer/)

#### 大域最適化
**v9.0以降の非凸QP/QCQP大域最適化**:
- 双線形ソルバー導入（v9.0）
- 非凸二次目的関数・制約の**大域最適解**を求める（多くの非線形ソルバーが局所最適のみ）
出典: [Non-Convex Quadratic Optimization - Gurobi](https://www.gurobi.com/events/non-convex-quadratic-optimization/), [What types of models can Gurobi solve?](https://support.gurobi.com/hc/en-us/articles/360013156432-What-types-of-models-can-Gurobi-solve)

#### v12の非線形強化
- **非凸MIQCP劇的高速化**
- **非線形制約導入**: Python nlfuncヘルパー関数、C/C++/.NET/Java Expression Trees
出典: [What types of models can Gurobi solve?](https://support.gurobi.com/hc/en-us/articles/360013156432-What-types-of-models-can-Gurobi-solve)

#### 限界
- 一般NLP（任意の非線形関数）には非対応
- 対応範囲: 二次形式 + 双線形 + v12の限定的非線形関数
出典: [Gurobi MIQCP](https://www.gurobi.com/faqs/gurobi-miqcp/)

### 分析（非線形観点）

**強み:**
- 非凸QP/QCQP大域最適化（他商用ソルバーにない）
- v12で非凸MIQCP大幅高速化
- GPU対応（Linux、大規模LP）

**弱み:**
- 一般NLP非対応（KNITRO/BARON対象外問題は扱えない）
- 非線形機能はKNITRO/BARON比で限定的

**適用場面（非線形）:**
- 非凸二次計画（大域最適解が必要）
- 双線形制約問題
- 二次錐計画（MIPと組合せ）

---

## 8. CPLEX（非線形観点）

### 事実

#### 非線形対応範囲
- **QP（二次計画）/MIQP（混合整数二次計画）**
- **QCP（二次制約付き計画）/MIQCP（混合整数二次制約付き計画）**
- **SOCP（二次錐計画）**: 二次錐制約、回転二次錐制約、より一般的な凸二次制約
出典: [CPLEX - Wikipedia](https://en.wikipedia.org/wiki/CPLEX), [CPLEX - GAMS](https://www.gams.com/latest/docs/S_CPLEX.html)

#### 凸・非凸対応
- **凸QP/QCP**: 内点法、active-set法で高速求解
- **非凸QP**: バリアアルゴリズム、分枝限定法。局所最適または大域最適解を求める手法あり
出典: [IBM ILOG CPLEX - QP](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/UsrMan/topics/cont_optim/qp/01_QP_title_synopsis.html), [Nonlinear Optimization with CPLEX](https://optimization.community/article/Nonlinear_Optimization_with_CPLEX.html)

**注意**: 非凸QPの解決は理論的に計算複雑性保証なし。凸QP同等次元の非凸QPは桁違いに時間がかかる場合あり。
出典: [Nonconvex quadratic programming comparisons - YALMIP](https://yalmip.github.io/example/nonconvexquadraticprogramming/)

#### 限界
- 一般NLP（任意非線形関数）非対応
- 対応は二次形式のみ
出典: [Modeling Nonlinear Optimization Problems in CPLEX - AMPL](https://ampl.com/modeling-nonlinear-optimization-problems-in-cplex-methods-and-best-practices/)

### 分析（非線形観点）

**強み:**
- 凸QP/QCPで成熟した性能
- SOCP対応（ロバスト最適化）
- 並列MIP処理（非凸QPの分枝限定）

**弱み:**
- 一般NLP非対応
- 非凸QP性能がGurobi v12劣る可能性（Gurobi大幅高速化）
- GPU対応記載なし

**適用場面（非線形）:**
- 凸QP/QCP（ポートフォリオ、供給網最適化）
- SOCP（ロバスト最適化）
- 並列処理活用の非凸QP（IBM製品統合環境）

---

## 9. FICO Xpress（非線形観点）

### 事実

#### 非線形対応範囲
- **LP, MIP, QP, MIQP, 非線形, MINLP**
- 制約プログラミング（Constraint Programming）
出典: [FICO Xpress - Wikipedia](https://en.wikipedia.org/wiki/FICO_Xpress)

#### MINLP性能
- 2020年比で全体68%高速、100秒超モデルで**5.3倍高速**
出典: [FICO Xpress MIP Performance](https://www.fico.com/blogs/blogs/experience-faster-mixed-integer-programming-optimization-with-xpress)

#### アルゴリズム
- **LP/QP**: Primal simplex, Dual simplex, Barrier interior-point, PDHG
- **MIP/非凸**: Branch-and-bound + Cutting-plane
出典: [FICO Xpress - Wikipedia](https://en.wikipedia.org/wiki/FICO_Xpress)

#### GPU対応
- GPU対応PDHG実装（v9.8ベータ版）
- Linux (x86_64, ARM64), Windows (x86_64)
- 大規模LP最大**50倍高速化**
出典: [FICO Xpress - Wikipedia](https://en.wikipedia.org/wiki/FICO_Xpress), [FICO Xpress Solver](https://www.fico.com/en/latest-thinking/solution-sheet/fico-xpress-solver)

### 分析（非線形観点）

**強み:**
- MINLP性能大幅向上（5.3倍高速）
- GPU対応（Windows可、Gurobiより広い）
- 非線形・制約プログラミング統合

**弱み:**
- GPU対応ベータ版（安定性要確認）
- 大域最適化非保証
- ベンチマーク非参加（第三者評価困難）

**適用場面（非線形）:**
- Windows環境でGPU活用MINLP
- 非線形+制約プログラミング複合問題
- NumPy中心Pythonワークフロー

---

## 10. SNOPT (Sparse Nonlinear OPTimizer)

### 事実

#### 概要
大規模疎非線形計画問題（線形・非線形）のためのソフトウェアパッケージ。疎SQPアルゴリズム + limited-memory準ニュートン近似（ラグランジアンのヘシアン）。
出典: [SNOPT - Wikipedia](https://en.wikipedia.org/wiki/SNOPT), [SNOPT - CCoM](https://ccom.ucsd.edu/~optimizers/solvers/snopt/)

#### アルゴリズム
**Sequential Quadratic Programming (SQP)法**
- MINOSやNPSOLより少ない行列計算
- 関数・勾配評価回数が少ない（高価な関数評価に有効）
出典: [SNOPT - MINOS Solver](https://tomopt.com/docs/snoptref/tomlab_snopt007.php), [SNOPT Paper](https://www.ccom.ucsd.edu/~peg/papers/snpaper.pdf)

#### 性能
**2023年ベンチマーク**:
- 収束率はKNITRO (interior-point/D) よりわずかに低いが、**著しく高速**
出典: [Comparative Analysis of Nonlinear Programming Solvers](https://www.mdpi.com/2504-446X/7/8/487)

#### 商用利用
- 航空宇宙、工学、制御システム最適化で効果的
- AMPL、GAMSで利用可（自動スケーリング、warm-start、精緻なソルバー統合）
出典: [Nonlinear Solvers - AMPL](https://ampl.com/products/solvers/nonlinear-solvers/), [SNOPT - GAMS](https://www.gams.com/latest/docs/S_SNOPT.html)

### 分析

**強み:**
- 高価な関数評価問題に特化（評価回数少）
- 大規模疎行列に最適化
- 航空宇宙分野で実績

**弱み:**
- MINLP非対応（NLPのみ）
- 大域最適化非保証
- 知名度がKNITRO/BARON劣る

**適用場面:**
- 航空宇宙・制御系（関数評価が高価）
- 大規模疎NLP
- warm-start活用の逐次最適化

---

## 11. MINOS

### 事実

#### 概要
大規模疎非線形計画問題のための確立されたソルバー。滑らかな制約を持つ問題に対応。
出典: [SNOPT - MINOS Solver](https://tomopt.com/docs/snoptref/tomlab_snopt007.php)

#### アルゴリズム
- **GRG (Generalized Reduced Gradient)法**
- **Simplex法**（線形問題）
出典: [Nonlinear Solvers - AMPL](https://ampl.com/products/solvers/nonlinear-solvers/)

#### 機能
- 自動問題スケーリング
- warm-start能力
- ソルバー統合拡張
出典: [Nonlinear Solvers - AMPL](https://ampl.com/products/solvers/nonlinear-solvers/)

### 分析

**強み:**
- 長年の実績（確立されたソルバー）
- 自動スケーリング、warm-start

**弱み:**
- SNOPTより性能劣る（行列計算多、評価回数多）
- MINLP非対応
- 近年の開発活動が不明

**適用場面:**
- レガシーシステムとの互換性
- Simplexベースのハイブリッド最適化

---

## 12. CONOPT

### 事実

#### 概要
実行可能パス法ベースのNLPソルバー。古典的GRG法に多数の拡張。2024年にGAMSが買収、継続開発保証。
出典: [CONOPT - Nonlinear Solver](https://conopt.gams.com/), [The Year 2025 for GAMS Solvers](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/)

#### アルゴリズム
**GRG (Generalized Reduced Gradient)法 + 拡張**:
- 前処理 (preprocessing)
- 特殊phase 0
- 線形モード反復 (linear mode iterations)
- SLP (Sequential Linear Programming)コンポーネント
- SQP (Sequential Quadratic Programming)コンポーネント
出典: [The CONOPT Algorithm](https://conopt.gams.com/algorithm/), [CONOPT - GAMS](https://www.gams.com/latest/docs/S_CONOPT.html)

#### 性能（2026年改善）
- SQP法で**共役勾配の動的スケーリング選択**導入。多数superbasicモデルで性能改善
- presolve改善（より多くの制約除去可能）
- さらなる基盤改修進行中（今後数年の大幅改善の基礎）
出典: [The Year 2025 for GAMS Solvers](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/)

#### 特徴
- 非常に非線形なモデルに好適
- 実行可能性達成が困難なモデルに効果的
- マルチメソッドアーキテクチャ + 自動最適手法選択
出典: [The CONOPT Algorithm](https://conopt.gams.com/algorithm/), [CONOPT - GAMS](https://www.gams.com/latest/docs/S_CONOPT.html)

### 分析

**強み:**
- 難非線形問題・実行可能性困難問題に強い
- 2026年も継続改善（共役勾配、presolve）
- GAMS統合（ソルバー切替容易）

**弱み:**
- MINLP非対応（NLPのみ）
- 大域最適化非保証
- 単体製品としての入手経路がGAMS依存

**適用場面:**
- 高度に非線形な化学プロセス最適化
- 実行可能性が困難な問題
- GAMS環境での全般NLP

---

## 13. その他の商用非線形ソルバー

### LOQO
- **タイプ**: 内点法ソルバー
- **強み**: 一般制約付き非線形最適化
- **ライセンス**: ライセンス料（制限付き学生版無料）
出典: [Benchmark of open-source smooth NLP solvers](https://www.alglib.net/nonlinear-programming/smooth-nlp-benchmark.php)

### NLPQL / NLPQLP
- **タイプ**: SQP法（滑らか連続微分可能目的関数・制約）
- **強み**: 数万変数・制約を効率処理
- **ライセンス**: 要確認
出典: [NLPQL - pyOpt](https://www.pyopt.org/reference/optimizers.nlpql.html), [Sequential quadratic programming - Wikipedia](https://en.wikipedia.org/wiki/Sequential_quadratic_programming)

### SLSQP
- **タイプ**: SQP法（open-source）
- **状況**: 1980年代開発、更新限定的。SciPy経由で広く利用
出典: [OpenSQP - arXiv](https://arxiv.org/html/2512.05392v1)

---

## 14. アルゴリズム種別比較

| アルゴリズム | 採用ソルバー | 特徴 | 適用場面 |
|------------|------------|------|---------|
| **内点法 (Interior-point)** | KNITRO, MOSEK, NAG, Gurobi, CPLEX, Xpress, LOQO, IPOPT | バリア関数、連続凸問題に強い | 大規模LP/QP、錐最適化 |
| **SQP (Sequential Quadratic Programming)** | KNITRO, SNOPT, NLPQL, NAG, CONOPT | 二次近似反復、滑らか非線形に強い | 高価関数評価、制約多数 |
| **拡大ラグランジアン (Augmented Lagrangian)** | KNITRO v15 | 退化問題対応 | 退化制約問題 |
| **Active-set法** | KNITRO, NAG, CPLEX | 制約活性集合追跡 | 中規模NLP |
| **GRG (Generalized Reduced Gradient)** | CONOPT, MINOS | 実行可能パス法 | 実行可能性困難 |
| **Branch-and-reduce** | BARON | 分枝限定+凸緩和+領域削減 | 大域MINLP |
| **Branch-and-bound** | KNITRO (MINLP), Gurobi, CPLEX, Xpress | MIP拡張 | MINLP |
| **PDHG (Primal-Dual Hybrid Gradient)** | Gurobi (GPU), Xpress (GPU) | GPU加速 | 巨大LP |

---

## 15. 並列化対応比較

| ソルバー | 並列化方式 | 特記事項 |
|---------|----------|---------|
| **KNITRO** | 並列分枝限定（MINLP、4スレッド5倍） | v13以降 |
| **BARON** | 並列下限計算（CPLEX/CBC並列モード） | 整数変数含むMINLP |
| **MOSEK** | 内点法・MIP並列化 | 小問題（<60秒）逆効果 |
| **Gurobi** | デフォルト全コア、GPU対応（Linux） | PDHG、NVIDIA H100推奨 |
| **CPLEX** | ノード並列、strong branching並列、分散並列 | 動的タスク分解（2025研究） |
| **Xpress** | マルチスレッド、GPU対応（Win/Linux） | GPU最大50倍高速 |
| **SNOPT/MINOS/CONOPT** | 記載なし | 単一コア前提 |

---

## 16. 価格・ライセンス比較

| ソルバー | 学術ライセンス | 商用価格 | 備考 |
|---------|--------------|---------|------|
| **KNITRO** | 無料（教育用、12ヶ月更新） | 要問合せ | full-featured |
| **BARON** | 学術割引（CMU/UIUC/UGA無料） | 月額/年額/永久 | sales@minlp.com |
| **MOSEK** | 学術プログラム有 | 要問合せ | 詳細要確認 |
| **Gurobi** | 無料（制限なし） | サブスク制 | 学術最強 |
| **CPLEX** | 無料（1年更新） | サブスク制 | IBM Academic Initiative |
| **Xpress** | 無料（学術） | カスタム見積 | |
| **NAG** | 要確認 | ライセンス料 | |
| **SNOPT/MINOS** | 要確認 | ライセンス料 | |
| **CONOPT** | GAMS経由 | GAMS経由 | GAMS買収 |
| **IPOPT** | **OSS（EPL、無料）** | 商用利用可 | 第三者コンポーネント要確認 |

**傾向**: 全商用ソルバーとも価格非公開（カスタム見積）。学術ライセンスは無料～割引。

---

## 17. 業界採用状況

### BARON
- Fortune 500企業、国立研究所、大学など1,000超の顧客
- エネルギー、金融、医療、材料、テクノロジー分野
出典: [The Leader in Global Optimization](https://minlp.com/)

### KNITRO
- 世界中数百サイト
出典: [Artelys Knitro - Wikipedia](https://en.wikipedia.org/wiki/Artelys_Knitro)

### MOSEK
- テクノロジー、金融、エネルギー、林業
出典: [MOSEK ApS](https://www.mosek.com/)

### SNOPT
- 航空宇宙、工学、制御システム
出典: [Nonlinear Solvers - AMPL](https://ampl.com/products/solvers/nonlinear-solvers/)

---

## 18. 性能ベンチマーク総括

### 2023-2024年比較研究結果

**KNITRO vs BARON（速度 vs 収束率）**:
- KNITRO (interior-point/D): 第2位収束率、BARON比CPU時間**2桁高速**
- BARON: 局所・大域ソルバー上回る性能（2024）
出典: [Comparative Analysis of Nonlinear Programming Solvers](https://www.mdpi.com/2504-446X/7/8/487), [Solving continuous and discrete nonlinear programs with BARON](https://link.springer.com/article/10.1007/s10589-024-00633-0)

**SNOPT（速度特化）**:
- 収束率KNITRO比やや低、著しく高速
出典: [Comparative Analysis of Nonlinear Programming Solvers](https://www.mdpi.com/2504-446X/7/8/487)

**MOSEK（大規模LP/錐）**:
- 大規模問題でGurobi上回る（MINLP研究）
出典: [arXiv - MINLP Study](https://arxiv.org/pdf/2303.04216)

### トレードオフ
- **速度重視**: SNOPT > KNITRO >> BARON（大域最適化コスト）
- **収束品質**: KNITRO (interior-point/D) > SNOPT
- **大域最適保証**: BARON > Gurobi（非凸QP）> その他（局所のみ）

---

## 19. 選定ガイドライン

| 要件 | 推奨ソルバー | 理由 |
|-----|------------|------|
| **大域MINLP（決定論的保証）** | BARON | 唯一の決定論的大域最適保証 |
| **高速MINLP（局所最適可）** | KNITRO | 26%高速化、内点法/SQP選択可 |
| **非凸QP大域最適化** | Gurobi v12 | 非凸MIQCP劇的高速化 |
| **錐最適化（SOCP/SDP）** | MOSEK | 内点法最強、指数錐対応 |
| **高価な関数評価NLP** | SNOPT, KNITRO (MISQP) | 評価回数最小化 |
| **難非線形・実行可能性困難** | CONOPT | GRG実行可能パス法 |
| **巨大LP（GPU活用）** | Gurobi (Linux), Xpress (Win/Linux) | GPU 50倍高速 |
| **学術研究（予算制約）** | Gurobi, CPLEX, IPOPT | 無料学術ライセンス |
| **金融工学（ポートフォリオ）** | MOSEK | SOCP/SDP、錐定式化 |
| **航空宇宙・制御系** | SNOPT | 疎行列、warm-start |
| **化学プロセス** | CONOPT | 高非線形対応 |

---

## 20. 技術トレンド（2025-2026）

### 1. 大域最適化の民主化
- BARON: 30年研究成果、決定論的保証
- Gurobi: 非凸QP大域最適化拡大（v12）

### 2. GPU活用の拡大（限定的）
- Gurobi: Linux限定、NVIDIA H100推奨
- Xpress: Windows/Linux/ARM64、ベータ版
- **課題**: 安定性、プラットフォーム制約

### 3. MINLP性能競争
- KNITRO: v13 5倍→v14 26%追加高速化
- Xpress: 5.3倍高速化（MINLP）
- BARON: 2025年9回アップデート

### 4. アルゴリズム多様化
- KNITRO: 拡大ラグランジアン法追加（v15、退化問題）
- CONOPT: 共役勾配動的スケーリング（2026）

### 5. 錐の拡張
- MOSEK: 指数錐・累乗錐（v9）→幾何計画、エントロピー最適化

### 6. オープンソース統合
- NAG: IPOPT統合（商用サポート付きOSS活用）
- IPOPT: Eclipse Public License、商用利用可

---

## 21. 商用 vs OSS の性能差（非線形）

### NLP性能差
- 商用（KNITRO/SNOPT）: 2023年ベンチマークでIPOPT/SLSQP（OSS）より高速かつ高収束率
- IPOPT（OSS）: 商用に近い性能（NAGが統合採用）
出典: [Comparative Analysis of Nonlinear Programming Solvers](https://www.mdpi.com/2504-446X/7/8/487)

### MINLP性能差
- 商用（KNITRO/BARON）: OSS（Bonmin等）より桁違い高速（BARON大域保証付き）
- 理由: 数十年開発投資、並列化、ヒューリスティクス洗練度
出典: 既存commercial_solvers.mdより

### 使い分け
- **学術研究**: 商用学術ライセンス（無料・高性能）
- **商用・大規模NLP/MINLP**: KNITRO/BARON（桁違い高速）
- **商用・小規模/予算制約**: IPOPT（OSS、実用的性能）

---

## 22. 情報源

### KNITRO
- [Knitro - Artelys](https://www.artelys.com/solvers/knitro/)
- [Artelys Knitro - Wikipedia](https://en.wikipedia.org/wiki/Artelys_Knitro)
- [Algorithms — Artelys Knitro 15.1 User's Manual](https://www.artelys.com/app/docs/knitro/2_userGuide/algorithms.html)
- [Mixed-integer nonlinear programming — Artelys Knitro 15.1](https://www.artelys.com/app/docs/knitro/2_userGuide/minlp.html)
- [Artelys Knitro 14 new release](https://www.artelys.com/news/artelys-knitro-14-new-release-nonlinear-optimization-solver/)
- [Artelys Knitro 13 solves MINLP 5 times faster](https://www.artelys.com/news/artelys-knitro-13-solves-minlp-problems-5-times-faster/)
- [Knitro Solver - AMPL](https://ampl.com/products/solvers/nonlinear-solvers/knitro/)

### BARON
- [BARON Solver](https://www.minlp.com/baron-solver)
- [BARON - Wikipedia](https://en.wikipedia.org/wiki/BARON)
- [BARON - GAMS](https://www.gams.com/latest/docs/S_BARON.html)
- [BARON Licenses](https://www.minlp.com/baron-licenses)
- [BARON Solver - AMPL](https://ampl.com/products/solvers/global-solvers/baron/)
- [Solving continuous and discrete nonlinear programs with BARON](https://link.springer.com/article/10.1007/s10589-024-00633-0)

### NAG
- [library.opt Submodule — NAG Library for Python 31.1.0.0](https://support.nag.com/numeric/py/nagdoc_latest/naginterfaces.library.opt.html)
- [naginterfaces.library.opt.handle_solve_ipopt](https://support.nag.com/numeric/py/nagdoc_latest/naginterfaces.library.opt.handle_solve_ipopt.html)
- [naginterfaces.library.opt.handle_solve_ssqp](https://support.nag.com/numeric/py/nagdoc_latest/naginterfaces.library.opt.handle_solve_ssqp.html)
- [Interior Point Method for Nonlinear Optimization - NAG](https://support.nag.com/industryarticles/nlp_ipm_mk26_v1.1.2.pdf)

### GAMS
- [GAMS - Solver Manuals](https://www.gams.com/latest/docs/S_MAIN.html)
- [Good NLP Formulations - GAMS](https://www.gams.com/latest/docs/UG_NLP_GoodFormulations.html)
- [DICOPT - GAMS](https://www.gams.com/latest/docs/S_DICOPT.html)
- [SBB - GAMS](https://www.gams.com/latest/docs/S_SBB.html)
- [AlphaECP - GAMS](https://www.gams.com/latest/docs/S_ALPHAECP.html)

### MOSEK
- [MOSEK - Wikipedia](https://en.wikipedia.org/wiki/MOSEK)
- [MOSEK Optimization Suite Release 11.1.5](https://docs.mosek.com/11.1/intro.pdf)
- [3 Conic quadratic optimization — MOSEK Modeling Cookbook 3.4.0](https://docs.mosek.com/modeling-cookbook/cqo.html)
- [MOSEK - GAMS](https://www.gams.com/latest/docs/S_MOSEK.html)
- [arXiv - MINLP Study](https://arxiv.org/pdf/2303.04216)

### Gurobi (非線形)
- [What types of models can Gurobi solve?](https://support.gurobi.com/hc/en-us/articles/360013156432-What-types-of-models-can-Gurobi-solve)
- [Non-Convex Quadratic Optimization - Gurobi](https://www.gurobi.com/events/non-convex-quadratic-optimization/)
- [Gurobi MIQCP](https://www.gurobi.com/faqs/gurobi-miqcp/)

### CPLEX (非線形)
- [IBM ILOG CPLEX - QP](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/UsrMan/topics/cont_optim/qp/01_QP_title_synopsis.html)
- [Nonlinear Optimization with CPLEX](https://optimization.community/article/Nonlinear_Optimization_with_CPLEX.html)
- [Modeling Nonlinear Optimization Problems in CPLEX - AMPL](https://ampl.com/modeling-nonlinear-optimization-problems-in-cplex-methods-and-best-practices/)

### SNOPT
- [SNOPT - Wikipedia](https://en.wikipedia.org/wiki/SNOPT)
- [SNOPT - CCoM](https://ccom.ucsd.edu/~optimizers/solvers/snopt/)
- [SNOPT Paper](https://www.ccom.ucsd.edu/~peg/papers/snpaper.pdf)
- [SNOPT - GAMS](https://www.gams.com/latest/docs/S_SNOPT.html)

### MINOS
- [SNOPT - MINOS Solver](https://tomopt.com/docs/snoptref/tomlab_snopt007.php)
- [Nonlinear Solvers - AMPL](https://ampl.com/products/solvers/nonlinear-solvers/)

### CONOPT
- [CONOPT - Nonlinear Solver](https://conopt.gams.com/)
- [The CONOPT Algorithm](https://conopt.gams.com/algorithm/)
- [CONOPT - GAMS](https://www.gams.com/latest/docs/S_CONOPT.html)
- [The Year 2025 for GAMS Solvers](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/)

### ベンチマーク・比較
- [Comparative Analysis of Nonlinear Programming Solvers](https://www.mdpi.com/2504-446X/7/8/487)
- [ResearchGate - Comparative Analysis](https://www.researchgate.net/publication/372596120_Comparative_Analysis_of_Nonlinear_Programming_Solvers_Performance_Evaluation_Benchmarking_and_Multi-UAV_Optimal_Path_Planning)
- [A Review and Comparison of Solvers for Convex MINLP](https://egon.cheme.cmu.edu/Papers/ConvexMINLPReview2018_OO.pdf)

### その他
- [IPOPT - Wikipedia](https://en.wikipedia.org/wiki/IPOPT)
- [GitHub - coin-or/Ipopt](https://github.com/coin-or/Ipopt)
- [Sequential quadratic programming - Wikipedia](https://en.wikipedia.org/wiki/Sequential_quadratic_programming)
- [OpenSQP - arXiv](https://arxiv.org/html/2512.05392v1)

---

**調査完了日**: 2026-02-12
**調査担当**: ashigaru2 (Sonnet 4.5)
**調査方法**: Web検索（2026年最新情報）、既存commercial_solvers.md参照、事実と分析の明確分離
