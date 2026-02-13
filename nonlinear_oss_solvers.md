# 非線形OSSソルバー総合調査

## 事実（Facts with Sources）

### 1. Ipopt (Interior Point Optimizer)

#### 対応問題種別
- Nonlinear Programming (NLP)
- 大規模非線形最適化問題
- 非凸関数も対応可能（ただし2階連続微分可能であること）
- Source: [Ipopt Documentation](https://coin-or.github.io/Ipopt/), [Ipopt Wikipedia](https://en.wikipedia.org/wiki/IPOPT)

#### ライセンス
- Eclipse Public License (EPL)
- COIN-ORからオープンソース提供
- Source: [Ipopt Documentation](https://coin-or.github.io/Ipopt/)

#### アルゴリズム
- Primal-dual interior point method
- Filter methodベースのline search
- Sparse symmetric indefinite linear system solverに強く依存
- Source: [Ipopt Documentation](https://coin-or.github.io/Ipopt/), [GAMS IPOPT](https://www.gams.com/latest/docs/S_IPOPT.html)

#### 並列化対応
- 並列線形ソルバー経由で並列化対応:
  - MKL Pardiso
  - HSL MA86, HSL MA97
  - SPRAL (Sparse Parallel Robust Algorithms Library)
  - MUMPS (MUltifrontal Massively Parallel sparse direct Solver)
- 最適化アルゴリズム自体ではなく、内部線形ソルバーによる並列化
- Source: [Ipopt Documentation](https://coin-or.github.io/Ipopt/), [Exploring Benefits of Linear Solver Parallelism (arXiv)](https://arxiv.org/abs/1909.08104)

#### 開発活発さ
- COIN-ORプロジェクトとして継続メンテナンス
- 2026年時点の具体的リリース情報は限定的

#### 言語バインディング
- C, C++, Fortran, Java, R, Python, その他多数の言語から呼び出し可能
- Source: [Ipopt Documentation](https://coin-or.github.io/Ipopt/)

---

### 2. Bonmin (Basic Open-source Nonlinear Mixed INteger programming)

#### 対応問題種別
- Mixed Integer Nonlinear Programming (MINLP)
- 凸問題に対しては厳密解、非凸問題に対してはヒューリスティック
- Source: [BONMIN — AMPL Resources](https://dev.ampl.com/solvers/bonmin/index.html), [Bonmin GitHub](https://github.com/coin-or/Bonmin)

#### アルゴリズム
- B-BB (NLP-based branch-and-bound algorithm)
- B-OA (outer-approximation decomposition algorithm)
- B-iFP (iterated feasibility pump algorithm)
- B-QG (Quesada and Grossmann's branch-and-cut algorithm)
- B-Hyb (hybrid outer-approximation based branch-and-cut algorithm)
- B-Ecp (variant of B-QG with ECP cuts)
- Source: [BONMIN Users' Manual](https://www.coin-or.org/Bonmin/Intro.html)

#### ライセンス
- Eclipse Public License (EPL)
- OSI Certified Open Source Software
- Source: [Bonmin Home Page](https://www.coin-or.org/Bonmin/)

#### 開発活発さ
- GitHub Stars: 141
- 最新リリース: 1.8.9 (2023年1月30日)
- 近年の開発活動は低調
- Source: [Bonmin GitHub](https://github.com/coin-or/Bonmin)

#### 実装
- 実装言語: C++
- IpoptとCbcの上に構築
- COIN-ORプロジェクト（2004年、IBM & Carnegie Mellon University）
- Source: [Bonmin GitHub](https://github.com/coin-or/Bonmin)

#### Pythonバインディング
- pyMIQP: Mixed Integer Quadratic Programming for Python (using MINLP-solver Bonmin)
- Source: [pyMIQP GitHub](https://github.com/sschnug/pyMIQP)

---

### 3. Couenne (Convex Over and Under ENvelopes for Nonlinear Estimation)

#### 対応問題種別
- Mixed Integer Nonlinear Programming (MINLP)
- **Global optimization（大域最適化）に特化**
- 非凸MINLPの大域最適解を求めることが目的
- Source: [Couenne](https://www.coin-or.org/Couenne/), [Couenne Wikipedia](https://en.wikipedia.org/wiki/Couenne)

#### アルゴリズム
- Spatial branch & bound algorithm
- 線形化（linearization）、境界縮小（bound reduction）、分岐（branching）
- Reformulation procedureにより線形計画近似を構築
- Source: [Couenne — AMPL Resources](https://dev.ampl.com/solvers/couenne/index.html), [Couenne GitHub](https://github.com/coin-or/Couenne)

#### ライセンス
- Eclipse Public License (EPL)
- OSI Certified Open Source Software
- Source: [Couenne](https://www.coin-or.org/Couenne/)

#### 開発活発さ
- GitHub Stars: 83
- Forks: 11
- 近年の開発活動は低調（2026年時点で具体的リリース情報なし）
- Source: [Couenne GitHub](https://github.com/coin-or/Couenne)

#### 開発履歴
- 2006年にIBMとCarnegie Mellon Universityの共同プロジェクトとして開始
- Source: [Couenne](https://www.coin-or.org/Couenne/)

---

### 4. SCIP (Solving Constraint Integer Programs)

#### 対応問題種別（非線形部分）
- Mixed-Integer Nonlinear Programming (MINLP) - convex/nonconvex両対応
- Source: [SCIP Official Site](https://www.scipopt.org/), [SCIP Suite 10.0 Paper](https://arxiv.org/html/2511.18580v1)

#### ライセンス
- Apache 2.0 または GNU Lesser General Public License (LGPL)
- Source: [SCIP Suite 10.0 Paper](https://arxiv.org/html/2511.18580v1)

#### 開発活発さ
- **最新リリース: SCIP 10.0.1 (2026年2月3日)**
- SCIP Optimization Suite 10.0 (2025年11月リリース)
- 非常に活発な開発
- Source: [SCIP GitHub Releases](https://github.com/scipopt/scip/releases)

#### 非線形対応の詳細
- GAMS 53でSCIP 10が利用可能、CONOPTをNLPソルバーとして使用可能
- Benders' Decompositionフレームワーク対応
- Source: [The Year 2025 for GAMS Solvers](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/)

---

### 5. SCS (Splitting Conic Solver)

#### 対応問題種別
- Linear Programs (LPs)
- Second-Order Cone Programs (SOCPs)
- Semidefinite Programs (SDPs)
- Exponential Cone Programs (ECPs)
- Power Cone Programs (PCPs)
- または上記の任意の組み合わせ
- Source: [SCS Documentation](https://www.cvxgrp.org/scs/), [SCS GitHub](https://github.com/cvxgrp/scs)

#### ライセンス
- **MIT License**
- Source: [SCS LICENSE.txt](https://github.com/cvxgrp/scs/blob/master/LICENSE.txt)

#### 性能特性
- 大規模問題向けに設計
- 他のCVXソルバーが扱えない大規模問題を解くための実験的ソルバー
- 精度は控えめなレベル（商用ソルバーより低精度だが高速）
- Source: [Solver Features - CVXPY](https://www.cvxpy.org/tutorial/solvers/index.html)

#### 開発活発さ
- **GitHub Stars: 601**
- **現行バージョン: 3.2.11** (3.2.9が2024年10月リリース)
- 活発な開発継続中
- Source: [SCS GitHub Releases](https://github.com/cvxgrp/scs/releases)

#### 実装
- 実装言語: C
- ADMM (Alternating Direction Method of Multipliers) ベース
- Source: [SCS Documentation](https://www.cvxgrp.org/scs/)

#### 言語バインディング
- C, Python, Julia, R, MATLAB, Ruby, JavaScript (WebAssembly経由)
- Source: [SCS GitHub](https://github.com/cvxgrp/scs)

---

### 6. ECOS (Embedded Conic Solver)

#### 対応問題種別
- Second-Order Cone Programs (SOCPs)
- Linear Programs (LPs)
- Positive orthant cone, second-order cones
- ECOS_BB拡張により Mixed-Integer / Mixed-Boolean Programsも対応
- Source: [ECOS GitHub](https://github.com/embotech/ecos), [ECOS: An SOCP Solver for Embedded Systems](https://web.stanford.edu/~boyd/papers/ecos.html)

#### アルゴリズム
- Primal-dual Mehrotra predictor-corrector method
- Nesterov-Todd scaling
- Self-dual embedding
- Symmetric indefinite KKT system
- Source: [ECOS GitHub](https://github.com/embotech/ecos)

#### ライセンス
- **GNU General Public License v3.0**
- Source: [ECOS GitHub](https://github.com/embotech/ecos)

#### 開発活発さ
- **GitHub Stars: 465**
- Forks: 122
- **最新リリース: v2.0.14 (2024年6月18日)** (ecos-python)
- 活発な開発（特にPythonインターフェース）
- Source: [ECOS GitHub](https://github.com/embotech/ecos), [ecos-python Releases](https://github.com/embotech/ecos-python/releases)

#### 実装
- 実装言語: ANSI-C (low footprint, single-threaded, library-free)
- 組み込みアプリケーション向けに設計
- CVXOPT ConeLPソルバーをベースにしているが線形システム処理が異なる
- Source: [ECOS GitHub](https://github.com/embotech/ecos)

#### 言語バインディング
- Python (CVXPYのデフォルトソルバーの一つ)
- MATLAB
- Source: [ECOS PyPI](https://pypi.org/project/ecos/)

---

### 7. NLopt (Nonlinear Optimization Library)

#### 対応問題種別
- Nonlinear local and global optimization
- 勾配あり/なし関数の両対応
- 非線形制約（一部アルゴリズム）: 不等式制約（MMA, ORIG_DIRECT）、等式制約（COBYLA, SLSQP, ISRES）
- Source: [NLopt Documentation](https://nlopt.readthedocs.io/), [NLopt GitHub](https://github.com/stevengj/nlopt)

#### 特徴
- **複数のfree/open-source非線形最適化ライブラリのラッパー/統一インターフェース**
- 共通インターフェースで多様なアルゴリズム切り替え可能
- 一部アルゴリズムは大域最適化、一部は局所最適化
- 勾配ベース/勾配フリーの両方のアルゴリズム含む
- Source: [NLopt Documentation](https://nlopt.readthedocs.io/)

#### ライセンス
- オープンソース（具体的ライセンスは検索結果では未確認）
- Source: [NLopt GitHub](https://github.com/stevengj/nlopt)

#### 言語バインディング
- **多言語対応**: C, C++, Fortran, MATLAB/Octave, Python, GNU Guile, Java, Julia, GNU R, Lua, OCaml, Rust, Crystal
- Source: [NLopt Documentation](https://nlopt.readthedocs.io/)

#### 開発活発さ
- GitHubで継続メンテナンス
- Source: [NLopt GitHub](https://github.com/stevengj/nlopt)

---

### 8. CasADi (Computer Algebra System with Automatic Differentiation)

#### 対応問題種別
- Nonlinear Programming (NLP)
- Mixed-Integer Nonlinear Programming (MINLP)
- Optimal Control
- Source: [CasADi](https://web.casadi.org/), [CasADi Paper](https://link.springer.com/article/10.1007/s12532-018-0139-4)

#### 特徴
- **フレームワーク/モデリングツール（ソルバーラッパー）**
- 自動微分（Automatic Differentiation: AD）とシンボリック式処理
- Forward/Reverse mode AD
- Jacobian/Hessian自動構築（graph coloringアプローチ）
- **外部ソルバーインターフェース**: IPOPT, BONMIN, BlockSQP, WORHP, KNITRO, SNOPT, SLEQP, Alpaqa
- Source: [CasADi](https://web.casadi.org/), [CasADi Paper (PDF)](https://optimization-online.org/wp-content/uploads/2018/01/6420.pdf)

#### ライセンス
- オープンソース（具体的ライセンスは検索結果では未確認）
- Source: [CasADi](https://web.casadi.org/)

#### 実装
- 実装言語: self-contained C++
- Python, MATLAB, Octaveの完全機能インターフェース提供
- Pythonインターフェースがベストドキュメンテッド、最も安定
- Source: [CasADi](https://web.casadi.org/)

#### 開発活発さ
- 2009年後半から開発
- 学術教育、プロセス制御、ロボティクス、航空宇宙等で広く利用
- Source: [CasADi](https://web.casadi.org/)

#### Pythonサポート
- pip経由でインストール可能
- Source: [An Introduction to CasADi with Python (Medium)](https://medium.com/@shoaib6174/an-introduction-to-casadi-with-python-12055f8e652f)

---

### 9. OSQP (Operator Splitting Quadratic Program)

#### 対応問題種別
- Quadratic Programs (QPs)
- 凸QPのみ（問題データが凸であることが唯一の要件）
- Source: [OSQP](https://osqp.org/), [OSQP GitHub](https://github.com/osqp/osqp)

#### アルゴリズム
- Alternating Direction Method of Multipliers (ADMM)
- カスタムスパース線形代数ルーチン
- Absolutely division free after setup
- Primal/Dual infeasibility検出機能（first-order methodsベースQPソルバーで初）
- Source: [OSQP Paper (PDF)](https://web.stanford.edu/~boyd/papers/pdf/osqp.pdf)

#### ライセンス
- **Apache 2.0**
- Source: [OSQP](https://osqp.org/)

#### 性能
- 多くの商用/学術ソルバーを上回る性能（ベンチマーク結果）
- Source: [OSQP](https://osqp.org/)

#### 並列化対応
- 単一問題のスレッド並列化は未サポート（ワークスペース共有制約）
- ベンチマークツールは複数問題の並列実行に対応
- **CUDA実装（cuosqp）が存在**
- Source: [OSQP GitHub Issues](https://github.com/osqp/osqp/issues/272), [cuosqp GitHub](https://github.com/osqp/cuosqp)

#### 開発活発さ
- **2025年の複数リポジトリで更新確認**:
  - qdldl: 2025年11月24日
  - osqp_benchmarks: 2025年11月13日
  - osqp.rs (Rust): 2025年4月21日
  - OSQP.jl (Julia): 2025年3月7日
- 活発な開発
- Source: [OSQP GitHub](https://github.com/osqp)

#### 実装
- 実装言語: Pure C
- Library-free embedded solver (制御・ロボティクスアプリ向け)
- Source: [OSQP](https://osqp.org/)

#### 言語バインディング
- Python, Julia, MATLAB, R
- CVXPY, JuMP, YALMIPから利用可能
- Source: [OSQP](https://osqp.org/)

---

### 10. CVXOPT

#### 対応問題種別
- Linear Programs (LPs)
- Second-Order Cone Programs (SOCPs)
- Semidefinite Programs (SDPs)
- Nonlinear convex optimization
- Source: [CVXOPT](https://cvxopt.org/), [CVXOPT GitHub](https://github.com/cvxopt/cvxopt)

#### ライセンス
- **GNU General Public License (GPL)**
- Source: [CVXOPT LICENSE](https://github.com/cvxopt/cvxopt/blob/master/LICENSE)

#### 開発活発さ
- **最新版: 1.3.2 (Python 3.13サポート)**
- 活発なメンテナンス
- Source: [CVXOPT PyPI](https://pypi.org/project/cvxopt/)

#### 実装
- 実装言語: Python
- LAPACK, BLASルーチンへのインターフェース
- DSDP5 (Semidefinite Programming solver) へのインターフェース
- Source: [CVXOPT](https://cvxopt.org/)

#### 開発者
- Martin Andersen, Joachim Dahl, Lieven Vandenberghe
- Source: [CVXOPT](https://cvxopt.org/)

#### 言語バインディング
- Python native
- Source: [CVXOPT](https://cvxopt.org/)

---

### 11. CSDP (C SemiDefinite Programming)

#### 対応問題種別
- Semidefinite Programs (SDPs)
- Source: [CSDP GitHub](https://github.com/coin-or/Csdp), [pycsdp GitHub](https://github.com/mghasemi/pycsdp)

#### 実装
- 実装言語: C
- BLAS, LAPACKサブルーチンに依存
- 呼び出し可能ライブラリとして提供
- Source: [CSDP Sage Documentation](https://doc.sagemath.org/html/en/reference/spkg/csdp.html)

#### ライセンス
- COIN-ORプロジェクト（EPLと推測されるが検索結果では未確認）
- Source: [CSDP GitHub](https://github.com/coin-or/Csdp)

#### 統合
- Sage内でデフォルト利用可能、CVXPYからも利用可能
- Source: [Semidefinite Programming - Sage](https://doc.sagemath.org/html/en/reference/numerical/sage/numerical/sdp.html)

#### Pythonバインディング
- pycsdp: Python library for the fast SDP solver CSDP
- Source: [pycsdp GitHub](https://github.com/mghasemi/pycsdp)

---

## 各領域のOSS充実度マッピング

### NLP (Nonlinear Programming)

#### OSS選択肢
1. **Ipopt** - COIN-OR, EPL, 内点法, 並列対応（線形ソルバー経由）
2. **NLopt** - 統一インターフェース, 多アルゴリズム対応, 多言語バインディング
3. **CasADi** - フレームワーク, AD機能, 複数ソルバーラッパー

#### 成熟度
- **高**: Ipoptは長年の実績、COIN-ORプロジェクトとして継続メンテナンス
- **高**: NLoptは多様なアルゴリズムを統一インターフェースで提供、多言語対応
- **高**: CasADiは学術・産業で広く利用、自動微分機能が強力

#### OSSが商用に迫る程度
- **局所最適化**: OSSは商用と遜色ない（Ipoptは広く使われている）
- **大域最適化**: 商用（BARON, KNITRO global）が優位
- **スケール**: 超大規模問題では商用が有利だが、中規模以下ではOSSで十分なケース多数

### MINLP (Mixed-Integer Nonlinear Programming)

#### OSS選択肢
1. **SCIP** - Apache 2.0/LGPL, convex/nonconvex対応, **2026年2月最新版リリース**
2. **Bonmin** - EPL, convex exact / nonconvex heuristic, COIN-OR
3. **Couenne** - EPL, global optimization特化, COIN-OR

#### 成熟度
- **高（SCIP）**: 活発な開発、最新リリースあり、商用に次ぐ性能
- **中（Bonmin）**: 安定だが開発低調、凸問題に適用可能
- **中（Couenne）**: 安定だが開発低調、大域最適化には有用

#### OSSが商用に迫る程度
- **Convex MINLP**: SCIPは商用に近い性能（2018年レビューで評価）
- **Nonconvex MINLP local**: Bonminはヒューリスティック、商用（KNITRO, BARON）が優位
- **Nonconvex MINLP global**: Couenneは有用だが、商用BARON等が依然優位
- **性能ギャップ**: 商用ソルバーとのギャップは線形MIPより大きい（~2桁差）

### QP / QCQP (Quadratic Programming / Quadratically Constrained QP)

#### OSS選択肢
1. **OSQP** - Apache 2.0, QP特化, ADMM, 活発開発（2025年更新多数）
2. **HiGHS** - MIT, QP対応, 線形ソルバーとしても優秀（既存調査参照）
3. **qpOASES** - LGPL, real-time QP, 組込み向け
4. **QCQP**: ALGLIB等が対応（商用/OSS混在）

#### 成熟度
- **高（OSQP）**: 活発開発、ベンチマークで商用を上回るケースあり
- **高（HiGHS）**: 2026年も活発、GPU対応、MIT License
- **中（qpOASES）**: 安定、real-time QP特化

#### OSSが商用に迫る程度
- **QP**: OSSは商用と競合レベル（OSQPはGurobi, MOSEKと比較して優秀な結果）
- **QCQP**: 商用（Gurobi, MOSEK, CPLEX）が優位、OSS選択肢が限定的

### SOCP / SDP (Second-Order Cone Programming / Semidefinite Programming)

#### OSS選択肢
1. **SCS** - MIT, LP/SOCP/SDP/ECP/PCP, **601 stars, 2024年更新**
2. **ECOS** - GPL v3.0, SOCP特化, 組込み向け, **465 stars, 2024年6月リリース**
3. **CVXOPT** - GPL, SOCP/SDP対応, Python, **1.3.2 (Python 3.13対応)**
4. **CSDP** - COIN-OR, SDP特化, C実装

#### 成熟度
- **高（SCS, ECOS）**: 活発開発、CVXPYのデフォルトソルバー、実績豊富
- **高（CVXOPT）**: 長年の実績、Python生態系で広く利用
- **中（CSDP）**: 安定、COIN-ORプロジェクト

#### OSSが商用に迫る程度
- **SOCP**: OSSは商用と競合レベル（ECOS, SCSは広く採用）
- **SDP**: 中規模問題ではOSSで十分、大規模・高精度では商用（MOSEK等）が優位
- **精度 vs 速度**: SCSは大規模問題向け（精度控えめだが高速）、商用は高精度・高速両立

---

## 比較表

| ソルバー | 問題種別 | ライセンス | 並列化 | GitHub Stars | 最新活動 | 主要言語 | 成熟度 |
|---------|---------|-----------|--------|-------------|---------|---------|--------|
| **Ipopt** | NLP | EPL | 線形ソルバー経由 | - | 継続メンテナンス | C++ | ⭐⭐⭐⭐⭐ |
| **Bonmin** | MINLP | EPL | - | 141 | 低調（2023年最終） | C++ | ⭐⭐⭐ |
| **Couenne** | MINLP global | EPL | - | 83 | 低調 | C++ | ⭐⭐⭐ |
| **SCIP** | MINLP/CP | Apache 2.0/LGPL | UG並列 | - | **2026年2月** | C++ | ⭐⭐⭐⭐⭐ |
| **SCS** | LP/SOCP/SDP/ECP/PCP | **MIT** | - | **601** | **2024年10月** | C | ⭐⭐⭐⭐ |
| **ECOS** | SOCP | **GPL v3.0** | - | **465** | **2024年6月** | C | ⭐⭐⭐⭐ |
| **NLopt** | NLP | OSS | - | - | 継続メンテナンス | C/C++ | ⭐⭐⭐⭐ |
| **CasADi** | NLP/MINLP/OC | OSS | ソルバー依存 | - | 活発 | C++ | ⭐⭐⭐⭐⭐ |
| **OSQP** | QP | **Apache 2.0** | CUDA版あり | - | **2025年11月** | C | ⭐⭐⭐⭐⭐ |
| **CVXOPT** | LP/SOCP/SDP/NL | **GPL** | - | - | **1.3.2 (Python 3.13)** | Python | ⭐⭐⭐⭐ |
| **CSDP** | SDP | COIN-OR (EPL推測) | - | - | 継続メンテナンス | C | ⭐⭐⭐ |

---

## 分析（Analysis / 足軽の意見）

### 領域別OSS充実度総括

#### 1. NLP（非線形計画）: OSS充実度 ⭐⭐⭐⭐⭐
- **Ipoptの存在**: 業界標準レベル。商用ソルバーと比較しても遜色ない局所最適化性能。
- **NLoptの柔軟性**: 統一インターフェースで多様なアルゴリズムを試せる。研究・プロトタイピングに最適。
- **CasADiの革新性**: 自動微分機能により、モデリングから最適化までシームレス。学術・産業で広く採用。
- **商用との差**: 局所最適化ではOSSで十分。大域最適化では商用BARON, KNITRO globalが優位だが、用途次第でOSSで対応可能。

#### 2. MINLP（混合整数非線形計画）: OSS充実度 ⭐⭐⭐⭐
- **SCIPの強さ**: 2026年2月最新版リリース、活発開発。Convex MINLPでは商用に次ぐ性能。
- **Bonmin/Couenneの役割**: 開発低調だが、特定用途（convex MINLP, global optimization）で依然有用。
- **商用との差**: Convex MINLPではギャップ縮小傾向（SCIPの進化）。Nonconvex/Globalでは商用BARON, KNITRO等が依然2桁程度優位。
- **実用判断**: 中規模convex MINLPならSCIPで十分。Large-scale/nonconvexは商用検討。

#### 3. QP / QCQP（二次計画）: OSS充実度 ⭐⭐⭐⭐⭐
- **OSQPの台頭**: Apache 2.0, 活発開発、ベンチマークで商用を上回る結果。QP分野でOSSの勝利。
- **HiGHSの汎用性**: QPも対応、MIT License、GPU対応。線形/二次両対応で利便性高い。
- **QCQP課題**: QCQPはOSS選択肢が限定的。商用Gurobi, MOSEK, CPLEXが優位。
- **推奨**: QPならOSQP/HiGHSで十分。QCQPは商用検討。

#### 4. SOCP / SDP（錐計画/半正定値計画）: OSS充実度 ⭐⭐⭐⭐
- **SCS/ECOSの実績**: CVXPYデフォルトソルバー、活発開発、広く採用。
- **CVXOPTの安定性**: 長年の実績、Python生態系で標準的存在。
- **商用との差**: 中規模問題ではOSSで十分。大規模・高精度ではMOSEK等商用が優位。
- **精度 vs 速度**: SCSは大規模問題を控えめ精度で高速処理（first-order method）。商用は高精度・高速両立。
- **推奨**: 中規模SOCP/SDPならOSSで対応可能。大規模・高精度要求なら商用MOSEK検討。

### ライセンス戦略

#### 最も自由度の高い選択肢
- **MIT (SCS)**: 最も制約少ない、商用利用完全自由
- **Apache 2.0 (SCIP, OSQP)**: MIT同等の自由度、特許条項あり

#### 弱いコピーレフト
- **EPL (Ipopt, Bonmin, Couenne)**: 実用上問題少ない、COIN-OR標準ライセンス

#### 強いコピーレフト（注意）
- **GPL (CVXOPT, ECOS)**: 派生物もGPL化必要。商用製品組み込みは要検討。

### 開発活発さのトレンド（2026年時点）

#### 活発開発（推奨）
- **SCIP**: 2026年2月最新版、MINLP分野のOSS筆頭
- **OSQP**: 2025年11月複数リポジトリ更新、QP分野の標準
- **SCS**: 2024年10月リリース、conic solver筆頭
- **ECOS**: 2024年6月リリース、SOCP組込み向け
- **CVXOPT**: 2026年Python 3.13対応、安定継続

#### 安定だが低調（用途次第）
- **Bonmin**: 2023年最終リリース、convex MINLP限定で有用
- **Couenne**: 開発低調、global optimization特化で依然価値あり
- **Ipopt**: 大きな変更なし、成熟製品として安定

### 推奨選定基準（2026年版）

| 用途 | 推奨OSSソルバー | 理由 | 商用検討ライン |
|-----|----------------|------|---------------|
| **NLP局所最適化** | Ipopt, NLopt | 業界標準、商用と遜色なし | 超大規模・高速要求 |
| **NLP大域最適化** | NLopt (global), Couenne | OSS選択肢限定的 | **BARON, KNITRO推奨** |
| **Convex MINLP** | SCIP | 活発開発、商用に次ぐ性能 | 大規模・高速要求 |
| **Nonconvex MINLP** | SCIP, Bonmin | ヒューリスティック、限界あり | **BARON, KNITRO推奨** |
| **Global MINLP** | Couenne, SCIP | OSS選択肢限定的 | **BARON推奨** |
| **QP** | OSQP, HiGHS | 商用と競合、活発開発 | QCQPは商用（Gurobi） |
| **SOCP** | SCS, ECOS | 標準的選択肢、実績豊富 | 大規模・高精度ならMOSEK |
| **SDP** | SCS, CVXOPT, CSDP | 中規模で十分 | 大規模・高精度ならMOSEK |
| **モデリング/AD** | CasADi | OSS最強フレームワーク | 不要（OSSで十分） |

### 商用ソルバーとのギャップ

#### 線形/凸問題: ギャップ縮小傾向
- **LP/MIP**: HiGHSにより約1桁差に縮小（Gurobi比）
- **QP**: OSQPは商用を上回る結果も
- **SOCP/SDP**: 中規模では遜色なし

#### 非線形/非凸問題: 依然ギャップ大
- **Nonconvex NLP**: 商用KNITRO, SNOPTが優位
- **Nonconvex MINLP**: 商用BARON, KNITROが約2桁優位
- **Global optimization**: 商用BARONがほぼ独占状態

#### 性能以外の差
- **サポート**: 商用は専門サポートあり、OSSはコミュニティ依存
- **安定性**: 商用は厳格なテスト、OSSはプロジェクト次第
- **スケーラビリティ**: 商用は超大規模問題に強い

### 2026年の注目トレンド

1. **SCIP 10.0系の進化**: MINLP性能向上、Benders対応、CONOPT統合
2. **OSQP/SCS/ECOSの継続成長**: Conic/QP分野でOSSが商用に迫る
3. **GPU対応の広がり**: HiGHS PDHG, cuOSQP等、今後の加速期待
4. **Python統合の深化**: CVXOPT, SCS, ECOS等、Python 3.13対応
5. **低調プロジェクトの固定化**: Bonmin, Couenneは開発停滞だが安定利用可能

### 総括

- **NLP局所最適化、QP、SOCP**: OSSで商用と同等以上の選択が可能
- **Convex MINLP**: OSS（特にSCIP）で実用レベル、商用との差は縮小傾向
- **Nonconvex/Global最適化**: 商用ソルバー（BARON, KNITRO）が依然優位、OSSは限定的
- **ライセンス**: MIT/Apache 2.0のOSS増加（SCS, OSQP, SCIP）、商用利用の自由度向上
- **開発活発さ**: SCIP, OSQP, SCS等の活発開発が頼もしい。Bonmin/Couenneは用途次第で選択。

**殿への一言**: 非線形OSS分野は線形ほど成熟していないが、局所最適化・凸問題では十分実用的。大域最適化・非凸MINLPは商用ソルバーへの投資価値あり。用途を明確にして選定すべし。
