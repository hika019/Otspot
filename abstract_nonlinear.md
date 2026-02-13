# 非線形ソルバー調査アブストラクト

作成日: 2026-02-12
作成者: ashigaru2
プロジェクト: solver (Phase 1 市場調査)
入力: nonlinear_oss_solvers.md, nonlinear_commercial_solvers.md

---

## 1. 概要

本調査は非線形最適化ソルバー市場の現状把握を目的とし、OSS（オープンソース）11種および商用12種の計23種の非線形ソルバーを調査した。対象問題種別はNLP（非線形計画）、MINLP（混合整数非線形計画）、QP/QCQP（二次計画）、SOCP（二次錐計画）、SDP（半正定値計画）の5領域。OSS・商用それぞれの技術的優位性、性能差、ライセンス形態、開発活発さを分析し、領域別の参入余地と競争状況を評価する。

---

## 2. 主要知見（事実・ソース付き）

### 2.1 非線形OSSソルバーの現状と主要プレイヤー

#### NLP（非線形計画）
- **Ipopt**: COIN-OR, EPL, 内点法。業界標準レベルの局所最適化性能。並列線形ソルバー経由で並列化対応（MKL Pardiso, HSL MA86/MA97, SPRAL, MUMPS）。
  - 出典: [Ipopt Documentation](https://coin-or.github.io/Ipopt/), [Exploring Benefits of Linear Solver Parallelism (arXiv)](https://arxiv.org/abs/1909.08104)
- **NLopt**: 統一インターフェースで複数free/open-source非線形最適化ライブラリをラップ。多言語対応（C, C++, Fortran, MATLAB, Python, Java, Julia, R, Lua, OCaml, Rust, Crystal）。
  - 出典: [NLopt Documentation](https://nlopt.readthedocs.io/)
- **CasADi**: フレームワーク/モデリングツール。自動微分（AD）とシンボリック式処理。外部ソルバー統合（IPOPT, BONMIN, KNITRO, SNOPT等）。Python/MATLAB/Octaveインターフェース。
  - 出典: [CasADi](https://web.casadi.org/), [CasADi Paper](https://link.springer.com/article/10.1007/s12532-018-0139-4)

#### MINLP（混合整数非線形計画）
- **SCIP**: Apache 2.0/LGPL。Convex/nonconvex両対応。**最新リリース: 10.0.1 (2026年2月3日)**。非常に活発な開発。
  - 出典: [SCIP GitHub Releases](https://github.com/scipopt/scip/releases), [SCIP Suite 10.0 Paper](https://arxiv.org/html/2511.18580v1)
- **Bonmin**: EPL。Convex問題に厳密解、nonconvex問題にヒューリスティック。GitHub Stars 141、最新リリース1.8.9 (2023年1月)。近年開発低調。
  - 出典: [Bonmin GitHub](https://github.com/coin-or/Bonmin)
- **Couenne**: EPL。**Global optimization（大域最適化）特化**。非凸MINLPの大域最適解を探索。GitHub Stars 83。近年開発低調。
  - 出典: [Couenne](https://www.coin-or.org/Couenne/), [Couenne GitHub](https://github.com/coin-or/Couenne)

#### QP（二次計画）
- **OSQP**: Apache 2.0。ADMM (Alternating Direction Method of Multipliers) ベース。**2025年複数リポジトリで更新**（qdldl: 2025年11月、osqp_benchmarks: 2025年11月、osqp.rs: 2025年4月）。ベンチマークで多くの商用/学術ソルバーを上回る性能。CUDA実装（cuosqp）あり。
  - 出典: [OSQP](https://osqp.org/), [OSQP GitHub](https://github.com/osqp), [cuosqp GitHub](https://github.com/osqp/cuosqp)
- **HiGHS**: MIT License。QP対応、線形ソルバーとしても優秀。GPU対応。2026年も活発。
  - 出典: 既存調査（linear_oss_solvers.md参照）

#### SOCP / SDP（錐計画 / 半正定値計画）
- **SCS**: **MIT License**。LP/SOCP/SDP/ECP/PCP対応。**GitHub Stars 601**。現行バージョン3.2.11 (2024年10月リリース)。大規模問題向け、精度控えめだが高速（first-order method）。
  - 出典: [SCS Documentation](https://www.cvxgrp.org/scs/), [SCS GitHub](https://github.com/cvxgrp/scs)
- **ECOS**: GPL v3.0。SOCP特化。組込みアプリケーション向け（ANSI-C, low footprint）。**GitHub Stars 465**。最新リリースv2.0.14 (2024年6月)。CVXPYのデフォルトソルバーの一つ。
  - 出典: [ECOS GitHub](https://github.com/embotech/ecos), [ecos-python Releases](https://github.com/embotech/ecos-python/releases)
- **CVXOPT**: GPL。LP/SOCP/SDP/Nonlinear convex optimization。**最新版1.3.2（Python 3.13サポート）**。Pythonネイティブ。
  - 出典: [CVXOPT PyPI](https://pypi.org/project/cvxopt/)
- **CSDP**: COIN-OR（EPL推測）。SDP特化。C実装。BLAS/LAPACK依存。SageやCVXPYから利用可能。
  - 出典: [CSDP GitHub](https://github.com/coin-or/Csdp), [CSDP Sage Documentation](https://doc.sagemath.org/html/en/reference/spkg/csdp.html)

### 2.2 非線形商用ソルバーの優位性と技術的核心

#### 大域最適化（Global Optimization）
- **BARON**: 決定論的大域最適化を**保証**する唯一の商用MINLPソルバー。Branch-and-reduce法（分枝限定＋制約伝播＋区間解析＋凸緩和＋領域削減）。30年の学術研究（INFORMS Computing Society Prize、Beale-Orchard-Hays Prize受賞）。2025年に9回アップデート。
  - 出典: [BARON Solver](https://www.minlp.com/baron-solver), [The Year 2025 for GAMS Solvers](https://www.gams.com/blog/2026/01/the-year-2025-for-gams-solvers/)
- **Gurobi v9.0以降**: 非凸QP/QCQP大域最適化。v12で非凸MIQCP劇的高速化。多くの非線形ソルバーが局所最適のみ提供する中、大域最適解を求める。
  - 出典: [Non-Convex Quadratic Optimization - Gurobi](https://www.gurobi.com/events/non-convex-quadratic-optimization/)

#### 局所最適化の性能優位性
- **KNITRO**: NLP/MINLP専門特化。3つのアルゴリズム（内点法/active-set/拡大ラグランジアン）選択可能。v14で凸MINLP 26%高速化。v13で並列分枝限定（4スレッドで5倍高速化）。2023年ベンチマークでBARONより**CPU時間2桁高速**。
  - 出典: [Artelys Knitro 14](https://www.artelys.com/news/artelys-knitro-14-new-release-nonlinear-optimization-solver/), [Comparative Analysis of Nonlinear Programming Solvers](https://www.mdpi.com/2504-446X/7/8/487)
- **SNOPT**: 大規模疎NLP向けSQP法。関数・勾配評価回数が少ない（高価な関数評価に有効）。2023年ベンチマークで収束率はKNITRO比やや低いが、**著しく高速**。
  - 出典: [SNOPT Paper](https://www.ccom.ucsd.edu/~peg/papers/snpaper.pdf), [Comparative Analysis](https://www.mdpi.com/2504-446X/7/8/487)

#### 錐最適化の専門性
- **MOSEK**: SOCP/SDP/指数錐/累乗錐の圧倒的専門性。同次内点法（homogeneous interior-point method）で連続LP/QP/錐問題最先端。v11（2026年2月リリース）で幾何計画、エントロピー最適化を錐定式化で効率的に解決。大規模問題でGurobi上回る結果（MINLP研究）。
  - 出典: [MOSEK - Wikipedia](https://en.wikipedia.org/wiki/MOSEK), [arXiv - MINLP Study](https://arxiv.org/pdf/2303.04216)

#### GPU対応（限定的）
- **Gurobi**: Linux限定、PDHG (Primal-Dual Hybrid Gradient)、NVIDIA H100推奨。
- **FICO Xpress**: Windows/Linux/ARM64対応。v9.8ベータ版。大規模LPで**最大50倍高速化**。
  - 出典: [FICO Xpress Solver](https://www.fico.com/en/latest-thinking/solution-sheet/fico-xpress-solver)

### 2.3 OSSと商用の差（どの領域が大きいか）

#### 差が小さい領域（OSS競合レベル）
- **NLP局所最適化**: Ipoptは商用と遜色ない。NAGがIPOPTを統合採用している事実が証左。
  - 出典: [naginterfaces.library.opt.handle_solve_ipopt](https://support.nag.com/numeric/py/nagdoc_latest/naginterfaces.library.opt.handle_solve_ipopt.html)
- **QP**: OSQPは商用（Gurobi, MOSEK）と同等以上の性能（ベンチマーク結果）。
  - 出典: [OSQP](https://osqp.org/)
- **SOCP**: SCS/ECOSは商用と競合。CVXPYのデフォルトソルバー。中規模問題では遜色なし。
  - 出典: [SCS Documentation](https://www.cvxgrp.org/scs/)

#### 差が中程度の領域
- **Convex MINLP**: SCIPは商用に次ぐ性能（2018年レビューで評価）。ただし商用KNITRO/BARONとは性能差あり。
  - 出典: [A Review and Comparison of Solvers for Convex MINLP](https://egon.cheme.cmu.edu/Papers/ConvexMINLPReview2018_OO.pdf)
- **SDP**: 中規模問題ではOSSで十分。大規模・高精度要求ではMOSEK等商用が優位。

#### 差が大きい領域（商用優位、約2桁差）
- **Nonconvex MINLP local**: Bonminはヒューリスティックのみ。商用KNITRO/BARONが優位。
- **Nonconvex MINLP global**: Couenneは有用だが、商用BARONが決定論的大域最適保証で圧倒。
- **一般NLP大域最適化**: 商用BARON, KNITRO globalが優位。OSS選択肢は限定的（Couenne, NLopt global）。
  - 出典: OSS調査比較表、商用調査ベンチマーク総括

### 2.4 各問題種別のOSS充実度

| 問題種別 | OSS充実度 | 主要OSSソルバー | 商用との差 |
|---------|----------|---------------|----------|
| **NLP局所最適化** | ⭐⭐⭐⭐⭐ | Ipopt, NLopt, CasADi | 小（商用と遜色なし） |
| **NLP大域最適化** | ⭐⭐ | NLopt (global), Couenne | 大（商用BARON優位） |
| **Convex MINLP** | ⭐⭐⭐⭐ | SCIP | 中（商用に次ぐ性能） |
| **Nonconvex MINLP** | ⭐⭐⭐ | SCIP, Bonmin | 大（商用2桁優位） |
| **Global MINLP** | ⭐⭐ | Couenne, SCIP | 大（BARON保証、2桁優位） |
| **QP** | ⭐⭐⭐⭐⭐ | OSQP, HiGHS | 小（商用と同等以上） |
| **QCQP** | ⭐⭐ | ALGLIB等（選択肢限定） | 大（商用優位） |
| **SOCP** | ⭐⭐⭐⭐ | SCS, ECOS | 小～中（中規模で十分） |
| **SDP** | ⭐⭐⭐⭐ | SCS, CVXOPT, CSDP | 中（大規模・高精度は商用） |

出典: OSS調査「領域別OSS充実度総括」、商用調査「商用 vs OSS の性能差」

---

## 3. 分析・意見（足軽/家老の見解）

### 3.1 非線形領域の競争状況評価

#### 線形と非線形の構造的差異
線形最適化（LP/MIP）ではHiGHSの登場によりOSSが商用に約1桁差まで迫った（Gurobi比）。しかし非線形領域では状況が異なる：

1. **凸問題**: OSS充実（NLP局所最適化、QP、SOCP）。商用との差は小～中。
2. **非凸問題**: 商用が圧倒（Nonconvex MINLP、Global optimization）。約2桁性能差。
3. **大域最適化**: BARONが30年の学術研究で決定論的保証を実現。OSSでは代替困難。

#### 開発活発さの二極化
- **活発開発（OSS）**: SCIP (2026年2月), OSQP (2025年11月), SCS (2024年10月), ECOS (2024年6月), CVXOPT (Python 3.13対応)
- **低調開発（OSS）**: Bonmin (2023年最終), Couenne (リリース情報なし)
- **商用継続投資**: KNITRO (v13→v14で31%累積高速化), BARON (2025年9回更新), Xpress (5.3倍MINLP高速化), CONOPT (2026年共役勾配改善)

線形ソルバーと異なり、非線形OSSプロジェクトは商用並みの継続投資が得られていない領域（MINLP特に）がある。

#### ライセンス戦略の進化
OSSライセンスがビジネスフレンドリーに進化:
- **MIT/Apache 2.0**: SCS (MIT), OSQP (Apache 2.0), SCIP (Apache 2.0/LGPL), HiGHS (MIT)
- **従来EPL**: Ipopt, Bonmin, Couenne（弱いコピーレフト、実用上問題少）
- **GPL注意**: CVXOPT, ECOS（派生物GPL化必要、商用製品組込み要検討）

商用ソルバーは全て学術ライセンス無料または大幅割引（KNITRO, Gurobi, CPLEX, BARON等）。学術研究では商用を無料利用できるため、OSSの優位性は「商用利用の自由度」に集約される。

### 3.2 参入余地が大きい領域

#### 高優先度: Nonconvex MINLP
**理由**:
1. **需要**: 多数の産業応用（エネルギー最適化、スケジューリング、設備配置、投資計画）
2. **商用独占**: KNITRO/BARON が実質的独占（約2桁性能差）
3. **OSS限界**: Bonmin/Couenne開発低調。SCIPは改善継続だが商用に劣後
4. **参入障壁**: アルゴリズム複雑性高いが、30年研究蓄積あり（公開論文、COIN-ORコード）
5. **市場規模**: BARON顧客1,000超（Fortune 500含む）、KNITROは数百サイト

**戦略オプション**:
- SCIP拡張（Apache 2.0で商用利用可、既存基盤活用）
- Bonmin/Couenneモダナイゼーション（EPLコード再構築、並列化、ヒューリスティクス追加）
- 新規実装（最新論文ベース、MIT/Apache 2.0、GPU対応）

#### 中優先度: Global NLP
**理由**:
1. **商用独占**: BARON が決定論的保証で独占。Gurobi は二次形式のみ。
2. **OSS限定**: Couenne/NLopt global（性能・保証で劣後）
3. **需要**: 金融（投資最適化）、エネルギー、材料設計で大域最適解必須
4. **参入障壁**: Branch-and-reduce、凸緩和、区間解析の高度実装必要

**リスク**: BARON の30年蓄積は非常に厚い。追従に長期投資必要。

#### 低優先度（既にOSS充実）
- **NLP局所最適化**: Ipopt/NLopt で十分。NAG がIPOPT採用。
- **QP**: OSQP が商用並み性能。HiGHS も対応。
- **SOCP/SDP**: SCS/ECOS/CVXOPT で中規模対応可。

#### QCQP（二次制約付き二次計画）
**現状**: OSS選択肢限定的。商用Gurobi/MOSEK/CPLEX優位。
**需要**: ポートフォリオ最適化、ロバスト最適化。
**参入余地**: 中程度。QP技術の延長だが、非凸QCQP は大域最適化必要。

### 3.3 リスクと不確実性

#### 技術リスク
1. **アルゴリズム複雑性**: Nonconvex MINLP/Global NLP は論文実装と商用製品の間に巨大な性能差（ヒューリスティクス、並列化、メモリ管理の洗練度）。
2. **ベンチマーク依存**: 商用ソルバーの性能主張は自社ベンチマーク中心。第三者比較研究は限定的（2023年MDPI研究、2018年CMU MINLP研究程度）。
3. **GPU加速未成熟**: Gurobi（Linux限定）、Xpress（ベータ版）。LP以外のGPU効果は未実証。

#### 市場リスク
1. **学術ライセンス無料**: 商用ソルバーが学術研究者に無料提供。OSS優位性は商用利用の自由度のみ。
2. **商用継続投資**: KNITRO v13→v14で31%高速化、BARON 2025年9回更新。OSS（特にBonmin/Couenne）は低調。
3. **サポート差**: 商用は専門サポート、厳格テスト、長期保証。OSSはコミュニティ依存。

#### 不確実性
1. **需要規模**: 非線形最適化市場規模の定量データ不足。BARON顧客1,000超、KNITRO数百サイトだが、売上非公開。
2. **OSS採用意欲**: 企業がOSSを選ぶ条件（性能何割差まで許容？サポート要否？）不明。
3. **GPU普及**: 非線形最適化のGPU加速は現状限定的（Xpressベータ版）。今後の加速は不確実。

#### 競合リスク
1. **商用買収**: GAMS が CONOPT買収（2024年）。商用企業がOSS脅威を買収で排除する可能性。
2. **Gurobi拡張**: Gurobi v12で非凸MIQCP大幅高速化。今後一般NLP対応の可能性（現状は二次形式限定）。
3. **SCIP進化**: Apache 2.0でSCIPが商用並み性能達成なら、新規参入の価値低下。

### 3.4 総括（足軽の意見）

**非線形最適化ソルバー市場は「凸/非凸」で二分される**:
- **凸領域（NLP局所、QP、SOCP、Convex MINLP）**: OSSが商用に迫る。参入余地小。
- **非凸領域（Nonconvex MINLP、Global NLP/MINLP）**: 商用独占、約2桁性能差。**参入余地大**。

**最大機会: Nonconvex MINLP**。需要大、商用独占、OSS低調。SCIP拡張またはBonmin/Couenneモダナイゼーションが現実的。MIT/Apache 2.0ライセンスで商用利用自由度を提供すれば差別化可能。

**最大障壁: 商用の継続投資**。KNITRO/BARON は30年蓄積＋年次改善継続。単発実装では追従困難。長期コミットメント＋コミュニティ育成必須。

**殿への具申**: 非線形参入は線形（HiGHS）より難易度高。Nonconvex MINLP 狙いなら、SCIP拡張（既存基盤活用）またはGPU特化新規実装（差別化）を推奨。市場規模検証（顧客ヒアリング）後に判断すべし。
