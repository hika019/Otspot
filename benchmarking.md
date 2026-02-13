# ソルバー性能ベンチマーク手法調査

## 1. 標準テストセット

### 1.1 MIPLIB（Mixed Integer Programming Library）

**■ 事実（情報源付き）:**

- **歴史**: 1992年にRobert E. Bixby、E.A. Boyd、R.R. Indivinaらによって初めてリリースされた電子利用可能な純粋整数および混合整数計画問題のライブラリ ([Benchmarking Optimization Software with Performance Profiles](https://arxiv.org/pdf/cs/0102001))
- **最新バージョン**: MIPLIB 2017（第6版）が最新の公開バージョン。初期プール5,721インスタンスから1,065インスタンスを選定し、そのうち240インスタンスがベンチマーク専用のサブセット「Benchmark Set」として特別選定されている ([MIPLIB 2017 – The Mixed Integer Programming Library](https://miplib.zib.de/), [MIPLIB 2017: Data-Driven Compilation of the 6th Mixed-Integer Programming Library](https://link.springer.com/article/10.1007/s12532-020-00194-3))
- **今後の展開**: 2024年にMIPLIB 2024への投稿受付が開始されたことが確認されている ([MIPLIB 2024 Submission Now Open](https://forum.gams.com/t/miplib-2024-submission-now-open/7222))
- **選定基準**: Benchmark Setのインスタンスは、今日のソルバーで（和集合として）解けること、および数値安定性などの制約条件を考慮して選ばれている ([MIPLIB 2017 – The Mixed Integer Programming Library](https://miplib.zib.de/))
- **使われ方**: 混合整数最適化ソルバーの性能比較のための標準テストセットとして広く使用されている ([MIPLIB 2017 – The Mixed Integer Programming Library](https://miplib.zib.de/))

**■ 足軽の意見/分析:**

MIPLIB 2017は現在の実質的な標準であり、ベンチマーク専用の240インスタンスに絞ることで、ソルバー間の公平な比較が可能になっている。2024版への移行が進行中である可能性が高いが、2026年2月時点では2017版が主流と推定される。

---

### 1.2 Netlib（Linear Programming Test Set）

**■ 事実（情報源付き）:**

- **概要**: 様々なソースから集められた実生活のLinear Programming（線形計画）問題のコレクション ([The NETLIB LP Test Problem Set](https://www.numerical.rl.ac.uk/cute/netlib.html))
- **形式**: MPS形式（SIF形式のサブセット）で提供されており、MPC圧縮ユーティリティで圧縮されている。同ディレクトリ内のEMPSで解凍する必要がある ([The NETLIB LP Test Problem Set](https://www.numerical.rl.ac.uk/cute/netlib.html))
- **規模**: ベンチマーク利用例では90問のテストセットが使われており、問題サイズは数百～数千変数で、最大はN=13,525変数、M=3,000制約 ([Benchmarking ALGLIB and other LP solvers](https://www.alglib.net/linear-programming/benchmark.php))
- **アクセス先**: https://www.netlib.org/lp/ および https://www.cuter.rl.ac.uk/Problems/netlib.html ([netlib/lp](https://www.netlib.org/lp/), [The NETLIB LP Test Problem Set](https://www.cuter.rl.ac.uk/Problems/netlib.html))

**■ 足軽の意見/分析:**

NetlibはLP専用テストセットとして古典的かつ実用的。MIPLIBと異なり純粋な線形計画問題に特化しているため、LPソルバー部分のベンチマークに有用。

---

### 1.3 その他の標準テストセット

**■ 事実（情報源付き）:**

- **Hans Mittelmannベンチマーク**: Arizona State UniversityのHans D. Mittelmann教授がplato.asu.edu/bench.htmlで維持している独立ベンチマーク ([Decison Tree for Optimization Software](https://plato.asu.edu/bench.html), [Hans D Mittelmann](https://plato.asu.edu/))
  - 歴史的にCPLEX、Gurobi、XPRESSを含んでいたが、2018年INFORMS年次会合でGurobiの行動により、IBMとFICOが自社ソルバーの結果削除を要求。2024年8月にGurobi、2024年12月にMindOptも撤退 ([Decison Tree for Optimization Software](https://plato.asu.edu/bench.html))
  - カバー範囲: Simplexおよび並列LPソルバー、SDPコード、MISOCPおよび大規模SOCP問題、MILPベンチマークなど広範なカテゴリ ([Decison Tree for Optimization Software](https://plato.asu.edu/bench.html))

**■ 足軽の意見/分析:**

Mittelmannベンチマークは独立性・公平性で高く評価されてきたが、2024年に主要商用ソルバーが相次いで撤退した点は注目に値する。今後はオープンソースソルバー中心の比較となる可能性がある。

---

## 2. 評価指標

### 2.1 主要メトリクス

**■ 事実（情報源付き）:**

- **解答時間（Solving Time）**: 最も基本的な性能指標。問題インスタンスを解くのにかかった時間 ([Benchmarking Optimization Software with Performance Profiles](https://arxiv.org/pdf/cs/0102001))
- **ギャップ（Optimality Gap）**: 現在の上界と下界の差。最適性ギャップは「最良解と最良境界の絶対差÷最良解の絶対値」で計算され、ギャップがゼロのとき最適性が証明される ([MIP Models - Gurobi Optimization](https://www.gurobi.com/documentation/9.5/refman/mip_models.html), [Mixed-Integer Programming (MIP/MILP) – A Primer on the Basics](https://www.gurobi.com/resources/mixed-integer-programming-mip-a-primer-on-the-basics/))
- **ノード数（Branch-and-Bound Nodes）**: 分枝限定法で探索された枝刈りノードの数。NodeLimitパラメータで上限設定可能 ([Terminating MIP Optimization](https://www-eio.upc.es/lceio/manuals/cplex-11/html/usrcplex/solveMIP10.html), [Classes — Python-MIP documentation](https://python-mip.readthedocs.io/en/latest/classes.html))
- **メモリ使用量（Memory Usage）**: GurobiではMemLimitパラメータでメモリ上限を設定でき、超過するとOUT_OF_MEMORYエラーで中断。深さ優先探索は未探索ノードを少なく保つためメモリ枯渇リスクが低い ([MIP Models - Gurobi Optimization](https://www.gurobi.com/documentation/9.5/refman/mip_models.html))

**■ 足軽の意見/分析:**

これら4指標は独立ではなくトレードオフ関係にある。例えば深さ優先探索はメモリを節約するがノード数が増える可能性がある。ベンチマークでは通常、解答時間を主指標とし、他はサブ指標として報告される。

---

### 2.2 Shifted Geometric Mean

**■ 事実（情報源付き）:**

- **定義**: n個の数の幾何平均はそれらの積のn乗根。Shifted geometric meanでは、各数に正のシフト値sを加えてから掛け合わせ、n乗根を取った後にsを引く ([arXiv:2302.01164v1](https://arxiv.org/pdf/2302.01164))
- **MIPにおける標準**: 計算MIPでは実行時間のshifted geometric meanが標準的な性能指標 ([arXiv:2302.01164v1](https://arxiv.org/pdf/2302.01164))
- **利点**: 非常に大きな外れ値（算術平均の弱点）にも非常に小さな外れ値（幾何平均の弱点）にも妥協しない ([arXiv:2302.01164v1](https://arxiv.org/pdf/2302.01164))
- **解釈**: Shifted-geometric-mean実行時間Yのソルバーは、テストセット全体で最速ソルバーよりY倍遅い。通常、最速ソルバーを1.0にスケールして比較する ([Visualizations of Mittelmann benchmarks](https://mattmilten.github.io/mittelmann-plots/))
- **シフト値**: Mittelmannベンチマークではs=10が使用されている ([The State-of-the-Art in Optimization Software](https://plato.asu.edu/talks/ismp2015.pdf), [qpbenchmark](https://pypi.org/project/qpbenchmark/))

**■ 足軽の意見/分析:**

シフト値s=10は実務的な標準であり、非常に高速（<1秒）と非常に低速（>1時間）の両極端な問題がテストセットに含まれる場合のバランスを取るのに適している。自作ソルバーのベンチマーク報告時にはこの指標を採用すべき。

---

## 3. 他ソルバーとの比較方法

### 3.1 Performance Profiles

**■ 事実（情報源付き）:**

- **概要**: 複数のソルバーを大規模テストセットで実行した際の性能指標の分布関数として機能するツール。最適化ソフトウェアのベンチマークおよび評価に広く使用されている ([A Note on Performance Profiles for Benchmarking Software](https://dl.acm.org/doi/10.1145/2950048), [Benchmarking Optimization Software with Performance Profiles](https://arxiv.org/abs/cs/0102001))
- **歴史**: Elizabeth DolanとJorge Moréによる2002年の論文「Benchmarking Optimization Software with Performance Profiles」（Mathematical Programming掲載）で広まった。Google Scholarで約5,000件の引用 ([A Note on Performance Profiles for Benchmarking Software](https://centaur.reading.ac.uk/74694/1/perform_toms.pdf))
- **仕組み**: 定義されたテスト集合に対して複数のソルバーを実行し、選択したメトリック（例: CPU時間）を測定する。選択されたコストメトリックは正の値で、小さい値ほど性能が良い ([Benchmarking Optimization Software with Performance Profiles](https://arxiv.org/pdf/cs/0102001))
- **利点**: 多様なインスタンス集合における複数アルゴリズムの性能を包括的に比較でき、解釈しやすいグラフィカル表現を提供 ([Benchmarking Optimization Software with Performance Profiles](https://ftp.mcs.anl.gov/pub/tech_reports/reports/P861.pdf))
- **注意点**: Performance profilesを用いてソルバーの相対性能を評価する際には解釈に注意が必要 ([Nested Performance Profiles for Benchmarking Software](https://arxiv.org/abs/1809.06270))
- **ツール**: Julia言語ではBenchmarkProfiles.jlおよびSolverBenchmark.jlがperformance profiles作成用の効果的なツールとして利用可能 ([Performance Profile Benchmarking Tool](https://tmigot.github.io/posts/2024/06/teaching/))

**■ 足軽の意見/分析:**

Performance profilesは視覚的に直感的であり、論文発表時の標準手法として推奨される。ただし単一のグラフで全てを語れるわけではなく、問題カテゴリ別のサブプロットや補足的な表も併用すべき。

---

### 3.2 Mittelmannベンチマークの活用

**■ 事実（情報源付き）:**

- **サイト**: plato.asu.edu/bench.htmlで継続的に更新されている ([Decison Tree for Optimization Software](https://plato.asu.edu/bench.html))
- **可視化ツール**: Matt Miltenによるインタラクティブチャートがあり、ベンチマーク内の各ソルバー間のペアワイズ実行時間要因を全インスタンスについて表示可能 ([Visualizations of Mittelmann benchmarks](https://mattmilten.github.io/mittelmann-plots/))
- **現状**: 2024年に主要商用ソルバー（Gurobi、CPLEX、XPRESS、MindOpt）が相次いで撤退 ([Decison Tree for Optimization Software](https://plato.asu.edu/bench.html))

**■ 足軽の意見/分析:**

商用ソルバー撤退後も、オープンソースソルバー（HiGHS、SCIP等）との比較には引き続き有用。自作ソルバーをMittelmannベンチマークに投稿することは、第三者評価としての信頼性を得る有力な手段となる。

---

### 3.3 公平な比較のための条件設定

**■ 事実（情報源付き）:**

- **ハードウェア統一**: 異なるベンチマークや異なる最適化定式化は異なる結果をもたらす可能性がある。同一ハードウェア環境での実行が重要 ([🧠 MIP Solvers Unleashed](https://medium.com/operations-research-bit/mip-solvers-unleashed-a-beginners-guide-to-pulp-cplex-gurobi-google-or-tools-and-pyomo-0150d4bd3999))
- **時間制限**: 一般的にはインスタンスごとに時間制限（例: 1時間、2時間）を設定し、その範囲内での性能を比較
- **パラメータ設定**: デフォルトパラメータでの比較が原則。特殊なチューニングを施す場合はその旨を明記 ([Benchmarks for Current Linear and Mixed Integer Optimization Solvers](https://www.researchgate.net/publication/288179452_Benchmarks_for_Current_Linear_and_Mixed_Integer_Optimization_Solvers))
- **バイアス回避**: LP/MILPソルバーの研究では、バイアスなく測定する方法と結果の解釈方法が重要 ([Benchmarks for Current Linear and Mixed Integer Optimization Solvers](https://ideas.repec.org/a/mup/actaun/actaun_2015063061923.html))

**■ 足軽の意見/分析:**

「公平な比較」の定義自体が難しい。例えばGurobiが有利な問題セット選定をすれば、それは不公平とみなされる（実際に過去に議論があった）。透明性を確保するため、テストセット選定理由、ハードウェアスペック、パラメータ設定を全て公開すべき。

---

## 4. 自作ソルバーをベンチマークする場合の実践的手順

**■ 事実（情報源付き）:**

- **標準テストセットの使用**: MIPLIB 2017のBenchmark Set（240インスタンス）を使用することが推奨される。これにより既存研究との直接比較が可能 ([MIPLIB 2017 – The Mixed Integer Programming Library](https://miplib.zib.de/))
- **ベンチマークリソース**: Python-MIPドキュメントでは、LPfeas（PD実行可能点探索）、LPopt（最適基底解探索）、Large Network-LP、MILP（MIPLIB2017使用）などのベンチマークが利用可能 ([Benchmarks — Python-MIP documentation](https://python-mip.readthedocs.io/en/latest/bench.html))
- **カスタムソルバー開発**: ケーススタディでは、特定の問題クラス向けにソルバーフレームワークを使用してカスタムソルバーを開発し、ブラックボックスソルバーとのベンチマークを行う例が紹介されている ([Decison Tree for Optimization Software](https://plato.asu.edu/bench.html))

**■ 足軽の意見/分析:**

自作ソルバーのベンチマーク実践手順として、以下のステップを推奨する:

1. **初期検証**: 小規模問題（Netlib等）で正しさを確認
2. **性能測定**: MIPLIB 2017 Benchmark Set（または一部サブセット）で時間/ノード数/メモリを記録
3. **比較対象選定**: 同クラスのソルバー（例: オープンソース同士）を選び、同一環境で実行
4. **メトリクス計算**: Shifted geometric mean (s=10)を計算し、最速ソルバーを1.0として相対比を算出
5. **Performance Profile作成**: JuliaのBenchmarkProfiles.jl等を使い視覚化
6. **結果報告**:
   - テストセット詳細（インスタンス数、選定基準）
   - ハードウェア仕様（CPU、メモリ、OS）
   - ソルバーバージョンおよびパラメータ設定
   - 生データ（各インスタンスの時間/ギャップ/ノード数）の付録掲載
7. **再現性確保**: ベンチマークスクリプトとログをGitHub等で公開

この手順により、第三者による検証可能性と学術的信頼性を担保できる。

---

## 情報源一覧（Sources）

- [MIPLIB 2017 – The Mixed Integer Programming Library](https://miplib.zib.de/)
- [MIPLIB 2017: Data-Driven Compilation of the 6th Mixed-Integer Programming Library](https://link.springer.com/article/10.1007/s12532-020-00194-3)
- [MIPLIB 2024 Submission Now Open](https://forum.gams.com/t/miplib-2024-submission-now-open/7222)
- [netlib/lp](https://www.netlib.org/lp/)
- [The NETLIB LP Test Problem Set](https://www.numerical.rl.ac.uk/cute/netlib.html)
- [Benchmarking ALGLIB and other LP solvers](https://www.alglib.net/linear-programming/benchmark.php)
- [Decison Tree for Optimization Software](https://plato.asu.edu/bench.html)
- [Hans D Mittelmann](https://plato.asu.edu/)
- [A Note on Performance Profiles for Benchmarking Software](https://dl.acm.org/doi/10.1145/2950048)
- [Benchmarking Optimization Software with Performance Profiles](https://arxiv.org/pdf/cs/0102001)
- [Visualizations of Mittelmann benchmarks](https://mattmilten.github.io/mittelmann-plots/)
- [arXiv:2302.01164v1](https://arxiv.org/pdf/2302.01164)
- [The State-of-the-Art in Optimization Software](https://plato.asu.edu/talks/ismp2015.pdf)
- [qpbenchmark](https://pypi.org/project/qpbenchmark/)
- [MIP Models - Gurobi Optimization](https://www.gurobi.com/documentation/9.5/refman/mip_models.html)
- [Mixed-Integer Programming (MIP/MILP) – A Primer on the Basics](https://www.gurobi.com/resources/mixed-integer-programming-mip-a-primer-on-the-basics/)
- [Classes — Python-MIP documentation](https://python-mip.readthedocs.io/en/latest/classes.html)
- [Terminating MIP Optimization](https://www-eio.upc.es/lceio/manuals/cplex-11/html/usrcplex/solveMIP10.html)
- [Benchmarks — Python-MIP documentation](https://python-mip.readthedocs.io/en/latest/bench.html)
- [Benchmarks for Current Linear and Mixed Integer Optimization Solvers](https://www.researchgate.net/publication/288179452_Benchmarks_for_Current_Linear_and_Mixed_Integer_Optimization_Solvers)
- [Performance Profile Benchmarking Tool](https://tmigot.github.io/posts/2024/06/teaching/)
- [Nested Performance Profiles for Benchmarking Software](https://arxiv.org/abs/1809.06270)
- [🧠 MIP Solvers Unleashed](https://medium.com/operations-research-bit/mip-solvers-unleashed-a-beginners-guide-to-pulp-cplex-gurobi-google-or-tools-and-pyomo-0150d4bd3999)
