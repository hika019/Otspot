# 天下を取るための戦略調査 + OSSアピール方法

作成日: 2026-02-12
担当: ashigaru2
任務ID: subtask_011f

---

## 1. 既存OSSソルバーに勝つための戦略

### 1.1 HiGHSの弱点（事実）

**性能差**
- LP問題において、HiGHSは最高の商用ソルバー(COPT)より20倍遅い [1]
- MIP問題では、Gurobiとの間に約1桁（10倍）の性能差がある [1]
- 大規模な実問題では、Gurobiに対して60-100倍遅くなる場合もある [5]

**並列化の制限**
- 基本的にシングルスレッド動作 [1]
- MIP求解では一部のコンポーネントのみマルチコア活用可能 [1]
- 新しいHiPO内点法ソルバーはマルチスレッド対応だが、メモリ使用量増加と精度低下のトレードオフがある [1]

**API・エコシステム**
- 商用ソルバーに比べてインターフェースの洗練度が低い [2]

### 1.2 SCIPの弱点（事実）

**アーキテクチャ上の制約**
- 内蔵LPソルバーを持たず、外部LPソルバーとのインターフェース経由で動作 [1]
- 外部LPソルバーとの通信コストが高く、一部情報にアクセスできない [1]
- primal/dual simplexのみ実装（アルゴリズムの柔軟性が低い） [1]

**メモリ使用量**
- OSSソルバーの中で最もメモリ増加が大きい（最小モデル172MB → 最大モデル261MB） [1]

### 1.3 OSS全般の弱点（事実）

**開発投資の差**
- 商用MIPソルバーは「数十人年」規模の開発投資を受けている [1]
- 高額なライセンス料によって継続的な開発が資金調達されている [1]

**技術的な差**
- 並列木探索の実装が不十分 [1]
- 問題クラスごとの最適化トリックが少ない [2]
- パラメータチューニングや内部最適化が不足 [2]

### 1.4 攻めるべき隙（足軽の分析）

**性能以外の差別化要素**
- エコシステム統合: HiGHSの成功例（SciPy、NAG、MathWorks、PyPSAのデフォルトソルバー） [5]
- 利用障壁の低さ: MITライセンスによる広範な採用 [5]
- 特化領域での性能: 小～中規模問題ではHiGHSがGurobiと同等の性能 [5]

**技術的ブレークスルーの可能性**
- 並列化アルゴリズムの改善（HiGHSでも進行中）
- 特定問題クラス（例: LP特化、特殊構造MIP特化）での最適化
- ML技術の活用（ハイパーパラメータ選択、分岐戦略の学習等）

---

## 2. 商用ソルバーに迫るための鍵

### 2.1 商用ソルバーの技術的アドバンテージ（事実）

**コアテクノロジー**

1. **Presolve（前処理）** [8]
   - 分枝限定法開始前に問題サイズを削減・定式化を強化
   - Cutting planeと並んで商用MIPソルバーの力の最重要要素

2. **Cutting Planes（切除平面法）** [8]
   - CPLEXは分枝カット法を採用し、presolve戦略とcutting planesを組み合わせてモデルを再定式化
   - Gurobiは対称性検出も含む最新のcutting plane実装

3. **Heuristics（発見的手法）** [8]
   - Gurobiは多様な種類の複数ヒューリスティクスを実装
   - 特にGurobiは整数計画問題を加速する高度なheuristic cutting-plane技術を統合 [2]

4. **並列処理** [8]
   - Gurobiは共有メモリ並列化に対応し、任意数のプロセッサ・コアを同時活用可能
   - MIPソルバーは並列実行され、主な並列化源はMIP木探索の各ノードを独立処理できること

**性能差の実態**
- 商用ソルバー（CPLEX, XPRESS, Gurobi）はOSS（HiGHS, CBC, SCIP）より約1～2桁高速 [2]
- 大規模問題では性能差がさらに拡大 [2]
- 特定の種類のモデルでは、商用ソルバーなら簡単に解ける問題が、無料ソルバーでは絶望的 [2]

**その他のアドバンテージ**
- 洗練されたインターフェース [2]
- 複数プログラミング言語向けAPI [2]
- クラウドコンピューティング対応 [2]

### 2.2 追いつける部分・追いつけない部分（足軽の分析）

**追いつきやすい領域**
- **小～中規模問題**: HiGHSは既にGurobiと同等性能を示している [5]
- **特定問題クラス**: LPやQPなど限定的な問題種別に特化すれば性能差を縮められる
- **インターフェース改善**: OSS独自の強み（他ツールとの統合、言語バインディング）

**追いつきにくい領域**
- **大規模MIP**: 開発投資の差が顕著に出る
- **汎用性**: あらゆる問題クラスに対する「トリック」の蓄積
- **商用サポート**: 企業向けサポート体制・SLA保証

**攻め口（足軽の分析）**
1. **並列化**: GPU活用、分散並列化など新しいアプローチ
2. **ML活用**: 深層学習による分岐戦略・パラメータ最適化
3. **特定領域特化**: 特殊構造を持つ問題（例: ネットワークフロー、スケジューリング）での最適化
4. **実装言語**: Rust等による安全性・性能の両立（推測）
5. **クラウドネイティブ設計**: 分散環境での性能最適化

---

## 3. OSSとしてのアピール方法

### 3.1 成功したOSSプロジェクトの事例（事実）

**PyTorch** [3]
- **コミュニティ規模**: 約100人のコアメンバー（Facebook内外）+ 900人以上のOSSコントリビューター + 6人のメンテナー
- **開発哲学**: 最初からOSSをDNAに組み込み、Facebookのエンジニアリング文化の一部として位置づけ
- **CI/CD**: CircleCIを活用し、多数のOSSコントリビューターに対してスムーズで信頼性の高いPRプロセスを提供
- **エコシステム**: 世界中の研究者に重要なツールを提供し、巨大な相互接続プロジェクトネットワークを構築
- **コミュニティイベント**: 2026年4月7-8日にパリで2日間のコミュニティ会議開催

**PostgreSQL** [3]
- **歴史**: 2026年で30周年。「おもちゃに過ぎない」と言われたプロジェクトから発展
- **ガバナンス**: 単一企業が所有せず、複数の競合する利益が調整されながら開発
- **コミュニティ**: 活発なアップストリームコミュニティ + 多数の企業・製品がPostgres周辺に構築
- **グローバル**: 異なる国・大陸の人々が協力して作業

**HiGHS** [5]
- **採用実績**:
  - SciPy 1.6.0からLPソルバーとして、1.9.0からMIPソルバーとして採用
  - NAGライブラリのMIPソルバーのベース
  - MathWorks Optimization ToolboxのデフォルトLP/MIPソルバー
  - PyPSAヨーロッパ多部門モデルのWebベース版でデフォルトソルバー（2022年2月～）
- **技術**: C++で実装、MITライセンス
- **言語バインディング**: C, Python, Julia, Rust, R, JavaScript, Fortran, C#

### 3.2 コミュニティ形成戦略（事実 + 足軽の分析）

**ベストプラクティス（事実）** [3]
- 活発なリポジトリ
- エンゲージメントの高いコミュニティ
- 明確なコントリビューションガイドライン
- これらは長期的なインパクトと可視性を確保する

**足軽の分析**
- **早期からのOSS文化**: PyTorchのように組織の文化として位置づける
- **低い参加障壁**: HiGHSのMITライセンスのような寛容なライセンス選択
- **継続的エンゲージメント**: 定期的なコミュニティイベント、活発なdiscussion/issue対応
- **企業スポンサー戦略**: 複数企業が利益を持つエコシステム形成（PostgreSQLモデル）

### 3.3 ドキュメント・チュートリアル・エコシステム（足軽の分析）

**重要性**
- 技術的に優れていても、利用障壁が高ければ採用されない
- HiGHSの成功は「デフォルトソルバー」として組み込まれたことが大きい

**推奨アプローチ**
- 初学者向けチュートリアル（PyTorchスタイル）
- 既存ツール（SciPy, OR-Tools等）への統合パス提供
- 実問題の解法例（ベンチマークだけでなく実用例）
- API設計のシンプルさ（HiGHSの成功要因）

### 3.4 ライセンス選択の影響（事実）

**ライセンスの戦略的意味** [6]
- **最大限のオープン性**: GPL
- **最大限の配布**: MITライセンス
- **企業プロジェクト（特許保護付き）**: Apacheライセンス

**ライセンス統計** [6]
- OSSプロジェクトの60%以上がMIT、GPL、Apache 2.0のいずれかを使用

**ライセンスの影響** [6]
- **Copyleftライセンス（GPL）**: 変更を同じライセンスで公開必要 → ビジネス戦略・製品開発に大きく影響
- **Permissiveライセンス（MIT, Apache）**: プロプライエタリソフトウェアへの組み込み可能 → 公開義務なし
- **企業環境**: プロプライエタリソリューション開発の場合、MITやApacheが好まれる

**戦略的選択の指針** [6]
- **最大限の採用を目指す場合**: MIT/Apache（オープン・クローズ双方で利用可能）
- ライセンス選択はコントリビューションインセンティブ、採用、互換性、商業化パスを形成する

**足軽の分析**
- HiGHSの成功はMITライセンスが大きい（SciPy, MathWorks等への組み込み容易性）
- ソルバー領域では「広く使われる」ことがエコシステム形成の鍵
- GPL選択はアカデミア主体のSCIPでは機能したが、産業界統合ではMIT/Apacheが有利

---

## 4. 差別化のポジショニング

### 4.1 「全部やる」vs「一点突破」の比較

**SCIP: 包括的カバレッジ戦略（事実）** [7]

- **哲学**: 数理計画の専門家が解法プロセスを完全制御し、ソルバーの内部情報に詳細アクセスできるフレームワーク
- **対応問題クラス**: 混合整数線形計画、混合整数二次計画、混合整数半正定値計画、混合整数非線形計画、擬似ブール最適化など膨大な範囲
- **拡張性**: 純粋なMIP/MINLPソルバーとして使用可能、またはbranch-cut-and-priceフレームワークとして利用可能
- **拡張例**: 受賞歴のあるSteiner tree solver（SCIP-Jack）、混合整数半正定値計画ソルバー（SCIP-SDP）
- **性能**: 学術開発ソルバーの中で最速クラスのMIP/MINLPソルバー
- **評価**: ベンチマークの膨大な配列で競争力を維持

**HiGHS: 特化戦略（事実）** [5]

- **対象**: 線形計画（LP）、混合整数計画（MIP）、凸二次計画（QP）
- **強み**: 小～中規模問題ではGurobiと同等性能
- **採用戦略**: シンプルなAPI + 主要ライブラリへのデフォルト組み込み
- **ライセンス**: MIT（商用・学術双方で自由に使用可能）

### 4.2 足軽の分析：どちらが有利か

**一点突破（HiGHSモデル）の利点**
1. **開発リソース集中**: 限られたリソースで高い性能を達成しやすい
2. **明確な価値提案**: 「LP/MIPなら最速のOSS」という訴求がシンプル
3. **エコシステム統合**: SciPy, MathWorks等への組み込みが容易（APIシンプル、依存関係少ない）
4. **性能ベンチマーク**: 限定的な問題クラスなら商用ソルバーとの性能差を縮めやすい

**包括的カバレッジ（SCIPモデル）の利点**
1. **多様なユーザー**: アカデミア・研究者向けには強力（研究テーマが多様）
2. **拡張性**: フレームワークとして使えるため、研究プラットフォームになる
3. **ロングテール需要**: 非線形、半正定値など特殊問題クラスのユーザーも取り込める
4. **エコシステムの厚み**: 拡張プロジェクト（SCIP-Jack, SCIP-SDP等）が生まれやすい

**市場状況に基づく判断**
- **商用ソルバー対抗**: 一点突破が有利（HiGHSの成功が証明）
- **学術・研究用途**: 包括的カバレッジが有利（SCIPの成功が証明）
- **産業界採用**: 一点突破 + シンプルAPI + Permissiveライセンスが最強（HiGHSモデル）

### 4.3 推奨ポジショニング（足軽の意見）

**Phase 1: 一点突破で橋頭堡確保**
- LP/MIPに特化し、中規模問題でGurobi並み性能を目指す
- MITライセンスで広範な採用を促進
- SciPy/Pandas/NumPy等のPythonエコシステムへの組み込みを最優先
- シンプルなAPI設計（学習コスト最小化）

**Phase 2: エコシステム拡大**
- 成功事例・ユースケースの蓄積
- コミュニティ形成（定期イベント、コントリビューター支援）
- 企業スポンサー獲得（複数企業が利益を持つ構造）

**Phase 3: 段階的拡張**
- 性能が確立した後、周辺問題クラス（QP, SOCP等）に拡大
- ただしコア性能（LP/MIP）を犠牲にしない範囲で

**差別化ポイント**
1. **並列化・GPU活用**: 商用ソルバーがまだ弱い領域
2. **クラウドネイティブ**: 分散環境での最適化
3. **ML統合**: ハイパーパラメータ自動調整、問題構造の学習
4. **特殊構造の活用**: ネットワークフロー、スケジューリング等の実問題に特化した最適化

---

## まとめ

### 天下を取るための戦略

1. **初期フェーズ**: HiGHSモデル（一点突破）が有利
   - LP/MIP特化、MITライセンス、シンプルAPI
   - Pythonエコシステムへの統合最優先

2. **技術的差別化**:
   - 並列化/GPU活用（商用ソルバーの弱点）
   - ML技術統合（学習ベースのパラメータ選択）
   - 特殊構造最適化（実問題特化）

3. **コミュニティ戦略**:
   - PyTorchモデル: 企業バックアップ + OSSコミュニティ
   - PostgreSQLモデル: 複数企業が利益を持つエコシステム
   - 明確なコントリビューションガイドライン + 活発なエンゲージメント

4. **商用ソルバーに対する現実的目標**:
   - 小～中規模問題で同等性能（達成可能 - HiGHSが証明済み）
   - 大規模問題では「十分使える」レベル（1桁差まで縮める）
   - 特定領域で商用超え（並列化、特殊構造活用等）

---

## 情報源

[1] [Do we know why open source MIP solvers are significantly slower than commercial ones? · ERGO-Code/HiGHS · Discussion #1683](https://github.com/ERGO-Code/HiGHS/discussions/1683)

[2] [Open-Source Solvers vs. Gurobi: Key Considerations - Gurobi Optimization](https://www.gurobi.com/resources/open-source-solvers-vs-gurobi-key-considerations/)

[3] [Leading open source ML advancements - CircleCI](https://circleci.com/case-studies/pytorch/) および [FOSDEM 2026 - Building the next generation of open source contributors – Lessons from 30 years of Postgres](https://fosdem.org/2026/schedule/event/JDZXFE-next-generation-contributors-lessons-from-postgres/)

[4] [24 Open Source Projects to Contribute to in 2026 | ClickUp](https://clickup.com/blog/top-open-source-projects-to-contribute/)

[5] [HiGHS optimization solver - Wikipedia](https://en.wikipedia.org/wiki/HiGHS_optimization_solver) および [About · HiGHS Documentation](https://ergo-code.github.io/HiGHS/dev/)

[6] [Understanding Open Source Licenses: GPL, MIT, Apache Compared - credativ®](https://www.credativ.de/en/blog/credativ-inside/understanding-open-source-licenses-gpl-mit-apache-compared/) および [MIT, Apache, or GPL? Open Source Licenses Made Simple | by Ritik Singh | Medium](https://medium.com/@bitsbyritik/mit-apache-or-gpl-open-source-licenses-made-simple-c8fcb54f413c)

[7] [The SCIP Optimization Suite 9.0](https://arxiv.org/html/2402.17702v2) および [Why SCIP? — PySCIPOpt documentation](https://pyscipopt.readthedocs.io/en/latest/whyscip.html)

[8] [Recent Advancements in Commercial Integer Optimization Solvers for Business Intelligence Applications | IntechOpen](https://www.intechopen.com/chapters/73051) および [Mixed-Integer Programming (MIP/MILP) – A Primer on the Basics - Gurobi Optimization](https://www.gurobi.com/resources/mixed-integer-programming-mip-a-primer-on-the-basics/)

[9] [MIP solver choice 2023 · google/or-tools · Discussion #3969](https://github.com/google/or-tools/discussions/3969)

---

**記載方針**
- 事実はすべて情報源付きで記載
- 足軽の分析・意見は「足軽の分析」「足軽の意見」と明記して分離
