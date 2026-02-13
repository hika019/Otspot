# ソルバー戦略討論 — 実用主義派 Opening Statement

**立場**: Go言語による実用優先・段階的拡張アプローチ
**執筆**: 足軽三番
**日付**: 2026-02-14

---

## 1. 参入領域: 実用主義的アプローチ

### 1.1 両派の検討と独自提案

ES派（NLP/MINLP GPU特化）とReq派（LP/MIP Rust実装）の両立場を検討した結果、**どちらも正しいが、どちらも不完全**であると判断する。

**ES派の正しさ:**
- GPU-NLPが商用・OSS双方で空白地帯であることは事実
- 化学・エネルギー・航空宇宙での大規模NLP需要も実在
- 「唯一無二のカテゴリー」創出は戦略的に正しい

**ES派の盲点:**
- 1年でGPU-NLP MVPは**技術的リスクが極めて高い**。NLP向けPDHG理論が未確立（executive_summary.md §3.1参照）であり、Ipoptを3-5倍高速化する保証がない
- C++実装は開発速度・チーム拡大のボトルネックになる（メモリ安全性欠如、並列化でのデバッグ困難）

**Req派の正しさ:**
- LP/MIPは産業需要の中核。全ての基盤である
- Rustのメモリ安全性・並列安全性は並列化ファースト設計で決定的に重要
- Apache-2.0（特許保護付き）は戦略的に正しい

**Req派の盲点:**
- Rustの学習曲線は**現実的な開発速度を阻害する**（初期フェーズでの速度が致命的に遅い）
- 「並列化ファースト設計」は理想だが、並列化の前に**まず動くコードが必要**
- 数値計算ライブラリの成熟度がC++に劣る（nalgebra/ndarrayはBLAS/LAPACKの完全代替ではない）

### 1.2 実用主義派の提案: ハイブリッド段階戦略

**Phase 1（0-12ヶ月）: LP特化MVP — 最速市場投入**
- 参入領域: LP専用（MIPは後回し）
- 実装言語: Go（開発速度最優先）
- アルゴリズム: Simplexのみ（Primal/Dual両方）
- 目標性能: HiGHS比80%（劣っても良い。「使える」が重要）
- エコシステム統合: Python bindings（PyPI公開）、SciPy/pandas/numpy連携
- ライセンス: MIT（最大限の採用促進）

**Phase 2（12-24ヶ月）: MIP追加 + 並列化**
- MIP基本実装（Branch-and-cut）
- goroutineによるノード探索並列化（Go最大の強み活用）
- 性能目標: HiGHS同等
- ライセンス移行検討: MITからApache-2.0へ（特許保護強化）

**Phase 3（24-36ヶ月）: GPU/ML統合 + Rust移行検討**
- CUDA/ROCmバインディング（CGO経由）
- ML統合（Cut Ranking等）
- **コア部分のRust再実装を検討**（ここで初めてRustの投資対効果が正当化される）

### 1.3 「完璧な戦略」より「実行可能で修正可能な戦略」の価値

ES派の「1年でGPU-NLP MVP」、Req派の「Rust並列化ファースト設計」は共に**理想的だが実行リスクが高い**。

実用主義派は以下を優先する:
1. **12ヶ月以内に動くものを出す**（市場検証・フィードバック獲得）
2. **技術的負債を許容する**（Go→Rustの段階的移行を前提）
3. **ユーザー数で勝つ**（PyPI公開、SciPy統合で採用促進）

HiGHSの成功は「最速」ではなく「デフォルトソルバー」になったこと（SciPy 1.6.0採用、winning_strategy.md §4.1）。**速度より普及が戦略的に重要**。

---

## 2. 実装言語: なぜGoか

### 2.1 開発速度とチーム拡大のしやすさ

**事実:**
- Go学習曲線: 数日～数週間（Rust: 数ヶ月～1年）
- Go構文: Cライク、メモリ管理自動（GC）、エラーハンドリング明示的
- Rust構文: 所有権・借用・ライフタイム学習が必須、コンパイラエラーとの格闘

**実測:**
- Go 1.26（2026年2月リリース）: CGO baseline overhead 30%削減（[Go 1.26 Release Notes](https://go.dev/doc/go1.26)）
- Goプロジェクト初期フェーズ: 数週間でプロトタイプ可能
- Rustプロジェクト初期フェーズ: 数ヶ月（所有権システムとの格闘）

**推論:**
12ヶ月以内のMVP達成を目標とする場合、開発速度の差が致命的になる。ES派C++も同様の問題を抱えるが、Goは**メモリ安全性をGCで自動化**することで、C++のメモリバグを回避しつつ開発速度を確保する。

### 2.2 goroutineによる並行処理の自然さ

**事実（Gonum事例）:**
- gonum/optimize: グローバル最適化（Simulated annealing）実装済み（[Gonum Optimization Package](https://pkg.go.dev/gonum.org/v1/gonum/optimize)）
- gonum/lp: Simplex algorithm（Dantzig法）をPure Go実装（[Gonum LP Package](https://pkg.go.dev/gonum.org/v1/gonum/optimize/convex/lp)）
- goroutine: 軽量スレッド（1goroutineあたりメモリ数KB）、数万並列可能

**ソルバーへの応用:**
- MIPのBranch-and-boundノード探索: 各ノードをgoroutineで並列実行
- LP行列演算: 行ごとの処理をgoroutineで分散
- 実装例: `go func() { ... }()` で並列化完了（Rustのthread::spawnより直感的）

**Req派との比較:**
- Rust: 所有権システムによりデータ競合をコンパイル時防止（理論的最強）
- Go: 実行時データ競合検出（`go run -race`）で実用上十分
- **トレードオフ**: Goは実行時検出のため完璧ではないが、開発速度とのバランスで優位

### 2.3 シングルバイナリデプロイ

**事実:**
- Go: 静的リンクのシングルバイナリ生成（依存ライブラリ込み）
- Rust: 同様にシングルバイナリ生成可能（Goと同等）
- C++: 共有ライブラリ地獄（libc, libstdc++等のバージョン依存問題）

**ユーザー体験:**
```bash
# Goソルバー（理想）
curl -LO https://example.com/gosolver && chmod +x gosolver && ./gosolver

# C++ソルバー（現実）
apt-get install libopenblas0 liblapack3 ... # 依存地獄
```

HiGHSの成功要因の一つは**組み込みやすさ**（MIT License + C++だがヘッダーオンリー化進行）。Goのシングルバイナリはさらに優位。

### 2.4 CGO経由のBLAS/LAPACK/CUDA連携

**現状調査結果:**
- Gonum: LAPACK部分実装 + CGOラッパーでC-based実装を併用（[Gonum LAPACK Package](https://pkg.go.dev/gonum.org/v1/gonum/lapack)）
- CGO overhead: ~40ns/call（[CGO Performance in Go 1.21](https://shane.ai/posts/cgo-performance-in-go1.21/)）
- Go 1.26改善: CGO baseline overhead 30%削減（~28ns/callに改善）

**BLAS/LAPACK連携の実態:**
- 行列演算コアはCGO経由でOpenBLAS/Intel MKL呼び出し
- 1回の行列積（1000x1000）で数百万FLOPS → CGO overhead無視可能
- 頻繁な小行列演算（100x100を数万回）では overhead問題あり

**CUDA連携の可能性:**
- CGO経由でCUDA Runtime API呼び出し可能
- 先行事例: [golp](https://github.com/draffensperger/golp)（LPSolveバインディング）、[goop](https://github.com/mit-drl/goop)（MIP in Go、Gurobiバックエンド対応）
- cuOptバインディングも技術的に可能（CGO + CUDA C wrapper）

### 2.5 C++やRustに対する率直な劣位点

**Go vs Rust:**
| 項目 | Go | Rust | 評価 |
|------|----|----|------|
| メモリ安全性 | GC（実行時overhead） | コンパイル時保証 | **Rust優位** |
| 並列安全性 | 実行時検出（-race） | コンパイル時保証 | **Rust優位** |
| 数値計算エコシステム | gonum（薄い） | nalgebra/ndarray（Goより厚いが、C++に劣る） | **両者C++に劣る** |
| 開発速度 | 極めて高速 | 低速（学習曲線） | **Go優位** |
| ゼロコスト抽象化 | GC overhead存在 | ゼロコスト | **Rust優位** |
| CGO/FFI性能 | 28ns/call（1.26） | unsafe経由で高速 | **Rust優位** |

**Go vs C++:**
| 項目 | Go | C++ | 評価 |
|------|----|----|------|
| メモリ安全性 | GC保証 | 手動管理（バグ源） | **Go優位** |
| 並列化実装難易度 | goroutine（容易） | pthread/OpenMP（困難） | **Go優位** |
| 数値計算エコシステム | 薄い | 最強（Eigen, BLAS等） | **C++優位** |
| コンパイル速度 | 高速 | 遅い | **Go優位** |

**Go最大の弱点: GC overhead**

数値計算における実測（[No GC in Go Benchmarks](https://blog.devgenius.io/no-garbage-collection-in-go-performance-benchmarks-eca6c2fb8307)）:
- GC無効化: 10-40%高速化（メモリリーク許容で実験）
- Go 1.26 Green Tea GC: 10-40% GC overhead削減（[Go 1.26 GC Improvements](https://www.infoworld.com/article/4131097/go-1-26-unleashes-performance-boosting-green-tea-gc.html)）
- ベクトル命令最適化（Ice Lake/Zen 4+）: 追加10%削減

**結論**: GC overheadは存在するが、1.26で大幅改善。大規模行列演算（計算bound）ではGC影響は相対的に小さい。

### 2.6 総合的な優位の主張

**Phase 1での最適解はGo:**
1. **12ヶ月MVP達成**が至上命題 → 開発速度でGo圧勝
2. **並列化の容易さ**がHiGHS超えの鍵 → goroutineで実装加速
3. **デプロイ簡便性**がエコシステム統合で重要 → シングルバイナリで優位
4. **GC overhead**は大規模問題で相対的に小さい（計算boundなため）

**Phase 3でのRust移行オプション:**
- ユーザーベース確立後、コア部分のRust再実装を検討
- Go実装を「実証済み仕様書」として、Rust移行リスク低減
- ハイブリッドアプローチ: Goフロントエンド + Rustコア（CGO経由連携）

---

## 3. ライセンス: 推奨と根拠

### 3.1 Phase 1: MIT License

**根拠:**
- HiGHSの成功はMIT Licenseが決定的要因（SciPy 1.6.0/1.9.0採用、winning_strategy.md §3.4）
- 商用組み込み自由 → 企業採用促進 → エコシステム拡大
- GPL（GLPKモデル）では商用敬遠 → 普及阻害

**Apache-2.0との比較:**
- Apache-2.0: 特許保護条項あり（Req派推奨）
- MIT: 特許保護なし（リスク残存）

**Phase 1でMIT選択の理由:**
- 初期段階では**特許トロールリスクより普及速度が重要**
- MITの「シンプルさ」が採用障壁を最小化
- SCIPがApache-2.0移行（SCIP 10.0）したのは成熟後（oss_solvers.md §2）

### 3.2 Phase 2-3: Apache-2.0移行検討

**条件:**
- ユーザーベース1000+達成（コミュニティ確立）
- 商用採用3社+達成（特許リスク顕在化）
- コアアルゴリズム特許出願検討時

**Apache-2.0の優位:**
- MIT同等の自由度 + 特許保護
- Rustエコシステム慣例（Apache-2.0/MITデュアル）
- SCIP実績（Apache-2.0移行で商用採用維持）

**ライセンス移行リスク:**
- 既存ユーザーへの影響分析必須
- MIT→Apache-2.0は一般に許容される（逆は困難）

---

## 4. ES派・Req派への批判と提案

### 4.1 C++の問題点（ES派への指摘）

**技術的問題:**
1. **メモリ安全性欠如**: use-after-free, buffer overflow → 大規模行列操作で致命的
2. **並列化のデバッグ困難**: data race検出が実行時のみ、再現困難
3. **コンパイル遅延**: 大規模プロジェクトで数分〜数十分

**戦略的問題:**
- ES派提案「1年でGPU-NLP MVP、C++コア」は**リスクが高すぎる**
- GPU-NLP理論未確立（PDHG理論がNLPで未確立、executive_summary.md §3.1）
- C++デバッグ困難 → 開発遅延 → 1年MVP破綻リスク

**提案:**
- Phase 1: GPU-NLPを回避し、**GPU-LP**（理論確立済み、PDHG実績あり）から開始
- 言語: GoでMVP → Phase 2でC++/Rust検討
- リスク分散: LPで市場検証 → NLP拡張判断

### 4.2 Rustの問題点（Req派への指摘）

**開発速度問題:**
- Req派「Rust並列化ファースト設計」は理想だが、**初期速度犠牲が大きすぎる**
- 所有権システム学習曲線 → 初期3-6ヶ月は生産性低迷
- 「設計してから実装」より「実装して設計検証」が実用的

**エコシステム問題:**
- nalgebra/ndarrayはBLAS/LAPACKの完全代替ではない
- CGO/FFI overhead（Rustも存在、unsafeブロック必要）
- Rust数値計算プロジェクト実績: russcip（SCIPバインディング）のみ目立つ

**提案:**
- Phase 1: Goで市場検証 → ユーザーフィードバック獲得
- Phase 2-3: フィードバック基にRust再実装検討
- Rust投資を**後回し**にすることで、「何を作るべきか」明確化してから投資

### 4.3 両派の参入戦略の盲点

**ES派の盲点:**
- GPU-NLPは「世界初」だが、**需要検証が不十分**
- 化学・エネルギー企業ヒアリング実施せず（executive_summary.md §5には顧客ヒアリング未実施）
- 「作れば売れる」は危険（Field of Dreamsの罠）

**Req派の盲点:**
- LP/MIP正面競争はHiGHS/SCIP/商用ソルバーとの**性能勝負**
- 「並列化ファースト」だけではHiGHS超え困難（HiGHSも並列化進行中、research_trends.md §4）
- 差別化要素不明確（「Rustで安全」は技術者アピールのみ、ユーザー価値薄い）

**実用主義派の差別化:**
1. **開発速度**: 12ヶ月MVP達成 → 先行者利得
2. **エコシステム**: Python統合（PyPI/SciPy）最優先 → HiGHSモデル
3. **段階的拡張**: LP→MIP→QP（失敗リスク分散）
4. **技術的負債許容**: Go→Rust移行前提（完璧主義回避）

### 4.4 現実的な勝ち筋

**HiGHSに学ぶ:**
- HiGHS成功は「最速」ではなく「デフォルトソルバー」採用（SciPy, MathWorks, winning_strategy.md §4.1）
- MIT License + シンプルAPI + 組み込みやすさ = 普及

**実用主義派の勝ち筋:**
1. **Phase 1（0-12ヶ月）**: Go LP MVP、MIT、PyPI公開 → 初期ユーザー獲得
2. **Phase 2（12-24ヶ月）**: MIP追加、goroutine並列化 → HiGHS同等達成
3. **Phase 3（24-36ヶ月）**: GPU/ML/Rust検討 → 技術的優位確立

**ES派・Req派への提言:**
- ES派: GPU-NLPはPhase 3以降。Phase 1はGPU-LP（リスク低）
- Req派: Rustは「正しい最終形」だが、「正しい初手」ではない。Go→Rust段階移行を

---

## 結語

**完璧な戦略は存在しない。実行可能な戦略のみが価値を持つ。**

ES派の「GPU-NLP世界初」、Req派の「Rust並列化ファースト」は技術的に魅力的だが、12ヶ月MVP達成には**リスクが高すぎる**。

実用主義派は以下を主張する:
- **Go言語でLP特化MVP** → 開発速度最優先
- **MIT License** → 普及最優先
- **段階的拡張** → 技術的負債許容、失敗リスク分散
- **Phase 3でRust移行検討** → 完璧主義を後回し

「世界を変える完璧なソルバー」を5年かけて作るより、「実用的なソルバー」を1年で市場投入し、ユーザーフィードバックで改善する方が**戦略的に正しい**。

---

## 情報源

### Go言語・数値計算
- [Gonum: Numerical Computing in Go](https://www.gonum.org/)
- [Gonum Optimization Package](https://pkg.go.dev/gonum.org/v1/gonum/optimize)
- [Gonum LP Package - Simplex](https://pkg.go.dev/gonum.org/v1/gonum/optimize/convex/lp)
- [Gonum LAPACK Package](https://pkg.go.dev/gonum.org/v1/gonum/lapack)
- [Scientific Computing in Golang with Gonum](https://www.freecodecamp.org/news/scientific-computing-in-golang-using-gonum/)
- [Awesome Scientific Go](https://github.com/samuell/awesome-scientific-go)

### Go CGO/FFI Performance
- [Go 1.26 Release Notes](https://go.dev/doc/go1.26)
- [CGO Performance in Go 1.21](https://shane.ai/posts/cgo-performance-in-go1.21/)
- [CGO Performance Issue (Go 1.22.5)](https://github.com/golang/go/issues/68587)

### Go Garbage Collection
- [Go 1.26 Green Tea GC](https://www.infoworld.com/article/4131097/go-1-26-unleashes-performance-boosting-green-tea-gc.html)
- [Green Tea GC Real-World Test (DoltHub)](https://www.dolthub.com/blog/2025-09-26-greentea-gc-with-dolt/)
- [No GC in Go: Performance Benchmarks](https://blog.devgenius.io/no-garbage-collection-in-go-performance-benchmarks-eca6c2fb8307)
- [Go GC Guide (Official)](https://go.dev/doc/gc-guide)

### Go Solver Projects
- [golp - Go bindings for LPSolve](https://github.com/draffensperger/golp)
- [goop - Generalized MIP in Go (MIT-DRL)](https://github.com/mit-drl/goop)
- [Nextmv SDK MIP Package](https://pkg.go.dev/github.com/nextmv-io/sdk/mip)
- [CLP - COIN-OR Linear Programming (LANL)](https://github.com/lanl/clp)
- [go-glpk - Go bindings for GLPK](https://pkg.go.dev/github.com/lukpank/go-glpk/glpk)

### ソルバー戦略（調査資料）
- executive_summary.md (本プロジェクト)
- requirements.md (本プロジェクト)
- winning_strategy.md (本プロジェクト)
- oss_solvers.md (本プロジェクト)
- commercial_solvers.md (本プロジェクト)
- research_trends.md (本プロジェクト)
