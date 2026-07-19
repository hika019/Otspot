# otspot

[English](README.md) | **日本語**

Rust で書かれた**数理最適化ソルバー**。

LP: 修正シンプレックス法（疎LU分解、Ruiz均衡スケーリング、最急勾配価格決定）。
QP: 内点法（Mehrotra predictor–corrector / IP-PMM）＋非凸QPの空間 branch-and-bound（α-BB / McCormick）。
MILP / 凸MIQP: branch-and-bound。
`Optimal` は**proof-carrying**（完全なKKT検証済み）。実行不可能・非有界も証明付きで返す。

## 機能

- **代数モデリングAPI** — 自然な数式記法。`x * x` / `x * y` による二次目的を含む
- **修正シンプレックス法（LP）** — 疎LU分解、Markowitz閾値ピボット、最急勾配価格決定
- **内点法（QP）** — 凸QPに対するMehrotra predictor–corrector / IP-PMM
- **非凸QP（大域）** — 空間B&B（α-BB / McCormick）。大域最適は証明書付き、局所解は `NonconvexLocal`
- **混合整数（MILP / 凸MIQP）** — branch-and-bound（GMI/MIR/cover/clique/implied-bound カット、reliability 分岐、RINS、競合分析）
- **感度分析（LP）** — RHS および目的関数係数の変動幅解析（ranging）
- **証明付き最適性** — `Optimal` は完全KKT証明書を要求。証明不能な解は降格
- **実行不可能・非有界の判定**
- **双対解出力** — 双対変数、簡約費用、スラック
- **入力フォーマット** — MPS（LP）、QPS / QPLIB（QP）

## クイックスタート

必要環境: Rust（edition 2021, stable）。

```toml
[dependencies]
otspot = "0.7"
```

```bash
git clone https://github.com/hika019/otspot.git
cd otspot
cargo run --release --example solve_lp   # LP の最小例
cargo run --release --example solve_qp   # QP の最小例
```

### LP

```rust
use otspot::model::{constraint, Model};

fn main() {
    // minimize  x + 2y   s.t.  2x + 3y <= 12,  x + y >= 3,  x >= 0, y in [0,10]
    let mut model = Model::new("example");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, 10.0);
    model.add_constraint(constraint!((2.0 * x + 3.0 * y) <= 12.0));
    model.add_constraint(constraint!((x + y) >= 3.0));
    model.minimize(x + 2.0 * y);

    let result = model.solve().unwrap();
    println!("obj={} x={} y={}", result.objective_value, result[x], result[y]);
}
```

`constraint!` は単一変数形式（`constraint!(x <= 7.0)`）や式メソッド API（`.leq()` / `.geq()` / `.eq_constraint()`）も使える。最大化は `model.maximize(...)` を使う。

許容誤差 / オプション:

```rust
use otspot::Tolerance;
model.set_tolerance(Tolerance::High); // 1e-8; Medium (1e-6, 既定), Fast, Custom(f64)
model.set_timeout(60.0);
```

### QP

```rust
use otspot::model::{constraint, Model};

fn main() {
    // minimize  x² + y²   s.t.  x + y >= 1
    let mut model = Model::new("qp");
    let x = model.add_var("x", f64::NEG_INFINITY, f64::INFINITY);
    let y = model.add_var("y", f64::NEG_INFINITY, f64::INFINITY);
    model.add_constraint(constraint!((x + y) >= 1.0));
    model.minimize(x * x + y * y);

    let result = model.solve().unwrap();
    println!("obj={:.4} x={:.4} y={:.4}", result.objective_value, result[x], result[y]);
}
```

### 低レベルAPI

```rust
use otspot::{problem::LpProblem, sparse::CscMatrix, solve};

let c = vec![-1.0, -1.0];
let rows = vec![0usize, 0, 1, 2];
let cols = vec![0usize, 1, 0, 1];
let vals = vec![1.0, 1.0, 1.0, 1.0];
let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 2).unwrap();
let problem = LpProblem::new(c, a, vec![4.0, 3.0, 3.0]).unwrap();
let result = solve(&problem);
println!("{} {}", result.status, result.objective); // Optimal -4
```

### MPS入力

```rust
use otspot::{io::mps, solve};
let prob = mps::parse_mps_file("problem.mps".as_ref()).unwrap();
let result = solve(&prob);
```

## 性能

標準公開セットでの求解率ベンチ。otspot-dev の benchmark harness（shell スクリプト — **`cargo bench` ではない**）で計測、`timeout = 1000s`:

| 問題種別 | セット | 問題数 | @1e-6 | @1e-8 |
|---|---|---:|---|---|
| 実行可能 LP | Netlib | 109 | 最適解 109 | 最適解 108、SuboptimalSolution 1 |
| 凸 QP | Maros–Mészáros | 138 | 最適解 121、SuboptimalSolution 1、Stalled 9、MaxIterations 2、OBJ_MISMATCH 1、参照値なし 4 | 最適解 93、SuboptimalSolution 42、TIMEOUT 1、参照値なし 2 |
| QCQP | QPLIB | 41 | 最適解 11、SuboptimalSolution 3、Stalled 8、TIMEOUT 3、NOT_SUPPORTED 11、SKIP 5 | 最適解 8、SuboptimalSolution 4、Stalled 9、TIMEOUT 4、NOT_SUPPORTED 11、SKIP 5 |
| MILP | MIPLIB 2017 small | 20 | 最適解 5、TIMEOUT 15、ERROR 0 | 最適解 5、TIMEOUT 15、ERROR 0 |
| SOCP | Mittelmann Large-SOCP | 18 | Optimal 4 @1000s（3600s で 6）、他 TIMEOUT、OOM 1 | n/a（1e-6 のみ） |
| 実行不可能 LP | Netlib | 29 | 正答 29 | 正答 29 |
| 非有界 LP | 合成 | 12 | 正答 12 | 正答 12 |

**最適解** = 既知最適値と照合済み（proof-carrying KKT）。**Stalled** = 反復・時間予算内でこれ以上進展せず、解を主張しない誠実な非収束 status（旧 taxonomy では SuboptimalSolution に丸められていた）。LP/QP/QCQP/MILP 行は `timeout = 1000s`、`jobs = 6`。SOCP 行は Mittelmann ベンチに合わせ `jobs = 1`（逐次。大規模問題は 1 問あたり最大 ~18 GB RSS）で計測し、timeout は下記の SOCP 注記に従う。

LP: @1e-6 は 109/109 最適解、timeout 0。@1e-8 は 108/109 最適解、timeout 0。ミスは `greenbea`（より厳しい primal 証明ゲートで SuboptimalSolution）——この列は上記 taxonomy 分割より前の計測で、今回は再測定していない。

QP: @1e-6 は 121/138 最適解、timeout 0。ミスは SuboptimalSolution 1 件 (`UBH1`)、Stalled 9 件（非収束で解を主張しない）、MaxIterations 2 件、OBJ_MISMATCH 1 件 (`LISWET7`)、公開参照値なしの検査済み 4 件。@1e-8 は 93/138 最適解、SuboptimalSolution 42 件、TIMEOUT 1 件 (`POWELL20`)、公開参照値なしの検査済み 2 件（この列は上記 taxonomy 分割より前の計測で、今回は再測定していない）。

QCQP（QPLIB、`bench_qplib` による単発 IPM — 本ベンチは `--global` の空間 B&B 経路を使わない）: @1e-6 は 11/41 最適解、TIMEOUT 3 件。非該当の内訳は SuboptimalSolution 3 件、Stalled 8 件（非収束の IPM iterate、解を主張しない — 旧 taxonomy が SuboptimalSolution へ丸めていたものの誠実な置き換え）、NOT_SUPPORTED 11 件（非凸 McCormick 緩和は全変数の有限境界を要求するが、これらの問題には非有界な変数がある）、SKIP 5 件（parse 時点の対象外: 整数変数または非対応の制約型）。@1e-8 は 8/41 最適解、SuboptimalSolution 4 件、Stalled 9 件（TIMEOUT 4 件。NOT_SUPPORTED/SKIP は eps に依存しないため不変）。@1e-4 まで緩めると 15/41 最適解、SuboptimalSolution 2 件、Stalled 5 件、TIMEOUT 3 件まで回復する。

MILP: @1e-6 / @1e-8 とも 5/20 最適解（`flugpl`、`gr4x6`、`gt2`、`khb05250`、`p0201`）。どちらも `TOTAL` 内に TIMEOUT 15 件、ERROR 0 件を計上する。`noswot` と `timtab1` は tree-cut separation の panic ではなく TIMEOUT になる。

SOCP: Otspot を Hans Mittelmann の [Large Second-Order Cone benchmark](https://plato.asu.edu/ftp/socp.html)（CBLIB 18 問、2026-06-29 版）で計測する。同ページには MOSEK/ECOS/KNITRO/COPT/cuOpt の公開実行時間（timeout 1 時間）が併載されている。これは従来の恣意的な 22 問 self-baseline を置き換えるもので、下表の商用/OSS 時間は Otspot 自身の値ではなく*外部の物差し*である。**Otspot の時間はメモリ制約のある 19 GB QEMU VM（8 vCPU）での計測であり、Mittelmann の Intel i7-11700K / 64 GB ではない**。したがって絶対秒数は方向性の目安であり、`firL2Linfalph`（1.22 億 nonzeros、入力 2.76 GB）がここで 18 GB 上限に達して OOM するのは 64 GB のメモリ余裕がないためである。Otspot の `Optimal` は 1e-6 での proof-carrying KKT 収束を指す。CBLIB/Mittelmann は目的値を公表しないため、これらは商用解との数値照合はしていない。

Otspot は **標準 1000s では 4/18** を解く。ベンチの 3600s 上限に延長すると `firL2L1alph`（1064s）と `firL2Linfeps`（1841s）が加わり **6/18** となる（他の timeout 問題は 3600s まで網羅的には再測していないため、真の 3600s 到達数はさらに多い可能性がある）。最も近い OSS SOCP ソルバである ECOS は 11/18 を解く。商用の MOSEK / COPT は 18 問すべてを解き、Mittelmann の公開 shifted geometric mean は 1.35 / 約 1（最速ソルバを 1 とする無次元の正規化値、彼のハードウェア。各問の生の秒数は下表）。

実行時間（秒。Otspot は実測、MOSEK/ECOS/COPT は Mittelmann の公開値。`f` = そのソルバが失敗、`nnz` = 非ゼロ要素数）:

| 問題 | nnz | Otspot | MOSEK | ECOS | COPT |
|---|---:|---|---:|---:|---:|
| chainsing-50000-1 | 0.9M | **6** | 3 | f | 3 |
| chainsing-50000-2 | 0.75M | **8** | 4 | f | 3 |
| chainsing-50000-3 | 0.6M | **6** | 3 | f | 2 |
| beam7 | 15M | 481 | 17 | 206 | 18 |
| firL2L1alph | 10M | 1064 | 6 | 202 | 5 |
| firL2Linfeps | 19M | 1841 | 25 | 687 | 14 |
| firL1Linfeps | 9.9M | timeout (>3600s) | 26 | 2531 | 13 |
| firL1 | 40M | timeout (>1000s) | 16 | 1305 | 9 |
| firL1Linfalph | 80M | timeout (>1000s) | 56 | 2847 | 23 |
| firL2L1eps | 40M | timeout (>1000s) | 14 | 797 | 9 |
| firL2a | 50M | timeout (>1000s) | 3 | 945 | 4 |
| firLinf | 80M | timeout (>1000s) | 95 | 3479 | 27 |
| wbNRL | 39M | timeout (>1000s) | 9 | 1333 | 7 |
| dsNRL | 67M | timeout (>1000s) | 56 | f | 27 |
| beam30 | 64M | timeout (>1000s) | 99 | 2465 | 84 |
| db-joint-soerensen | 6M | timeout (>1000s) | 29 | f | 46 |
| db-plate-yield-line | 1.5M | timeout (>1000s) | 6 | f | 5 |
| firL2Linfalph | 122M | OOM (>18 GB) | 27 | f | 25 |
| **solved** | | **6/18** | 18/18 | 11/18 | 18/18 |

Otspot が勝つ点: `chainsing-50000` 3 問（5 万回転錐、約 100 万 nonzeros）を 6〜8s で解く一方、ECOS は 3 問とも失敗する（MOSEK/COPT は約 3s）。Otspot が負ける点: 大規模で密なヤコビアンを持つ `fir`/`db` 系（1000 万〜1.22 億 nonzeros）は timeout するかメモリを使い果たす—MOSEK/COPT が数秒、ECOS（解ける 11 問）が数分で片付ける規模に、Otspot の錐 IPM はまだスケールしない。Otspot は発展途上の OSS SOCP ソルバであり、構造的に疎な錐問題では競争力があるが、大規模で密な問題ではまだ及ばない。（ソルバは錐種 `F`/`L±`/`L=`/`Q`/`QR` と分枝限定による MISOCP にも対応する。`EXP` と PSD 錐は非対応として拒否する。）

再現（データは gitignored、[ベンチマークデータ](#ベンチマークデータ)参照）:

```bash
for eps in 1e-6 1e-8; do
  bash scripts/run_lp_bench.sh --suite standard --eps "$eps" --jobs 6 --timeout 1000
  bash scripts/run_lp_bench.sh --suite infeas --eps "$eps" --jobs 6 --timeout 1000
  bash scripts/bench_parallel.sh --data-dir data/lp_problems_unbounded --eps "$eps" --jobs 6 \
       --timeout 1000 --output "/tmp/lp_unbounded_${eps}.txt"
  bash scripts/bench_parallel.sh --data-dir data/maros_meszaros --eps "$eps" --jobs 6 \
       --timeout 1000 --output "/tmp/qp_maros_${eps}.txt"
  bash scripts/bench_parallel.sh --data-dir data/qplib --eps "$eps" --jobs 6 \
       --timeout 1000 --output "/tmp/qplib_${eps}.txt"
  bash scripts/bench_parallel.sh --data-dir data/miplib_small --eps "$eps" --jobs 6 \
       --timeout 1000 --output "/tmp/miplib_small_${eps}.txt"  # ERROR / 外部timeout がある場合は非ゼロ終了
done
```

SOCP には専用の `bench_parallel.sh` ハーネスがまだない（同スクリプトは `.mps`/`.qps`/`.qplib` のみ対応）。`solve_cbf` example を外部 timeout で直接実行し、問題ごとに結果ファイルへ出力する。**必ず逐次実行する（jobs=1）**—大規模な `fir` 系は 1 問あたり最大 ~18 GB RSS を要するため、並列実行すると集団でメモリを使い果たし abort する。`timeout` の終了コードは直接取得する（`solve_cbf` を `grep` にパイプしてから `$?` を読むと、パイプの `$?` は grep の終了コードになり 124 の timeout を取りこぼす）:

```bash
cargo build --release --example solve_cbf
out=/tmp/socp18; mkdir -p "$out"
# Mittelmann Large-SOCP 18。CBLIB のダウンロード名は問題ごとに異なる（多くは
# 2013_<name> だが beam7/beam30/chainsing-* は 2013_ 接頭辞なし）—
# plato.asu.edu/ftp/socp.html に従って調整する。
for n in beam7 beam30 chainsing-50000-1 chainsing-50000-2 chainsing-50000-3 \
         db-joint-soerensen db-plate-yield-line dsNRL firL1 firL1Linfalph \
         firL1Linfeps firL2L1alph firL2L1eps firL2Linfalph firL2Linfeps \
         firL2a firLinf wbNRL; do
  f="data/cblib/$n.cbf"
  timeout 3600 ./target/release/examples/solve_cbf --eps 1e-6 "$f" > "$out/$n.csv" 2>/dev/null
  rc=$?   # パイプなし: 124 = timeout、137/134 = OOM、0 = CSV の status を参照
  [ "$rc" = 124 ] && echo "$n,Timeout,,,3600.0" >> "$out/$n.csv"
done
grep -hv '^problem,' "$out"/*.csv   # 問題ごとの status,objective,iters,time
```

## テスト

```bash
cargo nextest run --release --test-threads 6          # 全スイート (data/ 必須)
cargo nextest run --release --profile lib-only       # lib + bin テスト (kind=lib + kind=bin)、統合データ不要
cargo test --doc --release
```

統合テストは `data/` の存在を assert し、なければ panic する（`--profile lib-only` で回避）。

## 開発環境 (Docker)

```bash
docker build -f Dockerfile.dev -t otspot-dev .
docker run -it --rm -v "$PWD":/workspace -w /workspace otspot-dev bash
```

### ベンチマークデータ

```bash
bash scripts/download_all_bench_data.sh          # Netlib LP + Maros-Meszaros + QPLIB + 合成 QP
bash scripts/download_all_bench_data.sh --lp     # LP のみ
bash scripts/download_all_bench_data.sh --check  # 取得状況確認
```

合成 QP データ生成には `numpy scipy cvxpy clarabel` が必要（`pip install`）。

## プロジェクト構造

Cargo workspace:

```
otspot/          # facade クレート — core / io / model の公開 re-export
otspot-core/     # ソルバーエンジン (simplex, IPM, B&B, presolve, linalg, sparse)
otspot-io/       # ファイルパーサ (MPS, QPS, QPLIB)
otspot-model/    # 代数モデリング API (Model, Variable, constraint! マクロ)
otspot-dev/      # dev 専用バイナリ (qps_benchmark, qp_runner など。非公開)
examples/        # solve_lp, solve_qp
tests/           # 統合テスト
scripts/         # データ生成スクリプト
```

## ライセンス

[GNU Affero General Public License v3.0 (AGPL-3.0-only)](LICENSE)
