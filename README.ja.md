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

### Python

`otspot-py` クレート（PyO3/maturin）は同じ `Model` API をシンボル単位で Python に公開する:

```python
import otspot

model = otspot.Model("example")
x = model.add_var("x", 0.0, float("inf"))
y = model.add_var("y", 0.0, 10.0)
model.add_constraint((2.0 * x + 3.0 * y).leq(12.0))
model.add_constraint((x + y).geq(3.0))
model.minimize(x + 2.0 * y)

result = model.solve()
print(result.status, result.objective_value, result.value(x), result[y])
```

インストール方法、フル API（二次目的、整数/バイナリ変数、エラーハンドリング）、注意点（演算子オーバーロードが
Rust の `constraint!` マクロと異なる、スレッド安全性、`solve()` 中の GIL 解放、pickle 化）については
[`otspot-py/README.md`](otspot-py/README.md) を参照。

## 性能

標準公開セットでの求解率ベンチ。`otspot-dev` の benchmark harness（shell スクリプト — **`cargo bench` ではない**）で計測。2026-08-08 に commits `e4895ff3`–`e2f184d3` 時点（この間の差分は docs のみで、ソルバコードは同一）で実測:
`jobs = 6`、`timeout = 1000s`、`eps ∈ {1e-6, 1e-8}`（SOCP は `jobs = 1`、詳細は後述）。

| 問題種別 | セット | 問題数 | @1e-6 | @1e-8 |
|---|---|---:|---|---|
| 実行可能 LP | Netlib | 109 | 最適解 109 | 最適解 108、Suboptimal 1 |
| 凸 QP | Maros–Mészáros | 138 | 最適解 121、Suboptimal 1、Stalled 10、mismatch 2、no-ref 4 | 最適解 97、Suboptimal 4、Stalled 34、Timeout 1、no-ref 2 |
| QCQP | QPLIB | 41 | 最適解 12、Suboptimal 3、Stalled 7、Timeout 3、not-supported 11、skip 5 | 最適解 10、Suboptimal 4、Stalled 7、Timeout 4、not-supported 11、skip 5 |
| MILP | MIPLIB 2017 small | 20 | 最適解 7、Timeout 13、error 0 | 最適解 7、Timeout 13、error 0 |
| 実行不可能 LP | Netlib | 29 | 正答 29 | 正答 29 |
| 非有界 LP | 合成 | 12 | 正答 12 | 正答 12 |

**最適解** = 既知の目的値と照合済み（proof-carrying KKT 検証）。**Stalled** = IPM が反復・時間予算内でこれ以上進展せず、解を主張しない status。**Suboptimal** = 解いたが厳密な最適性証明には失敗。**no-ref** = 公開された目的値がなく照合できないまま解いた。

LP: ミスは `greenbea`（@1e-8 で Suboptimal）のみ。

QP: 主なミスは `UBH1`（両 eps で Suboptimal）、`LISWET7`（@1e-6 で目的値 mismatch、@1e-8 で Stalled）、`POWELL20`（@1e-8 で Timeout）。

QCQP（QPLIB、`bench_qplib` による単発 IPM — `--global` の空間 B&B 経路は使わない）: `not-supported` は McCormick 緩和が全変数の有限境界を要求するが非有界な変数を含むケース、`skip` は整数変数または非対応の制約型を含むケース。いずれも eps に依存しない。

MILP: 両 eps で最適解は `dcmulti`、`flugpl`、`gr4x6`、`gt2`、`khb05250`、`markshare_4_0`、`p0201`。`noswot` と `timtab1` は両 eps で timeout する。

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
       --timeout 1000 --output "/tmp/miplib_small_${eps}.txt"  # ERROR / 外部 timeout がある場合は非ゼロ終了
done
```

### SOCP

Otspot を Hans Mittelmann の [Large Second-Order Cone benchmark](https://plato.asu.edu/ftp/socp.html)（CBLIB 18 問）で計測する。同ページは MOSEK・ECOS・COPT の実行時間（1 時間上限）を公開しており、下表のこの 3 列は Mittelmann の公開値でここでの実測ではない。CBLIB/Mittelmann は目的値を公表しないため、Otspot の結果は商用ソルバの最適値との数値照合はしていない（ここでの `Optimal` は 1e-6 での KKT 収束のみを指す）。Otspot 自身の実行は **`jobs = 1`** — 大規模問題は 1 問あたり最大 ~18 GB RSS を要し、並列実行するとメモリを使い果たすため — かつメモリ制約のある 19 GB VM（8 vCPU）上での計測であり、Mittelmann の 64 GB マシンではないため、絶対秒数は方向性の目安である。

下表の 18 行はすべて今回の実測であり、2026-08-08〜09 に `eps = 1e-6`、`timeout = 1000s`、メモリ上限 14 GB のスコープ下で計測した。

| 問題 | nnz | Otspot | MOSEK | ECOS | COPT |
|---|---:|---|---:|---:|---:|
| chainsing-50000-1 | 0.9M | **5.8** | 3 | f | 3 |
| chainsing-50000-2 | 0.75M | **7.0** | 4 | f | 3 |
| chainsing-50000-3 | 0.6M | **4.9** | 3 | f | 2 |
| beam7 | 15M | 434.5 | 17 | 206 | 18 |
| db-plate-yield-line | 1.5M | timeout (>1000s) | 6 | f | 5 |
| db-joint-soerensen | 6M | timeout (>1000s) | 29 | f | 46 |
| firL2L1alph | 10M | timeout (>1000s) | 6 | 202 | 5 |
| firL1Linfeps | 9.9M | timeout (>1000s) | 26 | 2531 | 13 |
| firL2Linfeps | 19M | timeout (>1000s) | 25 | 687 | 14 |
| firL1 | 40M | timeout (>1000s) | 16 | 1305 | 9 |
| firL1Linfalph | 80M | timeout (>1000s) | 56 | 2847 | 23 |
| firL2L1eps | 40M | timeout (>1000s) | 14 | 797 | 9 |
| firL2a | 50M | timeout (>1000s) | 3 | 945 | 4 |
| firLinf | 80M | timeout (>1000s) | 95 | 3479 | 27 |
| wbNRL | 39M | timeout (>1000s) | 9 | 1333 | 7 |
| dsNRL | 67M | timeout (>1000s) | 56 | f | 27 |
| beam30 | 64M | timeout (>1000s) | 99 | 2465 | 84 |
| firL2Linfalph | 122M | OOM (>14 GB cap) | 27 | f | 25 |
| **solved** | | **4/18** | 18/18 | 11/18 | 18/18 |

Otspot は `chainsing-50000` 3 問（回転錐、約 100 万 nonzeros）を 5〜7s で解く一方、ECOS は 3 問とも失敗する。大規模で密なヤコビアンを持つ `fir`/`db`/`beam30`/`dsNRL` 系（600 万〜1.22 億 nonzeros）は timeout するかメモリを使い果たす。対応する錐種は `F`/`L±`/`L=`/`Q`/`QR` と分枝限定による MISOCP。`EXP` と PSD 錐は非対応として拒否する。

SOCP には専用の `bench_parallel.sh` ハーネスがまだない（同スクリプトは `.mps`/`.qps`/`.qplib` のみ対応）。`solve_cbf` example を外部 timeout で直接実行し、問題ごとに結果ファイルへ出力する。**必ず逐次実行する（jobs=1）**（理由は上記のメモリ要件のとおり）。`timeout` の終了コードは直接取得する（`solve_cbf` を `grep` にパイプしてから `$?` を読むと、パイプの `$?` は grep の終了コードになり 124 の timeout を取りこぼす）:

```bash
cargo build --release --example solve_cbf
out=/tmp/socp18; mkdir -p "$out"
# Mittelmann Large-SOCP 18、data/cblib_full/ に配置。CBLIB のダウンロード名は
# 問題ごとに異なり（fir*/dsNRL/wbNRL は 2013_ 接頭辞つき、beam7/beam30/
# chainsing-*/db-* はなし。plato.asu.edu/ftp/socp.html 参照）、一部は
# gzip 圧縮のまま残っている。
for n in beam7 beam30 chainsing-50000-1 chainsing-50000-2 chainsing-50000-3 \
         db-joint-soerensen db-plate-yield-line dsNRL firL1 firL1Linfalph \
         firL1Linfeps firL2L1alph firL2L1eps firL2Linfalph firL2Linfeps \
         firL2a firLinf wbNRL; do
  case "$n" in fir*|dsNRL|wbNRL) stem="2013_$n" ;; *) stem="$n" ;; esac
  f="data/cblib_full/$stem.cbf"
  if [ ! -f "$f" ]; then
    # フレッシュクローンで 0 バイトを作らない: .gz 必須、一時ファイルへ展開し
    # 成功時のみ mv する（gunzip 失敗で壊れた .cbf を再利用させない）。
    [ -f "$f.gz" ] || { echo "$n,Missing,,," >> "$out/$n.csv"; continue; }
    gunzip -c "$f.gz" > "$f.tmp" && mv "$f.tmp" "$f" \
      || { echo "$n,DecompressFailed,,," >> "$out/$n.csv"; continue; }
  fi
  timeout 1000 ./target/release/examples/solve_cbf --eps 1e-6 "$f" > "$out/$n.csv" 2>/dev/null
  rc=$?   # パイプなし: 0 = status は CSV 内; それ以外は書き込み前に異常終了
  case "$rc" in
    0) : ;;
    124) echo "$n,Timeout,,,1000.0" >> "$out/$n.csv" ;;       # timeout 1000
    137|134) echo "$n,OOM,,," >> "$out/$n.csv" ;;             # 14 GB 上限
    *) echo "$n,Error$rc,,," >> "$out/$n.csv" ;;
  esac
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
otspot-py/       # Python バインディング (PyO3/maturin)。default-members には含まれない
otspot-dev/      # dev 専用バイナリ (qps_benchmark, qp_runner など。非公開)
examples/        # solve_lp, solve_qp
tests/           # 統合テスト
scripts/         # データ生成スクリプト
```

## ライセンス

[GNU Affero General Public License v3.0 (AGPL-3.0-only)](LICENSE)
