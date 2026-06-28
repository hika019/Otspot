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
| 実行可能 LP | Netlib | 109 | 最適解 108、SuboptimalSolution 1 | 最適解 107、PFEAS_FAIL 1、SuboptimalSolution 1 |
| 凸 QP | Maros–Mészáros | 138 | 最適解 121、SuboptimalSolution 12、OBJ_MISMATCH 1、参照値なし 4 | 最適解 93、SuboptimalSolution 42、TIMEOUT 1、参照値なし 2 |
| MILP | MIPLIB 2017 small | 20 | 最適解 5、TIMEOUT 13、ERROR 2 | 最適解 5、TIMEOUT 13、ERROR 2 |
| 実行不可能 LP | Netlib | 29 | 正答 29 | 正答 29 |
| 非有界 LP | 合成 | 12 | 正答 12 | 正答 12 |

**最適解** = 既知最適値と照合済み（proof-carrying KKT）。`timeout = 1000s`、`jobs = 6` で計測。

LP: @1e-6 は 108/109 最適解、timeout 0。ミスは `cycle` (SuboptimalSolution)。@1e-8 は 107/109 最適解、timeout 0。ミスは `greenbea` (PFEAS_FAIL) と `cycle` (SuboptimalSolution)。

QP: @1e-6 は 121/138 最適解、timeout 0。ミスは SuboptimalSolution 12 件、OBJ_MISMATCH 1 件 (`LISWET7`)、公開参照値なしの検査済み 4 件。@1e-8 は 93/138 最適解、SuboptimalSolution 42 件、TIMEOUT 1 件 (`POWELL20`)、公開参照値なしの検査済み 2 件。

MILP: @1e-6 / @1e-8 とも 5/20 最適解（`flugpl`、`gr4x6`、`gt2`、`khb05250`、`p0201`）。どちらも `TOTAL` 内に TIMEOUT 13 件、ERROR 2 件を計上した。ERROR は `noswot` と `timtab1`（`no_output_exit=101`）。

再現（データは gitignored、[ベンチマークデータ](#ベンチマークデータ)参照）:

```bash
for eps in 1e-6 1e-8; do
  bash scripts/run_lp_bench.sh --suite standard --eps "$eps" --jobs 6 --timeout 1000
  bash scripts/run_lp_bench.sh --suite infeas --eps "$eps" --jobs 6 --timeout 1000
  bash scripts/bench_parallel.sh --data-dir data/lp_problems_unbounded --eps "$eps" --jobs 6 \
       --timeout 1000 --output "/tmp/lp_unbounded_${eps}.txt"
  bash scripts/bench_parallel.sh --data-dir data/maros_meszaros --eps "$eps" --jobs 6 \
       --timeout 1000 --output "/tmp/qp_maros_${eps}.txt"
  bash scripts/bench_parallel.sh --data-dir data/miplib_small --eps "$eps" --jobs 6 \
       --timeout 1000 --output "/tmp/miplib_small_${eps}.txt"  # ERROR / 外部timeout がある場合は非ゼロ終了
done
```

## テスト

```bash
cargo nextest run --release --test-threads 3          # 全スイート (data/ 必須)
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
