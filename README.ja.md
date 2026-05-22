# otspot

[English](README.md) | **日本語**

Rust で書かれた**数理最適化ソルバー**。

LP には**修正シンプレックス法**（疎LU分解、Ruiz均衡スケーリング、最急勾配価格決定）、QP には**内点法**（Mehrotra predictor–corrector / IP-PMM）を実装し、混合整数問題の branch-and-bound を備える。実行不可能・非有界も判定し、完全な主双対情報を返す。

## 機能

- **代数モデリングAPI** — 自然な数式記法で問題を表現
- **修正シンプレックス法（LP）** — 疎LU分解と Markowitz 閾値ピボット
- **内点法（QP）** — 凸QPに対するMehrotra predictor–corrector / IP-PMM
- **混合整数（MILP / 凸MIQP）** — 連続緩和（MILPはLP緩和、凸MIQPはQP緩和）上の baseline branch-and-bound（most-fractional 分枝）。非凸MIQPはスコープ外。カット・主発見的手法・SOS制約・より高度な分枝戦略は未実装。
- **実行不可能・非有界の判定** — 単なる失敗ではなく明示的なステータスを返す
- **Ruiz均衡化** — 数値条件を改善する行/列スケーリング前処理
- **最急勾配価格決定** — 改善された変数選択で収束を高速化
- **双対解出力** — 双対変数、簡約費用、制約スラック
- **入力フォーマット** — MPS（LP）と QPS / QPLIB（QP）
- **設定可能なオプション** — 許容誤差、反復回数上限、LU再分解閾値
- **ベンチ + ファズテスト** — criterion マイクロベンチと proptest ランダム化テスト

## クイックスタート

必要環境: Rust（edition 2021, stable）。crates.io には未公開のため、git 依存またはソースビルドで利用する。

git 依存として:

```toml
[dependencies]
otspot = { git = "https://github.com/hika019/otspot" }
```

ソースからビルド・動作確認:

```bash
git clone https://github.com/hika019/otspot.git
cd otspot
cargo build --release
cargo run --release --example solve_lp   # LP の最小例
cargo run --release --example solve_qp   # QP の最小例
```

### モデリングAPI

LP問題を定義して解く推奨の方法:

```rust
use otspot::model::{Model, constraint};

fn main() {
    // Problem:
    //   minimize    x + 2y
    //   subject to  2x + 3y <= 12
    //               x  +  y >= 3
    //               x in [0, +inf), y in [0, 10]

    let mut model = Model::new("production");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, 10.0);

    model.add_constraint(constraint!((2.0 * x + 3.0 * y) <= 12.0));
    model.add_constraint(constraint!((x + y) >= 3.0));
    model.minimize(x + 2.0 * y);

    let result = model.solve().unwrap();
    println!("objective = {}", result.objective());
    println!("x = {}", result[x]);
    println!("y = {}", result[y]);
}
```

**出力:**
```
objective = 3
x = 3
y = 0
```

### `constraint!` マクロ

`constraint!` マクロは、単一変数および括弧で囲まれた式に対して自然な不等式構文をサポートする:

```rust
// Single variable
model.add_constraint(constraint!(x <= 7.0));
model.add_constraint(constraint!(y >= 0.0));
model.add_constraint(constraint!(x == 5.0));

// Expression on the left-hand side (wrap in parentheses)
model.add_constraint(constraint!((2.0 * x + y) <= 10.0));
```

または、式に直接メソッドAPIを使用することもできる:

```rust
model.add_constraint((x + 2.0 * y).leq(8.0));
model.add_constraint((x - y).geq(0.0));
model.add_constraint((x + y).eq_constraint(5.0));
```

### 最大化

```rust
let mut model = Model::new("revenue");
let x = model.add_var("x", 0.0, f64::INFINITY);
let y = model.add_var("y", 0.0, f64::INFINITY);

model.add_constraint(constraint!((x + y) <= 10.0));
model.maximize(3.0 * x + 5.0 * y);

let result = model.solve().unwrap();
println!("max revenue = {}", result.objective());
```

### 許容誤差とオプション

収束許容誤差（eps）やよく使う設定は model に直接指定できる。`Tolerance` には
preset `High`（1e-8）/ `Medium`（1e-6, 既定）/ `Fast` と `Custom(f64)` がある:

```rust
use otspot::Tolerance;

model.set_tolerance(Tolerance::Custom(1e-8)); // eps = 1e-8
model.set_presolve(true);                      // presolve の有効/無効 (既定: 有効)
model.set_timeout(60.0);                       // 実時間上限 (秒)
```

より細かい制御は低レベルの `solve_with` に `SolverOptions` を渡す:

```rust
use otspot::SolverOptions;
use otspot::solve_with;

let opts = SolverOptions {
    primal_tol: 1e-8,   // LP simplex の最適性 / 実行可能性 許容誤差
    max_etas: 50,       // LU 再分解閾値 (0 = auto)
    clamp_tol: 1e-14,   // 解の微小値クランプ
    ..Default::default()
};
let result = solve_with(&problem, &opts); // problem: &LpProblem
```

### 双対解

主解に加えて、`model.solve()` は双対変数・簡約費用・制約スラックを返す（いずれも `Option<Vec<f64>>`）:

```rust
let result = model.solve().unwrap();
println!("dual (shadow): {:?}", result.dual_solution);
println!("reduced costs: {:?}", result.reduced_costs);
println!("slacks:        {:?}", result.slack);
```

### 二次計画法（QP）

QP は LP と同じモデリング API で書ける — 二次の目的関数を加えるだけ。`set_diagonal_q`
（対角 Q の簡易版）または `set_quadratic_objective`（`CscMatrix` 全体）で設定する。目的関数は
「1/2あり」規約 min ½·xᵀQx + cᵀx で、線形項 c は `minimize` / `maximize` で与える。

```rust
use otspot::model::{constraint, Model};

// min  x² + y²        (= ½·xᵀQx, Q = diag(2, 2))
// s.t. x + y >= 1
fn main() {
    let mut model = Model::new("qp");
    let x = model.add_var("x", f64::NEG_INFINITY, f64::INFINITY);
    let y = model.add_var("y", f64::NEG_INFINITY, f64::INFINITY);

    model.add_constraint(constraint!((x + y) >= 1.0));
    model.set_diagonal_q(&[2.0, 2.0]); // Q = diag(2, 2)
    model.minimize(0.0 * x + 0.0 * y); // 線形項 c = 0

    let result = model.solve().unwrap();
    println!("objective = {:.4}", result.objective());
    println!("x = {:.4}, y = {:.4}", result[x], result[y]);
}
```

**出力:**
```
objective = 0.5000
x = 0.5000, y = 0.5000
```

行列を直接渡したい場合は低レベルの `qp::solve_qp` / `QpProblem` API（Q, c, A, b, bounds を配列で）も使える。SQP 反復での warm-start には `solve_qp_warm`（前回解の活性集合を引き継ぐ）。

## 応用

高性能が求められるアプリケーションでは、制約行列をCSCフォーマットで直接構築し、低レベルAPIを呼び出せ:

```rust
use otspot::problem::LpProblem;
use otspot::sparse::CscMatrix;
use otspot::solve;

// minimize  -x1 - x2
// s.t.       x1 + x2 <= 4
//            x1      <= 3
//                 x2 <= 3
//            x1, x2 >= 0

let c = vec![-1.0, -1.0];

let rows = vec![0, 0, 1, 2];
let cols = vec![0, 1, 0, 1];
let vals = vec![1.0, 1.0, 1.0, 1.0];
let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 2).unwrap();

let b = vec![4.0, 3.0, 3.0];

let problem = LpProblem::new(c, a, b).unwrap();
let result = solve(&problem);

println!("status:    {}", result.status);    // Optimal
println!("objective: {}", result.objective);  // -4
println!("solution:  {:?}", result.solution); // [1.0, 3.0]
```

## MPS入力

MPSファイルからLP問題を読み込む:

```rust
use std::path::Path;
use otspot::io::mps;
use otspot::solve;

let prob = mps::parse_mps_file(Path::new("problem.mps")).expect("MPS parse error");
let result = solve(&prob);
println!("status: {}", result.status);
```

パーサは Netlib LP セットおよび Maros–Mészáros QP セットで検証済みである（[性能](#性能)参照）。

## 性能

標準的な公開ベンチマークセットで計測（`timeout = 1000s`、6並列、2つの許容誤差）。
**最適解** = 既知最適値に対し最適性を検証済み。**有効解** = ソルバーの最適性判定は満たすが
外部参照値が無く検証できない実行可能解（*最適解* とは別カウント）。実行不可能・非有界の
セットは正しい証明書を返せたかで判定する。

| 問題種別 | セット | 問題数 | @1e-6 | @1e-8 |
|---|---|---:|---|---|
| 実行可能 LP | Netlib | 109 | 最適解 109 | 最適解 106 |
| 凸 QP | Maros–Mészáros | 138 | 最適解 128・有効解 7 | 最適解 123・有効解 4 |
| 実行不可能 LP | Netlib | 29 | 正答 28¹ | 正答 28¹ |
| 非有界 LP | 合成 | 12 | 正答 12 | 正答 12 |

¹ 1問 未解決（`klein3`: simplex が停滞し打ち切り）。

`1e-6` の QP 未達は `LISWET12`（timeout）、`LISWET9`（目的関数 約9%ずれ。cond ≈ 1e16 の
f64 LDLᵀ 精度限界）、`QBORE3D`（双対残差 7.5e-4 で停滞）。より厳しい `1e-8` では
ill-conditioned 問題（主に LISWET 系）が主双対残差の閾値をわずかに超過 — 精度フロアの影響で収束失敗ではない。

ベンチデータは gitignore 対象かつ再現可能（[ベンチマークデータ](#ベンチマークデータ)参照）。

## ベンチマーク（criterion）

4種類の criterion マイクロベンチマークスイートが含まれている:

```bash
# All benchmarks
cargo bench

# Individual suites
cargo bench --bench scaling_pricing   # Ruiz scaling + steepest-edge pricing
cargo bench --bench lu_bench          # LU factorization throughput
cargo bench --bench solve_bench       # end-to-end LP solve
cargo bench --bench qp_bench          # QP solve
```

HTMLレポートは `target/criterion/` に生成される。

## テスト

```bash
# Full test suite (unit + Netlib + proptest) — bench data 必須
cargo nextest run --release

# unit / bin test のみ (bench data 不要)
cargo nextest run --release --profile lib-only

# doc test
cargo test --doc --release
```

`tests/*.rs` の多くは `data/lp_problems_*/`, `data/qplib/` 等を `assert!(path.exists())` で
要求し、data 欠落で **panic** する（プロジェクト原則「SKIP せず panic」で検証空白を作らない）。
data を整備していない環境では `--profile lib-only` で unit + bin test のみ走らせる。

bench data 取得は [開発環境 (Docker)](#開発環境-docker) section 参照、もしくは
`bash scripts/download_all_bench_data.sh`。

テストスイートに含まれるもの:
- 全モジュールの**ユニットテスト**
- 実世界検証のための**Netlibインスタンス integration**
- ランダム化ファズテストのための**proptest**
- 基本的なAPIカバレッジのための**スモークテスト**

## 開発環境 (Docker)

`Dockerfile.dev` は他環境で test / bench を再現するための開発用 container 定義（Rust 1.83
+ python3 (numpy/scipy/cvxpy/clarabel) + cargo-nextest + Netlib emps decoder pre-compile）。

```bash
# 1. Build (初回 ~5-10 分)
docker build -f Dockerfile.dev -t otspot-dev .

# 2. Interactive 開発 (source を host と共有、保存即反映)
docker run -it --rm -v "$PWD":/workspace -w /workspace otspot-dev bash

# 3. One-shot test
docker run --rm -v "$PWD":/workspace -w /workspace otspot-dev \
  cargo nextest run --release
```

### Bare host 実行 (Docker を使わない場合)

bench data 生成 script (`scripts/gen_*.py`) は以下の Python pkg を要求:

```bash
pip install numpy scipy cvxpy clarabel
```

`cvxpy` / `clarabel` は `osqp_bench` 系 (`osqp_bench`, `osqp_bench_*`, `qp_dense_a`)
の生成器でのみ必要。LP suite (`lp_problems*`) は `curl` + `emps` のみで動く。

`scripts/download_all_bench_data.sh` は QP モード突入時に numpy/scipy の
有無を check し、不在なら Docker / pip の手順を案内して exit する。

### ベンチマークデータ

`data/` 配下は `.gitignore` 対象なので、clone 後に bench data を自前生成する必要がある:

```bash
# 全部 (Netlib LP + 合成 QP)
bash scripts/download_all_bench_data.sh

# LP のみ (Netlib 取得 + 合成、決定論的)
bash scripts/download_all_bench_data.sh --lp

# 取得状況確認
bash scripts/download_all_bench_data.sh --check
```

合成系は固定 seed なので任意環境で同一出力を再現可。Maros–Mészáros / QPLIB は
download script 未整備で手動配置が必要 (URL ヒントは `download_all_bench_data.sh` 内)。
取得状況は `--check` で確認できる。

## プロジェクト構造

```
src/
├── lib.rs              # クレートのエントリポイント・公開API再エクスポート
├── model/              # 高レベル代数モデリングAPI (Model、constraint!マクロ)
├── lp.rs               # LP求解エントリ
├── simplex/            # 修正シンプレックス (primal / dual)
├── qp/                 # QP求解 (内点法 IPM / IP-PMM、postsolve)
├── mip/                # 混合整数 (MILP / MIQP) branch-and-bound
├── presolve/           # 前処理 (Ruizスケーリング、postsolve)
├── linalg/             # 線形代数 (LU、LDLᵀ)
├── basis/              # 基底管理
├── sparse/             # CSC疎行列・疎ベクトル
├── problem/            # LpProblem / QpProblem、SolverResult、SolveStatus
├── screening.rs        # 問題スクリーニング
├── options.rs          # SolverOptions
├── tolerances.rs       # 数値許容誤差定数
├── error.rs            # SolverError
├── io/                 # 入力パーサ (mps / qps / qplib)
└── bin/                # CLIツール (qp_runner、qp_diag、qps_benchmark ほか)
examples/               # 利用例 (solve_lp、solve_qp)
benches/                # Criterionベンチ (lu_bench、qp_bench、solve_bench、scaling_pricing)
```

## ライセンス

[GNU Affero General Public License v3.0 (AGPL-3.0-only)](LICENSE)
