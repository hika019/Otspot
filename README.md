# solver

Rustで書かれた高性能な線形計画法（LP）ソルバー。

疎LU分解、Ruiz均衡スケーリング、最急勾配価格決定を備えた**修正シンプレックス法**を実装しており、実世界のLPインスタンスに対して高い性能を発揮する。

## 機能

- **代数モデリングAPI** — 自然な数式記法でLP問題を表現
- **修正シンプレックス法** — 疎LU分解とMarkowitz閾値ピボットによるPhase I/II
- **Ruiz均衡化** — 数値条件を改善するための行/列スケーリング前処理
- **最急勾配価格決定** — 収束を高速化する改善された変数選択
- **双対解出力** — 双対変数、簡約費用、制約スラック
- **MPSファイル入力** — 業界標準MPSフォーマット読み込み；23件のNetlibインスタンスで検証済み
- **設定可能なオプション** — 許容誤差、反復回数上限、LU再分解閾値
- **ベンチマーク** — スケーリング、LU分解、ソルブのcriterionベースベンチマーク
- **ファズテスト** — proptestベースのランダム化テスト

## クイックスタート

必要環境: Rust (edition 2021, stable)。crates.io には未公開のため、git 依存またはソースビルドで利用する。

別プロジェクトから依存する場合（git 依存）:

```toml
[dependencies]
solver = { git = "https://github.com/hika019/solver" }
```

（左辺 `solver` は crate 名（`Cargo.toml` の `name`）、URL はリポジトリ。両者は独立に決まる。）

ソースからビルド・動作確認:

```bash
git clone https://github.com/hika019/solver.git
cd solver
cargo build --release
cargo run --release --example solve_lp   # LP の最小例
cargo run --release --example solve_qp   # QP の最小例
```

### モデリングAPI

LP問題を定義して解く推奨の方法:

```rust
use solver::model::{Model, constraint};

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

### SolverOptions

ソルバーの動作を細かく調整する:

```rust
use solver::SolverOptions;
use solver::problem::LpProblem;
use solver::simplex;

let opts = SolverOptions {
    primal_tol: 1e-8,          // optimality / feasibility tolerance
    max_iterations: Some(500), // None = auto (100*(m+n)+1000)
    max_etas: 50,              // LU refactorization threshold
    clamp_tol: 1e-14,          // solution micro-value clamp
};

let result = simplex::solve_with(&problem, &opts);
```

### 双対解

低レベルの `simplex::solve` と `simplex::solve_with` は完全な双対情報を含む `SolverResult` を返す:

```rust
use solver::problem::SolverResult;

let result: SolverResult = simplex::solve(&problem);
println!("primal:        {:?}", result.solution);
println!("dual (shadow): {:?}", result.dual_solution);
println!("reduced costs: {:?}", result.reduced_costs);
println!("slacks:        {:?}", result.slack);
```

### 二次計画法（QP）

`solve_qp` APIで二次計画問題を解く:

```rust
use solver::qp::{solve_qp, QpProblem};
use solver::sparse::CscMatrix;

// min  x^2 + y^2
// s.t. x + y >= 1
// (「1/2あり」規約: Q = [[2,0],[0,2]], min 1/2 x^T Q x)
fn main() {
    let q = CscMatrix::from_triplets(
        &[0, 1], &[0, 1], &[2.0, 2.0], 2, 2
    ).unwrap();
    let c = vec![0.0, 0.0];

    // x + y >= 1 → -x - y <= -1 (Ax <= b 形式)
    let a = CscMatrix::from_triplets(
        &[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2
    ).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];

    let problem = QpProblem::new(q, c, a, b, bounds).unwrap();
    let result = solve_qp(&problem);

    println!("status:    {:?}", result.status);
    println!("solution:  {:?}", result.solution);   // ≈ [0.5, 0.5]
    println!("objective: {:.4}", result.objective); // ≈ 0.5
}
```

**出力:**
```
status:    Optimal
solution:  [0.5, 0.5]
objective: 0.5000
```

SQP反復でのWarm-startには `solve_qp_warm` を使用する（前回解の活性集合を引き継ぎ収束を高速化）。

## 応用

高性能が求められるアプリケーションでは、制約行列をCSCフォーマットで直接構築し、低レベルAPIを呼び出せ:

```rust
use solver::problem::LpProblem;
use solver::sparse::CscMatrix;
use solver::simplex;

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
let result = simplex::solve(&problem);

println!("status:    {}", result.status);   // Optimal
println!("objective: {}", result.objective); // -4
println!("solution:  {:?}", result.solution);// [1.0, 3.0]
```

## MPS入力

MPSファイルからLP問題を読み込む:

```rust
use std::path::Path;
use solver::io::mps;
use solver::simplex;

let prob = mps::parse_mps_file(Path::new("problem.mps")).expect("MPS parse error");
let result = simplex::solve(&prob);
println!("status: {}", result.status);
```

このソルバーは23件のNetlibベンチマークインスタンス（adlittle、afiro、sc50a、sc50b、kb2、brandy、scorpion、fit1d、share1bなど）で検証済みである。

## ベンチマーク

3種類のcriterionベースベンチマークスイートが含まれている:

```bash
# All benchmarks
cargo bench

# Individual suites
cargo bench --bench scaling_pricing   # Ruiz scaling + steepest-edge pricing
cargo bench --bench lu_bench          # LU factorization throughput
cargo bench --bench solve_bench       # End-to-end LP solve
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
要求し、data 欠落で **panic** する (CLAUDE.md 原則「SKIP せず panic」で検証空白を作らない)。
data を整備していない環境では `--profile lib-only` で unit + bin test のみ走らせる。

bench data 取得は [開発環境 (Docker)](#開発環境-docker) section 参照、もしくは
`bash scripts/download_all_bench_data.sh`。

テストスイートに含まれるもの:
- 全モジュールの**ユニットテスト**
- 実世界検証のための**Netlibインスタンス integration**
- ランダム化ファズテストのための**proptest**
- 基本的なAPIカバレッジのための**スモークテスト**

## 開発環境 (Docker)

`Dockerfile.dev` は他環境で test / bench を再現するための開発用 container 定義 (Rust 1.83
+ python3 (numpy/scipy/cvxpy/clarabel) + cargo-nextest + Netlib emps decoder pre-compile)。

```bash
# 1. Build (初回 ~5-10 分)
docker build -f Dockerfile.dev -t solver-dev .

# 2. Interactive 開発 (source を host と共有、保存即反映)
docker run -it --rm -v "$PWD":/workspace -w /workspace solver-dev bash

# 3. One-shot test
docker run --rm -v "$PWD":/workspace -w /workspace solver-dev \
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

### ベンチマークデータの取得

`data/` 配下は `.gitignore` 対象なので、clone 後に bench data を自前生成する必要がある:

```bash
# 全部 (LP 234 + QP 出来る範囲 = ~570 問、~10-20 分)
bash scripts/download_all_bench_data.sh

# LP のみ (Netlib 取得 + 合成、決定論的)
bash scripts/download_all_bench_data.sh --lp

# 取得状況確認
bash scripts/download_all_bench_data.sh --check
```

| dir | 件数 | source | 自動化 |
|---|---|---|---|
| lp_problems | 109 | Netlib | ✓ |
| lp_problems_infeas | 29 | Netlib | ✓ |
| lp_problems_extra | 4 | Mittelmann | ✓ |
| lp_problems_hard | 53 | various | ✓ |
| lp_problems_canary | 27 | symlink | ✓ |
| lp_problems_unbounded | 12 | 合成 (固定 seed) | ✓ |
| osqp_bench | 62 | external + gen | ✓ |
| osqp_bench_extra | 238 | 合成 (固定 seed) | ✓ |
| osqp_bench_illscaled | 126 | 合成 (固定 seed) | ✓ |
| osqp_bench_xl | 2 | 合成 (固定 seed) | ✓ |
| mpc_qp | 64 | external | ✓ |
| qp_dense_a | 8 | 合成 (固定 seed) | ✓ |
| qp_infeasible | 12 | 合成 (固定 seed) | ✓ |
| qp_unbounded | 9 | 合成 (固定 seed) | ✓ |
| qplib_nonconvex | 45 | 合成 (固定 seed) | ✓ |
| **maros_meszaros** | 139 | YimingYAN/QP-Test-Problems | **手動** |
| **qplib** | 41 | QPLIB.zib.de | **手動** |
| **qplib_unsupported** | 11 | QPLIB.zib.de | **手動** |

合成系は固定 seed なので任意環境で同一出力を再現可。Maros / QPLIB (計 191 問) は
download script 未整備、手動配置が必要 (URL ヒントは `download_all_bench_data.sh` 内に記載)。

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
