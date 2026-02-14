# solver

Rustで実装された高性能数理最適化ソルバー

## プロジェクト概要

`solver`は線形計画問題（LP）、混合整数計画問題（MIP）、非線形計画問題（NLP）を段階的に扱う次世代最適化ソルバーです。Rustによりメモリ安全性と高性能を両立し、並列処理を前提とした設計を採用しています。

**Phase 1 M1（現在）**: Primal Simplex法によるLP求解の基本実装が完了しています。

## 前提条件

- Rust toolchain（1.70以降推奨）

```bash
# Rustのインストール（未導入の場合）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## ビルド方法

```bash
# リリースビルド
cargo build --release

# デバッグビルド
cargo build
```

## テスト実行

```bash
# 全テスト実行
cargo test

# 詳細出力付き
cargo test -- --nocapture
```

## 使い方

### 基本的なLP問題の定義と求解

以下は2変数の線形計画問題を解く例です:

**問題**:
```
minimize    -x1 - x2
subject to  x1 + x2 <= 4
            x1 <= 3
            x2 <= 3
            x1, x2 >= 0
```

**実装例**:

```rust
use solver::problem::LpProblem;
use solver::sparse::CscMatrix;
use solver::simplex;

fn main() {
    // 目的関数ベクトル c: min c^T x
    let c = vec![-1.0, -1.0];

    // 制約行列 A を疎行列（CSC形式）で定義
    // 3行2列の行列（3つの制約、2つの変数）
    let rows = vec![0, 0, 1, 2];  // 行インデックス
    let cols = vec![0, 1, 0, 1];  // 列インデックス
    let vals = vec![1.0, 1.0, 1.0, 1.0];  // 値
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 2)
        .expect("行列構築失敗");

    // 右辺ベクトル b: Ax <= b
    let b = vec![4.0, 3.0, 3.0];

    // LP問題の構築
    let problem = LpProblem::new(c, a, b).expect("LP問題構築失敗");

    // 求解
    let result = simplex::solve(&problem);

    // 結果の表示
    println!("求解ステータス: {}", result.status);
    println!("目的関数値: {}", result.objective);
    println!("解: {:?}", result.solution);
}
```

**出力例**:
```
求解ステータス: Optimal
目的関数値: -4
解: [1.0, 3.0]
```

### より複雑な例（3変数）

```rust
use solver::problem::LpProblem;
use solver::sparse::CscMatrix;
use solver::simplex;

fn solve_3var_problem() {
    // 問題: min -2x1 - 3x2 - x3
    //       s.t. x1 + x2 + x3 <= 10
    //            2x1 + x2 <= 14
    //            x2 + x3 <= 8
    //            x1, x2, x3 >= 0

    let c = vec![-2.0, -3.0, -1.0];

    let rows = vec![0, 0, 0, 1, 1, 2, 2];
    let cols = vec![0, 1, 2, 0, 1, 1, 2];
    let vals = vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 3).unwrap();

    let b = vec![10.0, 14.0, 8.0];

    let problem = LpProblem::new(c, a, b).unwrap();
    let result = simplex::solve(&problem);

    println!("{}", result);
}
```

### CSC疎行列の基本操作

```rust
use solver::sparse::CscMatrix;

// 単位行列の生成
let identity = CscMatrix::identity(3);

// ベクトルとの乗算
let x = vec![1.0, 2.0, 3.0];
let y = identity.mat_vec_mul(&x).unwrap();
assert_eq!(y, vec![1.0, 2.0, 3.0]);

// 転置
let matrix_t = identity.transpose();
```

## 現在の実装状況

**Phase 1 M1（完了）**:
- ✅ Primal Simplex法（Phase I/II法による2段階アルゴリズム）
- ✅ CSC（Compressed Sparse Column）形式の疎行列実装
- ✅ LP問題の定義と求解API
- ✅ 退化問題対応（Bland's rule）
- ✅ 非実行可能・非有界問題の検出

## プロジェクト構成

```
src/
├── lib.rs           # クレートのエントリポイント
├── sparse/
│   └── mod.rs       # 疎行列（CscMatrix）の実装
├── problem/
│   └── mod.rs       # LP問題定義（LpProblem, SolverResult等）
└── simplex/
    └── mod.rs       # Primal Simplex法の実装
```

### 主要なAPI

- **`solver::sparse::CscMatrix`**: 疎行列（CSC形式）
  - `new(nrows, ncols)`: 空行列生成
  - `from_triplets(rows, cols, vals, nrows, ncols)`: COO形式から構築
  - `mat_vec_mul(x)`: 行列ベクトル積
  - `transpose()`: 転置行列

- **`solver::problem::LpProblem`**: LP問題定義
  - `new(c, a, b)`: LP問題の構築（min c^T x, s.t. Ax <= b, x >= 0）

- **`solver::simplex::solve(problem)`**: LP問題の求解
  - 戻り値: `SolverResult`（status, objective, solution）

## ライセンス

このプロジェクトはデュアルライセンスです:

- **Apache License 2.0** ([LICENSE-APACHE](LICENSE-APACHE))
- **MIT License** ([LICENSE-MIT](LICENSE-MIT))

どちらかを選択して使用できます。

## 今後の開発予定

### Phase 1（0-12ヶ月）: LP Simplex MVP
- M1: Primal Simplex（完了）
- M2: Dual Simplex（計画中、3-6ヶ月）
- M3: Python bindings（PyO3、6-9ヶ月）
- M4: PyPI公開とSciPy PR提出（9-12ヶ月）

### Phase 2（12-24ヶ月）: MIP + 並列化
- Branch-and-cut
- 並列木探索（Rustの安全性を活用）
- 切除平面法

### Phase 3（24-36ヶ月）: NLP/GPU拡張
- GPU対応LP/MIP、またはNLPソルバー（Phase 1-2の結果次第で判断）
