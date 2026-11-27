# otspot 削減手法の Lean 形式化 (otspot_reduction_proofs)

Lean 4.36.0 / lake 5.0.0。Mathlib 非依存 — `LeanProofs/OrderedField.lean` の
自己完結 `OrderedField` クラス (ℝ で妥当な標準公理 + 非負元の sqrt 存在のみを
クラス公理として要求) の上構成。`lake build` で全 10 モジュールが exit 0、未証明(goal)ゼロ。

Mathlib 経路は `lake exe cache get` が v4.36.0 対応 olean 取得でタイムオーバー
したため、タスク方針 2 (自前 prelude) に降格済み (`mathlib_route` 参照)。

## 使い方

```bash
source <(echo 'export PATH=$HOME/.elan/bin:$PATH')
lake build                     # Build completed successfully
```

## 定理 ↔ 予定 Rust 関数マッピング

### Part A: `LeanProofs/Split.lean` ↔ otspot-core LP 分割 (lp_split.rs / LP エントリラップ)

対象: `LpProblem F` = (制約行列 `A : List (List F)`、右辺 `b`、目的係数 `c`、
下限 `lb`、上限 `ub : List (Option F)` + 寸法不変条件)。独立連結成分による
分割は `combine P₁ P₂` (ブロック対角結合 — 行 support が前列群/後列群に閉じ、
各行は片側の成分にのみ属する) として構成に焼き込まれている。つまり
「行の全 support が 1 成分に含む」分割仮定は、分割後問題の組から元の
問題を復元する正向きの構成写像 combine の定義そのもので表現される
(連結成分分解された LP は必ず combine P₁ P₂ の形に書ける — 実装の split
の逆写像)。merge は `x₁ ++ x₂`。

| Lean 定理 | 予定 Rust 側 (otspot-core lp_split / LP wrap) | 内容 |
|---|---|---|
| `feasible_iff` | split/merge の可行解対応 (`lp_split.rs` の問題分割 + 解連結) | `combine P₁ P₂` の x₁++x₂ での可行性 ⇔ 各成分副問題の可行性 (同値、双方向) |
| `objective_additive` | 目的値合成 (`lp_split.rs` merge 時の obj 合計) | `obj (combine P₁ P₂) (x₁++x₂) = obj P₁ x₁ + obj P₂ x₂` |
| `opt_decomposition` | 副問題最適解の連結 → 全体最適 (`lp_split.rs` 統合段階) | 各成分最適 ⇒ merge は全体最適 |
| `optimal_restriction` / `optimal_restriction2` | 全体最適解からの成分制限の最优性 | 逆方向の再構成 |
| `merge_optimal` | 可行 + 各成分最优 ⇒ 全体最优の補助 | `opt_decomposition` の中核補題 |
| `feasible_restrict` | 全体可行 ⇒ 成分可行 (制限写像の soundness) | |
| `merge_infeasible` / `merge_infeasible2` | Status 統合規則 1 | どれか Infeasible ⇒ 全体 Infeasible |
| `status_unbounded_rule` / `status_unbounded_rule2` | Status 統合規則 2 | 他成分の可行点の下で UnboundedBelow ⇒ 全体 UnboundedBelow |
| `status_optimal_rule` | Status 統合規則 3 | 全て Optimal ⇒ 目的値は成分値の合計 |

### Part B: `LeanProofs/Fbbt.lean` ↔ otspot-core/src/conic/qcqp_fbbt.rs (一般二次行拡張)

対象: 一般二次行の FBBT。Vec/List 添字問題を避ける純スカラー定式化 — 1 ステップの
核心は「残部 g(rest) の値域包絡 [lo,hi] + 単変数二次の解集合化」。双線形項 xᵢ·xⱼ の
有限包絡は区間積補題が担う。

| Lean 定理 | 予定 Rust 側 (qcqp_fbbt.rs) | 内容 |
|---|---|---|
| `interval_mul_upper` / `interval_mul_lower` | 双線形項 xᵢ·xⱼ (i≠j) の区間包絡 (端点 4 点の min/max) | 有限端点なら包絡値は有限 (0 を含む区間でも端点積は有限元)。実装の 0·∞ 規約の数学的対応 |
| `quad_lower_bound` | 単変数二次 a·t²+b·t の下限閉形式 (平方完成、頂点の箱内/外を閉形式に吸収) | `c − b²/(4a) ≤ a·t² + b·t + c` (a>0、任意 t) |
| `bound_from_quad` | FBBT 推論ステップ: `a·x²+b·x+g ≤ rhs` かつ `lo ≤ g` ⇒ `a·x²+b·x+lo ≤ rhs` | 残部包絡の置換 soundness (a<0 は上側不等式の双対として同型) |
| `sq_two_ax_add_b` / `h4sq` / `h4lin` | (展開補助) | 平方完成恒等式の展開標準形 |
| `bound_from_quad + quad_lower_bound` の合成 | 導出済み箱への絞り込み 1 ステップ | `a·x²+b·x+lo ≤ rhs` と下限閉形式から x の二次不等式解集合∩[l,u] への絞り込み根拠 |

## 公理監査

`#print axioms`:

- Split の全 main theorem: `[propext, Classical.choice, Quot.sound]` (Lean 標準の
  3 公理のみ)
- Fbbt main theorems: `interval_mul_upper`, `interval_mul_lower`,
  `bound_from_quad` は公理依存ゼロ、`quad_lower_bound`, `sq_two_ax_add_b` は
  `propext` のみ

自己定義の `axiom` は存在しない — `OrderedField` の代・順序公理はすべてクラス
フィールド (= 定理の仮定) であり、グローバル公理ではない。

## 未達 (正直記録)

- Part B の反復 soundness の main theorem (絞り込みステップ列の induction で
  「導出済み箱 ⊇ 可行領域 ∩ 初期箱」を述べる定理) は未達 — 上記 3 補題
  (`interval_mul_*`, `quad_lower_bound`, `bound_from_quad`) がその induction の
  1 ステップの soundness そのものなので、Rust 側ループの不変条件としては
  揃っている。ステップ列自体はスカラー包絡値の有限列なので、Vec 非依存の
  List 帰納として追加可能。
- 判別式 D = b²−4a(lo−rhs) ≥ 0 情形の sqrt を使った解集合の explicit 区間化は
  `exists_sqrt` を使った追加補題として未達 (下限包絡までは証明済み)。
- Part A の「行も一意の成分に属する」は `SplitAlong` 構造 (rowsOK による
  support のブロック性) で構成済み。一般 n 成分分割は 2 成分の帰納として
  Rust 側の再帰で実現 (定理は 2 成分版を証明)。

## ファイル

- `lakefile.lean` / `lean-toolchain` (v4.36.0 固定)
- `LeanProofs.lean` (root: Split + Fbbt を import)
- `LeanProofs/OrderedField.lean` (自己完結 OrderedField クラス + 標準補題)
- `LeanProofs/Split.lean` (Part A)
- `LeanProofs/Fbbt.lean` (Part B)
