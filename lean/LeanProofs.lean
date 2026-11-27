/-
# otspot 削減手法の Lean 形式化

  - `LeanProofs.Split` : Part A — 独立連結成分による LP 分割の正しさ
    (feasible_iff / objective_additive / opt_decomposition / Status 統合規則)
  - `LeanProofs.Fbbt`  : Part B — 一般二次制約からの FBBT 型 bound 導出の soundness
    (区間積 4 点包絡 / 平方完成下限 / FBBT 推論ステップ)

代数・順序の基礎事実は `LeanProofs.OrderedField` の自己完結クラスに公理化して
いる (ℝ で妥当な公理のみ。Mathlib 非依存)。
-/
import LeanProofs.OrderedField
import LeanProofs.Split
import LeanProofs.Fbbt
