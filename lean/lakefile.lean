import Lake
open Lake DSL

package «otspot_reduction_proofs» where
  -- Lean 形式化: (A) LP 連結成分分割の正しさ / (B) 一般二次制約 FBBT の soundness。
  -- 数学lib経路は cache get 失敗(タイムオーバー)のため、自前 OrderedField prelude で構成する。

@[default_target]
lean_lib «LeanProofs» where
  roots := #[`LeanProofs]
  globs := #[.andSubmodules `LeanProofs]
