/-
# Part B: 一般二次制約からの FBBT 型 bound 導出の soundness (スカラー核心部)

Vec/List の添字問題を避けるため純スカラーで定式化する。FBBT の 1 ステップは
「箱の上で制約を単変数二次不等式に包絡し、その解集合で箱を絞る」操作で、
残りの変数への依存は包絡値 (lo, hi) に集約されるので、スカラー定式化で
十分な一般性がある。双線形項 xᵢ·xⱼ (i≠j) は区間積包絡補題
`interval_mul_upper` / `interval_mul_lower` が 4 端点積で包絡する —
有限端点なら包絡値は有限 (0 を含む区間でも端点積は有限元なので有限包絡が
成立する。実装の 0·∞ 規約の数学的対応)。

公理は LeanProofs/OrderedField.lean の `OrderedField` クラスのみで、
これは ℝ で妥当な公理に限定されている。よって本ファイルの定理は
実体として ℝ に対する正当な証明である。

main theorems:
  - interval_mul_upper / interval_mul_lower : 区間積の 4 点包絡
  - quad_lower_bound : 平方完成による a>0 の二次式の下限閉形式
  - bound_from_quad  : FBBT 推論ステップの核心 (a<0 は上側の双対として同型)
-/
import LeanProofs.OrderedField

namespace LeanProofs.Fbbt
variable {F : Type} [OrderedField F]

/-- 順序反転 (減算なしの純加法誘導)。 -/
theorem neg_le_neg' {a b : F} (h : a ≤ b) : -b ≤ -a := by
  have h1 : (0 : F) ≤ b - a := (sub_nonneg b a).mpr h
  rw [sub_eq] at h1
  have h2 := add_le_add_left h1 (-b)
  rw [add_zero, ← add_assoc, add_left_neg, zero_add] at h2
  exact h2

theorem mul_left_comm (a b c : F) : a * (b * c) = b * (a * c) := by
  rw [mul_comm a (b * c), mul_assoc, mul_comm c a]

theorem add_left_comm (a b c : F) : a + (b + c) = b + (a + c) := by
  rw [← add_assoc, add_comm a b, add_assoc b a c]

/-- 平方完成の中核恒等式 (展開標準形)。 -/
theorem sq_two_ax_add_b (a b x : F) :
    (2 * a * x + b) * (2 * a * x + b) =
      (2 * a) * (2 * a) * (x * x) + (2 * a) * (2 * (b * x)) + b * b := by
  have hw : (2 * a * x) * (2 * a * x) = 2 * a * (2 * a) * (x * x) := by
    simp only [mul_comm, mul_assoc, mul_left_comm]
  have hb : (2 * a * x) * b + b * (2 * a * x) = 2 * a * (2 * (b * x)) := by
    rw [mul_comm b (2 * a * x), ← right_distrib, ← two_mul]
    simp only [mul_comm, mul_assoc, mul_left_comm]
  rw [left_distrib, right_distrib, right_distrib,
    add_assoc (2 * a * x * (2 * a * x)) (2 * a * x * b) (b * (2 * a * x) + b * b),
    ← add_assoc (2 * a * x * b) (b * (2 * a * x)) (b * b), hw, hb,
    ← add_assoc (2 * a * (2 * a) * (x * x)) (2 * a * (2 * (b * x))) (b * b)]

theorem h4sq (a x : F) : (2 * a) * (2 * a) * (x * x) = (4 * a) * (a * (x * x)) := by
  rw [four_eq2]
  simp only [mul_comm, mul_assoc, mul_left_comm]

theorem h4lin (a b x : F) : (2 * a) * (2 * (b * x)) = (4 * a) * (b * x) := by
  rw [mul_assoc (2 : F) a (2 * (b * x)), mul_left_comm a 2 (b * x),
    ← mul_assoc (2 : F) (2 : F) (a * (b * x)), ← four_eq2,
    ← mul_assoc (4 : F) a (b * x)]

/-- **区間積の上側 4 点包絡**: `x ∈ [l₁,u₁]`, `y ∈ [l₂,u₂]` (有限端点) なら
`x·y` は端点積 `l₁l₂, l₁u₂, u₁l₂, u₁u₂` の任意共通上界以下。
双線形項の包絡が 0 を含む区間でも有限であることの実証。 -/
theorem interval_mul_upper (l₁ u₁ l₂ u₂ x y p : F)
    (hx₁ : l₁ ≤ x) (hx₂ : x ≤ u₁) (hy₁ : l₂ ≤ y) (hy₂ : y ≤ u₂)
    (h11 : l₁ * l₂ ≤ p) (h12 : l₁ * u₂ ≤ p) (h21 : u₁ * l₂ ≤ p) (h22 : u₁ * u₂ ≤ p) :
    x * y ≤ p := by
  rcases lt_or_ge' y 0 with hy₀ | hy₀
  · -- y ≤ 0: x·y ≤ l₁·y (端点符号非依存: (l₁−x)·y ≥ 0)
    have hxy0 : (l₁ - x) * y = l₁ * y - x * y := by
      rw [sub_eq, left_distrib, neg_mul, ← sub_eq]
    have hsub : 0 ≤ l₁ * y - x * y := by
      rw [← hxy0]
      exact mul_nonneg_of_nonpos_of_nonpos ((sub_nonpos _ _).mpr hx₁) hy₀
    have hxy : x * y ≤ l₁ * y := (sub_nonneg _ _).mp hsub
    rcases lt_or_ge' l₁ 0 with hl₁ | hl₁
    · -- l₁ ≤ 0: l₁·y ≤ l₁·l₂ (y ≥ l₂, 符号反転)
      have h2a : l₁ * (y - l₂) = l₁ * y - l₁ * l₂ := by
        rw [sub_eq, right_distrib, mul_neg, ← sub_eq]
      have hh := mul_nonneg (neg_nonneg hl₁) ((sub_nonneg _ _).mpr hy₁)
      rw [neg_mul l₁ (y - l₂)] at hh
      have hs2 : l₁ * y - l₁ * l₂ ≤ 0 := by
        rw [← h2a]
        exact le_of_neg_nonneg hh
      exact le_trans _ _ _ hxy (le_trans _ _ _ ((sub_nonpos _ _).mp hs2) h11)
    · -- 0 < l₁: l₁·y ≤ l₁·u₂
      exact le_trans _ _ _ hxy
        (le_trans _ _ _ (mul_le_mul_of_nonneg_left hy₂ (le_of_lt hl₁)) h12)
  · -- 0 < y: x·y ≤ u₁·y
    have hxy : x * y ≤ u₁ * y := mul_le_mul_of_nonneg_right hx₂ (le_of_lt hy₀)
    rcases lt_or_ge' u₁ 0 with hu₁ | hu₁
    · -- u₁ ≤ 0: u₁·y ≤ u₁·l₂
      have h2a : u₁ * (y - l₂) = u₁ * y - u₁ * l₂ := by
        rw [sub_eq, right_distrib, mul_neg, ← sub_eq]
      have hh := mul_nonneg (neg_nonneg hu₁) ((sub_nonneg _ _).mpr hy₁)
      rw [neg_mul u₁ (y - l₂)] at hh
      have hs2 : u₁ * y - u₁ * l₂ ≤ 0 := by
        rw [← h2a]
        exact le_of_neg_nonneg hh
      exact le_trans _ _ _ hxy (le_trans _ _ _ ((sub_nonpos _ _).mp hs2) h21)
    · -- 0 < u₁: u₁·y ≤ u₁·u₂
      exact le_trans _ _ _ hxy
        (le_trans _ _ _ (mul_le_mul_of_nonneg_left hy₂ (le_of_lt hu₁)) h22)

/-- **区間積の下側 4 点包絡** (`interval_mul_upper` を -x に適用)。 -/
theorem interval_mul_lower (l₁ u₁ l₂ u₂ x y q : F)
    (hx₁ : l₁ ≤ x) (hx₂ : x ≤ u₁) (hy₁ : l₂ ≤ y) (hy₂ : y ≤ u₂)
    (h11 : q ≤ l₁ * l₂) (h12 : q ≤ l₁ * u₂) (h21 : q ≤ u₁ * l₂) (h22 : q ≤ u₁ * u₂) :
    q ≤ x * y := by
  have hc1 : -(u₁ * l₂) = (-u₁) * l₂ := (neg_mul u₁ l₂).symm
  have hc2 : -(u₁ * u₂) = (-u₁) * u₂ := (neg_mul u₁ u₂).symm
  have hc3 : -(l₁ * l₂) = (-l₁) * l₂ := (neg_mul l₁ l₂).symm
  have hc4 : -(l₁ * u₂) = (-l₁) * u₂ := (neg_mul l₁ u₂).symm
  have h := interval_mul_upper (-u₁) (-l₁) l₂ u₂ (-x) y (-q)
    (neg_le_neg' hx₂) (neg_le_neg' hx₁) hy₁ hy₂
    (hc1 ▸ neg_le_neg' h21) (hc2 ▸ neg_le_neg' h22)
    (hc3 ▸ neg_le_neg' h11) (hc4 ▸ neg_le_neg' h12)
  rw [neg_mul] at h
  have h5 := add_le_add h (le_refl (x * y))
  rw [add_left_neg, add_comm] at h5
  exact (sub_nonneg _ _).mp h5

/-- **二次式の下限 (平方完成)**: a > 0 のとき任意の t で
`a·t² + b·t + c ≥ c − b²/(4a)`。頂点の箱内/外 case 分岐を閉形式に吸収した形。 -/
theorem quad_lower_bound (a b c : F) (ha : 0 < a) (t : F) :
    c - (4 * a)⁻¹ * (b * b) ≤ a * t * t + b * t + c := by
  have ha4 : (0 : F) < 4 * a := mul_pos four_pos ha
  have hne : 4 * a ≠ 0 := ne_of_pos ha4
  have hkey : (4 * a)⁻¹ * ((2 * a * t + b) * (2 * a * t + b)) =
      a * t * t + b * t + (4 * a)⁻¹ * (b * b) := by
    rw [sq_two_ax_add_b, right_distrib, right_distrib, h4sq, h4lin,
      ← mul_assoc (4 * a)⁻¹ (4 * a) (a * (t * t)), ← mul_assoc (4 * a)⁻¹ (4 * a) (b * t),
      mul_comm (4 * a)⁻¹ (4 * a), mul_inv_cancel (4 * a) hne, one_mul, ← mul_assoc a t t, one_mul]
  have hge : 0 ≤ a * t * t + b * t + (4 * a)⁻¹ * (b * b) := by
    rw [← hkey]
    exact mul_nonneg (inv_nonneg (le_of_lt ha4)) (sq_nonneg _)
  have hsub : 0 ≤ (a * t * t + b * t + c) - (c - (4 * a)⁻¹ * (b * b)) := by
    rw [sub_eq, sub_eq, neg_add, neg_neg,
      add_assoc (a * t * t + b * t) c (Neg.neg c + (4 * a)⁻¹ * (b * b)),
      ← add_assoc c (Neg.neg c) ((4 * a)⁻¹ * (b * b)), add_right_neg c, zero_add]
    exact hge
  exact (sub_nonneg _ _).mp hsub

/-- **FBBT 推論ステップ**: 制約 `a·xᵢ² + b·xᵢ + g ≤ rhs` が点 xᵢ で成立し、
残部の値域包絡が `lo ≤ g` なら `a·xᵢ² + b·xᵢ + lo ≤ rhs`。
(`quad_lower_bound` と合わせれば xᵢ の絞り込み根拠が得られる。a < 0 は
上側不等式の双対として同型。) -/
theorem bound_from_quad (a b lo rhs gval : F) (x : F)
    (hge : lo ≤ gval) (hcon : a * x * x + b * x + gval ≤ rhs) :
    a * x * x + b * x + lo ≤ rhs :=
  le_trans _ _ _ (add_le_add_left hge (a * x * x + b * x)) hcon

end LeanProofs.Fbbt
