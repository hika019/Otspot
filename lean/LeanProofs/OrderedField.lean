/-
# OrderedField: 順序付き体の公理限定 (自己完結, Mathlib 不使用)

## 設計方針

Lean 4.36 の core には `Field`/`LinearOrder`/`LinearOrderedField` 型クラスが
存在しない (これらは Mathlib 側の提供物) ため, 本プロジェクトは順序付き体に
関する「証明で実際に invoke する標準事実そのもの」をクラスフィールド (= 公理)
として宣言する.

**これら公理はすべて実数 ℝ に対して妥当である** (各事実は ℝ の通常の演算・
順序・平方根で成立する). したがって本クラスのインスタンス — 特に ℝ — について
証明された定理は, ℝ に対する正当な証明である. 公理リストは「ℝ の標準性質の
列挙」に限定されており, 既存の数学と矛盾する仮定は含まれない.

- 減算 `a - b = a + -b`, 除算 `a / b = a * b⁻¹`, 順序 `a < b = a ≤ b ∧ ¬ b ≤ a`
  は定義 (公理ではない).
- `min`/`max` は `if a ≤ b then ...` による自己完結定義 (`minO`/`maxO`).
- 非負元への平方根存在 `exists_sqrt` は ℝ の通常 sqrt が満たす.

Mathlib の型クラス・ tactic は一切使わない.
-/

namespace LeanProofs

open Classical

/-- 順序付き体の基底公理 ( notation 完備のため先に宣言 ). -/
class OrderedFieldCore (F : Type u) extends Zero F, One F, Add F, Mul F, Neg F, Inv F, LE F where
  add_assoc : ∀ a b c : F, a + b + c = a + (b + c)
  add_comm : ∀ a b : F, a + b = b + a
  zero_add : ∀ a : F, 0 + a = a
  add_zero : ∀ a : F, a + 0 = a
  add_left_neg : ∀ a : F, -a + a = 0
  mul_assoc : ∀ a b c : F, a * b * c = a * (b * c)
  mul_comm : ∀ a b : F, a * b = b * a
  one_mul : ∀ a : F, 1 * a = a
  mul_one : ∀ a : F, a * 1 = a
  mul_inv_cancel : ∀ a : F, a ≠ 0 → a * a⁻¹ = 1
  inv_zero : (0 : F)⁻¹ = 0
  left_distrib : ∀ a b c : F, (a + b) * c = a * c + b * c
  right_distrib : ∀ a b c : F, a * (b + c) = a * b + a * c
  zero_ne_one : (0 : F) ≠ 1
  le_refl : ∀ a : F, a ≤ a
  le_trans : ∀ a b c : F, a ≤ b → b ≤ c → a ≤ c
  le_antisymm : ∀ a b : F, a ≤ b → b ≤ a → a = b
  le_total : ∀ a b : F, a ≤ b ∨ b ≤ a
  add_le_add_left : ∀ a b : F, a ≤ b → ∀ c : F, c + a ≤ c + b
  mul_nonneg : ∀ a b : F, 0 ≤ a → 0 ≤ b → 0 ≤ a * b
  sq_nonneg : ∀ a : F, 0 ≤ a * a
  exists_sqrt : ∀ a : F, 0 ≤ a → ∃ s : F, 0 ≤ s ∧ s * s = a

/-- 減算の定義 (体公理ではなく定義). -/
instance (priority := 100) instSub [OrderedFieldCore F] : Sub F := ⟨fun a b => a + -b⟩

/-- 除算の定義 (体公理ではなく定義). -/
instance (priority := 100) instDiv [OrderedFieldCore F] : Div F := ⟨fun a b => a * b⁻¹⟩

/-- 順序 `<` の定義. -/
instance (priority := 100) instLt [OrderedFieldCore F] : LT F :=
  ⟨fun a b => a ≤ b ∧ ¬ b ≤ a⟩

/-- 自然数定数 (core に汎用 `OfNat` 構成がないため自前で定義).
    注: `1` は core の `instOfNatOne` (One に由来) が勝つのでこの再帰は停止する. -/
def natRec (F : Type u) [Zero F] [One F] [Add F] : Nat → F
  | 0 => 0
  | k + 1 => natRec F k + 1

instance (priority := 100) instOfNatOF [z : Zero F] [o : One F] [a : Add F] (n : Nat) :
    OfNat F n where
  ofNat := natRec F n

/-- 順序付き体の標準補題群 (公理として宣言; すべて ℝ で成立).
    証明で invoke する事実だけを列挙している. -/
class OrderedField (F : Type u) where
  core : OrderedFieldCore F
  add_right_neg : ∀ a : F, a + -a = 0
  add_left_cancel : ∀ a b c : F, a + b = a + c → b = c
  add_right_cancel : ∀ a b c : F, a + b = c + b → a = c
  neg_eq_of_add : ∀ a b : F, b + a = 0 → -a = b
  neg_add : ∀ a b : F, -(a + b) = -a + -b
  neg_neg : ∀ a : F, -(-a) = a
  neg_inj : ∀ a b : F, -a = -b → a = b
  mul_zero : ∀ a : F, a * 0 = 0
  zero_mul : ∀ a : F, 0 * a = 0
  mul_neg : ∀ a b : F, a * -b = -(a * b)
  neg_mul : ∀ a b : F, -a * b = -(a * b)
  neg_mul_neg : ∀ a b : F, -a * -b = a * b
  sub_self : ∀ a : F, a - a = 0
  sub_zero : ∀ a : F, a - 0 = a
  sub_add_cancel : ∀ a b : F, a - b + b = a
  add_sub_cancel : ∀ a b : F, a + b - b = a
  sub_eq_zero : ∀ a b : F, a - b = 0 ↔ a = b
  zero_ne_one' : (1 : F) ≠ 0
  le_of_lt : ∀ a b : F, a < b → a ≤ b
  not_le_of_lt : ∀ a b : F, a < b → ¬ b ≤ a
  not_lt_of_le : ∀ a b : F, a ≤ b → ¬ b < a
  lt_of_not_ge : ∀ a b : F, ¬ b ≤ a → a < b
  lt_of_lt_of_le : ∀ a b c : F, a < b → b ≤ c → a < c
  lt_of_le_of_lt : ∀ a b c : F, a ≤ b → b < c → a < c
  lt_trans : ∀ a b c : F, a < b → b < c → a < c
  lt_or_ge : ∀ a b : F, a < b ∨ b ≤ a
  lt_or_ge' : ∀ a b : F, a ≤ b ∨ b < a
  le_of_not_lt : ∀ a b : F, ¬ a < b → b ≤ a
  ne_of_lt : ∀ a b : F, a < b → a ≠ b
  lt_of_ne_of_le : ∀ a b : F, a ≠ b → a ≤ b → a < b
  ne_of_pos : ∀ a : F, 0 < a → a ≠ 0
  add_le_add : ∀ a b c d : F, a ≤ b → c ≤ d → a + c ≤ b + d
  add_lt_add_left : ∀ a b : F, a < b → ∀ c : F, c + a < c + b
  add_lt_add_right : ∀ a b : F, a < b → ∀ c : F, a + c < b + c
  add_lt_add : ∀ a b c d : F, a < b → c < d → a + c < b + d
  lt_of_add_lt_add_left : ∀ a b c : F, a + b < a + c → b < c
  lt_of_add_lt_add_right : ∀ a b c : F, a + c < b + c → a < b
  le_of_add_le_add_left : ∀ a b c : F, a + b ≤ a + c → b ≤ c
  le_of_add_le_add_right : ∀ a b c : F, a + c ≤ b + c → a ≤ b
  sub_nonneg : ∀ a b : F, 0 ≤ a - b ↔ b ≤ a
  sub_nonpos : ∀ a b : F, a - b ≤ 0 ↔ a ≤ b
  sub_pos : ∀ a b : F, 0 < a - b ↔ b < a
  neg_nonneg : ∀ a : F, a ≤ 0 → 0 ≤ -a
  neg_nonpos : ∀ a : F, 0 ≤ a → -a ≤ 0
  le_of_neg_le : ∀ a : F, -a ≤ 0 → 0 ≤ a
  le_of_neg_nonneg : ∀ a : F, 0 ≤ -a → a ≤ 0
  neg_pos_of_neg : ∀ a : F, a < 0 → 0 < -a
  neg_lt_zero : ∀ a : F, 0 < -a → a < 0
  lt_neg_neg : ∀ a b : F, a < -b → b < -a
  sub_lt_zero_of_lt : ∀ a b : F, a - b < 0 → a < b
  mul_le_mul_of_nonneg_right : ∀ a b c : F, a ≤ b → 0 ≤ c → a * c ≤ b * c
  mul_le_mul_of_nonneg_left : ∀ a b c : F, a ≤ b → 0 ≤ c → c * a ≤ c * b
  mul_nonneg_of_nonpos_of_nonpos : ∀ a b : F, a ≤ 0 → b ≤ 0 → 0 ≤ a * b
  mul_nonpos_of_nonneg_of_nonpos : ∀ a b : F, 0 ≤ a → b ≤ 0 → a * b ≤ 0
  mul_ne_zero : ∀ a b : F, a ≠ 0 → b ≠ 0 → a * b ≠ 0
  mul_pos : ∀ a b : F, 0 < a → 0 < b → 0 < a * b
  mul_lt_mul_of_pos_right : ∀ a b c : F, a < b → 0 < c → a * c < b * c
  mul_lt_mul_of_pos_left : ∀ a b c : F, a < b → 0 < c → c * a < c * b
  le_of_mul_le_mul_left : ∀ a b c : F, 0 < c → c * a ≤ c * b → a ≤ b
  le_of_mul_le_mul_left_of_neg : ∀ a b c : F, c < 0 → c * a ≤ c * b → b ≤ a
  sq_le_sq : ∀ x y : F, 0 ≤ x → x ≤ y → x * x ≤ y * y
  sq_lt_sq : ∀ x y : F, 0 ≤ x → x < y → x * x < y * y
  sq_abs_bound : ∀ s t : F, 0 ≤ s → t * t ≤ s * s → -s ≤ t ∧ t ≤ s
  inv_mul_cancel : ∀ a : F, a ≠ 0 → a⁻¹ * a = 1
  inv_unique : ∀ a b : F, a ≠ 0 → a * b = 1 → b = a⁻¹
  inv_ne_zero : ∀ a : F, a ≠ 0 → a⁻¹ ≠ 0
  inv_inv : ∀ a : F, a ≠ 0 → a⁻¹⁻¹ = a
  inv_mul : ∀ a b : F, a ≠ 0 → b ≠ 0 → (a * b)⁻¹ = a⁻¹ * b⁻¹
  inv_nonneg : ∀ a : F, 0 ≤ a → 0 ≤ a⁻¹
  inv_pos : ∀ a : F, 0 < a → 0 < a⁻¹
  one_pos : (0 : F) < 1
  two_eq : (2 : F) = 1 + 1
  two_mul : ∀ x : F, (2 : F) * x = x + x
  two_pos : (0 : F) < 2
  two_ne_zero : (2 : F) ≠ 0
  four_eq2 : (4 : F) = 2 * 2
  four_pos : (0 : F) < 4
  four_ne_zero : (4 : F) ≠ 0

/-- `OrderedField` から基底公理インスタンスへ. -/
instance (priority := 100) instOFC [self : OrderedField F] : OrderedFieldCore F := self.core

section Basic

variable {F : Type u} [OrderedField F]

theorem add_assoc (a b c : F) : a + b + c = a + (b + c) := OrderedFieldCore.add_assoc a b c
theorem add_comm (a b : F) : a + b = b + a := OrderedFieldCore.add_comm a b
theorem zero_add (a : F) : 0 + a = a := OrderedFieldCore.zero_add a
theorem add_zero (a : F) : a + 0 = a := OrderedFieldCore.add_zero a
theorem add_left_neg (a : F) : -a + a = 0 := OrderedFieldCore.add_left_neg a
theorem add_right_neg (a : F) : a + -a = 0 := OrderedField.add_right_neg a
theorem add_left_cancel (a b c : F) (h : a + b = a + c) : b = c := OrderedField.add_left_cancel a b c h
theorem add_right_cancel (a b c : F) (h : a + b = c + b) : a = c := OrderedField.add_right_cancel a b c h
protected theorem neg_eq_of_add (a b : F) (h : b + a = 0) : -a = b := OrderedField.neg_eq_of_add a b h
theorem neg_add (a b : F) : -(a + b) = -a + -b := OrderedField.neg_add a b
theorem neg_neg (a : F) : -(-a) = a := OrderedField.neg_neg a
theorem neg_inj (a b : F) (h : -a = -b) : a = b := OrderedField.neg_inj a b h
theorem mul_assoc (a b c : F) : a * b * c = a * (b * c) := OrderedFieldCore.mul_assoc a b c
theorem mul_comm (a b : F) : a * b = b * a := OrderedFieldCore.mul_comm a b
theorem one_mul (a : F) : 1 * a = a := OrderedFieldCore.one_mul a
theorem mul_one (a : F) : a * 1 = a := OrderedFieldCore.mul_one a
theorem mul_inv_cancel (a : F) (ha : a ≠ 0) : a * a⁻¹ = 1 := OrderedFieldCore.mul_inv_cancel a ha
theorem inv_zero : (0 : F)⁻¹ = 0 := OrderedFieldCore.inv_zero
theorem left_distrib (a b c : F) : (a + b) * c = a * c + b * c := OrderedFieldCore.left_distrib a b c
theorem right_distrib (a b c : F) : a * (b + c) = a * b + a * c := OrderedFieldCore.right_distrib a b c
theorem mul_zero (a : F) : a * 0 = 0 := OrderedField.mul_zero a
theorem zero_mul (a : F) : 0 * a = 0 := OrderedField.zero_mul a
theorem mul_neg (a b : F) : a * -b = -(a * b) := OrderedField.mul_neg a b
theorem neg_mul (a b : F) : -a * b = -(a * b) := OrderedField.neg_mul a b
theorem neg_mul_neg (a b : F) : -a * -b = a * b := OrderedField.neg_mul_neg a b
theorem sub_self (a : F) : a - a = 0 := OrderedField.sub_self a
theorem sub_zero (a : F) : a - 0 = a := OrderedField.sub_zero a
theorem sub_add_cancel (a b : F) : a - b + b = a := OrderedField.sub_add_cancel a b
theorem add_sub_cancel (a b : F) : a + b - b = a := OrderedField.add_sub_cancel a b
theorem sub_eq_zero (a b : F) : a - b = 0 ↔ a = b := OrderedField.sub_eq_zero a b
theorem zero_ne_one' : (1 : F) ≠ 0 := OrderedField.zero_ne_one'
theorem le_refl (a : F) : a ≤ a := OrderedFieldCore.le_refl a
theorem le_trans (a b c : F) (hab : a ≤ b) (hbc : b ≤ c) : a ≤ c :=
  OrderedFieldCore.le_trans a b c hab hbc
theorem le_antisymm (a b : F) (hab : a ≤ b) (hba : b ≤ a) : a = b :=
  OrderedFieldCore.le_antisymm a b hab hba
theorem le_total (a b : F) : a ≤ b ∨ b ≤ a := OrderedFieldCore.le_total a b
theorem add_le_add_left {a b : F} (hab : a ≤ b) (c : F) : c + a ≤ c + b :=
  OrderedFieldCore.add_le_add_left a b hab c
theorem mul_nonneg {a b : F} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a * b :=
  OrderedFieldCore.mul_nonneg a b ha hb
theorem sq_nonneg (a : F) : 0 ≤ a * a := OrderedFieldCore.sq_nonneg a
theorem exists_sqrt (a : F) (ha : 0 ≤ a) : ∃ s : F, 0 ≤ s ∧ s * s = a :=
  OrderedFieldCore.exists_sqrt a ha
theorem le_of_lt {a b : F} (h : a < b) : a ≤ b := OrderedField.le_of_lt a b h
theorem not_le_of_lt {a b : F} (h : a < b) : ¬ b ≤ a := OrderedField.not_le_of_lt a b h
theorem not_lt_of_le {a b : F} (h : a ≤ b) : ¬ b < a := OrderedField.not_lt_of_le a b h
theorem lt_of_not_ge {a b : F} (h : ¬ b ≤ a) : a < b := OrderedField.lt_of_not_ge a b h
theorem lt_of_lt_of_le {a b c : F} (hab : a < b) (hbc : b ≤ c) : a < c :=
  OrderedField.lt_of_lt_of_le a b c hab hbc
theorem lt_of_le_of_lt {a b c : F} (hab : a ≤ b) (hbc : b < c) : a < c :=
  OrderedField.lt_of_le_of_lt a b c hab hbc
theorem lt_trans {a b c : F} (hab : a < b) (hbc : b < c) : a < c := OrderedField.lt_trans a b c hab hbc
theorem lt_or_ge (a b : F) : a < b ∨ b ≤ a := OrderedField.lt_or_ge a b
theorem lt_or_ge' (a b : F) : a ≤ b ∨ b < a := OrderedField.lt_or_ge' a b
theorem le_of_not_lt {a b : F} (h : ¬ a < b) : b ≤ a := OrderedField.le_of_not_lt a b h
theorem ne_of_lt {a b : F} (h : a < b) : a ≠ b := OrderedField.ne_of_lt a b h
theorem lt_of_ne_of_le {a b : F} (hn : a ≠ b) (hab : a ≤ b) : a < b :=
  OrderedField.lt_of_ne_of_le a b hn hab
theorem ne_of_pos {a : F} (h : 0 < a) : a ≠ 0 := OrderedField.ne_of_pos a h
theorem add_le_add {a b c d : F} (hac : a ≤ b) (hbd : c ≤ d) : a + c ≤ b + d :=
  OrderedField.add_le_add a b c d hac hbd
theorem add_lt_add_left {a b : F} (hab : a < b) (c : F) : c + a < c + b :=
  OrderedField.add_lt_add_left a b hab c
theorem add_lt_add_right {a b : F} (hab : a < b) (c : F) : a + c < b + c :=
  OrderedField.add_lt_add_right a b hab c
theorem add_lt_add {a b c d : F} (hac : a < b) (hbd : c < d) : a + c < b + d :=
  OrderedField.add_lt_add a b c d hac hbd
theorem lt_of_add_lt_add_left {a b c : F} (h : a + b < a + c) : b < c :=
  OrderedField.lt_of_add_lt_add_left a b c h
theorem lt_of_add_lt_add_right {a b c : F} (h : a + c < b + c) : a < b :=
  OrderedField.lt_of_add_lt_add_right a b c h
theorem le_of_add_le_add_left {a b c : F} (h : a + b ≤ a + c) : b ≤ c :=
  OrderedField.le_of_add_le_add_left a b c h
theorem le_of_add_le_add_right {a b c : F} (h : a + c ≤ b + c) : a ≤ b :=
  OrderedField.le_of_add_le_add_right a b c h
theorem sub_nonneg (a b : F) : 0 ≤ a - b ↔ b ≤ a := OrderedField.sub_nonneg a b
theorem sub_nonpos (a b : F) : a - b ≤ 0 ↔ a ≤ b := OrderedField.sub_nonpos a b
theorem sub_pos (a b : F) : 0 < a - b ↔ b < a := OrderedField.sub_pos a b
theorem neg_nonneg {a : F} (h : a ≤ 0) : 0 ≤ -a := OrderedField.neg_nonneg a h
theorem neg_nonpos {a : F} (h : 0 ≤ a) : -a ≤ 0 := OrderedField.neg_nonpos a h
theorem le_of_neg_le {a : F} (h : -a ≤ 0) : 0 ≤ a := OrderedField.le_of_neg_le a h
theorem le_of_neg_nonneg {a : F} (h : 0 ≤ -a) : a ≤ 0 := OrderedField.le_of_neg_nonneg a h
theorem neg_pos_of_neg {a : F} (h : a < 0) : 0 < -a := OrderedField.neg_pos_of_neg a h
theorem neg_lt_zero {a : F} (h : 0 < -a) : a < 0 := OrderedField.neg_lt_zero a h
theorem lt_neg_neg {a b : F} (h : a < -b) : b < -a := OrderedField.lt_neg_neg a b h
theorem sub_lt_zero_of_lt {a b : F} (h : a - b < 0) : a < b := OrderedField.sub_lt_zero_of_lt a b h
theorem mul_le_mul_of_nonneg_right {a b c : F} (hab : a ≤ b) (hc : 0 ≤ c) : a * c ≤ b * c :=
  OrderedField.mul_le_mul_of_nonneg_right a b c hab hc
theorem mul_le_mul_of_nonneg_left {a b c : F} (hab : a ≤ b) (hc : 0 ≤ c) : c * a ≤ c * b :=
  OrderedField.mul_le_mul_of_nonneg_left a b c hab hc
theorem mul_nonneg_of_nonpos_of_nonpos {a b : F} (ha : a ≤ 0) (hb : b ≤ 0) : 0 ≤ a * b :=
  OrderedField.mul_nonneg_of_nonpos_of_nonpos a b ha hb
theorem mul_nonpos_of_nonneg_of_nonpos {a b : F} (ha : 0 ≤ a) (hb : b ≤ 0) : a * b ≤ 0 :=
  OrderedField.mul_nonpos_of_nonneg_of_nonpos a b ha hb
theorem mul_ne_zero {a b : F} (ha : a ≠ 0) (hb : b ≠ 0) : a * b ≠ 0 :=
  OrderedField.mul_ne_zero a b ha hb
theorem mul_pos {a b : F} (ha : 0 < a) (hb : 0 < b) : 0 < a * b := OrderedField.mul_pos a b ha hb
protected theorem mul_lt_mul_of_pos_right {a b c : F} (hab : a < b) (hc : 0 < c) : a * c < b * c :=
  OrderedField.mul_lt_mul_of_pos_right a b c hab hc
protected theorem mul_lt_mul_of_pos_left {a b c : F} (hab : a < b) (hc : 0 < c) : c * a < c * b :=
  OrderedField.mul_lt_mul_of_pos_left a b c hab hc
theorem le_of_mul_le_mul_left {a b c : F} (hc : 0 < c) (h : c * a ≤ c * b) : a ≤ b :=
  OrderedField.le_of_mul_le_mul_left a b c hc h
theorem le_of_mul_le_mul_left_of_neg {a b c : F} (hc : c < 0) (h : c * a ≤ c * b) : b ≤ a :=
  OrderedField.le_of_mul_le_mul_left_of_neg a b c hc h
theorem sq_le_sq {x y : F} (hx : 0 ≤ x) (hxy : x ≤ y) : x * x ≤ y * y :=
  OrderedField.sq_le_sq x y hx hxy
theorem sq_lt_sq {x y : F} (hx : 0 ≤ x) (hxy : x < y) : x * x < y * y :=
  OrderedField.sq_lt_sq x y hx hxy
theorem sq_abs_bound {s t : F} (hs : 0 ≤ s) (h : t * t ≤ s * s) : -s ≤ t ∧ t ≤ s :=
  OrderedField.sq_abs_bound s t hs h
theorem inv_mul_cancel (a : F) (ha : a ≠ 0) : a⁻¹ * a = 1 := OrderedField.inv_mul_cancel a ha
theorem inv_unique (a b : F) (ha : a ≠ 0) (h : a * b = 1) : b = a⁻¹ := OrderedField.inv_unique a b ha h
theorem inv_ne_zero {a : F} (ha : a ≠ 0) : a⁻¹ ≠ 0 := OrderedField.inv_ne_zero a ha
theorem inv_inv (a : F) (ha : a ≠ 0) : a⁻¹⁻¹ = a := OrderedField.inv_inv a ha
theorem inv_mul (a b : F) (ha : a ≠ 0) (hb : b ≠ 0) : (a * b)⁻¹ = a⁻¹ * b⁻¹ :=
  OrderedField.inv_mul a b ha hb
theorem inv_nonneg {a : F} (ha : 0 ≤ a) : 0 ≤ a⁻¹ := OrderedField.inv_nonneg a ha
theorem inv_pos {a : F} (ha : 0 < a) : 0 < a⁻¹ := OrderedField.inv_pos a ha
theorem one_pos : (0 : F) < 1 := OrderedField.one_pos
theorem two_eq : (2 : F) = 1 + 1 := OrderedField.two_eq
theorem two_mul (x : F) : (2 : F) * x = x + x := OrderedField.two_mul x
theorem two_pos : (0 : F) < 2 := OrderedField.two_pos
theorem two_ne_zero : (2 : F) ≠ 0 := OrderedField.two_ne_zero
theorem four_eq2 : (4 : F) = 2 * 2 := OrderedField.four_eq2
theorem four_pos : (0 : F) < 4 := OrderedField.four_pos
theorem four_ne_zero : (4 : F) ≠ 0 := OrderedField.four_ne_zero
@[simp] theorem sub_eq (a b : F) : a - b = a + -b := rfl
@[simp] theorem div_eq (a b : F) : a / b = a * b⁻¹ := rfl

-- AC 組み換え用 (公理から 1 行)
theorem add_left_comm (a b c : F) : a + (b + c) = b + (a + c) := by
  rw [← add_assoc, add_comm a b, add_assoc]
theorem add_right_comm (a b c : F) : (a + b) + c = (a + c) + b := by
  rw [add_assoc, add_comm b c, ← add_assoc]
theorem mul_left_comm (a b c : F) : a * (b * c) = b * (a * c) := by
  rw [← mul_assoc, mul_comm a b, mul_assoc]
theorem mul_right_comm (a b c : F) : (a * b) * c = (a * c) * b := by
  rw [mul_assoc, mul_comm b c, ← mul_assoc]

-- min / max (自己完結定義; classical 判定に依存するため noncomputable)
noncomputable def maxO (a b : F) : F := if a ≤ b then b else a
noncomputable def minO (a b : F) : F := if a ≤ b then a else b

theorem le_maxO_left (a b : F) : a ≤ maxO a b := by
  rw [maxO]
  split
  · assumption
  · exact le_refl _

theorem le_maxO_right (a b : F) : b ≤ maxO a b := by
  rw [maxO]
  split
  · exact le_refl _
  · exact Or.resolve_right (le_total b a) ‹¬ _›

theorem minO_le_left (a b : F) : minO a b ≤ a := by
  rw [minO]
  split
  · exact le_refl _
  · exact Or.resolve_right (le_total b a) ‹¬ _›

theorem minO_le_right (a b : F) : minO a b ≤ b := by
  rw [minO]
  split
  · assumption
  · exact le_refl _

theorem maxO_le {a b c : F} (hac : a ≤ c) (hbc : b ≤ c) : maxO a b ≤ c := by
  rw [maxO]
  split
  · exact hbc
  · exact hac

theorem le_minO {a b c : F} (hac : c ≤ a) (hbc : c ≤ b) : c ≤ minO a b := by
  rw [minO]
  split
  · exact hac
  · exact hbc

end Basic

end LeanProofs
