/-
# Part A: 独立連結成分による LP 分割の正しさ

LpProblem を (A : 行のリスト, b, c, lb ub : List (Option F) + 寸法不変条件) と定義
(境界の Option は実装の `Option<f64>` 境界と 1 対 1. 独立成分側の枠は `none` が
そのまま無限側の枠に対応).

各行の support を hyperedge とする変数上の連結関係の連結成分による列の分割は
「行の全 support が単一の成分に含まれる」性質を持つ. この性質を持つ 2 成分分割は
列を成分順に並べ替えればブロック対角そのものであるから, ブロック対角結合
`combine` を分割の標準形 (merge = リスト連結 `++`, split = take/drop) として
その正しさを証明する. k 成分一般は 2 成分結合の帰納 (README 参照).

main theorems:
  - feasible_iff        : x₁ ++ x₂ が元の可行解 ⇔ 各成分制限が副問題で可行
  - objective_additive  : merge した x の目的値 = Σ 成分目的値
  - merge_optimal       : 各成分最適解 ⇒ マージが元の最適解
  - optimal_restriction / optimal_restriction2 / opt_decomposition
  - status_infeasible_rule / status_unbounded_rule / status_optimal_rule
    (Status 統合規則; Unbounded 規則は「他の成分の可行点の存在」を仮定に置く)
-/
import LeanProofs.OrderedField

namespace LeanProofs

section Lists
variable {F : Type} [OrderedField F]

def vzeros (n : Nat) : List F := List.replicate n 0

/-- 内積 (長さ一致下で通常のドット積). -/
def dot : List F → List F → F
  | [], _ => 0
  | _, [] => 0
  | p :: a, q :: x => p * q + dot a x

theorem dot_zeros_r (a : List F) (n : Nat) : dot a (vzeros n) = 0 := by
  induction a generalizing n with
  | nil => exact rfl
  | cons p a ih =>
      cases n with
      | zero => exact rfl
      | succ k =>
          show p * 0 + dot a (vzeros k) = 0
          rw [mul_zero, ih, add_zero]

theorem dot_zeros_l (x : List F) (n : Nat) : dot (vzeros n) x = 0 := by
  induction n generalizing x with
  | zero => exact rfl
  | succ k ih =>
      cases x with
      | nil => exact rfl
      | cons q x =>
          show 0 * q + dot (vzeros k) x = 0
          rw [zero_mul, ih, add_zero]

/-- 長さ一致の下でのドット積の append 分解. -/
theorem dot_append (a b x y : List F) (h : a.length = x.length) :
    dot (a ++ b) (x ++ y) = dot a x + dot b y := by
  induction a generalizing x with
  | nil =>
      cases x with
      | nil => exact (zero_add (dot b y)).symm
      | cons q x =>
          exfalso
          simp at h
  | cons p a ih =>
      cases x with
      | nil =>
          exfalso
          simp at h
      | cons q x =>
          have ha : a.length = x.length := by simpa using h
          show p * q + dot (a ++ b) (x ++ y) = (p * q + dot a x) + dot b y
          rw [ih x ha, add_assoc]

theorem vzeros_length (F : Type) [OrderedField F] (n : Nat) :
    (List.replicate n (0 : F)).length = n :=
  List.length_replicate

/-- 前半行ブロック: 内積は前半成分に一致. -/
theorem dot_pad1 (r x₁ x₂ : List F) (k : Nat) (h : r.length = x₁.length) :
    dot (r ++ vzeros k) (x₁ ++ x₂) = dot r x₁ := by
  rw [dot_append _ _ _ _ h, dot_zeros_l, add_zero]

/-- 後半行ブロック: 内積は後半成分に一致. -/
theorem dot_pad2 (r x₁ x₂ : List F) (h : x₁.length = k) :
    dot (vzeros k ++ r) (x₁ ++ x₂) = dot r x₂ := by
  have hz : (vzeros k).length = x₁.length :=
    show (vzeros k).length = x₁.length from (@vzeros_length F _ k).trans h.symm
  rw [dot_append _ _ _ _ hz, dot_zeros_l, zero_add]

omit [OrderedField F] in
theorem take_of_append (x₁ : List F) (x₂ : List F) (h : x₁.length = n₁) :
    (x₁ ++ x₂).take n₁ = x₁ := by
  subst h
  induction x₁ with
  | nil => exact rfl
  | cons a x₁ ih => exact congrArg (List.cons a) ih

omit [OrderedField F] in
theorem drop_of_append (x₁ : List F) (x₂ : List F) (h : x₁.length = n₁) :
    (x₁ ++ x₂).drop n₁ = x₂ := by
  subst h
  induction x₁ generalizing x₂ with
  | nil => exact rfl
  | cons a x₁ ih => exact ih x₂

omit [OrderedField F] in
theorem take_append_drop (y : List F) (n : Nat) : y.take n ++ y.drop n = y := by
  induction y generalizing n with
  | nil => simp
  | cons a y ih =>
      cases n with
      | zero => exact rfl
      | succ k => exact congrArg (List.cons a) (ih k)

/-- 全成分 `l ≤ x`. `none` 成分は制約なし (実装の無限境界に対応). -/
def geOK : List (Option F) → List F → Prop
  | [], _ => True
  | _ :: _, [] => False
  | none :: l, _ :: x => geOK l x
  | some p :: l, q :: x => p ≤ q ∧ geOK l x

/-- 全成分 `x ≤ u`. `none` 成分は制約なし. -/
def leOK : List (Option F) → List F → Prop
  | [], _ => True
  | _ :: _, [] => False
  | none :: l, _ :: x => leOK l x
  | some p :: l, q :: x => q ≤ p ∧ leOK l x

theorem geOK_append (l₁ l₂ : List (Option F)) (x₁ x₂ : List F)
    (h : l₁.length = x₁.length) :
    geOK (l₁ ++ l₂) (x₁ ++ x₂) ↔ (geOK l₁ x₁ ∧ geOK l₂ x₂) := by
  induction l₁ generalizing x₁ with
  | nil =>
      cases x₁ with
      | nil => exact ⟨fun e => ⟨trivial, e⟩, fun e => e.2⟩
      | cons a x₁ =>
          exfalso
          simp at h
  | cons o l ih =>
      cases x₁ with
      | nil =>
          exfalso
          simp at h
      | cons q x₁ =>
          have ha : l.length = x₁.length := by simpa using h
          cases o with
          | none =>
              constructor
              · intro e
                have hi := ih x₁ ha |>.mp e
                exact ⟨hi.1, hi.2⟩
              · intro e
                exact ih x₁ ha |>.mpr ⟨e.1, e.2⟩
          | some p =>
              constructor
              · intro e
                have e' : p ≤ q ∧ geOK (l ++ l₂) (x₁ ++ x₂) := e
                have hi := ih x₁ ha |>.mp e'.2
                exact ⟨⟨e'.1, hi.1⟩, hi.2⟩
              · intro e
                show p ≤ q ∧ geOK (l ++ l₂) (x₁ ++ x₂)
                exact ⟨e.1.1, ih x₁ ha |>.mpr ⟨e.1.2, e.2⟩⟩

theorem leOK_append (u₁ u₂ : List (Option F)) (x₁ x₂ : List F)
    (h : u₁.length = x₁.length) :
    leOK (u₁ ++ u₂) (x₁ ++ x₂) ↔ (leOK u₁ x₁ ∧ leOK u₂ x₂) := by
  induction u₁ generalizing x₁ with
  | nil =>
      cases x₁ with
      | nil => exact ⟨fun e => ⟨trivial, e⟩, fun e => e.2⟩
      | cons a x₁ =>
          exfalso
          simp at h
  | cons o u ih =>
      cases x₁ with
      | nil =>
          exfalso
          simp at h
      | cons q x₁ =>
          have ha : u.length = x₁.length := by simpa using h
          cases o with
          | none =>
              constructor
              · intro e
                exact ih x₁ ha |>.mp e
              · intro e
                exact ih x₁ ha |>.mpr ⟨e.1, e.2⟩
          | some p =>
              constructor
              · intro e
                have e' : q ≤ p ∧ leOK (u ++ u₂) (x₁ ++ x₂) := e
                have hi := ih x₁ ha |>.mp e'.2
                exact ⟨⟨e'.1, hi.1⟩, hi.2⟩
              · intro e
                show q ≤ p ∧ leOK (u ++ u₂) (x₁ ++ x₂)
                exact ⟨e.1.1, ih x₁ ha |>.mpr ⟨e.1.2, e.2⟩⟩

/-- 行制約 (行と RHS を同時にたどる). -/
def rowsOK : List (List F) → List F → List F → Prop
  | [], _, _ => True
  | _ :: _, [], _ => False
  | r :: rows, bb :: bs, x => dot r x ≤ bb ∧ rowsOK rows bs x

theorem rowsOK_append (A₁ A₂ : List (List F)) (b₁ b₂ x : List F)
    (h : A₁.length = b₁.length) :
    rowsOK (A₁ ++ A₂) (b₁ ++ b₂) x ↔ (rowsOK A₁ b₁ x ∧ rowsOK A₂ b₂ x) := by
  induction A₁ generalizing b₁ with
  | nil =>
      cases b₁ with
      | nil => exact ⟨fun e => ⟨trivial, e⟩, fun e => e.2⟩
      | cons a bs =>
          exfalso
          simp at h
  | cons r A ih =>
      cases b₁ with
      | nil =>
          exfalso
          simp at h
      | cons bb bs =>
          have ha : A.length = bs.length := by simpa using h
          constructor
          · intro e
            have e' : dot r x ≤ bb ∧ rowsOK (A ++ A₂) (bs ++ b₂) x := e
            have hi := ih bs ha |>.mp e'.2
            exact ⟨⟨e'.1, hi.1⟩, hi.2⟩
          · intro e
            show dot r x ≤ bb ∧ rowsOK (A ++ A₂) (bs ++ b₂) x
            exact ⟨e.1.1, ih bs ha |>.mpr ⟨e.1.2, e.2⟩⟩

end Lists

section LP
variable {F : Type} [OrderedField F]

/-- LP と寸法不変条件 (n := c.length, m := A.length). -/
structure LpProblem (F : Type) where
  A : List (List F)
  b : List F
  c : List F
  lb : List (Option F)
  ub : List (Option F)
  hb : b.length = A.length
  hrow : ∀ r ∈ A, r.length = c.length
  hlb : lb.length = c.length
  hub : ub.length = c.length

/-- 目的値. -/
def objOf (P : LpProblem F) (x : List F) : F := dot P.c x

/-- 可行解. -/
def feasible (P : LpProblem F) (x : List F) : Prop :=
  x.length = P.c.length ∧ geOK P.lb x ∧ leOK P.ub x ∧ rowsOK P.A P.b x

def IsOpt (P : LpProblem F) (x : List F) : Prop :=
  feasible P x ∧ ∀ y, feasible P y → objOf P x ≤ objOf P y

def UnboundedBelow (P : LpProblem F) : Prop :=
  ∀ B : F, ∃ x, feasible P x ∧ objOf P x ≤ B

def Infeasible (P : LpProblem F) : Prop := ¬ ∃ x, feasible P x

/-- 独立 2 成分のブロック対角結合 (merge = `++` での列結合に対応). -/
def combine (P₁ P₂ : LpProblem F) : LpProblem F where
  A := P₁.A.map (fun r => r ++ vzeros P₂.c.length) ++
        P₂.A.map (fun r => vzeros P₁.c.length ++ r)
  b := P₁.b ++ P₂.b
  c := P₁.c ++ P₂.c
  lb := P₁.lb ++ P₂.lb
  ub := P₁.ub ++ P₂.ub
  hb := by
    simp only [List.length_append, List.length_map, P₁.hb, P₂.hb]
  hrow := by
    intro r hr
    simp only [List.mem_append, List.mem_map] at hr
    rcases hr with ⟨r₁, hm₁, rfl⟩ | ⟨r₂, hm₂, rfl⟩
    · rw [List.length_append,
        show (vzeros P₂.c.length).length = P₂.c.length from vzeros_length F _,
        P₁.hrow r₁ hm₁, List.length_append]
    · rw [List.length_append,
        show (vzeros P₁.c.length).length = P₁.c.length from vzeros_length F _,
        P₂.hrow r₂ hm₂, List.length_append]
  hlb := by
    simp only [List.length_append, P₁.hlb, P₂.hlb]
  hub := by
    simp only [List.length_append, P₁.hub, P₂.hub]

/-- 前半行ブロックの行制約 ⇔ 前半成分の行制約. -/
theorem rowsOK_map_pad1 (A : List (List F)) (b x₁ x₂ : List F) (k : Nat)
    (hA : A.length = b.length) (hr : ∀ r ∈ A, r.length = x₁.length) :
    rowsOK (A.map (fun r => r ++ vzeros k)) b (x₁ ++ x₂) ↔ rowsOK A b x₁ := by
  induction A generalizing b with
  | nil => exact ⟨fun _ => trivial, fun _ => trivial⟩
  | cons r A ih =>
      cases b with
      | nil =>
          exfalso
          simp at hA
      | cons bb bs =>
          have hA' : A.length = bs.length := by simpa using hA
          have hr' : ∀ r' ∈ A, r'.length = x₁.length :=
            fun r' hm => hr r' (List.mem_cons_of_mem r hm)
          constructor
          · intro e
            have hd : dot (r ++ vzeros k) (x₁ ++ x₂) ≤ bb := e.1
            rw [dot_pad1 r x₁ x₂ k (hr r List.mem_cons_self)] at hd
            exact ⟨hd, ih bs hA' hr' |>.mp e.2⟩
          · intro e
            refine ⟨?_, ih bs hA' hr' |>.mpr e.2⟩
            rw [dot_pad1 r x₁ x₂ k (hr r List.mem_cons_self)]
            exact e.1

/-- 後半行ブロックの行制約 ⇔ 後半成分の行制約. -/
theorem rowsOK_map_pad2 (A : List (List F)) (b x₁ x₂ : List F)
    (hA : A.length = b.length) (hx₁ : x₁.length = k)
    (hr : ∀ r ∈ A, r.length = x₂.length) :
    rowsOK (A.map (fun r => vzeros k ++ r)) b (x₁ ++ x₂) ↔ rowsOK A b x₂ := by
  induction A generalizing b with
  | nil => exact ⟨fun _ => trivial, fun _ => trivial⟩
  | cons r A ih =>
      cases b with
      | nil =>
          exfalso
          simp at hA
      | cons bb bs =>
          have hA' : A.length = bs.length := by simpa using hA
          have hr' : ∀ r' ∈ A, r'.length = x₂.length :=
            fun r' hm => hr r' (List.mem_cons_of_mem r hm)
          constructor
          · intro e
            have hd : dot (vzeros k ++ r) (x₁ ++ x₂) ≤ bb := e.1
            rw [dot_pad2 r x₁ x₂ hx₁] at hd
            exact ⟨hd, ih bs hA' hr' |>.mp e.2⟩
          · intro e
            refine ⟨?_, ih bs hA' hr' |>.mpr e.2⟩
            rw [dot_pad2 r x₁ x₂ hx₁]
            exact e.1

/-- **可行解の分解定理** (main theorem A-1). -/
theorem feasible_iff (P₁ P₂ : LpProblem F) (x₁ x₂ : List F)
    (h₁ : x₁.length = P₁.c.length) (h₂ : x₂.length = P₂.c.length) :
    feasible (combine P₁ P₂) (x₁ ++ x₂) ↔ (feasible P₁ x₁ ∧ feasible P₂ x₂) := by
  have hlb : geOK (combine P₁ P₂).lb (x₁ ++ x₂) ↔
      (geOK P₁.lb x₁ ∧ geOK P₂.lb x₂) :=
    geOK_append P₁.lb P₂.lb x₁ x₂ (P₁.hlb.trans h₁.symm)
  have hub : leOK (combine P₁ P₂).ub (x₁ ++ x₂) ↔
      (leOK P₁.ub x₁ ∧ leOK P₂.ub x₂) :=
    leOK_append P₁.ub P₂.ub x₁ x₂ (P₁.hub.trans h₁.symm)
  have hpadc : (List.map (fun r : List F => r ++ vzeros P₂.c.length) P₁.A).length =
      P₁.b.length :=
    (List.length_map (fun r : List F => r ++ vzeros P₂.c.length)).trans P₁.hb.symm
  have hpadd : (List.map (fun r : List F => vzeros P₁.c.length ++ r) P₂.A).length =
      P₂.b.length :=
    (List.length_map (fun r : List F => vzeros P₁.c.length ++ r)).trans P₂.hb.symm
  have happ : rowsOK (combine P₁ P₂).A (combine P₁ P₂).b (x₁ ++ x₂) ↔
      (rowsOK (List.map (fun r : List F => r ++ vzeros P₂.c.length) P₁.A) P₁.b (x₁ ++ x₂) ∧
        rowsOK (List.map (fun r : List F => vzeros P₁.c.length ++ r) P₂.A) P₂.b (x₁ ++ x₂)) :=
    rowsOK_append
      (List.map (fun r : List F => r ++ vzeros P₂.c.length) P₁.A)
      (List.map (fun r : List F => vzeros P₁.c.length ++ r) P₂.A)
      P₁.b P₂.b (x₁ ++ x₂) hpadc
  have h1 := rowsOK_map_pad1 P₁.A P₁.b x₁ x₂ P₂.c.length P₁.hb.symm
    (fun r hm => (P₁.hrow r hm).trans h₁.symm)
  have h2 := rowsOK_map_pad2 P₂.A P₂.b x₁ x₂ P₂.hb.symm h₁
    (fun r hm => (P₂.hrow r hm).trans h₂.symm)
  have hrows : rowsOK (combine P₁ P₂).A (combine P₁ P₂).b (x₁ ++ x₂) ↔
      (rowsOK P₁.A P₁.b x₁ ∧ rowsOK P₂.A P₂.b x₂) := by
    constructor
    · intro e
      exact ⟨h1.mp (happ.mp e |>.1), h2.mp (happ.mp e |>.2)⟩
    · intro e
      exact happ.mpr ⟨h1.mpr e.1, h2.mpr e.2⟩

  constructor
  · intro h
    obtain ⟨hlen, hblb, hubb, hrr⟩ := h
    exact ⟨⟨h₁, hlb.mp hblb |>.1, hub.mp hubb |>.1, hrows.mp hrr |>.1⟩,
      ⟨h₂, hlb.mp hblb |>.2, hub.mp hubb |>.2, hrows.mp hrr |>.2⟩⟩
  · intro h
    obtain ⟨⟨hl1, hb1, hu1, hr1⟩, ⟨hl2, hb2, hu2, hr2⟩⟩ := h
    refine ⟨by rw [List.length_append, hl1, hl2]; exact List.length_append.symm,
      hlb.mpr ⟨hb1, hb2⟩, hub.mpr ⟨hu1, hu2⟩, hrows.mpr ⟨hr1, hr2⟩⟩

/-- 目的値の加法性 (main theorem A-2). -/
theorem objective_additive (P₁ P₂ : LpProblem F) (x₁ x₂ : List F)
    (h₁ : x₁.length = P₁.c.length) :
    objOf (combine P₁ P₂) (x₁ ++ x₂) = objOf P₁ x₁ + objOf P₂ x₂ := by
  unfold objOf combine
  rw [dot_append _ _ _ _ h₁.symm]

/-- 制限による可行解の分解 (任意長の y を take/drop で分割). -/
theorem feasible_restrict (P₁ P₂ : LpProblem F) (y : List F)
    (hy : y.length = (combine P₁ P₂).c.length) :
    feasible (combine P₁ P₂) y ↔
      (feasible P₁ (y.take P₁.c.length) ∧ feasible P₂ (y.drop P₁.c.length)) := by
  have hleny : y.length = P₁.c.length + P₂.c.length :=
    hy.trans List.length_append
  have hn₁ : (y.take P₁.c.length).length = P₁.c.length := by
    rw [List.length_take, Nat.min_eq_left (hleny ▸ Nat.le_add_right _ _)]
  have hn₂ : (y.drop P₁.c.length).length = P₂.c.length := by
    rw [List.length_drop, hleny, Nat.add_sub_cancel_left]
  have hsplit : y.take P₁.c.length ++ y.drop P₁.c.length = y :=
    take_append_drop y P₁.c.length
  constructor
  · intro h
    rw [← hsplit] at h
    exact (feasible_iff P₁ P₂ (y.take P₁.c.length) (y.drop P₁.c.length) hn₁ hn₂).mp h
  · intro h
    rw [← hsplit]
    exact (feasible_iff P₁ P₂ (y.take P₁.c.length) (y.drop P₁.c.length) hn₁ hn₂).mpr h

/-- どれか成分が不可行なら全体が不可行. -/
theorem merge_infeasible (P₁ P₂ : LpProblem F) (h : Infeasible P₁) : Infeasible (combine P₁ P₂) := by
  intro hex
  obtain ⟨y, hy⟩ := hex
  exact h ⟨y.take P₁.c.length,
    (feasible_restrict P₁ P₂ y hy.1).mp hy |>.1⟩

theorem merge_infeasible2 (P₁ P₂ : LpProblem F) (h : Infeasible P₂) : Infeasible (combine P₁ P₂) := by
  intro hex
  obtain ⟨y, hy⟩ := hex
  exact h ⟨y.drop P₁.c.length,
    (feasible_restrict P₁ P₂ y hy.1).mp hy |>.2⟩

/-- **最適解の再構成** (main theorem A-3). -/
theorem merge_optimal (x₁ x₂ : List F) (hf1 : feasible P₁ x₁) (hf2 : feasible P₂ x₂)
    (h1 : IsOpt P₁ x₁) (h2 : IsOpt P₂ x₂) : IsOpt (combine P₁ P₂) (x₁ ++ x₂) := by
  refine ⟨(feasible_iff P₁ P₂ x₁ x₂ hf1.1 hf2.1).mpr ⟨h1.1, h2.1⟩, ?_⟩
  intro y hy
  have hyt := (feasible_restrict P₁ P₂ y hy.1).mp hy
  calc objOf (combine P₁ P₂) (x₁ ++ x₂)
      = objOf P₁ x₁ + objOf P₂ x₂ := objective_additive P₁ P₂ x₁ x₂ hf1.1
    _ ≤ objOf P₁ (y.take P₁.c.length) + objOf P₂ (y.drop P₁.c.length) :=
        add_le_add (h1.2 _ hyt.1) (h2.2 _ hyt.2)
    _ = objOf (combine P₁ P₂) (y.take P₁.c.length ++ y.drop P₁.c.length) :=
        (objective_additive P₁ P₂ (y.take P₁.c.length) (y.drop P₁.c.length)
          (feasible_restrict P₁ P₂ y hy.1 |>.mp hy |>.1 |>.1)).symm
    _ = objOf (combine P₁ P₂) y := by rw [take_append_drop]

/-- 最適解の前半成分への制限は前半副問題の最適解. -/
theorem optimal_restriction (x₁ x₂ : List F) (feas1 : feasible P₁ x₁)
    (feas2 : feasible P₂ x₂) (hopt : IsOpt (combine P₁ P₂) (x₁ ++ x₂)) :
    IsOpt P₁ x₁ := by
  refine ⟨feas1, ?_⟩
  intro y₁ hy₁
  have h := hopt.2 _ ((feasible_iff P₁ P₂ y₁ x₂ hy₁.1 feas2.1).mpr ⟨hy₁, feas2⟩)
  rw [objective_additive P₁ P₂ x₁ x₂ feas1.1,
    objective_additive P₁ P₂ y₁ x₂ hy₁.1] at h
  exact le_of_add_le_add_right h

/-- 最適解の後半成分への制限は後半副問題の最適解. -/
theorem optimal_restriction2 (x₁ x₂ : List F) (feas1 : feasible P₁ x₁)
    (feas2 : feasible P₂ x₂) (hopt : IsOpt (combine P₁ P₂) (x₁ ++ x₂)) :
    IsOpt P₂ x₂ := by
  refine ⟨feas2, ?_⟩
  intro y₂ hy₂
  have h := hopt.2 _ ((feasible_iff P₁ P₂ x₁ y₂ feas1.1 hy₂.1).mpr ⟨feas1, hy₂⟩)
  rw [objective_additive P₁ P₂ x₁ x₂ feas1.1,
    objective_additive P₁ P₂ x₁ y₂ feas1.1] at h
  exact le_of_add_le_add_left h

/-- **最適分解の同値性** (main theorem A-4). -/
theorem opt_decomposition (x₁ x₂ : List F) (feas1 : feasible P₁ x₁)
    (feas2 : feasible P₂ x₂) :
    IsOpt (combine P₁ P₂) (x₁ ++ x₂) ↔ (IsOpt P₁ x₁ ∧ IsOpt P₂ x₂) :=
  ⟨fun h => ⟨optimal_restriction x₁ x₂ feas1 feas2 h,
    optimal_restriction2 x₁ x₂ feas1 feas2 h⟩,
    fun h => merge_optimal x₁ x₂ feas1 feas2 h.1 h.2⟩

/-- Status 統合規則 1: どれか不可行 ⇒ Infeasible. -/
theorem status_infeasible_rule (P₁ P₂ : LpProblem F) (h : Infeasible P₁ ∨ Infeasible P₂) :
    Infeasible (combine P₁ P₂) := by
  cases h with
  | inl h1 => exact merge_infeasible P₁ P₂ h1
  | inr h2 => exact merge_infeasible2 P₁ P₂ h2

/-- Status 統合規則 2: 実行可能でどれか Unbounded ⇒ Unbounded
(他成分の可行点の存在を仮定に置く — 減算の向きが対称なため). -/
theorem status_unbounded_rule (P₁ P₂ : LpProblem F) (h1 : UnboundedBelow P₁)
    (hfeas2 : ∃ x, feasible P₂ x) : UnboundedBelow (combine P₁ P₂) := by
  intro B
  obtain ⟨z₂, hz₂f⟩ := hfeas2
  obtain ⟨w₁, hw₁f, hw₁o⟩ := h1 (B - objOf P₂ z₂)
  refine ⟨w₁ ++ z₂, (feasible_iff P₁ P₂ w₁ z₂ hw₁f.1 hz₂f.1).mpr ⟨hw₁f, hz₂f⟩, ?_⟩
  calc objOf (combine P₁ P₂) (w₁ ++ z₂)
      = objOf P₁ w₁ + objOf P₂ z₂ := objective_additive P₁ P₂ w₁ z₂ hw₁f.1
    _ ≤ (B - objOf P₂ z₂) + objOf P₂ z₂ := add_le_add hw₁o (le_refl _)
    _ = B := sub_add_cancel _ _

theorem status_unbounded_rule2 (P₁ P₂ : LpProblem F) (h2 : UnboundedBelow P₂)
    (hfeas1 : ∃ x, feasible P₁ x) : UnboundedBelow (combine P₁ P₂) := by
  intro B
  obtain ⟨z₁, hz₁f⟩ := hfeas1
  obtain ⟨w₂, hw₂f, hw₂o⟩ := h2 (B - objOf P₁ z₁)
  refine ⟨z₁ ++ w₂, (feasible_iff P₁ P₂ z₁ w₂ hz₁f.1 hw₂f.1).mpr ⟨hz₁f, hw₂f⟩, ?_⟩
  calc objOf (combine P₁ P₂) (z₁ ++ w₂)
      = objOf P₁ z₁ + objOf P₂ w₂ := objective_additive P₁ P₂ z₁ w₂ hz₁f.1
    _ ≤ objOf P₁ z₁ + (B - objOf P₁ z₁) := add_le_add (le_refl _) hw₂o
    _ = B := by rw [add_comm, sub_add_cancel]

/-- Status 統合規則 3: 全て Optimal ⇒ 最適値は合計. -/
theorem status_optimal_rule (P₁ P₂ : LpProblem F) (x₁ x₂ : List F) (hopt : IsOpt P₁ x₁ ∧ IsOpt P₂ x₂) :
    IsOpt (combine P₁ P₂) (x₁ ++ x₂) :=
  merge_optimal x₁ x₂ hopt.1.1 hopt.2.1 hopt.1 hopt.2

end LP

end LeanProofs
