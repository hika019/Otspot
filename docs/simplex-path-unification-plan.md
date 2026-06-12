# MIP node-LP path 統一計画

調査日: 2026-06-13

対象 worktree: `/home/whisky/develop/Otspot-wt/unify`, branch `investigate/simplex-path-unify`

## 結論

MIP node LP の scaling reuse を exact に入れる前に、node LP の canonical path を統一するべきである。

理由は、現状の node LP が同じ `LpProblem` から次の複数表現へ分岐し、行数・列数・basis index 空間・upper-bound 状態の意味が一致しないためである。

- bounded path: `BoundedStandardForm`, UB は `upper_bounds` に保持、`m = original constraints`
- bounded-artificial path: `BoundedStandardForm` に一時 artificial columns を追加、Phase I/II を bounded primal core で解く
- legacy path: `StandardForm`, UB は明示 UB rows + slacks に展開、`m = original constraints + finite UB rows`
- special path: `m == 0` / `n == 0` は simplex/scaling へ入らない
- GMI cut generation: solver path ではないが、legacy standard-form tableau に明示依存

`feat/mip-node-scale-reuse` の 4cb7e10 -> 166a85a -> 81250f7 は、この分裂を incremental に埋めようとしたが、最終的に legacy cache を外し、bounded 系だけを cache 対象に縮退している。`gt2 --cuts` の tree exact が崩れた真因は、単なる cache bug ではなく、同一 node が bounded / bounded-artificial / legacy のどれで解かれるかにより scaling と warm-start basis の意味が変わることにある。

## 再現実測

現 staging 相当の worktree で以下を実行した。

```text
cargo build --release
cargo run --release -p otspot-dev --bin milp_solve -- /home/whisky/develop/Otspot/data/miplib_small/gt2.mps --cuts --timeout 60 --eps 1e-6
```

結果:

```text
status: Optimal
objective: 21166.000000000
wall_ms: 3204.143
root_lp_bound: 21166
nodes: 219
incumbent_updates: 1
fp_incumbent_found: false
max_depth: 7
pruned: 218
lp_presolve_us: 0
lp_solve_us: 2891619
lp_postsolve_us: 0
per_node_us: presolve=0.0 solve=13203.7 postsolve=0.0
```

`docs/simplex-core-redesign.md` の c991780 実測では、同じ `gt2 --cuts` は `nodes=219`, `lp_solve_us=2,863,417us`, descendant `scale_us=2,378,109us` で、descendant `lp_solve_us` の 83.9% が scaling だった。今回の再測も node count / status / objective は一致し、solve time は同水準である。

## Path Map

| path | 選択条件 | standard form | Ruiz scaling | warm-start |
|---|---|---|---|---|
| special zero-var | `solve_without_presolve_inner`, `n == 0` | 構築しない | なし | なし |
| special no-row | `solve_without_presolve_inner`, `m == 0` | 構築しない | なし | なし |
| bounded | `SimplexMethod::Auto/DualAdvanced`, finite UB が1つ以上、`BoundedStandardForm.num_artificial == 0`, bounded dispatch enabled | `build_bounded_standard_form_with_deadline`: variable shift/split + original constraints slacks。finite UB は UB rows にせず `upper_bounds` | `LpEquilibration::scale_with_deadline(&bsf.a, &bsf.b, &bsf.c)`。`scale_upper_bounds` で UB も scaled space に写す | `warm.basis.len() == bsf.m && idx < bsf.n_total` のときだけ採用。`at_upper` は保存されず `vec![false; n_total]` から復元するため UB 側状態は失われる |
| bounded-artificial | finite UB があり、`bsf.num_artificial > 0` | 同じ `BoundedStandardForm` を元に `build_a_aug_for_eq` で artificial columns を一時追加。UB rows は増やさない | bounded と同じ BSF scaling 後、scaled `a` に artificial identity columns を追加 | cold path 中心。最終 basis が `j < bsf.n_total` のときだけ `WarmStartBasis` を返す。artificial が残ると legacy warm path と index 空間が合わないので返さない |
| legacy warm/cold advanced | finite UB なし、または bounded が `None` を返す (`UbViolationOutOfScope`, Phase I inconclusive, crash infeasible, dispatch disabled) | `build_standard_form_with_deadline`: finite lower/upper に応じて shift/splitし、finite boxed UB は追加 UB rows + slacks に展開 | `LpEquilibration::scale_with_deadline(&sf.a, &sf.b, &sf.c)` を legacy SF に対して毎回実行 | `warm.basis.len() == sf.m && idx < sf.n_total` のとき採用。bounded basis は `sf.m` と合わないことがあり自然に落ちる |
| legacy primal/two-phase | `SimplexMethod::Primal/Dual`、または advanced の Ge/Eq cold fallback | 同じ `StandardForm` | primal/dual 側で `StandardForm` に対して Ruiz scaling | primal は Phase I/II 後に `basis + x_b` を返す。legacy dual/primal は `at_upper` を持たない |
| GMI cut tableau | `mip/cuts.rs` の root cut generation | `build_standard_form` を意図的に使用。UB rows 展開済み、nonbasic は 0 前提 | cut LP は primal solve の結果 basis から unscaled tableau を再構築する | root cut 生成のための legacy SF basis が必要。これは solver path 削除後も tableau adapter として残す必要がある |

コード上の根拠:

- MIP node は `MilpProblem::solve` が bounds だけを差し替えて `solve_lp_with` へ渡す。
- MIP driver は integer node で `recover_warm_start_basis = true`, `presolve = false` にし、親 basis を子へ渡す。
- `solve_without_presolve_inner` は先に `build_standard_form_with_deadline` を作り、その後 `dual_advanced::solve_dual_advanced` に渡す。
- `solve_dual_advanced` は finite UB があると先に `build_bounded_standard_form_with_deadline` で bounded path を試し、失敗時に引数 `sf` の legacy path に落ちる。
- GMI cuts は file header で legacy standard form tableau 依存を明記し、`solve_cut_lp` は `SimplexMethod::Primal`, `presolve=false`, `recover_warm_start_basis=true` を強制する。

## Scale-Reuse 失敗の検証

### 4cb7e10

最初の実装は `CachedMipLpRelaxation` に bounded template と legacy template の両方を持たせた。root で `BoundedStandardForm` を作り、さらに `wrap_to_legacy(&bounded_template)` で legacy template を作り、両方を `scale_with_deadline_and_recipe` で scale した。

descendant solve は node bounds を差し替えた `node_problem` から bounded form を派生し、cached scaling recipe で `A/b/c` を再 scale し、bounded が `None` なら fresh bounded または cached legacy に落ちる設計だった。

問題点:

- bounded と legacy は同じ LP の別表現ではあるが、行数と列数が異なる。finite UB が legacy では rows/slacks、bounded では `upper_bounds` なので、root recipe を descendant の別表現へ安全に共有するには厳密な layout contract が必要。
- branch による bound 変更で `lb` shift、effective UB、row sign、`needs_artificial` が変わる。4cb7e10 の `derive_bounded_standard_form_for_bounds` は `initial_basis` / `needs_artificial` 変化を reject しているが、reject 後の fallback が再び cached recipe を使うため、fresh solve と同じ pivot path になる保証が弱い。
- legacy cache は `wrap_to_legacy(root bounded)` 由来なので、現 node の `build_standard_form(node_problem)` と pattern が一致しない場合に recipe fallback を使う。ここで row/col scale が fresh legacy scaling と変わると、reduced costs、tie-break、Farkas/guard 判定が変わり、B&B tree が変わる。

### 166a85a

この commit は legacy cache を削除し、cache 対象を bounded form だけに絞った。テスト名も `cached_mip_lp_legacy_fallback_does_not_reuse_root_scaling` に変わり、bounded dispatch disabled のとき `solve_with_node_lp_cache` が `None` を返すことを sentinel 化している。

これは「legacy の cached scaling を小さく直した」のではなく、「legacy には cached scaling を渡さない」という撤退である。つまり incremental 修正で exact 化できる範囲は bounded path に限られる、という実装上の結論になっている。

### 81250f7

bounded-artificial path については、cached solve と fresh solve の `status/objective/solution/warm_start_basis.basis/warm_start_basis.x_b` が bit exact で一致する sentinel を追加した。これは bounded-artificial 内部では root recipe reuse を exact にできる可能性を示す。

ただし sentinel は小さい Ge+UB MILP の単体 node であり、gt2 の B&B 全体で「どの node も bounded-artificial から legacy へ落ちない」ことや、「warm-start basis kind が常に一致する」ことまでは証明していない。

### 91a25d7

この commit 自体にコード差分はなく、結論コミットである。前段の diff から読み取れる実態は、legacy cache を撤去しても gt2 exact 問題が残り、bounded-artificial の局所 bit exact sentinel だけでは B&B tree exact まで届かなかった、というもの。

## 真因

真因は「Ruiz scaling cache の実装ミス」より広い。

1. B&B node は同じ logical LP でも path により canonical matrix が違う。
2. scaling recipe は canonical matrix の row/col 空間に属するので、BSF recipe と SF recipe は互換でない。
3. `WarmStartBasis` は `basis + x_b` だけで、basis が bounded/legacy/augmented のどの空間に属するか、nonbasic-at-upper がどれかを表せない。
4. bounded path は `UbViolationOutOfScope` や Phase I inconclusive で legacy に fall through する。したがって「cache を受け取った node」と「fresh solve が通る path」が一致しない場合がある。
5. GMI cuts は root で legacy tableau を使い、cuts 後の B&B node は finite UB + Ge cut rows を持つ。これが bounded-artificial と legacy fallback の境界を踏みやすい。

小修正で exact 化できる反例は、限定条件つきなら存在する。例えば「bounded dispatch が必ず成功する Le-only/Ge+UB の小問題」では 81250f7 の sentinel のように exact 化できる。しかし gt2 --cuts のような全 B&B tree exact では、すべての fallback と basis-kind 変換を塞ぐ必要があり、それは既に path 統一に近い作業量になる。

Verdict: 「基盤統一が必要」は支持する。ただし「legacy standard form 表現を即削除する」ではない。solver execution path を統一し、GMI tableau 用 legacy adapter は検証付きで残すのが現実的である。

## 統一設計

中心は `UnifiedBoundedForm` と `SimplexBasisState` である。

`UnifiedBoundedForm`:

- variable shift/split と row normalization は一箇所で行う。
- finite UB は常に explicit `upper_bounds` として保持する。
- artificial columns は form へ恒久追加せず、Phase I view として扱う。
- `StandardForm` UB-row expansion は solver entry から外し、GMI/tableau adapter としてだけ提供する。

`SimplexBasisState`:

- `basis: Vec<usize>`
- `x_b: Vec<f64>`
- `at_upper: Vec<bool>`
- `space: BasisSpace` (`Bounded`, `BoundedAugmented { n_struct }`, `LegacyTableau` など)
- `scaling_id` / canonical layout fingerprint
- optional pricing state

重要な invariant:

- solver core は `BasisSpace::Bounded` または `BoundedAugmented` だけを受ける。
- B&B child warm-start は parent の `space` と current canonical fingerprint が一致するときだけ採用する。
- legacy tableau basis は GMI adapter 内で消費し、B&B node warm-start と混ぜない。
- scaling reuse は `UnifiedBoundedForm` の row/col fingerprint が一致し、node で変わる値が `b`, `upper_bounds`, `obj_offset` に限定される場合だけ許可する。

## 段階計画

### Stage 0: path census と profile gate

目的: 統一前に現 path の使用数を常設で測る。

touched files:

- `otspot-core/src/simplex/dual_advanced/mod.rs`
- `otspot-core/src/simplex/entry.rs`
- `otspot-core/src/mip/mod.rs`
- `otspot-core/src/problem.rs` または stats 定義周辺

規模: 150-250 lines

sentinel:

- env OFF で stats がゼロ/overhead なし。
- env ON で bounded / bounded-artificial / legacy fallback / zero-row counts の合計が LP solve count と一致。
- no-op fail: bounded fallback counter を増やさない変更で `gt2 --cuts` profile test が失敗する。

### Stage 1: `SimplexBasisState` 型を追加し、既存 `WarmStartBasis` へ lossless bridge

目的: basis の所属空間と `at_upper` を型で表せるようにする。ただし solver behavior は変えない。

touched files:

- `otspot-core/src/options.rs`
- `otspot-core/src/problem.rs`
- `otspot-core/src/simplex/dual_advanced/mod.rs`
- `otspot-core/src/simplex/primal/mod.rs`
- `otspot-core/src/mip/mod.rs`

規模: 250-450 lines

sentinel:

- current `WarmStartBasis` roundtrip が従来結果と bit/near exact。
- bounded path の `at_upper` を artificial に全 false へ落とす no-op では、UB-active synthetic warm-start が反復数または objective sentinel で fail。
- mismatched `BasisSpace` を渡すと warm-start 採用されず cold-start になる。

### Stage 2: bounded standard form を唯一の node solver canonical form にする

目的: `solve_dual_advanced` の finite UB dispatch を「試す」形から、bounded form を常に作る形へ移す。legacy fallback はまだ残すが、fallback 入口で理由を typed にする。

touched files:

- `otspot-core/src/simplex/standard_form.rs`
- `otspot-core/src/simplex/dual_advanced/mod.rs`
- `otspot-core/src/simplex/dual_advanced/bounded_core.rs`
- `otspot-core/src/simplex/entry.rs`

規模: 400-800 lines

sentinel:

- `tests_bounded_form` の BSF/SF equivalence は維持。
- Ge+UB が bounded Phase I path に入る sentinel は維持。
- no-op fail: finite UB 問題を legacy へ直行させると dispatch counter sentinel が fail。
- objective guard: original-space objective recompute と KKT guard は全 path で維持。

### Stage 3: bounded-artificial を `BoundedAugmented` view に分離

目的: artificial columns を一時 view として扱い、Phase I 後に `SimplexBasisState` を `Bounded` へ戻す変換を明示する。

touched files:

- `otspot-core/src/simplex/dual_advanced/mod.rs`
- `otspot-core/src/simplex/dual_advanced/bounded_core.rs`
- `otspot-core/src/simplex/primal/*`

規模: 500-900 lines

sentinel:

- artificial が basic に残る場合は `Bounded` warm-start を返さない。
- Phase I residual artificial > tol は Infeasible。
- no-op fail: artificial pin-to-zero を外すと multi-artificial sentinel が fail。
- cached/fresh bounded-artificial bit exact sentinel を gt2-derived small fixture へ拡張。

### Stage 4: legacy solver fallback を削除し、legacy tableau adapter へ隔離

目的: B&B node solve が legacy `StandardForm` solver core へ落ちないようにする。GMI は legacy tableau adapter として残す。

touched files:

- `otspot-core/src/simplex/dual_advanced/mod.rs`
- `otspot-core/src/simplex/dual.rs`
- `otspot-core/src/mip/cuts.rs`
- `otspot-core/src/simplex/tests_bounded_form.rs`

規模: 600-1000 lines

sentinel:

- `bounded_dispatch_disabled` 系テストを削除/置換し、fallback count がゼロであることを assert。
- GMI brute-force validity tests は維持。
- `solve_cut_lp` は `LegacyTableauBasis` を返し、B&B child warm-start に使えない型にする。
- no-op fail: GMI に bounded basis を渡すと type/test が fail。

legacy 依存の実コード確認:

- `mip/cuts.rs` は legacy SF tableau を使うと明記している。
- `simplex/tests_bounded_form.rs` は `wrap_to_legacy(build_bounded_standard_form(lp)) == build_standard_form(lp)` を sentinel 化している。
- `dual_advanced` tests には legacy warm-start lb violation、bounded dispatch disabled、legacy fallback correctness の sentinel がある。Stage 4 でこれらを「fallback が必要」から「fallback が発生しない」テストへ置換する。

### Stage 5: Phase 2A scaling reuse を再導入

目的: unified bounded form の fingerprint を key にして node 間 scaling を reuse する。

touched files:

- `otspot-core/src/mip/mod.rs`
- `otspot-core/src/mip/problem.rs`
- `otspot-core/src/simplex/entry.rs`
- `otspot-core/src/simplex/standard_form.rs`
- `otspot-core/src/presolve/scaling.rs`

規模: 350-700 lines

sentinel:

- cache enabled/disabled で `gt2 --cuts` status/objective/nodes が完全一致。
- cached descendant は Ruiz scale call count が増えない。
- fresh/cached single-node comparison は `status/objective/solution/basis/x_b/at_upper` を bit exact または documented tolerance で比較。
- no-op fail: row_scale と col_scale のどちらかを 1 に固定すると objective/KKT sentinel が fail。

## Silent wrong objective リスクと対策

主リスク:

- scaled upper bound と original upper bound の混同
- `obj_offset` の二重加算または欠落
- nonbasic-at-upper を 0 として解釈する誤り
- artificial basic at zero を clean bounded basis と誤認
- legacy tableau basis を B&B node warm-start に混入

対策:

- solver result は常に original objective を recompute する guard を通す。
- `SimplexBasisState.space` と canonical fingerprint が一致しない warm-start は reject。
- `at_upper` を含む basis-state equality sentinel を入れる。
- `gt2 --cuts`, `flugpl --cuts`, Ge+UB synthetic, UB-active warm-start synthetic を CI heavy / targeted test に分ける。

## 統一後に scaling reuse が clean に乗る理由

統一後は、MIP node 間で変わるものが `b`, `upper_bounds`, `obj_offset`, warm-start state に限定される。`A` の sparsity pattern と row/col scaling の所属空間は `UnifiedBoundedForm` に固定されるため、root の scaling recipe を descendant に適用する contract を単純化できる。

現状のように「bounded recipe」「legacy recipe」「augmented artificial recipe」が solver fallback に応じて切り替わることがなくなる。cache miss は canonical fingerprint mismatch の一種類になり、miss 時は fresh unified bounded solve へ戻せばよい。これにより 4cb7e10 のような cross-path cached scaling と、166a85a のような legacy 除外による tree 差分の両方を避けられる。
