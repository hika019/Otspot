# Architecture

Otspot は下向きの依存関係を持つレイヤ構成へ移行している。

```text
facade / model / I/O
        ↓
solver algorithms (`otspot-core`) ──→ numerics (`otspot-num`)
```

## レイヤ

- `otspot-num`: sparse storage、AMD、LDL/DD-LDL、MINRES、Ruiz、KKT backend、
  fixpoint制御 (`run_fixpoint`/`PipelineStop`)、timeout/cancellation。問題型には
  依存しない。
- `otspot-core`: simplex/IPM/MIP/conic と旧公開API。presolveのtransform数学
  (singleton row、bounds tightening 等) はここに同居し、共有するのはfixpointの
  制御セマンティクスのみ — 住所は `otspot-num` に一本化する。旧 `sparse`/`linalg`
  パスは `otspot-num` の互換re-export。

未publishの間に本番呼び出しゼロだった統一IR (`otspot-ir`、`solve_ir`アダプタ) は
撤去済み。presolve orchestration専用crateだった `otspot-presolve` も解体し、
共有制御ロジックのみ `otspot-num` へ移設した。

## 機械ゲート

`Architecture` workflowで以下を検査する。

- レイヤの依存方向とcanonical型・traitの実装所有権
- legacy facadeの薄さと旧実装ディレクトリの再導入
- otspot-core内部コードによる`crate::sparse`/`crate::linalg`/`crate::error::SolverError`/`crate::SolverError`経由の参照（`otspot_num`への直接依存を強制）
- foundation crateとmodule rootのファイルサイズ
- 220行超の関数の新規追加、およびbaseline登録済み長大関数の肥大化（`otspot-core`/`otspot-num`/`otspot-io`/`otspot-model`/`otspot-dev`のsrcを走査）

ゲート本体にもfixture self-testを設ける。既存長大関数の縮小・分割・削除は許可する。

`CscMatrix`/`SparseVec`のstorage field (`col_ptr`/`row_ind`/`values`/`nrows`/`ncols`、`indices`/`len`) は `otspot-num/src/sparse/` 定義側で `pub(crate)` — otspot-core/otspot-io/otspot-model/otspot-dev からの直接dot-accessもstruct-literal構築も、コンパイルエラーとしてRustコンパイラが強制する。旧来はテキストベースの正規表現ratchet (`scripts/check_csc_encapsulation.py`) で「既存箇所は減少のみ許可」を検査していたが、全箇所をアクセサ (`col_ptr()`等) 経由へ移行しフィールドを非公開化したことでratchet自体が不要になり撤去した。
