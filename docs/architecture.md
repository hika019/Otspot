# Architecture

Otspot は下向きの依存関係を持つレイヤ構成へ移行している。

```text
facade / model / I/O
        ↓
solver algorithms (`otspot-core`)
   ├──→ canonical IR (`otspot-ir`) ──→ numerics (`otspot-num`)
   └──→ presolve pipeline (`otspot-presolve`) ──┘
```

## レイヤ

- `otspot-num`: sparse storage、AMD、LDL/DD-LDL、MINRES、Ruiz、KKT backend、
  timeout/cancellation。問題型には依存しない。
- `otspot-ir`: LP/QP/QCQP/SOCP と整数派生を表す `OptimizationProblem`、
  `SolveContext`、`SolveOutcome`、`Solver`。
- `otspot-presolve`: LP/QPで共有するfixpoint、pass-limit、deadline/cancel制御。
- `otspot-core`: simplex/IPM/MIP/conic と旧公開API。旧 `sparse`/`linalg` パスは
  `otspot-num` の互換re-export。`solve_ir`でcanonical IRをproduction solverへ配線。

## 機械ゲート

`Architecture` workflowで以下を検査する。

- レイヤの依存方向とcanonical型・traitの実装所有権
- legacy facadeの薄さと旧実装ディレクトリの再導入
- otspot-core内部コードによる`crate::sparse`/`crate::linalg`/`crate::error::SolverError`/`crate::SolverError`経由の参照（`otspot_num`への直接依存を強制）
- foundation crateとmodule rootのファイルサイズ
- 220行超の関数の新規追加、およびbaseline登録済み長大関数の肥大化
- sparse storage内部fieldへの直接アクセスの増加（既存箇所は減少のみ許可）

ゲート本体にもfixture self-testを設ける。既存長大関数の縮小・分割・削除は許可する。
