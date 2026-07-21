---
name: bench
description: ベンチマーク実行時に必ず読む。実行コマンド・基準値 (timeout/eps/jobs)・順次実行・baseline との per-problem diff による退化検出手順を定義する。
---

# ベンチ実行

## 基準値
- timeout=1000s (バグ再現の短縮時は 400s 可)、eps=1e-6 (任意で 1e-4 / 1e-8 の3点)。
- jobs=6 固定。CPU 取り合いを避けるため、問題集 (suite) は並行させず順次実行する。
- 実行は haiku subagent へ委譲し、background で完了通知を待つ。

## コマンド
`scripts/bench_parallel.sh` を使う (solver_bench.sh 経由。バイナリ直接呼び出しは結果集計が壊れるため使わない):
```
bash scripts/bench_parallel.sh --data-dir data/<suite> --timeout 1000 --eps 1e-6 --jobs 6 --output <file>
```
- --jobs は必須 (暗黙デフォルトなし)。形式混在ディレクトリ (.mps と .qps 等) は非対応。
- README の性能表は netlib standard (data/lp_problems, 109問) / Maros-Mészáros (data/maros_meszaros, 138問) 等、README に書かれた suite と同一条件で取る。

## suite → data ディレクトリ
LP: lp_problems (standard) / _extra / _hard / _canary / _infeas / _unbounded。
QP: maros_meszaros, osqp_bench(+_illscaled), mpc_qp, qp_dense_a, qp_infeasible, qp_unbounded。
QPLIB: qplib, qplib_nonconvex(_official)。MIP: miplib_small。Conic: cblib_socp。(qplib_unsupported は未対応形式の隔離置き場でベンチ対象外。)
参照値: data/baseline_objectives/*.csv。

## 結果の扱い
- 合計 PASS 数だけで判定しない。**per-problem diff** を前回結果/baseline と取り、PASS→非PASS の1問でも出たら退化として扱い、bug-frontier skill の6状態 (①regression か ②既存バグ表面化か、⑤TIMEOUT か ⑥非収束か) で分類する。
- TIMEOUT は反復軌跡 (iters/残差の推移) を実測してから ⑤/⑥ を判定する。status 文字列で決めない。
- 報告には suite 名・eps・timeout・jobs・PASS/FAIL 数・退化/改善の問題名リストを必ず含める。
- baseline CSV の修正はベンチ結果のレビューとは独立に、それ自体のレビュー (出典・別ソルバでの再計算根拠) を要する。

## 既知の罠
- bench の iters は純粋な IPM 反復数ではなく multi-attempt driver の累積課金値。iters の差 ≠ IPM の変化。
- QP の SUBOPT を KKT 許容誤差の緩和 (データノルム相対 tol 化) で「直す」のは過去2回 false-Optimal (BOYD1 等) を生んで revert 済みの罠。per-component `1+|v|` 相対化は意図的に正しい。残差側を IR/拡張精度で下げることで対処する。
