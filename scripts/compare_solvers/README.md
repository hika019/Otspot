# compare_solvers — HiGHS / SCIP per-problem comparison harness

Otspot が非 PASS の問題を HiGHS / SCIP が解けるかを
per-problem で突合し、「他ソルバ可・Otspot 不可」のフロンティアを確定する基盤。

## 使い方

```sh
# 前提: 変換ツールのビルド (qplib/cbf を扱う場合のみ必要)
cargo build --release --example dump_problem

# suite 単位でソルバを実行 (逐次、CSV 出力: problem,status,objective,time_sec)
bash scripts/compare_solvers/run_highs.sh data/lp_problems 1000 lp_highs.csv
bash scripts/compare_solvers/run_scip.sh  data/lp_problems 1000 lp_scip.csv

# Otspot 側の既存結果と突合
python3 scripts/compare_solvers/compare.py \
    --otspot bench_results/<suite>/<result>.txt \
    --highs lp_highs.csv --scip lp_scip.csv --out compare.csv

# テスト
python3 scripts/compare_solvers/test_compare.py       # compare.py の単体テスト
python3 scripts/compare_solvers/solve_one_scip.py --self-test  # obj_offset 回帰
```

`compare.py --otspot` は 2 形式を自動判別する:
`bench_parallel.sh` のテキスト出力 (.txt) と `examples/solve_cbf.rs` の
CSV 出力 (.csv、cblib_socp 用)。

## フォーマット対応

| suite の形式 | HiGHS (highspy 1.14.0) | SCIP (PySCIPOpt 6.2.1 / SCIP 10.0.2) | 経路 |
|---|---|---|---|
| .mps / .qps / .QPS | 可 (LP/QP/MILP) | 可 (LP/QP/MILP/MIQP) | 直接。`.mps` 拡張子リネーム (HiGHS) / `readProblem(extension="mps")` (SCIP) |
| .qplib (LP/QP/MILP/MIQP) | 可 | 可 | `dump_problem` 変換 |
| .qplib (QCQP: 二次制約あり) | 不可 `Unsupported(QCQP-...)` | 可 (一般二次制約) | 同上 |
| .cbf (SOCP/MISOCP) | 対象外 `Unsupported(no-SOCP-...)` | 可 (SOC を `t≥0 ∧ Σu²≤t²` の二次制約で表現) | 同上 |

どちらのソルバもこの環境では `.qplib` / `.cbf` のネイティブ reader を持たない
(pyscipopt PyPI wheel の libscip に reader_cbf/reader_qplib が未リンクであることを
`strings` で確認済み)。`dump_problem` は otspot-io の既存パーサを唯一の
変換実装として再利用する (Python 側での二重実装を避ける)。

QPLIB の `maximize` 宣言は `dump_problem` が明示的に拒否する — otspot-io は
parse 時に目的関数を符号反転して minimize 問題として返し、sense が失われる
ため、外部ソルバの目的値が黙って符号反転してしまう。現 data/ は全問
minimize (125/125 実測)。

## 公平性に関する注記 (tolerance / 設定差)

結果を解釈する際は以下の設定差を前提にすること:

| 項目 | Otspot bench | HiGHS (このランナー) | SCIP (このランナー) |
|---|---|---|---|
| feasibility tol | eps=1e-6 (primal+dual) | 既定値 primal/dual 1e-7 | 既定値 feastol 1e-6 / dualfeastol 1e-7 |
| MIP gap | gap_tol=1e-6 | 既定値 mip_rel_gap 1e-4 | 既定値 limits/gap 0 |
| time limit | 1000s (基準) | `--timeout` (solver 内部 + 外部 timeout+30s) | 同左 |
| threads | single | 既定 (LP/QP はシングル) | 既定 |

- ランナーは tolerance を明示設定しない (ソルバ既定のまま)。フロンティア
  検出 (解けた/解けない) が目的であり、目的値の 1e-6 級の照合をする場合は
  この差を考慮するか、lp_vs_highs.sh のように options を明示すること。
- **SCIP はヒューリスティクス OFF で実行される** (非線形制約を含むモデルのみ。
  純 LP/MILP は既定のまま)。このマシンの仮想 CPU (QEMU, AVX/AVX2/FMA 無し・
  SSE4.2 まで) では SCIP の非線形制約向けヒューリスティクスが
  `Illegal instruction` でクラッシュするための回避。**SCIP の実力を
  過小評価する方向のバイアス**であり、SCIP が解けなかった問題を
  「SCIP でも不可」と断定する材料にはできない (逆方向 — SCIP が解けた事実 —
  は有効)。compare.csv の `notes` 列 (`scip_heuristics_off_on_nonlinear`)
  にこの注記が入る。
- SOCP を汎用二次制約として SCIP に渡すため、専用 conic ソルバ
  (ECOS/Mosek/Clarabel 等) より大幅に遅い。cblib での SCIP timelimit は
  「conic としては不明」と解釈すること。

## compare.csv の列

- `otspot_status` / `highs_status` / `scip_status`, 各 time/obj
- `otspot_unverified_claim`: solve_cbf 形式の Infeasible/Unbounded は
  baseline 照合がない生の主張 (cblib の baseline は otspot_self 由来で
  独立検証にならない) — True の行は PASS 扱いしない。外部ソルバが同一結論
  なら合意として `other_solver_wins=False`、異なる結論なら True (要トリアージ)。
- `other_solver_wins`: フロンティア (他ソルバが決着・Otspot 非 PASS)。
- `notes`: SCIP 行への公平性注記。

## 環境メモ

- HiGHS: `pip install --user highspy` (導入済み 1.14.0)
- SCIP: `pip install --user --break-system-packages pyscipopt` (6.2.1,
  SCIP 10.0.2 同梱。apt/sudo 不可・ensurepip 欠如で venv 不可のため)
- ランナーの終了ラベル: ソルバ内 timeout = 各ソルバの status
  (`TimeLimit`/`timelimit`)、外部 kill (timeout+30s) = `TIMEOUT_HARD`、
  プロセス異常終了 = `CRASH(exit=N)` (132=SIGILL, 139=SIGSEGV)。
