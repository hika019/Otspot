# LP Timeout Probe 運用手順（Noether修正 前後比較）

この手順は `cont1 / cont11 / cont4` について、Noether の timeout 根因修正の**前後**を
同条件で再計測し、`internal_timeout` と `external_timeout` の内訳を比較するためのものです。

対象スクリプト:

- `scripts/lp_timeout_probe.sh`

## 1. 前提

1. `data/lp_problems_hard` に `cont1.QPS`, `cont11.QPS`, `cont4.QPS` が存在すること
2. `bench_parallel.sh` が通常どおり実行できること
3. 計測中は同一マシン負荷条件を維持すること（比較のため）

## 2. 比較条件（固定）

前後比較では、以下を**必ず同じ値**に固定します。

- `--timeout`
- `--eps`
- `--jobs`
- `--ext-timeout-buffer`

例（短時間確認用）:

- `timeout=10`
- `eps=1e-6`
- `jobs=1`
- `ext-timeout-buffer=5`

## 3. 修正前の計測

`<TAG>` は任意識別子（例: `before_noether_fix`）。

```bash
bash scripts/lp_timeout_probe.sh \
  --timeout 10 \
  --eps 1e-6 \
  --jobs 1 \
  --ext-timeout-buffer 5 \
  --bench-output /private/tmp/lp_timeout_probe_<TAG>.bench.txt \
  --report /private/tmp/lp_timeout_probe_<TAG>.report.txt \
  --class-tsv /private/tmp/lp_timeout_probe_<TAG>.class.tsv
```

この時点で保存する成果物:

- `*.bench.txt`（生ログ）
- `*.report.txt`（人間向け判定）
- `*.class.tsv`（機械可読判定）

## 4. Noether修正を取り込んだ後の計測

修正後も**完全に同一条件**で実行します（パラメータ変更禁止）。

```bash
bash scripts/lp_timeout_probe.sh \
  --timeout 10 \
  --eps 1e-6 \
  --jobs 1 \
  --ext-timeout-buffer 5 \
  --bench-output /private/tmp/lp_timeout_probe_after_noether_fix.bench.txt \
  --report /private/tmp/lp_timeout_probe_after_noether_fix.report.txt \
  --class-tsv /private/tmp/lp_timeout_probe_after_noether_fix.class.tsv
```

## 5. 差分確認（最小）

### 5.1 問題ごとの判定差分

```bash
diff -u \
  /private/tmp/lp_timeout_probe_before_noether_fix.class.tsv \
  /private/tmp/lp_timeout_probe_after_noether_fix.class.tsv
```

### 5.2 件数比較

```bash
echo "[before]"
awk -F'\t' 'NR>1 { c[$2]++ } END { for (k in c) print k, c[k] }' \
  /private/tmp/lp_timeout_probe_before_noether_fix.class.tsv | sort

echo "[after]"
awk -F'\t' 'NR>1 { c[$2]++ } END { for (k in c) print k, c[k] }' \
  /private/tmp/lp_timeout_probe_after_noether_fix.class.tsv | sort
```

## 6. 判定ルール（`lp_timeout_probe.sh` と一致）

- `TIMEOUT` かつ detail に `external_timeout=` を含む: `external_timeout`
- `TIMEOUT` だが `external_timeout=` を含まない: `internal_timeout`
- `SUBOPTIMAL` を含む: `suboptimal`
- `PASS` を含む: `optimal_or_pass`
- `FAIL` または `ERROR` を含む: `failure`

## 7. 手戻り防止の運用メモ

1. 比較対象2回（前後）で timeout や buffer を変えない
2. `class.tsv` を比較の正本にする（`report.txt` は可読用）
3. 必要なら `--from-bench-output` で過去ログを再分類し、判定ロジックを統一する

