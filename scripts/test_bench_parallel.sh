#!/bin/bash
# test_bench_parallel.sh — bench_parallel.sh の worker abend 対処 sentinel。
#
# 過去観測の二大退化 (TMPDIR_BASE 消失で worker が group_000 を spin、
# gtimeout 強制終了で問題が集計から脱落) を再現し、修正後の挙動を fact 化する。
#
# 使い方: bash scripts/test_bench_parallel.sh
# 実行時間: ~5s。CI / 手動 sanity 用。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_SCRIPT="$SCRIPT_DIR/bench_parallel.sh"

# 数値定数
WORKER_EXIT_WAIT_S=10   # TMPDIR 消滅後 worker が止まるまでの最大待ち秒
SLOW_WORKER_SLEEP_S=2   # TMPDIR 削除 race fixture が完走しないための待機秒

PASS=0
FAIL=0
TMP_ROOT=$(mktemp -d /tmp/test_bench_parallel.XXXXXX)
trap 'rm -rf "$TMP_ROOT"' EXIT

assert_eq() {
  local actual="$1" expected="$2" label="$3"
  if [[ "$actual" == "$expected" ]]; then
    echo "  PASS: $label"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: $label (expected='$expected' actual='$actual')" >&2
    FAIL=$((FAIL + 1))
  fi
}

assert_contains() {
  local haystack="$1" needle="$2" label="$3"
  if grep -qF -- "$needle" <<<"$haystack"; then
    echo "  PASS: $label"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: $label (missing: '$needle')" >&2
    echo "  --- haystack ---" >&2
    echo "$haystack" >&2
    echo "  ----------------" >&2
    FAIL=$((FAIL + 1))
  fi
}

# --- shared fixture: 偽 solver と data dir ---
DATA_DIR="$TMP_ROOT/data"
mkdir -p "$DATA_DIR"
# qps_benchmark/bench_qplib の拡張子判定のため .qps を 3 つ用意
for n in p1 p2 p3; do
  cat > "$DATA_DIR/${n}.qps" <<EOF
NAME          ${n}
ROWS
 N  COST
COLUMNS
    X1  COST  1.0
RHS
BOUNDS
ENDATA
EOF
done

# 偽 solver_bench.sh を持つ branch 用 SOLVER_DIR を用意
FAKE_SOLVER_ROOT="$TMP_ROOT/fake_solver"
mkdir -p "$FAKE_SOLVER_ROOT/scripts"
mkdir -p "$FAKE_SOLVER_ROOT/data/baseline_objectives"
touch "$FAKE_SOLVER_ROOT/data/baseline_objectives/netlib_lp.csv"

cat > "$FAKE_SOLVER_ROOT/scripts/solver_bench.sh" <<'SOL'
#!/bin/bash
# テスト用 stub: BENCH_TEST_MODE で挙動切替
#   ok        → 即時 PASS Summary
#   slow      → 1s sleep してから PASS（TMPDIR 削除の race を観測しやすくする）
#   timeout   → exit 124 (gtimeout 強制終了相当)
#   fail      → exit 7   (真の異常終了相当)
#   partial   → Summary を一部だけ出して exit 7
set -u
mode="${BENCH_TEST_MODE:-ok}"
echo "[fake] mode=$mode run=${BENCH_TEST_RUN_ID:-}" >&2
case "$mode" in
  timeout) exit 124 ;;
  fail)    exit 7   ;;
  partial) prob=$(basename "$(find "$2" -type l -o -type f | head -1)"); echo "  $prob  PASS"; echo "=== Summary ==="; echo "    PASS: 1"; echo "    ERROR: 0"; echo "    TOTAL: 1"; exit 7 ;;
  slow)    sleep "${SLOW_WORKER_SLEEP_S:-5}"; echo "  PASS"; echo "    PASS: 1"; echo "    TOTAL: 1" ;;
  *)
    prob=$(basename "$(find "$2" -type l -o -type f | head -1)")
    echo "  $prob  PASS"
    echo "    PASS:    1"
    echo "    TOTAL:   1"
    ;;
esac
SOL
chmod +x "$FAKE_SOLVER_ROOT/scripts/solver_bench.sh"

# bench_parallel.sh は \$SCRIPT_DIR/solver_bench.sh を呼ぶので、本物の bench_parallel.sh
# を fake scripts/ にコピーして fake solver_bench.sh と同居させる
cp "$BENCH_SCRIPT" "$FAKE_SOLVER_ROOT/scripts/bench_parallel.sh"
FAKE_BENCH="$FAKE_SOLVER_ROOT/scripts/bench_parallel.sh"

# ---------------------------------------------------------------
echo "=== Test 1: child exit=124 (gtimeout 強制) → EXTERNAL_TIMEOUT 集計 + exit 1 ==="
OUT="$TMP_ROOT/t1.out"
export BENCH_TEST_MODE=timeout
set +e
SUMMARY=$(SOLVER_DIR="$FAKE_SOLVER_ROOT" \
  bash "$FAKE_BENCH" \
  --data-dir "$DATA_DIR" \
  --timeout 1 \
  --eps 1e-6 \
  --jobs 1 \
  --output "$OUT" 2>&1)
T1_EXIT=$?
unset BENCH_TEST_MODE
assert_contains "$SUMMARY" "外部timeout発火" "gtimeout 検知 log 出力"
TIMEOUT_LINE=$(grep -E "^\s+TIMEOUT:" "$OUT" | head -1 | awk '{print $2}')
EXTERNAL_TIMEOUT_LINE=$(grep -E "^\s+EXTERNAL_TIMEOUT:" "$OUT" | head -1 | awk '{print $2}')
assert_eq "$TIMEOUT_LINE" "0" "通常 TIMEOUT には混入しない"
assert_eq "$EXTERNAL_TIMEOUT_LINE" "3" "3 件全て EXTERNAL_TIMEOUT 集計"
assert_eq "$T1_EXIT" "1" "EXTERNAL_TIMEOUT > 0 は bench_parallel exit 1"
# FAILED_GROUPS 行は出てはいけない（gtimeout は正常な safety net 動作）
if grep -q "★ 異常終了グループ" "$OUT"; then
  echo "  FAIL: gtimeout 経路で 異常終了グループ ラベル混入" >&2
  FAIL=$((FAIL + 1))
else
  echo "  PASS: gtimeout は 異常終了 扱いされない"
  PASS=$((PASS + 1))
fi

# ---------------------------------------------------------------
echo "=== Test 2: TMPDIR_BASE 消滅 → worker fail-fast (group_000 spin なし) ==="
# 偽 bench を background で起動し、TMPDIR_BASE を即座に rm -rf して
# worker の挙動を観測。修正前は group_000 を無限ループしていた。
OUT="$TMP_ROOT/t2.out"
LOG="$TMP_ROOT/t2.log"
# slow モード = 各 group SLOW_WORKER_SLEEP_S 秒。worker が active な間に
# 途中で TMPDIR を消すと fail-fast 経路に確実に入る。
export BENCH_TEST_MODE=slow
export SLOW_WORKER_SLEEP_S
export BENCH_TEST_RUN_ID="t2-$$"
(
  SOLVER_DIR="$FAKE_SOLVER_ROOT" \
  bash "$FAKE_BENCH" \
    --data-dir "$DATA_DIR" \
    --timeout 60 \
    --eps 1e-6 \
    --jobs 1 \
    --output "$OUT" >"$LOG" 2>&1 || true
) &
BENCH_PID=$!

# worker が稼働中になるまで待つ。 counter ファイルは worker 起動直前に作られ、
# group_001.log は worker_func が最初に open するため両方の存在を確認。
for _ in $(seq 1 100); do
  DIR=""
  for candidate in /tmp/bench_parallel.*; do
    if [[ -d "$candidate" && -f "$candidate/counter" && -f "$candidate/group_001.log" ]] \
      && grep -qF "[fake] mode=slow run=$BENCH_TEST_RUN_ID" "$candidate/group_001.log" 2>/dev/null; then
      DIR="$candidate"
      break
    fi
  done
  [[ -n "$DIR" ]] && break
  sleep 0.05
done
if [[ -z "${DIR:-}" || ! -d "$DIR" || ! -f "$DIR/counter" ]]; then
  echo "  FAIL: TMPDIR_BASE/worker 起動検出失敗" >&2
  FAIL=$((FAIL + 1))
  kill -9 $BENCH_PID 2>/dev/null || true
else
  rm -rf "$DIR"
  # worker fail-fast を待つ（最大 WORKER_EXIT_WAIT_S で抜けることを保証）
  ELAPSED=0
  while kill -0 $BENCH_PID 2>/dev/null && [[ $ELAPSED -lt $WORKER_EXIT_WAIT_S ]]; do
    sleep 0.5
    ELAPSED=$((ELAPSED + 1))
  done
  if kill -0 $BENCH_PID 2>/dev/null; then
    kill -9 $BENCH_PID 2>/dev/null || true
    echo "  FAIL: TMPDIR 消滅後も worker が ${WORKER_EXIT_WAIT_S}s 内に終了せず (spin 退化)" >&2
    FAIL=$((FAIL + 1))
  else
    echo "  PASS: TMPDIR 消滅で worker ${WORKER_EXIT_WAIT_S}s 以内停止"
    PASS=$((PASS + 1))
  fi
  if grep -q "group_000" "$LOG" 2>/dev/null; then
    echo "  FAIL: group_000 spin 痕跡検出" >&2
    FAIL=$((FAIL + 1))
  else
    echo "  PASS: group_000 spin なし"
    PASS=$((PASS + 1))
  fi
  # fail-fast 経路は TMPDIR 消失 検知 もしくは 不正 group_num (state破損) 検知
  if grep -qE "TMPDIR_BASE.*消失|不正 group_num" "$LOG" 2>/dev/null; then
    echo "  PASS: fail-fast log 出力"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: fail-fast log 未出力" >&2
    echo "  --- LOG ($LOG) ---" >&2
    cat "$LOG" >&2
    echo "  ------------------" >&2
    FAIL=$((FAIL + 1))
  fi
fi
unset BENCH_TEST_RUN_ID

# ---------------------------------------------------------------
echo "=== Test 3: 真の異常終了 (exit=7) → FAILED_GROUPS 扱い ==="
OUT="$TMP_ROOT/t3.out"
LOG="$TMP_ROOT/t3.log"
unset BENCH_TEST_MODE
export BENCH_TEST_MODE=fail
SOLVER_DIR="$FAKE_SOLVER_ROOT" \
bash "$FAKE_BENCH" \
  --data-dir "$DATA_DIR" \
  --timeout 5 \
  --eps 1e-6 \
  --jobs 1 \
  --output "$OUT" >"$LOG" 2>&1 || true
assert_contains "$(cat "$LOG")" "異常終了 (exit=7)" "真の exit≠124 は 異常終了 維持"
if grep -q "★ 異常終了グループ" "$OUT"; then
  echo "  PASS: FAILED_GROUPS に集約"
  PASS=$((PASS + 1))
else
  echo "  FAIL: FAILED_GROUPS に集約されず" >&2
  FAIL=$((FAIL + 1))
fi
ERROR_LINE=$(grep -E "^\s+ERROR:" "$OUT" | head -1 | awk '{print $2}')
TOTAL_LINE=$(grep -E "^\s+TOTAL:" "$OUT" | head -1 | awk '{print $2}')
assert_eq "$ERROR_LINE" "3" "異常終了も ERROR に計上"
assert_eq "$TOTAL_LINE" "3" "異常終了も TOTAL に計上"

# ---------------------------------------------------------------
echo "=== Test 3b: 異常終了 + partial Summary → 未集計分を ERROR/TOTAL に補完 ==="
OUT="$TMP_ROOT/t3b.out"
LOG="$TMP_ROOT/t3b.log"
export BENCH_TEST_MODE=partial
SOLVER_DIR="$FAKE_SOLVER_ROOT" \
bash "$FAKE_BENCH" \
  --data-dir "$DATA_DIR" \
  --timeout 5 \
  --eps 1e-6 \
  --jobs 1 \
  --output "$OUT" >"$LOG" 2>&1 || true
unset BENCH_TEST_MODE
ERROR_LINE=$(grep -E "^\s+ERROR:" "$OUT" | head -1 | awk '{print $2}')
TOTAL_LINE=$(grep -E "^\s+TOTAL:" "$OUT" | head -1 | awk '{print $2}')
assert_eq "$ERROR_LINE" "3" "partial Summary 後の異常終了も ERROR に補完"
assert_eq "$TOTAL_LINE" "3" "partial Summary 後の異常終了も TOTAL に補完"
assert_contains "$(cat "$OUT")" "worker_exit=7" "Summary 後の worker 非0終了を detail に明示"

# ---------------------------------------------------------------
echo "=== Test 3c: solver_bench milp_solve 内部の外部timeout → EXTERNAL_TIMEOUT/TOTAL ==="
MIP_DATA="$TMP_ROOT/mip_data"
FAKE_TIMEOUT_DIR="$TMP_ROOT/fake_timeout_bin"
mkdir -p "$MIP_DATA" "$FAKE_TIMEOUT_DIR"
: > "$MIP_DATA/p.mps"
cat > "$FAKE_TIMEOUT_DIR/timeout" <<'FAKE_TIMEOUT'
#!/bin/bash
exit 124
FAKE_TIMEOUT
chmod +x "$FAKE_TIMEOUT_DIR/timeout"
MIP_OUT="$TMP_ROOT/t3c.out"
(
  cd "$REPO_ROOT" && \
  _BENCH_PARALLEL_CALLER=1 PATH="$FAKE_TIMEOUT_DIR:$PATH" \
  bash "$SCRIPT_DIR/solver_bench.sh" milp_solve "$MIP_DATA" --eps 1e-6 --timeout 5
) > "$MIP_OUT" 2>&1
MIP_EXT_LINE=$(grep -E "^\s+EXTERNAL_TIMEOUT:" "$MIP_OUT" | head -1 | awk '{print $2}')
MIP_TOTAL_LINE=$(grep -E "^\s+TOTAL:" "$MIP_OUT" | head -1 | awk '{print $2}')
assert_eq "$MIP_EXT_LINE" "1" "MIP per-file 外部timeoutを EXTERNAL_TIMEOUT に計上"
assert_eq "$MIP_TOTAL_LINE" "1" "MIP per-file 外部timeoutを TOTAL に計上"

# ---------------------------------------------------------------
echo "=== Test 4: per-category aggregation (KKT_FAIL counter + CATEGORY_SUM 整合性) ==="
# 目的: bench_qplib.rs に新 STATUS が追加された際、bench_parallel.sh の
# Summary printer / CATEGORY_SUM 整合性 check に集計漏れが起きないことを検知する。
# 過去 #114 で KKT_FAIL が aggregator に未配線で fence post bench 集計から漏れた。
#
# 合成 fixture (per-file synthetic Summary; 1 file = 1 group, jobs=3 で 3 group):
#   p1.qps → PASS=3, TOTAL=3
#   p2.qps → KKT_FAIL=4, TOTAL=4
#   p3.qps → KKT_FAIL=3, TOTAL=3
# 期待: TOTAL_PASS=3, TOTAL_KKT_FAIL=7, TOTAL=10, カテゴリ合算==TOTAL (エラー 0 件)。
FIXTURE_PASS=3
FIXTURE_KKT_FAIL_G2=4
FIXTURE_KKT_FAIL_G3=3
FIXTURE_KKT_FAIL_TOTAL=$(( FIXTURE_KKT_FAIL_G2 + FIXTURE_KKT_FAIL_G3 ))
FIXTURE_TOTAL=$(( FIXTURE_PASS + FIXTURE_KKT_FAIL_TOTAL ))

unset BENCH_TEST_MODE
# 既存 fake (BENCH_TEST_MODE 切替) を per-file 集計を吐くものに置き換え。
# Test 4 は本 script 末尾のため後続 test への副作用なし。
cat > "$FAKE_SOLVER_ROOT/scripts/solver_bench.sh" <<SOL
#!/bin/bash
# Test 4 専用 stub: group_dir 内のファイル名で per-group Summary を切替。
set -u
prob=\$(basename "\$(find "\$2" -type l -o -type f | head -1)")
case "\$prob" in
  p1.qps)
    echo "  \$prob  PASS"
    echo "    PASS:    ${FIXTURE_PASS}"
    echo "    TOTAL:   ${FIXTURE_PASS}"
    ;;
  p2.qps)
    echo "  \$prob  KKT_FAIL"
    echo "    KKT_FAIL: ${FIXTURE_KKT_FAIL_G2}"
    echo "    TOTAL:    ${FIXTURE_KKT_FAIL_G2}"
    ;;
  p3.qps)
    echo "  \$prob  KKT_FAIL"
    echo "    KKT_FAIL: ${FIXTURE_KKT_FAIL_G3}"
    echo "    TOTAL:    ${FIXTURE_KKT_FAIL_G3}"
    ;;
  *)
    echo "  \$prob  ERROR"
    echo "    ERROR: 1"
    echo "    TOTAL: 1"
    ;;
esac
SOL
chmod +x "$FAKE_SOLVER_ROOT/scripts/solver_bench.sh"

OUT="$TMP_ROOT/t4.out"
LOG="$TMP_ROOT/t4.log"
SOLVER_DIR="$FAKE_SOLVER_ROOT" \
bash "$FAKE_BENCH" \
  --data-dir "$DATA_DIR" \
  --timeout 1 \
  --eps 1e-6 \
  --jobs 3 \
  --output "$OUT" >"$LOG" 2>&1 || true

KKT_FAIL_LINE=$(grep -E "^\s+KKT_FAIL:" "$OUT" | head -1 | awk '{print $2}')
PASS_LINE=$(grep -E "^\s+PASS:" "$OUT" | head -1 | awk '{print $2}')
TOTAL_LINE=$(grep -E "^\s+TOTAL:" "$OUT" | head -1 | awk '{print $2}')
assert_eq "$PASS_LINE" "$FIXTURE_PASS" "TOTAL_PASS 集計 (=${FIXTURE_PASS})"
assert_eq "$KKT_FAIL_LINE" "$FIXTURE_KKT_FAIL_TOTAL" "TOTAL_KKT_FAIL 集計 (=${FIXTURE_KKT_FAIL_TOTAL})"
assert_eq "$TOTAL_LINE" "$FIXTURE_TOTAL" "TOTAL 集計 (=${FIXTURE_TOTAL})"
if grep -q "エラー: カテゴリ合算" "$LOG"; then
  echo "  FAIL: 正常 fixture でカテゴリ合算 mismatch エラーが出力された" >&2
  echo "  --- LOG ($LOG) ---" >&2
  grep "エラー:" "$LOG" >&2 || true
  FAIL=$((FAIL + 1))
else
  echo "  PASS: カテゴリ合算 == TOTAL (エラー 0 件)"
  PASS=$((PASS + 1))
fi

# toggle proof (no-op 書換実証, ref: feedback_sentinel_must_fail_under_noop):
# bench_parallel.sh の TOTAL_KKT_FAIL 集計行 (line 363: `${kkt_fail:-0}` 加算) を
# sed で除去した複製を同じ FAKE_SOLVER_ROOT/scripts 内に配置して実行する。
# (TOGGLE_BENCH が SCRIPT_DIR から solver_bench.sh を探すため同居必須。)
# pattern `kkt_fail:-0` は line 363 のみで unique。
# CATEGORY_SUM が KKT_FAIL=7 分だけ目減りして mismatch エラーが出ることを確認する。
# エラー line 形式: "エラー: カテゴリ合算(N) ≠ TOTAL(M)"
TOGGLE_BENCH="$FAKE_SOLVER_ROOT/scripts/bench_parallel_no_kkt.sh"
sed '/kkt_fail:-0/d' "$FAKE_BENCH" > "$TOGGLE_BENCH"
# sed が aggregation 行を正しく削除したか前提検証 (見つからない/複数 hit を防ぐ)
TOGGLE_REMOVED=$(( $(grep -c "kkt_fail:-0" "$FAKE_BENCH") - $(grep -c "kkt_fail:-0" "$TOGGLE_BENCH") ))
assert_eq "$TOGGLE_REMOVED" "1" "toggle sed: TOTAL_KKT_FAIL 集計行を 1 行削除"
chmod +x "$TOGGLE_BENCH"
TOGGLE_OUT="$TMP_ROOT/t4_toggle.out"
TOGGLE_LOG="$TMP_ROOT/t4_toggle.log"
SOLVER_DIR="$FAKE_SOLVER_ROOT" \
bash "$TOGGLE_BENCH" \
  --data-dir "$DATA_DIR" \
  --timeout 1 \
  --eps 1e-6 \
  --jobs 3 \
  --output "$TOGGLE_OUT" >"$TOGGLE_LOG" 2>&1
TOGGLE_EXIT=$?
EXPECT_TOGGLE_WARN="エラー: カテゴリ合算(${FIXTURE_PASS}) ≠ TOTAL(${FIXTURE_TOTAL})"
if [[ $TOGGLE_EXIT -ne 0 ]] && grep -qF "$EXPECT_TOGGLE_WARN" "$TOGGLE_LOG"; then
  echo "  PASS: toggle で sentinel 発火 (exit=$TOGGLE_EXIT, CATEGORY_SUM=${FIXTURE_PASS} ≠ TOTAL=${FIXTURE_TOTAL})"
  PASS=$((PASS + 1))
else
  echo "  FAIL: toggle で sentinel 未発火 (KKT_FAIL 集計欠落を検知できず)" >&2
  echo "  --- expected ---" >&2
  echo "  exit!=0 and $EXPECT_TOGGLE_WARN" >&2
  echo "  --- TOGGLE_LOG ($TOGGLE_LOG) ---" >&2
  grep "エラー:" "$TOGGLE_LOG" >&2 || echo "  (no エラー line found)" >&2
  echo "  ----------------" >&2
  FAIL=$((FAIL + 1))
fi

# ---------------------------------------------------------------
echo ""
echo "=== Result: PASS=$PASS FAIL=$FAIL ==="
[[ $FAIL -eq 0 ]] || exit 1
exit 0
