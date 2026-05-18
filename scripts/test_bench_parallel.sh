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
set -u
mode="${BENCH_TEST_MODE:-ok}"
echo "[fake] mode=$mode" >&2
case "$mode" in
  timeout) exit 124 ;;
  fail)    exit 7   ;;
  slow)    sleep 1; echo "  PASS"; echo "    PASS: 1"; echo "    TOTAL: 1" ;;
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
echo "=== Test 1: child exit=124 (gtimeout 強制) → TIMEOUT 集計 ==="
OUT="$TMP_ROOT/t1.out"
export BENCH_TEST_MODE=timeout
SUMMARY=$(SOLVER_DIR="$FAKE_SOLVER_ROOT" \
  bash "$FAKE_BENCH" \
  --data-dir "$DATA_DIR" \
  --timeout 1 \
  --eps 1e-6 \
  --jobs 1 \
  --output "$OUT" 2>&1) || true
unset BENCH_TEST_MODE
assert_contains "$SUMMARY" "外部timeout発火" "gtimeout 検知 log 出力"
TIMEOUT_LINE=$(grep -E "^\s+TIMEOUT:" "$OUT" | head -1 | awk '{print $2}')
assert_eq "$TIMEOUT_LINE" "3" "3 件全て TIMEOUT 集計"
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
# slow モード = 各 group 1s。3 group = 3s 程度 worker が active なので
# 途中で TMPDIR を消すと fail-fast 経路に確実に入る。
export BENCH_TEST_MODE=slow
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
  DIR=$(ls -dt /tmp/bench_parallel.* 2>/dev/null | head -1)
  if [[ -n "$DIR" && -d "$DIR" && -f "$DIR/counter" && -f "$DIR/group_001.log" ]]; then
    break
  fi
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

# ---------------------------------------------------------------
echo ""
echo "=== Result: PASS=$PASS FAIL=$FAIL ==="
[[ $FAIL -eq 0 ]] || exit 1
exit 0
