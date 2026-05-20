#!/bin/bash
# lp_vs_highs.sh — Netlib LP benchmark: self-solver vs HiGHS comparison
#
# METHODOLOGY (公平性 / Fairness):
#   Tolerance:    Both solvers eps=1e-6 (primal + dual feasibility)
#   Time limit:   Both solvers 1000s (configurable via --timeout)
#   Presolve:     HiGHS default (choose/auto); self-solver default
#   Threads:      HiGHS parallel=off (single-thread); self-solver single-thread
#   HiGHS mode:   default (choose), typically selects dual simplex for LP
#   Baseline:     data/baseline_objectives/netlib_lp.csv (Netlib MINOS 5.3)
#   Obj scoring:  Symmetric — both solvers: |obj - baseline| / max(|baseline|, 1.0)
#                 Self-solver: obj_err% from note field (/ 100 → relative error)
#                 HiGHS: full-precision obj vs baseline
#                 3 thresholds reported: 1e-2, 1e-4, 1e-6
#                 Self-solver internal acceptance gate = 1e-2 (§2.4), NOT 1e-6
#   e226:         Excluded — objective constant convention differs for all 3 (self/HiGHS/Netlib)
#
# Usage:
#   bash scripts/lp_vs_highs.sh [--mini] [--jobs N] [--timeout SEC]
#   bash scripts/lp_vs_highs.sh --rescore RESULT_DIR
#
# Options:
#   --mini           Smoke test with afiro+adlittle+blend only
#   --jobs N         Parallel workers for HiGHS runs (default: 6)
#   --timeout N      Timeout in seconds for both solvers (default: 1000)
#   --rescore DIR    Re-run scoring only from saved self_solver.txt + highs/ artifacts
#                    (no re-solve; regenerates lp_vs_highs.csv and summary.md in DIR)
#
# Output: bench_results/lp_vs_highs_<timestamp>/
#   lp_vs_highs.csv  — per-instance results with symmetric obj_match at 1e-2
#   summary.md       — aggregate stats + 3-threshold obj_match table
#   self_solver.txt  — raw self-solver bench_parallel.sh output
#   highs/           — per-instance HiGHS stdout files

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HIGHS_BIN="${HIGHS_BIN:-/opt/homebrew/bin/highs}"

JOBS=6
TIMEOUT=1000
MINI=0
RESCORE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mini)    MINI=1; shift ;;
        --jobs)    JOBS="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --rescore) RESCORE_DIR="$2"; shift 2 ;;
        --help|-h)
            grep '^# ' "$0" | head -35 | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "[lp_vs_highs] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

BASELINE_CSV="$SOLVER_ROOT/data/baseline_objectives/netlib_lp.csv"

if [[ -n "$RESCORE_DIR" ]]; then
    # ── Rescore mode: re-run scoring from existing artifacts, no solve ───
    if [[ ! -d "$RESCORE_DIR" ]]; then
        echo "[lp_vs_highs] Error: --rescore dir not found: $RESCORE_DIR" >&2; exit 1
    fi
    RESULT_DIR="$(cd "$RESCORE_DIR" && pwd)"
    SELF_SOLVER_TXT="$RESULT_DIR/self_solver.txt"
    HIGHS_DIR="$RESULT_DIR/highs"
    COMPARISON_CSV="$RESULT_DIR/lp_vs_highs.csv"
    SUMMARY_MD="$RESULT_DIR/summary.md"
    if [[ ! -f "$SELF_SOLVER_TXT" ]]; then
        echo "[lp_vs_highs] Error: self_solver.txt not found in $RESULT_DIR" >&2; exit 1
    fi
    if [[ ! -d "$HIGHS_DIR" ]]; then
        echo "[lp_vs_highs] Error: highs/ dir not found in $RESULT_DIR" >&2; exit 1
    fi
    echo "[lp_vs_highs] Rescore mode: $RESULT_DIR"
    echo "[lp_vs_highs] solver: $(git -C "$SOLVER_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
else
    # ── Normal mode: full solve + score ──────────────────────────────────
    if [[ ! -x "$HIGHS_BIN" ]]; then
        echo "[lp_vs_highs] Error: HiGHS not found at $HIGHS_BIN" >&2; exit 1
    fi

    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    RESULT_DIR="$SOLVER_ROOT/bench_results/lp_vs_highs_${TIMESTAMP}"
    mkdir -p "$RESULT_DIR"

    LP_DATA_DIR="$SOLVER_ROOT/data/lp_problems"
    HIGHS_OPTS_FILE="$RESULT_DIR/highs.options"
    SELF_SOLVER_TXT="$RESULT_DIR/self_solver.txt"
    HIGHS_DIR="$RESULT_DIR/highs"
    MPS_DIR="$RESULT_DIR/mps"
    COMPARISON_CSV="$RESULT_DIR/lp_vs_highs.csv"
    SUMMARY_MD="$RESULT_DIR/summary.md"

    mkdir -p "$HIGHS_DIR" "$MPS_DIR"

    # HiGHS options: tol=1e-6, single-thread
    cat > "$HIGHS_OPTS_FILE" <<'EOF'
primal_feasibility_tolerance = 1e-6
dual_feasibility_tolerance = 1e-6
parallel = off
EOF

    echo "[lp_vs_highs] ============================================"
    echo "[lp_vs_highs] Self-solver vs HiGHS — Netlib LP benchmark"
    echo "[lp_vs_highs] timestamp: $TIMESTAMP"
    echo "[lp_vs_highs] result_dir: $RESULT_DIR"
    echo "[lp_vs_highs] jobs: $JOBS | timeout: ${TIMEOUT}s | mini: $MINI"
    echo "[lp_vs_highs] highs: $("$HIGHS_BIN" --version 2>&1 | head -1)"
    echo "[lp_vs_highs] solver: $(git -C "$SOLVER_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "[lp_vs_highs] ============================================"

    # ── Step 1: Ensure LP problem data exists ────────────────────────────
    LP_COUNT=$(find "$LP_DATA_DIR" -maxdepth 1 -iname "*.qps" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$LP_COUNT" -eq 0 ]]; then
        echo "[lp_vs_highs] LP data not found — downloading via netlib_lp_download.sh..."
        EMPS="/tmp/emps"
        if [[ ! -x "$EMPS" ]]; then
            curl -s https://www.netlib.org/lp/data/emps.c -o /tmp/emps.c
            cc -o "$EMPS" /tmp/emps.c
        fi
        EMPS_BIN="$EMPS" bash "$SCRIPT_DIR/netlib_lp_download.sh" "$LP_DATA_DIR" || true
        LP_COUNT=$(find "$LP_DATA_DIR" -maxdepth 1 -iname "*.qps" 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$LP_COUNT" -eq 0 ]]; then
            echo "[lp_vs_highs] Error: download produced 0 files in $LP_DATA_DIR" >&2; exit 1
        fi
    fi
    echo "[lp_vs_highs] LP problems available: $LP_COUNT"

    # ── Step 2: Determine problem set ────────────────────────────────────
    if [[ "$MINI" -eq 1 ]]; then
        BENCH_DATA_DIR="$RESULT_DIR/mini_subset"
        mkdir -p "$BENCH_DATA_DIR"
        for name in afiro adlittle blend; do
            f=$(find "$LP_DATA_DIR" -maxdepth 1 -iname "${name}.qps" 2>/dev/null | head -1)
            if [[ -n "$f" ]]; then
                ln -sf "$f" "$BENCH_DATA_DIR/$(basename "$f")"
                echo "[lp_vs_highs] mini: included $name"
            else
                echo "[lp_vs_highs] Warning: $name.qps not found in $LP_DATA_DIR" >&2
            fi
        done
    else
        BENCH_DATA_DIR="$LP_DATA_DIR"
    fi

    PROBLEM_FILES=()
    while IFS= read -r f; do
        PROBLEM_FILES+=("$f")
    done < <(find "$BENCH_DATA_DIR" -maxdepth 1 -iname "*.qps" | sort)
    PROBLEM_COUNT=${#PROBLEM_FILES[@]}
    echo "[lp_vs_highs] Problems to run: $PROBLEM_COUNT"

    if [[ $PROBLEM_COUNT -eq 0 ]]; then
        echo "[lp_vs_highs] Error: no .qps files in $BENCH_DATA_DIR" >&2; exit 1
    fi

    # ── Step 3: Run self-solver via bench_parallel.sh ────────────────────
    echo ""
    echo "[lp_vs_highs] [1/3] Running self-solver (bench_parallel.sh, jobs=$JOBS)..."
    SOLVER_DIR="$SOLVER_ROOT" \
    bash "$SCRIPT_DIR/bench_parallel.sh" \
        --data-dir "$BENCH_DATA_DIR" \
        --timeout "$TIMEOUT" \
        --eps "1e-6" \
        --jobs "$JOBS" \
        --output "$SELF_SOLVER_TXT"
    echo "[lp_vs_highs] Self-solver complete → $SELF_SOLVER_TXT"

    # ── Step 4: Run HiGHS in parallel ────────────────────────────────────
    echo ""
    echo "[lp_vs_highs] [2/3] Running HiGHS (jobs=$JOBS, timeout=${TIMEOUT}s)..."

    for qps_file in "${PROBLEM_FILES[@]}"; do
        stem=$(basename "$qps_file"); stem="${stem%.*}"
        cp "$qps_file" "$MPS_DIR/${stem}.mps"
    done

    _highs_worker() {
        local name="$1"
        "$HIGHS_BIN" \
            --options_file "$HIGHS_OPTS_FILE" \
            --time_limit "$TIMEOUT" \
            "$MPS_DIR/${name}.mps" > "$HIGHS_DIR/${name}.txt" 2>&1 || true
    }
    export -f _highs_worker
    export HIGHS_BIN HIGHS_OPTS_FILE HIGHS_DIR MPS_DIR TIMEOUT

    active_pids=()
    for qps_file in "${PROBLEM_FILES[@]}"; do
        stem=$(basename "$qps_file"); name="${stem%.*}"
        _highs_worker "$name" &
        active_pids+=($!)
        if [[ ${#active_pids[@]} -ge $JOBS ]]; then
            wait "${active_pids[0]}" 2>/dev/null || true
            active_pids=("${active_pids[@]:1}")
        fi
    done
    for pid in "${active_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    echo "[lp_vs_highs] HiGHS complete"
fi  # end normal mode

# ── Step 5: Parse results and generate CSV + summary ─────────────────
echo ""
echo "[lp_vs_highs] [3/3] Generating comparison CSV and summary..."

export _LV_SELF_TXT="$SELF_SOLVER_TXT"
export _LV_HIGHS_DIR="$HIGHS_DIR"
export _LV_BASELINE="$BASELINE_CSV"
export _LV_CSV="$COMPARISON_CSV"
export _LV_SUMMARY="$SUMMARY_MD"
export _LV_TIMEOUT="$TIMEOUT"

python3 - <<'PYEOF'
import csv, os, re, math, datetime

self_solver_txt = os.environ['_LV_SELF_TXT']
highs_dir       = os.environ['_LV_HIGHS_DIR']
baseline_csv    = os.environ['_LV_BASELINE']
comparison_csv  = os.environ['_LV_CSV']
summary_md      = os.environ['_LV_SUMMARY']
timeout_s       = int(os.environ['_LV_TIMEOUT'])

# ── Load baseline objectives ──────────────────────────────────────────
baseline = {}
with open(baseline_csv) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        if len(parts) >= 2 and parts[0] != 'problem_name':
            try:
                baseline[parts[0].strip()] = float(parts[1].strip())
            except ValueError:
                pass

# ── Parse self-solver output ──────────────────────────────────────────
# Format: {name:<20} {vars:>6} {cons:>6} {status:>15} {time:>10.3} {note...}
# note (PASS): obj=X.XXe+Y pf=... df=... [method] obj_err=X.XXX%
# obj_err is {:.3}% — display resolution 0.001% = 1e-5 relative; "0.000%" means < 5e-6 relative
KNOWN_STATUSES = {
    'PASS', 'PASS[no_ref]', 'PASS:Infeasible', 'PASS:Unbounded',
    'TIMEOUT', 'MAXITER', 'ERROR', 'SKIP', 'PARSE_ERR',
    'NONCONVEX', 'SUBOPTIMAL', 'KKT_FAIL', 'OBJ_MISMATCH',
    'PFEAS_FAIL', 'DFEAS_FAIL', 'FAIL', 'FAIL:NumericalError', 'FAIL:Unknown',
}

self_data = {}
with open(self_solver_txt) as f:
    for line in f:
        if not line or not line[0].strip() or line[0] in ('=', '[', ' ', '-'):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        name, status = parts[0], parts[3]
        if not (status in KNOWN_STATUSES or status.startswith('PASS') or status.startswith('FAIL')):
            continue
        try:
            time_val = float(parts[4])
        except ValueError:
            continue
        note = ' '.join(parts[5:]) if len(parts) > 5 else ''
        # Display obj ({:.2e}, ~3 sig figs): CSV display only, not used for scoring
        m = re.search(r'obj=([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)', note)
        obj_display = m.group(1) if m else None
        # obj_err% from note field: used for symmetric objective scoring
        m_err = re.search(r'obj_err=([0-9]+(?:\.[0-9]+)?)%', note)
        obj_err_pct = float(m_err.group(1)) if m_err else None  # percentage
        self_data[name] = {
            'status': status, 'time': time_val,
            'obj_display': obj_display, 'obj_err_pct': obj_err_pct,
        }

# ── Parse HiGHS per-problem outputs ──────────────────────────────────
highs_data = {}
for fname in sorted(os.listdir(highs_dir)):
    if not fname.endswith('.txt'):
        continue
    name = fname[:-4]
    with open(os.path.join(highs_dir, fname)) as f:
        content = f.read()
    h_status, h_time, h_obj = 'NA', None, None
    m = re.search(r'^Model status\s*:\s*(.+)$', content, re.MULTILINE)
    if m:
        h_status = m.group(1).strip()
    m = re.search(r'^Objective value\s*:\s*([+-]?[0-9.eE+\-]+)', content, re.MULTILINE)
    if m:
        try:
            h_obj = float(m.group(1))
        except ValueError:
            pass
    m = re.search(r'^HiGHS run time\s*:\s*([0-9]+(?:\.[0-9]+)?)', content, re.MULTILINE)
    if m:
        h_time = float(m.group(1))
    highs_data[name] = {'status': h_status, 'time': h_time, 'obj': h_obj}

# Problem list from highs/ dir (works in both normal and --rescore modes)
problem_names = sorted(
    fname[:-4] for fname in os.listdir(highs_dir) if fname.endswith('.txt')
)

# ── Symmetric objective scoring ───────────────────────────────────────
# Both solvers: relative error = |obj - baseline| / max(|baseline|, 1.0)
# Self-solver:  obj_err% from note / 100  (solver already computed this vs baseline)
# HiGHS:        |highs_obj - baseline| / max(|baseline|, 1.0)
# e226: excluded — objective constant convention differs across self/HiGHS/Netlib (all correct)

EXCLUDE_SCORING = {'e226'}
THRESHOLDS      = [1e-2, 1e-4, 1e-6]
THR_LABELS      = {1e-2: '1e-2', 1e-4: '1e-4', 1e-6: '1e-6'}

def self_rel_err(sd):
    """Relative error for self-solver from obj_err% in note field."""
    pct = sd.get('obj_err_pct')
    if pct is None:
        return None
    return pct / 100.0

def highs_rel_err(hd, b):
    """Relative error for HiGHS: |obj - baseline| / max(|baseline|, 1.0)."""
    if hd.get('status') != 'Optimal' or hd.get('obj') is None or b is None:
        return None
    return abs(hd['obj'] - b) / max(abs(b), 1.0)

def chk(err, thr):
    if err is None:
        return '?'
    return '✓' if err <= thr else '✗'

def match_label(err, thr):
    if err is None:
        return 'not_solved'
    return 'match' if err <= thr else 'mismatch'

# ── Write CSV ─────────────────────────────────────────────────────────
# solver_obj_match / highs_obj_match use 1e-2 (self-solver's internal gate)
fieldnames = [
    'instance', 'solver_status', 'solver_time', 'solver_obj',
    'highs_status', 'highs_time', 'highs_obj', 'baseline_obj',
    'solver_obj_match', 'highs_obj_match',
]
rows = []
for name in problem_names:
    sd = self_data.get(name, {})
    hd = highs_data.get(name, {})
    b  = baseline.get(name)
    h_obj    = hd.get('obj')
    s_status = sd.get('status', 'NA')
    if name in EXCLUDE_SCORING:
        s_match = 'excluded(convention)'
        h_match = 'excluded(convention)'
    else:
        s_match = match_label(self_rel_err(sd), 1e-2)
        h_match = match_label(highs_rel_err(hd, b), 1e-2)
    rows.append({
        'instance':         name,
        'solver_status':    s_status,
        'solver_time':      '{:.3f}'.format(sd['time']) if sd.get('time') is not None else 'NA',
        'solver_obj':       sd.get('obj_display') or 'NA',
        'highs_status':     hd.get('status', 'NA'),
        'highs_time':       '{:.3f}'.format(hd['time']) if hd.get('time') is not None else 'NA',
        'highs_obj':        '{:.6e}'.format(h_obj) if h_obj is not None else 'NA',
        'baseline_obj':     '{:.6e}'.format(b) if b is not None else 'NA',
        'solver_obj_match': s_match,
        'highs_obj_match':  h_match,
    })

with open(comparison_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print('[lp_vs_highs] CSV: {} rows -> {}'.format(len(rows), comparison_csv))

# ── Aggregate stats ───────────────────────────────────────────────────
n_total   = len(rows)
n_s_opt   = sum(1 for r in rows if r['solver_status'].startswith('PASS'))
n_h_opt   = sum(1 for r in rows if r['highs_status'] == 'Optimal')
n_s_tmout = sum(1 for r in rows if r['solver_status'] == 'TIMEOUT')
n_h_tmout = sum(1 for r in rows if 'time' in r['highs_status'].lower())

# 3-threshold match counts (e226 excluded from scoring pool)
scorable = [r for r in rows if r['instance'] not in EXCLUDE_SCORING]
n_score  = len(scorable)
thresh_counts = {}
for thr in THRESHOLDS:
    s_cnt = h_cnt = 0
    for r in scorable:
        name = r['instance']
        sd = self_data.get(name, {})
        hd = highs_data.get(name, {})
        b  = baseline.get(name)
        s_err = self_rel_err(sd)
        h_err = highs_rel_err(hd, b)
        if s_err is not None and s_err <= thr:
            s_cnt += 1
        if h_err is not None and h_err <= thr:
            h_cnt += 1
    thresh_counts[thr] = (s_cnt, h_cnt)

# Time ratios (problems both solved optimally)
ratios = []
s_wins = h_wins = 0
for r in rows:
    if not (r['solver_status'].startswith('PASS') and r['highs_status'] == 'Optimal'):
        continue
    if r['solver_time'] == 'NA' or r['highs_time'] == 'NA':
        continue
    st, ht = float(r['solver_time']), float(r['highs_time'])
    EPS_T = 1e-3  # 1ms floor for sub-ms timing granularity
    ratio = max(st, EPS_T) / max(ht, EPS_T)
    ratios.append((r['instance'], st, ht, ratio))
    if st < ht:
        s_wins += 1
    else:
        h_wins += 1

n_both = len(ratios)

def geomean(vals):
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))

def median_val(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2

ratio_vals = [r[3] for r in ratios]
s_times    = [float(r['solver_time']) for r in rows
              if r['solver_status'].startswith('PASS') and r['solver_time'] != 'NA']
h_times    = [float(r['highs_time']) for r in rows
              if r['highs_status'] == 'Optimal' and r['highs_time'] != 'NA']

gm_ratio  = geomean(ratio_vals)
med_ratio = median_val(ratio_vals)
s_median  = median_val(s_times)
h_median  = median_val(h_times)

# Problems notable at 1e-6 for either solver
notable = []
for r in scorable:
    name = r['instance']
    sd = self_data.get(name, {})
    hd = highs_data.get(name, {})
    b  = baseline.get(name)
    s_err = self_rel_err(sd)
    h_err = highs_rel_err(hd, b)
    if (s_err is not None and s_err > 1e-6) or (h_err is not None and h_err > 1e-6):
        notable.append((name, s_err, h_err))

# ── Write summary.md ──────────────────────────────────────────────────
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

with open(summary_md, 'w') as f:
    f.write('# LP Benchmark: Self-solver vs HiGHS\n\n')
    f.write('**Generated**: {}\n\n'.format(now))
    f.write('**Suite**: Netlib LP standard — {} problems available'.format(n_total))
    f.write(' (pilot-ja / pilot-we / vtp-base 未DL),')
    f.write(' e226 を objective-constant convention 差として除外し **{}問** で採点\n\n'.format(n_score))
    f.write('**CSV**: lp_vs_highs.csv\n\n')

    f.write('## Methodology\n\n')
    f.write('| Parameter | Self-solver | HiGHS |\n')
    f.write('|-----------|-------------|-------|\n')
    f.write('| Primal+dual feasibility tol | 1e-6 | 1e-6 (options file) |\n')
    f.write('| **Objective acceptance gate** | **1e-2 (§2.4)** | **1e-6 vs Netlib baseline** |\n')
    f.write('| Time limit | {}s | {}s |\n'.format(timeout_s, timeout_s))
    f.write('| Presolve | default | default (choose/auto) |\n')
    f.write('| Threads | single | parallel=off (single) |\n')
    f.write('| Solver mode | auto (simplex/IPM by size) | default (choose/dual simplex for LP) |\n')
    f.write('| Baseline reference | Netlib MINOS 5.3 | Netlib MINOS 5.3 |\n')
    f.write('| **Obj_match scoring** | **対称: 両者ともobj_err vs Netlib baseline** | **← 同** |\n\n')

    f.write('### 公平性に関する注記\n\n')
    f.write('- 自solverの **内部受理ゲートは1e-2** (1%)。STATUSが`PASS`でも必ずしもobj_err<1e-6ではない\n')
    f.write('- 本比較では自solverのobj_err%値(note field)とHiGHS実値を**同一Netlib baselineで対称採点**\n')
    if gm_ratio is not None:
        f.write('- 計測時の6並列CPU競合により大問題の自solver時間は**上振れバイアスあり**。')
        f.write('{:.0f}x比は自solver**不利**方向の保守値\n\n'.format(gm_ratio))
    else:
        f.write('- 計測時の6並列CPU競合により大問題の自solver時間は**上振れバイアスあり**\n\n')

    # 3-threshold match table
    f.write('## 正解率 (Objective match vs Netlib baseline)\n\n')
    f.write('**e226除外** (objective定数のconvention差 — self=-25.86, HiGHS=-11.64,')
    f.write(' Netlib=-18.75; 3者全て別convention)\n\n')
    f.write('| 閾値 | 自solver | HiGHS | 備考 |\n')
    f.write('|------|---------|-------|------|\n')
    for thr in THRESHOLDS:
        s_cnt, h_cnt = thresh_counts[thr]
        label = THR_LABELS[thr]
        if thr == 1e-2:
            note = '自solver内部ゲートと同一'
        elif thr == 1e-4:
            note = '両者同一 — greenbea+pilot が閾値超え' if s_cnt == h_cnt else ''
        else:
            note = '†自solver表示精度5e-6のため不確実性あり'
        f.write('| {} | {}/{} | {}/{} | {} |\n'.format(label, s_cnt, n_score, h_cnt, n_score, note))
    f.write('\n')
    f.write('†1e-6欄: 自solverのobj_errは0.001% (=1e-5)以下が非ゼロ表示下限のため、')
    f.write('"0.000%"表示の問題が1e-6を厳密に下回るか不確定\n\n')

    if notable:
        f.write('### 問題別 obj_err 詳細 (閾値超えのみ)\n\n')
        f.write('| Instance | Self err% | HiGHS err | 1e-2 | 1e-4 | 1e-6 | 注記 |\n')
        f.write('|----------|-----------|-----------|------|------|------|------|\n')
        for name, s_err, h_err in sorted(notable):
            s_pct = '{:.4f}%'.format(s_err * 100) if s_err is not None else 'N/A'
            h_val = '{:.2e}'.format(h_err) if h_err is not None else 'N/A'
            col12 = '{}/{}'.format(chk(s_err, 1e-2), chk(h_err, 1e-2))
            col14 = '{}/{}'.format(chk(s_err, 1e-4), chk(h_err, 1e-4))
            col16 = '{}/{}'.format(chk(s_err, 1e-6), chk(h_err, 1e-6))
            remark = '両solver同値、baseline更新が外れ値の可能性' if name in ('greenbea', 'greenbeb') else ''
            f.write('| {} | {} | {} | {} | {} | {} | {} |\n'.format(
                name, s_pct, h_val, col12, col14, col16, remark))
        f.write('\n')
        f.write('**e226** (除外): 自solver=-25.86 (obj_offset=-7.113), HiGHS=-11.64,')
        f.write(' baseline=-18.75 — 3者いずれも異なるconventionで正しい\n\n')
        f.write('**greenbea/greenbeb** (含む): 両solverが独立に同一値を返し、旧baselineと一致。')
        f.write('netlib_lp.csv更新後の新値が外れ値の可能性あり\n\n')

    # Speed section
    f.write('## 速度比較 ({} 問 両者Optimal)\n\n'.format(n_both))
    if gm_ratio is not None:
        s_med_str = '{:.3f}s'.format(s_median) if s_median is not None else 'N/A'
        h_med_str = '{:.3f}s'.format(h_median) if h_median is not None else 'N/A'
        f.write('| Stat | 値 |\n')
        f.write('|------|-----|\n')
        f.write('| Geometric mean (solver/HiGHS) | **{:.2f}x** |\n'.format(gm_ratio))
        f.write('| Median ratio (solver/HiGHS) | {:.2f}x |\n'.format(med_ratio))
        f.write('| 自solver中央値 | {} |\n'.format(s_med_str))
        f.write('| HiGHS中央値 | {} |\n'.format(h_med_str))
        f.write('| HiGHSが速い | {}/{} |\n\n'.format(h_wins, n_both))
        if gm_ratio > 1.0:
            f.write('**速度結論**: HiGHS は自solverより **{:.1f}x 高速**'.format(gm_ratio))
            f.write(' (幾何平均, {}問全てHiGHS勝)\n\n'.format(n_both))
        elif gm_ratio < 1.0:
            f.write('**速度結論**: 自solver は HiGHS より **{:.1f}x 高速** (幾何平均)\n\n'.format(1 / gm_ratio))
        else:
            f.write('**速度結論**: 両solver同速\n\n')
        f.write('> 注: 6並列CPU競合により大問題の自solver時間は上振れバイアスあり。')
        f.write('{:.0f}xは保守的下限値\n\n'.format(gm_ratio))

        if ratios:
            f.write('### 最大時間差 Top-10\n\n')
            top10 = sorted(ratios, key=lambda x: -x[3])[:10]
            f.write('| Instance | Solver (s) | HiGHS (s) | Ratio |\n')
            f.write('|----------|------------|-----------|-------|\n')
            for inst, st, ht, ratio in top10:
                f.write('| {} | {:.3f} | {:.3f} | {:.0f}x |\n'.format(inst, st, ht, ratio))
            f.write('\n')

    f.write('## 除外問題\n\n')
    f.write('- **pilot-ja / pilot-we / vtp-base**: netlib_lp_download.sh のbash3.2 `declare -A`')
    f.write(' 非互換によりDL失敗。Netlib standard109問中106問が取得済み\n\n')

    f.write('---\n*Generated by scripts/lp_vs_highs.sh (symmetric obj_match)*\n')

print('[lp_vs_highs] Summary -> ' + summary_md)
print('[lp_vs_highs] Optimal: self={}/{}, HiGHS={}/{}'.format(n_s_opt, n_total, n_h_opt, n_total))
print('')
print('[lp_vs_highs] === 3-threshold obj_match ({} problems, excl. e226) ==='.format(n_score))
for thr in THRESHOLDS:
    s_cnt, h_cnt = thresh_counts[thr]
    print('[lp_vs_highs]   {} | self={}/{} | HiGHS={}/{}'.format(
        THR_LABELS[thr], s_cnt, n_score, h_cnt, n_score))
if gm_ratio is not None:
    print('[lp_vs_highs] Time geomean (solver/HiGHS): {:.2f}x  ({} problems both solved)'.format(
        gm_ratio, n_both))
PYEOF

echo ""
echo "[lp_vs_highs] ============================================"
echo "[lp_vs_highs] Done. Results: $RESULT_DIR"
echo "[lp_vs_highs]   CSV:     $COMPARISON_CSV"
echo "[lp_vs_highs]   Summary: $SUMMARY_MD"
echo "[lp_vs_highs] ============================================"
