#!/bin/bash
# milp_vs_highs.sh — MILP benchmark: otspot vs HiGHS comparison.
#
# 公平性 (Fairness):
#   Tolerance:  両者 eps=1e-6 (MIP gap + integer feas / HiGHS mip_rel_gap + tol)
#   Time limit: 両者 timeout (既定 100s)
#   Threads:    両者 single-thread
#   Obj match:  |otspot_obj - highs_obj| / max(1, |highs_obj|) <= 1e-6
#               (HiGHS の最適値を基準とした相対誤差)
#
# Usage:
#   bash scripts/milp_vs_highs.sh [--data-dir DIR] [--timeout SEC] [--jobs N]
#
# 出力: bench_results/milp_vs_highs_<timestamp>/
#   results.csv   — instance ごとの status/obj/time
#   summary.md    — markdown 比較表 + 集計

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HIGHS_BIN="${HIGHS_BIN:-/opt/homebrew/bin/highs}"
MILP_SOLVE="$REPO_ROOT/target/release/milp_solve"

DATA_DIR="$REPO_ROOT/data/miplib_small"
TIMEOUT=100
JOBS=6
EPS=1e-6

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --timeout)  TIMEOUT="$2"; shift 2 ;;
    --jobs)     JOBS="$2"; shift 2 ;;
    --eps)      EPS="$2"; shift 2 ;;
    *) echo "[milp_vs_highs] unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -x "$HIGHS_BIN" ]]; then
  echo "[milp_vs_highs] HiGHS not found at $HIGHS_BIN" >&2; exit 1
fi
if [[ ! -x "$MILP_SOLVE" ]]; then
  echo "[milp_vs_highs] building milp_solve..."
  (cd "$REPO_ROOT" && cargo build --release --bin milp_solve)
fi

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULT_DIR="$REPO_ROOT/bench_results/milp_vs_highs_${TIMESTAMP}"
mkdir -p "$RESULT_DIR/otspot" "$RESULT_DIR/highs"

# bash 3.2 互換: mapfile を使わず while-read で配列化。
FILES=()
while IFS= read -r f; do
  FILES+=("$f")
done < <(find "$DATA_DIR" -maxdepth 1 -iname "*.mps" | sort)
echo "[milp_vs_highs] data=$DATA_DIR  problems=${#FILES[@]}  timeout=${TIMEOUT}s  jobs=$JOBS"
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "[milp_vs_highs] no .mps in $DATA_DIR (run miplib_small_download.sh first)" >&2; exit 1
fi

# HiGHS options: matched tolerances, single-thread.
HIGHS_OPTS="$RESULT_DIR/highs.options"
cat > "$HIGHS_OPTS" <<EOF
mip_rel_gap = $EPS
mip_feasibility_tolerance = $EPS
primal_feasibility_tolerance = $EPS
parallel = off
threads = 1
EOF

run_one() {
  local f="$1" name
  name=$(basename "$f" .mps)
  "$MILP_SOLVE" "$f" --timeout "$TIMEOUT" --eps "$EPS" > "$RESULT_DIR/otspot/$name.txt" 2>&1 || true
  "$HIGHS_BIN" --options_file "$HIGHS_OPTS" --time_limit "$TIMEOUT" "$f" \
      > "$RESULT_DIR/highs/$name.txt" 2>&1 || true
}
export -f run_one
export MILP_SOLVE HIGHS_BIN HIGHS_OPTS RESULT_DIR TIMEOUT EPS

# Worker pool: bash 3.2 compatible (macOS default; wait -n requires 4.3+).
# Mirrors bench_parallel.sh pattern: N workers pull from a shared counter via flock.
_POOL_COUNTER="$RESULT_DIR/.pool_counter"
_POOL_LOCK="$RESULT_DIR/.pool_lock"
_POOL_TOTAL=${#FILES[@]}
echo "0" > "$_POOL_COUNTER"
: > "$_POOL_LOCK"

_pool_worker() {
  local wid="$1"
  while true; do
    local idx
    idx=$(
      (
        flock -x 9
        n=$(cat "$_POOL_COUNTER")
        echo $(( n + 1 )) > "$_POOL_COUNTER"
        echo "$n"
      ) 9>"$_POOL_LOCK"
    ) || break
    [[ "$idx" =~ ^[0-9]+$ ]] || break
    [[ $idx -ge $_POOL_TOTAL ]] && break
    run_one "${FILES[$idx]}"
  done
}

WORKER_PIDS=()
for _w in $(seq 1 "$JOBS"); do
  _pool_worker "$_w" &
  WORKER_PIDS+=($!)
done
for _pid in "${WORKER_PIDS[@]}"; do wait "$_pid" 2>/dev/null || true; done
echo "[milp_vs_highs] solves complete; scoring..."

export _MV_RESULT="$RESULT_DIR" _MV_TIMEOUT="$TIMEOUT" _MV_EPS="$EPS" _MV_DATA="$DATA_DIR"
python3 - <<'PYEOF'
import os, re, math, glob, datetime

result_dir = os.environ['_MV_RESULT']
timeout_s  = float(os.environ['_MV_TIMEOUT'])
eps        = float(os.environ['_MV_EPS'])

def parse_otspot(path):
    d = {'status': 'NA', 'obj': None, 'time': None, 'nodes': None, 'n_int': None}
    if not os.path.exists(path):
        return d
    for line in open(path):
        m = re.match(r'(\w+):\s*(.+)', line.strip())
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if k == 'status':
            d['status'] = v
        elif k == 'objective':
            try: d['obj'] = float(v)
            except ValueError: d['obj'] = None
        elif k == 'wall_ms':
            try: d['time'] = float(v) / 1000.0
            except ValueError: pass
        elif k == 'nodes':
            try: d['nodes'] = int(v)
            except ValueError: pass
        elif k == 'n_int':
            try: d['n_int'] = int(v)
            except ValueError: pass
    return d

def parse_highs(path):
    d = {'status': 'NA', 'obj': None, 'time': None}
    if not os.path.exists(path):
        return d
    content = open(path).read()
    m = re.search(r'^\s*Status\s+(.+)$', content, re.MULTILINE)
    if m: d['status'] = m.group(1).strip()
    m = re.search(r'^\s*([+-]?[0-9.]+(?:[eE][+-]?[0-9]+)?)\s*\(objective\)', content, re.MULTILINE)
    if m:
        try: d['obj'] = float(m.group(1))
        except ValueError: pass
    m = re.search(r'^\s*Timing\s+([0-9.]+)', content, re.MULTILINE)
    if not m:
        m = re.search(r'HiGHS run time\s*:\s*([0-9.]+)', content)
    if m: d['time'] = float(m.group(1))
    return d

names = sorted(os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob(os.path.join(result_dir, 'otspot', '*.txt')))

rows = []
for name in names:
    o = parse_otspot(os.path.join(result_dir, 'otspot', f'{name}.txt'))
    h = parse_highs(os.path.join(result_dir, 'highs', f'{name}.txt'))
    # obj comparison relative to HiGHS optimum (these MPS are all minimization).
    #   match    : |rel| <= 1e-6 (agrees with HiGHS optimum)
    #   close    : |rel| <= 1e-4
    #   worse    : otspot obj larger than HiGHS — expected when otspot did not converge
    #   INVALID  : otspot obj smaller than HiGHS proven optimum → infeasible/wrong incumbent (BUG)
    #   MISMATCH : otspot Optimal but obj differs > 1e-4 (real concern: bug or convention)
    match = '?'
    if o['obj'] is not None and h['obj'] is not None:
        signed = (o['obj'] - h['obj']) / max(1.0, abs(h['obj']))
        rel = abs(signed)
        if signed < -1e-6 and h['status'] == 'Optimal':
            match = 'INVALID'
        elif rel <= 1e-6:
            match = 'match'
        elif rel <= 1e-4:
            match = 'close'
        elif o['status'] == 'Optimal':
            match = 'MISMATCH'
        else:
            match = 'worse'  # otspot not converged; larger obj is expected, not a bug
    rows.append({'name': name, 'o': o, 'h': h, 'match': match})

# CSV
csv_path = os.path.join(result_dir, 'results.csv')
with open(csv_path, 'w') as f:
    f.write('instance,n_int,otspot_status,otspot_obj,otspot_time,otspot_nodes,'
            'highs_status,highs_obj,highs_time,obj_match\n')
    for r in rows:
        o, h = r['o'], r['h']
        f.write(','.join(str(x) for x in [
            r['name'], o['n_int'], o['status'],
            f"{o['obj']:.6f}" if o['obj'] is not None else 'NA',
            f"{o['time']:.3f}" if o['time'] is not None else 'NA',
            o['nodes'] if o['nodes'] is not None else 'NA',
            h['status'],
            f"{h['obj']:.6f}" if h['obj'] is not None else 'NA',
            f"{h['time']:.3f}" if h['time'] is not None else 'NA',
            r['match'],
        ]) + '\n')

# aggregate
def otspot_solved(s): return s in ('Optimal',)
def highs_solved(s):  return s == 'Optimal'
n = len(rows)
n_o_opt = sum(1 for r in rows if otspot_solved(r['o']['status']))
n_h_opt = sum(1 for r in rows if highs_solved(r['h']['status']))
n_mismatch = sum(1 for r in rows if r['match'] == 'MISMATCH')
n_invalid  = sum(1 for r in rows if r['match'] == 'INVALID')
invalids   = [r['name'] for r in rows if r['match'] == 'INVALID']
mismatches = [r['name'] for r in rows if r['match'] == 'MISMATCH']
# speed on both-Optimal
ratios = []
for r in rows:
    if otspot_solved(r['o']['status']) and highs_solved(r['h']['status']) \
       and r['o']['time'] is not None and r['h']['time'] is not None:
        st, ht = max(r['o']['time'], 1e-3), max(r['h']['time'], 1e-3)
        ratios.append(st / ht)
gm = math.exp(sum(math.log(x) for x in ratios)/len(ratios)) if ratios else None

md = os.path.join(result_dir, 'summary.md')
with open(md, 'w') as f:
    f.write('# MILP Benchmark: otspot vs HiGHS\n\n')
    f.write(f"**Generated**: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
    f.write(f"**Set**: MIPLIB 2017 small subset — {n} problems | "
            f"timeout {timeout_s:.0f}s | eps {eps:g} | single-thread\n\n")
    f.write(f"**Solved (Optimal)**: otspot {n_o_opt}/{n} | HiGHS {n_h_opt}/{n}\n\n")
    f.write(f"**INVALID (otspot obj better than HiGHS optimum = wrong incumbent / BUG)**: "
            f"{n_invalid}{(' — ' + ', '.join(invalids)) if invalids else ''}\n\n")
    f.write(f"**MISMATCH (otspot Optimal but obj differs >1e-4)**: "
            f"{n_mismatch}{(' — ' + ', '.join(mismatches)) if mismatches else ''}\n\n")
    if gm is not None:
        f.write(f"**Speed (both Optimal, {len(ratios)} probs)**: "
                f"otspot/HiGHS geomean = {gm:.1f}x\n\n")
    f.write('| instance | n_int | otspot status | otspot obj | otspot t(s) | nodes | '
            'HiGHS status | HiGHS obj | HiGHS t(s) | obj |\n')
    f.write('|---|---|---|---|---|---|---|---|---|---|\n')
    for r in rows:
        o, h = r['o'], r['h']
        f.write('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n'.format(
            r['name'], o['n_int'] if o['n_int'] is not None else '?',
            o['status'],
            f"{o['obj']:.4g}" if o['obj'] is not None else 'NA',
            f"{o['time']:.2f}" if o['time'] is not None else 'NA',
            o['nodes'] if o['nodes'] is not None else 'NA',
            h['status'],
            f"{h['obj']:.4g}" if h['obj'] is not None else 'NA',
            f"{h['time']:.2f}" if h['time'] is not None else 'NA',
            r['match'],
        ))
    f.write('\n*obj 列: match (rel≤1e-6) / close (≤1e-4) / MISMATCH / ? (片方未解)*\n')

print(f"[milp_vs_highs] CSV     -> {csv_path}")
print(f"[milp_vs_highs] summary -> {md}")
print(f"[milp_vs_highs] Optimal: otspot {n_o_opt}/{n}, HiGHS {n_h_opt}/{n}; "
      f"INVALID={n_invalid}, MISMATCH={n_mismatch}")
if gm is not None:
    print(f"[milp_vs_highs] speed geomean otspot/HiGHS = {gm:.1f}x ({len(ratios)} both-Optimal)")
# echo full table to stdout for the caller
print('\n' + open(md).read())
PYEOF

echo "[milp_vs_highs] done -> $RESULT_DIR"
