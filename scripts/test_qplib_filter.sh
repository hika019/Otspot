#!/bin/bash
# Smoke test for scripts/qplib_filter.sh — feeds a synthetic instances.html
# covering convex/non-convex × continuous/binary/mixed × Q/D/L/C objectives,
# then checks bucket counts and membership.

set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

OUT="$WORK/out"
CACHE="$OUT/cache"
mkdir -p "$CACHE"

cat > "$CACHE/instances.html" <<'HTML'
<HTML><BODY><TABLE id="instancelisting">
<THEAD>
<TR><TH>Instance</TH><TH>Cvx</TH><TH>O</TH><TH>density</TH><TH>probev</TH>
<TH>V</TH><TH>nvars</TH><TH>nbin</TH><TH>nint</TH><TH>C</TH><TH>ncons</TH>
<TH>nquad</TH><TH>nz</TH></TR>
</THEAD>
<TBODY>
<TR><TD align="left"><A href=QPLIB_0001.html>0001</A></TD>
 <TD align="center">&#10004;</TD><TD align="center">Q</TD>
 <TD>100.0</TD><TD>0.0</TD><TD align="center">C</TD>
 <TD>50</TD><TD>&nbsp;</TD><TD>&nbsp;</TD>
 <TD align="center">L</TD><TD>10</TD><TD>0</TD><TD>100</TD></TR>
<TR><TD align="left"><A href=QPLIB_0002.html>0002</A></TD>
 <TD align="center">-</TD><TD align="center">Q</TD>
 <TD>100.0</TD><TD>48.0</TD><TD align="center">C</TD>
 <TD>500</TD><TD>&nbsp;</TD><TD>&nbsp;</TD>
 <TD align="center">L</TD><TD>5</TD><TD>0</TD><TD>200</TD></TR>
<TR><TD align="left"><A href=QPLIB_0003.html>0003</A></TD>
 <TD align="center">&#10004;</TD><TD align="center">D</TD>
 <TD>1.0</TD><TD>0.0</TD><TD align="center">C</TD>
 <TD>2000</TD><TD>&nbsp;</TD><TD>&nbsp;</TD>
 <TD align="center">L</TD><TD>1500</TD><TD>0</TD><TD>5000</TD></TR>
<TR><TD align="left"><A href=QPLIB_0004.html>0004</A></TD>
 <TD align="center">&#10004;</TD><TD align="center">C</TD>
 <TD>0.0</TD><TD>0.0</TD><TD align="center">B</TD>
 <TD>150</TD><TD>150</TD><TD>&nbsp;</TD>
 <TD align="center">L</TD><TD>5</TD><TD>0</TD><TD>50</TD></TR>
<TR><TD align="left"><A href=QPLIB_0005.html>0005</A></TD>
 <TD align="center">-</TD><TD align="center">L</TD>
 <TD>0.0</TD><TD>0.0</TD><TD align="center">C</TD>
 <TD>1000</TD><TD>&nbsp;</TD><TD>&nbsp;</TD>
 <TD align="center">Q</TD><TD>800</TD><TD>10</TD><TD>3000</TD></TR>
<TR><TD align="left"><A href=QPLIB_0006.html>0006</A></TD>
 <TD align="center">-</TD><TD align="center">Q</TD>
 <TD>50.0</TD><TD>50.0</TD><TD align="center">M</TD>
 <TD>80</TD><TD>40</TD><TD>&nbsp;</TD>
 <TD align="center">L</TD><TD>20</TD><TD>0</TD><TD>80</TD></TR>
<TR><TD align="left"><A href=QPLIB_0007.html>0007</A></TD>
 <TD align="center">&#10004;</TD><TD align="center">Q</TD>
 <TD>10.0</TD><TD>0.0</TD><TD align="center">C</TD>
 <TD>100</TD><TD>&nbsp;</TD><TD>&nbsp;</TD>
 <TD align="center">L</TD><TD>5</TD><TD>0</TD><TD>50</TD></TR>
<TR><TD align="left"><A href=QPLIB_0008.html>0008</A></TD>
 <TD align="center">-</TD><TD align="center">Q</TD>
 <TD>80.0</TD><TD>10.0</TD><TD align="center">C</TD>
 <TD>1500</TD><TD>&nbsp;</TD><TD>&nbsp;</TD>
 <TD align="center">B</TD><TD>0</TD><TD>0</TD><TD>3000</TD></TR>
</TBODY></TABLE></BODY></HTML>
HTML

# Run filter (small=100, medium=1000 by default).
bash scripts/qplib_filter.sh "$OUT" >/dev/null

fail=0
check() {
  local file="$1" expected="$2"
  local got
  got=$(wc -l < "$OUT/$file" | tr -d ' ')
  if [[ "$got" != "$expected" ]]; then
    echo "FAIL $file: expected $expected got $got" >&2
    fail=1
  fi
}

check_contains() {
  local file="$1" id="$2"
  if ! grep -qx "$id" "$OUT/$file"; then
    echo "FAIL $file: missing $id" >&2
    fail=1
  fi
}

check_not_contains() {
  local file="$1" id="$2"
  if grep -qx "$id" "$OUT/$file"; then
    echo "FAIL $file: unexpected $id" >&2
    fail=1
  fi
}

# Total rows.
check all.txt 8
check convex.txt 4
check nonconvex.txt 4

# Objective.
check obj_quadratic.txt 6   # Q×5 + D×1
check obj_linear.txt 1

# Variable.
check var_continuous.txt 6
check var_binary.txt 1
check var_mixed.txt 1
check var_integer.txt 0

# Constraint.
check con_linear.txt 6
check con_box_or_none.txt 1
check con_quadratic.txt 1

# Solver scope: O∈{Q,D}, V=C, C∈{L,B,N}.
#   0001 convex small, 0003 convex large, 0007 convex small/medium boundary (n=100),
#   0002 nonconvex medium, 0008 nonconvex large.
check in_scope_convex.txt 3
check in_scope_nonconvex.txt 2
check_contains in_scope_convex.txt 0001
check_contains in_scope_convex.txt 0003
check_contains in_scope_convex.txt 0007
check_contains in_scope_nonconvex.txt 0002
check_contains in_scope_nonconvex.txt 0008
check_not_contains in_scope_convex.txt 0004   # binary
check_not_contains in_scope_nonconvex.txt 0005 # LP
check_not_contains in_scope_nonconvex.txt 0006 # mixed-integer

# Size buckets (small ≤100, medium ≤1000, large >1000).
check_contains size_small.txt 0001     # n=50
check_contains size_small.txt 0007     # n=100 (boundary)
check_contains size_medium.txt 0002    # n=500
check_contains size_large.txt 0003     # n=2000
check_contains size_large.txt 0008     # n=1500

# Scope×size splits.
check in_scope_convex_small.txt 2      # 0001, 0007
check in_scope_convex_medium.txt 0
check in_scope_convex_large.txt 1      # 0003
check in_scope_nonconvex_medium.txt 1  # 0002
check in_scope_nonconvex_large.txt 1   # 0008

# Custom thresholds via env (small=10, medium=200).
SIZE_SMALL=10 SIZE_MEDIUM=200 bash scripts/qplib_filter.sh "$OUT" >/dev/null
got=$(wc -l < "$OUT/size_small.txt" | tr -d ' ')
[[ "$got" == "0" ]] || { echo "FAIL custom-thresholds size_small: $got"; fail=1; }

if (( fail )); then
  echo "[test] FAIL"
  exit 1
fi

echo "[test] PASS (qplib filter smoke)"
