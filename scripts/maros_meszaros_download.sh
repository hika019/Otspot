#!/bin/bash
# Maros-Meszaros QP test set (138 problems) downloader.
#
# Source: https://github.com/YimingYAN/QP-Test-Problems (MAT_Files/)
# .mat を取得 -> scipy.io でロード -> scripts/qp_to_qps.py の write_qps() で .QPS 化。
#
# Usage:
#   bash scripts/maros_meszaros_download.sh [OUT_DIR]
# OUT_DIR 省略時は data/maros_meszaros
#
# 依存: python3 + scipy + numpy (qp_to_qps.py が import)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="${1:-data/maros_meszaros}"
# OUT_DIR が relative なら REPO_ROOT 基準で解決
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$REPO_ROOT/$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

MAT_CACHE="${MAROS_MAT_CACHE:-/tmp/maros_meszaros_mat}"
mkdir -p "$MAT_CACHE"

BASE_URL="https://raw.githubusercontent.com/YimingYAN/QP-Test-Problems/master/MAT_Files"

# 138 problem 名 (PROBLEM_NAMES from scripts/run_maros_all.py、fact-based)
NAMES=(
  AUG2D AUG2DC AUG2DCQP AUG2DQP
  AUG3D AUG3DC AUG3DCQP AUG3DQP
  BOYD1 BOYD2
  CONT-050 CONT-100 CONT-101 CONT-200 CONT-201 CONT-300
  CVXQP1_L CVXQP1_M CVXQP1_S
  CVXQP2_L CVXQP2_M CVXQP2_S
  CVXQP3_L CVXQP3_M CVXQP3_S
  DPKLO1 DTOC3
  DUAL1 DUAL2 DUAL3 DUAL4
  DUALC1 DUALC2 DUALC5 DUALC8
  EXDATA
  GENHS28
  GOULDQP2 GOULDQP3
  HS118 HS21 HS268 HS35 HS35MOD HS51 HS52 HS53 HS76
  HUES-MOD HUESTIS
  KSIP
  LASER
  LISWET1 LISWET10 LISWET11 LISWET12
  LISWET2 LISWET3 LISWET4 LISWET5 LISWET6 LISWET7 LISWET8 LISWET9
  LOTSCHD
  MOSARQP1 MOSARQP2
  POWELL20
  PRIMAL1 PRIMAL2 PRIMAL3 PRIMAL4
  PRIMALC1 PRIMALC2 PRIMALC5 PRIMALC8
  Q25FV47 QADLITTL QAFIRO QBANDM QBEACONF QBORE3D
  QBRANDY QCAPRI QE226 QETAMACR QFFFFF80 QFORPLAN
  QGFRDXPN QGROW15 QGROW22 QGROW7 QISRAEL
  QPCBLEND QPCBOEI1 QPCBOEI2 QPCSTAIR
  QPILOTNO QPTEST QRECIPE QSC205
  QSCAGR25 QSCAGR7 QSCFXM1 QSCFXM2 QSCFXM3
  QSCORPIO QSCRS8 QSCSD1 QSCSD6 QSCSD8
  QSCTAP1 QSCTAP2 QSCTAP3 QSEBA
  QSHARE1B QSHARE2B QSHELL
  QSHIP04L QSHIP04S QSHIP08L QSHIP08S QSHIP12L QSHIP12S
  QSIERRA QSTAIR QSTANDAT
  S268
  STADAT1 STADAT2 STADAT3
  STCQP1 STCQP2
  TAME UBH1 VALUES YAO ZECEVIC2
)

total=${#NAMES[@]}
ok=0
skip=0
fail=0
fail_names=()

# .mat -> .QPS 変換は scripts/qp_to_qps.py を再利用 (新規 logic 書かない)
CONVERTER=$(cat <<'PYEOF'
import sys
from pathlib import Path
import numpy as np
import scipy.io
import scipy.sparse as spa

sys.path.insert(0, sys.argv[3])  # repo_root/scripts
from qp_to_qps import write_qps

mat_path = Path(sys.argv[1])
qps_path = Path(sys.argv[2])
name = sys.argv[4]

mat = scipy.io.loadmat(str(mat_path))
Q = mat['Q']
c = np.asarray(mat['c']).flatten().astype(float)
A = mat['A']
rl = np.asarray(mat['rl']).flatten().astype(float)
ru = np.asarray(mat['ru']).flatten().astype(float)
lb = np.asarray(mat['lb']).flatten().astype(float)
ub = np.asarray(mat['ub']).flatten().astype(float)

# Maros の ±inf convention: ±1e20 → ±inf
INF_THR = 1e19
rl = np.where(rl <= -INF_THR, -np.inf, rl)
ru = np.where(ru >=  INF_THR,  np.inf, ru)
lb = np.where(lb <= -INF_THR, -np.inf, lb)
ub = np.where(ub >=  INF_THR,  np.inf, ub)

# Q が片側 triangle 格納の場合に備え symmetrize:
#   full sym 格納なら Q≈Q.T → そのまま
#   triangle のみなら Q+Q.T-diag(Q) で full sym 化
Q = spa.csc_matrix(Q)
diff = Q - Q.T
if abs(diff).max() if diff.nnz > 0 else 0.0 > 1e-10:
    Q = Q + Q.T - spa.diags(Q.diagonal())

write_qps(
    name=name[:8],
    P=Q,
    q=c,
    A=A,
    l=rl,
    u=ru,
    out_path=qps_path,
    var_lb=lb,
    var_ub=ub,
)
PYEOF
)

for name in "${NAMES[@]}"; do
  out="$OUT_DIR/${name}.QPS"
  if [[ -s "$out" ]]; then
    skip=$((skip + 1))
    continue
  fi

  mat_cached="$MAT_CACHE/${name}.mat"
  if [[ ! -s "$mat_cached" ]]; then
    tmp=$(mktemp)
    if ! curl -fsSL "$BASE_URL/${name}.mat" -o "$tmp"; then
      rm -f "$tmp"
      fail=$((fail + 1))
      fail_names+=("$name(download)")
      echo "[fail] $name (download)"
      continue
    fi
    mv "$tmp" "$mat_cached"
  fi

  # 変換は atomic: 一時 file に書いて成功時のみ rename
  tmp_qps=$(mktemp --suffix=.QPS 2>/dev/null || mktemp -t maros)
  if python3 -c "$CONVERTER" "$mat_cached" "$tmp_qps" "$REPO_ROOT/scripts" "$name" 2>/tmp/maros_dl_err; then
    mv "$tmp_qps" "$out"
    ok=$((ok + 1))
    echo "[ok]   $name"
  else
    rm -f "$tmp_qps"
    fail=$((fail + 1))
    fail_names+=("$name(convert)")
    echo "[fail] $name (convert: $(head -1 /tmp/maros_dl_err 2>/dev/null))"
  fi
done

echo ""
echo "[summary] maros_meszaros: ok=$ok skip=$skip fail=$fail total=$total -> $OUT_DIR"
if (( fail > 0 )); then
  echo "[fail-names] ${fail_names[*]}"
  exit 1
fi
