#!/bin/bash
# Build the Netlib emps decoder used by LP benchmark-data download scripts.

set -euo pipefail

EMPS_OUT="${EMPS_BIN:-/tmp/emps}"
EMPS_SRC="${EMPS_SRC:-/tmp/emps.c}"
EMPS_URL="${EMPS_URL:-https://www.netlib.org/lp/data/emps.c}"

CURL_RETRY="${EMPS_CURL_RETRY:-5}"
CURL_RETRY_DELAY="${EMPS_CURL_RETRY_DELAY:-3}"
CURL_CONNECT_TIMEOUT="${EMPS_CURL_CONNECT_TIMEOUT:-20}"
CURL_MAX_TIME="${EMPS_CURL_MAX_TIME:-180}"

if [[ -x "$EMPS_OUT" ]]; then
  echo "[ensure_emps] exists: $EMPS_OUT"
  exit 0
fi

tmp_src="$(mktemp "${EMPS_SRC}.XXXXXX")"
cleanup() {
  rm -f "$tmp_src"
}
trap cleanup EXIT

echo "[ensure_emps] download: $EMPS_URL"
curl \
  --fail \
  --show-error \
  --location \
  --retry "$CURL_RETRY" \
  --retry-delay "$CURL_RETRY_DELAY" \
  --retry-all-errors \
  --connect-timeout "$CURL_CONNECT_TIMEOUT" \
  --max-time "$CURL_MAX_TIME" \
  "$EMPS_URL" \
  --output "$tmp_src"

if [[ ! -s "$tmp_src" ]]; then
  echo "[ensure_emps] error: downloaded source is empty: $EMPS_URL" >&2
  exit 1
fi

mkdir -p "$(dirname "$EMPS_OUT")"
cc -x c -o "$EMPS_OUT" "$tmp_src"

if [[ ! -x "$EMPS_OUT" ]]; then
  echo "[ensure_emps] error: build did not produce executable: $EMPS_OUT" >&2
  exit 1
fi

cp "$tmp_src" "$EMPS_SRC" 2>/dev/null || true
echo "[ensure_emps] built: $EMPS_OUT"
