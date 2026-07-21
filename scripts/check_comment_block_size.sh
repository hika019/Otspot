#!/bin/bash
# Fail if a production .rs file has an over-long consecutive comment block.
#
# CLAUDE.md, "実装" section: "コメントは OSS 水準で少なめ。メモ書き・作業ログ的
# コメントは書かない。" rustdoc (exactly `///`, or `//!`) and plain notes (`//`,
# plus `////`+ which Rust does NOT treat as doc) get separate limits, but a
# single contiguous comment run is ONE block: a run's doc lines and its note
# lines are each tallied over the whole run and checked against their own limit.
# Interleaving one `///` into a long `//` run (or one `//` into a long `///`
# run) therefore cannot split the run into sub-threshold pieces — the tally
# resets only at a non-comment line. rustdoc may run longer (API/algorithm
# docs); scratch notes must stay short.
#
# Thresholds come from the repo-wide block-length distribution (run-based,
# measured over otspot-{core,io,model,dev}/src): of 3272 doc blocks, 3258 are
# <= 17 lines and the single longest is 22 (conic/qcqp.rs, the Higham
# eigenvalue-perturbation derivation) — a genuine rustdoc block practically
# never needs more. MAX_DOC=24 gives that longest legitimate block a 2-line
# margin so it does not sit exactly on the threshold (any edit that grows it
# by one line would otherwise fail the gate) while still catching 25+ line
# outliers. All 3179 note blocks are <= 12 lines; MAX_MEMO=14 gives the same
# 2-line margin over that longest legitimate note block, since a design note
# needing 15+ consecutive `//` lines is almost always a work log or content
# that belongs in rustdoc.
#
# Marker-only lines (a bare `///`/`//!`/`//`) are paragraph breaks and are not
# counted: a rustdoc block split into paragraphs — which clippy's
# doc_lazy_continuation lint requires after a list — must not be penalised for
# the separators. They still keep the surrounding run contiguous, so inserting
# blanks cannot game the limit.
set -euo pipefail

# Max consecutive rustdoc (`///` / `//!`) lines allowed in one block.
readonly MAX_DOC=24
# Max consecutive plain-comment (`//`, `////`+) lines allowed in one block.
readonly MAX_MEMO=14

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

HITS=""
while IFS= read -r f; do
  out=$(awk -v maxdoc="$MAX_DOC" -v maxmemo="$MAX_MEMO" -v fname="$f" '
    function flush(   ) {
      if (dcnt > maxdoc) {
        printf "%s:%d-%d: rustdoc (///,//!) block %d lines (> MAX_DOC=%d)\n",
          fname, rstart, rend, dcnt, maxdoc
      }
      if (mcnt > maxmemo) {
        printf "%s:%d-%d: note (//) block %d lines (> MAX_MEMO=%d)\n",
          fname, rstart, rend, mcnt, maxmemo
      }
      dcnt = 0; mcnt = 0; rstart = 0; rend = 0
    }
    {
      is_comment = ($0 ~ /^[[:space:]]*\/\//)
      if (is_comment) {
        # rustdoc is exactly three slashes (`///` then non-slash/EOL) or `//!`;
        # `////`+ is a plain note, so a stray extra slash cannot buy the higher
        # MAX_DOC budget.
        is_doc = ($0 ~ /^[[:space:]]*(\/\/\/([^\/]|$)|\/\/!)/)
        # A marker-only line (bare `///`/`//!`/`//`) is a paragraph break: it
        # keeps the run contiguous but is not counted.
        is_blank = ($0 ~ /^[[:space:]]*\/\/[\/!]?[[:space:]]*$/)
        if (rstart == 0) rstart = NR
        rend = NR
        if (!is_blank) { if (is_doc) dcnt++; else mcnt++ }
      } else {
        flush()
      }
    }
    END { flush() }
  ' "$f")
  [ -n "$out" ] && HITS+="$out"$'\n'
done < <(find otspot-core/src otspot-io/src otspot-model/src otspot-dev/src \
  -name '*.rs' -type f)

if [ -n "${HITS//[[:space:]]/}" ]; then
  echo "::error::Comment block too long (rustdoc > ${MAX_DOC} or note > ${MAX_MEMO} lines):"
  echo "$HITS"
  exit 1
fi
echo "comment block size check: OK"
