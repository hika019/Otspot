#!/bin/bash
# QPLIB metadata-driven mechanical filter (453 instances).
#
# Downloads the official instances.html, parses every row, and writes ID lists
# to data/qplib_filtered/ grouped by convexity, objective/variable/constraint
# type and size bucket. No .qplib payload is fetched here — use
# scripts/qplib_download.sh against a chosen ID list to materialise the data.
#
# Field reference (qplib.zib.de/doc.html#probtype):
#   probtype = O V C  (objective / variable / constraint type)
#     O: Q quadratic | L linear | C constant | D diagonal
#     V: C continuous | B binary | I integer | M binary+continuous | G general+continuous
#     C: N none | B box | L linear | Q quadratic | D diagonal
#   Cvx: ✔ if the continuous relaxation is convex, '-' otherwise.
#
# Size buckets are defined by SIZE_SMALL/SIZE_MEDIUM (defaults 100/1000).
#
# Usage:
#   bash scripts/qplib_filter.sh [OUT_DIR]
# OUT_DIR defaults to data/qplib_filtered.

set -euo pipefail

OUT_DIR="${1:-data/qplib_filtered}"
CACHE_DIR="${CACHE_DIR:-${OUT_DIR}/cache}"
META_URL="${META_URL:-https://qplib.zib.de/instances.html}"
META_FILE="$CACHE_DIR/instances.html"
SIZE_SMALL="${SIZE_SMALL:-100}"
SIZE_MEDIUM="${SIZE_MEDIUM:-1000}"

mkdir -p "$OUT_DIR" "$CACHE_DIR"

if [[ ! -s "$META_FILE" ]]; then
  echo "[fetch] $META_URL -> $META_FILE"
  tmp=$(mktemp)
  if ! curl -fsSL "$META_URL" -o "$tmp"; then
    rm -f "$tmp"
    echo "[fail] metadata fetch failed; set META_FILE= to point at a cached copy" >&2
    exit 1
  fi
  mv "$tmp" "$META_FILE"
fi

TSV="$OUT_DIR/instances.tsv"

python3 - "$META_FILE" "$TSV" "$SIZE_SMALL" "$SIZE_MEDIUM" "$OUT_DIR" <<'PY'
import html
import re
import sys
from pathlib import Path

meta_path, tsv_path, small_s, medium_s, out_dir = sys.argv[1:6]
small = int(small_s)
medium = int(medium_s)
out = Path(out_dir)

raw = Path(meta_path).read_text(encoding="utf-8")

# Slice the <TBODY>...</TBODY> region to avoid the <THEAD> sample cells.
m = re.search(r"<TBODY>(.*?)</TBODY>", raw, re.DOTALL | re.IGNORECASE)
if not m:
    sys.exit("instances.html: <TBODY> not found")
body = m.group(1)

row_re = re.compile(r"<TR\b[^>]*>(.*?)</TR>", re.DOTALL | re.IGNORECASE)
td_re = re.compile(r"<TD\b[^>]*>(.*?)</TD>", re.DOTALL | re.IGNORECASE)
id_re = re.compile(r"QPLIB_(\d+)\.html")
tag_re = re.compile(r"<[^>]+>")


def clean(cell: str) -> str:
    text = html.unescape(tag_re.sub("", cell)).strip()
    return text.replace("\xa0", "").strip()


def to_int(cell: str) -> int:
    text = clean(cell)
    return int(text) if text and text.lstrip("-").isdigit() else 0


rows = []
for tr in row_re.findall(body):
    tds = td_re.findall(tr)
    if len(tds) != 13:
        continue
    id_match = id_re.search(tds[0])
    if not id_match:
        continue
    iid = id_match.group(1)
    cvx_raw = clean(tds[1])
    convex = cvx_raw in ("✔", "✓")
    rows.append({
        "id": iid,
        "convex": convex,
        "obj": clean(tds[2]),
        "var": clean(tds[5]),
        "con": clean(tds[9]),
        "nvars": to_int(tds[6]),
        "nbin": to_int(tds[7]),
        "nint": to_int(tds[8]),
        "ncons": to_int(tds[10]),
        "nquad": to_int(tds[11]),
    })

if not rows:
    sys.exit("instances.html: no rows parsed (HTML layout changed?)")

# Normalised TSV.
with open(tsv_path, "w", encoding="utf-8") as f:
    f.write("id\tconvex\tO\tV\tC\tnvars\tnbin\tnint\tncons\tnquad\n")
    for r in rows:
        f.write(
            f"{r['id']}\t{int(r['convex'])}\t{r['obj']}\t{r['var']}\t{r['con']}\t"
            f"{r['nvars']}\t{r['nbin']}\t{r['nint']}\t{r['ncons']}\t{r['nquad']}\n"
        )


def size_bucket(n: int) -> str:
    if n <= small:
        return "small"
    if n <= medium:
        return "medium"
    return "large"


buckets: dict[str, list[str]] = {
    "all": [],
    "convex": [],
    "nonconvex": [],
    "obj_quadratic": [],
    "obj_linear": [],
    "var_continuous": [],
    "var_binary": [],
    "var_integer": [],
    "var_mixed": [],
    "con_linear": [],
    "con_box_or_none": [],
    "con_quadratic": [],
    # Solver scope = continuous QP with linear (or box/none) constraints.
    "in_scope_convex": [],
    "in_scope_nonconvex": [],
    "out_of_scope": [],
    "size_small": [],
    "size_medium": [],
    "size_large": [],
}

scope_size: dict[str, list[str]] = {
    "in_scope_convex_small": [],
    "in_scope_convex_medium": [],
    "in_scope_convex_large": [],
    "in_scope_nonconvex_small": [],
    "in_scope_nonconvex_medium": [],
    "in_scope_nonconvex_large": [],
}

for r in rows:
    iid = r["id"]
    buckets["all"].append(iid)
    buckets["convex" if r["convex"] else "nonconvex"].append(iid)

    if r["obj"] in ("Q", "D"):
        buckets["obj_quadratic"].append(iid)
    elif r["obj"] == "L":
        buckets["obj_linear"].append(iid)

    if r["var"] == "C":
        buckets["var_continuous"].append(iid)
    elif r["var"] == "B":
        buckets["var_binary"].append(iid)
    elif r["var"] == "I":
        buckets["var_integer"].append(iid)
    elif r["var"] in ("M", "G"):
        buckets["var_mixed"].append(iid)

    if r["con"] == "L":
        buckets["con_linear"].append(iid)
    elif r["con"] in ("B", "N"):
        buckets["con_box_or_none"].append(iid)
    elif r["con"] == "Q":
        buckets["con_quadratic"].append(iid)

    in_scope = r["obj"] in ("Q", "D") and r["var"] == "C" and r["con"] in ("L", "B", "N")
    if in_scope:
        bucket = "in_scope_convex" if r["convex"] else "in_scope_nonconvex"
        buckets[bucket].append(iid)
        scope_size[f"{bucket}_{size_bucket(r['nvars'])}"].append(iid)
    else:
        buckets["out_of_scope"].append(iid)

    buckets[f"size_{size_bucket(r['nvars'])}"].append(iid)


def write_list(name: str, ids: list[str]) -> None:
    (out / f"{name}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


for name, ids in buckets.items():
    write_list(name, ids)
for name, ids in scope_size.items():
    write_list(name, ids)

# Human-readable summary.
summary_lines = [
    f"# QPLIB filter summary",
    f"total: {len(rows)}",
    f"convex: {len(buckets['convex'])}",
    f"nonconvex: {len(buckets['nonconvex'])}",
    "",
    "## objective type",
    f"  Q+D (quadratic/diagonal): {len(buckets['obj_quadratic'])}",
    f"  L (linear):               {len(buckets['obj_linear'])}",
    "",
    "## variable type",
    f"  C continuous: {len(buckets['var_continuous'])}",
    f"  B binary:     {len(buckets['var_binary'])}",
    f"  I integer:    {len(buckets['var_integer'])}",
    f"  M/G mixed:    {len(buckets['var_mixed'])}",
    "",
    "## constraint type",
    f"  L linear:        {len(buckets['con_linear'])}",
    f"  B/N box-or-none: {len(buckets['con_box_or_none'])}",
    f"  Q quadratic:     {len(buckets['con_quadratic'])}",
    "",
    f"## solver scope (O in Q/D, V=C, C in L/B/N), size split n<={small}/{medium}",
    f"  in_scope_convex:    {len(buckets['in_scope_convex'])}"
    f"  (small {len(scope_size['in_scope_convex_small'])}"
    f" / medium {len(scope_size['in_scope_convex_medium'])}"
    f" / large {len(scope_size['in_scope_convex_large'])})",
    f"  in_scope_nonconvex: {len(buckets['in_scope_nonconvex'])}"
    f"  (small {len(scope_size['in_scope_nonconvex_small'])}"
    f" / medium {len(scope_size['in_scope_nonconvex_medium'])}"
    f" / large {len(scope_size['in_scope_nonconvex_large'])})",
    f"  out_of_scope:       {len(buckets['out_of_scope'])}",
]
(out / "SUMMARY.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
print("\n".join(summary_lines))

# Audit currently-tracked data/qplib/ entries (if present) against the metadata
# so #4 integrity issues surface here rather than at bench time.
tracked_dir = Path("data/qplib")
if tracked_dir.is_dir():
    by_id = {r["id"]: r for r in rows}
    tracked = sorted(
        p.stem.removeprefix("QPLIB_") for p in tracked_dir.glob("QPLIB_*.qplib")
    )
    lines = ["# Audit of data/qplib/ (currently tracked subset)",
             f"tracked total: {len(tracked)}\n",
             "id\tconvex\tO\tV\tC\tnvars\tncons\tin_scope"]
    in_scope_count = 0
    out_of_scope_ids: list[str] = []
    for iid in tracked:
        r = by_id.get(iid)
        if r is None:
            lines.append(f"{iid}\t?\t?\t?\t?\t?\t?\tMETADATA_MISSING")
            continue
        scope = (r["obj"] in ("Q", "D")
                 and r["var"] == "C"
                 and r["con"] in ("L", "B", "N"))
        if scope:
            in_scope_count += 1
        else:
            out_of_scope_ids.append(iid)
        cvx = "convex" if r["convex"] else "nonconvex"
        lines.append(
            f"{iid}\t{cvx}\t{r['obj']}\t{r['var']}\t{r['con']}\t"
            f"{r['nvars']}\t{r['ncons']}\t{'yes' if scope else 'no'}"
        )
    lines.append("")
    lines.append(f"in_scope (continuous QP): {in_scope_count}")
    lines.append(f"out_of_scope: {len(out_of_scope_ids)}"
                 + (f" -> {','.join(out_of_scope_ids)}" if out_of_scope_ids else ""))
    (out / "tracked_subset_audit.txt").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")
PY

echo ""
echo "[done] filter outputs -> $OUT_DIR"
