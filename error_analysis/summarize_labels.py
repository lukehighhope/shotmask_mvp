#!/usr/bin/env python3
"""Pareto summary on labeled FP/FN (human_primary taxonomy id filled in manifest JSONL).

Label flow:
  1) Export: python error_analysis/export_manifest.py --use-split --cnn-only -o outputs/audit.jsonl
  2) Edit audit.jsonl — set `"human_primary": "FP_ECHO"` (ids from taxonomy.json)
  3) Summarize:

     python error_analysis/summarize_labels.py outputs/audit.jsonl
     python error_analysis/summarize_labels.py outputs/audit.jsonl --target-pct 0.80 --combined

Or merge from CSV (--csv labels.csv columns audit_id,human_primary) into counts without editing JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _pareto_table(counter: Counter, total: int | None, target_pct: float, title: str):
    rows = counter.most_common()
    print(f"\n=== {title} (labeled n={total}) ===")
    if not rows or total is None or total <= 0:
        print("(no labeled rows)")
        return
    acc = 0
    thresh = target_pct * total
    top_k_hit = None
    acc_at_hit = None
    print(f"{'#':>3} {'id':<34} {'n':>5} {'pct':>8} {'cum':>9}")
    for i, (lab, cnt) in enumerate(rows, start=1):
        acc += cnt
        pct = 100.0 * cnt / total
        cum = 100.0 * acc / total
        print(f"{i:>3} {lab:<34} {cnt:>5} {pct:>7.1f}% {cum:>8.1f}%")
        if top_k_hit is None and acc >= thresh:
            top_k_hit = i
            acc_at_hit = acc
    if top_k_hit is not None and acc_at_hit is not None:
        print(
            f"--> ≥{target_pct:.0%} of errors (~{thresh:.1f}/{total}) in top {top_k_hit} ids "
            f"(cumulative count {acc_at_hit})"
        )
    else:
        print(f"--> Could not reach {target_pct:.0%} cumulative with disjoint tail (need more dominant categories or bigger sample)")

    labeled_fp: Counter[str] = Counter()
    labeled_fn: Counter[str] = Counter()
    labeled_all: Counter[str] = Counter()
    csv_map: dict[str, str] = {}
    if csv_override and csv_override.is_file():
        with open(csv_override, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                aid = (row.get("audit_id") or "").strip()
                lab = (row.get("human_primary") or "").strip()
                if aid and lab:
                    csv_map[aid] = lab

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record_type") not in {"fp", "fn"}:
                continue
            lab = rec.get("human_primary")
            if isinstance(lab, str):
                lab = lab.strip()
            if not lab:
                aid = rec.get("audit_id", "")
                lab = csv_map.get(aid, "") or ""

            if not lab:
                continue

            rt = rec["record_type"]
            if rt == "fp":
                labeled_fp[lab] += 1
            elif rt == "fn":
                labeled_fn[lab] += 1
            labeled_all[lab] += 1

    tf = sum(labeled_fp.values())
    tn = sum(labeled_fn.values())
    _pareto_table(labeled_fp, tf, target_pct, "False positives by human_primary")
    _pareto_table(labeled_fn, tn, target_pct, "False negatives by human_primary")
    if combined:
        _pareto_table(labeled_all, tf + tn, target_pct, "Combined FP+FN by human_primary")

    print(f"\nTotals: labeled_fp={tf} labeled_fn={tn} combined={tf + tn}")
    print("(Unlabeled rows are ignored; export sets human_primary null until you annotate.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path, help="labeled JSONL from export_manifest")
    ap.add_argument("--csv", type=Path, default=None, help="optional audit_id,human_primary merge")
    ap.add_argument("--target-pct", type=float, default=0.80, help="Pareto line (default 0.80)")
    ap.add_argument("--combined", action="store_true", help="also FP+FN combined table")
    args = ap.parse_args()
    summarize_jsonl(args.manifest, args.csv, float(args.target_pct), args.combined)


if __name__ == "__main__":
    main()
