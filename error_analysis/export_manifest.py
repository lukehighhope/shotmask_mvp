#!/usr/bin/env python3
"""
Export one JSON Line per FP / FN for human error taxonomy (see taxonomy.json).

Example:
  python error_analysis/export_manifest.py --use-split --cnn-only --out outputs/err_manifest.jsonl
  python error_analysis/export_manifest.py --folder traning\\\\data\\\\dz01032026 --out outputs/err_manifest.jsonl

Later: merge labels in spreadsheet (audit_id → human_primary_id) and run summarize_labels.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluate_multivideo import (
    NMS_TIME_WINDOW,
    TOL,
    analyze_false_negatives,
    analyze_false_positives,
    gunshot_detection_context_for_video,
    _fp_automatic_hints,
)


def _write_record(fh, obj):
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_manifest(video_paths: list[str], *, cnn_only, tol, threshold, nms, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp_i = fn_i = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        _write_record(
            fh,
            {
                "record_type": "_meta",
                "tol": tol,
                "nms": nms,
                "threshold_override": threshold,
                "cnn_only": cnn_only,
                "n_videos_requested": len(video_paths),
            },
        )

        for vp in video_paths:
            ctx, reason = gunshot_detection_context_for_video(
                vp, cnn_only=cnn_only, tol=tol, threshold=threshold, nms=nms
            )
            if ctx is None:
                _write_record(
                    fh,
                    {
                        "record_type": "skip",
                        "video": os.path.basename(vp),
                        "video_path": os.path.abspath(vp),
                        "reason": reason,
                    },
                )
                continue

            name = ctx["video"]
            ref_times = ctx["ref_times"]
            shots = ctx["shots_final"]
            cands = ctx["candidates_stage1"]
            beep_v = ctx["beep_t_video_axis"]

            fps = analyze_false_positives(ref_times, shots, tol)
            for row in fps:
                fp_i += 1
                det_t = float(row["t"])
                hints = _fp_automatic_hints(det_t, beep_v, ref_times, tol)
                audit_id = f"{name}|FP|{fp_i:04d}"
                _write_record(
                    fh,
                    {
                        "record_type": "fp",
                        "audit_id": audit_id,
                        "video": name,
                        "video_path": ctx["video_path"],
                        "audio_extracted_path": ctx["audio_path"],
                        "det_time_ref_axis_sec": round(det_t, 4),
                        "confidence": row["confidence"],
                        "distance_to_nearest_gt": row["distance_to_nearest_gt"],
                        "hints": hints,
                        "human_primary": None,
                        "human_secondary": [],
                        "notes": "",
                    },
                )

            fnrows = analyze_false_negatives(ref_times, shots, cands, tol)
            for r in fnrows:
                fn_i += 1
                gap = r["gap_sec_after_prev_gt"]
                audit_id = f"{name}|FN|{fn_i:04d}"
                cdt = r["nearest_cand_peak_dt"]
                cdt_f = float(cdt) if cdt is not None else None
                cc_raw = r.get("nearest_cand_confidence")
                cc = float(cc_raw) if cc_raw is not None else None
                fn_hints = {
                    "might_fast_follow": gap is not None and gap < 0.28,
                    "might_classifier_reject": cdt_f is not None and cdt_f <= 0.12 and (cc is not None and cc < 0.55),
                    "might_no_stage1": cdt_f is None or cdt_f > 0.10,
                    "seconds_after_beep": round(float(r["gt_t_ref_axis"]) - beep_v, 4),
                }
                _write_record(
                    fh,
                    {
                        "record_type": "fn",
                        "audit_id": audit_id,
                        "video": name,
                        "video_path": ctx["video_path"],
                        "audio_extracted_path": ctx["audio_path"],
                        **r,
                        "hints": fn_hints,
                        "human_primary": None,
                        "human_secondary": [],
                        "notes": "",
                    },
                )


def main():
    ap = argparse.ArgumentParser(description="Export FP/FN rows for taxonomy labeling")
    ap.add_argument("--use-split", action="store_true", help="validation videos from dataset_split")
    ap.add_argument("--folder", default=None, help="folder of mp4 (if not --use-split)")
    ap.add_argument("--video", default=None, help="single mp4 basename inside --folder")
    ap.add_argument("--cnn-only", action="store_true")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--nms", type=float, default=NMS_TIME_WINDOW)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", "-o", type=Path, default=Path("outputs/error_audit_manifest.jsonl"))
    args = ap.parse_args()

    if args.use_split:
        from dataset_split import get_val_video_paths

        videos = get_val_video_paths()
    elif args.folder:
        folder = os.path.abspath(args.folder)
        if args.video:
            vpath = os.path.join(folder, args.video)
            if not os.path.isfile(vpath):
                raise SystemExit(f"not found: {vpath}")
            videos = [vpath]
        else:
            videos = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mp4"))
    else:
        raise SystemExit("need --use-split or --folder")

    if not videos:
        raise SystemExit("no videos")

    build_manifest(
        videos,
        cnn_only=args.cnn_only,
        tol=float(args.tol),
        threshold=args.threshold,
        nms=float(args.nms),
        out_path=args.out,
    )
    print(f"wrote {args.out}  (videos={len(videos)})")


if __name__ == "__main__":
    main()
