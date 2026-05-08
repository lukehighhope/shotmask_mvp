"""
Prototype CLI: detect shots, cluster Stage2 spectral features, keep dominant cluster.

  python dominant_pov_shots.py --video \"traning data/.../foo.mp4\"

Requires: sklearn (same as AST training).
"""

import argparse
import json
import os
import sys

import numpy as np

from detectors.dominant_pov import annotate_shots_pov_cluster, filter_shots_dominant_pov
from detectors.shot_audio import detect_shots
from train_logreg_multivideo import extract_audio, get_fps_duration


def main() -> int:
    ap = argparse.ArgumentParser(description="POV dominant gunshot clustering (prototype)")
    ap.add_argument("--video", required=True, help="Path to .mp4")
    ap.add_argument("--k", type=int, default=2, help="KMeans clusters (>=2)")
    ap.add_argument("--min-matched", type=int, default=6, help="Min Stage2-matched shots to cluster")
    ap.add_argument(
        "--mode",
        choices=("filter", "annotate"),
        default="filter",
        help="filter: drop non-dominant; annotate: keep all, set pov_dominant true/false",
    )
    ap.add_argument("--json-out", default=None, help="Write shots + meta JSON here")
    args = ap.parse_args()

    vp = os.path.abspath(args.video)
    if not os.path.isfile(vp):
        print("Not found:", vp, file=sys.stderr)
        return 1

    try:
        audio_path = extract_audio(vp)
        fps, _ = get_fps_duration(vp)
    except Exception as e:
        print("Audio extract failed:", e, file=sys.stderr)
        return 1

    res = detect_shots(audio_path, fps, use_calibrated=True, return_candidates=True)
    if isinstance(res, tuple) and len(res) == 2:
        shots, cands = res
    else:
        print("detect_shots did not return (shots, candidates); use improved+calibrated path.", file=sys.stderr)
        return 1

    kw = dict(
        k_clusters=args.k,
        min_matched=args.min_matched,
        time_match_tol=0.10,
    )
    if args.mode == "filter":
        out_shots, meta = filter_shots_dominant_pov(shots, cands, unmatched_policy="keep", **kw)
    else:
        out_shots, meta = annotate_shots_pov_cluster(shots, cands, **kw)

    print(json.dumps(meta, indent=2))
    for s in sorted(out_shots, key=lambda x: float(x["t"])):
        t = float(s["t"])
        conf = float(s.get("confidence", 0))
        extra = ""
        if "pov_dominant" in s:
            extra = f"  pov_dominant={s['pov_dominant']}"
        print(f"{t:.4f}s  conf={conf:.3f}{extra}")

    if args.json_out:
        outp = os.path.abspath(args.json_out)
        jd = os.path.dirname(outp)
        if jd:
            os.makedirs(jd, exist_ok=True)
        payload = {"meta": meta, "shots": out_shots}
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("Wrote", args.json_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
