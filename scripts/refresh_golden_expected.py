#!/usr/bin/env python3
"""Refresh golden_gunshot_expected.json using CPU Torch (matches GitHub Actions).

Run after changing golden_gunshot.wav:
  python scripts/refresh_golden_expected.py

Edits committed ``min_confidence_floor`` stays conservative (manual 0.64 in JSON afterwards if needed).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from detectors import config_paths  # noqa: E402
from detectors import shot_audio  # noqa: E402
from detectors.shot_audio import detect_shots  # noqa: E402


def main():
    wav = REPO_ROOT / "tests" / "fixtures" / "golden_gunshot.wav"
    if not wav.is_file():
        raise SystemExit(f"missing {wav}")

    fx_path = REPO_ROOT / "tests" / "fixtures" / "golden_gunshot_fixture.json"
    fps = 30
    if fx_path.is_file():
        meta = json.loads(fx_path.read_text(encoding="utf-8"))
        fps = int(meta.get("fps_for_tests", 30))

    repo = Path(config_paths.project_root())
    raw = json.loads((repo / "calibrated_detector_params.json").read_text(encoding="utf-8"))
    cal = dict(raw)
    cal["cnn_gunshot_path"] = str((repo / "outputs" / "cnn_gunshot.pt").resolve())
    cal["ast_gunshot_path"] = None
    cal["use_ast_gunshot"] = False
    cal["use_cnn_only"] = True

    orig = shot_audio.load_calibrated_params
    shot_audio.load_calibrated_params = lambda: shot_audio._resolve_cal_paths(dict(cal))
    try:
        shots = detect_shots(str(wav), fps=float(fps), use_calibrated=True)
    finally:
        shot_audio.load_calibrated_params = orig

    ordered = sorted(shots, key=lambda s: float(s["t"]))
    times = [round(float(s["t"]), 4) for s in ordered]

    blob = {
        "_source_attribution_file": "tests/fixtures/GOLDEN_AUDIO_ATTRIBUTION.txt",
        "_note": (
            "Production gate: calibrated min_confidence_threshold + CNN-only (AST off). "
            "Times snapshot from scripts/refresh_golden_expected.py (CPU Torch)."
        ),
        "fps": fps,
        "expected_shot_times_sec": times,
        "time_tolerance_sec": 0.05,
        "min_confidence_floor": 0.64,
    }
    out = REPO_ROOT / "tests" / "fixtures" / "golden_gunshot_expected.json"
    out.write_text(json.dumps(blob, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} n={len(times)} times={times}")


if __name__ == "__main__":
    main()
