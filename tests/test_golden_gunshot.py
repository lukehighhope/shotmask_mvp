"""
Golden regression on a short PD gunshot-derived clip.

Uses production calibrated ``min_confidence_threshold`` + CNN-only (AST off via cal patch).

See ``tests/fixtures/GOLDEN_AUDIO_ATTRIBUTION.txt`` for redistribution rights.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from detectors import config_paths
from detectors.shot_audio import detect_shots, load_calibrated_params, _resolve_cal_paths
from train_logreg_multivideo import extract_audio

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_WAV_PATH = _FIXTURES / "golden_gunshot.wav"
_REPO_ROOT = Path(config_paths.project_root())
_CNN_PATH = _REPO_ROOT / "outputs" / "cnn_gunshot.pt"


def _expected_profile():
    ep = _FIXTURES / "golden_gunshot_expected.json"
    raw = json.loads(ep.read_text(encoding="utf-8"))
    tol = float(raw["time_tolerance_sec"])
    fps = int(raw.get("fps", 30))
    exp_times = sorted(float(t) for t in raw["expected_shot_times_sec"])
    floor = float(raw.get("min_confidence_floor", 0.64))
    return exp_times, tol, fps, floor


def _cal_for_golden():
    base = load_calibrated_params() or {}
    cal = dict(base)
    if not _CNN_PATH.is_file():
        pytest.skip(f"CNN checkpoint missing: {_CNN_PATH} (tracked in repo for CI/regression)")
    cal["cnn_gunshot_path"] = str(_CNN_PATH.resolve())
    cal["ast_gunshot_path"] = None
    cal["use_ast_gunshot"] = False
    cal["use_cnn_only"] = True
    return _resolve_cal_paths(cal)


@pytest.fixture(name="golden_cal")
def fx_golden_cal(monkeypatch):
    merged = _cal_for_golden()
    monkeypatch.setattr(
        "detectors.shot_audio.load_calibrated_params",
        lambda: dict(merged),
    )
    yield merged


@pytest.mark.usefixtures("golden_cal")
def test_golden_wav_detect_shot_times_confidence():
    if not _WAV_PATH.is_file():
        pytest.fail(f"golden WAV missing; run scripts/build_real_golden_wav.py: {_WAV_PATH}")

    expected_sorted, tol, fps, floor = _expected_profile()

    shots = detect_shots(
        str(_WAV_PATH),
        fps=float(fps),
        use_calibrated=True,
    )
    sorted_shots = sorted(shots, key=lambda s: float(s["t"]))
    got = [round(float(s["t"]), 4) for s in sorted_shots]

    assert len(got) == len(expected_sorted), f"shot count mismatch: got {got}, expected ~{expected_sorted}"

    for a, b in zip(expected_sorted, got):
        assert abs(a - b) <= tol + 1e-6, f"time drift: expected {a}, got {b} (tol {tol}s)"

    for s in sorted_shots:
        assert float(s["confidence"]) >= floor - 1e-6


@pytest.mark.usefixtures("golden_cal")
def test_golden_mp4_extract_then_detect(golden_gunshot_mp4_path, tmp_path):
    expected_sorted, tol, fps, floor = _expected_profile()

    out_wav = extract_audio(golden_gunshot_mp4_path, out_dir=str(tmp_path / "wav_out"))
    shots = detect_shots(out_wav, fps=float(fps), use_calibrated=True)
    sorted_shots = sorted(shots, key=lambda s: float(s["t"]))
    got = [round(float(s["t"]), 4) for s in sorted_shots]

    assert len(got) == len(expected_sorted)
    for a, b in zip(expected_sorted, got):
        assert abs(a - b) <= tol + 1e-6
    for s in sorted_shots:
        assert float(s["confidence"]) >= floor - 1e-6
