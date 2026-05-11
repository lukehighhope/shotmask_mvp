"""Pure evaluation helpers (imports full evaluate_multivideo for submodule coverage)."""
import json
from pathlib import Path

from evaluate_multivideo import (
    TOL,
    analyze_false_positives,
    append_benchmark_jsonl,
    evaluate_gunshot_detection_on_videos,
    evaluate_shots,
    non_maximum_suppression,
)


def test_evaluate_shots_perfect_match():
    refs = [1.0, 2.0]
    shots = [{"t": 1.01}, {"t": 2.005}]
    tp, fp, fn, p, r, f1 = evaluate_shots(refs, shots, tol=0.06)
    assert tp == 2 and fp == 0 and fn == 0
    assert p == 1.0 and r == 1.0 and f1 == 1.0


def test_evaluate_shots_one_fn():
    refs = [1.0, 3.0]
    shots = [{"t": 1.0}]
    tp, fp, fn, p, r, f1 = evaluate_shots(refs, shots, tol=0.06)
    assert tp == 1 and fp == 0 and fn == 1


def test_evaluate_shots_fp():
    refs = [1.0]
    shots = [{"t": 1.0}, {"t": 2.5}]
    tp, fp, fn, _, _, _ = evaluate_shots(refs, shots, tol=0.06)
    assert tp == 1 and fp == 1 and fn == 0


def test_evaluate_shots_one_shot_cannot_double_count():
    refs = [1.0, 1.02]
    shots = [{"t": 1.01}]
    tp, fp, fn, _, _, _ = evaluate_shots(refs, shots, tol=0.06)
    assert tp == 1 and fn == 1


def test_non_maximum_suppression_keeps_higher_confidence():
    d = [{"t": 0.10, "confidence": 0.5}, {"t": 0.12, "confidence": 0.9}]
    out = non_maximum_suppression(d, time_window=0.06)
    assert len(out) == 1
    assert out[0]["confidence"] == 0.9


def test_non_maximum_suppression_empty():
    assert non_maximum_suppression([], time_window=0.06) == []


def test_analyze_false_positives_basic():
    refs = [1.0]
    shots = [{"t": 1.0, "confidence": 0.9}, {"t": 3.0, "confidence": 0.7}]
    fps = analyze_false_positives(refs, shots, tol=0.06)
    assert len(fps) == 1
    assert fps[0]["t"] == 3.0


def test_evaluate_gunshot_empty_videos():
    r = evaluate_gunshot_detection_on_videos([], cnn_only=True, verbose=False)
    assert r["pooled"]["tp"] == 0
    assert r["counts"]["n_input_videos"] == 0
    assert r["params"]["cnn_only"] is True


def test_append_benchmark_jsonl_writes_line(tmp_path, monkeypatch):
    monkeypatch.delenv("SHOTMASK_TRAINING_DATA_ROOT", raising=False)
    jp = Path(tmp_path) / "benchmark.jsonl"
    result = evaluate_gunshot_detection_on_videos([], cnn_only=True, verbose=False)
    append_benchmark_jsonl(str(jp), result, tag="unit", notes="n/a")
    text = jp.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 1
    row = json.loads(text[0])
    assert row["meta"]["tag"] == "unit"
    assert row["pooled"]["tp"] == 0
    assert "recorded_utc" in row["meta"]
