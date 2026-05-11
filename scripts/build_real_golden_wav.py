"""
Rebuild tests/fixtures/golden_gunshot.wav from Wikimedia Commons (public domain).

Source: Commons file "Gunshots 8.ogg" (PdSounds / LibriVox-style simulated gunshots).
Expected SHA256 of downloaded OGG: 0107c3a14e256c2a4c2e94f7494df6994edba75f4de05fced2d8900cf19b4c0b

Requirements: pip install -r requirements-dev.txt  (needs imageio-ffmpeg)

Usage (from repo root):
  python scripts/build_real_golden_wav.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import imageio_ffmpeg as iif

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OGG_SHA256 = "0107c3a14e256c2a4c2e94f7494df6994edba75f4de05fced2d8900cf19b4c0b"
SOURCE_URL = "https://upload.wikimedia.org/wikipedia/commons/e/ee/Gunshots_8.ogg"
DURATION_SEC = 3.5


def main() -> None:
    tmp_ogg = REPO_ROOT / "tmp" / "_golden_dl_gunshots.ogg"
    tmp_ogg.parent.mkdir(parents=True, exist_ok=True)
    if not tmp_ogg.is_file():
        print("Downloading", SOURCE_URL)
        payload = urllib.request.urlopen(SOURCE_URL, timeout=120).read()  # noqa: S310 pinned URL only
        tmp_ogg.write_bytes(payload)
    h = hashlib.sha256(tmp_ogg.read_bytes()).hexdigest()
    if h != EXPECTED_OGG_SHA256:
        raise SystemExit(f"SHA256 mismatch for OGG ({h}); URL may have changed; update script & expected hashes.")

    out_wav = REPO_ROOT / "tests" / "fixtures" / "golden_gunshot.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") or iif.get_ffmpeg_exe()

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(tmp_ogg),
            "-ac",
            "1",
            "-ar",
            "48000",
            "-to",
            str(DURATION_SEC),
            str(out_wav),
        ],
        check=True,
        capture_output=True,
    )

    meta = REPO_ROOT / "tests" / "fixtures" / "golden_gunshot_fixture.json"
    meta.write_text(
        json.dumps(
            {
                "source": "Gunshots 8.ogg — Wikimedia Commons (public domain). See GOLDEN_AUDIO_ATTRIBUTION.txt",
                "source_url": "https://commons.wikimedia.org/wiki/File:Gunshots_8.ogg",
                "download_sha256_ogg": EXPECTED_OGG_SHA256,
                "sr": 48000,
                "channels_mono": True,
                "slice_from_sec": 0,
                "slice_duration_sec": DURATION_SEC,
                "fps_for_tests": 30,
                "format_version": 2,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    wav_hash = hashlib.sha256(out_wav.read_bytes()).hexdigest()[:16]
    print(f"Wrote {out_wav}  sha256_prefix={wav_hash}")
    print("Next: PYTHONHASHSEED irrelevant — run:")
    print(r'  $env:CUDA_VISIBLE_DEVICES="" ; python scripts/refresh_golden_expected.py')


if __name__ == "__main__":
    main()
