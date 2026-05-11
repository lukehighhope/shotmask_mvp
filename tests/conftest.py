import shutil
import subprocess
from pathlib import Path

import pytest

_FIXTURES = Path(__file__).resolve().parent / "fixtures"

def _ffmpeg_executable():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg as iif

        return iif.get_ffmpeg_exe()
    except Exception:
        return None


@pytest.fixture(scope="session")
def golden_gunshot_mp4_path(tmp_path_factory):
    """Mux committed WAV into a minimal H.264+AAC clip (needs ffmpeg executable)."""
    wav = _FIXTURES / "golden_gunshot.wav"
    if not wav.is_file():
        pytest.skip("golden_gunshot.wav missing")

    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        pytest.skip("ffmpeg unavailable (PATH or pip imageio-ffmpeg)")

    out_dir = tmp_path_factory.mktemp("golden_mux")
    mp4 = out_dir / "golden_gunshot.mp4"
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=160x120:r=30",
        "-i",
        str(wav),
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(mp4),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not mp4.is_file():
        pytest.skip(f"ffmpeg mux failed: {r.stderr[-500:] if r.stderr else 'no stderr'}")
    return str(mp4)
