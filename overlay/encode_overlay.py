import subprocess
import shutil


def _get_ffmpeg():
    return shutil.which("ffmpeg") or "ffmpeg"


def encode_webm(frames_dir, fps, out, ffmpeg_cmd=None):
    ff = ffmpeg_cmd or _get_ffmpeg()
    # Use yuv420p (no alpha). Scale to 720p to avoid libvpx-vp9/libx264 memory allocation errors on some systems.
    cmd = (
        f'"{ff}" -y -framerate {fps} '
        f'-i "{frames_dir}/%06d.png" '
        f'-vf "scale=1280:720,format=yuv420p" -c:v libvpx-vp9 -pix_fmt yuv420p -b:v 0 -crf 28 "{out}"'
    )
    subprocess.run(cmd, shell=True, check=True)
