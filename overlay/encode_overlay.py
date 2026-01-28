import subprocess
import shutil


def _get_ffmpeg():
    return shutil.which("ffmpeg") or "ffmpeg"


def encode_webm(frames_dir, fps, out, ffmpeg_cmd=None):
    ff = ffmpeg_cmd or _get_ffmpeg()
    cmd = (
        f'"{ff}" -y -framerate {fps} '
        f'-i "{frames_dir}/%06d.png" '
        f'-c:v libvpx-vp9 -pix_fmt yuva420p '
        f'-b:v 0 -crf 28 "{out}"'
    )
    subprocess.run(cmd, shell=True, check=True)
