"""
Extract motion features from reference shot windows.
Usage: python extract_motion_features.py --video 1.mp4
"""
import argparse
import os
import sys

from detectors.beep import detect_beeps
from detectors.shot_motion import extract_motion_features_at_ref_shots, save_motion_features, analyze_motion_features
from reference_splits import ref_shot_times as get_ref_shot_times


def get_fps_from_video(video):
    """Get fps from video via ffprobe"""
    import subprocess
    import shutil
    
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        # Try common Windows paths
        common_paths = [
            r"C:\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe",
            r"C:\ffmpeg\bin\ffprobe.exe",
            r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
        ]
        for path in common_paths:
            if os.path.exists(path):
                ffprobe = path
                break
    
    if not ffprobe:
        return None
    
    cmd = (
        f'"{ffprobe}" -v error -select_streams v:0 '
        f'-show_entries stream=r_frame_rate -of csv=p=0 "{video}"'
    )
    out = subprocess.check_output(cmd, shell=True).decode().strip()
    if "/" in out:
        num, den = map(int, out.split("/"))
        return num / den if den else 30.0
    return float(out) if out else 30.0


def extract_audio(video, ffmpeg, output_audio="tmp/audio.wav", channels=1):
    """Extract audio from video."""
    import subprocess
    os.makedirs(os.path.dirname(output_audio), exist_ok=True)
    print(f"Extracting audio ({channels}ch): {output_audio}")
    cmd = f'"{ffmpeg}" -y -i "{video}" -ac {channels} -ar 48000 -vn "{output_audio}"'
    subprocess.run(cmd, shell=True, check=True)
    return output_audio


def get_ffmpeg_cmd():
    """Get ffmpeg command."""
    import shutil
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    common_paths = [
        r"C:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return None


def main(video, output_json="outputs/motion_features.json"):
    ffmpeg = get_ffmpeg_cmd()
    if not ffmpeg:
        print("ffmpeg not detected, please install first.")
        raise SystemExit(1)
    
    # Extract audio for beep detection
    audio_mono = extract_audio(video, ffmpeg, "tmp/audio.wav", channels=1)
    
    # Detect beep to get reference shot times
    fps = get_fps_from_video(video)
    if fps is None:
        print("Could not get video FPS")
        raise SystemExit(1)
    
    print(f"Video FPS: {fps}")
    
    beeps = detect_beeps(audio_mono, fps)
    if not beeps:
        print("No beep detected, cannot determine reference shot times")
        raise SystemExit(1)
    
    beep_time = beeps[0]["t"]
    print(f"Beep time: {beep_time:.4f}s")
    
    ref_shot_times = get_ref_shot_times(beep_time)
    print(f"Reference shot times: {len(ref_shot_times)} shots")
    
    # Extract motion features
    print("\nExtracting motion features from reference shot windows...")
    features_list = extract_motion_features_at_ref_shots(
        video, fps, ref_shot_times, method="diff",
        window_before=0.02, window_after=0.08
    )
    
    if not features_list:
        print("No features extracted!")
        return
    
    print(f"Extracted features for {len(features_list)} shots")
    
    # Analyze features
    analyze_motion_features(features_list)
    
    # Save features
    save_motion_features(features_list, output_json)
    
    print(f"\nDone! Features saved to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract motion features from reference shot windows")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--output", default="outputs/motion_features.json", help="Output JSON path")
    args = parser.parse_args()
    
    main(args.video, args.output)
