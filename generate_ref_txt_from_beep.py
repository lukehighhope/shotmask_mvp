"""
根据 beep_overrides.json 中的验证 beep 时间，将同目录下 S1.txt～S8.txt 从「相对 beep 的秒数」转为「视频内绝对时间」，写回对应 .txt。
这样 ref 文件自包含，不依赖运行时 beep 检测。

Usage:
  python generate_ref_txt_from_beep.py --folder "traning data/outdoor-20260208T235429Z-1-001"
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description="Generate ref .txt with absolute times from beep_overrides.json")
    ap.add_argument("--folder", required=True, help="Folder containing beep_overrides.json and S1.txt ... S8.txt")
    args = ap.parse_args()
    folder = os.path.abspath(args.folder)
    override_path = os.path.join(folder, "beep_overrides.json")
    if not os.path.isfile(override_path):
        print(f"No beep_overrides.json in {folder}")
        return 1
    with open(override_path, "r", encoding="utf-8") as f:
        beep_times = json.load(f)

    for key in sorted(beep_times.keys()):
        beep_t = float(beep_times[key])
        txt_path = os.path.join(folder, key + ".txt")
        if not os.path.isfile(txt_path):
            print(f"  {key}: no {key}.txt, skip")
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        offsets = []
        for line in lines:
            try:
                offsets.append(float(line))
            except ValueError:
                continue
        if not offsets:
            print(f"  {key}: no numbers in {key}.txt, skip")
            continue
        absolute = [round(beep_t + x, 4) for x in offsets]
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(str(t) for t in absolute))
        print(f"  {key}.txt: beep={beep_t:.2f}s, {len(absolute)} ref -> absolute times (e.g. {absolute[0]:.2f} .. {absolute[-1]:.2f}s)")
    print(f"\nDone: updated ref .txt in {folder} with absolute times.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
