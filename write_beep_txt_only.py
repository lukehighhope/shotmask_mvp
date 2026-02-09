"""
不改变 timer 起点：只增加 S1beep.txt～S8beep.txt，每文件只写对应 beep 时间（一行）。
若 S1.txt 等已被改成绝对时间，则先根据 beep_overrides.json 还原为相对 beep 的秒数，再写 beep 文件。

Usage:
  python write_beep_txt_only.py --folder "traning data/outdoor-20260208T235429Z-1-001"
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description="Write S1beep.txt ... S8beep.txt with beep time; restore S1.txt to relative if needed")
    ap.add_argument("--folder", required=True)
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
        beep_txt_path = os.path.join(folder, key + "beep.txt")

        # Write beep file
        with open(beep_txt_path, "w", encoding="utf-8") as f:
            f.write(f"{beep_t:.4f}\n")
        print(f"  {key}beep.txt: {beep_t:.4f}s")

        # If ref .txt exists and values look like absolute (all > beep_t), restore to relative
        if not os.path.isfile(txt_path):
            continue
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
        except Exception:
            continue
        nums = []
        for line in lines:
            try:
                nums.append(float(line))
            except ValueError:
                continue
        if not nums:
            continue
        if min(nums) >= beep_t - 0.5:
            relative = [round(x - beep_t, 4) for x in nums]
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(str(t) for t in relative))
            print(f"  {key}.txt: restored to relative (since beep), {len(relative)} times")
    print(f"\nDone: added *beep.txt in {folder}, ref .txt kept as relative to beep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
