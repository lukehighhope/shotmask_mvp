import json
from pathlib import Path

data_root = Path('traning data')
split_file = data_root / 'dataset_split.json'
split = json.load(open(split_file, encoding='utf-8-sig'))

total_shots = 0
total_beeps = 0
video_count = 0
missing_cali = []

for split_name in ['train', 'val']:
    for rel in split.get(split_name, []):
        base = str(data_root / rel).replace('.mp4', '')
        cali = Path(base + 'cali.txt')
        beep = Path(base + 'beep.txt')
        has_cali = cali.exists()
        has_beep = beep.exists()
        if has_cali:
            shots = [l.strip() for l in open(cali, encoding='utf-8') if l.strip()]
            total_shots += len(shots)
            video_count += 1
        if has_beep:
            beeps = [l.strip() for l in open(beep, encoding='utf-8') if l.strip()]
            total_beeps += len(beeps)
        if not has_cali:
            missing_cali.append(rel)

print(f'Videos with cali.txt :  {video_count}')
print(f'Total shot annotations: {total_shots}')
print(f'Total beep annotations: {total_beeps}')
print(f'Avg shots per video  :  {total_shots/max(video_count,1):.1f}')
if missing_cali:
    print(f'\nMissing cali ({len(missing_cali)}):')
    for m in missing_cali:
        print(f'  {m}')
