from extract_shell_features import process_video
import numpy as np
from pathlib import Path

video = Path('traning data/outdoor/dz01032026/1.mp4')
cali  = Path('traning data/outdoor/dz01032026/1cali.txt')

print(f'Processing: {video.name}')
X, y = process_video(video, cali, 'test')
if X is not None:
    pos = X[y==1]
    neg = X[y==0]
    labels = ['scr_A','scr_B','spike','mean','pk_A','actv','t_pk','rel_sp']
    print(f'\n{len(pos)} positive, {len(neg)} negative\n')
    print(f'{"Feature":<10}  {"POS mean":>9}  {"NEG mean":>9}  {"Diff":>7}')
    print('-' * 42)
    for i, lbl in enumerate(labels):
        pm = pos[:,i].mean()
        nm = neg[:,i].mean()
        print(f'{lbl:<10}  {pm:>9.3f}  {nm:>9.3f}  {pm-nm:>+7.3f}')

    import os; os.makedirs('outputs', exist_ok=True)
    np.savez('outputs/shell_test_1video.npz', X=X, y=y)
    print('\nSaved to outputs/shell_test_1video.npz')
