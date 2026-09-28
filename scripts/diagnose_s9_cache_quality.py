"""
檢查 s9 的快取影格資料品質，排除「跟s2一樣受8/27那個嘴唇偵測信心度bug影響」的可能性。
bug特徵：受影響的影片快取是全黑/單一數值畫面 → 原始(未標準化前)像素標準差接近0。

執行方式（WSL）：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/diagnose_s9_cache_quality.py
"""
import os
import glob
import numpy as np

os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

files = sorted(glob.glob(os.path.join('data', 's9', '*.mpg')))
print(f's9 共 {len(files)} 支影片')

bad = []
stds = []
for fp in files:
    file_name = os.path.splitext(os.path.basename(fp))[0]
    cache_path = None
    for suffix in ['_cached', '_dlib_cached']:
        p = os.path.join('data', f's9{suffix}', f'{file_name}.npy')
        if os.path.exists(p):
            cache_path = p
            break
    if cache_path is None:
        bad.append((file_name, '找不到快取檔', None))
        continue
    frames = np.load(cache_path)
    raw_std = float(np.std(frames))
    stds.append(raw_std)
    if frames.shape != (75, 46, 140, 1):
        bad.append((file_name, f'shape異常: {frames.shape}', raw_std))
    elif raw_std < 1.0:  # 正常影片像素標準差通常遠大於1；全黑/單一數值畫面std會接近0
        bad.append((file_name, f'std過低(疑似全黑/空白幀): {raw_std:.4f}', raw_std))
    elif np.isnan(frames).any():
        bad.append((file_name, 'frames含NaN', raw_std))

print(f'\n異常筆數: {len(bad)} / {len(files)}')
for name, problem, std in bad[:30]:
    print(f'  {name}: {problem}')
if len(bad) > 30:
    print(f'  ...(還有 {len(bad)-30} 筆，只顯示前30)')

if stds:
    stds_arr = np.array(stds)
    print(f'\n全部 std 分布: min={stds_arr.min():.2f}, max={stds_arr.max():.2f}, '
          f'mean={stds_arr.mean():.2f}, median={np.median(stds_arr):.2f}')
    print(f'std < 5 的筆數: {(stds_arr < 5).sum()}（越接近0越像全黑/異常幀）')

print('\n' + '=' * 60)
if len(bad) == 0:
    print('結論：s9快取資料品質正常，沒有發現全黑/異常幀，排除資料品質問題')
else:
    pct = len(bad) / len(files) * 100
    print(f'結論：s9有 {len(bad)}/{len(files)} ({pct:.1f}%) 筆快取資料異常，'
          f'{"可能是導致0%正確率的部分原因，需要修復後重測" if pct > 5 else "比例很低，不太可能是0%的主因，但仍建議修復"}')
print('=' * 60)
