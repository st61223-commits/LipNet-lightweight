"""
比較三代嘴唇偵測器（YOLOv5 / YOLOv8 / YOLOv10）在CPU上的推論速度。
同一台機器、強制CPU、同一張假畫面(640x480x3)，各跑N次取平均。

YOLOv5的權重是用舊版獨立的 ultralytics/yolov5 專案訓練，跟v8/v10用的新版
統一 `ultralytics` 套件的載入器不相容(torch_safe_load會丟
"NOT forwards compatible with YOLOv8"錯誤)，所以v5改用torch.hub在本機
yolov5 repo(C:\\Users\\Tno\\OneDrive\\Lipnet_nchu\\yolov5)以它原生的方式載入，
v8/v10維持用新版ultralytics.YOLO()——兩者都是呼叫各自公開介面做「單張圖片
前處理+推論+NMS」的end-to-end時間，方法論上仍可比較。

用法：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/benchmark_yolo_detector_speed.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
import numpy as np
import torch

N_WARMUP = 3
N_RUNS = 30
dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

results = {}

# ---- YOLOv5：用torch.hub本機載入(舊版repo原生方式) ----
label = 'YOLOv5（原始偵測器）'
path = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5\runs\v5s\weights\best.pt'
repo = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5'
print(f'載入 {label} ({path}) ...')
t0 = time.time()
model_v5 = torch.hub.load(repo, 'custom', path=path, source='local')
model_v5.to('cpu')
print(f'  載入耗時: {time.time() - t0:.2f}秒')

for _ in range(N_WARMUP):
    model_v5(dummy)

times = []
for _ in range(N_RUNS):
    t0 = time.time()
    model_v5(dummy)
    times.append((time.time() - t0) * 1000)
times = np.array(times)
results[label] = times
print(f'  推論 {N_RUNS} 次: 平均 {times.mean():.1f}ms, 中位數 {np.median(times):.1f}ms, '
      f'最快 {times.min():.1f}ms, 最慢 {times.max():.1f}ms\n')
del model_v5

# ---- YOLOv8 / YOLOv10：用新版ultralytics.YOLO() ----
from ultralytics import YOLO

MODELS = {
    'YOLOv8（前一版現役）': r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov8_lip\lip_detect\weights\best.pt',
    'YOLOv10（現役）': r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov10_lip\lip_detect\weights\best.pt',
}
for label, path in MODELS.items():
    print(f'載入 {label} ({path}) ...')
    t0 = time.time()
    model = YOLO(path)
    print(f'  載入耗時: {time.time() - t0:.2f}秒')

    for _ in range(N_WARMUP):
        model(dummy, verbose=False, device='cpu')

    times = []
    for _ in range(N_RUNS):
        t0 = time.time()
        model(dummy, verbose=False, device='cpu')
        times.append((time.time() - t0) * 1000)

    times = np.array(times)
    results[label] = times
    print(f'  推論 {N_RUNS} 次: 平均 {times.mean():.1f}ms, 中位數 {np.median(times):.1f}ms, '
          f'最快 {times.min():.1f}ms, 最慢 {times.max():.1f}ms\n')
    del model

print('=' * 70)
print('對照結果（CPU推論，同一台機器同一張假畫面）')
print('=' * 70)
labels = list(results.keys())
baseline_mean = results[labels[0]].mean()
for label in labels:
    m = results[label].mean()
    ratio = baseline_mean / m
    print(f'{label:20s} 平均 {m:7.1f}ms   相對YOLOv5速度: {ratio:.2f}x')
print('=' * 70)
