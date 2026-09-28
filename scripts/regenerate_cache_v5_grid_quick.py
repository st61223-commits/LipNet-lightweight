"""
修正版：用原始YOLOv5偵測器重製「真正GRID語料庫」說話者的快取，只取GRID s1前100支，
取代之前誤用s99_8/s34_3（都不是標準GRID資料）的快速測試。

輸出：data/s1_cached_v5_gridquick/{file}.npy

用法：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/regenerate_cache_v5_grid_quick.py
"""
import os
import glob
import time
import pathlib
import warnings

pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)

SPEAKER = 's1'
N_LIMIT = 100
CACHE_SUFFIX = '_cached_v5_gridquick'
CONF_THRESHOLD = 0.5
input_shape = (75, 46, 140, 1)

import cv2
import numpy as np
import torch

print('載入原始 YOLOv5（torch.hub本機載入）...')
YOLOV5_REPO = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5'
YOLOV5_WEIGHTS = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5\runs\v5s\weights\best.pt'
yolo_model = torch.hub.load(YOLOV5_REPO, 'custom', path=YOLOV5_WEIGHTS, source='local')
yolo_model.conf = CONF_THRESHOLD
print('載入完成！\n')

t_start = time.time()
cache_dir = os.path.join('data', f'{SPEAKER}{CACHE_SUFFIX}')
os.makedirs(cache_dir, exist_ok=True)
videos = sorted(glob.glob(os.path.join('data', SPEAKER, '*.mpg')))[:N_LIMIT]
print(f'=== {SPEAKER}（取前{len(videos)}支影片，真正GRID語料庫）===', flush=True)

for vi, video_path in enumerate(videos, 1):
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = os.path.join(cache_dir, f'{file_name}.npy')
    if os.path.exists(cache_path):
        continue

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model(frame)
        det = results.xyxy[0]
        if len(det) > 0:
            x1, y1, x2, y2 = [int(v) for v in det[0][:4]]
            lip = frame[y1:y2, x1:x2]
            if lip.size > 0:
                lip = cv2.resize(lip, (input_shape[2], input_shape[1]))
                lip = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
                frames.append(lip.astype(np.float32))
    cap.release()

    T = 75
    frames = frames[:T]
    if not frames:
        frames = [np.zeros((input_shape[1], input_shape[2]), dtype=np.float32)]
    while len(frames) < T:
        frames.append(frames[-1])
    arr = np.expand_dims(np.array(frames), -1)
    np.save(cache_path, arr)

    if vi % 25 == 0:
        print(f'  {SPEAKER}: {vi}/{len(videos)}', flush=True)

print(f'\n全部完成，總耗時 {(time.time()-t_start)/60:.1f} 分鐘')
