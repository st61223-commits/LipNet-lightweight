"""
快速版：只用原始YOLOv5偵測器重製2位說話者(s99_8/s34_3，共299支影片)的訓練快取，
目的是幫v11(學長基準模型)補測WER正確率(之前沒測過)。

背景：v11當初訓練用的_cached(無後綴)快取資料夾已於9/15磁碟清理時刪除，
無法直接拿舊快取測。若直接拿_cached_yolov8/_cached_yolov10去測v11，會踩到
這個專題最早發現的同一個雷(訓練/測試偵測器不一致，結果不公平)。這裡改用
torch.hub本機載入yolov5(跟新版ultralytics.YOLO()不相容，見
benchmark_yolo_detector_speed.py的踩坑記錄)，重製一份跟v11訓練條件一致的
小規模快取，先求快、不求完整16位說話者規模。

輸出：data/{speaker}_cached_v5_quick/{file}.npy，全新後綴，不覆蓋任何既有檔案。

用法：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/regenerate_cache_v5_quick.py
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

SPEAKER_DIRS = ['s99_8', 's34_3']  # 快速測試，只測2位(100+199支影片)
CACHE_SUFFIX = '_cached_v5_quick'
CONF_THRESHOLD = 0.5
input_shape = (75, 46, 140, 1)

import cv2
import numpy as np
import torch

print('載入原始 YOLOv5（torch.hub本機載入，跟v11訓練時同一顆權重）...')
YOLOV5_REPO = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5'
YOLOV5_WEIGHTS = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5\runs\v5s\weights\best.pt'
yolo_model = torch.hub.load(YOLOV5_REPO, 'custom', path=YOLOV5_WEIGHTS, source='local')
yolo_model.conf = CONF_THRESHOLD
print('載入完成！\n')

t_start = time.time()
for spk in SPEAKER_DIRS:
    cache_dir = os.path.join('data', f'{spk}{CACHE_SUFFIX}')
    os.makedirs(cache_dir, exist_ok=True)
    videos = sorted(glob.glob(os.path.join('data', spk, '*.mpg')))
    print(f'=== {spk}（{len(videos)}支影片）===', flush=True)

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
            det = results.xyxy[0]  # tensor: [x1,y1,x2,y2,conf,cls]
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

        if vi % 50 == 0:
            print(f'  {spk}: {vi}/{len(videos)}', flush=True)

    print(f'  {spk} 完成', flush=True)

print(f'\n全部完成，總耗時 {(time.time()-t_start)/60:.1f} 分鐘')
