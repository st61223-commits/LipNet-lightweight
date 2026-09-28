"""
完整版：用原始 YOLOv5 偵測器重製 v11 benchmark 需要的 13 個說話者資料夾快取（共7885支影片），
目的是公平測 v11(學長基準模型) 的 WER 正確率。
沿用 regenerate_cache_v5_quick.py 的做法，只是擴大到完整 SPEAKERS 清單。
輸出：data/{speaker}_cached_v5_full/{file}.npy
"""
import os, glob, time, pathlib, warnings
pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)

SPEAKER_DIRS = ['s99_1', 's99_6', 's99_6_new', 's99_7', 's99_7_new', 's99_8',
                's34_3', 's1', 's2', 's5', 's6', 's7', 's13']
CACHE_SUFFIX = '_cached_v5_full'
CONF_THRESHOLD = 0.5
input_shape = (75, 46, 140, 1)

import cv2
import numpy as np
import torch

print('載入原始 YOLOv5（torch.hub本機載入，跟v11訓練時同一顆權重）...', flush=True)
YOLOV5_REPO = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5'
YOLOV5_WEIGHTS = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5\runs\v5s\weights\best.pt'
yolo_model = torch.hub.load(YOLOV5_REPO, 'custom', path=YOLOV5_WEIGHTS, source='local')
yolo_model.conf = CONF_THRESHOLD
print('載入完成！\n', flush=True)

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

        if vi % 100 == 0:
            elapsed = (time.time()-t_start)/60
            print(f'  {spk}: {vi}/{len(videos)}  (累計耗時 {elapsed:.1f} 分鐘)', flush=True)

    print(f'  {spk} 完成', flush=True)

print(f'\n全部完成，總耗時 {(time.time()-t_start)/60:.1f} 分鐘')
