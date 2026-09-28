"""
深入排查「at」介係詞誤判問題：比對官方快取的像素資料，跟「重新用完全相同邏輯
再跑一次」得到的像素資料，是否真的逐位元組一致。

背景：diagnose_resize_mismatch.py已經證實「完全比照regenerate_cache_yolov8.py
的cv2/BGR邏輯」重新跑，還是得不到跟官方快取一致的高正確率(先前測試對象剛好是
其他難例影片)。這支腳本换成直接比較bar5n(唯一在8支影片測試中只錯1個字"at"的
案例)的官方快取 vs 重新運算的結果，逐幀比較像素值，排除是不是cv2.VideoCapture
解碼本身就有一點點不確定性(不同次解碼同一支.mpg，抓到的frame是否真的位元組相同)。

執行方式：
  python diagnose_pixel_level_diff.py
"""
import os
import sys
import pathlib
import warnings

pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)

import cv2
import numpy as np
from ultralytics import YOLO

CONF_THRESHOLD = 0.5
input_shape = (75, 46, 140, 1)

print('載入 YOLOv8...')
yolo_model = YOLO(r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov8_lip\lip_detect\weights\best.pt')
print('載入完成\n')


def crop_video_cv2style(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    boxes = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        results = yolo_model(frame, verbose=False)
        matched = False
        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) >= CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    lip = frame[y1:y2, x1:x2]
                    if lip.size == 0:
                        continue
                    lip = cv2.resize(lip, (input_shape[2], input_shape[1]))
                    lip = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
                    frames.append(lip.astype(np.float32))
                    boxes.append((frame_idx, x1, y1, x2, y2, float(box.conf[0])))
                    matched = True
                    break
            if matched:
                break
    cap.release()
    T = 75
    frames = frames[:T]
    if not frames:
        return None, boxes
    while len(frames) < T:
        frames.append(frames[-1])
    return np.expand_dims(np.array(frames), -1), boxes


video_path = os.path.join('data', 's6', 'bbar5n.mpg')
official_cache = np.load('data/s6_cached_yolov8/bbar5n.npy')

print('重新對 bbar5n.mpg 跑一次一模一樣的偵測/裁切流程...')
recomputed, boxes = crop_video_cv2style(video_path)

print(f'\n官方快取 shape: {official_cache.shape}, dtype: {official_cache.dtype}')
print(f'重新運算 shape: {recomputed.shape}, dtype: {recomputed.dtype}')

diff = np.abs(official_cache.astype(np.float64) - recomputed.astype(np.float64))
n_identical_frames = sum(1 for i in range(75) if np.array_equal(official_cache[i], recomputed[i]))
print(f'\n逐幀完全位元組相同的幀數: {n_identical_frames}/75')
print(f'整體差異: max={diff.max():.4f}, mean={diff.mean():.6f}, 有差異的像素數={np.sum(diff > 0)}/{diff.size}')

if n_identical_frames < 75:
    print('\n找出第一個不同的幀，比較細節：')
    for i in range(75):
        if not np.array_equal(official_cache[i], recomputed[i]):
            d = np.abs(official_cache[i].astype(np.float64) - recomputed[i].astype(np.float64))
            print(f'  frame {i}: max_diff={d.max():.2f}, mean_diff={d.mean():.4f}, 官方box=? 重算box={boxes[i] if i < len(boxes) else "N/A(padding)"}')
            if i >= 3:
                break

print('\n' + '=' * 60)
if n_identical_frames == 75:
    print('結論：官方快取 vs 重新運算 逐位元組完全相同！代表cv2解碼+YOLO偵測本身')
    print('是確定性的(deterministic)，"at"誤判的原因不是解碼不確定性，而是')
    print('即時管線(有人臉偵測+box平滑+低門檻)造成的裁切範圍/內容真的不同。')
else:
    print(f'結論：官方快取 vs 重新運算 有 {75-n_identical_frames} 幀不同！代表就算')
    print('偵測邏輯完全相同，同一支影片重新解碼/偵測仍會得到不同的裁切結果')
    print('(可能是YOLO推論本身有微小的非決定性，或cv2解碼在不同次執行間有差異)。')
print('=' * 60)
