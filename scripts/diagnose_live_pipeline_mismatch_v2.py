"""
根因排查：LipNetConnect1.py的即時攝影機辨識管線(process_frame)在headless模擬測試
(test_sliding_window_pipeline.py)中，對s6的4支測試影片全部辨識失敗，但用訓練用的
離線快取(data/s6_cached/*.npy)跑同一支影片卻100%正確——證實不是模型問題，是
「即時管線的前處理」跟「訓練資料產生管線」不一致造成的。

兩條管線已知的差異（比對 LipNetConnect1.py vs train_grid_multi.py）：
  1. 偵測器不同：即時管線用YOLOv8(yolov8_lip/lip_detect)，訓練快取用YOLOv5(yolov5s)
  2. 信心度門檻不同：即時0.1，訓練0.5
  3. 即時管線多了Haar人臉偵測先框一次臉、YOLO只在臉部範圍內找；訓練管線直接對全幀跑YOLO
  4. 即時管線用box_history(10幀滑動平均)平滑座標；訓練管線每幀獨立偵測，無平滑
  5. resize方式不同：即時用tf.image.resize(RGB)，訓練用cv2.resize(BGR)+之後轉灰階

這支腳本逐一測試「只改回YOLOv5+訓練門檻，其他跟即時管線一樣(含face-cascade/box平滑)」
是否就能修復辨識結果，藉此定位問題主因是不是偵測器本身。

執行方式：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/diagnose_live_pipeline_mismatch.py
"""
import os
import sys
import time
import warnings
import pathlib

pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)
sys.path.insert(0, r'C:\Users\Tno\claude-code')
from vocab_correction import correct_sentence

import tensorflow as tf
import cv2
import numpy as np
import torch
from collections import deque
from tensorflow.keras.models import load_model

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except Exception:
    pass

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)


print('載入 LipNet 模型...')
lipnet_model = load_model(os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_width05_datafix.h5'),
                           custom_objects={'CTCLoss': CTCLoss}, compile=False)

print('載入 YOLOv5（跟訓練快取產生時同一顆權重）...')
yolov5s = torch.hub.load(
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5', 'custom',
    path=r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5\runs\v5s\weights\best.pt', source='local')
print('載入完成！\n')


def load_ground_truth(video_path):
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    project_name = os.path.basename(os.path.dirname(video_path))
    align_path = os.path.join('data', 'alignments', project_name, f'{file_name}.align')
    words = []
    with open(align_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[2] not in ('sil', 'sp'):
                words.append(parts[2])
    return ' '.join(words)


def run_inference_on_frames(frames_bgr_list):
    """frames_bgr_list: 75張(46,140)灰階numpy array，比照train_grid_multi.py的產生方式"""
    T = 75
    frames = frames_bgr_list[:T]
    if not frames:
        return "(無偵測到任何幀)"
    while len(frames) < T:
        frames.append(frames[-1])
    arr = np.expand_dims(np.array(frames, dtype=np.float32), -1)
    mean, std = np.mean(arr), np.std(arr) + 1e-6
    arr = ((arr - mean) / std).astype(np.float32)
    x = np.expand_dims(arr, 0)
    yhat = lipnet_model.predict(x, verbose=0)
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
    raw = tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode()
    return correct_sentence(raw.strip())


def process_video_yolov5_style(video_path):
    """完全比照train_grid_multi.py的偵測方式：YOLOv5、conf>=0.5、無臉部偵測、無box平滑、
    每幀獨立偵測(不用前一幀的框)，cv2.resize+BGR2GRAY，用來跟即時管線(YOLOv8+多重機制)對照。"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 這裡刻意完全複製train_grid_multi.py原始寫法，不加break：
        # 如果同一幀有多個信心度>=0.5的偵測框，會append多筆(等於重複這幀)，
        # 這是原始程式碼的實際行為，不是bug，先如實重現看看是否是造成
        # 前面對照實驗殘餘誤差的原因
        detections = yolov5s(frame)
        for det in detections.pred:
            for *xyxy, conf, cls in det:
                if int(cls) == 0 and conf >= 0.5:
                    x1, y1, x2, y2 = map(int, xyxy)
                    lip = frame[y1:y2, x1:x2]
                    if lip.size == 0:
                        continue
                    lip = cv2.resize(lip, (140, 46))
                    lip = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
                    frames.append(lip.astype(np.float32))
    cap.release()
    return run_inference_on_frames(frames)


videos = [
    os.path.join('data', 's6', 'bbae7n.mpg'),
    os.path.join('data', 's6', 'bbae8s.mpg'),
]

for vp in videos:
    gt = load_ground_truth(vp)
    print(f'=== {vp} ===')
    print(f'  正解: "{gt}"')
    t0 = time.time()
    pred = process_video_yolov5_style(vp)
    dt = time.time() - t0
    match = "✅ 一致" if pred.strip() == gt.strip() else "❌ 不一致"
    print(f'  YOLOv5+訓練門檻(比照train_grid_multi.py) 預測: "{pred}"  {match}  ({dt:.1f}s)')
    print()

print('=' * 70)
print('若上面YOLOv5版本能正確辨識（對照之前headless測試YOLOv8版本全部失敗），')
print('代表根因確認是「即時管線用YOLOv8、訓練快取用YOLOv5，兩個偵測器框出的')
print('嘴唇區域不一致」，而非box平滑或人臉偵測等其他因素。')
print('=' * 70)
