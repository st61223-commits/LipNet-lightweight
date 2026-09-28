"""
無GUI/無攝影機環境下，用一支真實測試影片模擬完整的通話系統辨識邏輯
（人臉偵測→YOLO嘴唇偵測→box平滑→緩衝區→滑動視窗提早預測→CTC解碼），
邏輯完全比照 LipNetConnect1.py 的 process_frame，只是把攝影機來源換成
讀取現成的.mpg檔逐幀餵進去，藉此在沒有真實硬體的情況下驗證整條pipeline
是否會crash、提早預測的文字有沒有隨著幀數增加逐漸收斂到正確答案。

用法：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/test_sliding_window_pipeline.py [影片路徑]
  不給參數預設用 data/s6/bbae7n.mpg（正解："bin blue at e seven now"）
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
from collections import deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model

SLIDE_STEP = 5
MIN_FRAMES = 15

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


import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('video', nargs='?', default=os.path.join('data', 's6', 'bbae7n.mpg'))
_parser.add_argument('--model', default='trained_model_grid_multi_width05_datafix.h5',
                      help='models/ 資料夾內的模型檔名（不含路徑），方便測試不同版本，例如 trained_model_grid_multi_width05_yolov8.h5')
_args = _parser.parse_args()

H5_MODEL_PATH = os.path.join(BASE_DIR, 'models', _args.model)

print('載入模型與偵測器...')
lipnet_model = load_model(H5_MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)
yolo_model = YOLO(r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov8_lip\lip_detect\weights\best.pt')
face_cascade = cv2.CascadeClassifier(r'C:\Users\Tno\miniconda3\envs\py39\Library\etc\haarcascades\haarcascade_frontalface_default.xml')
print('載入完成！\n')

input_shape = (75, 46, 140, 1)


def preprocess_lip_shape(lip_region):
    return tf.image.resize(lip_region, (input_shape[1], input_shape[2]))


def preprocess_lip_std(lip_regionf):
    mean = tf.math.reduce_mean(lip_regionf)
    std = tf.math.reduce_std(tf.cast(lip_regionf, tf.float32))
    return tf.cast((lip_regionf - mean), tf.float32) / std


def run_inference(frames_list):
    lip_region_processed = preprocess_lip_std(frames_list)
    x = tf.expand_dims(lip_region_processed, axis=0)
    yhat = lipnet_model.predict(x, verbose=0)
    # greedy=False(beam search)+過濾單一字母，跟官方benchmark腳本的解碼方式一致
    # （2026-09-03修正，原本用greedy=True且沒過濾單字母，會跟正解對不齊）
    decoded = tf.keras.backend.ctc_decode(tf.cast(yhat, tf.float32), input_length=[75], greedy=False)[0][0].numpy()
    predicted_texts = [tf.strings.reduce_join([num_to_char(word) for word in sentence]).numpy().decode('utf-8') for sentence in decoded]
    raw_result = " ".join(w for w in " ".join(predicted_texts).split() if len(w) > 1) if predicted_texts else "No prediction"
    return correct_sentence(raw_result)


def load_ground_truth(video_path):
    # 2026-09-03踩坑修正：官方train_grid_multi*.py/benchmark*.py的load_alignments
    # 用 len(word) > 1 把單一字母(GRID文法的letter欄位，例如"e"/"l"/"f")從正解跟
    # 預測裡都濾掉，模型從來沒被要求要答對這個字。這支腳本原本沒套用同樣的過濾，
    # 導致正解多了一個字，讓原本答對的被誤判成答錯，一度誤以為即時管線完全失效。
    # 這裡補上跟官方一致的過濾規則，才能公平比對。
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    project_name = os.path.basename(os.path.dirname(video_path))
    align_path = os.path.join('data', 'alignments', project_name, f'{file_name}.align')
    if not os.path.exists(align_path):
        return None
    words = []
    with open(align_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[2] not in ('sil', 'sp') and len(parts[2]) > 1:
                words.append(parts[2])
    return ' '.join(words)


video_path = _args.video
GROUND_TRUTH = load_ground_truth(video_path) or "(找不到對齊檔)"
print(f'測試模型: {_args.model}')
print(f'測試影片: {video_path}')
print(f'（若用預設影片，正解應為: "{GROUND_TRUTH}"）\n')

cap = cv2.VideoCapture(video_path)

lip_regions = deque(maxlen=75)
box_history = deque(maxlen=1)  # 2026-09-03修正：比照LipNetConnect1.py拿掉box平滑
no_detect_count = 0
frame_counter = 0
n_predictions = 0
errors = 0
t_start = time.time()

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_counter += 1
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # process_frame預期輸入是RGB(跟VideoDisplay吐出的一致)

    try:
        frame_bgr2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 2026-09-03修正：比照LipNetConnect1.py拿掉人臉偵測預篩選，直接對全畫面
        # 做YOLO偵測（見project_lipnet.md / 實驗記錄.md「實驗七」的診斷實驗）
        detected = False
        results = yolo_model(frame_bgr2, verbose=False)
        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) >= 0.1:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_history.append((x1, y1, x2, y2))
                    detected = True
                    break
            if detected:
                break

        if detected:
            no_detect_count = 0
        else:
            no_detect_count += 1

        if len(box_history) > 0 and no_detect_count <= 20:
            x1 = int(sum(b[0] for b in box_history) / len(box_history))
            y1 = int(sum(b[1] for b in box_history) / len(box_history))
            x2 = int(sum(b[2] for b in box_history) / len(box_history))
            y2 = int(sum(b[3] for b in box_history) / len(box_history))
            lip_region = frame[y1:y2, x1:x2]
            if lip_region.size > 0:
                lip_region = preprocess_lip_shape(lip_region)
                lip_region = tf.image.rgb_to_grayscale(lip_region)
                lip_regions.append(lip_region)

        if len(lip_regions) >= MIN_FRAMES and frame_counter % SLIDE_STEP == 0:
            frames_list = list(lip_regions)
            is_early = len(frames_list) < 75
            if is_early:
                pad_count = 75 - len(frames_list)
                frames_list = frames_list + [frames_list[-1]] * pad_count
            mode_tag = f"提早預測 {len(lip_regions)}/75" if is_early else "完整75幀"

            t0 = time.time()
            text = run_inference(frames_list)
            dt = time.time() - t0
            n_predictions += 1
            match = "✅" if text.strip() == GROUND_TRUTH else "  "
            print(f'  frame={frame_counter:3d} [{mode_tag:14s}] ({dt*1000:6.1f}ms) → "{text}" {match}')

    except Exception as e:
        import traceback
        errors += 1
        print(f'  [例外] frame={frame_counter}: {type(e).__name__}: {e}')
        traceback.print_exc()

cap.release()
total_dt = time.time() - t_start

print('\n' + '=' * 70)
print(f'總幀數: {frame_counter}，總預測次數: {n_predictions}，例外次數: {errors}')
print(f'總耗時: {total_dt:.1f}秒')
if errors == 0:
    print('結論：整條pipeline跑完全程沒有crash，滑動視窗+提早預測機制運作正常')
else:
    print(f'結論：跑完但出現 {errors} 次例外，需要進一步排查')
print('=' * 70)
