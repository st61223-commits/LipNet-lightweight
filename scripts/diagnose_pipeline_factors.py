"""
拆解即時管線的三個額外機制（人臉偵測預篩選／box_history 10幀平滑／信心度門檻），
逐一開關測試，找出究竟是哪個因素造成 width05_yolov8 模型在完整即時管線下辨識失敗。

背景：width05_yolov8模型在「乾淨YOLOv8裁切」(regenerate_cache_yolov8.py風格：
無人臉偵測、無平滑、conf>=0.5)的benchmark上s6有95.0%；但用完整即時管線
(LipNetConnect1.py風格：人臉偵測+YOLO+10幀平滑+conf>=0.1)跑同樣4支測試影片
卻全部辨識錯誤。這支腳本把「人臉偵測」「平滑」「信心度門檻」三個變因獨立拆開，
每次只開一個，看單獨開哪個會讓正確率掉下去，藉此定位真正的元兇。

用法：
  python diagnose_pipeline_factors.py --face-cascade --conf 0.1          # 只開人臉偵測
  python diagnose_pipeline_factors.py --smoothing --conf 0.1             # 只開平滑
  python diagnose_pipeline_factors.py --conf 0.1                         # 只調門檻(其餘都關)
  python diagnose_pipeline_factors.py --face-cascade --smoothing --conf 0.1   # 全開(=完整即時管線)
  不加任何旗標 --conf預設0.5 = 訓練資料產生時的簡單風格(基準對照組)
"""
import os
import sys
import time
import warnings
import pathlib
import argparse

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


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='trained_model_grid_multi_width05_yolov8.h5')
parser.add_argument('--face-cascade', action='store_true', help='開啟Haar人臉偵測預篩選')
parser.add_argument('--smoothing', action='store_true', help='開啟10幀box_history平滑')
parser.add_argument('--conf', type=float, default=0.5, help='YOLO信心度門檻')
parser.add_argument('--videos', nargs='*', default=[
    'bbae7n', 'bbafza', 'bbal1n', 'bbal2s', 'bbal3p', 'bbal4a', 'bbar5n', 'bbar6s'])  # 8支width05_yolov8官方快取答對的影片
args = parser.parse_args()

label = f"face_cascade={args.face_cascade} smoothing={args.smoothing} conf={args.conf}"
print(f'=== 設定: {label} ===\n')

H5_MODEL_PATH = os.path.join(BASE_DIR, 'models', args.model)
lipnet_model = load_model(H5_MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)
yolo_model = YOLO(r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov8_lip\lip_detect\weights\best.pt')
face_cascade = cv2.CascadeClassifier(r'C:\Users\Tno\miniconda3\envs\py39\Library\etc\haarcascades\haarcascade_frontalface_default.xml')

input_shape = (75, 46, 140, 1)


def preprocess_lip_shape(lip_region):
    return tf.image.resize(lip_region, (input_shape[1], input_shape[2]))


def preprocess_lip_std(lip_regionf):
    mean = tf.math.reduce_mean(lip_regionf)
    std = tf.math.reduce_std(tf.cast(lip_regionf, tf.float32))
    return tf.cast((lip_regionf - mean), tf.float32) / std


def run_inference(frames_list):
    # 2026-09-03修正：套用官方評分規則(過濾單一字母+beam search解碼)，
    # 跟test_sliding_window_pipeline.py同樣的修正，避免重蹈評分bug覆轍
    lip_region_processed = preprocess_lip_std(frames_list)
    x = tf.expand_dims(lip_region_processed, axis=0)
    yhat = lipnet_model.predict(x, verbose=0)
    decoded = tf.keras.backend.ctc_decode(tf.cast(yhat, tf.float32), input_length=[75], greedy=False)[0][0].numpy()
    predicted_texts = [tf.strings.reduce_join([num_to_char(word) for word in sentence]).numpy().decode('utf-8') for sentence in decoded]
    raw_result = " ".join(w for w in " ".join(predicted_texts).split() if len(w) > 1) if predicted_texts else "No prediction"
    return correct_sentence(raw_result)


def load_ground_truth(video_path):
    # 2026-09-03修正：套用官方評分規則，過濾單一字母(len(word)>1)，
    # 避免正解含有模型從未被要求答對的字，見project_lipnet.md的踩坑記錄
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    project_name = os.path.basename(os.path.dirname(video_path))
    align_path = os.path.join('data', 'alignments', project_name, f'{file_name}.align')
    words = []
    with open(align_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[2] not in ('sil', 'sp') and len(parts[2]) > 1:
                words.append(parts[2])
    return ' '.join(words)


def process_video(video_path):
    GROUND_TRUTH = load_ground_truth(video_path)
    cap = cv2.VideoCapture(video_path)

    lip_regions = deque(maxlen=75)
    box_history = deque(maxlen=10 if args.smoothing else 1)  # 平滑關掉時只留最新1筆=不平滑
    no_detect_count = 0
    frame_counter = 0
    final_pred = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_counter += 1
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_bgr2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        detected = False
        if args.face_cascade:
            gray = cv2.cvtColor(frame_bgr2, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                pad = int(fh * 0.1)
                fx1 = max(0, fx - pad); fy1 = max(0, fy - pad)
                fx2 = min(frame.shape[1], fx + fw + pad); fy2 = min(frame.shape[0], fy + fh + pad)
                face_crop = frame_bgr2[fy1:fy2, fx1:fx2]
                results = yolo_model(face_crop, verbose=False)
                for r in results:
                    for box in r.boxes:
                        if float(box.conf[0]) >= args.conf:
                            lx1, ly1, lx2, ly2 = map(int, box.xyxy[0])
                            x1, y1, x2, y2 = fx1 + lx1, fy1 + ly1, fx1 + lx2, fy1 + ly2
                            box_history.append((x1, y1, x2, y2))
                            detected = True
                            break
                    if detected:
                        break

        if not detected:
            results = yolo_model(frame_bgr2, verbose=False)
            for r in results:
                for box in r.boxes:
                    if float(box.conf[0]) >= args.conf:
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
            if len(frames_list) < 75:
                frames_list = frames_list + [frames_list[-1]] * (75 - len(frames_list))
            final_pred = run_inference(frames_list)

    cap.release()
    match = final_pred is not None and final_pred.strip() == GROUND_TRUTH.strip()
    return GROUND_TRUTH, final_pred, match


results = []
for name in args.videos:
    vp = os.path.join('data', 's6', f'{name}.mpg')
    gt, pred, match = process_video(vp)
    mark = "✅" if match else "❌"
    print(f'  {name}: 正解="{gt}"  預測="{pred}"  {mark}')
    results.append(match)

n_correct = sum(results)
print(f'\n=== 結果: {label} → {n_correct}/{len(results)} 正確 ===')
