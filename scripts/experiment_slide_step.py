"""
測試不同 SLIDE_STEP（滑動視窗每隔幾幀重新推論一次）對兩件事的影響：
1. CPU總運算負擔——SLIDE_STEP越大，重新推論次數越少，總耗時越低
2. 字幕反應速度——SLIDE_STEP越大，多久才能"收斂"到最終正確答案的間隔會拉長

背景：現行LipNetConnect1.py用SLIDE_STEP=5，每次都把完整75幀重新丟進模型算一次，
即使新增的只有5幀。這支腳本比照test_sliding_window_pipeline.py已修正的管線邏輯
(無人臉偵測預篩選、box_history不平滑、官方評分規則)，掃過幾組候選SLIDE_STEP值，
量化「調大SLIDE_STEP換取更少運算量」的實際代價，幫忙找一個平衡點。

用法：
  CUDA_VISIBLE_DEVICES=-1 /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/experiment_slide_step.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 強制CPU，模擬低階硬體情境(跟目標硬體方向一致)
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

MIN_FRAMES = 15
SLIDE_STEPS = [5, 10, 15, 25]
VIDEOS = ['bbae7n', 'bbafza', 'bbal1n', 'bbal2s', 'bbal3p', 'bbal4a', 'bbar5n', 'bbar6s']

print(f'可見GPU數量: {len(tf.config.list_physical_devices("GPU"))}（應為0=強制CPU）\n')

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)


print('載入模型與偵測器（整個實驗只載入一次，跨SLIDE_STEP/影片重複使用）...')
H5_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_width05_yolov8.h5')
lipnet_model = load_model(H5_MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)
yolo_model = YOLO(r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov8_lip\lip_detect\weights\best.pt')
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
    decoded = tf.keras.backend.ctc_decode(tf.cast(yhat, tf.float32), input_length=[75], greedy=False)[0][0].numpy()
    predicted_texts = [tf.strings.reduce_join([num_to_char(word) for word in sentence]).numpy().decode('utf-8') for sentence in decoded]
    raw_result = " ".join(w for w in " ".join(predicted_texts).split() if len(w) > 1) if predicted_texts else "No prediction"
    return correct_sentence(raw_result)


def load_ground_truth(video_path):
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


# 先把每支影片的原始幀+ground truth快取起來，避免每個SLIDE_STEP都重新解碼影片浪費時間
print('預先讀取8支影片的所有幀（跨SLIDE_STEP重複使用，避免重複解碼影片）...')
video_frames_cache = {}
video_gt_cache = {}
for name in VIDEOS:
    vp = os.path.join('data', 's6', f'{name}.mpg')
    cap = cv2.VideoCapture(vp)
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    video_frames_cache[name] = frames
    video_gt_cache[name] = load_ground_truth(vp)
print('讀取完成！\n')


def run_one(name, slide_step):
    frames = video_frames_cache[name]
    gt = video_gt_cache[name]
    lip_regions = deque(maxlen=75)
    box_history = deque(maxlen=1)
    no_detect_count = 0
    n_infer = 0
    infer_time = 0.0
    final_pred = None
    first_stable_frame = None  # 第一個「之後都不再變」且等於最終預測的frame
    pred_log = []  # [(frame_counter, text)]

    for frame_counter, frame in enumerate(frames, start=1):
        frame_bgr2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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

        if len(lip_regions) >= MIN_FRAMES and frame_counter % slide_step == 0:
            frames_list = list(lip_regions)
            if len(frames_list) < 75:
                frames_list = frames_list + [frames_list[-1]] * (75 - len(frames_list))
            t0 = time.time()
            text = run_inference(frames_list)
            infer_time += time.time() - t0
            n_infer += 1
            final_pred = text
            pred_log.append((frame_counter, text))

    # 從後往前找「第一個之後都不再變」的frame（=收斂點），只在最終預測正確時才有意義
    if pred_log:
        last_text = pred_log[-1][1]
        first_stable_frame = pred_log[-1][0]
        for fc, text in reversed(pred_log):
            if text == last_text:
                first_stable_frame = fc
            else:
                break

    match = final_pred is not None and final_pred.strip() == gt.strip()
    return {
        'n_infer': n_infer,
        'infer_time': infer_time,
        'match': match,
        'first_stable_frame': first_stable_frame if match else None,
    }


print('=' * 78)
print(f'{"SLIDE_STEP":>10s} | {"總推論次數":>10s} | {"總推論耗時":>12s} | {"正確率":>8s} | {"平均收斂幀數":>12s}')
print('-' * 78)

all_results = {}
for slide_step in SLIDE_STEPS:
    total_infer = 0
    total_time = 0.0
    n_correct = 0
    stable_frames = []
    for name in VIDEOS:
        r = run_one(name, slide_step)
        total_infer += r['n_infer']
        total_time += r['infer_time']
        if r['match']:
            n_correct += 1
            if r['first_stable_frame'] is not None:
                stable_frames.append(r['first_stable_frame'])
    avg_stable = (sum(stable_frames) / len(stable_frames)) if stable_frames else float('nan')
    all_results[slide_step] = {
        'total_infer': total_infer, 'total_time': total_time,
        'n_correct': n_correct, 'avg_stable': avg_stable,
    }
    acc_str = f'{n_correct}/{len(VIDEOS)}'
    print(f'{slide_step:>10d} | {total_infer:>10d} | {total_time:>10.2f}s | '
          f'{acc_str:>8s} | {avg_stable:>10.1f}幀')

print('=' * 78)
print('\n說明：')
print('- 總推論次數/總推論耗時：8支影片加總，數字越小代表CPU負擔越低')
print('- 正確率：最終字幕是否等於正解（8支影片中答對幾支）')
print('- 平均收斂幀數：只算答對的影片，從影片開始算起，第幾幀之後字幕就不再變動、')
print('  穩定停在最終正確答案（幀數越小代表反應越快）；75幀=3秒，所以幀數/25=收斂所需秒數')
