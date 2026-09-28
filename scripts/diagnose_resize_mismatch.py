"""
排查另一層可能的差異：resize/色彩空間處理方式。

發現：diagnose_pipeline_factors.py的「基準組」(face_cascade=False, smoothing=False,
conf=0.5，理論上完全比照regenerate_cache_yolov8.py的偵測邏輯)在4支測試影片上
還是0/4全錯，但同一顆模型在benchmark_width05_yolov8.py(用regenerate_cache_yolov8.py
產生的_cached_yolov8快取)上s6有95.0%正確率。矛盾！代表「偵測邏輯」以外，還有
別的環節沒有真正比照，最可疑的是：diagnose_pipeline_factors.py沿用了
test_sliding_window_pipeline.py的寫法——frame轉成RGB、用tf.image.resize縮放、
用tf.image.rgb_to_grayscale轉灰階；但regenerate_cache_yolov8.py(產生真正訓練
資料那支)其實是直接對BGR frame用cv2.resize縮放、cv2.cvtColor(BGR2GRAY)轉灰階，
兩者從沒被證實完全等價。

這支腳本完全複製regenerate_cache_yolov8.py的裁切/resize/灰階邏輯(cv2/BGR)，
搭配一樣「無人臉偵測、無平滑、conf=0.5」的偵測邏輯，測同樣4支影片，
看是否終於能重現benchmark的高正確率。

執行方式：
  python diagnose_resize_mismatch.py
"""
import os
import sys
import pathlib
import warnings

pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)
sys.path.insert(0, r'C:\Users\Tno\claude-code')
from vocab_correction import correct_sentence

import tensorflow as tf
import cv2
import numpy as np
from ultralytics import YOLO
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


print('載入模型...')
model = load_model(os.path.join('models', 'trained_model_grid_multi_width05_yolov8.h5'),
                    custom_objects={'CTCLoss': CTCLoss}, compile=False)
yolo_model = YOLO(r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov8_lip\lip_detect\weights\best.pt')
print('載入完成！\n')

input_shape = (75, 46, 140, 1)
CONF_THRESHOLD = 0.5


def crop_video_cv2style(video_path):
    """完全比照regenerate_cache_yolov8.py：cv2.resize + cv2.cvtColor(BGR2GRAY)，
    直接對原始BGR frame操作，不轉RGB。"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
                    matched = True
                    break
            if matched:
                break
    cap.release()
    T = 75
    frames = frames[:T]
    if not frames:
        return None
    while len(frames) < T:
        frames.append(frames[-1])
    return np.expand_dims(np.array(frames), -1)


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


def run_inference(frames):
    mean, std = np.mean(frames), np.std(frames) + 1e-6
    frames = ((frames - mean) / std).astype(np.float32)
    x = np.expand_dims(frames, 0)
    yhat = model.predict(x, verbose=0)
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
    raw = tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode()
    return correct_sentence(raw.strip())


results = []
for name in ['bbae7n', 'bbae8s', 'bbae9p', 'bbafza']:
    vp = os.path.join('data', 's6', f'{name}.mpg')
    gt = load_ground_truth(vp)
    frames = crop_video_cv2style(vp)
    pred = run_inference(frames) if frames is not None else "(無偵測)"
    match = pred.strip() == gt.strip()
    mark = "✅" if match else "❌"
    print(f'  {name}: 正解="{gt}"  預測="{pred}"  {mark}')
    results.append(match)

n_correct = sum(results)
print(f'\n=== 結果(cv2/BGR風格，完全比照regenerate_cache_yolov8.py): {n_correct}/4 正確 ===')
print('若這裡正確率明顯回升，證實問題是resize/色彩空間處理方式(cv2 vs tf.image)不一致，')
print('不是人臉偵測/平滑/信心度門檻。')
