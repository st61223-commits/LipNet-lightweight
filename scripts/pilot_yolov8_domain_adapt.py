"""
小規模先導實驗：模型能不能透過微調適應YOLOv8的裁切風格？

背景：已確認即時管線(YOLOv8)跟訓練快取(YOLOv5)裁切方式不一致，導致實機辨識
大幅失敗（見 project_lipnet.md）。方案2「全部資料用YOLOv8重新產生+重新訓練」
工程量大(21GB快取+數小時)，這支腳本先用小規模、**完全不寫入磁碟**（YOLOv8裁切
出來的畫面只留在記憶體）的方式，快速驗證「模型微調後能不能適應YOLOv8裁切風格」
這個假設，避免大動作之前先確認方向有沒有機會成功。

跟s9個人化實驗（同樣是小樣本微調）的關鍵差異：s9是「換一個全新的人」，是
每個人獨特、難以類化的差異；這裡是「換一種固定的裁切方式」，理論上是所有
影片都適用同一種系統性偏移，應該比「認識新的人」更容易透過微調學起來。

做法：
1. 從s1/s6/s7各抽樣訓練用+測試用影片
2. 用YOLOv8(跟LipNetConnect1.py部署同一顆權重)即時裁切成(75,46,140,1)陣列，
   只存在記憶體，不寫入.npy（磁碟安全）
3. 用YOLOv8裁切的訓練集微調現役datafix模型（全模型解凍、低學習率、
   validation+early stopping）
4. 用YOLOv8裁切的測試集（訓練沒看過的影片）驗證微調後正確率

執行方式：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/pilot_yolov8_domain_adapt.py
"""
import os
import sys
import glob
import time
import pathlib
import warnings
import numpy as np

pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)
sys.path.insert(0, r'C:\Users\Tno\claude-code')
from vocab_correction import correct_sentence

import cv2
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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
model = load_model(os.path.join('models', 'trained_model_grid_multi_width05_datafix.h5'),
                    custom_objects={'CTCLoss': CTCLoss}, compile=False)
yolo_model = YOLO(r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov8_lip\lip_detect\weights\best.pt')
print('載入完成！\n')

CONF_THRESHOLD = 0.5


def yolov8_crop_video(video_path):
    """比照train_grid_multi.py風格：每幀獨立偵測、無平滑、無人臉偵測，只是換成YOLOv8"""
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
                    lip = cv2.resize(lip, (140, 46))
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


def load_alignment(project_name, file_name):
    align_path = os.path.join('data', 'alignments', project_name, f'{file_name}.align')
    tokens = []
    with open(align_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[2] not in ('sil', 'sp') and len(parts[2]) > 1:
                tokens.extend([' ', parts[2]])
    tokens_flat = tokens[1:]
    if not tokens_flat:
        return tf.zeros([1], dtype=tf.int64)
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens_flat, input_encoding='UTF-8'), (-1)))


def build_dataset(video_paths, label):
    xs, ys = [], []
    for i, vp in enumerate(video_paths, 1):
        frames = yolov8_crop_video(vp)
        if frames is None:
            continue
        mean, std = np.mean(frames), np.std(frames) + 1e-6
        frames = ((frames - mean) / std).astype(np.float32)
        project_name = os.path.basename(os.path.dirname(vp))
        file_name = os.path.splitext(os.path.basename(vp))[0]
        align = load_alignment(project_name, file_name)
        xs.append(frames)
        ys.append(align.numpy())
        if i % 20 == 0:
            print(f'  [{label}] {i}/{len(video_paths)} 支已裁切', flush=True)
    return xs, ys


def pad_labels(ys, max_len=40):
    out = np.zeros((len(ys), max_len), dtype=np.int64)
    for i, y in enumerate(ys):
        L = min(len(y), max_len)
        out[i, :L] = y[:L]
    return out


def test_accuracy(mdl, xs, ys_raw):
    ok = 0
    for x, y in zip(xs, ys_raw):
        yhat = mdl.predict(np.expand_dims(x, 0), verbose=0)
        dec = tf.keras.backend.ctc_decode(tf.cast(yhat, tf.float32), [75], greedy=False)[0][0].numpy()
        orig = ' '.join(tf.strings.reduce_join(num_to_char(y)).numpy().decode().strip().split())
        raw = ' '.join(w for w in tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split() if len(w) > 1)
        corr = ' '.join(correct_sentence(raw).split())
        if orig == corr:
            ok += 1
    return ok, len(xs)


# ── 抽樣資料：s1/s6/s7各抽一些，訓練跟測試不重疊
TRAIN_PER_SPEAKER = 40
TEST_PER_SPEAKER = 15
SPEAKERS = ['s1', 's6', 's7']

train_videos, test_videos = [], []
for spk in SPEAKERS:
    files = sorted(glob.glob(os.path.join('data', spk, '*.mpg')))
    train_videos += files[:TRAIN_PER_SPEAKER]
    test_videos += files[TRAIN_PER_SPEAKER:TRAIN_PER_SPEAKER + TEST_PER_SPEAKER]

print(f'訓練集: {len(train_videos)} 支，測試集: {len(test_videos)} 支\n')

t0 = time.time()
print('=== YOLOv8裁切訓練集（記憶體中，不寫入磁碟）===')
train_x, train_y = build_dataset(train_videos, '訓練集')
print(f'\n=== YOLOv8裁切測試集 ===')
test_x, test_y = build_dataset(test_videos, '測試集')
print(f'\n裁切完成，耗時 {(time.time()-t0)/60:.1f} 分鐘\n')

# 微調前先測一次基準（模型從沒看過YOLOv8裁切的畫面，預期很差，比照之前對照實驗發現）
print('=== 微調前基準（模型只看過YOLOv5裁切風格）===')
ok0, n0 = test_accuracy(model, test_x, test_y)
print(f'微調前: {ok0}/{n0} = {ok0/n0*100:.1f}%\n')

# 切一部分訓練集當validation
n_val = max(1, len(train_x) // 6)
val_x, val_y = train_x[-n_val:], train_y[-n_val:]
fit_x, fit_y = train_x[:-n_val], train_y[:-n_val]

fit_x_arr = np.stack(fit_x)
val_x_arr = np.stack(val_x)
fit_y_arr = pad_labels(fit_y)
val_y_arr = pad_labels(val_y)

for layer in model.layers:
    layer.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5, clipnorm=1.0), loss=CTCLoss)

print(f'=== 開始微調（全模型解凍,lr=1e-5）：{len(fit_x)}筆訓練+{len(val_x)}筆驗證，最多30輪，early stop patience=5 ===')
callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)]
history = model.fit(fit_x_arr, fit_y_arr, validation_data=(val_x_arr, val_y_arr),
                     batch_size=2, epochs=30, verbose=2, callbacks=callbacks)

best_val = min(history.history['val_loss'])
print(f'\n最佳 val_loss: {best_val:.4f}')

print('\n=== 微調後：測試集（全部沒在訓練/驗證用過）===')
ok1, n1 = test_accuracy(model, test_x, test_y)
pct1 = ok1 / n1 * 100 if n1 else 0
print(f'微調後: {ok1}/{n1} = {pct1:.1f}%')

print('\n' + '=' * 60)
print(f'結論pilot：YOLOv8裁切風格適應微調 → 測試集正確率從 {ok0/n0*100:.1f}% 提升到 {pct1:.1f}%')
if pct1 > ok0 / n0 * 100 + 20:
    print('顯著進步！代表方案2(全部用YOLOv8重新產生+重新訓練)這個方向大有可為')
elif pct1 > ok0 / n0 * 100:
    print('有進步但幅度不大，可能需要更多資料/更多輪次才能看到方案2的完整效果')
else:
    print('沒有明顯進步，需要進一步排查（可能是取樣資料量還是太小，60+15support有限）')
print('=' * 60)
