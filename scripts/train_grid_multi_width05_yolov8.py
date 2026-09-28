"""
【方案2「根本修復」的訓練腳本，尚未執行，等 regenerate_cache_yolov8.py 跑完才能用】
F_Width0.5 改用YOLOv8裁切資料重新訓練（暖啟動，接續現役datafix模型繼續訓練）

背景：2026-09-02凌晨發現即時通話系統用YOLOv8裁切嘴唇，但全部訓練資料是用
YOLOv5裁切產生的快取，兩者裁切風格不一致，導致即時辨識大幅失準（詳見
project_lipnet.md、實驗記錄.md「實驗六第三階段」）。小規模先導實驗
（scripts/pilot_yolov8_domain_adapt.py，100筆資料+30輪微調）已證實模型能
透過微調適應YOLOv8裁切風格（正確率24.4%→31.1%，val_loss持續下降未收斂），
這支腳本是把先導實驗的做法擴大到完整16位說話者資料集的正式訓練版本。

**⚠️ 執行前必讀**：
1. 這支腳本讀取 `data/{speaker}_cached_yolov8/` 快取，必須先執行
   `scripts\\regenerate_cache_yolov8.py` 產生這些快取才能跑（腳本開頭有
   防呆檢查，快取不存在會直接報錯，不會靜默訓練在錯誤/空白的資料上）
2. 暖啟動起點是現役的 width05_datafix.h5（含資料修復的最佳模型），不是從零開始

做法：跟 train_grid_multi_width05_datafix.py 同樣模式——暖啟動、同樣16位
說話者（s9依然排除，留作乾淨的held-out測試）、同樣的resume/checkpoint機制，
只差讀取快取的資料夾後綴從 `_cached` 換成 `_cached_yolov8`。

存檔用新檔名（不覆蓋現役模型），要先用 benchmark 驗證真的有進步、且用
scripts/test_sliding_window_pipeline.py 模擬即時管線確認辨識恢復正常，
才手動替換現役模型、更新通話系統。

執行方式（WSL）：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/train_grid_multi_width05_yolov8.py
"""

import os
import sys
import json
import time
import glob
import pathlib
import warnings
import numpy as np
import tensorflow as tf

pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional,
    MaxPool3D, Activation, TimeDistributed, Flatten,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

BASE_DIR      = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
CURRENT_MODEL = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_width05_datafix.h5')
SAVE_PATH     = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_width05_yolov8.h5')

RESUME_WEIGHTS = os.path.join(BASE_DIR, 'models', 'checkpoint_width05_yolov8_resume.weights.h5')
STATE_FILE     = os.path.join(BASE_DIR, 'models', 'train_width05_yolov8_state.json')
DONE_FLAG      = os.path.join(BASE_DIR, 'models', 'train_width05_yolov8_DONE.flag')

CACHE_SUFFIX = '_cached_yolov8'  # 跟 regenerate_cache_yolov8.py 產生的資料夾後綴一致

os.chdir(BASE_DIR)

# ── 資料集：跟 width05_datafix 同樣16位說話者，s9依然排除(留當held-out測試)
SPEAKER_DIRS = [
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's13',
    's99_1', 's99_6', 's99_6_new', 's99_7', 's99_7_new', 's99_8', 's34_3',
]
all_patterns = [f'data\\{spk}\\*.mpg' for spk in SPEAKER_DIRS]

# ── 前置防呆：確認 _cached_yolov8 快取真的存在，避免在缺資料的情況下靜默訓練
missing = [spk for spk in SPEAKER_DIRS if not os.path.isdir(os.path.join('data', f'{spk}{CACHE_SUFFIX}'))]
if missing:
    print(f'[中止] 以下說話者還沒有 {CACHE_SUFFIX} 快取: {missing}')
    print(f'請先執行 scripts\\regenerate_cache_yolov8.py 產生快取後再跑這支訓練腳本。')
    sys.exit(1)

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
VOCAB_SIZE = char_to_num.vocabulary_size()


def load_video(path: str) -> np.ndarray:
    project_name = os.path.basename(os.path.dirname(path))
    file_name = os.path.splitext(os.path.basename(path))[0]
    cache_path = os.path.join('data', f'{project_name}{CACHE_SUFFIX}', f'{file_name}.npy')
    if os.path.exists(cache_path):
        frames = np.load(cache_path)
        mean = np.mean(frames)
        std = np.std(frames) + 1e-6
        return ((frames - mean) / std).astype(np.float32)
    raise FileNotFoundError(f'找不到YOLOv8快取：{project_name}/{file_name}（記得先跑regenerate_cache_yolov8.py）')


def load_alignments(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens.extend([' ', line[2]])
    tokens_flat = tokens[1:]
    return char_to_num(tf.reshape(
        tf.strings.unicode_split(tokens_flat, input_encoding='UTF-8'), (-1)
    ))


def load_data(path):
    try:
        path = bytes.decode(path.numpy())
        file_name = os.path.splitext(os.path.basename(path))[0]
        parts = path.replace('\\\\', '/').replace('\\', '/').split('/')
        parts = [p for p in parts if p]
        project_name = parts[-2]

        video_path = os.path.join('data', project_name, f'{file_name}.mpg')
        align_name = project_name.replace('_new', '')
        alignment_path = os.path.join('data', 'alignments', align_name, f'{file_name}.align')

        frames = load_video(video_path)
        alignments = load_alignments(alignment_path)
    except Exception as e:
        print(f'[SKIPPED] {path} → {e}')
        frames = np.zeros((75, 46, 140, 1), dtype=np.float32)
        alignments = np.zeros((1,), dtype=np.int32)
    return frames, alignments


def mappable_function(path):
    features, labels = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    features.set_shape([75, None, None, 1])
    labels.set_shape([40])
    return features, labels


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def build_model_width05(width_mult=0.5):
    c1 = max(8, int(128 * width_mult))
    c2 = max(8, int(256 * width_mult))
    c3 = max(8, int(75 * width_mult))
    m = Sequential([
        Conv3D(c1, 3, input_shape=(75, 46, 140, 1), padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1, 2, 2)),
        Conv3D(c2, 3, padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1, 2, 2)),
        Conv3D(c3, 3, padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1, 2, 2)),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Dense(VOCAB_SIZE + 1, kernel_initializer='he_normal', activation='softmax'),
    ], name='F_Width0.5_yolov8')
    return m


BATCH_SIZE = 2

data = tf.data.Dataset.list_files(all_patterns)
data = data.shuffle(1000, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(BATCH_SIZE, padded_shapes=([75, None, None, 1], [40]))
data = data.prefetch(tf.data.AUTOTUNE)

total = tf.data.experimental.cardinality(data).numpy()
train_size = int(total * 0.8)
val_size = total - train_size

train = data.take(train_size)
val = data.skip(train_size)

print(f'訓練集：{train_size} batch，驗證集：{val_size} batch（batch_size={BATCH_SIZE}）')

CKPT_PATH = os.path.join('models', 'checkpoint_width05_yolov8')
CKPT_EVERY = os.path.join('models', 'checkpoint_width05_yolov8_epoch{epoch:02d}')


class ResumeCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, weights_path, state_path, save_every_sec=180):
        super().__init__()
        self.weights_path = weights_path
        self.state_path = state_path
        self.save_every_sec = save_every_sec
        self.last_save = time.time()

    def on_train_batch_end(self, batch, logs=None):
        now = time.time()
        if now - self.last_save >= self.save_every_sec:
            self.model.save_weights(self.weights_path)
            self.last_save = now

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(self.weights_path)
        val_loss = logs.get('val_loss')
        with open(self.state_path, 'w') as f:
            json.dump({'next_epoch': epoch + 1, 'val_loss': float(np.array(val_loss))}, f)


model = build_model_width05(0.5)
model.summary(line_length=80)

initial_epoch = 0
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    initial_epoch = state.get('next_epoch', 0)

if os.path.exists(RESUME_WEIGHTS):
    print(f'偵測到先前中斷的訓練進度，接續讀取權重：{RESUME_WEIGHTS}（從第 {initial_epoch + 1} 輪繼續）')
    model.load_weights(RESUME_WEIGHTS)
else:
    print(f'暖啟動：載入現役模型權重當起點：{CURRENT_MODEL}')
    base_model = load_model(CURRENT_MODEL, custom_objects={'CTCLoss': CTCLoss}, compile=False)
    model.set_weights(base_model.get_weights())

# lr比照width05_datafix訓練的量級——這次也是模型第一次真正看到大量新分布的資料
# (YOLOv8裁切風格)，需要足夠學習率真的學進去，但不能太大以免破壞其他已學好的部分。
# 先導實驗(pilot_yolov8_domain_adapt.py)用同樣lr=1e-5、100筆資料30輪就有進步，
# 這次資料量大上百倍，可以用同樣lr觀察是否需要調整。
model.compile(optimizer=Adam(learning_rate=1e-5, clipnorm=1.0), loss=CTCLoss)

MAX_EPOCHS = 60  # 2026-09-03延長：第一輪30輪跑完val_loss仍在下降(5.32,未plateau)，
                  # 且benchmark整體(-6.8%)+即時管線比較都還沒明顯勝過datafix，延長看能否收斂更好

callbacks = [
    ModelCheckpoint(CKPT_EVERY, monitor='val_loss',
                     save_weights_only=True, save_best_only=False, verbose=1),
    ModelCheckpoint(CKPT_PATH, monitor='val_loss',
                     save_weights_only=True, save_best_only=True, verbose=0),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ResumeCheckpoint(RESUME_WEIGHTS, STATE_FILE, save_every_sec=180),
]

print(f'\n開始訓練 F_Width0.5 YOLOv8裁切版（暖啟動，lr=1e-5，最多 {MAX_EPOCHS} 輪）...\n')
history = model.fit(
    train,
    validation_data=val,
    epochs=MAX_EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=callbacks
)

model.save(SAVE_PATH)
print(f'\n模型已儲存：{SAVE_PATH}')

with open(DONE_FLAG, 'w') as f:
    f.write('done')
print('已寫入完成標記。')

print('\nEpoch | Train Loss | Val Loss')
print('-' * 35)
for i, (t, v) in enumerate(zip(
    history.history['loss'], history.history['val_loss']
), 1):
    print(f'  {i:2d}  |  {t:.4f}    |  {v:.4f}')
