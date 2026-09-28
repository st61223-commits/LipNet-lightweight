"""
F_Width0.5 資料修復後續訓練（暖啟動，接續現役模型繼續訓練）

背景：2026-08-27 發現 YOLOv5 嘴唇偵測信心度門檻(0.25/0.5)太嚴格，導致全專案
約20%訓練資料的快取檔案是全黑空白影片，其中 s2 高達 82.7%！已經全面修復
（詳見 實驗記錄.md 實驗五）。現役模型 trained_model_grid_multi_width05.h5
是用「有缺陷的舊資料」訓練出來的，需要用「修復後的完整資料」重新訓練，
才能真正吃到這次資料修復的紅利（尤其是 s2，之前幾乎沒真正學過）。

做法：不從零開始訓練（那樣會浪費掉已經訓練好的部分），而是「暖啟動」——
載入現役模型的權重當起點，用跟原本一模一樣的16位說話者資料（* s9 除外，
s9 特意保留當作乾淨的「情況A」測試對象，不參與訓練 *），現在資料已經修復
完整，繼續訓練。

存檔用新檔名（不直接覆蓋現役模型），要先用 benchmark 驗證真的有進步，
才手動替換現役模型、更新通話系統。

執行方式（WSL）：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/train_grid_multi_width05_datafix.py
"""

import os
import json
import time
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
CURRENT_MODEL = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_width05.h5')
SAVE_PATH     = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_width05_datafix.h5')

# 獨立的一組 resume 檔案，不會跟原本第一次訓練的 state 檔案衝突
RESUME_WEIGHTS = os.path.join(BASE_DIR, 'models', 'checkpoint_width05_datafix_resume.weights.h5')
STATE_FILE     = os.path.join(BASE_DIR, 'models', 'train_width05_datafix_state.json')
DONE_FLAG      = os.path.join(BASE_DIR, 'models', 'train_width05_datafix_DONE.flag')

os.chdir(BASE_DIR)

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
VOCAB_SIZE = char_to_num.vocabulary_size()


def load_video(path: str) -> np.ndarray:
    project_name = os.path.basename(os.path.dirname(path))
    file_name = os.path.splitext(os.path.basename(path))[0]
    for suffix in ['_cached', '_dlib_cached']:
        cache_path = os.path.join('data', f'{project_name}{suffix}', f'{file_name}.npy')
        if os.path.exists(cache_path):
            frames = np.load(cache_path)
            mean = np.mean(frames)
            std = np.std(frames) + 1e-6
            return ((frames - mean) / std).astype(np.float32)
    raise FileNotFoundError(f'找不到快取：{project_name}/{file_name}')


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
    ], name='F_Width0.5_datafix')
    return m


# ── 資料集：跟原本一模一樣的16位說話者，s9 刻意不放進來（留當乾淨的情況A測試）
all_patterns = [
    r'data\s1\*.mpg',
    r'data\s2\*.mpg',
    r'data\s3\*.mpg',
    r'data\s4\*.mpg',
    r'data\s5\*.mpg',
    r'data\s6\*.mpg',
    r'data\s7\*.mpg',
    r'data\s8\*.mpg',
    r'data\s13\*.mpg',
    r'data\s99_1\*.mpg',
    r'data\s99_6\*.mpg',
    r'data\s99_6_new\*.mpg',
    r'data\s99_7\*.mpg',
    r'data\s99_7_new\*.mpg',
    r'data\s99_8\*.mpg',
    r'data\s34_3\*.mpg',
]

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

CKPT_PATH = os.path.join('models', 'checkpoint_width05_datafix')
CKPT_EVERY = os.path.join('models', 'checkpoint_width05_datafix_epoch{epoch:02d}')


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

# lr 比從零訓練的1e-4小，但比小幅微調的量級大一些——因為這次s2等說話者
# 相當於第一次真正看到大量新資料，需要足夠的學習率讓模型真的學進去，
# 但又不能太大以免把其他已經學好的說話者弄壞。
model.compile(optimizer=Adam(learning_rate=1e-5, clipnorm=1.0), loss=CTCLoss)

MAX_EPOCHS = 30  # 暖啟動不需要像從零訓練那樣65輪，先看30輪的效果

callbacks = [
    ModelCheckpoint(CKPT_EVERY, monitor='val_loss',
                     save_weights_only=True, save_best_only=False, verbose=1),
    ModelCheckpoint(CKPT_PATH, monitor='val_loss',
                     save_weights_only=True, save_best_only=True, verbose=0),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ResumeCheckpoint(RESUME_WEIGHTS, STATE_FILE, save_every_sec=180),
]

print(f'\n開始訓練 F_Width0.5 資料修復版（暖啟動，lr=1e-5，最多 {MAX_EPOCHS} 輪）...\n')
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
