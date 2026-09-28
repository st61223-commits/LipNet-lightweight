"""
BatchNorm 架構訓練腳本 v3 - 第二階段續訓
背景：v3 第一階段完成（10輪，val_loss 卡在 99~100，幾乎沒進步）。
第二階段（解凍全部，lr=2e-6）訓練到第4輪（存檔於 checkpoint_bn_v3_p2_epoch04），
第5輪跑到一半電腦疑似過熱重開機，訓練中斷，沒有存下最終模型。
這支腳本從第4輪的權重接續訓練第二階段，並加入自動接續機制（仿照 v11），
避免電腦又意外重開機時要從頭開始。
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
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, BatchNormalization, Activation, MaxPool3D,
    TimeDistributed, Flatten, Bidirectional, LSTM, Dropout, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

BASE_DIR   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
SAVE_PATH  = os.path.join(BASE_DIR, 'models', 'trained_model_bn_v3.h5')

# 上次中斷前，第二階段最後完整存下的權重（第4輪，2026-06-24 16:25 存的）
START_WEIGHTS = os.path.join(BASE_DIR, 'models', 'checkpoint_bn_v3_p2_epoch04')

# 這支續訓腳本自己如果又中斷，靠這三個檔案接續，不用重新讀 START_WEIGHTS
RESUME_WEIGHTS = os.path.join(BASE_DIR, 'models', 'checkpoint_bn_v3_p2r_resume.weights.h5')
STATE_FILE     = os.path.join(BASE_DIR, 'models', 'train_bn_v3_p2r_state.json')
DONE_FLAG      = os.path.join(BASE_DIR, 'models', 'train_bn_v3_p2r_DONE.flag')

os.chdir(BASE_DIR)

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)
VOCAB_SIZE = char_to_num.vocabulary_size()


def load_video(path: str) -> np.ndarray:
    project_name = os.path.basename(os.path.dirname(path))
    file_name = os.path.splitext(os.path.basename(path))[0]
    for suffix in ['_cached', '_dlib_cached']:
        cache_path = os.path.join('data', f'{project_name}{suffix}', f'{file_name}.npy')
        if os.path.exists(cache_path):
            frames = np.load(cache_path)
            mean = np.mean(frames)
            std  = np.std(frames) + 1e-6
            return ((frames - mean) / std).astype(np.float32)
    raise FileNotFoundError(f'找不到快取：{project_name}/{file_name}')


def load_alignments(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split()
        word = parts[2]
        if word not in ('sil', 'sp') and len(word) > 1:
            tokens.extend([' ', word])
    tokens_flat = tokens[1:]
    if not tokens_flat:
        return tf.zeros([1], dtype=tf.int64)
    return char_to_num(tf.reshape(
        tf.strings.unicode_split(tokens_flat, input_encoding='UTF-8'), (-1)))


def load_data(path):
    try:
        path = bytes.decode(path.numpy())
        file_name = os.path.splitext(os.path.basename(path))[0]
        parts = [p for p in path.replace('\\\\', '/').replace('\\', '/').split('/') if p]
        project_name = parts[-2]
        video_path = os.path.join('data', project_name, f'{file_name}.mpg')
        align_name = project_name.replace('_new', '')
        alignment_path = os.path.join('data', 'alignments', align_name, f'{file_name}.align')
        frames     = load_video(video_path)
        alignments = load_alignments(alignment_path)
    except Exception as e:
        frames     = np.zeros((75, 46, 140, 1), dtype=np.float32)
        alignments = np.zeros((1,), dtype=np.int32)
    return frames, alignments


def mappable_function(path):
    features, labels = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    features.set_shape([75, None, None, 1])
    labels.set_shape([40])
    return features, labels


def CTCLoss(y_true, y_pred):
    batch_len    = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


# ── 資料集（跟原本 v3 一樣，才是同一個實驗的延續）────────────────
all_patterns = [
    r'data\s1\*.mpg',  r'data\s2\*.mpg',  r'data\s3\*.mpg',
    r'data\s4\*.mpg',  r'data\s5\*.mpg',  r'data\s6\*.mpg',
    r'data\s7\*.mpg',  r'data\s8\*.mpg',  r'data\s13\*.mpg',
    r'data\s99_1\*.mpg', r'data\s99_6\*.mpg',
    r'data\s99_7\*.mpg', r'data\s99_7_new\*.mpg',
    r'data\s99_8\*.mpg', r'data\s34_3\*.mpg',
]

data = tf.data.Dataset.list_files(all_patterns)
data = data.shuffle(1000, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes=([75, None, None, 1], [40]))
data = data.prefetch(tf.data.AUTOTUNE)

total      = tf.data.experimental.cardinality(data).numpy()
train_size = int(total * 0.8)
val_size   = total - train_size
train = data.take(train_size)
val   = data.skip(train_size)
print(f'訓練集：{train_size} batch，驗證集：{val_size} batch')

# ── 重建跟原本 v3 完全相同的模型架構（形狀要一樣才能載入權重）───
print('\n建立 BatchNorm 模型（跟 v3 相同架構）...')
bn_model = Sequential([
    Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same', name='conv3d_0'),
    BatchNormalization(name='bn_0'),
    Activation('relu'),
    MaxPool3D((1, 2, 2)),

    Conv3D(256, 3, padding='same', name='conv3d_1'),
    BatchNormalization(name='bn_1'),
    Activation('relu'),
    MaxPool3D((1, 2, 2)),

    Conv3D(75, 3, padding='same', name='conv3d_2'),
    BatchNormalization(name='bn_2'),
    Activation('relu'),
    MaxPool3D((1, 2, 2)),

    TimeDistributed(Flatten()),
    Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
    Dropout(0.5),
    Dense(VOCAB_SIZE + 1, kernel_initializer='he_normal', activation='softmax'),
], name='LipNet_BN_v3')

bn_model.build((None, 75, 46, 140, 1))

# ── 讀取權重，決定從第幾輪接續 ───────────────────────────────
initial_epoch = 4
if os.path.exists(RESUME_WEIGHTS) and os.path.exists(STATE_FILE):
    print(f'偵測到這支續訓腳本自己中斷過的紀錄，讀取：{RESUME_WEIGHTS}')
    bn_model.load_weights(RESUME_WEIGHTS)
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    initial_epoch = state.get('next_epoch', 4)
else:
    print(f'讀取 v3 第二階段上次中斷前最後存檔（第4輪）：{START_WEIGHTS}')
    bn_model.load_weights(START_WEIGHTS)

for layer in bn_model.layers:
    layer.trainable = True

print(f'從第 {initial_epoch + 1} 輪開始接續訓練，lr=2e-6，最多到第 50 輪')
bn_model.compile(optimizer=Adam(2e-6), loss=CTCLoss)


class ResumeCheckpoint(tf.keras.callbacks.Callback):
    """每隔一段時間存一次最新權重，並記錄目前跑到第幾輪，讓電腦意外重開機後可以接續訓練。"""

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
        with open(self.state_path, 'w') as f:
            json.dump({'next_epoch': epoch + 1, 'val_loss': logs.get('val_loss')}, f)


callbacks = [
    ModelCheckpoint(os.path.join('models', 'checkpoint_bn_v3_p2_epoch{epoch:02d}'),
                    monitor='val_loss', save_weights_only=True, save_best_only=False, verbose=1),
    ModelCheckpoint(os.path.join('models', 'checkpoint_bn_v3_p2r_best'),
                    monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=0),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ResumeCheckpoint(RESUME_WEIGHTS, STATE_FILE, save_every_sec=180),
]

print('\n開始接續訓練第二階段...\n')
history = bn_model.fit(
    train, validation_data=val, epochs=50,
    initial_epoch=initial_epoch, callbacks=callbacks
)

bn_model.save(SAVE_PATH)
print(f'\n模型已儲存：{SAVE_PATH}')

with open(DONE_FLAG, 'w') as f:
    f.write('done')
print('已寫入完成標記，之後開機不會再自動重跑這個訓練。')

print('\n【第二階段續訓】Epoch | Train Loss | Val Loss')
print('-' * 40)
for i, (t, v) in enumerate(zip(history.history['loss'], history.history['val_loss']), 1):
    print(f'  {i:2d}  |  {t:.4f}    |  {v:.4f}')
