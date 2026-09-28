"""
多說話者 GRID 訓練腳本 — F_Width0.5（輕量化，寬度縮放版本）

跟 train_grid_multi_v11.py 用同一套資料（16 位說話者、同樣的 80/20 切分、同樣的
CTCLoss），但模型架構換成 model_experiment.py 裡驗證過的 F_Width0.5：
  BatchNorm 版 LipNet，三層 Conv3D 通道數全部乘 0.5（128→64、256→128、75→38）

跟 v11 最大的不同：v11 是「載入 v10 舊模型繼續微調」，這支腳本是全新架構，
沒有相容的舊權重可以載入，只能「從頭開始訓練」（random initialization）。
這代表第一次訓練結果不會馬上追上 v11 的 67.9%，需要像 v2→v11 一樣分好幾次
訓練才會慢慢進步，這是預期中的事，不是失敗。

先跑 15 epoch 建立第一個版本（width05_v1），之後可以用同一套 resume 機制繼續訓練。
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

# GPU 記憶體「用多少要多少」，避免一開始就把整張卡鎖住（之前 model_experiment.py
# 在這台 6GB GPU 上吃過 OOM 的虧，這裡先加上去）
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 注意：這裡刻意「不」開 mixed_precision。CTC loss 對數值精度敏感，
# 之前 BatchNorm 移植（bn_v1/v2）就吃過 loss 暴衝到 100~200 的虧，
# 這次是全新架構從頭訓練，先求穩定，不額外疊加 mixed_precision 的風險。

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional,
    MaxPool3D, Activation, TimeDistributed, Flatten,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

BASE_DIR  = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
SAVE_PATH = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_width05.h5')

# 電腦如果無預警重開機（例如過熱斷電，之前 v11/bn_v3 訓練都發生過），
# 這三個檔案讓訓練可以自動接續，不用從頭開始
RESUME_WEIGHTS = os.path.join(BASE_DIR, 'models', 'checkpoint_grid_multi_width05_resume.weights.h5')
STATE_FILE     = os.path.join(BASE_DIR, 'models', 'train_width05_state.json')
DONE_FLAG      = os.path.join(BASE_DIR, 'models', 'train_width05_DONE.flag')

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
    """跟 model_experiment.py 的 model_bn_width() 完全一樣的架構。"""
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
    ], name='F_Width0.5_production')
    return m


# ── 資料集（跟 v11 完全同一套 16 位說話者、同樣切分方式）───────
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

CKPT_PATH = os.path.join('models', 'checkpoint_grid_multi_width05')
CKPT_EVERY = os.path.join('models', 'checkpoint_grid_multi_width05_epoch{epoch:02d}')


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
    print('沒有中斷紀錄，全新架構從頭開始訓練（random initialization）')

model.compile(optimizer=Adam(learning_rate=1e-4, clipnorm=1.0), loss=CTCLoss)

MAX_EPOCHS = 65  # 15→35（val_loss持續下降,46.2%）,這次接續再訓練到65（比照v2→v11的漸進模式）

callbacks = [
    ModelCheckpoint(CKPT_EVERY, monitor='val_loss',
                     save_weights_only=True, save_best_only=False, verbose=1),
    ModelCheckpoint(CKPT_PATH, monitor='val_loss',
                     save_weights_only=True, save_best_only=True, verbose=0),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ResumeCheckpoint(RESUME_WEIGHTS, STATE_FILE, save_every_sec=180),
]

print(f'\n開始訓練 F_Width0.5 正式版（lr=1e-4，從頭訓練，最多 {MAX_EPOCHS} 輪）...\n')
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
