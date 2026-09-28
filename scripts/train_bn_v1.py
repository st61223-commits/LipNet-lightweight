"""
BatchNorm 架構訓練腳本 v1
- 在每個 Conv3D 後加 BatchNormalization
- 從 v8 搬移 Conv3D / LSTM / Dense 權重（不從頭學）
- 所有層可訓練，lr=1e-4，patience=10，最多 50 輪
"""

import os
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

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import (
    Conv3D, BatchNormalization, Activation, MaxPool3D,
    TimeDistributed, Flatten, Bidirectional, LSTM, Dropout, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

BASE_DIR    = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
V8_MODEL    = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_v8.h5')
SAVE_PATH   = os.path.join(BASE_DIR, 'models', 'trained_model_bn_v1.h5')

os.chdir(BASE_DIR)

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)
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

        frames     = load_video(video_path)
        alignments = load_alignments(alignment_path)
    except Exception as e:
        print(f'[SKIPPED] {path} → {e}')
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


# ── 資料集 ────────────────────────────────────────────────────
all_patterns = [
    r'data\s1\*.mpg',
    r'data\s2\*.mpg',
    r'data\s5\*.mpg',
    r'data\s6\*.mpg',
    r'data\s7\*.mpg',
    r'data\s13\*.mpg',
    r'data\s99_1\*.mpg',
    r'data\s99_6\*.mpg',
    r'data\s99_7\*.mpg',
    r'data\s99_7_new\*.mpg',
    r'data\s99_8\*.mpg',
    r'data\s34_3\*.mpg',
]

data = tf.data.Dataset.list_files(all_patterns)
data = data.shuffle(1000, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes=([75, None, None, 1], [40]))
data = data.prefetch(tf.data.AUTOTUNE)

total = tf.data.experimental.cardinality(data).numpy()
train_size = int(total * 0.8)
val_size   = total - train_size

train = data.take(train_size)
val   = data.skip(train_size)
print(f'訓練集：{train_size} batch，驗證集：{val_size} batch')

# ── 建立 BatchNorm 模型 ───────────────────────────────────────
print('\n建立 BatchNorm 模型...')
bn_model = Sequential([
    Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool3D((1, 2, 2)),

    Conv3D(256, 3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool3D((1, 2, 2)),

    Conv3D(75, 3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool3D((1, 2, 2)),

    TimeDistributed(Flatten()),
    Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
    Dropout(0.5),
    Dense(VOCAB_SIZE + 1, kernel_initializer='he_normal', activation='softmax'),
], name='LipNet_BN')

# 先 compile 一次以初始化權重
bn_model.compile(optimizer=Adam(1e-4), loss=CTCLoss)
bn_model.build((None, 75, 46, 140, 1))

# ── 從 v8 搬移權重 ────────────────────────────────────────────
print(f'載入 v8 模型：{V8_MODEL}')
v8 = load_model(V8_MODEL, custom_objects={'CTCLoss': CTCLoss}, compile=False)

# v8 層索引 → bn_model 層索引
# v8:  conv3d(0)  conv3d_1(3)  conv3d_2(6)  time_dist(9)  bi(10)  bi_1(12)  dense(14)
# bn:  conv3d(0)  conv3d_1(4)  conv3d_2(8)  time_dist(12) bi(13)  bi_1(15)  dense(17)
transfer_map = [(0, 0), (3, 4), (6, 8), (9, 12), (10, 13), (12, 15), (14, 17)]

for v8_idx, bn_idx in transfer_map:
    bn_model.layers[bn_idx].set_weights(v8.layers[v8_idx].get_weights())

print(f'已搬移 {len(transfer_map)} 層權重（Conv3D x3, LSTM x2, Dense x1, TimeDistributed x1）')
del v8

# ── 訓練 ──────────────────────────────────────────────────────
CKPT_EVERY = os.path.join('models', 'checkpoint_bn_v1_epoch{epoch:02d}')
CKPT_BEST  = os.path.join('models', 'checkpoint_bn_v1_best')

callbacks = [
    ModelCheckpoint(CKPT_EVERY, monitor='val_loss',
                    save_weights_only=True, save_best_only=False, verbose=1),
    ModelCheckpoint(CKPT_BEST, monitor='val_loss',
                    save_weights_only=True, save_best_only=True, verbose=0),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
]

print('\n開始訓練 BatchNorm v1（lr=1e-4，最多 50 輪）...\n')
history = bn_model.fit(
    train,
    validation_data=val,
    epochs=50,
    callbacks=callbacks,
)

bn_model.save(SAVE_PATH)
print(f'\n模型已儲存：{SAVE_PATH}')

print('\nEpoch | Train Loss | Val Loss')
print('-' * 35)
for i, (t, v) in enumerate(zip(
    history.history['loss'], history.history['val_loss']
), 1):
    print(f'  {i:2d}  |  {t:.4f}    |  {v:.4f}')
