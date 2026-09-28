"""
BatchNorm 架構訓練腳本 v3
失敗原因修正：bn_v1/v2 搬移 LSTM+Dense 導致 loss 暴衝
修正策略：
  第一階段 — 只搬 Conv3D 權重（從 v10），凍結 Conv3D，BN+LSTM+Dense 從頭學，lr=1e-3
  第二階段 — 解凍全部，lr=2e-6 微調
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

BASE_DIR   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
V10_MODEL  = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_v10.h5')
SAVE_PATH  = os.path.join(BASE_DIR, 'models', 'trained_model_bn_v3.h5')

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


# ── 資料集（同 v10）────────────────────────────────────────────
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

# ── 建立 BatchNorm 模型 ───────────────────────────────────────
print('\n建立 BatchNorm 模型...')
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

bn_model.compile(optimizer=Adam(1e-3), loss=CTCLoss)
bn_model.build((None, 75, 46, 140, 1))

# ── 只搬 Conv3D 權重（不搬 LSTM/Dense）───────────────────────
print(f'載入 v10 模型：{V10_MODEL}')
v10 = load_model(V10_MODEL, custom_objects={'CTCLoss': CTCLoss}, compile=False)

# v10 Conv3D 索引: 0, 3, 6 → bn_model Conv3D 索引: 0, 4, 8
conv_transfer = [(0, 0), (3, 4), (6, 8)]
for v10_idx, bn_idx in conv_transfer:
    bn_model.layers[bn_idx].set_weights(v10.layers[v10_idx].get_weights())
del v10
print('已搬移 3 層 Conv3D 權重（BN/LSTM/Dense 從頭學）')

# ── 第一階段：凍結 Conv3D，只訓練 BN+LSTM+Dense ──────────────
for layer in bn_model.layers:
    if 'conv3d' in layer.name:
        layer.trainable = False

trainable_count = sum(1 for l in bn_model.layers if l.trainable)
print(f'\n【第一階段】凍結 Conv3D，可訓練層數：{trainable_count}，lr=1e-3，最多 30 輪')

bn_model.compile(optimizer=Adam(1e-3), loss=CTCLoss)

callbacks_p1 = [
    ModelCheckpoint(os.path.join('models', 'checkpoint_bn_v3_p1_best'),
                    monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=0),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
]

history_p1 = bn_model.fit(train, validation_data=val, epochs=30, callbacks=callbacks_p1)

print('\n第一階段完成！')
print('\nEpoch | Train Loss | Val Loss')
print('-' * 35)
for i, (t, v) in enumerate(zip(history_p1.history['loss'], history_p1.history['val_loss']), 1):
    print(f'  {i:2d}  |  {t:.4f}    |  {v:.4f}')

# ── 第二階段：解凍全部，低 lr 微調 ──────────────────────────
for layer in bn_model.layers:
    layer.trainable = True

print(f'\n【第二階段】解凍全部，lr=2e-6，最多 50 輪')
bn_model.compile(optimizer=Adam(2e-6), loss=CTCLoss)

callbacks_p2 = [
    ModelCheckpoint(os.path.join('models', 'checkpoint_bn_v3_p2_epoch{epoch:02d}'),
                    monitor='val_loss', save_weights_only=True, save_best_only=False, verbose=1),
    ModelCheckpoint(os.path.join('models', 'checkpoint_bn_v3_p2_best'),
                    monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=0),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
]

history_p2 = bn_model.fit(train, validation_data=val, epochs=50, callbacks=callbacks_p2)

bn_model.save(SAVE_PATH)
print(f'\n模型已儲存：{SAVE_PATH}')

print('\n【第二階段】Epoch | Train Loss | Val Loss')
print('-' * 40)
for i, (t, v) in enumerate(zip(history_p2.history['loss'], history_p2.history['val_loss']), 1):
    print(f'  {i:2d}  |  {t:.4f}    |  {v:.4f}')
