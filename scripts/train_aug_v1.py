"""
資料增強訓練腳本 v1
- 從 trained_model_grid_multi_v8.h5 出發
- 訓練時隨機套用：水平翻轉、亮度、對比度
- lr=1e-5，Conv3D 凍結，patience=10，最多 50 輪
"""

import os
import random
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

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

BASE_DIR    = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
START_MODEL = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_v8.h5')
SAVE_PATH   = os.path.join(BASE_DIR, 'models', 'trained_model_aug_v1.h5')

os.chdir(BASE_DIR)

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def augment_frames(frames: np.ndarray) -> np.ndarray:
    """對 (75, H, W, 1) 的 frames 做隨機增強"""
    # 水平翻轉（50%）
    if random.random() < 0.5:
        frames = np.flip(frames, axis=2).copy()

    # 隨機亮度 ±15%
    frames = frames * (1.0 + random.uniform(-0.15, 0.15))

    # 隨機對比度 ±10%（以均值為中心縮放）
    mean = np.mean(frames)
    frames = mean + (1.0 + random.uniform(-0.10, 0.10)) * (frames - mean)

    return frames.astype(np.float32)


def load_video(path: str, is_training: bool = False) -> np.ndarray:
    project_name = os.path.basename(os.path.dirname(path))
    file_name = os.path.splitext(os.path.basename(path))[0]
    for suffix in ['_cached', '_dlib_cached']:
        cache_path = os.path.join('data', f'{project_name}{suffix}', f'{file_name}.npy')
        if os.path.exists(cache_path):
            frames = np.load(cache_path)
            mean = np.mean(frames)
            std  = np.std(frames) + 1e-6
            frames = ((frames - mean) / std).astype(np.float32)
            if is_training:
                frames = augment_frames(frames)
            return frames
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


IS_TRAINING = True   # 訓練集用，驗證集不增強（由 load_data_val 控制）

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

        frames     = load_video(video_path, is_training=True)
        alignments = load_alignments(alignment_path)
    except Exception as e:
        print(f'[SKIPPED] {path} → {e}')
        frames     = np.zeros((75, 46, 140, 1), dtype=np.float32)
        alignments = np.zeros((1,), dtype=np.int32)
    return frames, alignments


def load_data_val(path):
    """驗證集不做增強"""
    try:
        path = bytes.decode(path.numpy())
        file_name = os.path.splitext(os.path.basename(path))[0]
        parts = path.replace('\\\\', '/').replace('\\', '/').split('/')
        parts = [p for p in parts if p]
        project_name = parts[-2]

        video_path = os.path.join('data', project_name, f'{file_name}.mpg')
        align_name = project_name.replace('_new', '')
        alignment_path = os.path.join('data', 'alignments', align_name, f'{file_name}.align')

        frames     = load_video(video_path, is_training=False)
        alignments = load_alignments(alignment_path)
    except Exception as e:
        frames     = np.zeros((75, 46, 140, 1), dtype=np.float32)
        alignments = np.zeros((1,), dtype=np.int32)
    return frames, alignments


def mappable_train(path):
    features, labels = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    features.set_shape([75, None, None, 1])
    labels.set_shape([40])
    return features, labels


def mappable_val(path):
    features, labels = tf.py_function(load_data_val, [path], (tf.float32, tf.int64))
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

all_files = tf.data.Dataset.list_files(all_patterns, shuffle=False)
total = tf.data.experimental.cardinality(all_files).numpy()
train_size = int(total * 0.8)

# 訓練集：有增強
train_files = all_files.take(train_size).shuffle(1000)
train = train_files.map(mappable_train)
train = train.padded_batch(2, padded_shapes=([75, None, None, 1], [40]))
train = train.prefetch(tf.data.AUTOTUNE)

# 驗證集：無增強
val_files = all_files.skip(train_size)
val = val_files.map(mappable_val)
val = val.padded_batch(2, padded_shapes=([75, None, None, 1], [40]))
val = val.prefetch(tf.data.AUTOTUNE)

train_batches = tf.data.experimental.cardinality(train).numpy()
val_batches   = tf.data.experimental.cardinality(val).numpy()
print(f'訓練集：{train_batches} batch，驗證集：{val_batches} batch')

# ── 載入模型 ──────────────────────────────────────────────────
print(f'\n載入 v8 模型：{START_MODEL}')
model = load_model(START_MODEL, custom_objects={'CTCLoss': CTCLoss})

for layer in model.layers:
    if 'conv3d' in layer.name:
        layer.trainable = False

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f'可訓練層數：{trainable_count}（Conv3D 已凍結）')

model.compile(optimizer=Adam(learning_rate=1e-5, clipnorm=1.0), loss=CTCLoss)

# ── 訓練 ──────────────────────────────────────────────────────
CKPT_EVERY = os.path.join('models', 'checkpoint_aug_v1_epoch{epoch:02d}')
CKPT_BEST  = os.path.join('models', 'checkpoint_aug_v1_best')

callbacks = [
    ModelCheckpoint(CKPT_EVERY, monitor='val_loss',
                    save_weights_only=True, save_best_only=False, verbose=1),
    ModelCheckpoint(CKPT_BEST, monitor='val_loss',
                    save_weights_only=True, save_best_only=True, verbose=0),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
]

print('\n開始訓練 aug_v1（資料增強，lr=1e-5，最多 50 輪）...\n')
history = model.fit(
    train,
    validation_data=val,
    epochs=50,
    callbacks=callbacks,
)

model.save(SAVE_PATH)
print(f'\n模型已儲存：{SAVE_PATH}')

print('\nEpoch | Train Loss | Val Loss')
print('-' * 35)
for i, (t, v) in enumerate(zip(
    history.history['loss'], history.history['val_loss']
), 1):
    print(f'  {i:2d}  |  {t:.4f}    |  {v:.4f}')
