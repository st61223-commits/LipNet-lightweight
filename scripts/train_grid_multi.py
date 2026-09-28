"""
多說話者 GRID 訓練腳本
- 起點：freeze_conv.h5（Conv3D 已凍結）
- 訓練：s1+s2+s5+s6+s7+s13（GRID）+ s99_1+s99_6+s99_7+s99_8（防遺忘）
- 測試集：s34_3（不進訓練，最後才評分）
"""

import os
import sys
import glob
import pathlib
import warnings
import cv2
import numpy as np
import tensorflow as tf
import torch

# Windows/Linux 路徑相容
pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

# ── GPU 設定 ──
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from typing import List

# ── 路徑設定 ──
BASE_DIR    = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
YOLO_REPO   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5'
YOLO_WEIGHT = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5\runs\v5s\weights\best.pt'
START_MODEL = os.path.join(BASE_DIR, 'models', 'trained_model_freeze_conv.h5')
SAVE_PATH   = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi.h5')

os.chdir(BASE_DIR)

# ── 詞彙表 ──
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# ── YOLOv5 ──
print('載入 YOLOv5...')
yolov5s = torch.hub.load(YOLO_REPO, 'custom', path=YOLO_WEIGHT, source='local')
print('YOLOv5 載入完成')

input_shape = (75, 46, 140, 1)


def load_video(path: str) -> np.ndarray:
    project_name = os.path.basename(os.path.dirname(path))
    cache_dir = os.path.join('data', f'{project_name}_cached')
    file_name = os.path.splitext(os.path.basename(path))[0]
    cache_path = os.path.join(cache_dir, f'{file_name}.npy')

    if os.path.exists(cache_path):
        frames = np.load(cache_path)
        mean = np.mean(frames)
        std  = np.std(frames) + 1e-6
        return ((frames - mean) / std).astype(np.float32)

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        detections = yolov5s(frame)
        for det in detections.pred:
            for *xyxy, conf, cls in det:
                if int(cls) == 0 and conf >= 0.5:
                    x1, y1, x2, y2 = map(int, xyxy)
                    lip = frame[y1:y2, x1:x2]
                    lip = cv2.resize(lip, (input_shape[2], input_shape[1]))
                    lip = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
                    frames.append(lip.astype(np.float32))
    cap.release()

    T = 75
    frames = frames[:T]
    if not frames:
        frames = [np.zeros((input_shape[1], input_shape[2]), dtype=np.float32)]
    while len(frames) < T:
        frames.append(frames[-1])
    frames = np.expand_dims(np.array(frames), -1)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, frames)

    mean = np.mean(frames)
    std  = np.std(frames) + 1e-6
    return ((frames - mean) / std).astype(np.float32)


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


def scheduler(epoch, lr):
    return lr if epoch < 30 else lr * tf.math.exp(-0.1)


# ── 建立 Dataset ──
GRID_PATTERNS = [
    r'data\s1\*.mpg',
    r'data\s2\*.mpg',
    r'data\s5\*.mpg',
    r'data\s6\*.mpg',
    r'data\s7\*.mpg',
    r'data\s13\*.mpg',
]
S99_PATTERNS = [
    r'data\s99_1\*.mpg',
    r'data\s99_6\*.mpg',
    r'data\s99_7\*.mpg',
    r'data\s99_7_new\*.mpg',
    r'data\s99_8\*.mpg',
]

all_patterns = GRID_PATTERNS + S99_PATTERNS
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

# ── 載入模型（自動判斷從 checkpoint 繼續或從頭開始）──
CKPT_PATH    = os.path.join('models', 'checkpoint_grid_multi')
CKPT_EVERY   = os.path.join('models', 'checkpoint_grid_multi_epoch{epoch:02d}')
EPOCH_FILE   = os.path.join('models', 'grid_multi_epoch.txt')

def read_last_epoch():
    if os.path.exists(EPOCH_FILE):
        try:
            return int(open(EPOCH_FILE).read().strip())
        except:
            pass
    return 0

initial_epoch = read_last_epoch()

print(f'載入起點模型：{START_MODEL}')
model = load_model(START_MODEL, custom_objects={'CTCLoss': CTCLoss})

for layer in model.layers:
    if 'conv3d' in layer.name:
        layer.trainable = False

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f'可訓練層數：{trainable_count}（Conv3D 已凍結）')

model.compile(optimizer=Adam(learning_rate=1e-5, clipnorm=1.0), loss=CTCLoss)

# 嘗試載入最近的 epoch checkpoint
latest = tf.train.latest_checkpoint(os.path.dirname(CKPT_EVERY.format(epoch=0)) if '/' in CKPT_EVERY else 'models')
if initial_epoch > 0 and latest:
    model.load_weights(latest)
    print(f'[續訓] 從第 {initial_epoch} epoch 繼續，載入權重：{latest}')
else:
    initial_epoch = 0
    print('[新訓] 從頭開始訓練')

# ── Callbacks ──
class SaveEpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        open(EPOCH_FILE, 'w').write(str(epoch + 1))

callbacks = [
    ModelCheckpoint(CKPT_EVERY, monitor='val_loss',
                    save_weights_only=True, save_best_only=False, verbose=1),
    ModelCheckpoint(CKPT_PATH, monitor='val_loss',
                    save_weights_only=True, save_best_only=True, verbose=0),
    LearningRateScheduler(scheduler),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    SaveEpochCallback(),
]

# ── 訓練 ──
print('\n開始訓練...\n')
history = model.fit(
    train,
    validation_data=val,
    epochs=20,
    initial_epoch=initial_epoch,
    callbacks=callbacks
)

# ── 儲存 ──
model.save(SAVE_PATH)
print(f'\n模型已儲存：{SAVE_PATH}')

# ── 訓練摘要 ──
print('\nEpoch | Train Loss | Val Loss')
print('-' * 35)
for i, (t, v) in enumerate(zip(
    history.history['loss'], history.history['val_loss']
), 1):
    print(f'  {i:2d}  |  {t:.4f}    |  {v:.4f}')
