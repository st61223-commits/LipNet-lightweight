"""
多說話者 GRID 訓練腳本 v11
- 起點：trained_model_grid_multi_v10.h5（加權平均 67.2%）
- 關鍵改動：lr 從 5e-7 降到 2e-7，繼續微調
- 補上 s99_6_new（100筆，v10 漏掉）
- Conv3D 維持解凍
- patience=5，最多 20 輪
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

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

BASE_DIR    = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
START_MODEL = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_v10.h5')
SAVE_PATH   = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_v11.h5')

# 電腦如果無預警重開機（例如過熱斷電），這三個檔案讓訓練可以自動接續，不用從頭開始
RESUME_WEIGHTS = os.path.join(BASE_DIR, 'models', 'checkpoint_grid_multi_v11_resume.weights.h5')
STATE_FILE     = os.path.join(BASE_DIR, 'models', 'train_v11_state.json')
DONE_FLAG      = os.path.join(BASE_DIR, 'models', 'train_v11_DONE.flag')

os.chdir(BASE_DIR)

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


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
    r'data\s99_6_new\*.mpg',   # v10 漏掉，補上
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

CKPT_PATH  = os.path.join('models', 'checkpoint_grid_multi_v11')
CKPT_EVERY = os.path.join('models', 'checkpoint_grid_multi_v11_epoch{epoch:02d}')


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

print(f'載入 v10 模型：{START_MODEL}')
model = load_model(START_MODEL, custom_objects={'CTCLoss': CTCLoss})

initial_epoch = 0
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    initial_epoch = state.get('next_epoch', 0)

if os.path.exists(RESUME_WEIGHTS):
    print(f'偵測到先前中斷的訓練進度，接續讀取權重：{RESUME_WEIGHTS}（從第 {initial_epoch + 1} 輪繼續）')
    model.load_weights(RESUME_WEIGHTS)
else:
    print('沒有中斷紀錄，從 v10 模型全新開始訓練')

for layer in model.layers:
    layer.trainable = True

trainable_count = sum(1 for l in model.layers if l.trainable)
conv_count = sum(1 for l in model.layers if 'conv3d' in l.name)
print(f'可訓練層數：{trainable_count}（含 {conv_count} 個 Conv3D，全部解凍）')

model.compile(optimizer=Adam(learning_rate=2e-7, clipnorm=1.0), loss=CTCLoss)

callbacks = [
    ModelCheckpoint(CKPT_EVERY, monitor='val_loss',
                    save_weights_only=True, save_best_only=False, verbose=1),
    ModelCheckpoint(CKPT_PATH, monitor='val_loss',
                    save_weights_only=True, save_best_only=True, verbose=0),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ResumeCheckpoint(RESUME_WEIGHTS, STATE_FILE, save_every_sec=180),
]

print('\n開始訓練 v11（lr=2e-7，Conv3D 全解凍，最多 20 輪）...\n')
history = model.fit(
    train,
    validation_data=val,
    epochs=20,
    initial_epoch=initial_epoch,
    callbacks=callbacks
)

model.save(SAVE_PATH)
print(f'\n模型已儲存：{SAVE_PATH}')

with open(DONE_FLAG, 'w') as f:
    f.write('done')
print('已寫入完成標記，之後開機不會再自動重跑這個訓練。')

print('\nEpoch | Train Loss | Val Loss')
print('-' * 35)
for i, (t, v) in enumerate(zip(
    history.history['loss'], history.history['val_loss']
), 1):
    print(f'  {i:2d}  |  {t:.4f}    |  {v:.4f}')
