"""
多說話者 GRID 訓練腳本 v7
- 起點：trained_model_grid_multi_v6.h5（benchmark 加權平均 63.9%）
- 訓練資料同 v6：s1+s2+s5+s6+s7+s13 + s99_1+s99_6+s99_7+s99_8 + s34_3
- 學習率降為 2e-6（v6 用 5e-6 持續進步，再降低學習率讓模型更精細地調整）
- 繼續訓練 30 epoch，patience=5
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

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

BASE_DIR    = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
START_MODEL = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_v6.h5')
SAVE_PATH   = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_v7.h5')

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


def scheduler(epoch, lr):
    return lr if epoch < 30 else lr * tf.math.exp(-0.1)


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

CKPT_PATH  = os.path.join('models', 'checkpoint_grid_multi_v7')
CKPT_EVERY = os.path.join('models', 'checkpoint_grid_multi_v7_epoch{epoch:02d}')

print(f'載入起點模型：{START_MODEL}')
model = load_model(START_MODEL, custom_objects={'CTCLoss': CTCLoss})

for layer in model.layers:
    if 'conv3d' in layer.name:
        layer.trainable = False

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f'可訓練層數：{trainable_count}（Conv3D 已凍結）')

model.compile(optimizer=Adam(learning_rate=2e-6, clipnorm=1.0), loss=CTCLoss)

callbacks = [
    ModelCheckpoint(CKPT_EVERY, monitor='val_loss',
                    save_weights_only=True, save_best_only=False, verbose=1),
    ModelCheckpoint(CKPT_PATH, monitor='val_loss',
                    save_weights_only=True, save_best_only=True, verbose=0),
    LearningRateScheduler(scheduler),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
]

print('\n開始訓練...\n')
history = model.fit(
    train,
    validation_data=val,
    epochs=30,
    callbacks=callbacks
)

model.save(SAVE_PATH)
print(f'\n模型已儲存：{SAVE_PATH}')

print('\nEpoch | Train Loss | Val Loss')
print('-' * 35)
for i, (t, v) in enumerate(zip(
    history.history['loss'], history.history['val_loss']
), 1):
    print(f'  {i:2d}  |  {t:.4f}    |  {v:.4f}')
