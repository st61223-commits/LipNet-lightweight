"""
架構升級實驗：Transformer 後端（Self-Attention）取代 BiLSTM，完整規模訓練

背景：小規模先導測試（claude-code/model_experiment_transformer.py，60筆/10epoch）
顯示 Transformer 版參數量減半（4.1M vs 8.47M）、loss 降得比 BiLSTM 基準更低
（60.75 vs 74.05）。這支腳本把先導架構套用到完整16位說話者資料集，驗證趨勢
是否延續（比照過去 DSConv/width05 等實驗「先小規模再全量」的做法，先小規模
看過方向沒問題才投入全量訓練時間）。

架構：跟 train_grid_multi_width05_yolov8.py 一樣的 Conv3D+BatchNorm 前端
（未砍寬度，維持128/256/75通道，因為這次改動的是後端不是前端寬度，避免同時
改兩個變因看不出來是哪個造成差異），後端把 BiLSTM 換成 2 層 Transformer
Encoder（Self-Attention）。CTC loss跟現有模型完全一致（不像DSConv蒸餾那樣
混了額外的loss項，這次val_loss理論上可以跟其他模型直接比較）。

**從零開始訓練**（無法從BiLSTM架構暖啟動，架構不相容），比照F_Width0.5/
DSConv第一次從頭訓練的做法。

執行前必讀：讀取 `data/{speaker}_cached_yolov8/` 快取（跟現役width05_yolov8
訓練用同一批資料，維持訓練/部署管線一致，避免重蹈Training-Inference Mismatch
覆轍），必須已執行過 regenerate_cache_yolov8.py。

執行方式（WSL）：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/train_grid_multi_transformer.py
"""

import os
import sys
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

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, Dense, Dropout, MaxPool3D, Activation,
    TimeDistributed, Flatten, BatchNormalization,
    MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

BASE_DIR  = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
SAVE_PATH = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_transformer.h5')

RESUME_WEIGHTS = os.path.join(BASE_DIR, 'models', 'checkpoint_transformer_resume.weights.h5')
STATE_FILE     = os.path.join(BASE_DIR, 'models', 'train_transformer_state.json')
DONE_FLAG      = os.path.join(BASE_DIR, 'models', 'train_transformer_DONE.flag')

CACHE_SUFFIX = '_cached_yolov8'  # 跟現役width05_yolov8同一批快取，維持訓練/部署管線一致

os.chdir(BASE_DIR)

# ── 資料集：跟 width05_yolov8 同樣16位說話者，s9依然排除(留當held-out測試)
SPEAKER_DIRS = [
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's13',
    's99_1', 's99_6', 's99_6_new', 's99_7', 's99_7_new', 's99_8', 's34_3',
]
all_patterns = [f'data\\{spk}\\*.mpg' for spk in SPEAKER_DIRS]

missing = [spk for spk in SPEAKER_DIRS if not os.path.isdir(os.path.join('data', f'{spk}{CACHE_SUFFIX}'))]
if missing:
    print(f'[中止] 以下說話者還沒有 {CACHE_SUFFIX} 快取: {missing}')
    print('請先執行 scripts\\regenerate_cache_yolov8.py 產生快取後再跑這支訓練腳本。')
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
        print(f'[SKIPPED] {path} -> {e}')
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


def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)


def transformer_block(x, d_model=256, num_heads=4, ff_dim=512, dropout=0.1, name=''):
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                                   name=f'{name}_mha')(x, x)
    attn_out = Dropout(dropout)(attn_out)
    x = Add()([x, attn_out])
    x = LayerNormalization(epsilon=1e-6)(x)

    ffn = Dense(ff_dim, activation='relu')(x)
    ffn = Dense(d_model)(ffn)
    ffn = Dropout(dropout)(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def build_model_transformer():
    """Conv3D+BN 前端（維持原始寬度，只換後端）＋ 2層 Transformer Encoder 後端"""
    d_model = 256
    inp = Input(shape=(75, 46, 140, 1))
    x = Conv3D(128, 3, padding='same')(inp); x = BatchNormalization()(x); x = Activation('relu')(x); x = MaxPool3D((1, 2, 2))(x)
    x = Conv3D(256, 3, padding='same')(x);   x = BatchNormalization()(x); x = Activation('relu')(x); x = MaxPool3D((1, 2, 2))(x)
    x = Conv3D(75,  3, padding='same')(x);   x = BatchNormalization()(x); x = Activation('relu')(x); x = MaxPool3D((1, 2, 2))(x)
    x = TimeDistributed(Flatten())(x)
    x = Dense(d_model)(x)
    pe = positional_encoding(75, d_model)
    x = x + pe
    x = transformer_block(x, d_model=d_model, num_heads=4, ff_dim=512, dropout=0.1, name='blk1')
    x = transformer_block(x, d_model=d_model, num_heads=4, ff_dim=512, dropout=0.1, name='blk2')
    out = Dense(VOCAB_SIZE + 1, kernel_initializer='he_normal', activation='softmax')(x)
    return Model(inp, out, name='Transformer_full')


BATCH_SIZE = 1  # 先導實驗用1、Transformer的attention記憶體用量比LSTM高，先求穩定不OOM

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

CKPT_PATH = os.path.join('models', 'checkpoint_transformer')
CKPT_EVERY = os.path.join('models', 'checkpoint_transformer_epoch{epoch:02d}')


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


model = build_model_transformer()
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
    print('從零開始訓練（Transformer架構無法從BiLSTM模型暖啟動）')

# lr比照小規模先導實驗(model_experiment_transformer.py)驗證過能正常收斂的設定
model.compile(optimizer=Adam(learning_rate=1e-4, clipnorm=1.0), loss=CTCLoss)

MAX_EPOCHS = 60  # 跟width05_yolov8同樣規模，方便直接比較

callbacks = [
    ModelCheckpoint(CKPT_EVERY, monitor='val_loss',
                     save_weights_only=True, save_best_only=False, verbose=1),
    ModelCheckpoint(CKPT_PATH, monitor='val_loss',
                     save_weights_only=True, save_best_only=True, verbose=0),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ResumeCheckpoint(RESUME_WEIGHTS, STATE_FILE, save_every_sec=180),
]

print(f'\n開始訓練 Transformer 後端版（從零訓練，lr=1e-4，最多 {MAX_EPOCHS} 輪）...\n')
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
