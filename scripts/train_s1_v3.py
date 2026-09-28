"""
train_s1_v3.py
目標：讓模型學會辨識 GRID s1 說話者

修正 balanced_v2 的兩個問題：
  1. load_alignments 改為排除單字元位置字母（與測試腳本對齊）
  2. s1:s99_1 = 80:20（更積極讓 s1 梯度主導），LR 提高到 2e-5
"""
import os, glob, random, pathlib, warnings, io, sys
pathlib.PosixPath = pathlib.WindowsPath
warnings.filterwarnings('ignore')

os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print(f'GPU 已啟用: {physical_devices[0]}')

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

def load_video_from_cache(path):
    project_name = os.path.basename(os.path.dirname(path))
    file_name = os.path.splitext(os.path.basename(path))[0]
    for suffix in ['_cached', '_dlib_cached']:
        cache_path = os.path.join('data', f'{project_name}{suffix}', f'{file_name}.npy')
        if os.path.exists(cache_path):
            frames = np.load(cache_path)
            mean, std = np.mean(frames), np.std(frames) + 1e-6
            return ((frames - mean) / std).astype(np.float32)
    raise FileNotFoundError(f'找不到快取: {project_name}/{file_name}')

def load_alignments(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split()
        word = parts[2]
        # 修正：排除 sil、sp 以及單字元位置字母（如 f、l、g）
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
        frames = load_video_from_cache(os.path.join('data', project_name, f'{file_name}.mpg'))
        align_name = project_name.replace('_new', '')
        alignments = load_alignments(
            os.path.join('data', 'alignments', align_name, f'{file_name}.align'))
    except Exception as e:
        frames = np.zeros((75, 46, 140, 1), dtype=np.float32)
        alignments = np.zeros((1,), dtype=np.int32)
    return frames, alignments

def mappable_function(path):
    f, l = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    f.set_shape([75, None, None, 1])
    l.set_shape([40])
    return f, l

def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)

# 每輪結束後印一個 s1 和一個 s99_1 的預測範例
class ProduceExample(tf.keras.callbacks.Callback):
    def __init__(self, s1_ds, s99_ds):
        self._s1_iter = s1_ds.as_numpy_iterator()
        self._s99_iter = s99_ds.as_numpy_iterator()
        self._s1_ds = s1_ds
        self._s99_ds = s99_ds

    def _next(self, it, ds_ref):
        try:
            return it.next()
        except StopIteration:
            new_it = ds_ref.as_numpy_iterator()
            return new_it.next()

    def on_epoch_end(self, epoch, logs=None):
        for tag, it, ds in [('s1', self._s1_iter, self._s1_ds),
                             ('s99_1', self._s99_iter, self._s99_ds)]:
            data = self._next(it, ds)
            yhat = self.model.predict(data[0], verbose=0)
            dec = tf.keras.backend.ctc_decode(
                tf.cast(yhat, tf.float32), [75], greedy=False)[0][0].numpy()
            orig = tf.strings.reduce_join(num_to_char(data[1][0])).numpy().decode().strip()
            pred = tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip()
            print(f'  [{tag}] 答案: {orig!r}   預測: {pred!r}')

# ── 資料集準備 ──
s1_files   = sorted(glob.glob(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s1\*.mpg'))
s99_1_files = sorted(glob.glob(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_1\*.mpg'))

random.seed(42)
random.shuffle(s1_files)
random.shuffle(s99_1_files)

# s1 全部 1000 筆；s99_1 取 250 筆（80:20 比例）
n_s1   = len(s1_files)          # 1000
n_s99  = min(250, len(s99_1_files))  # 250
s99_1_sampled = random.sample(s99_1_files, n_s99)

print(f's1: {n_s1} 筆，s99_1: {n_s99} 筆（比例 {n_s1/(n_s1+n_s99)*100:.0f}:{n_s99/(n_s1+n_s99)*100:.0f}）')

all_files = s1_files + s99_1_sampled
random.shuffle(all_files)

data_all = tf.data.Dataset.from_tensor_slices(all_files)
data_all = data_all.shuffle(len(all_files), reshuffle_each_iteration=False)
data_all = data_all.map(mappable_function)
data_all = data_all.padded_batch(1, padded_shapes=([75, None, None, 1], [40]))
data_all = data_all.prefetch(tf.data.AUTOTUNE)

total    = tf.data.experimental.cardinality(data_all).numpy()
train_sz = int(total * 0.9)
val_sz   = total - train_sz
print(f'總計 {total} 筆 → 訓練: {train_sz}，驗證: {val_sz}')

train_ds = data_all.take(train_sz)
val_ds   = data_all.skip(train_sz)

# 範例用（各取 1 筆固定做對照）
s1_sample_ds  = tf.data.Dataset.from_tensor_slices(s1_files[:5]).map(mappable_function)\
    .padded_batch(1, padded_shapes=([75, None, None, 1], [40]))
s99_sample_ds = tf.data.Dataset.from_tensor_slices(s99_1_files[:5]).map(mappable_function)\
    .padded_batch(1, padded_shapes=([75, None, None, 1], [40]))

# ── 載入 freeze_conv 並解凍 LSTM/Dense ──
print('\n載入 trained_model_freeze_conv.h5 ...')
model = load_model(
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\trained_model_freeze_conv.h5',
    custom_objects={'CTCLoss': CTCLoss})

frozen_layers = []
for layer in model.layers:
    if 'conv3d' in layer.name:
        layer.trainable = False
        frozen_layers.append(layer.name)
    else:
        layer.trainable = True

print(f'凍結 Conv3D: {frozen_layers}')
print(f'可訓練參數：{sum(np.prod(v.shape) for v in model.trainable_variables):,}')

model.compile(optimizer=Adam(learning_rate=2e-5, clipnorm=1.0), loss=CTCLoss)
print('開始訓練（最多 30 輪，lr=2e-5，patience=5）...\n')

os.makedirs('models', exist_ok=True)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[
        ModelCheckpoint(
            os.path.join('models', 'checkpoint_s1_v3'),
            monitor='val_loss', save_weights_only=True, save_best_only=True,
            verbose=1),
        EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1),
        ProduceExample(s1_sample_ds, s99_sample_ds),
    ]
)

model.save(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\trained_model_s1_v3.h5')
print('[OK] 已儲存 trained_model_s1_v3.h5')

# 印出最終 val_loss 曲線
print('\n每輪 val_loss:')
for i, v in enumerate(history.history.get('val_loss', []), 1):
    print(f'  Epoch {i:2d}: {v:.4f}')
