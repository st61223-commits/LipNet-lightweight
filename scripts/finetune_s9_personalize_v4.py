"""
「新使用者個人化校準」實驗第3輪——測試「凍結Conv3D」是不是v1/v2失敗的真正原因。

v1（50筆校準，無validation，lr=1e-4，凍結Conv3D）：訓練loss降到0.79但測試集0.0%（過擬合）
v2（100筆校準，有validation+early stop，lr=3e-5，凍結Conv3D）：early stop在val_loss 39.4，
   測試集依然0.0%——val_loss遠高於「有可用正確率」的門檻(~15以下)，代表凍結Conv3D後
   決策層(LSTM+Dense)光靠通用視覺特徵，資訊量不夠分辨這位新使用者的嘴型

這次改成全模型解凍（含Conv3D），用很低的學習率(1e-5，比照v10→v11微調時的量級)，
測試「讓視覺特徵也一起適應這個人」是否才是關鍵。

其餘設計比照v2：100筆校準(85訓練/15驗證)、EarlyStopping、其餘850筆當測試集。

執行方式（WSL）：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/finetune_s9_personalize_v3.py
"""
import os
import sys
import glob
import pathlib
import warnings
import numpy as np
import tensorflow as tf

pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)
sys.path.append(r'C:\Users\Tno\claude-code')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from vocab_correction import correct_sentence

CALIB_N = 300
CALIB_VAL_N = 15
MAX_EPOCHS = 40
BATCH_SIZE = 2
BASE_MODEL_PATH = os.path.join('models', 'trained_model_grid_multi_width05_datafix.h5')
SAVE_PATH = os.path.join('models', 'trained_model_width05_datafix_s9personalized_v4.h5')

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)


def _load_frames(path):
    project_name = os.path.basename(os.path.dirname(path))
    file_name = os.path.splitext(os.path.basename(path))[0]
    for suffix in ['_cached', '_dlib_cached']:
        cache_path = os.path.join('data', f'{project_name}{suffix}', f'{file_name}.npy')
        if os.path.exists(cache_path):
            frames = np.load(cache_path)
            mean, std = np.mean(frames), np.std(frames) + 1e-6
            return ((frames - mean) / std).astype(np.float32)
    raise FileNotFoundError(f'找不到快取：{project_name}/{file_name}')


def load_alignments(path):
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
        p = bytes.decode(path.numpy())
        file_name = os.path.splitext(os.path.basename(p))[0]
        parts = [x for x in p.replace('\\\\', '/').replace('\\', '/').split('/') if x]
        project_name = parts[-2]
        frames = _load_frames(os.path.join('data', project_name, f'{file_name}.mpg'))
        alignments = load_alignments(os.path.join('data', 'alignments', project_name, f'{file_name}.align'))
    except Exception as e:
        print(f'[SKIPPED] {path} -> {e}')
        frames = np.zeros((75, 46, 140, 1), dtype=np.float32)
        alignments = np.zeros((1,), dtype=np.int32)
    return frames, alignments


def mappable(path):
    f, l = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    f.set_shape([75, None, None, 1])
    l.set_shape([40])
    return f, l


def make_train_ds(file_list, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    if shuffle:
        ds = ds.shuffle(len(file_list), reshuffle_each_iteration=True)
    ds = ds.map(mappable)
    ds = ds.padded_batch(BATCH_SIZE, padded_shapes=([75, None, None, 1], [40]))
    return ds.prefetch(tf.data.AUTOTUNE)


def make_eval_ds(file_list):
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    ds = ds.map(mappable)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds.padded_batch(1, padded_shapes=([75, None, None, 1], [40])).prefetch(tf.data.AUTOTUNE)


def test_model(mdl, ds):
    ok = n = 0
    for frames, labels in ds.as_numpy_iterator():
        try:
            yhat = mdl.predict(frames, verbose=0)
            dec = tf.keras.backend.ctc_decode(
                tf.cast(yhat, tf.float32), [75] * len(yhat), greedy=False)[0][0].numpy()
            orig = ' '.join(tf.strings.reduce_join(num_to_char(labels[0])).numpy().decode().strip().split())
            raw = ' '.join(w for w in tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split() if len(w) > 1)
            corr = ' '.join(correct_sentence(raw).split())
            if orig == corr:
                ok += 1
            n += 1
        except Exception:
            pass
    return ok, n


all_files = sorted(glob.glob(os.path.join('data', 's9', '*.mpg')))
print(f's9 總筆數: {len(all_files)}')
calib_files = all_files[:CALIB_N]
calib_train = calib_files[:-CALIB_VAL_N]
calib_val = calib_files[-CALIB_VAL_N:]
test_files = all_files[CALIB_N:]
print(f'校準訓練: {len(calib_train)} 筆，校準驗證: {len(calib_val)} 筆，測試集: {len(test_files)} 筆')

print('\n載入 width05_datafix 模型當起點...')
model = load_model(BASE_MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)

# ── 這次全模型解凍（含Conv3D），測試視覺特徵也一起適應是否才是關鍵
for layer in model.layers:
    layer.trainable = True
print(f'全部 {len(model.layers)} 層皆可訓練（含Conv3D）')

model.compile(optimizer=Adam(learning_rate=1e-5, clipnorm=1.0), loss=CTCLoss)

train_ds = make_train_ds(calib_train, shuffle=True)
val_ds = make_train_ds(calib_val, shuffle=False)

print(f'\n=== 開始個人化微調 v4（全模型解凍，lr=1e-5，300筆校準）：{len(calib_train)}筆訓練+{len(calib_val)}筆驗證，最多{MAX_EPOCHS}輪，早停patience=5 ===')
callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)]
history = model.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS, verbose=2, callbacks=callbacks)

model.save(SAVE_PATH)
print(f'\n個人化模型v4已存: {SAVE_PATH}')

best_val = min(history.history['val_loss'])
print(f'最佳 val_loss: {best_val:.4f}（第{history.history["val_loss"].index(best_val)+1}輪）')

print(f'\n=== 微調後：測試集全部 {len(test_files)} 筆 ===')
test_ds = make_eval_ds(test_files)
ok1, n1 = test_model(model, test_ds)
pct1 = ok1 / n1 * 100 if n1 > 0 else 0
print(f'微調後（測試集全部）: {ok1}/{n1} = {pct1:.1f}%')

print('\n' + '=' * 60)
print(f'結論v4：s9個人化微調(全模型解凍,lr=1e-5,300筆校準) → 測試集正確率 {pct1:.1f}%')
print(f'（對照：v2凍結Conv3D是0.0%(val_loss卡在39.4)，v1是0.0%(過擬合)，無微調基準是0.1%）')
print('=' * 60)
