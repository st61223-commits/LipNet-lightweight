"""
「新使用者個人化」第5個嘗試——換一個完全不同的技術：BatchNorm統計量重新校準
(test-time BN adaptation / AdaBN，查文獻確認是domain adaptation領域公認技術，
見 project_lipnet_direction.md 記錄的文獻)。

跟v1~v4(梯度微調LSTM+Dense或全模型)完全不同：這裡**不做任何梯度訓練**，
只是把 s9 校準資料以 training=True 模式做幾次前向傳播，讓 Conv3D 後面
三層 BatchNormalization 的 moving_mean/moving_variance 更新成「適合這個
新使用者視覺特徵分布」的統計量（不動任何學到的權重，包括BN自己的
gamma/beta），理論上完全不會過擬合(沒有標籤、沒有梯度)，計算成本也極低。

假設：v1~v4已經證實問題出在Conv3D產生的視覺特徵分布跟這位新使用者不合，
如果純粹是「統計量沒對齊」(例如這個人的膚色/打光造成的特徵分布偏移)，
不需要重新學權重，只要讓BN知道新的均值/變異數就可能有幫助。如果BN
重新校準也沒用，代表問題更深層(是特徵本身的判別力不夠，不只是沒對齊)。

做法：跟v4一樣切s9前300筆當校準、其餘700筆當測試，方便直接比較。

執行方式（WSL）：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/bn_adapt_s9.py
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
from vocab_correction import correct_sentence

CALIB_N = 300          # 跟v4用一樣的切分，方便直接比較
ADAPT_PASSES = 8       # 前向傳播跑幾輪讓moving average真正收斂到新統計量
BATCH_SIZE = 2
BASE_MODEL_PATH = os.path.join('models', 'trained_model_grid_multi_width05_datafix.h5')
SAVE_PATH = os.path.join('models', 'trained_model_width05_datafix_s9_bnadapt.h5')

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


def make_ds(file_list, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    if shuffle:
        ds = ds.shuffle(len(file_list), reshuffle_each_iteration=True)
    ds = ds.map(mappable)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds.padded_batch(BATCH_SIZE, padded_shapes=([75, None, None, 1], [40])).prefetch(tf.data.AUTOTUNE)


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
calib_files = all_files[:CALIB_N]
test_files = all_files[CALIB_N:]
print(f's9 總筆數: {len(all_files)}，校準(BN統計量用): {len(calib_files)}，測試: {len(test_files)}')

print('\n載入 width05_datafix 模型...')
model = load_model(BASE_MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)

bn_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
print(f'找到 {len(bn_layers)} 個 BatchNormalization 層，將重新校準其 moving_mean/moving_variance')
for l in bn_layers:
    print(f'  - {l.name}: 校準前 momentum={l.momentum}')

print(f'\n=== 開始BN統計量重新校準：{len(calib_files)}筆校準資料，跑{ADAPT_PASSES}輪前向傳播（無梯度、無標籤使用）===')
calib_ds = make_ds(calib_files, shuffle=True)
for p in range(ADAPT_PASSES):
    n_batches = 0
    for frames, labels in calib_ds:
        _ = model(frames, training=True)  # 只前向傳播，觸發BN內部moving average更新，不呼叫optimizer/梯度
        n_batches += 1
    print(f'  第 {p+1}/{ADAPT_PASSES} 輪前向傳播完成（{n_batches} batch）')

model.save(SAVE_PATH)
print(f'\nBN校準後模型已存: {SAVE_PATH}')

print(f'\n=== 校準後：測試集全部 {len(test_files)} 筆 ===')
test_ds = make_eval_ds(test_files)
ok1, n1 = test_model(model, test_ds)
pct1 = ok1 / n1 * 100 if n1 > 0 else 0
print(f'BN校準後（測試集全部）: {ok1}/{n1} = {pct1:.1f}%')

print('\n' + '=' * 60)
print(f'結論bn_adapt：s9的BN統計量重新校準(無梯度,{CALIB_N}筆校準,{ADAPT_PASSES}輪前向) → 測試集正確率 {pct1:.1f}%')
print(f'（對照：v1~v4梯度微調系列全部是0.0%，無任何校準的基準是0.1%）')
print('=' * 60)
