"""
測試 trained_model_s1_v3.h5（或 checkpoint）對 GRID s1 的真正準確率
同時和 freeze_conv 做對比
"""
import os, sys, pathlib, warnings, io
pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
sys.path.append(r'C:\Users\Tno\claude-code')

import numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from vocab_correction import correct_sentence

physical_devices = tf.config.list_physical_devices('GPU')
try: tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: pass
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

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
    with open(path, 'r') as f: lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split(); word = parts[2]
        if word not in ('sil', 'sp') and len(word) > 1:
            tokens.extend([' ', word])
    tokens_flat = tokens[1:]
    if not tokens_flat: return tf.zeros([1], dtype=tf.int64)
    return char_to_num(tf.reshape(
        tf.strings.unicode_split(tokens_flat, input_encoding='UTF-8'), (-1)))

def load_data(path):
    try:
        p = bytes.decode(path.numpy())
        file_name = os.path.splitext(os.path.basename(p))[0]
        parts = [x for x in p.replace('\\\\', '/').replace('\\', '/').split('/') if x]
        project_name = parts[-2]
        frames = _load_frames(os.path.join('data', project_name, f'{file_name}.mpg'))
        alignments = load_alignments(os.path.join(
            'data', 'alignments', project_name.replace('_new', ''), f'{file_name}.align'))
    except:
        frames = np.zeros((75, 46, 140, 1), dtype=np.float32)
        alignments = np.zeros((1,), dtype=np.int32)
    return frames, alignments

def mappable(path):
    f, l = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    f.set_shape([75, None, None, 1]); l.set_shape([40])
    return f, l

def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)

def make_ds(patterns):
    ds = tf.data.Dataset.list_files(patterns).map(mappable)
    return ds.padded_batch(1, padded_shapes=([75, None, None, 1], [40])).prefetch(tf.data.AUTOTUNE)

def test_model(mdl, ds, label, show_examples=5):
    raw_ok = corr_ok = n = 0
    examples = []
    for frames, labels in ds.as_numpy_iterator():
        yhat = mdl.predict(frames, verbose=0)
        dec = tf.keras.backend.ctc_decode(
            tf.cast(yhat, tf.float32), [75] * len(yhat), greedy=False)[0][0].numpy()
        orig = ' '.join(tf.strings.reduce_join(num_to_char(labels[0])).numpy().decode().strip().split())
        raw  = ' '.join(w for w in tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split() if len(w) > 1)
        corr = ' '.join(correct_sentence(raw).split())
        if orig == raw:  raw_ok  += 1
        if orig == corr: corr_ok += 1
        if n < show_examples:
            examples.append((orig, raw, corr, orig == corr))
        n += 1
    r = raw_ok  / n * 100
    c = corr_ok / n * 100
    print(f'  {label:<30} {n:>5} 筆   原始 {r:>5.1f}%   校正後 {c:>5.1f}%')
    for orig, raw, corr, ok in examples:
        mark = '✓' if ok else '✗'
        print(f'    {mark} 答案:{orig!r:35s}  原始:{raw!r:35s}  校正:{corr!r}')
    return r, c

SPEAKERS = [
    ('s99_6（200支）',     [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6\*.mpg',
                             r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6_new\*.mpg']),
    ('s99_1（1000支）',    [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_1\*.mpg']),
    ('GRID s1（1000支）',  [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s1\*.mpg']),
]

print('載入模型中...')

# 嘗試載入完整訓練後的 s1_v3 模型，若不存在則從 checkpoint 重建
s1v3_path = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\trained_model_s1_v3.h5'
ckpt_path  = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\checkpoint_s1_v3'

if os.path.exists(s1v3_path):
    print(f'載入 trained_model_s1_v3.h5 ...')
    new_model = load_model(s1v3_path, custom_objects={'CTCLoss': CTCLoss})
    model_label = 'trained_model_s1_v3.h5'
else:
    print(f'找不到 trained_model_s1_v3.h5，從 checkpoint 重建 ...')
    new_model = load_model(
        r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\trained_model_freeze_conv.h5',
        custom_objects={'CTCLoss': CTCLoss})
    for layer in new_model.layers:
        if 'conv3d' in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True
    new_model.compile(optimizer=Adam(learning_rate=2e-5, clipnorm=1.0), loss=CTCLoss)
    new_model.load_weights(ckpt_path)
    model_label = 'checkpoint_s1_v3（訓練中最佳）'
    print(f'  已載入 checkpoint 權重')

old_model = load_model(
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\trained_model_freeze_conv.h5',
    custom_objects={'CTCLoss': CTCLoss})
print('模型載入完成！\n')

print('=' * 80)
print(f'【新模型】{model_label}')
print('=' * 80)
for label, patterns in SPEAKERS:
    ds = make_ds(patterns)
    test_model(new_model, ds, label, show_examples=3)

print('\n' + '=' * 80)
print('【基準】trained_model_freeze_conv.h5')
print('=' * 80)
for label, patterns in SPEAKERS:
    ds = make_ds(patterns)
    test_model(old_model, ds, label, show_examples=3)
