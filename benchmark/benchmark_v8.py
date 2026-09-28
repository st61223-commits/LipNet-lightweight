"""
Benchmark：grid_multi v8 單模型測試
"""
import os, sys, pathlib, warnings, io
pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
sys.path.append(r'C:\Users\Tno\claude-code')

import numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
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

def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl,1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl,1), dtype="int64")
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
        parts = [x for x in p.replace('\\\\','/').replace('\\','/').split('/') if x]
        project_name = parts[-2]
        frames = _load_frames(os.path.join('data', project_name, f'{file_name}.mpg'))
        alignments = load_alignments(os.path.join(
            'data', 'alignments', project_name.replace('_new',''), f'{file_name}.align'))
    except:
        frames = np.zeros((75,46,140,1), dtype=np.float32)
        alignments = np.zeros((1,), dtype=np.int32)
    return frames, alignments

def mappable(path):
    f, l = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    f.set_shape([75,None,None,1]); l.set_shape([40])
    return f, l

def make_ds(patterns):
    ds = tf.data.Dataset.list_files(patterns, shuffle=False).map(mappable)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds.padded_batch(1, padded_shapes=([75,None,None,1],[40])).prefetch(tf.data.AUTOTUNE)

def test_model(mdl, ds):
    ok = n = 0
    for frames, labels in ds.as_numpy_iterator():
        try:
            yhat = mdl.predict(frames, verbose=0)
            dec = tf.keras.backend.ctc_decode(
                tf.cast(yhat, tf.float32), [75]*len(yhat), greedy=False)[0][0].numpy()
            orig = ' '.join(tf.strings.reduce_join(num_to_char(labels[0])).numpy().decode().strip().split())
            raw  = ' '.join(w for w in tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split() if len(w)>1)
            corr = ' '.join(correct_sentence(raw).split())
            if orig == corr: ok += 1
            n += 1
        except Exception:
            pass
    return ok, n

SPEAKERS = [
    ('s99_1  ', 1000, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_1\*.mpg']),
    ('s99_6  ',  200, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6\*.mpg',
                       r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6_new\*.mpg']),
    ('s99_7  ',  400, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7\*.mpg',
                       r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7_new\*.mpg']),
    ('s99_8  ',  100, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_8\*.mpg']),
    ('s99_3+5',   40, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_3\*.mpg',
                       r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_5\*.mpg']),
    ('s34_3  ',  199, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s34_3\*.mpg']),
    ('GRID s1', 1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s1\*.mpg']),
    ('GRID s2', 1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s2\*.mpg']),
    ('GRID s3', 1000, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s3\*.mpg']),   # 新說話者
    ('GRID s4',  966, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s4\*.mpg']),   # 新說話者
    ('GRID s5', 1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s5\*.mpg']),
    ('GRID s6',  987, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s6\*.mpg']),
    ('GRID s7', 1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s7\*.mpg']),
    ('GRID s8', 1000, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s8\*.mpg']),   # 新說話者
    ('GRID s13',1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s13\*.mpg']),
]

# v7 參考數字（2026-06-18 benchmark）
V7_REF = {
    's99_1  ': 99.9, 's99_6  ': 100.0, 's99_7  ': 99.5,
    's99_8  ': 100.0, 's99_3+5': 0.0,  's34_3  ': 78.9,
    'GRID s1': 42.7, 'GRID s2': 10.5,  'GRID s3': 0.0,
    'GRID s4': 0.0,  'GRID s5': 66.4,  'GRID s6': 93.8,
    'GRID s7': 83.5, 'GRID s8': 0.0,   'GRID s13': 42.9,
}

print('載入 v8 模型...')
m_v8 = load_model(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_grid_multi_v8.h5',
                  custom_objects={'CTCLoss': CTCLoss}, compile=False)
print('載入完成！\n')

results = []
for label, n_expect, patterns in SPEAKERS:
    print(f'  測試 {label.strip()}...', flush=True)
    ds = make_ds(patterns)
    ok, n = test_model(m_v8, ds)
    results.append((label, n, ok))
    pct = ok/n*100 if n > 0 else 0
    v7 = V7_REF.get(label, 0)
    diff = pct - v7
    diff_s = f'+{diff:.1f}%' if diff >= 0 else f'{diff:.1f}%'
    print(f'    → {ok}/{n} = {pct:.1f}%  (v7: {v7:.1f}%  差異: {diff_s})', flush=True)

print('\n')
print('=' * 70)
print(f'{"說話者":<10} {"筆數":>5}   {"gm_v8":>10}   {"gm_v7":>10}   {"差異":>8}')
print('=' * 70)

total_n = v8_wok = v7_wok_total = 0
for label, n, ok in results:
    pct = ok/n*100 if n > 0 else 0
    v7  = V7_REF.get(label, 0)
    diff = pct - v7
    diff_s = f'+{diff:.1f}%' if diff >= 0 else f'{diff:.1f}%'
    print(f'{label:<10} {n:>5}   {ok:>4}/{n:<5} {pct:>5.1f}%   {"":>5} {v7:>5.1f}%   {diff_s:>8}')
    total_n    += n
    v8_wok     += n * pct
    v7_wok_total += n * v7

print('-' * 70)
v8_avg = v8_wok / total_n
v7_avg = v7_wok_total / total_n
diff_avg = v8_avg - v7_avg
diff_s = f'+{diff_avg:.1f}%' if diff_avg >= 0 else f'{diff_avg:.1f}%'
print(f'{"加權平均":<10} {total_n:>5}   {"":>11} {v8_avg:>5.1f}%   {"":>11} {v7_avg:>5.1f}%   {diff_s:>8}')
print('=' * 70)
