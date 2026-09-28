"""
完整 Benchmark：freeze_conv vs s1_v3
涵蓋所有說話者
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
    ('GRID s5', 1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s5\*.mpg']),
    ('GRID s6',  987, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s6\*.mpg']),
    ('GRID s7', 1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s7\*.mpg']),
    ('GRID s13',1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s13\*.mpg']),
]

print('載入模型...')
m_fc  = load_model(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_freeze_conv.h5',
                   custom_objects={'CTCLoss':CTCLoss}, compile=False)
m_s1  = load_model(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_s1_v3.h5',
                   custom_objects={'CTCLoss':CTCLoss}, compile=False)
m_gm  = load_model(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_grid_multi.h5',
                   custom_objects={'CTCLoss':CTCLoss}, compile=False)
m_v3  = load_model(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_grid_multi_v3.h5',
                   custom_objects={'CTCLoss':CTCLoss}, compile=False)
m_v4  = load_model(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_grid_multi_v4.h5',
                   custom_objects={'CTCLoss':CTCLoss}, compile=False)
m_v5  = load_model(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_grid_multi_v5.h5',
                   custom_objects={'CTCLoss':CTCLoss}, compile=False)
m_v6  = load_model(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_grid_multi_v6.h5',
                   custom_objects={'CTCLoss':CTCLoss}, compile=False)
print('載入完成！\n')

fc_results = []
s1_results = []
gm_results = []
v3_results = []
v4_results = []
v5_results = []
v6_results = []

for label, n_expect, patterns in SPEAKERS:
    print(f'  測試 {label}...', flush=True)
    ds = make_ds(patterns)
    ok_fc, n_fc = test_model(m_fc, ds)
    ds = make_ds(patterns)
    ok_s1, n_s1 = test_model(m_s1, ds)
    ds = make_ds(patterns)
    ok_gm, n_gm = test_model(m_gm, ds)
    ds = make_ds(patterns)
    ok_v3, n_v3 = test_model(m_v3, ds)
    ds = make_ds(patterns)
    ok_v4, n_v4 = test_model(m_v4, ds)
    ds = make_ds(patterns)
    ok_v5, n_v5 = test_model(m_v5, ds)
    ds = make_ds(patterns)
    ok_v6, n_v6 = test_model(m_v6, ds)
    fc_results.append((label, n_fc, ok_fc))
    s1_results.append((label, n_s1, ok_s1))
    gm_results.append((label, n_gm, ok_gm))
    v3_results.append((label, n_v3, ok_v3))
    v4_results.append((label, n_v4, ok_v4))
    v5_results.append((label, n_v5, ok_v5))
    v6_results.append((label, n_v6, ok_v6))

# ── 輸出總表 ──
print('\n')
print('=' * 146)
print(f'{"說話者":<10} {"筆數":>5}   {"freeze_conv":>12} {"s1_v3":>10} {"gm_v2":>10} {"gm_v3":>10} {"gm_v4":>10} {"gm_v5":>10} {"gm_v6":>10}   {"v6差異":>7}')
print('=' * 146)

total_n = fc_wok = s1_wok = gm_wok = v3_wok = v4_wok = v5_wok = v6_wok = 0
for (label, n, fc_ok), (_, _, s1_ok), (_, _, gm_ok), (_, _, v3_ok), (_, _, v4_ok), (_, _, v5_ok), (_, _, v6_ok) in zip(fc_results, s1_results, gm_results, v3_results, v4_results, v5_results, v6_results):
    fc_pct = fc_ok/n*100
    s1_pct = s1_ok/n*100
    gm_pct = gm_ok/n*100
    v3_pct = v3_ok/n*100
    v4_pct = v4_ok/n*100
    v5_pct = v5_ok/n*100
    v6_pct = v6_ok/n*100
    diff   = v6_pct - v5_pct
    diff_s = f'+{diff:.1f}%' if diff >= 0 else f'{diff:.1f}%'
    print(f'{label:<10} {n:>5}   {fc_ok:>5}/{n:<5} {fc_pct:>5.1f}%   {s1_ok:>4}/{n:<5} {s1_pct:>5.1f}%   {gm_ok:>4}/{n:<5} {gm_pct:>5.1f}%   {v3_ok:>4}/{n:<5} {v3_pct:>5.1f}%   {v4_ok:>4}/{n:<5} {v4_pct:>5.1f}%   {v5_ok:>4}/{n:<5} {v5_pct:>5.1f}%   {v6_ok:>4}/{n:<5} {v6_pct:>5.1f}%   {diff_s:>7}')
    total_n += n; fc_wok += n*fc_pct; s1_wok += n*s1_pct; gm_wok += n*gm_pct; v3_wok += n*v3_pct; v4_wok += n*v4_pct; v5_wok += n*v5_pct; v6_wok += n*v6_pct

print('-' * 146)
fc_avg = fc_wok/total_n; s1_avg = s1_wok/total_n; gm_avg = gm_wok/total_n; v3_avg = v3_wok/total_n; v4_avg = v4_wok/total_n; v5_avg = v5_wok/total_n; v6_avg = v6_wok/total_n
diff_avg = v6_avg - v5_avg
diff_s = f'+{diff_avg:.1f}%' if diff_avg >= 0 else f'{diff_avg:.1f}%'
print(f'{"加權平均":<10} {total_n:>5}   {"":>11} {fc_avg:>5.1f}%   {"":>11} {s1_avg:>5.1f}%   {"":>11} {gm_avg:>5.1f}%   {"":>11} {v3_avg:>5.1f}%   {"":>11} {v4_avg:>5.1f}%   {"":>11} {v5_avg:>5.1f}%   {"":>11} {v6_avg:>5.1f}%   {diff_s:>7}')
print('=' * 146)
