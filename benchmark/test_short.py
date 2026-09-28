"""
比較 trained_model_balanced_s1_short.h5（5輪保守訓練）
vs  trained_model_freeze_conv.h5（原始最佳）
"""
import os, sys, pathlib, warnings
pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore")

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

def test_model(mdl, ds, label):
    raw_ok = corr_ok = n = 0
    for frames, labels in ds.as_numpy_iterator():
        yhat = mdl.predict(frames, verbose=0)
        dec = tf.keras.backend.ctc_decode(
            tf.cast(yhat, tf.float32), [75] * len(yhat), greedy=False)[0][0].numpy()
        orig = ' '.join(tf.strings.reduce_join(num_to_char(labels[0])).numpy().decode().strip().split())
        raw  = ' '.join(w for w in tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split() if len(w) > 1)
        corr = ' '.join(correct_sentence(raw).split())
        if orig == raw:  raw_ok  += 1
        if orig == corr: corr_ok += 1
        n += 1
    r = raw_ok  / n * 100
    c = corr_ok / n * 100
    print(f'  {label:<28} {n:>5} 筆   原始 {r:>5.1f}%   校正後 {c:>5.1f}%')
    return r, c

SPEAKERS = [
    ('s99_6（200支）',      200,  [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6\*.mpg',
                                   r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6_new\*.mpg']),
    ('s99_7+new（400支）',  400,  [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7\*.mpg',
                                   r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7_new\*.mpg']),
    ('s99_8（100支）',      100,  [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_8\*.mpg']),
    ('s99_1（1000支）',    1000,  [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_1\*.mpg']),
    ('s34_3（199支）',      199,  [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s34_3\*.mpg']),
    ('s99_3+s99_5（40支）',  40,  [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_3\*.mpg',
                                   r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_5\*.mpg']),
    ('GRID s1（1000支）',  1000,  [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s1\*.mpg']),
]

print('載入模型中...')
new_model = load_model(
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\trained_model_balanced_v2.h5',
    custom_objects={'CTCLoss': CTCLoss})
old_model = load_model(
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\trained_model_freeze_conv.h5',
    custom_objects={'CTCLoss': CTCLoss})
print('模型載入完成！\n')

print('=' * 65)
print('【新模型】trained_model_balanced_v2.h5（10輪，50/50，lr=5e-6）')
print('=' * 65)
new_results = []
for label, n, patterns in SPEAKERS:
    ds = make_ds(patterns)
    r, c = test_model(new_model, ds, label)
    new_results.append((label, n, r, c))

print('\n' + '=' * 65)
print('【基準】trained_model_freeze_conv.h5')
print('=' * 65)
old_results = []
for label, n, patterns in SPEAKERS:
    ds = make_ds(patterns)
    r, c = test_model(old_model, ds, label)
    old_results.append((label, n, r, c))

print('\n\n' + '=' * 90)
print(f'{"比較總表":^90}')
print('=' * 90)
print(f'{"說話者":<24} {"筆數":>5}   {"新模型原始":>10} {"新模型校正":>10}   {"舊模型原始":>10} {"舊模型校正":>10} {"差異":>7}')
print('-' * 90)

total_vids = 0
new_wr = new_wc = old_wr = old_wc = 0

for (label, n, nr, nc), (_, _, or_, oc) in zip(new_results, old_results):
    diff = nc - oc
    diff_str = f'+{diff:.1f}%' if diff >= 0 else f'{diff:.1f}%'
    print(f'{label:<24} {n:>5}   {nr:>10.1f}% {nc:>10.1f}%   {or_:>10.1f}% {oc:>10.1f}% {diff_str:>7}')
    total_vids += n; new_wr += n*nr; new_wc += n*nc; old_wr += n*or_; old_wc += n*oc

print('-' * 90)
na_r = new_wr/total_vids; na_c = new_wc/total_vids
oa_r = old_wr/total_vids; oa_c = old_wc/total_vids
diff_avg = na_c - oa_c
diff_str = f'+{diff_avg:.1f}%' if diff_avg >= 0 else f'{diff_avg:.1f}%'
print(f'{"加權平均":<24} {total_vids:>5}   {na_r:>10.1f}% {na_c:>10.1f}%   {oa_r:>10.1f}% {oa_c:>10.1f}% {diff_str:>7}')
print('=' * 90)

print()
if diff_avg > 0:
    print(f'結論：新模型比舊模型提升 {diff_avg:.1f}%，建議換用新模型！')
elif diff_avg == 0:
    print('結論：兩個模型準確率相同。')
else:
    print(f'結論：新模型比舊模型下降 {abs(diff_avg):.1f}%，繼續使用 freeze_conv。')
