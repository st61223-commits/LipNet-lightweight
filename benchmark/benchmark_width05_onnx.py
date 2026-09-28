"""
Benchmark：F_Width0.5 的 ONNX (float32) 版本，驗證轉換後正確率有沒有受影響。
ONNX單次推論約158ms（比TFLite的2.35秒快15倍），所以這次直接跑全量資料集
（不像TFLite INT8benchmark那樣只能抽樣50筆/人），可以跟原本.h5的76.6%全量結果
直接對照。

在 tf215 環境跑（獨立環境，裝了 onnxruntime，不影響 py39）。
data/ 底下的快取檔案、alignments 都是純 numpy/文字檔，任何環境都讀得到，
不需要透過 py39。
"""
import os, sys, pathlib, warnings, io, time
pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
sys.path.append(r'C:\Users\Tno\claude-code')

import numpy as np, tensorflow as tf
import onnxruntime as ort
from vocab_correction import correct_sentence

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
    f.set_shape([75,46,140,1]); l.set_shape([40])
    return f, l

def make_ds(patterns):
    ds = tf.data.Dataset.list_files(patterns, shuffle=False).map(mappable)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds.padded_batch(1, padded_shapes=([75,46,140,1],[40])).prefetch(tf.data.AUTOTUNE)

ONNX_PATH = sys.argv[1] if len(sys.argv) > 1 else 'models/width05.onnx'
sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
IN_NAME = sess.get_inputs()[0].name
OUT_NAME = sess.get_outputs()[0].name

def predict_one(frames):
    return sess.run([OUT_NAME], {IN_NAME: frames.astype(np.float32)})[0]

def test_model(ds):
    ok = n = 0
    for frames, labels in ds.as_numpy_iterator():
        try:
            yhat = predict_one(frames)
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

# F_Width0.5 原始(.h5)版本全量benchmark參考數字（76.6%，2026-07-29）
WIDTH05_REF = {
    's99_1  ': 93.7, 's99_6  ': 100.0, 's99_7  ': 99.5,
    's99_8  ': 99.0, 's99_3+5': 0.0,  's34_3  ': 74.4,
    'GRID s1': 84.9, 'GRID s2': 13.1,  'GRID s5': 75.0,
    'GRID s6': 97.2, 'GRID s7': 93.5,  'GRID s13': 66.3,
}

print(f'ONNX模型: {ONNX_PATH}')
print(f'ONNX單次推論約158ms，比TFLite快15倍，這次直接跑全量資料集（不抽樣）\n')

t_start = time.time()
results = []
for label, n_expect, patterns in SPEAKERS:
    t0 = time.time()
    print(f'  測試 {label.strip()}（預期{n_expect}筆）...', flush=True)
    ds = make_ds(patterns)
    ok, n = test_model(ds)
    results.append((label, n, ok))
    pct = ok/n*100 if n > 0 else 0
    ref = WIDTH05_REF.get(label, 0)
    diff = pct - ref
    diff_s = f'+{diff:.1f}%' if diff >= 0 else f'{diff:.1f}%'
    print(f'    -> {ok}/{n} = {pct:.1f}%  (原始.h5全量: {ref:.1f}%  差異: {diff_s})  用時{time.time()-t0:.0f}秒', flush=True)

print(f'\n總耗時: {(time.time()-t_start)/60:.1f} 分鐘\n')
print('=' * 70)
print(f'{"說話者":<10} {"筆數":>5}   {"ONNX":>10}   {"原始.h5":>10}   {"差異":>8}')
print('=' * 70)

total_n = q_wok = ref_wok_total = 0
for label, n, ok in results:
    pct = ok/n*100 if n > 0 else 0
    ref = WIDTH05_REF.get(label, 0)
    diff = pct - ref
    diff_s = f'+{diff:.1f}%' if diff >= 0 else f'{diff:.1f}%'
    print(f'{label:<10} {n:>5}   {ok:>4}/{n:<5} {pct:>5.1f}%   {"":>5} {ref:>5.1f}%   {diff_s:>8}')
    total_n += n
    q_wok += n * pct
    ref_wok_total += n * ref

print('-' * 70)
q_avg = q_wok / total_n
ref_avg = ref_wok_total / total_n
diff_avg = q_avg - ref_avg
diff_s = f'+{diff_avg:.1f}%' if diff_avg >= 0 else f'{diff_avg:.1f}%'
print(f'{"加權平均":<10} {total_n:>5}   {"":>11} {q_avg:>5.1f}%   {"":>11} {ref_avg:>5.1f}%   {diff_s:>8}')
print('=' * 70)
