"""
Benchmark：grid_multi F_Width0.5「資料修復後續訓練版」（width05_datafix）

背景：2026-08-27 發現嘴唇偵測信心度門檻太嚴格，約20%訓練資料快取是全黑空白影片
（s2 高達82.7%）。已修復資料，並用現役 F_Width0.5 權重暖啟動，接續訓練30輪
（val_loss 9.83 → 4.80）。這支腳本驗證修復+續訓後正確率是否真的進步。

同時新增 s9（訓練時特意保留、完全沒訓練過的乾淨測試對象），檢驗資料修復
是否有改善「對全新使用者辨識率0%」的問題（見 實驗記錄.md 實驗五）。

三欄比較：datafix（新） vs F_Width0.5原版（現役，76.6%） vs v11（67.9%）
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
# 注意：這裡刻意不開 mixed_precision——訓練時也沒開，測試要跟訓練條件一致，
# 避免精度策略不同造成推論結果跟訓練時的行為不一致。

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
    ('GRID s5', 1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s5\*.mpg']),
    ('GRID s6',  987, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s6\*.mpg']),
    ('GRID s7', 1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s7\*.mpg']),
    ('GRID s13',1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s13\*.mpg']),
    # 新增：s9，訓練時特意保留、兩個模型都完全沒訓練過，測「對全新使用者」的真實表現
    ('s9     ', 1000, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s9\*.mpg']),
]

# v11 參考數字（2026-07-07，加權平均67.9%）
V11_REF = {
    's99_1  ': 99.9, 's99_6  ': 100.0, 's99_7  ': 100.0,
    's99_8  ': 100.0, 's99_3+5': 0.0,  's34_3  ': 78.4,
    'GRID s1': 49.2, 'GRID s2': 10.6,  'GRID s5': 67.5,
    'GRID s6': 94.9, 'GRID s7': 85.6,  'GRID s13': 46.2,
}

# F_Width0.5 原版參考數字（2026-07-29，65輪訓練完成，加權平均76.6%，現役模型）
W05_ORIG_REF = {
    's99_1  ': 93.7, 's99_6  ': 100.0, 's99_7  ': 99.5,
    's99_8  ': 99.0, 's99_3+5': 0.0,  's34_3  ': 74.4,
    'GRID s1': 84.9, 'GRID s2': 13.1,  'GRID s5': 75.0,
    'GRID s6': 97.2, 'GRID s7': 93.5,  'GRID s13': 66.3,
}

print('載入 width05_datafix 模型...')
m_dfx = load_model(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_grid_multi_width05_datafix.h5',
                   custom_objects={'CTCLoss': CTCLoss}, compile=False)
print('載入完成！\n')

results = []
for label, n_expect, patterns in SPEAKERS:
    print(f'  測試 {label.strip()}...', flush=True)
    ds = make_ds(patterns)
    ok, n = test_model(m_dfx, ds)
    results.append((label, n, ok))
    pct = ok/n*100 if n > 0 else 0
    w05 = W05_ORIG_REF.get(label)
    w05_s = f'{w05:.1f}%' if w05 is not None else 'N/A(新測)'
    print(f'    → {ok}/{n} = {pct:.1f}%  (F_Width0.5原版: {w05_s})', flush=True)

print('\n')
print('=' * 90)
print(f'{"說話者":<10} {"筆數":>5}   {"datafix":>10}   {"w05原版":>10}   {"差異":>8}   {"v11":>8}')
print('=' * 90)

total_n = dfx_wok = w05_wok_total = v11_wok_total = 0
for label, n, ok in results:
    pct = ok/n*100 if n > 0 else 0
    w05 = W05_ORIG_REF.get(label)
    v11 = V11_REF.get(label)
    if w05 is not None:
        diff = pct - w05
        diff_s = f'+{diff:.1f}%' if diff >= 0 else f'{diff:.1f}%'
        w05_s = f'{w05:>7.1f}%'
        w05_wok_total += n * w05
    else:
        diff_s = '(新測)'
        w05_s = '     N/A'
    v11_s = f'{v11:>7.1f}%' if v11 is not None else '     N/A'
    if v11 is not None:
        v11_wok_total += n * v11
    print(f'{label:<10} {n:>5}   {ok:>4}/{n:<5} {pct:>5.1f}%   {w05_s}   {diff_s:>8}   {v11_s}')
    total_n += n
    dfx_wok += n * pct

print('-' * 90)
dfx_avg_all = dfx_wok / total_n

# 只用「原版w05也有測過」的說話者做公平比較（排除s9這個新增的held-out測試）
n_comparable = 0
dfx_wok_comparable = 0
for label, n, ok in results:
    if W05_ORIG_REF.get(label) is not None:
        pct = ok/n*100 if n > 0 else 0
        n_comparable += n
        dfx_wok_comparable += n * pct
dfx_avg_comparable = dfx_wok_comparable / n_comparable if n_comparable else 0
w05_avg = w05_wok_total / n_comparable if n_comparable else 0
diff_avg = dfx_avg_comparable - w05_avg
diff_avg_s = f'+{diff_avg:.1f}%' if diff_avg >= 0 else f'{diff_avg:.1f}%'

print(f'{"加權平均(含s9)":<14} {total_n:>5}   {"":>11} {dfx_avg_all:>5.1f}%')
print(f'{"加權平均(可比對)":<14} {n_comparable:>5}   {"":>11} {dfx_avg_comparable:>5.1f}%   {w05_avg:>7.1f}%   {diff_avg_s:>8}   ← datafix vs w05原版，排除s9')
print('=' * 90)
