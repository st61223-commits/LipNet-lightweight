"""
Benchmark（WER版，修正版）：v11（學長基準／完整寬度架構）補算 WER。
修正上次 benchmark_v11_wer_quick.py 的兩個bug：
1. 漏了 mixed_precision.set_global_policy('mixed_float16')（v11原本的載入方式就有設定這行，
   benchmark_v11.py 裡有，quick版漏掉了，可能導致float16/float32精度不匹配讓輸出失真）
2. 快取路徑寫死指到錯誤資料夾(s1_cached_v5_gridquick)，這次改用新重建的完整版 _cached_v5_full
讀取 _cached_v5_full 快取（用原始YOLOv5權重重新產生，跟v11訓練條件一致）。
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

CACHE_SUFFIX = '_cached_v5_full'

def _load_frames(path):
    project_name = os.path.basename(os.path.dirname(path))
    file_name = os.path.splitext(os.path.basename(path))[0]
    for suffix in [CACHE_SUFFIX, '_cached', '_dlib_cached']:
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

MISS = {'count': 0}
def load_data(path):
    try:
        p = bytes.decode(path.numpy())
        file_name = os.path.splitext(os.path.basename(p))[0]
        parts = [x for x in p.replace('\\\\','/').replace('\\','/').split('/') if x]
        project_name = parts[-2]
        frames = _load_frames(os.path.join('data', project_name, f'{file_name}.mpg'))
        alignments = load_alignments(os.path.join(
            'data', 'alignments', project_name.replace('_new',''), f'{file_name}.align'))
    except Exception:
        MISS['count'] += 1
        frames = np.zeros((75,46,140,1), dtype=np.float32)
        alignments = np.zeros((1,), dtype=np.int32)
    return frames, alignments

def mappable(path):
    f, l = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    f.set_shape([75,None,None,1]); l.set_shape([40])
    return f, l

def make_ds(patterns, limit=None):
    ds = tf.data.Dataset.list_files(patterns, shuffle=False)
    if limit is not None:
        ds = ds.take(limit)
    ds = ds.map(mappable)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds.padded_batch(1, padded_shapes=([75,None,None,1],[40])).prefetch(tf.data.AUTOTUNE)

def word_edit_ops(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]

def test_model(mdl, ds):
    ok = ok_raw = n = 0
    S_raw=N_raw=0; S_corr=N_corr=0
    for frames, labels in ds.as_numpy_iterator():
        try:
            yhat = mdl.predict(frames, verbose=0)
            dec = tf.keras.backend.ctc_decode(
                tf.cast(yhat, tf.float32), [75]*len(yhat), greedy=False)[0][0].numpy()
            orig = ' '.join(tf.strings.reduce_join(num_to_char(labels[0])).numpy().decode().strip().split())
            raw  = ' '.join(w for w in tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split() if len(w)>1)
            corr = ' '.join(correct_sentence(raw).split())
            if orig == corr: ok += 1
            if orig == raw: ok_raw += 1
            n += 1
            ref_words = orig.split()
            S_raw += word_edit_ops(ref_words, raw.split()); N_raw += len(ref_words)
            S_corr += word_edit_ops(ref_words, corr.split()); N_corr += len(ref_words)
        except Exception:
            pass
    return dict(ok=ok, ok_raw=ok_raw, n=n, S_raw=S_raw, N_raw=N_raw, S_corr=S_corr, N_corr=N_corr)

SPEAKERS = [
    ('s99_1  ', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_1\*.mpg'], None),
    ('s99_6  ', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6\*.mpg',
                 r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6_new\*.mpg'], None),
    ('s99_7  ', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7\*.mpg',
                 r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7_new\*.mpg'], None),
    ('s99_8  ', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_8\*.mpg'], None),
    ('s34_3  ', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s34_3\*.mpg'], None),
    ('GRID s1', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s1\*.mpg'], None),
    ('GRID s2', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s2\*.mpg'], None),
    ('GRID s5', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s5\*.mpg'], None),
    ('GRID s6', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s6\*.mpg'], None),
    ('GRID s7', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s7\*.mpg'], None),
    ('GRID s13',[r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s13\*.mpg'], None),
]

print('載入 v11（學長基準／完整寬度架構）模型...')
m = load_model(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_grid_multi_v11.h5',
               custom_objects={'CTCLoss': CTCLoss}, compile=False)
print('載入完成！\n')

rows=[]
tot_n=tot_ok=tot_ok_raw=0; tot_Sr=tot_Nr=0; tot_Sc=tot_Nc=0
for label, patterns, limit in SPEAKERS:
    print(f'  測試 {label.strip()}...', flush=True)
    before_miss = MISS['count']
    ds = make_ds(patterns, limit=limit)
    r = test_model(m, ds)
    miss_this = MISS['count'] - before_miss
    if miss_this: print(f'    !! 警告：{miss_this} 筆讀取失敗(快取缺失)', flush=True)
    wer_raw = r['S_raw']/r['N_raw']*100 if r['N_raw'] else 0
    wer_corr = r['S_corr']/r['N_corr']*100 if r['N_corr'] else 0
    sent_acc = r['ok']/r['n']*100 if r['n'] else 0
    sent_acc_raw = r['ok_raw']/r['n']*100 if r['n'] else 0
    rows.append((label, r['n'], sent_acc, sent_acc_raw, wer_raw, wer_corr))
    print(f'    -> 整句(校正後)={sent_acc:.1f}% 整句(校正前)={sent_acc_raw:.1f}% | WER校正前={wer_raw:.1f}%(單字正確率{100-wer_raw:.1f}%) | WER校正後={wer_corr:.1f}%(單字正確率{100-wer_corr:.1f}%)', flush=True)
    tot_n+=r['n']; tot_ok+=r['ok']; tot_ok_raw+=r['ok_raw']; tot_Sr+=r['S_raw']; tot_Nr+=r['N_raw']; tot_Sc+=r['S_corr']; tot_Nc+=r['N_corr']

print('\n' + '='*110)
print(f'{"說話者":<10}{"筆數":>6}{"整句(校正後)":>13}{"整句(校正前)":>13}{"單字(校正前)":>13}{"單字(校正後)":>13}')
print('='*110)
for label, n, sent_acc, sent_acc_raw, wer_raw, wer_corr in rows:
    print(f'{label:<10}{n:>6}{sent_acc:>12.1f}%{sent_acc_raw:>12.1f}%{100-wer_raw:>12.1f}%{100-wer_corr:>12.1f}%')

overall_sent = tot_ok/tot_n*100
overall_sent_raw = tot_ok_raw/tot_n*100
overall_wer_raw = tot_Sr/tot_Nr*100
overall_wer_corr = tot_Sc/tot_Nc*100
print('-'*110)
print(f'{"整體(語料級)":<10}{tot_n:>6}{overall_sent:>12.1f}%{overall_sent_raw:>12.1f}%{100-overall_wer_raw:>12.1f}%{100-overall_wer_corr:>12.1f}%')
print('='*110)
print(f'\n總計快取缺失筆數：{MISS["count"]}')
