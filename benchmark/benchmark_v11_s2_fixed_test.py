"""單獨測 v11 在修正後 s2 快取上的表現，比對修正前後差異。"""
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

CACHE_SUFFIX = '_cached_v5_fixed'

def _load_frames(path):
    project_name = os.path.basename(os.path.dirname(path))
    file_name = os.path.splitext(os.path.basename(path))[0]
    cache_path = os.path.join('data', f'{project_name}{CACHE_SUFFIX}', f'{file_name}.npy')
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

def make_ds(patterns):
    ds = tf.data.Dataset.list_files(patterns, shuffle=False).map(mappable)
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
    ok = n = 0
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
            n += 1
            ref_words = orig.split()
            S_raw += word_edit_ops(ref_words, raw.split()); N_raw += len(ref_words)
            S_corr += word_edit_ops(ref_words, corr.split()); N_corr += len(ref_words)
        except Exception:
            pass
    return dict(ok=ok, n=n, S_raw=S_raw, N_raw=N_raw, S_corr=S_corr, N_corr=N_corr)

print('載入 v11 模型...')
m = load_model(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_grid_multi_v11.h5',
               custom_objects={'CTCLoss': CTCLoss}, compile=False)
print('載入完成！\n')

print('測試 GRID s2（修正後快取）...')
ds = make_ds([r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s2\*.mpg'])
r = test_model(m, ds)
wer_raw = r['S_raw']/r['N_raw']*100 if r['N_raw'] else 0
wer_corr = r['S_corr']/r['N_corr']*100 if r['N_corr'] else 0
sent_acc = r['ok']/r['n']*100 if r['n'] else 0
print(f'\n快取缺失：{MISS["count"]}筆')
print(f'整句 {r["ok"]}/{r["n"]}={sent_acc:.1f}% | WER校正前={wer_raw:.1f}%(單字正確率{100-wer_raw:.1f}%) | WER校正後={wer_corr:.1f}%(單字正確率{100-wer_corr:.1f}%)')
print(f'\n對照：舊版(有bug的conf=0.5快取)測出來的結果是 raw=11.6% corr=32.5%')
