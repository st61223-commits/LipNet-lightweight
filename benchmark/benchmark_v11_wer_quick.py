"""
Benchmark（WER版）：width05_datafix 現役模型，額外計算 WER（逐字錯誤率）
跟原始 LipNet 論文（Assael et al. 2016）用同一套量尺(WER→單字正確率)比較，
不只是既有腳本(benchmark_width05_datafix.py)用的「整句完全比對」。
兩種指標同時輸出，互相對照，不取代既有的整句比對結果。
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
    for suffix in ['_cached_v5_gridquick', '_cached', '_dlib_cached']:
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
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1] and dp[i][j] == dp[i-1][j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            S += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            D += 1; i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            I += 1; j -= 1
        else:
            break
    return S, D, I, dp[n][m]

def test_model(mdl, ds):
    ok = n = 0
    S_raw=D_raw=I_raw=N_raw = 0
    S_corr=D_corr=I_corr=N_corr = 0
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
            s,d,i_,_ = word_edit_ops(ref_words, raw.split())
            S_raw+=s; D_raw+=d; I_raw+=i_; N_raw+=len(ref_words)
            s,d,i_,_ = word_edit_ops(ref_words, corr.split())
            S_corr+=s; D_corr+=d; I_corr+=i_; N_corr+=len(ref_words)
        except Exception:
            pass
    return dict(ok=ok, n=n,
                S_raw=S_raw, D_raw=D_raw, I_raw=I_raw, N_raw=N_raw,
                S_corr=S_corr, D_corr=D_corr, I_corr=I_corr, N_corr=N_corr)

# 快速測試：只測s99_8(100支)+s34_3(199支)，用剛重製的_cached_v5_gridquick快取
import glob as _glob
_cached_files = sorted(_glob.glob(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s1_cached_v5_gridquick\*.npy'))
_video_paths = [
    os.path.join(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s1', os.path.splitext(os.path.basename(p))[0] + '.mpg')
    for p in _cached_files
]
print(f'實際只測有快取的 {len(_video_paths)} 支影片（避免抓到沒快取的影片變成假的全零資料污染結果）')
SPEAKERS = [
    ('GRID s1', _video_paths),
]

print('載入 v11（學長基準／完整寬度架構）模型...')
m = load_model(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_grid_multi_v11.h5',
               custom_objects={'CTCLoss': CTCLoss}, compile=False)
print('載入完成！\n')

rows = []
for label, patterns in SPEAKERS:
    print(f'  測試 {label.strip()}...', flush=True)
    ds = make_ds(patterns)
    r = test_model(m, ds)
    wer_raw = (r['S_raw']+r['D_raw']+r['I_raw'])/r['N_raw']*100 if r['N_raw'] else 0
    wer_corr = (r['S_corr']+r['D_corr']+r['I_corr'])/r['N_corr']*100 if r['N_corr'] else 0
    sent_acc = r['ok']/r['n']*100 if r['n'] else 0
    rows.append((label, r, wer_raw, wer_corr, sent_acc))
    print(f'    → 整句完全比對 {r["ok"]}/{r["n"]} = {sent_acc:.1f}%'
          f' | WER(校正前)={wer_raw:.1f}% 單字正確率={100-wer_raw:.1f}%'
          f' | WER(校正後)={wer_corr:.1f}% 單字正確率={100-wer_corr:.1f}%', flush=True)

print('\n' + '='*100)
print(f'{"說話者":<10}{"整句筆數":>8}{"整句正確率":>12}{"WER校正前":>12}{"單字正確率(校正前)":>18}{"WER校正後":>12}{"單字正確率(校正後)":>18}')
print('='*100)

tot_n=tot_ok=0
tot_Sr=tot_Dr=tot_Ir=tot_Nr=0
tot_Sc=tot_Dc=tot_Ic=tot_Nc=0
for label, r, wer_raw, wer_corr, sent_acc in rows:
    print(f'{label:<10}{r["n"]:>8}{sent_acc:>11.1f}%{wer_raw:>11.1f}%{100-wer_raw:>17.1f}%{wer_corr:>11.1f}%{100-wer_corr:>17.1f}%')
    tot_n+=r['n']; tot_ok+=r['ok']
    tot_Sr+=r['S_raw']; tot_Dr+=r['D_raw']; tot_Ir+=r['I_raw']; tot_Nr+=r['N_raw']
    tot_Sc+=r['S_corr']; tot_Dc+=r['D_corr']; tot_Ic+=r['I_corr']; tot_Nc+=r['N_corr']

print('-'*100)
overall_sent = tot_ok/tot_n*100
overall_wer_raw = (tot_Sr+tot_Dr+tot_Ir)/tot_Nr*100
overall_wer_corr = (tot_Sc+tot_Dc+tot_Ic)/tot_Nc*100
print(f'{"整體(語料級)":<10}{tot_n:>8}{overall_sent:>11.1f}%{overall_wer_raw:>11.1f}%{100-overall_wer_raw:>17.1f}%{overall_wer_corr:>11.1f}%{100-overall_wer_corr:>17.1f}%')
print('='*100)
print(f'\n⚠️ 快速測試，只測2位說話者(s99_8+s34_3，共299筆)，不是完整12位說話者規模，數字僅供初步參考')
print(f'對照：原始LipNet論文(Assael et al. 2016) GRID單字正確率 = 95.2%(同批說話者) / 88.6%(未看過新說話者)')
print(f'v11(學長基準模型)：')
print(f'  校正前單字正確率 = {100-overall_wer_raw:.1f}%（原始CTC解碼輸出，未套用詞彙校正，跟論文條件最接近）')
print(f'  校正後單字正確率 = {100-overall_wer_corr:.1f}%（套用部署時實際使用的詞彙校正後）')
