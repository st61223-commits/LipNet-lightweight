"""
Benchmark（WER版）：F_Width0.5 的 TFLite INT8 版本，補算 WER。
沿用既有方法論：每位說話者僅抽樣前50筆(TFLite單次推論約2.35秒，全量需5小時以上)。
"""
import os, sys, pathlib, warnings, io
pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
sys.path.append(r'C:\Users\Tno\claude-code')

import numpy as np, tensorflow as tf
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

SAMPLE_PER_SPEAKER = 50

def make_ds(patterns):
    ds = tf.data.Dataset.list_files(patterns, shuffle=False).take(SAMPLE_PER_SPEAKER).map(mappable)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds.padded_batch(1, padded_shapes=([75,46,140,1],[40])).prefetch(tf.data.AUTOTUNE)

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

TFLITE_PATH = 'models/width05_float32_baseline.tflite'

def predict_one(frames):
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(in_idx, frames.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(out_idx)

def test_model(ds):
    ok = n = 0
    S_raw=N_raw=0; S_corr=N_corr=0
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
            ref_words = orig.split()
            S_raw += word_edit_ops(ref_words, raw.split()); N_raw += len(ref_words)
            S_corr += word_edit_ops(ref_words, corr.split()); N_corr += len(ref_words)
        except Exception:
            pass
    return dict(ok=ok, n=n, S_raw=S_raw, N_raw=N_raw, S_corr=S_corr, N_corr=N_corr)

SPEAKERS = [
    ('s99_1  ', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_1\*.mpg']),
    ('s99_6  ', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6\*.mpg',
                 r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6_new\*.mpg']),
    ('s99_7  ', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7\*.mpg',
                 r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7_new\*.mpg']),
    ('s99_8  ', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_8\*.mpg']),
    ('s99_3+5', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_3\*.mpg',
                 r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_5\*.mpg']),
    ('s34_3  ', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s34_3\*.mpg']),
    ('GRID s1', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s1\*.mpg']),
    ('GRID s2', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s2\*.mpg']),
    ('GRID s5', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s5\*.mpg']),
    ('GRID s6', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s6\*.mpg']),
    ('GRID s7', [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s7\*.mpg']),
    ('GRID s13',[r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s13\*.mpg']),
]

print(f'TFLite模型(純格式轉換,不量化): {TFLITE_PATH}  (每位說話者抽樣前{SAMPLE_PER_SPEAKER}筆)\n')
tot_n=tot_ok=0; tot_Sr=tot_Nr=0; tot_Sc=tot_Nc=0
for label, patterns in SPEAKERS:
    print(f'  測試 {label.strip()}...', flush=True)
    ds = make_ds(patterns)
    r = test_model(ds)
    wer_raw = r['S_raw']/r['N_raw']*100 if r['N_raw'] else 0
    wer_corr = r['S_corr']/r['N_corr']*100 if r['N_corr'] else 0
    sent_acc = r['ok']/r['n']*100 if r['n'] else 0
    print(f'    -> 整句 {r["ok"]}/{r["n"]}={sent_acc:.1f}% | WER校正前={wer_raw:.1f}%(單字正確率{100-wer_raw:.1f}%) | WER校正後={wer_corr:.1f}%(單字正確率{100-wer_corr:.1f}%)', flush=True)
    tot_n+=r['n']; tot_ok+=r['ok']; tot_Sr+=r['S_raw']; tot_Nr+=r['N_raw']; tot_Sc+=r['S_corr']; tot_Nc+=r['N_corr']

overall_sent = tot_ok/tot_n*100
overall_wer_raw = tot_Sr/tot_Nr*100
overall_wer_corr = tot_Sc/tot_Nc*100
print('\n' + '='*80)
print(f'TFLite float32純格式轉換 (抽樣) 整體：n={tot_n} 整句正確率={overall_sent:.1f}%')
print(f'  單字正確率(校正前)={100-overall_wer_raw:.1f}%  單字正確率(校正後)={100-overall_wer_corr:.1f}%')
print('='*80)
