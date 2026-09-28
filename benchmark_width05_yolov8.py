"""
Benchmark：grid_multi F_Width0.5「YOLOv8裁切版」（width05_yolov8）

背景：2026-09-02發現即時通話系統用YOLOv8裁切嘴唇，但全部訓練資料是用YOLOv5
裁切產生的，兩者不一致導致實機辨識大幅失準（Training-Inference Mismatch，
詳見 project_lipnet.md）。已用YOLOv8重新產生全部16位說話者的訓練快取
（regenerate_cache_yolov8.py，_cached_yolov8後綴）並重新訓練
（train_grid_multi_width05_yolov8.py，暖啟動自width05_datafix）。

這支腳本讀取 _cached_yolov8 快取（不是_cached，因為要跟這個新模型的訓練
分布一致），驗證重新訓練後正確率是否維持/進步，跟現役width05_datafix
（89.4%，但那是用YOLOv5快取測的）比較。

⚠️ 這個benchmark測的是「YOLOv8快取上的正確率」，不是「即時管線的真實表現」。
真正驗證有沒有解決Training-Inference Mismatch問題，還要另外跑
test_sliding_window_pipeline.py（模擬真實即時管線）確認。

s9不在這支benchmark裡（regenerate_cache_yolov8.py沒有幫s9產生YOLOv8快取，
因為s9是held-out測試對象、不參與訓練資料重製）。
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
    # 只認 _cached_yolov8，不fallback到舊的_cached(YOLOv5)，避免混用兩種分布造成誤判
    cache_path = os.path.join('data', f'{project_name}_cached_yolov8', f'{file_name}.npy')
    if os.path.exists(cache_path):
        frames = np.load(cache_path)
        mean, std = np.mean(frames), np.std(frames) + 1e-6
        return ((frames - mean) / std).astype(np.float32)
    raise FileNotFoundError(f'找不到YOLOv8快取：{project_name}/{file_name}')

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

# 跟regenerate_cache_yolov8.py的SPEAKER_DIRS同一套(s9不在其中，故不測)
SPEAKERS = [
    ('s99_1  ', 1000, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_1\*.mpg']),
    ('s99_6  ',  200, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6\*.mpg',
                       r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6_new\*.mpg']),
    ('s99_7  ',  400, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7\*.mpg',
                       r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7_new\*.mpg']),
    ('s99_8  ',  100, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_8\*.mpg']),
    ('s34_3  ',  199, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s34_3\*.mpg']),
    ('GRID s1', 1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s1\*.mpg']),
    ('GRID s2', 1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s2\*.mpg']),
    ('GRID s3', 1000, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s3\*.mpg']),
    ('GRID s4', 1000, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s4\*.mpg']),
    ('GRID s5', 1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s5\*.mpg']),
    ('GRID s6',  987, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s6\*.mpg']),
    ('GRID s7', 1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s7\*.mpg']),
    ('GRID s8', 1000, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s8\*.mpg']),
    ('GRID s13',1001, [r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s13\*.mpg']),
]

# width05_datafix 參考數字（現役模型，YOLOv5快取上測的89.4%，2026-09-02）
# 注意：s3/s4/s8這三位當初datafix benchmark沒有單獨測過，留空
DATAFIX_REF = {
    's99_1  ': 99.5, 's99_6  ': 100.0, 's99_7  ': 99.8,
    's99_8  ': 100.0, 's34_3  ': 81.9,
    'GRID s1': 95.3, 'GRID s2': 60.9,  'GRID s5': 89.6,
    'GRID s6': 98.5, 'GRID s7': 97.9,  'GRID s13': 82.4,
}

MODEL_PATH = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_grid_multi_width05_yolov8.h5'

print('載入 width05_yolov8 模型...')
m = load_model(MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)
print('載入完成！\n')

results = []
for label, n_expect, patterns in SPEAKERS:
    print(f'  測試 {label.strip()}...', flush=True)
    ds = make_ds(patterns)
    ok, n = test_model(m, ds)
    results.append((label, n, ok))
    pct = ok/n*100 if n > 0 else 0
    ref = DATAFIX_REF.get(label)
    ref_s = f'{ref:.1f}%' if ref is not None else 'N/A(新測)'
    print(f'    → {ok}/{n} = {pct:.1f}%  (width05_datafix於YOLOv5快取: {ref_s})', flush=True)

print('\n')
print('=' * 90)
print(f'{"說話者":<10} {"筆數":>5}   {"yolov8版":>10}   {"datafix(v5)":>12}   {"差異":>8}')
print('=' * 90)

total_n = wok = ref_wok_total = 0
n_comparable = ref_wok_comparable = 0
for label, n, ok in results:
    pct = ok/n*100 if n > 0 else 0
    ref = DATAFIX_REF.get(label)
    if ref is not None:
        diff = pct - ref
        diff_s = f'+{diff:.1f}%' if diff >= 0 else f'{diff:.1f}%'
        ref_s = f'{ref:>9.1f}%'
        n_comparable += n
        ref_wok_comparable += n * ref
    else:
        diff_s = '(新測)'
        ref_s = '        N/A'
    print(f'{label:<10} {n:>5}   {ok:>4}/{n:<5} {pct:>5.1f}%   {ref_s}   {diff_s:>8}')
    total_n += n
    wok += n * pct

print('-' * 90)
avg_all = wok / total_n if total_n else 0
avg_comparable = (sum(n*(ok/n*100 if n>0 else 0) for label,n,ok in results if DATAFIX_REF.get(label) is not None) / n_comparable) if n_comparable else 0
ref_avg = ref_wok_comparable / n_comparable if n_comparable else 0
diff_avg = avg_comparable - ref_avg
diff_avg_s = f'+{diff_avg:.1f}%' if diff_avg >= 0 else f'{diff_avg:.1f}%'

print(f'{"加權平均(全部)":<14} {total_n:>5}   {"":>11} {avg_all:>5.1f}%')
print(f'{"加權平均(可比對)":<14} {n_comparable:>5}   {"":>11} {avg_comparable:>5.1f}%   {ref_avg:>9.1f}%   {diff_avg_s:>8}   ← yolov8版 vs datafix(YOLOv5快取)')
print('=' * 90)
print('\n提醒：這是在YOLOv8快取上測的分數，跟即時管線是否真的修好，還要另外跑')
print('test_sliding_window_pipeline.py 模擬真實攝影機管線確認。')
