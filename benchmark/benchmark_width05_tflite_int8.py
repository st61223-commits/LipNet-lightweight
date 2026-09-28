"""
Benchmark：F_Width0.5 的 TFLite INT8 量化版本，驗證量化後正確率有沒有受影響。
量化實驗已確認CPU推論速度沒有變快(甚至更慢，架構跟TFLite不相容導致)，
但檔案縮小到約1/10仍有儲存/傳輸價值，這裡補測正確率完成量化實驗最後一塊拼圖。
用TFLite Interpreter取代原本的Keras model.predict()。
"""
import os, sys, pathlib, warnings, io
pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # TFLite CPU推論，跟latency測試條件一致
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

SAMPLE_PER_SPEAKER = 50  # TFLite每次推論約2.35秒，全量~7900筆要跑5小時以上，
                          # 改抽樣驗證正確率有沒有掉，不追求跟其他benchmark一樣的全量筆數

def make_ds(patterns):
    ds = tf.data.Dataset.list_files(patterns, shuffle=False).take(SAMPLE_PER_SPEAKER).map(mappable)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds.padded_batch(1, padded_shapes=([75,46,140,1],[40])).prefetch(tf.data.AUTOTUNE)

TFLITE_PATH = 'models/width05_dynamic_int8.tflite'

# 重要踩坑：這個模型混用SELECT_TF_OPS(Flex delegate)，同一個interpreter物件
# 重複呼叫invoke()會在第2次開始輸出壞掉的固定垃圾模式(已用sanity_check_tflite_repeat.py
# 驗證確認)。修法：每筆樣本都重新建立一個新的interpreter，避免狀態污染。
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

# F_Width0.5 原始(.h5)版本參考數字（76.6%，量化前）
WIDTH05_REF = {
    's99_1  ': 93.7, 's99_6  ': 100.0, 's99_7  ': 99.5,
    's99_8  ': 99.0, 's99_3+5': 0.0,  's34_3  ': 74.4,
    'GRID s1': 84.9, 'GRID s2': 13.1,  'GRID s5': 75.0,
    'GRID s6': 97.2, 'GRID s7': 93.5,  'GRID s13': 66.3,
}

print(f'TFLite模型: {TFLITE_PATH}')
print(f'注意：每位說話者僅抽樣前{SAMPLE_PER_SPEAKER}筆（TFLite單次推論約2.35秒，全量7900+筆需5小時以上），非全量benchmark\n')

results = []
for label, n_expect, patterns in SPEAKERS:
    print(f'  測試 {label.strip()}...', flush=True)
    ds = make_ds(patterns)
    ok, n = test_model(ds)
    results.append((label, n, ok))
    pct = ok/n*100 if n > 0 else 0
    ref = WIDTH05_REF.get(label, 0)
    diff = pct - ref
    diff_s = f'+{diff:.1f}%' if diff >= 0 else f'{diff:.1f}%'
    print(f'    -> {ok}/{n} = {pct:.1f}%  (量化前: {ref:.1f}%  差異: {diff_s})', flush=True)

print('\n')
print('=' * 70)
print(f'{"說話者":<10} {"筆數":>5}   {"INT8量化":>10}   {"量化前":>10}   {"差異":>8}')
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
