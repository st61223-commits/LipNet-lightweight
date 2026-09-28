"""
方案B：分析 s1_v3 在 GRID s1 上「答對哪些句子」，找出瓶頸。

輸出：
  1. 每筆預測結果（CSV）
  2. 整體準確率
  3. 每個詞位置的準確率
  4. 每個單字的準確率
  5. 答對句子列表
"""
import os, sys, pathlib, warnings, io, csv
from collections import defaultdict
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

# ── 載入模型 ──
print('載入 s1_v3 模型...')
model = load_model(
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\trained_model_s1_v3.h5',
    custom_objects={'CTCLoss': CTCLoss})
print('載入完成！\n')

# ── 跑 GRID s1 ──
s1_pattern = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s1\*.mpg'
file_paths = tf.data.Dataset.list_files(s1_pattern)

# 先收集所有路徑（用於顯示檔名）
all_paths = sorted([p.numpy().decode() for p in file_paths])

ds = tf.data.Dataset.from_tensor_slices(all_paths)
ds = ds.map(lambda p: (p, *tf.py_function(
    lambda x: load_data(x), [p], (tf.float32, tf.int64))))

# 重新建立 dataset（含路徑）
def load_with_path(path):
    frames, labels = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    frames.set_shape([75, None, None, 1])
    labels.set_shape([40])
    return path, frames, labels

ds2 = tf.data.Dataset.from_tensor_slices(all_paths)
ds2 = ds2.map(load_with_path)
ds2 = ds2.padded_batch(1, padded_shapes=([], [75, None, None, 1], [40])).prefetch(tf.data.AUTOTUNE)

# ── 分析用資料結構 ──
results = []          # (filename, orig, raw, corr, is_correct)
pos_ok   = defaultdict(int)   # position → correct count
pos_tot  = defaultdict(int)   # position → total count
word_ok  = defaultdict(int)   # word → correct count
word_tot = defaultdict(int)   # word → total count

n = 0
print(f'開始預測 {len(all_paths)} 筆...\n')

for batch_paths, frames, labels in ds2.as_numpy_iterator():
    path_str = batch_paths[0].decode()
    filename = os.path.splitext(os.path.basename(path_str))[0]

    yhat = model.predict(frames, verbose=0)
    dec  = tf.keras.backend.ctc_decode(
        tf.cast(yhat, tf.float32), [75] * len(yhat), greedy=False)[0][0].numpy()

    orig = ' '.join(tf.strings.reduce_join(num_to_char(labels[0])).numpy().decode().strip().split())
    raw  = ' '.join(w for w in tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split() if len(w) > 1)
    corr = ' '.join(correct_sentence(raw).split())
    is_correct = (orig == corr)

    results.append((filename, orig, raw, corr, is_correct))

    # 詞位置分析
    orig_words = orig.split()
    corr_words = corr.split()
    for i, ow in enumerate(orig_words):
        pos_tot[i] += 1
        word_tot[ow] += 1
        if i < len(corr_words) and corr_words[i] == ow:
            pos_ok[i] += 1
            word_ok[ow] += 1

    n += 1
    if n % 100 == 0:
        correct_so_far = sum(1 for r in results if r[4])
        print(f'  已處理 {n}/{len(all_paths)}，目前準確率 {correct_so_far/n*100:.1f}%')

# ── 輸出結果 ──
correct_total = sum(1 for r in results if r[4])
print(f'\n{"="*70}')
print(f'總筆數：{n}   答對：{correct_total}   準確率：{correct_total/n*100:.1f}%')
print(f'{"="*70}')

# 1. 詞位置準確率
GRID_POS = ['指令(place/bin/lay/set)', '顏色', '介系詞(at/by/in/with)', '字母', '數字', '時間/again']
print('\n【每個詞位置的準確率】')
print(f'  {"位置":<5} {"名稱":<25} {"答對":>6} {"總計":>6} {"準確率":>8}')
print(f'  {"-"*55}')
for i in range(6):
    name = GRID_POS[i] if i < len(GRID_POS) else f'位置{i}'
    ok  = pos_ok.get(i, 0)
    tot = pos_tot.get(i, 0)
    pct = ok/tot*100 if tot > 0 else 0
    print(f'  {i:<5} {name:<25} {ok:>6} {tot:>6} {pct:>7.1f}%')

# 2. 每個單字準確率（按準確率排序）
print('\n【每個單字的準確率（出現 ≥ 10 次）】')
print(f'  {"單字":<15} {"答對":>6} {"總計":>6} {"準確率":>8}')
print(f'  {"-"*40}')
word_stats = [(w, word_ok[w], word_tot[w], word_ok[w]/word_tot[w]*100)
              for w in word_tot if word_tot[w] >= 10]
word_stats.sort(key=lambda x: -x[3])
for w, ok, tot, pct in word_stats:
    print(f'  {w:<15} {ok:>6} {tot:>6} {pct:>7.1f}%')

# 3. 答對的句子列表
correct_results = [r for r in results if r[4]]
print(f'\n【答對的 {len(correct_results)} 個句子】')
for filename, orig, raw, corr, _ in correct_results[:50]:
    print(f'  {filename:<12}  {orig}')
if len(correct_results) > 50:
    print(f'  ...（共 {len(correct_results)} 個，只顯示前 50）')

# 4. 常見錯誤模式（答錯的前 20 筆）
wrong_results = [r for r in results if not r[4]]
print(f'\n【錯誤範例（前 20 筆）】')
print(f'  {"檔名":<12}  {"答案":<30}  {"校正後預測":<30}')
print(f'  {"-"*75}')
for filename, orig, raw, corr, _ in wrong_results[:20]:
    print(f'  {filename:<12}  {orig:<30}  {corr:<30}')

# 5. 儲存完整 CSV
csv_path = r'C:\Users\Tno\claude-code\s1_v3_analysis.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'ground_truth', 'raw_pred', 'corrected_pred', 'correct'])
    for filename, orig, raw, corr, ok in results:
        writer.writerow([filename, orig, raw, corr, ok])
print(f'\n完整結果已儲存：{csv_path}')
