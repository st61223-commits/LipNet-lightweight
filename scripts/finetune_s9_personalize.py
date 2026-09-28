"""
「新使用者個人化校準」小實驗：用少量 s9（從沒訓練過的說話者）資料快速微調，
測試能不能解決「對全新使用者辨識率0%」的問題（見 實驗記錄.md 實驗五結論、
project_lipnet_direction.md 的 LoRA個人化文獻筆記）。

背景：資料修復(width05_datafix)雖然把已訓練說話者的加權平均從76.6%拉到89.4%，
但 s9(held-out) 依然是 0.1%——證實這不是資料品質問題，是模型從沒看過這個人。
查文獻發現「用少量新使用者資料做輕量微調」(LoRA個人化) 比追求zero-shot泛化
更實際，也更貼近真實部署場景（新使用者用前先錄一小段校準影片）。

做法（簡化版個人化，非真正LoRA——完整LoRA需要額外工程量，這裡先驗證「概念
是否有效」）：
1. s9 共1000筆，切成「校準集」(CALIB_N筆，模擬新使用者第一次用時錄的資料)
   跟「測試集」(其餘筆數，模擬之後真實使用時的資料)
2. 凍結 Conv3D+BatchNorm 特徵萃取層（假設嘴型視覺特徵是通用的），只微調
   LSTM+Dense decoder 部分，降低過擬合風險、微調也更快
3. 只在校準集上訓練幾輪，然後在測試集上跑 benchmark，看正確率有沒有從
   基準的 0.1% 明顯回升

**磁碟空間考量（2026-09-02 C槽只剩5.8GB可用）**：不存per-epoch checkpoint，
只在訓練結束後存一次最終模型（~48MB），把磁碟佔用降到最低。

執行方式（WSL）：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/finetune_s9_personalize.py
"""
import os
import sys
import glob
import pathlib
import warnings
import numpy as np
import tensorflow as tf

pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)
sys.path.append(r'C:\Users\Tno\claude-code')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from vocab_correction import correct_sentence

CALIB_N = 50          # 模擬新使用者第一次錄的校準資料筆數
FINETUNE_EPOCHS = 25  # 校準資料很少，多練幾輪讓模型真的學進去
BATCH_SIZE = 2
BASE_MODEL_PATH = os.path.join('models', 'trained_model_grid_multi_width05_datafix.h5')
SAVE_PATH = os.path.join('models', 'trained_model_width05_datafix_s9personalized.h5')

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
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
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split()
        word = parts[2]
        if word not in ('sil', 'sp') and len(word) > 1:
            tokens.extend([' ', word])
    tokens_flat = tokens[1:]
    if not tokens_flat:
        return tf.zeros([1], dtype=tf.int64)
    return char_to_num(tf.reshape(
        tf.strings.unicode_split(tokens_flat, input_encoding='UTF-8'), (-1)))


def load_data(path):
    try:
        p = bytes.decode(path.numpy())
        file_name = os.path.splitext(os.path.basename(p))[0]
        parts = [x for x in p.replace('\\\\', '/').replace('\\', '/').split('/') if x]
        project_name = parts[-2]
        frames = _load_frames(os.path.join('data', project_name, f'{file_name}.mpg'))
        alignments = load_alignments(os.path.join('data', 'alignments', project_name, f'{file_name}.align'))
    except Exception as e:
        print(f'[SKIPPED] {path} -> {e}')
        frames = np.zeros((75, 46, 140, 1), dtype=np.float32)
        alignments = np.zeros((1,), dtype=np.int32)
    return frames, alignments


def mappable(path):
    f, l = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    f.set_shape([75, None, None, 1])
    l.set_shape([40])
    return f, l


def make_train_ds(file_list):
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    ds = ds.shuffle(len(file_list), reshuffle_each_iteration=True)
    ds = ds.map(mappable)
    ds = ds.padded_batch(BATCH_SIZE, padded_shapes=([75, None, None, 1], [40]))
    return ds.prefetch(tf.data.AUTOTUNE)


def make_eval_ds(file_list):
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    ds = ds.map(mappable)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds.padded_batch(1, padded_shapes=([75, None, None, 1], [40])).prefetch(tf.data.AUTOTUNE)


def test_model(mdl, ds):
    ok = n = 0
    for frames, labels in ds.as_numpy_iterator():
        try:
            yhat = mdl.predict(frames, verbose=0)
            dec = tf.keras.backend.ctc_decode(
                tf.cast(yhat, tf.float32), [75] * len(yhat), greedy=False)[0][0].numpy()
            orig = ' '.join(tf.strings.reduce_join(num_to_char(labels[0])).numpy().decode().strip().split())
            raw = ' '.join(w for w in tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split() if len(w) > 1)
            corr = ' '.join(correct_sentence(raw).split())
            if orig == corr:
                ok += 1
            n += 1
        except Exception:
            pass
    return ok, n


# ── 切分 s9：前 CALIB_N 筆當校準資料，其餘當測試資料（固定用檔名排序，實驗可重現）
all_files = sorted(glob.glob(os.path.join('data', 's9', '*.mpg')))
print(f's9 總筆數: {len(all_files)}')
calib_files = all_files[:CALIB_N]
test_files = all_files[CALIB_N:]
print(f'校準集: {len(calib_files)} 筆，測試集: {len(test_files)} 筆')

print('\n載入 width05_datafix 模型當起點...')
model = load_model(BASE_MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)

# ── 凍結 Conv3D+BatchNorm 特徵萃取層，只微調 LSTM+Dense（輕量個人化，減少過擬合/破壞其他說話者的風險）
FREEZE_LAYER_TYPES = ('Conv3D', 'BatchNormalization', 'Activation', 'MaxPooling3D', 'MaxPool3D')
frozen, trainable = 0, 0
for layer in model.layers:
    if type(layer).__name__ in FREEZE_LAYER_TYPES or 'time_distributed' in layer.name.lower():
        layer.trainable = False
        frozen += 1
    else:
        layer.trainable = True
        trainable += 1
print(f'凍結層數: {frozen}，可訓練層數: {trainable}')

model.compile(optimizer=Adam(learning_rate=1e-4, clipnorm=1.0), loss=CTCLoss)

print(f'\n=== 基準：微調前，先測一次校準前的測試集正確率（應該接近0%，跟benchmark的0.1%一致）===')
baseline_ds = make_eval_ds(test_files[:200])  # 先抽200筆快速驗證基準，不用跑全部749筆
ok0, n0 = test_model(model, baseline_ds)
print(f'微調前（測試集前200筆抽樣）: {ok0}/{n0} = {ok0/n0*100:.1f}%')

print(f'\n=== 開始個人化微調：{CALIB_N}筆校準資料，{FINETUNE_EPOCHS}輪 ===')
train_ds = make_train_ds(calib_files)
model.fit(train_ds, epochs=FINETUNE_EPOCHS, verbose=2)

model.save(SAVE_PATH)
print(f'\n個人化模型已存: {SAVE_PATH}')

print(f'\n=== 微調後：測試集全部 {len(test_files)} 筆 ===')
test_ds = make_eval_ds(test_files)
ok1, n1 = test_model(model, test_ds)
pct1 = ok1 / n1 * 100 if n1 > 0 else 0
print(f'微調後（測試集全部）: {ok1}/{n1} = {pct1:.1f}%')

print('\n' + '=' * 60)
print(f'結論：s9個人化微調 {CALIB_N}筆校準資料 → 測試集正確率 {pct1:.1f}%')
print(f'（對照：完全沒微調的基準是 0.1%，來自全量benchmark）')
print('=' * 60)
