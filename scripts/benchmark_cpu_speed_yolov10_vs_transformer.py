"""
補上Transformer實驗報告裡明確列為「留待後續評估」的缺口：CPU推論速度對照。
跟之前 benchmark_cpu_speed_v11_vs_width05.py 同樣方法，同一台機器強制CPU、
同一組假資料、各跑N次取平均，比較「現役模型(width05_yolov10)」vs「Transformer後端版」。

用法：
  CUDA_VISIBLE_DEVICES=-1 /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/benchmark_cpu_speed_yolov10_vs_transformer.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import time
import warnings
import pathlib

pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

print(f'可見GPU數量: {len(tf.config.list_physical_devices("GPU"))}（應該是0，代表成功強制CPU）\n')


def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)


MODELS = {
    '現役 width05_yolov10（BiLSTM後端）': 'trained_model_grid_multi_width05_yolov10.h5',
    'Transformer 後端版': 'trained_model_grid_multi_transformer.h5',
}

N_WARMUP = 3
N_RUNS = 20
input_shape = (1, 75, 46, 140, 1)

results = {}
param_counts = {}
for label, fname in MODELS.items():
    path = os.path.join(BASE_DIR, 'models', fname)
    print(f'載入 {label} ({fname}) ...')
    t_load0 = time.time()
    model = load_model(path, custom_objects={'CTCLoss': CTCLoss}, compile=False)
    t_load = time.time() - t_load0
    print(f'  載入耗時: {t_load:.2f}秒')
    param_counts[label] = model.count_params()
    print(f'  參數量: {param_counts[label]:,}')

    dummy = np.random.rand(*input_shape).astype(np.float32)

    for _ in range(N_WARMUP):
        model.predict(dummy, verbose=0)

    times = []
    for _ in range(N_RUNS):
        t0 = time.time()
        model.predict(dummy, verbose=0)
        times.append((time.time() - t0) * 1000)

    times = np.array(times)
    results[label] = times
    print(f'  推論 {N_RUNS} 次: 平均 {times.mean():.1f}ms, 中位數 {np.median(times):.1f}ms, '
          f'最快 {times.min():.1f}ms, 最慢 {times.max():.1f}ms\n')
    del model
    tf.keras.backend.clear_session()

print('=' * 70)
print('對照結果（CPU推論，同一台機器同一批假資料，強制CUDA_VISIBLE_DEVICES=-1）')
print('=' * 70)
labels = list(results.keys())
baseline_mean = results[labels[0]].mean()
for label in labels:
    m = results[label].mean()
    ratio = baseline_mean / m
    print(f'{label:35s} 平均 {m:7.1f}ms   參數量 {param_counts[label]:>10,}   相對現役速度: {ratio:.2f}x')
print('=' * 70)
