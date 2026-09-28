"""
補上之前缺的一組直接對照實驗：「完整寬度模型(v11) vs 通道砍半模型(width05)」
在同一台機器、強制用CPU(不用GPU)的情況下，推論同一組假資料，各跑N次取平均，
直接量出「輕量化架構」對「推論速度」的真實影響，而不是用訓練速度去推論。

背景：2026-09-03跟使用者對話時發現，我們手上只有「width05模型 ONNX vs .h5」的
CPU推論速度對照(145ms vs 300ms)，還有「width05 vs 完整寬度 的訓練速度對照」(2.21x)，
但沒有「完整寬度 vs width05，同格式(.h5)、同硬體、CPU推論」這組最直接的數字。
這支腳本補上這個缺口。

用法：
  CUDA_VISIBLE_DEVICES=-1 /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/benchmark_cpu_speed_v11_vs_width05.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 強制不用GPU，模擬低階硬體情境
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
    'v11（完整寬度，舊基準）': 'trained_model_grid_multi_v11.h5',
    'width05_yolov8（現役，通道砍半）': 'trained_model_grid_multi_width05_yolov8.h5',
}

N_WARMUP = 3
N_RUNS = 20
input_shape = (1, 75, 46, 140, 1)

results = {}
for label, fname in MODELS.items():
    path = os.path.join(BASE_DIR, 'models', fname)
    print(f'載入 {label} ({fname}) ...')
    t_load0 = time.time()
    model = load_model(path, custom_objects={'CTCLoss': CTCLoss}, compile=False)
    t_load = time.time() - t_load0
    print(f'  載入耗時: {t_load:.2f}秒')

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
    print(f'{label:35s} 平均 {m:7.1f}ms   相對v11速度: {ratio:.2f}x')
print('=' * 70)
