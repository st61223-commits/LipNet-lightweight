"""
測試onnxruntime的SessionOptions線程設定對CPU推論速度的影響。

背景：查資料發現LipNetConnect1.py建立ONNX session時完全用預設設定
(ort.InferenceSession(path, providers=['CPUExecutionProvider']))，
沒有調整intra_op_num_threads(單一運算內部平行運算的執行緒數)。預設是
用滿所有實體核心，但文獻提到：(1)太多執行緒有時反而因為排程/競爭開銷
拖慢小模型的延遲 (2)如果同時還有YOLO偵測在跑，兩邊搶執行緒可能互相拖累。
這是純粹的「執行期設定」調整，不改模型、不改架構、零風險、可隨時調回預設。

用法：
  /mnt/c/Users/Tno/miniconda3/envs/tf215/python.exe scripts/tune_onnx_threads.py
"""
import os
import time
import multiprocessing

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)

import onnxruntime as ort
import numpy as np

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'width05_yolov8.onnx')
n_cores = multiprocessing.cpu_count()
print(f'onnxruntime版本: {ort.__version__}')
print(f'這台機器的CPU核心數(含邏輯核心): {n_cores}\n')

dummy = np.random.rand(1, 75, 46, 140, 1).astype(np.float32)
N_WARMUP, N_RUNS = 3, 20

configs = [
    ('預設(intra=0,自動用滿核心)', {'intra_op_num_threads': 0}),
    ('intra=1', {'intra_op_num_threads': 1}),
    ('intra=2', {'intra_op_num_threads': 2}),
    ('intra=4', {'intra_op_num_threads': 4}),
]

results = []
for label, opts in configs:
    so = ort.SessionOptions()
    if opts.get('intra_op_num_threads', 0) > 0:
        so.intra_op_num_threads = opts['intra_op_num_threads']
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    for _ in range(N_WARMUP):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(N_RUNS):
        t0 = time.time()
        session.run(None, {input_name: dummy})
        times.append((time.time() - t0) * 1000)
    times = np.array(times)
    print(f'{label:30s}: 平均 {times.mean():7.1f}ms, 中位數 {np.median(times):7.1f}ms, '
          f'最快 {times.min():7.1f}ms, 最慢 {times.max():7.1f}ms')
    results.append((label, times.mean()))
    del session

print('\n' + '=' * 60)
baseline = results[0][1]
best_label, best_mean = min(results, key=lambda x: x[1])
print(f'最快設定: {best_label} ({best_mean:.1f}ms)')
print(f'相對預設設定的改善: {(1 - best_mean/baseline)*100:.1f}%')
print('=' * 60)
