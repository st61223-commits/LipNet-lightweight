"""
測試 F_Width0.5 轉成 ONNX 後，用 onnxruntime 在純 CPU 環境推論的耗時，
跟原本 .h5（300.3ms）、TFLite（2345~2350ms）比較。

方法論盡量比照 benchmark_cpu_latency_width05.py / benchmark_tflite_latency.py：
固定 dummy 輸入 (1,75,46,140,1)、3 次暖機、正式測 20 次、含 CTC decode。

在 tf215 這個獨立環境跑（裝了 onnx/tf2onnx/onnxruntime，不會動到 py39）。
"""
import os
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import time
import numpy as np
import onnxruntime as ort
import tensorflow as tf  # 只用來做跟其他benchmark一致的 ctc_decode，不用來跑模型本身

ONNX_PATH = 'models/width05.onnx'
size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)

# 明確指定只用 CPUExecutionProvider，確保是純 CPU 推論
sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print('Providers:', sess.get_providers())
print('輸入名稱:', input_name, '輸出名稱:', output_name)

rng = np.random.default_rng(0)
dummy = rng.standard_normal((1, 75, 46, 140, 1)).astype('float32')
N = 20

for _ in range(3):  # 暖機
    _ = sess.run([output_name], {input_name: dummy})

times = []
for _ in range(N):
    t0 = time.perf_counter()
    out = sess.run([output_name], {input_name: dummy})[0]
    tf.keras.backend.ctc_decode(out, input_length=[75], greedy=True)[0][0].numpy()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)
times = np.array(times)

print(f'\n===== ONNX Runtime CPU 推論耗時（含 CTC decode，測 {N} 次）=====')
print(f'檔案大小: {size_mb:.2f} MB')
print(f'平均: {times.mean():.1f} ms, 中位數: {np.median(times):.1f} ms, 最慢: {times.max():.1f} ms')
print(f'每秒約可跑 {1000/times.mean():.2f} 次辨識')
print('\n對照：原始 .h5 (py39, TF2.10) = 300.3 ms | TFLite INT8 = 2350.3 ms')
