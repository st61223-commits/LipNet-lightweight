"""
測試 TFLite INT8 量化版本的 CPU 推論耗時，跟原始 F_Width0.5 (.h5) 的
CPU基準（300.3ms/次，見 project_lipnet_direction 記憶）做對照。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 強制純CPU，跟之前的基準測試條件一致
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import time
import numpy as np
import tensorflow as tf

import sys
TFLITE_PATH = f'models/width05_{sys.argv[1]}.tflite' if len(sys.argv) > 1 else 'models/width05_dynamic_int8.tflite'
size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
print(f'TFLite模型大小: {size_mb:.2f} MB')

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('輸入資訊:', input_details[0]['shape'], input_details[0]['dtype'])
print('輸出資訊:', output_details[0]['shape'], output_details[0]['dtype'])

rng = np.random.default_rng(0)
dummy = rng.standard_normal(input_details[0]['shape']).astype(input_details[0]['dtype'])

for _ in range(3):  # 暖機
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()

N = 20
times = []
for _ in range(N):
    t0 = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    tf.keras.backend.ctc_decode(tf.constant(out), input_length=[75], greedy=True)[0][0].numpy()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

times = np.array(times)
print(f'\n===== TFLite INT8量化版 CPU推論耗時（含CTC decode，共測{N}次）=====')
print(f'平均: {times.mean():.1f} ms')
print(f'中位數: {np.median(times):.1f} ms')
print(f'最快: {times.min():.1f} ms')
print(f'最慢: {times.max():.1f} ms')
print(f'\n對照：F_Width0.5原始(.h5)CPU平均 300.3 ms、v11原始(.h5)CPU平均 662.6 ms')
print(f'量化後 vs 量化前(.h5) 加速比: {300.3/times.mean():.2f}x')
print(f'量化後 vs v11(.h5) 加速比: {662.6/times.mean():.2f}x')
