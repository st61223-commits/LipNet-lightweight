import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
import numpy as np
import tensorflow as tf

N_WARMUP, N_RUNS = 3, 20
dummy = np.random.rand(1, 75, 46, 140, 1).astype(np.float32)

interp = tf.lite.Interpreter(model_path='models/width05_yolov8_float32_baseline.tflite')
interp.allocate_tensors()
in_idx = interp.get_input_details()[0]['index']
for _ in range(N_WARMUP):
    interp.set_tensor(in_idx, dummy); interp.invoke()

times = []
for _ in range(N_RUNS):
    interp2 = tf.lite.Interpreter(model_path='models/width05_yolov8_float32_baseline.tflite')
    interp2.allocate_tensors()
    idx_in = interp2.get_input_details()[0]['index']
    t0 = time.time()
    interp2.set_tensor(idx_in, dummy)
    interp2.invoke()
    times.append((time.time()-t0)*1000)
times = np.array(times)
print(f'TFLite float32 (純轉換): 平均 {times.mean():.1f}ms, 中位數 {np.median(times):.1f}ms')
