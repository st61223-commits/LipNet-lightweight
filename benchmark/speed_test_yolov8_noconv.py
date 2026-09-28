import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
import numpy as np
import onnxruntime as ort

N_WARMUP, N_RUNS = 3, 20
dummy = np.random.rand(1, 75, 46, 140, 1).astype(np.float32)
sess = ort.InferenceSession('models/width05_yolov8_int8_noconv.onnx', providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
for _ in range(N_WARMUP): sess.run(None, {in_name: dummy})
times = []
for _ in range(N_RUNS):
    t0 = time.time(); sess.run(None, {in_name: dummy}); times.append((time.time()-t0)*1000)
times = np.array(times)
print(f'ONNX INT8 (noconv): 平均 {times.mean():.1f}ms, 中位數 {np.median(times):.1f}ms')
