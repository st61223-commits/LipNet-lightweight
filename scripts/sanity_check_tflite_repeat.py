"""
測試TFLite interpreter重複呼叫invoke()多次，會不會累積出問題
（懷疑benchmark腳本裡對同一個interpreter物件連續跑幾百次，
內部狀態沒有正確重置，導致除了第一筆之外全部壞掉）。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import glob
import numpy as np
import tensorflow as tf

def _load_frames(path):
    project_name = os.path.basename(os.path.dirname(path))
    file_name = os.path.splitext(os.path.basename(path))[0]
    for suffix in ['_cached', '_dlib_cached']:
        cache_path = os.path.join('data', f'{project_name}{suffix}', f'{file_name}.npy')
        if os.path.exists(cache_path):
            frames = np.load(cache_path)
            mean, std = np.mean(frames), np.std(frames) + 1e-6
            return ((frames - mean) / std).astype(np.float32)
    raise FileNotFoundError(cache_path)

files = sorted(glob.glob('data/s99_1/*.mpg'))[:5]
print(f'測試 {len(files)} 支不同影片，同一個interpreter物件連續呼叫invoke()')

interpreter = tf.lite.Interpreter(model_path='models/width05_dynamic_int8.tflite')
interpreter.allocate_tensors()
in_idx = interpreter.get_input_details()[0]['index']
out_idx = interpreter.get_output_details()[0]['index']

print('\n===== 測試A：同一支影片，同一個interpreter連續invoke 5次 =====')
x = np.expand_dims(_load_frames(files[0]), 0).astype(np.float32)
prev_argmax = None
for i in range(5):
    interpreter.set_tensor(in_idx, x)
    interpreter.invoke()
    yhat = interpreter.get_tensor(out_idx)
    argmax = yhat[0].argmax(axis=-1)
    same_as_prev = 'N/A' if prev_argmax is None else f"{(argmax==prev_argmax).mean()*100:.1f}%"
    print(f'  第{i+1}次呼叫: argmax前10個={argmax[:10]}  跟上次一致比例={same_as_prev}')
    prev_argmax = argmax

print('\n===== 測試B：不同影片依序呼叫（模擬benchmark迴圈情境） =====')
for i, fp in enumerate(files):
    x = np.expand_dims(_load_frames(fp), 0).astype(np.float32)
    interpreter.set_tensor(in_idx, x)
    interpreter.invoke()
    yhat = interpreter.get_tensor(out_idx)
    argmax = yhat[0].argmax(axis=-1)
    print(f'  影片{i+1}({os.path.basename(fp)}): argmax前10個={argmax[:10]}  輸出mean={yhat.mean():.6f}')

print('\n===== 測試C：每支影片都重新建立新的interpreter物件（對照組） =====')
for i, fp in enumerate(files):
    interp2 = tf.lite.Interpreter(model_path='models/width05_dynamic_int8.tflite')
    interp2.allocate_tensors()
    x = np.expand_dims(_load_frames(fp), 0).astype(np.float32)
    interp2.set_tensor(interp2.get_input_details()[0]['index'], x)
    interp2.invoke()
    yhat = interp2.get_tensor(interp2.get_output_details()[0]['index'])
    argmax = yhat[0].argmax(axis=-1)
    print(f'  影片{i+1}({os.path.basename(fp)}): argmax前10個={argmax[:10]}  輸出mean={yhat.mean():.6f}')
