"""測 width05_yolov8 四種格式的CPU推論速度，用model.predict()，跟原始基準腳本方法一致。"""
import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
import numpy as np

N_WARMUP, N_RUNS = 3, 20
dummy = np.random.rand(1, 75, 46, 140, 1).astype(np.float32)
results = {}

print('=== .h5 (TensorFlow, model.predict) ===', flush=True)
import tensorflow as tf
from tensorflow.keras.models import load_model
def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl,1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl,1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)
model = load_model('models/trained_model_grid_multi_width05_yolov8.h5', custom_objects={'CTCLoss': CTCLoss}, compile=False)
for _ in range(N_WARMUP): model.predict(dummy, verbose=0)
times = []
for _ in range(N_RUNS):
    t0 = time.time(); model.predict(dummy, verbose=0); times.append((time.time()-t0)*1000)
times = np.array(times)
print(f'h5: 平均 {times.mean():.1f}ms, 中位數 {np.median(times):.1f}ms', flush=True)
results['h5'] = times.mean()
del model
tf.keras.backend.clear_session()

print('\n=== ONNX float32 ===', flush=True)
import onnxruntime as ort
sess = ort.InferenceSession('models/width05_yolov8.onnx', providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
for _ in range(N_WARMUP): sess.run(None, {in_name: dummy})
times = []
for _ in range(N_RUNS):
    t0 = time.time(); sess.run(None, {in_name: dummy}); times.append((time.time()-t0)*1000)
times = np.array(times)
print(f'ONNX float32: 平均 {times.mean():.1f}ms, 中位數 {np.median(times):.1f}ms', flush=True)
results['onnx_fp32'] = times.mean()

print('\n=== ONNX INT8 (uint8w) ===', flush=True)
sess2 = ort.InferenceSession('models/width05_yolov8_int8_uint8w.onnx', providers=['CPUExecutionProvider'])
in_name2 = sess2.get_inputs()[0].name
for _ in range(N_WARMUP): sess2.run(None, {in_name2: dummy})
times = []
for _ in range(N_RUNS):
    t0 = time.time(); sess2.run(None, {in_name2: dummy}); times.append((time.time()-t0)*1000)
times = np.array(times)
print(f'ONNX INT8: 平均 {times.mean():.1f}ms, 中位數 {np.median(times):.1f}ms', flush=True)
results['onnx_int8'] = times.mean()

print('\n=== TFLite INT8 ===', flush=True)
interp = tf.lite.Interpreter(model_path='models/width05_yolov8_dynamic_int8.tflite')
interp.allocate_tensors()
in_idx = interp.get_input_details()[0]['index']
out_idx = interp.get_output_details()[0]['index']
for _ in range(N_WARMUP):
    interp.set_tensor(in_idx, dummy); interp.invoke()
times = []
for _ in range(N_RUNS):
    interp2 = tf.lite.Interpreter(model_path='models/width05_yolov8_dynamic_int8.tflite')
    interp2.allocate_tensors()
    idx_in = interp2.get_input_details()[0]['index']
    idx_out = interp2.get_output_details()[0]['index']
    t0 = time.time()
    interp2.set_tensor(idx_in, dummy)
    interp2.invoke()
    times.append((time.time()-t0)*1000)
times = np.array(times)
print(f'TFLite INT8: 平均 {times.mean():.1f}ms, 中位數 {np.median(times):.1f}ms', flush=True)
results['tflite_int8'] = times.mean()

print('\n' + '='*60)
print('總結（width05_yolov8, CPU, model.predict/等效方法）')
print('='*60)
for k, v in results.items():
    ratio = results['h5'] / v
    print(f'{k:15s}: {v:8.1f}ms   相對h5: {ratio:.2f}x')
