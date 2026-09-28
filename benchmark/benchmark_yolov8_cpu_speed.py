"""測 width05_yolov8 四種格式的CPU推論速度，用假資料，比照之前v11/width05比較的方法論。"""
import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
import numpy as np

N_WARMUP, N_RUNS = 3, 20
dummy = np.random.rand(1, 75, 46, 140, 1).astype(np.float32)

print('=== .h5 (TensorFlow) ===')
import tensorflow as tf
from tensorflow.keras.models import load_model
def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl,1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl,1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)
model = load_model('models/trained_model_grid_multi_width05_yolov8.h5', custom_objects={'CTCLoss': CTCLoss}, compile=False)
for _ in range(N_WARMUP): model(dummy, training=False)
times = []
for _ in range(N_RUNS):
    t0 = time.time(); model(dummy, training=False); times.append((time.time()-t0)*1000)
times = np.array(times)
print(f'h5: 平均 {times.mean():.1f}ms, 中位數 {np.median(times):.1f}ms')
h5_time = times.mean()
del model
import gc; gc.collect()
