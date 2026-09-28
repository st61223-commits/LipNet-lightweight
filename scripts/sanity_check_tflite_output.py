"""
比對同一支影片，分別餵給「原始.h5模型」跟「TFLite INT8量化模型」，
直接看兩邊輸出的機率分布長什麼樣子，判斷正確率崩潰(0.3%)是量化真的
把模型弄壞了，還是我的TFLite benchmark腳本本身寫錯了。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)

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

import glob
sample_path = sorted(glob.glob('data/s99_1/*.mpg'))[0]
print('測試樣本:', sample_path)
frames = _load_frames(sample_path)
print('frames shape:', frames.shape, 'dtype:', frames.dtype)
x = np.expand_dims(frames, 0)  # (1,75,46,140,1)

print('\n===== 原始 .h5 模型 =====')
model = load_model('models/trained_model_grid_multi_width05.h5', custom_objects={'CTCLoss': CTCLoss}, compile=False)
yhat_h5 = model.predict(x, verbose=0)
print('輸出 shape:', yhat_h5.shape)
print('輸出範圍: min=%.6f max=%.6f mean=%.6f' % (yhat_h5.min(), yhat_h5.max(), yhat_h5.mean()))
print('每個time step機率總和(應該接近1，softmax輸出):', yhat_h5[0].sum(axis=-1)[:5], '...')
argmax_h5 = yhat_h5[0].argmax(axis=-1)
print('前20個time step的argmax:', argmax_h5[:20])

print('\n===== TFLite INT8量化模型 =====')
interpreter = tf.lite.Interpreter(model_path='models/width05_dynamic_int8.tflite')
interpreter.allocate_tensors()
in_idx = interpreter.get_input_details()[0]['index']
out_idx = interpreter.get_output_details()[0]['index']
interpreter.set_tensor(in_idx, x.astype(np.float32))
interpreter.invoke()
yhat_tflite = interpreter.get_tensor(out_idx)
print('輸出 shape:', yhat_tflite.shape)
print('輸出範圍: min=%.6f max=%.6f mean=%.6f' % (yhat_tflite.min(), yhat_tflite.max(), yhat_tflite.mean()))
print('每個time step機率總和(應該接近1，softmax輸出):', yhat_tflite[0].sum(axis=-1)[:5], '...')
argmax_tflite = yhat_tflite[0].argmax(axis=-1)
print('前20個time step的argmax:', argmax_tflite[:20])

print('\n===== 差異比較 =====')
diff = np.abs(yhat_h5 - yhat_tflite)
print('絕對誤差: max=%.6f mean=%.6f' % (diff.max(), diff.mean()))
same_argmax = (argmax_h5 == argmax_tflite).mean()
print(f'argmax(每個time step選的字元)一致比例: {same_argmax*100:.1f}%')
