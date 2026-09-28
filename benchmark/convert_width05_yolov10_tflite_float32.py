"""把 width05_yolov10 轉成 TFLite float32 純格式轉換版本（不量化）。"""
import os
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
import tensorflow as tf
from tensorflow.keras.models import load_model

def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)

MODEL_PATH = 'models/trained_model_grid_multi_width05_yolov10.h5'
model = load_model(MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)

concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
    tf.TensorSpec([1, 75, 46, 140, 1], model.inputs[0].dtype))

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()

out_path = 'models/width05_yolov10_float32_baseline.tflite'
with open(out_path, 'wb') as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / (1024 * 1024)
orig_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f'[成功] 已存: {out_path}')
print(f'原始 .h5 大小: {orig_mb:.2f} MB')
print(f'轉換後大小: {size_mb:.2f} MB')
