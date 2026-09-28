"""
把 F_Width0.5 匯出成 SavedModel 格式（固定 batch=1），給另一個獨立環境
（tf215，裝 tf2onnx/onnx/onnxruntime）轉成 ONNX 用。

之所以匯出成 SavedModel 而不是直接在原環境轉 ONNX：
onnx 套件需要新版 protobuf，會跟這裡 TensorFlow 2.10 需要的舊版 protobuf
衝突（裝下去會讓 TensorFlow 整個壞掉，已實測過一次），所以改成兩階段：
這裡（py39, TF 2.10）只負責匯出 SavedModel，轉 ONNX 的動作交給另一個
完全獨立、不影響 py39 的環境（tf215）去做。

用法：python export_width05_savedmodel.py
"""
import os
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import tensorflow as tf
from tensorflow.keras.models import load_model


def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)


MODEL_PATH = 'models/trained_model_grid_multi_width05.h5'
model = load_model(MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)

# 跟 TFLite 轉換一樣：Bidirectional LSTM 在動態 batch 下轉換會出問題，
# 固定 batch=1（通話系統本來就一次辨識一段影片，不影響功能）
concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
    tf.TensorSpec([1, 75, 46, 140, 1], model.inputs[0].dtype, name='input'))

out_dir = 'models/width05_savedmodel_onnx'
tf.saved_model.save(model, out_dir, signatures={'serving_default': concrete_func})

print(f'[成功] SavedModel 已存到: {out_dir}')
