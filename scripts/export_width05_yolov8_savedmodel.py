"""
把 F_Width0.5「資料修復後續訓練版」(width05_datafix) 匯出成 SavedModel 格式
（固定 batch=1），給另一個獨立環境（tf215，裝 tf2onnx/onnx/onnxruntime）轉成 ONNX 用。

做法比照 export_width05_savedmodel.py（原版模型），只是換成 datafix 權重，
輸出目錄也換成獨立的一份，不覆蓋原本 width05 的 SavedModel/ONNX。

用法：python export_width05_datafix_savedmodel.py
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


MODEL_PATH = 'models/trained_model_grid_multi_width05_yolov8.h5'
model = load_model(MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)

# 跟原版一樣：Bidirectional LSTM 在動態 batch 下轉換會出問題，固定 batch=1
concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
    tf.TensorSpec([1, 75, 46, 140, 1], model.inputs[0].dtype, name='input'))

out_dir = 'models/width05_yolov8_savedmodel_onnx'
tf.saved_model.save(model, out_dir, signatures={'serving_default': concrete_func})

print(f'[成功] SavedModel 已存到: {out_dir}')
