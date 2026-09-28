"""
把 F_Width0.5 轉成 TensorFlow Lite，測試量化效果。

用 sys.argv[1] 指定要跑哪一種量化方式，每次只跑一種、跑完就結束程式，
避免同一個process連續轉換多次造成GPU資源殘留、狀態互相干擾的問題
（之前把三種塞在同一支程式裡跑，跑到第二種就無預警當掉過一次）。

用法：python convert_width05_tflite.py [dynamic_int8|float16|float32_baseline]
"""
import os
import sys
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import tensorflow as tf
from tensorflow.keras.models import load_model

def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)

MODE = sys.argv[1] if len(sys.argv) > 1 else 'dynamic_int8'
assert MODE in ('dynamic_int8', 'float16', 'float32_baseline')

MODEL_PATH = 'models/trained_model_grid_multi_width05.h5'
model = load_model(MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)

# Bidirectional LSTM 在 batch size 是 None（動態）時，TFLite 轉換器會因為
# TensorListReserve 需要靜態 shape 而失敗。修法：先固定 batch size=1，
# 用 concrete function 轉換，繞過這個限制（通話系統本來就是一次辨識一段影片，
# batch=1 完全符合實際使用情境，不影響功能）。
concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
    tf.TensorSpec([1, 75, 46, 140, 1], model.inputs[0].dtype))

print(f'\n===== 嘗試轉換：{MODE} =====', flush=True)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
# MaxPool3D 這種3D運算，TFLite內建運算子集不支援，加SELECT_TF_OPS讓它退回
# 用完整TensorFlow的運算子執行（其他層例如LSTM/Dense仍可用TFLite原生格式+量化）
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]

if MODE == 'dynamic_int8':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
elif MODE == 'float16':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
# float32_baseline: 不加任何 optimizations，單純格式轉換對照組

tflite_model = converter.convert()

out_path = f'models/width05_{MODE}.tflite'
with open(out_path, 'wb') as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / (1024 * 1024)
orig_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f'[成功] 已存: {out_path}', flush=True)
print(f'原始 .h5 大小: {orig_mb:.2f} MB', flush=True)
print(f'轉換後大小: {size_mb:.2f} MB', flush=True)
print(f'壓縮比: {orig_mb/size_mb:.2f}x', flush=True)
