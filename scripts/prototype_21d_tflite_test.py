"""
今天研究線的最後一塊拼圖：最早(8/21)發現的TFLite量化問題根源是MaxPool3D不支援，
逼得整個模型要退回SELECT_TF_OPS完整TF引擎執行，速度暴增16倍。我們的(2+1)D設計
用的是MaxPool2D(不是MaxPool3D)，理論上應該能讓TFLite轉換時完全不需要退回
SELECT_TF_OPS。這支腳本驗證這個閉環：用今天v3驗證過的3層(2+1)D模型轉TFLite，
檢查轉換時是不是真的不再需要SELECT_TF_OPS。

用法：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/prototype_21d_tflite_test.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import time
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)
sys.path.insert(0, 'scripts')

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

INPUT_SHAPE = (75, 46, 140, 1)
TMP_DIR = os.path.join(BASE_DIR, 'scripts', '_prototype_tmp4')
os.makedirs(TMP_DIR, exist_ok=True)


def conv3d_style_block(x, filters):
    x = layers.Conv3D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool3D((1, 2, 2))(x)
    return x


def conv21d_block(x, filters):
    x = layers.TimeDistributed(layers.Conv2D(filters, 3, padding='same'))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.TimeDistributed(layers.MaxPool2D((2, 2)))(x)
    T, H, W, C = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
    x = layers.Permute((2, 3, 1, 4))(x)
    x = layers.Reshape((H * W, T, C))(x)
    x = layers.TimeDistributed(layers.Conv1D(filters, 3, padding='same'))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Reshape((H, W, T, filters))(x)
    x = layers.Permute((3, 1, 2, 4))(x)
    return x


def build(block_fn, name, c=(16, 32, 48)):
    inp = layers.Input(shape=INPUT_SHAPE, name='input')
    x = block_fn(inp, c[0])
    x = block_fn(x, c[1])
    x = block_fn(x, c[2])
    out = layers.GlobalAveragePooling3D()(x)
    return models.Model(inp, out, name=name)


def try_tflite_convert(model, name):
    """嘗試用最嚴格的設定轉換(只允許TFLITE_BUILTINS，不給SELECT_TF_OPS退路)，
    看轉換會不會失敗——如果失敗，代表這個架構還是有op不被原生支援。"""
    saved_dir = os.path.join(TMP_DIR, f'{name}_savedmodel')
    concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
        tf.TensorSpec([1] + list(INPUT_SHAPE), model.inputs[0].dtype, name='input'))
    tf.saved_model.save(model, saved_dir, signatures={'serving_default': concrete_func})

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    # 關鍵：只允許原生TFLite builtin ops，不給SELECT_TF_OPS退路，藉此測試「真的完全原生支援」
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    try:
        tflite_model = converter.convert()
        tflite_path = os.path.join(TMP_DIR, f'{name}.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        # 跑一次推論驗證真的能用，並計時
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        dummy = np.random.rand(*input_details[0]['shape']).astype(np.float32)

        for _ in range(3):
            interpreter.set_tensor(input_details[0]['index'], dummy)
            interpreter.invoke()

        times = []
        for _ in range(10):
            t0 = time.time()
            interpreter.set_tensor(input_details[0]['index'], dummy)
            interpreter.invoke()
            times.append((time.time() - t0) * 1000)
        return {'ok': True, 'size_mb': len(tflite_model) / 1024 / 1024,
                'mean_ms': float(np.mean(times))}
    except Exception as e:
        return {'ok': False, 'error': str(e)[:400]}


print('=' * 70)
print('TFLite轉換測試：只允許原生builtin ops(不給SELECT_TF_OPS退路)')
print('=' * 70)

print('\n建立並測試 Conv3D+MaxPool3D 3層堆疊...')
model_a = build(conv3d_style_block, 'conv3d_tflite_test')
r_a = try_tflite_convert(model_a, 'conv3d_tflite_test')
print(f'結果: {r_a}')

print('\n建立並測試 (2+1)D 3層堆疊...')
model_b = build(conv21d_block, 'conv21d_tflite_test')
r_b = try_tflite_convert(model_b, 'conv21d_tflite_test')
print(f'結果: {r_b}')

print('\n' + '=' * 70)
print('結論')
print('=' * 70)
print(f'Conv3D+MaxPool3D: {"純原生TFLite轉換成功！" if r_a.get("ok") else "轉換失敗，需要SELECT_TF_OPS退路 -> " + str(r_a.get("error"))[:150]}')
print(f'(2+1)D:           {"純原生TFLite轉換成功！" if r_b.get("ok") else "轉換失敗，需要SELECT_TF_OPS退路 -> " + str(r_b.get("error"))[:150]}')
print('=' * 70)
