"""
(2+1)D分解卷積的概念驗證實驗——不重新訓練完整LipNet，只用「隨機權重的小型測試模型」
驗證一個具體問題：把Conv3D換成「2D空間卷積+1D時間卷積」之後，INT8動態量化
是否真的能繞開今天卡住的ConvInteger/Conv3D不支援問題？

背景：查到Tran et al. 2018的R(2+1)D論文，主張2D+1D卷積在主流推論引擎上支援
更成熟。但這是別人在別的任務(動作辨識)上的結論，不確定在我們用的onnxruntime
版本上是否真的成立。與其直接冒險把整個LipNet架構改掉重新訓練(要花很多小時)，
先用小規模測試驗證這個技術假設是否成立，值得才投入大工程。

做法：蓋兩個小模型，輸入形狀比照LipNet的(75,46,140,1)：
  A. baseline：一層Conv3D（模擬LipNet現有架構的量化困境）
  B. (2+1)D：一層2D空間卷積(對每個時間幀獨立做) + 一層1D時間卷積(沿時間軸)
兩個都做INT8動態量化，比較「量化後能不能正常load並跑」、「量化後速度」。

用法：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/prototype_21d_quantization_test.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import time
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

INPUT_SHAPE = (75, 46, 140, 1)  # 跟LipNet輸入一樣：75幀、46x140、灰階
TMP_DIR = os.path.join(BASE_DIR, 'scripts', '_prototype_tmp')
os.makedirs(TMP_DIR, exist_ok=True)


def build_conv3d_model():
    """baseline：標準Conv3D，模擬LipNet現有架構的第一層"""
    inp = layers.Input(shape=INPUT_SHAPE, name='input')
    x = layers.Conv3D(32, (3, 5, 5), strides=(1, 2, 2), padding='same', activation='relu')(inp)
    x = layers.MaxPool3D((1, 2, 2))(x)
    out = layers.GlobalAveragePooling3D()(x)
    return models.Model(inp, out, name='conv3d_baseline')


def build_21d_model():
    """(2+1)D版本：2D空間卷積(TimeDistributed，對每幀獨立做) + 1D時間卷積"""
    inp = layers.Input(shape=INPUT_SHAPE, name='input')
    # 空間卷積：對75幀裡的每一幀，各自做2D卷積（用TimeDistributed包裝Conv2D）
    x = layers.TimeDistributed(
        layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='relu')
    )(inp)  # (batch, 75, 23, 70, 32)
    x = layers.TimeDistributed(layers.MaxPool2D((2, 2)))(x)  # (batch, 75, 11, 35, 32)
    # 時間卷積：把空間維度攤平，沿著時間軸做1D卷積
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3] * shape[4]))(x)  # (batch, 75, feat)
    x = layers.Conv1D(32, 3, strides=1, padding='same', activation='relu')(x)
    out = layers.GlobalAveragePooling1D()(x)
    return models.Model(inp, out, name='conv21d_prototype')


def count_params(model):
    return model.count_params()


def export_and_quantize(model, name):
    """匯出成SavedModel→ONNX，再做INT8動態量化，回傳(onnx路徑, 量化後路徑, 是否成功載入, 速度資訊)"""
    saved_dir = os.path.join(TMP_DIR, f'{name}_savedmodel')
    onnx_path = os.path.join(TMP_DIR, f'{name}.onnx')
    int8_path = os.path.join(TMP_DIR, f'{name}_int8.onnx')

    # 比照正式流程(export_width05_yolov8_savedmodel.py)的做法：固定batch=1匯出
    concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
        tf.TensorSpec([1] + list(INPUT_SHAPE), model.inputs[0].dtype, name='input'))
    tf.saved_model.save(model, saved_dir, signatures={'serving_default': concrete_func})

    # 用tf215環境的tf2onnx做轉換（跟正式流程一樣，透過subprocess呼叫）
    import subprocess
    cmd = [
        r'C:\Users\Tno\miniconda3\envs\tf215\python.exe', '-m', 'tf2onnx.convert',
        '--saved-model', saved_dir, '--output', onnx_path, '--opset', '15'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(onnx_path):
        return {'onnx_ok': False, 'error': result.stderr[-500:]}

    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024

    # 量化 + 測試載入速度，透過subprocess在tf215環境做（onnxruntime裝在那邊）
    test_script = os.path.join(TMP_DIR, f'_test_{name}.py')
    with open(test_script, 'w', encoding='utf-8') as f:
        f.write(f"""
import time, numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

quantize_dynamic(r'{onnx_path}', r'{int8_path}', weight_type=QuantType.QUInt8)

dummy = np.random.rand(1, 75, 46, 140, 1).astype(np.float32)
results = {{}}
for label, path in [('fp32', r'{onnx_path}'), ('int8', r'{int8_path}')]:
    try:
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        in_name = sess.get_inputs()[0].name
        for _ in range(3):
            sess.run(None, {{in_name: dummy}})
        times = []
        for _ in range(10):
            t0 = time.time()
            sess.run(None, {{in_name: dummy}})
            times.append((time.time()-t0)*1000)
        results[label] = ('OK', float(np.mean(times)))
    except Exception as e:
        results[label] = ('FAIL', str(e)[:200])

import json
print('RESULT_JSON:' + json.dumps(results))
""")
    result2 = subprocess.run(
        [r'C:\Users\Tno\miniconda3\envs\tf215\python.exe', test_script],
        capture_output=True, text=True
    )
    out = result2.stdout
    err = result2.stderr

    import json
    quant_result = None
    for line in out.splitlines():
        if line.startswith('RESULT_JSON:'):
            quant_result = json.loads(line[len('RESULT_JSON:'):])
    int8_size = os.path.getsize(int8_path) / 1024 / 1024 if os.path.exists(int8_path) else None

    return {
        'onnx_ok': True, 'onnx_size_mb': onnx_size, 'int8_size_mb': int8_size,
        'quant_result': quant_result, 'stderr_tail': err[-300:] if quant_result is None else None,
    }


print('=' * 70)
print('(2+1)D分解卷積 概念驗證實驗')
print('=' * 70)

print('\n建立 Conv3D baseline 模型...')
model_a = build_conv3d_model()
print(f'  參數量: {count_params(model_a):,}')

print('\n建立 (2+1)D 模型...')
model_b = build_21d_model()
print(f'  參數量: {count_params(model_b):,}')

print('\n匯出+量化 Conv3D baseline...')
r_a = export_and_quantize(model_a, 'conv3d')
print(f'  結果: {r_a}')

print('\n匯出+量化 (2+1)D 模型...')
r_b = export_and_quantize(model_b, 'conv21d')
print(f'  結果: {r_b}')

print('\n' + '=' * 70)
print('總結')
print('=' * 70)
for label, r in [('Conv3D baseline', r_a), ('(2+1)D', r_b)]:
    print(f'\n{label}:')
    if not r.get('onnx_ok'):
        print(f'  ONNX匯出失敗: {r.get("error")}')
        continue
    print(f'  ONNX大小: {r["onnx_size_mb"]:.2f}MB, INT8大小: {r["int8_size_mb"]:.2f}MB' if r["int8_size_mb"] else '  量化檔案未產生')
    qr = r.get('quant_result')
    if qr:
        for prec, (status, val) in qr.items():
            if status == 'OK':
                print(f'  {prec}: 成功, 平均 {val:.1f}ms')
            else:
                print(f'  {prec}: 失敗 - {val}')
    else:
        print(f'  量化測試沒有結果，stderr: {r.get("stderr_tail")}')
print('=' * 70)
