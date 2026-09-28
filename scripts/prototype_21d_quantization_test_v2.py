"""
延續 prototype_21d_quantization_test.py 的發現：上次的簡化玩具模型(單層Conv3D)
量化後雖然變慢但至少能跑，跟下午對完整LipNet量化「直接讀不進去」的狀況不一樣。
這次把玩具模型改得更接近LipNet真實架構——比照train_grid_multi_width05_yolov8.py
第155-157行的三層結構：[Conv3D(kernel=3,padding=same) → BatchNorm → ReLU → MaxPool3D]
重複3次，藉此縮小玩具模型跟真實模型的差距，看能不能重現「完全無法載入」的狀況，
藉此定位到底是哪個具體因素(BatchNorm折疊？MaxPool3D？三層堆疊？)造成完全失敗。

用法：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/prototype_21d_quantization_test_v2.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)

import tensorflow as tf
from tensorflow.keras import layers, models
import subprocess
import json

INPUT_SHAPE = (75, 46, 140, 1)
TMP_DIR = os.path.join(BASE_DIR, 'scripts', '_prototype_tmp2')
os.makedirs(TMP_DIR, exist_ok=True)


def build_lipnet_style_conv3d(c1=16, c2=32, c3=48):
    """完全比照train_grid_multi_width05_yolov8.py第155-157行的3層結構"""
    inp = layers.Input(shape=INPUT_SHAPE, name='input')
    x = layers.Conv3D(c1, 3, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool3D((1, 2, 2))(x)

    x = layers.Conv3D(c2, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool3D((1, 2, 2))(x)

    x = layers.Conv3D(c3, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool3D((1, 2, 2))(x)

    out = layers.GlobalAveragePooling3D()(x)
    return models.Model(inp, out, name='lipnet_style_conv3d')


def export_and_quantize(model, name):
    saved_dir = os.path.join(TMP_DIR, f'{name}_savedmodel')
    onnx_path = os.path.join(TMP_DIR, f'{name}.onnx')
    int8_path = os.path.join(TMP_DIR, f'{name}_int8.onnx')

    concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
        tf.TensorSpec([1] + list(INPUT_SHAPE), model.inputs[0].dtype, name='input'))
    tf.saved_model.save(model, saved_dir, signatures={'serving_default': concrete_func})

    cmd = [
        r'C:\Users\Tno\miniconda3\envs\tf215\python.exe', '-m', 'tf2onnx.convert',
        '--saved-model', saved_dir, '--output', onnx_path, '--opset', '15'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(onnx_path):
        return {'onnx_ok': False, 'error': result.stderr[-500:]}

    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024

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
        results[label] = ('FAIL', str(e)[:300])

import json
print('RESULT_JSON:' + json.dumps(results))
""")
    result2 = subprocess.run(
        [r'C:\Users\Tno\miniconda3\envs\tf215\python.exe', test_script],
        capture_output=True, text=True
    )
    out = result2.stdout
    err = result2.stderr

    quant_result = None
    for line in out.splitlines():
        if line.startswith('RESULT_JSON:'):
            quant_result = json.loads(line[len('RESULT_JSON:'):])
    int8_size = os.path.getsize(int8_path) / 1024 / 1024 if os.path.exists(int8_path) else None

    return {
        'onnx_ok': True, 'onnx_size_mb': onnx_size, 'int8_size_mb': int8_size,
        'quant_result': quant_result, 'stderr_tail': err[-500:] if quant_result is None else None,
    }


print('=' * 70)
print('LipNet風格3層Conv3D+BatchNorm+MaxPool3D堆疊 量化測試')
print('=' * 70)

model = build_lipnet_style_conv3d()
print(f'參數量: {model.count_params():,}')

r = export_and_quantize(model, 'lipnet_style')
print('\n結果:')
if not r.get('onnx_ok'):
    print(f'  ONNX匯出失敗: {r.get("error")}')
else:
    print(f'  ONNX大小: {r["onnx_size_mb"]:.3f}MB, INT8大小: {r["int8_size_mb"]:.3f}MB' if r["int8_size_mb"] else '  量化檔案未產生')
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
