"""
延續v2的發現：3層Conv3D+BatchNorm+MaxPool3D堆疊，量化後慢39倍（比單層Conv3D的21倍更誇張，
隨層數疊加惡化）。這次把同樣的3層堆疊，全部換成(2+1)D版本（2D空間卷積+1D時間卷積取代
每一層Conv3D），做最直接的蘋果比蘋果對照：3層Conv3D堆疊 vs 3層(2+1)D堆疊，量化後
分別慢多少倍？這是目前最接近「如果真的把LipNet改成(2+1)D架構」的預測性測試。

用法：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/prototype_21d_quantization_test_v3.py
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
TMP_DIR = os.path.join(BASE_DIR, 'scripts', '_prototype_tmp3')
os.makedirs(TMP_DIR, exist_ok=True)


def conv21d_block(x, filters):
    """一個(2+1)D區塊：TimeDistributed Conv2D(空間) → BatchNorm → ReLU → TimeDistributed MaxPool2D
    → 時間軸1D卷積(用Permute+Reshape把T移到跟C相鄰，每個空間位置獨立套用同一組時間卷積權重，
    正確保留(T,H,W,C)結構，不是v2版那個會讓參數量爆炸的錯誤flatten寫法) → BatchNorm → ReLU"""
    x = layers.TimeDistributed(layers.Conv2D(filters, 3, padding='same'))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.TimeDistributed(layers.MaxPool2D((2, 2)))(x)  # (batch,T,H,W,C)

    T, H, W, C = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
    # (batch,T,H,W,C) -> (batch,H,W,T,C)：把時間軸搬到跟channel相鄰，方便對每個空間位置獨立做時間卷積
    x = layers.Permute((2, 3, 1, 4))(x)
    # (batch,H,W,T,C) -> (batch,H*W,T,C)：合併空間維度，當成TimeDistributed的「時間」軸
    x = layers.Reshape((H * W, T, C))(x)
    # 對每個空間位置(H*W個)，各自套用「同一組」沿時間軸的1D卷積權重(空間共享，跟卷積定義一致)
    x = layers.TimeDistributed(layers.Conv1D(filters, 3, padding='same'))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # 换回(batch,H,W,T,C) -> (batch,T,H,W,C)
    x = layers.Reshape((H, W, T, filters))(x)
    x = layers.Permute((3, 1, 2, 4))(x)
    return x


def build_21d_3layer(c1=16, c2=32, c3=48):
    inp = layers.Input(shape=INPUT_SHAPE, name='input')
    x = conv21d_block(inp, c1)
    x = conv21d_block(x, c2)
    x = conv21d_block(x, c3)
    out = layers.GlobalAveragePooling3D()(x)
    return models.Model(inp, out, name='conv21d_3layer')


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
        return {'onnx_ok': False, 'error': result.stderr[-800:]}

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
        'quant_result': quant_result, 'stderr_tail': err[-800:] if quant_result is None else None,
    }


print('=' * 70)
print('3層(2+1)D堆疊 vs 3層Conv3D堆疊（v2結果：慢39倍）蘋果比蘋果對照')
print('=' * 70)

model = build_21d_3layer()
print(f'參數量: {model.count_params():,}（v2的Conv3D版是56,208）')

r = export_and_quantize(model, 'conv21d_3layer')
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
        if qr.get('fp32', ['', 0])[0] == 'OK' and qr.get('int8', ['', 0])[0] == 'OK':
            ratio = qr['int8'][1] / qr['fp32'][1]
            print(f'\n  量化後變慢倍數: {ratio:.1f}x（v2的3層Conv3D版是39x）')
    else:
        print(f'  量化測試沒有結果，stderr: {r.get("stderr_tail")}')
print('=' * 70)
