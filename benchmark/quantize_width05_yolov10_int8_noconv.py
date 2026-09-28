"""
對 width05_yolov10.onnx 做 INT8 動態量化，跳過 Conv（避開 Conv3D 的 ConvInteger 相容性問題），
只量化 MatMul/LSTM，比照之前 F_Width0.5 時代 width05_int8_noconv.onnx 的做法。
"""
import os, time
BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)

from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

SRC = os.path.join(BASE_DIR, 'models', 'width05_yolov10.onnx')
DST = os.path.join(BASE_DIR, 'models', 'width05_yolov10_int8_noconv.onnx')

print(f'onnxruntime 版本: {ort.__version__}')
print(f'原始模型大小: {os.path.getsize(SRC) / 1024 / 1024:.2f} MB')

t0 = time.time()
quantize_dynamic(SRC, DST, weight_type=QuantType.QUInt8, op_types_to_quantize=['MatMul', 'LSTM'])
print(f'量化耗時: {time.time()-t0:.1f}秒')
print(f'量化後大小: {os.path.getsize(DST) / 1024 / 1024:.2f} MB')
print(f'縮小比例: {(1 - os.path.getsize(DST)/os.path.getsize(SRC))*100:.1f}%')

print('\n驗證能否正常推論...')
import numpy as np
sess = ort.InferenceSession(DST, providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
dummy = np.random.rand(1, 75, 46, 140, 1).astype(np.float32)
out = sess.run(None, {in_name: dummy})[0]
print(f'[成功] 輸出shape={out.shape}, min={out.min():.4f}, max={out.max():.4f}')
