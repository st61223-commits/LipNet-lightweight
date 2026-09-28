"""
嘗試一個之前沒試過的方向：對ONNX模型做INT8動態量化（dynamic quantization）。

背景：8/21已經試過「TFLite量化」，結論是檔案有變小(47.15MB→4.95MB)、正確率幾乎沒掉，
但CPU推論速度反而變慢(300ms→2350ms)，原因是MaxPool3D這個運算子TFLite不支援，
量化後的模型要整個退回完整TF引擎執行，速度被這個相容性問題拖垮，不是量化本身的錯。

這次改用onnxruntime自帶的量化工具，對「已經在用的ONNX模型」(width05_yolov8.onnx)
做INT8動態量化——onnxruntime支援的運算子集合跟TFLite不同，理論上有機會繞開
MaxPool3D那個相容性地雷，值得試試看是否能拿到「檔案更小+速度更快」雙重好處，
而不是像TFLite那樣顧此失彼。

用法：
  /mnt/c/Users/Tno/miniconda3/envs/tf215/python.exe scripts/quantize_onnx_int8.py
"""
import os
import time

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)

from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np

SRC = os.path.join(BASE_DIR, 'models', 'width05_yolov8.onnx')
DST = os.path.join(BASE_DIR, 'models', 'width05_yolov8_int8.onnx')

print(f'onnxruntime 版本: {ort.__version__}')
print(f'原始模型: {SRC}')
print(f'  大小: {os.path.getsize(SRC) / 1024 / 1024:.2f} MB\n')

print('執行INT8動態量化...')
t0 = time.time()
try:
    quantize_dynamic(SRC, DST, weight_type=QuantType.QInt8)
    print(f'  量化耗時: {time.time()-t0:.1f}秒')
    print(f'  量化後大小: {os.path.getsize(DST) / 1024 / 1024:.2f} MB')
    print(f'  縮小比例: {(1 - os.path.getsize(DST)/os.path.getsize(SRC))*100:.1f}%\n')
except Exception as e:
    print(f'❌ 量化失敗: {type(e).__name__}: {e}')
    raise

# 速度比較：兩個模型各跑20次CPU推論
dummy = np.random.rand(1, 75, 46, 140, 1).astype(np.float32)
N_WARMUP, N_RUNS = 3, 20

for label, path in [('原始ONNX(float32)', SRC), ('量化ONNX(int8)', DST)]:
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    for _ in range(N_WARMUP):
        session.run(None, {input_name: dummy})
    times = []
    for _ in range(N_RUNS):
        t0 = time.time()
        session.run(None, {input_name: dummy})
        times.append((time.time() - t0) * 1000)
    times = np.array(times)
    print(f'{label:20s}: 平均 {times.mean():7.1f}ms, 中位數 {np.median(times):7.1f}ms')

print('\n量化模型已存到:', DST)
print('下一步：如果速度有改善，需要再驗證正確率是否保留（用真實測試資料跑CTC解碼比對）')
