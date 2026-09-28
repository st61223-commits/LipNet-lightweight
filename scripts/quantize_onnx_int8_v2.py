"""
重試ONNX INT8量化，這次改用QUInt8而不是QInt8作為權重類型。

背景：查到onnxruntime官方GitHub issue(#15888，跟我們遇到的錯誤逐字相同)，
根因是ConvInteger這個運算子最初只支援uint8×uint8這個型別組合，
uint8(動態量化的activation預設就是uint8)×int8(weight_type=QInt8，我們原本用的設定)
這個組合在較舊版本沒有實作。修復這個問題的官方PR(#26585)在2025-11-24才合併進主分支，
現在環境的onnxruntime是1.23.2，不確定是否已包含這個修復。與其等版本升級，
先直接試「把weight_type改成QUInt8」，讓量化出來剛好是uint8×uint8組合，
這個組合按issue的說法本來就有支援，理論上不需要等新版本就能繞開這個問題。

用法：
  /mnt/c/Users/Tno/miniconda3/envs/tf215/python.exe scripts/quantize_onnx_int8_v2.py
"""
import os
import time

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)

from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np

SRC = os.path.join(BASE_DIR, 'models', 'width05_yolov8.onnx')
DST = os.path.join(BASE_DIR, 'models', 'width05_yolov8_int8_uint8w.onnx')

print(f'onnxruntime 版本: {ort.__version__}')
print(f'原始模型大小: {os.path.getsize(SRC) / 1024 / 1024:.2f} MB\n')

print('執行INT8動態量化（weight_type=QUInt8，跟前一次QInt8不同）...')
t0 = time.time()
quantize_dynamic(SRC, DST, weight_type=QuantType.QUInt8)
print(f'  量化耗時: {time.time()-t0:.1f}秒')
print(f'  量化後大小: {os.path.getsize(DST) / 1024 / 1024:.2f} MB')
print(f'  縮小比例: {(1 - os.path.getsize(DST)/os.path.getsize(SRC))*100:.1f}%\n')

print('嘗試載入量化後模型並跑一次推論...')
try:
    session = ort.InferenceSession(DST, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy = np.random.rand(1, 75, 46, 140, 1).astype(np.float32)

    for _ in range(3):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(20):
        t0 = time.time()
        session.run(None, {input_name: dummy})
        times.append((time.time() - t0) * 1000)
    times = np.array(times)
    print(f'[成功] 成功載入並推論！平均 {times.mean():.1f}ms, 中位數 {np.median(times):.1f}ms, '
          f'最快 {times.min():.1f}ms, 最慢 {times.max():.1f}ms')

    # 跟原始float32模型比較
    session_fp32 = ort.InferenceSession(SRC, providers=['CPUExecutionProvider'])
    input_name_fp32 = session_fp32.get_inputs()[0].name
    for _ in range(3):
        session_fp32.run(None, {input_name_fp32: dummy})
    times_fp32 = []
    for _ in range(20):
        t0 = time.time()
        session_fp32.run(None, {input_name_fp32: dummy})
        times_fp32.append((time.time() - t0) * 1000)
    times_fp32 = np.array(times_fp32)
    print(f'\n原始float32模型: 平均 {times_fp32.mean():.1f}ms')
    print(f'量化uint8模型:   平均 {times.mean():.1f}ms')
    print(f'速度比: {times_fp32.mean()/times.mean():.2f}x')
except Exception as e:
    print(f'[失敗] 仍然失敗: {type(e).__name__}: {e}')
