"""
獨立、輕量的效能測試腳本，設計給「真正的舊筆電」用，不需要裝整套TensorFlow/conda環境，
只需要 onnxruntime + numpy 兩個套件（跟LipNetConnect1.py在沒有GPU時走的路徑一樣）。

這支腳本不需要攝影機、不需要YOLO、不需要GRID資料集，純粹測「這台筆電的CPU
跑LipNet模型一次要多久」，藉此驗證專題的目標硬體方向（舊筆電CPU-only可用）。

使用方式（在要測試的舊筆電上）：
  1. 安裝 Python 3.9 以上（https://www.python.org/downloads/ 下載安裝，
     安裝時記得勾選「Add Python to PATH」）
  2. 開命令提示字元(cmd)，輸入：pip install onnxruntime numpy
  3. 把這支腳本，跟 models/width05_yolov8.onnx 這個模型檔案，複製到同一個資料夾
     （模型檔案在 C:\\Users\\Tno\\OneDrive\\Lipnet_nchu\\LipNet\\models\\ 底下，
     可以用隨身碟或雲端硬碟複製過去，檔案約47MB）
  4. 在命令提示字元裡，切換到那個資料夾，執行：
       python benchmark_old_laptop.py
  5. 跑完會印出結果，把結果截圖或複製文字回報即可
"""
import os
import sys
import time
import platform

print('=' * 60)
print('LipNet 舊筆電CPU效能測試')
print('=' * 60)
print(f'作業系統: {platform.system()} {platform.release()}')
print(f'處理器: {platform.processor()}')
print(f'Python 版本: {sys.version.split()[0]}')
try:
    import multiprocessing
    print(f'CPU核心數: {multiprocessing.cpu_count()}')
except Exception:
    pass
print()

try:
    import numpy as np
except ImportError:
    print('❌ 找不到 numpy，請先執行: pip install numpy')
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print('❌ 找不到 onnxruntime，請先執行: pip install onnxruntime')
    sys.exit(1)

MODEL_PATH = 'width05_yolov8.onnx'
if not os.path.exists(MODEL_PATH):
    print(f'❌ 找不到模型檔案 {MODEL_PATH}')
    print('   請確認這支腳本跟 width05_yolov8.onnx 放在同一個資料夾')
    sys.exit(1)

print(f'onnxruntime 版本: {ort.__version__}')
print(f'載入模型: {MODEL_PATH} ...')

t0 = time.time()
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
load_time = time.time() - t0
print(f'  模型載入耗時: {load_time:.2f} 秒\n')

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f'模型輸入名稱: {input_name}, 形狀: {input_shape}')

# 用假資料測試（跟真正的嘴唇畫面形狀一樣：75幀、46x140像素、灰階），
# 我們只在乎「跑一次要多久」，內容是隨機數字不影響速度
dummy = np.random.rand(1, 75, 46, 140, 1).astype(np.float32)

N_WARMUP = 3
N_RUNS = 20

print(f'\n先跑 {N_WARMUP} 次熱身（不計入計時，讓CPU快取穩定）...')
for _ in range(N_WARMUP):
    session.run(None, {input_name: dummy})

print(f'正式測試 {N_RUNS} 次...')
times = []
for i in range(N_RUNS):
    t0 = time.time()
    session.run(None, {input_name: dummy})
    dt = (time.time() - t0) * 1000
    times.append(dt)
    print(f'  第{i+1:2d}次: {dt:.1f}ms')

times = np.array(times)
print('\n' + '=' * 60)
print('結果總結')
print('=' * 60)
print(f'平均推論時間: {times.mean():.1f} ms')
print(f'中位數:      {np.median(times):.1f} ms')
print(f'最快:        {times.min():.1f} ms')
print(f'最慢:        {times.max():.1f} ms')
print()
print('參考：這台開發用筆電（RTX 3060筆電，強制CPU模式）測出來的數字約是 145ms/次')
print('（用同一個width05_yolov8.onnx模型），可以拿這個數字互相對照，看老筆電慢多少。')
print('=' * 60)
