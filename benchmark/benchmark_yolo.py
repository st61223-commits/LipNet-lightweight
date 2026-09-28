"""
YOLOv5 vs YOLOv8 推論速度對比測試
測試項目：每幀推論時間、FPS、GPU 記憶體用量
"""
import os, time, warnings, pathlib
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
pathlib.PosixPath = pathlib.WindowsPath

import cv2
import numpy as np
import torch
from ultralytics import YOLO

TEST_VIDEO   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6\video1.mpg'
YOLOV5_DIR   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5'
YOLOV5_PT    = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5\runs\v5s\weights\best.pt'
YOLOV8_PT    = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov8_lip\lip_detect\weights\best.pt'
WARMUP_FRAMES = 10   # 暖機幀數（不計入統計）
TEST_FRAMES   = 75   # 測試幀數

def extract_frames(video_path, n):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def benchmark(model_fn, frames, label):
    """跑 warmup 後測試，回傳時間列表（毫秒）"""
    print(f'\n測試 {label}...')

    # 暖機
    for f in frames[:WARMUP_FRAMES]:
        model_fn(f)

    # 正式測試
    times = []
    detected = 0
    for f in frames:
        t0 = time.perf_counter()
        result = model_fn(f)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        if result:
            detected += 1

    avg = sum(times) / len(times)
    mn  = min(times)
    mx  = max(times)
    fps = 1000 / avg
    det_rate = detected / len(frames) * 100

    print(f'  平均推論時間：{avg:.1f} ms')
    print(f'  最快 / 最慢：{mn:.1f} ms / {mx:.1f} ms')
    print(f'  推論 FPS：{fps:.1f}')
    print(f'  偵測率：{det_rate:.1f}% ({detected}/{len(frames)} 幀)')
    return times, fps, det_rate

# ── 讀取測試影片 ──
print(f'讀取測試影片：{TEST_VIDEO}')
frames_bgr = extract_frames(TEST_VIDEO, TEST_FRAMES + WARMUP_FRAMES)
frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
print(f'共 {len(frames_bgr)} 幀')

# ── 載入模型 ──
print('\n載入 YOLOv5...')
v5 = torch.hub.load(YOLOV5_DIR, 'custom', path=YOLOV5_PT, source='local', verbose=False)
v5.conf = 0.1

print('載入 YOLOv8...')
v8 = YOLO(YOLOV8_PT)

# ── 定義偵測函式 ──
def run_v5(frame_rgb):
    res = v5(frame_rgb)
    for det in res.pred:
        for *_, conf, cls in det:
            if int(cls) == 0 and conf >= 0.1:
                return True
    return False

def run_v8(frame_bgr):
    res = v8(frame_bgr, verbose=False, conf=0.1)
    for r in res:
        if len(r.boxes) > 0:
            return True
    return False

# ── GPU 記憶體（測試前）──
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# ── 正式測試 ──
t5, fps5, det5 = benchmark(run_v5, frames_rgb[:TEST_FRAMES + WARMUP_FRAMES], 'YOLOv5s')
if torch.cuda.is_available():
    mem5 = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
else:
    mem5 = 0

t8, fps8, det8 = benchmark(run_v8, frames_bgr[:TEST_FRAMES + WARMUP_FRAMES], 'YOLOv8n')
if torch.cuda.is_available():
    mem8 = torch.cuda.max_memory_allocated() / 1024**2
else:
    mem8 = 0

# ── 比較總表 ──
avg5 = sum(t5) / len(t5)
avg8 = sum(t8) / len(t8)
speedup = avg5 / avg8

print('\n' + '=' * 50)
print(f'{"比較總表":^50}')
print('=' * 50)
print(f'{"指標":<20} {"YOLOv5s":>12} {"YOLOv8n":>12}')
print('-' * 50)
print(f'{"平均推論時間":<20} {avg5:>10.1f}ms {avg8:>10.1f}ms')
print(f'{"推論 FPS":<20} {fps5:>11.1f} {fps8:>11.1f}')
print(f'{"偵測率":<20} {det5:>10.1f}% {det8:>10.1f}%')
if mem5 > 0:
    print(f'{"GPU 記憶體峰值":<20} {mem5:>9.1f}MB {mem8:>9.1f}MB')
print('-' * 50)
if speedup >= 1:
    print(f'YOLOv8n 比 YOLOv5s 快 {speedup:.2f} 倍（{(speedup-1)*100:.1f}% 提升）')
else:
    print(f'YOLOv8n 比 YOLOv5s 慢 {1/speedup:.2f} 倍')
print('=' * 50)
