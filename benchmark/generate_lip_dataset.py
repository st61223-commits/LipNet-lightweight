"""
第一步：從現有影片自動產生 YOLOv8 訓練資料
用 YOLOv5 best.pt 跑每支影片，把偵測到的嘴唇框存成 YOLO 格式標記
"""
import os, sys, random, warnings
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import torch

# ── 設定 ──
YOLOV5_DIR   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5'
YOLOV5_PT    = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5\runs\v5s\weights\best.pt'
OUT_DIR      = r'C:\Users\Tno\OneDrive\Lipnet_nchu\lip_dataset'
CONF_THRESH  = 0.6    # 只保留信心度 >= 0.6 的偵測
FRAMES_PER_VIDEO = 5  # 每支影片取幾幀（均勻取樣）
TRAIN_RATIO  = 0.85   # 85% 訓練，15% 驗證

# 使用的資料夾（各說話者都取樣，增加多樣性）
VIDEO_DIRS = [
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6_new',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7_new',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_8',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_1',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s34_3',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_3',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_3_new',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_5',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_5_new',
]

# ── 建立輸出目錄 ──
for split in ['train', 'val']:
    os.makedirs(os.path.join(OUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'labels', split), exist_ok=True)

# ── 載入 YOLOv5 ──
print('載入 YOLOv5...')
model = torch.hub.load(YOLOV5_DIR, 'custom', path=YOLOV5_PT, source='local', verbose=False)
model.conf = CONF_THRESH
print('YOLOv5 載入完成\n')

# ── 收集所有影片路徑 ──
all_videos = []
for d in VIDEO_DIRS:
    if not os.path.exists(d):
        continue
    vids = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.mpg')]
    all_videos.extend(vids)

random.seed(42)
random.shuffle(all_videos)
print(f'共找到 {len(all_videos)} 支影片，每支取 {FRAMES_PER_VIDEO} 幀')
print(f'預計最多產生 {len(all_videos) * FRAMES_PER_VIDEO} 張圖片\n')

# ── 逐影片處理 ──
saved = 0
skipped = 0
img_paths = []

for idx, video_path in enumerate(all_videos):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        continue

    # 均勻取樣幀位置
    sample_positions = np.linspace(5, max(total_frames - 5, 5), FRAMES_PER_VIDEO, dtype=int)

    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue

        # 用 YOLOv5 偵測
        results = model(frame)
        preds = results.pred[0]

        # 找信心度最高的 class 0（嘴唇）
        best_box = None
        best_conf = 0
        for *xyxy, conf, cls in preds:
            if int(cls) == 0 and float(conf) > best_conf:
                best_conf = float(conf)
                best_box = [float(x) for x in xyxy]

        if best_box is None:
            skipped += 1
            continue

        # 轉成 YOLO 格式（正規化的中心點 + 寬高）
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = best_box
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        # 決定 train 或 val
        split = 'train' if random.random() < TRAIN_RATIO else 'val'

        # 存圖片
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        img_name = f'{video_name}_f{pos:04d}.jpg'
        img_out = os.path.join(OUT_DIR, 'images', split, img_name)
        cv2.imwrite(img_out, frame)

        # 存標記
        lbl_out = os.path.join(OUT_DIR, 'labels', split, img_name.replace('.jpg', '.txt'))
        with open(lbl_out, 'w') as f:
            f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')

        saved += 1
        img_paths.append(img_out)

    cap.release()

    if (idx + 1) % 100 == 0:
        print(f'  進度：{idx+1}/{len(all_videos)} 支影片，已存 {saved} 張，跳過 {skipped} 張')

print(f'\n完成！共存 {saved} 張圖片（跳過 {skipped} 張無偵測）')

# ── 產生 dataset.yaml ──
yaml_path = os.path.join(OUT_DIR, 'dataset.yaml')
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(f"""path: {OUT_DIR}
train: images/train
val: images/val

nc: 1
names:
  0: lip
""")
print(f'已產生 dataset.yaml：{yaml_path}')
print('\n下一步：執行 train_yolov8.py 開始訓練')
