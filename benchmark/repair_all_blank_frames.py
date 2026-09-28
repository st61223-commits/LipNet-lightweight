"""
全專案等級修復：掃過所有說話者的 .npy 快取資料夾，找出全黑（YOLOv5信心度<0.5被誤判成
「沒偵測到」而整支補零）的檔案，用已驗證有效的修正邏輯（不論信心度多少，只要有偵測到
就採用最高信心度的框）重新處理對應的原始影片，覆蓋掉壞的快取檔案。

背景：這個 bug 原本只在 s99_3/s99_5（真正沒訓練過的新說話者）被發現，但今天在幫 s9
建快取時意外發現，這其實是全專案性的問題——連已經被拿去訓練的 s2 都有高達 82.7% 的
快取是全黑的！詳見 實驗記錄.md 實驗五。

執行方式（WSL）：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe repair_all_blank_frames.py
"""

import os
import glob
import pathlib
import cv2
import numpy as np
import torch

pathlib.PosixPath = pathlib.WindowsPath

YOLO_REPO   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5'
YOLO_WEIGHT = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5\runs\v5s\weights\best.pt'
DATA_ROOT   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data'
INPUT_SHAPE = (75, 46, 140)

# 全專案所有已知有快取資料的說話者資料夾（跟 data/ 底下的 *_cached 對應）
SPEAKERS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's13', 's34_3',
            's99_1', 's99_6', 's99_7', 's99_8', 's99_3', 's99_5']


def load_yolov5():
    print('載入 YOLOv5 模型...', flush=True)
    model = torch.hub.load(YOLO_REPO, 'custom', path=YOLO_WEIGHT, source='local')
    # 關鍵：YOLOv5 AutoShape 物件本身有內建信心度門檻，預設0.25，會在我們自己的
    # 「不論信心度多少、取最高的框」邏輯看到結果之前就先濾掉低於0.25的偵測。
    # 上一輪漏加這行，導致s9這種信心度普遍在0.05~0.25之間的影片完全沒被修好。
    model.conf = 0.001
    print('YOLOv5 載入完成', flush=True)
    return model


def find_blank_files(speaker):
    cache_dir = os.path.join(DATA_ROOT, f'{speaker}_cached')
    files = sorted(glob.glob(os.path.join(cache_dir, '*.npy')))
    blanks = []
    for f in files:
        arr = np.load(f)
        if arr.std() < 1e-3:
            blanks.append(os.path.splitext(os.path.basename(f))[0])
    return blanks


def repair_one(speaker, stem, yolov5s):
    video_path = os.path.join(DATA_ROOT, speaker, stem + '.mpg')
    cache_path = os.path.join(DATA_ROOT, f'{speaker}_cached', stem + '.npy')
    if not os.path.exists(video_path):
        return 'no_source'

    cap = cv2.VideoCapture(video_path)
    frames = []
    no_detect = 0
    total = 0
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        total += 1
        detections = yolov5s(frame)
        best_box, best_conf = None, -1.0
        for det in detections.pred:
            for *xyxy, conf, cls in det:
                if int(cls) == 0 and float(conf) > best_conf:
                    best_conf = float(conf)
                    best_box = [int(v) for v in xyxy]
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            x1, y1 = max(0, x1), max(0, y1)
            lip = frame[y1:y2, x1:x2]
            if lip.size == 0:
                no_detect += 1
                frames.append(frames[-1] if frames else np.zeros((INPUT_SHAPE[1], INPUT_SHAPE[2]), dtype=np.float32))
                continue
            lip = cv2.resize(lip, (INPUT_SHAPE[2], INPUT_SHAPE[1]))
            lip = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frames.append(lip)
        else:
            no_detect += 1
            frames.append(frames[-1] if frames else np.zeros((INPUT_SHAPE[1], INPUT_SHAPE[2]), dtype=np.float32))
    cap.release()

    T = INPUT_SHAPE[0]
    frames = frames[:T]
    while len(frames) < T:
        frames.append(frames[-1] if frames else np.zeros((INPUT_SHAPE[1], INPUT_SHAPE[2]), dtype=np.float32))

    arr = np.array(frames)
    arr = np.expand_dims(arr, -1)
    np.save(cache_path, arr)
    still_blank = arr.std() < 1e-3
    return 'still_blank' if still_blank else f'fixed(no_detect={no_detect}/{total})'


def main():
    yolov5s = load_yolov5()

    print('\n第一步：掃描所有說話者，找出全黑檔案...', flush=True)
    to_repair = {}
    total_blank = 0
    for spk in SPEAKERS:
        cache_dir = os.path.join(DATA_ROOT, f'{spk}_cached')
        if not os.path.isdir(cache_dir):
            continue
        blanks = find_blank_files(spk)
        if blanks:
            to_repair[spk] = blanks
            total_blank += len(blanks)
        print(f'  {spk}: 全黑 {len(blanks)} 筆', flush=True)
    print(f'\n共需修復 {total_blank} 筆\n', flush=True)

    print('第二步：逐一修復...', flush=True)
    fixed_count = 0
    still_blank_count = 0
    no_source_count = 0
    for spk, stems in to_repair.items():
        print(f'\n--- 修復 {spk}（共{len(stems)}筆）---', flush=True)
        for i, stem in enumerate(stems):
            result = repair_one(spk, stem, yolov5s)
            if result == 'no_source':
                no_source_count += 1
                print(f'  [{i+1}/{len(stems)}] {stem}: 找不到原始影片，跳過', flush=True)
            elif result == 'still_blank':
                still_blank_count += 1
                print(f'  [{i+1}/{len(stems)}] {stem}: 修復後仍是全黑（真的偵測不到嘴唇）', flush=True)
            else:
                fixed_count += 1
                if (i + 1) % 20 == 0:
                    print(f'  [{i+1}/{len(stems)}] 進度中...（{result}）', flush=True)

    print('\n' + '=' * 60, flush=True)
    print(f'修復完成：成功修好 {fixed_count} 筆、仍全黑 {still_blank_count} 筆、找不到來源 {no_source_count} 筆', flush=True)

    print('\n第三步：複查全部說話者的全黑比例...', flush=True)
    for spk in SPEAKERS:
        cache_dir = os.path.join(DATA_ROOT, f'{spk}_cached')
        if not os.path.isdir(cache_dir):
            continue
        blanks = find_blank_files(spk)
        files = glob.glob(os.path.join(cache_dir, '*.npy'))
        pct = len(blanks) / len(files) * 100 if files else 0
        print(f'  {spk}: 全黑 {len(blanks)}/{len(files)} = {pct:.1f}%', flush=True)


if __name__ == '__main__':
    main()
