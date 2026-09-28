"""
前處理腳本：為 s3, s4, s8 建立 .npy 快取
執行方式（WSL）：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe scripts/precache_s3_s4_s8.py
"""

import os
import sys
import glob
import pathlib
import cv2
import numpy as np
import torch

pathlib.PosixPath = pathlib.WindowsPath

YOLO_REPO   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5'
YOLO_WEIGHT = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5\runs\v5s\weights\best.pt'
DATA_ROOT   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data'

NEW_SPEAKERS = ['s3', 's4', 's8']

INPUT_SHAPE = (75, 46, 140)  # (frames, height, width)


def load_yolov5():
    print('載入 YOLOv5 模型...')
    model = torch.hub.load(YOLO_REPO, 'custom', path=YOLO_WEIGHT, source='local')
    print('YOLOv5 載入完成')
    return model


def process_video(video_path: str, yolov5s, input_shape=INPUT_SHAPE):
    project_name = os.path.basename(os.path.dirname(video_path))
    cache_dir = os.path.join(DATA_ROOT, f'{project_name}_cached')
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = os.path.join(cache_dir, f'{file_name}.npy')

    if os.path.exists(cache_path):
        return 'skip'

    cap = cv2.VideoCapture(video_path)
    frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        detections = yolov5s(frame)
        for det in detections.pred:
            for *xyxy, conf, cls in det:
                if int(cls) == 0 and conf >= 0.5:
                    x1, y1, x2, y2 = map(int, xyxy)
                    lip = frame[y1:y2, x1:x2]
                    lip = cv2.resize(lip, (input_shape[2], input_shape[1]))
                    lip = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
                    frames.append(lip.astype(np.float32))
    cap.release()

    T = input_shape[0]
    frames = frames[:T]
    if not frames:
        frames = [np.zeros((input_shape[1], input_shape[2]), dtype=np.float32)]
    while len(frames) < T:
        frames.append(frames[-1])

    frames = np.array(frames)           # (75, 46, 140)
    frames = np.expand_dims(frames, -1) # (75, 46, 140, 1)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, frames)
    return 'done'


def main():
    yolov5s = load_yolov5()

    all_videos = []
    for speaker in NEW_SPEAKERS:
        pattern = os.path.join(DATA_ROOT, speaker, '*.mpg')
        videos = glob.glob(pattern)
        all_videos.extend(videos)
        print(f'  {speaker}: {len(videos)} 支影片')

    total = len(all_videos)
    print(f'\n共 {total} 支影片需要前處理\n')

    done_count = 0
    skip_count = 0
    error_count = 0

    for idx, video_path in enumerate(all_videos):
        try:
            result = process_video(video_path, yolov5s)
            if result == 'skip':
                skip_count += 1
            else:
                done_count += 1
        except Exception as e:
            error_count += 1
            print(f'  [ERROR] {os.path.basename(video_path)}: {e}')

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f'進度：{idx + 1}/{total}  （新增={done_count}, 跳過={skip_count}, 錯誤={error_count}）')

    print(f'\n[完成] 新增={done_count}, 跳過={skip_count}, 錯誤={error_count}')


if __name__ == '__main__':
    main()
