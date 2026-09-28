"""
前處理腳本：為全新、完全沒訓練過的 GRID 說話者 s9 建立 .npy 快取
目的：拿現役 F_Width0.5 模型（不重新訓練）測試「情況A」——
同樣是 GRID 攝影棚環境，換一個完全沒看過的人，正確率如何。

比照 precache_new_speakers.py 的邏輯，但修正了信心度門檻的舊 bug：
原本「信心度<0.5就整支影片補零」的做法，會把「其實有偵測到、只是沒那麼有信心」
的影片誤判成完全沒偵測到（這個 bug 是之前排查 s99_3/5 全黑快取檔案時發現的，
詳見 實驗記錄.md 實驗五）。這裡改成「不論信心度多少，只要有偵測到就採用
最高信心度的框」，避免重蹈覆轍。

執行方式（WSL）：
  /mnt/c/Users/Tno/miniconda3/envs/py39/python.exe precache_s9.py
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

SPEAKER = 's9'
INPUT_SHAPE = (75, 46, 140)  # (frames, height, width)


def load_yolov5():
    print('載入 YOLOv5 模型...')
    model = torch.hub.load(YOLO_REPO, 'custom', path=YOLO_WEIGHT, source='local')
    # YOLOv5 AutoShape 物件內建信心度門檻預設0.25，要關到很低，才能讓後面
    # 「不論信心度多少、取最高的框」的邏輯真的看到所有候選框（教訓見實驗記錄.md）。
    model.conf = 0.001
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
    no_detect = 0
    total_read = 0

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        total_read += 1
        detections = yolov5s(frame)
        best_box, best_conf = None, -1
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
                lip_gray = frames[-1] if frames else np.zeros((input_shape[1], input_shape[2]), dtype=np.float32)
            else:
                lip = cv2.resize(lip, (input_shape[2], input_shape[1]))
                lip_gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frames.append(lip_gray)
        else:
            no_detect += 1
            frames.append(frames[-1] if frames else np.zeros((input_shape[1], input_shape[2]), dtype=np.float32))
    cap.release()

    T = input_shape[0]
    frames = frames[:T]
    if not frames:
        frames = [np.zeros((input_shape[1], input_shape[2]), dtype=np.float32)]
    while len(frames) < T:
        frames.append(frames[-1])

    frames = np.array(frames)
    frames = np.expand_dims(frames, -1)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, frames)
    if no_detect > 0:
        print(f'    {file_name}: {no_detect}/{total_read} 幀沒偵測到（已用前一幀補），非全黑')
    return 'done'


def main():
    yolov5s = load_yolov5()

    pattern = os.path.join(DATA_ROOT, SPEAKER, '*.mpg')
    videos = glob.glob(pattern)
    print(f'{SPEAKER}: 共 {len(videos)} 支影片需要前處理\n')

    done_count = skip_count = error_count = 0
    for idx, video_path in enumerate(videos):
        try:
            result = process_video(video_path, yolov5s)
            if result == 'skip':
                skip_count += 1
            else:
                done_count += 1
            if (idx + 1) % 100 == 0:
                print(f'  進度：{idx+1}/{len(videos)}（完成{done_count}，跳過{skip_count}）')
        except Exception as e:
            error_count += 1
            print(f'  錯誤 {video_path}: {e}')

    print(f'\n完成！done={done_count} skip={skip_count} error={error_count}')


if __name__ == '__main__':
    main()
