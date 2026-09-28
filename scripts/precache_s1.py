import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

YOLO_MODEL = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov8_lip\lip_detect\weights\best.pt'
INPUT_SHAPE = (75, 46, 140)
CONF_THRESH = 0.5

yolo = YOLO(YOLO_MODEL)

def process_video(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo(frame, verbose=False)
        found = False
        for box in results[0].boxes:
            if box.conf[0] >= CONF_THRESH:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                lip = frame[y1:y2, x1:x2]
                if lip.size == 0:
                    continue
                lip = cv2.resize(lip, (INPUT_SHAPE[2], INPUT_SHAPE[1]))
                lip = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
                frames.append(lip.astype(np.float32))
                found = True
                break
        if not found:
            # 偵測不到嘴唇時補零幀
            frames.append(np.zeros((INPUT_SHAPE[1], INPUT_SHAPE[2]), dtype=np.float32))

    cap.release()

    frames = frames[:INPUT_SHAPE[0]]
    while len(frames) < INPUT_SHAPE[0]:
        frames.append(frames[-1] if frames else np.zeros((INPUT_SHAPE[1], INPUT_SHAPE[2]), dtype=np.float32))

    arr = np.array(frames)           # (75, 46, 140)
    arr = np.expand_dims(arr, -1)    # (75, 46, 140, 1)
    return arr


videos = glob.glob(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s1\*.mpg')
cache_dir = os.path.join('data', 's1_cached')
os.makedirs(cache_dir, exist_ok=True)

print(f'共 {len(videos)} 支影片，開始建立快取 → {cache_dir}')

done = 0
skip = 0
fail = 0

for video_path in videos:
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = os.path.join(cache_dir, f'{file_name}.npy')

    if os.path.exists(cache_path):
        skip += 1
        continue

    try:
        arr = process_video(video_path)
        np.save(cache_path, arr)
        done += 1
    except Exception as e:
        print(f'[FAIL] {file_name}: {e}')
        fail += 1

    if (done + skip) % 100 == 0:
        print(f'  進度：{done + skip}/{len(videos)}（新建 {done}，跳過 {skip}，失敗 {fail}）')

print(f'\n[完成] 新建 {done}，跳過 {skip}，失敗 {fail}')
print(f'快取位置：{os.path.abspath(cache_dir)}')
