"""
比照 regenerate_cache_yolov8.py，改用新微調好的 YOLOv10 嘴唇偵測器
（yolov10_lip/lip_detect/weights/best.pt）重新產生訓練快取，供之後
train_grid_multi_width05_yolov10.py 使用，評估 YOLOv8→YOLOv10 是否值得換。

輸出到新的快取資料夾（後綴 _cached_yolov10，不覆蓋既有 YOLOv8 版本的快取），
可以之後A/B比較，也保留隨時退回的空間。

執行方式：
  python3 scripts/regenerate_cache_yolov10.py --dry-run
  python3 scripts/regenerate_cache_yolov10.py
  python3 scripts/regenerate_cache_yolov10.py --only s6
"""
import argparse
import os
import sys
import glob
import shutil
import time
import pathlib
import warnings

pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)

DEFAULT_MIN_FREE_GB = 15

SPEAKER_DIRS = [
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's13',
    's99_1', 's99_6', 's99_6_new', 's99_7', 's99_7_new', 's99_8', 's34_3',
]

CACHE_SUFFIX = '_cached_yolov10'
CONF_THRESHOLD = 0.5
input_shape = (75, 46, 140, 1)


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='*', default=None, help='只處理指定的說話者資料夾名稱')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--min-free-gb', type=float, default=DEFAULT_MIN_FREE_GB,
                         help=f'磁碟防呆門檻(GB)，預設{DEFAULT_MIN_FREE_GB}；只剩少量說話者要補時可調低')
    args = parser.parse_args()
    min_free_gb = args.min_free_gb

    speakers = [s for s in SPEAKER_DIRS if (args.only is None or s in args.only)]

    total_videos = 0
    for spk in speakers:
        pattern = os.path.join('data', spk, '*.mpg')
        total_videos += len(glob.glob(pattern))

    est_gb = total_videos * (75 * 46 * 140 * 4) / 1e9
    print(f'預計處理 {len(speakers)} 位說話者，共 {total_videos} 支影片')
    print(f'預估新增快取空間需求: 約 {est_gb:.1f} GB')
    print(f'目前剩餘空間: {free_gb(BASE_DIR):.1f} GB')

    if args.dry_run:
        print('\n[dry-run模式，未實際執行任何偵測/寫入]')
        return

    if free_gb(BASE_DIR) < min_free_gb:
        print(f'\n[中止] 剩餘空間({free_gb(BASE_DIR):.1f}GB) 低於安全門檻({min_free_gb}GB)。')
        sys.exit(1)

    import cv2
    import numpy as np
    from ultralytics import YOLO

    print('\n載入 YOLOv10（新微調的嘴唇偵測器）...')
    yolo_model = YOLO(r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov10_lip\lip_detect\weights\best.pt')
    print('載入完成！\n')

    t_start = time.time()
    aborted = False
    for spk in speakers:
        if free_gb(BASE_DIR) < min_free_gb:
            print(f'[中止] 處理到 {spk} 時空間不足，停止')
            aborted = True
            break

        cache_dir = os.path.join('data', f'{spk}{CACHE_SUFFIX}')
        os.makedirs(cache_dir, exist_ok=True)
        videos = sorted(glob.glob(os.path.join('data', spk, '*.mpg')))
        print(f'=== {spk}（{len(videos)}支影片）===', flush=True)

        for vi, video_path in enumerate(videos, 1):
            file_name = os.path.splitext(os.path.basename(video_path))[0]
            cache_path = os.path.join(cache_dir, f'{file_name}.npy')
            if os.path.exists(cache_path):
                continue

            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = yolo_model(frame, verbose=False)
                for r in results:
                    matched = False
                    for box in r.boxes:
                        if float(box.conf[0]) >= CONF_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            lip = frame[y1:y2, x1:x2]
                            if lip.size == 0:
                                continue
                            lip = cv2.resize(lip, (input_shape[2], input_shape[1]))
                            lip = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
                            frames.append(lip.astype(np.float32))
                            matched = True
                            break
                    if matched:
                        break
            cap.release()

            T = 75
            frames = frames[:T]
            if not frames:
                frames = [np.zeros((input_shape[1], input_shape[2]), dtype=np.float32)]
            while len(frames) < T:
                frames.append(frames[-1])
            arr = np.expand_dims(np.array(frames), -1)
            np.save(cache_path, arr)

            if vi % 100 == 0:
                print(f'  {spk}: {vi}/{len(videos)}（剩餘空間 {free_gb(BASE_DIR):.1f}GB）', flush=True)

        print(f'  {spk} 完成', flush=True)

    if aborted:
        print(f'\n⚠️ 因空間不足中途中止，並未處理完全部指定的說話者！耗時 {(time.time()-t_start)/60:.1f} 分鐘')
        print(f'剩餘空間: {free_gb(BASE_DIR):.1f} GB')
    else:
        print(f'\n✅ 全部完成，總耗時 {(time.time()-t_start)/60:.1f} 分鐘')
        print(f'剩餘空間: {free_gb(BASE_DIR):.1f} GB')
        print('\n下一步：train_grid_multi_width05_yolov10.py 讀 *_cached_yolov10 快取重新訓練。')


if __name__ == '__main__':
    main()
