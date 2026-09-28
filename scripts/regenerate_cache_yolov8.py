"""
【方案2「根本修復」的準備腳本，尚未執行】用YOLOv8重新產生訓練快取，
解決「即時管線用YOLOv8、訓練資料用YOLOv5，兩者框出的嘴唇區域不一致」問題
（詳見 project_lipnet.md 2026-09-02凌晨的重大發現）。

背景：偵測器對照實驗證實，即使把即時管線換回YOLOv5，也無法完美復現訓練快取
的結果（懷疑YOLOv5權重可能已經漂移/更新過，難以精確重現當初的環境）。
唯一保證訓練/部署一致的做法，是讓訓練資料改用「現在部署中的同一顆YOLOv8
權重」重新產生，這樣訓練跟部署天生就是同一套偵測邏輯。

做法：比照 train_grid_multi.py 的 load_video 邏輯（每幀獨立偵測、無平滑、
無人臉偵測），只把YOLOv5換成YOLOv8（跟LipNetConnect1.py用同一顆權重：
yolov8_lip/lip_detect/weights/best.pt），confidence沿用訓練時的門檻0.5
（不是即時管線的0.1，因為這是產生訓練資料、要跟訓練慣例一致，不是即時辨識）。
輸出到新的快取資料夾（後綴 _cached_yolov8，不覆蓋既有YOLOv5版本的快取），
可以之後A/B比較，也保留隨時退回的空間。

**⚠️ 執行前必讀**：
1. 目前C槽只剩5.8GB，這個規模的快取(16位說話者、共約7900+支影片)預估需要
   15~20GB空間，執行前必須先跑 scripts\\cleanup_old_checkpoints.ps1 -Confirm
   清出空間（腳本內建磁碟防呆，空間不夠會自動中止）
2. 這只是「產生快取」，產生完之後還需要另外寫暖啟動訓練腳本（比照
   train_grid_multi_width05_datafix.py的模式）用新快取重新訓練，才能真正
   讓模型跟YOLOv8的裁切風格對齊，這支腳本不含訓練步驟
3. 全部16位說話者、每人耗時視影片數量而定，保守估計數小時等級（YOLOv8在GPU上
   每幀約20-50ms，7900支影片*75幀，實際會比這個粗估快，因為很多幀共用同一支
   影片的推論成本，但仍是「小時」量級的工作，適合排在使用者確認方向後於背景執行）

執行方式（WSL或Windows）：
  python3 scripts/regenerate_cache_yolov8.py --dry-run     # 先看看預估需要空間/時間
  python3 scripts/regenerate_cache_yolov8.py                # 正式執行（會先做磁碟防呆檢查）
  python3 scripts/regenerate_cache_yolov8.py --only s6      # 只處理指定說話者（先小量測試用）
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

DEFAULT_MIN_FREE_GB = 15  # 磁碟防呆門檻預設值，比照 download_missing_grid_speakers.py 的做法
# 2026-09-02踩坑：第一次完整執行時跑到第15位(s99_8)空間剩14GB、低於15GB門檻直接
# 中止，留下s99_8+s34_3(共299支影片、不到1GB需求)沒處理，而且結尾的「全部完成」
# 訊息沒有區分「真的處理完全部」還是「中途中止」，容易誤判已經跑完。
# 加 --min-free-gb 讓之後這種「只剩一點點要補」的情境可以彈性調低門檻。

# 跟 train_grid_multi_width05_datafix.py 的 all_patterns 同一套16位說話者
SPEAKER_DIRS = [
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's13',
    's99_1', 's99_6', 's99_6_new', 's99_7', 's99_7_new', 's99_8', 's34_3',
]

CACHE_SUFFIX = '_cached_yolov8'
CONF_THRESHOLD = 0.5  # 沿用訓練時的門檻（不是即時管線的0.1）
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

    est_gb = total_videos * (75 * 46 * 140 * 4) / 1e9  # float32快取，每支約1.8MB
    print(f'預計處理 {len(speakers)} 位說話者，共 {total_videos} 支影片')
    print(f'預估新增快取空間需求: 約 {est_gb:.1f} GB')
    print(f'目前剩餘空間: {free_gb(BASE_DIR):.1f} GB')

    if args.dry_run:
        print('\n[dry-run模式，未實際執行任何偵測/寫入]')
        return

    if free_gb(BASE_DIR) < min_free_gb:
        print(f'\n[中止] 剩餘空間({free_gb(BASE_DIR):.1f}GB) 低於安全門檻({min_free_gb}GB)，'
              f'請先執行 scripts\\cleanup_old_checkpoints.ps1 -Confirm 清出空間，'
              f'或用 --min-free-gb 調低門檻（如果確定剩下要處理的資料量很小）後再重新執行。')
        sys.exit(1)

    # 真正執行時才載入這些重的套件，dry-run不需要等它們載入
    import cv2
    import numpy as np
    from ultralytics import YOLO

    print('\n載入 YOLOv8（跟 LipNetConnect1.py 部署用同一顆權重）...')
    yolo_model = YOLO(r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov8_lip\lip_detect\weights\best.pt')
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
                continue  # 已處理過，支援中斷續跑

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
        print('請清理空間或用 --min-free-gb 調低門檻後，針對還沒完成的說話者用 --only 補跑。')
    else:
        print(f'\n✅ 全部完成，總耗時 {(time.time()-t_start)/60:.1f} 分鐘')
        print(f'剩餘空間: {free_gb(BASE_DIR):.1f} GB')
        print('\n下一步：需另外寫暖啟動訓練腳本，改讀 *_cached_yolov8 快取重新訓練，'
              '才能讓模型真正對齊YOLOv8的裁切風格。')


if __name__ == '__main__':
    main()
