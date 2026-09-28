"""
下載 GRID 語料庫其餘缺少的說話者（影片+對齊檔），擴充訓練說話者多樣性。

背景：實驗五(2026-08-27)結論指出，s99_3+5/s9這類全新使用者辨識率0%的
真正瓶頸是訓練說話者多樣性太窄(目前只用9位)，需要擴大訓練說話者數量才
能解決。實驗六(2026-09-02)用五種個人化校準技術都證實補不回這個缺口。
GRID語料庫官方共有 s1~s34（缺s21，官方本來就沒有這個編號，已用curl -I
確認回404）共33位說話者，目前本機只有 s1~s9+s13 共10位，缺23位。

**⚠️ 執行前必讀：這支腳本會下載約9~10GB資料，執行前必須先確認磁碟空間
充足**（2026-09-02發現C槽只剩5.8GB，需要使用者先執行
`scripts\\cleanup_old_checkpoints.ps1 -Confirm` 清出約27GB空間後才能跑這支）。
腳本本身有磁碟空間防呆機制(見下方MIN_FREE_GB)，空間不足會直接中止不會
硬幹到把C槽塞爆。

資料來源（官方 University of Sheffield GRID corpus 網站，已用curl -I驗證可連線）：
  影片: https://spandh.dcs.shef.ac.uk/gridcorpus/s{N}/video/s{N}.mpg_vcd.zip (~410MB/人)
  對齊: https://spandh.dcs.shef.ac.uk/gridcorpus/s{N}/align/s{N}.tar (~1MB/人)

做法：每位說話者依序下載→解壓→歸位到 data/s{N}/ 和 data/alignments/s{N}/
→ 刪除暫存壓縮檔（省空間），每位處理完都重新檢查剩餘空間。**只下載影片
+對齊，不下載音訊**(唇語辨識用不到)。下載完的.mpg還需要另外跑嘴唇偵測
產生 _cached 快取才能拿去訓練，這支腳本不含偵測步驟(沿用現有前處理流程)。

執行方式（WSL或Windows皆可，純Python標準庫+requests，不需要py39/tf215環境）：
  python3 scripts/download_missing_grid_speakers.py           # 下載全部缺少的
  python3 scripts/download_missing_grid_speakers.py --only 10 11 12   # 只下載指定幾位（先小量測試用）
  python3 scripts/download_missing_grid_speakers.py --dry-run  # 只列出缺哪些、預估空間，不下載
"""
import argparse
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet' if os.name == 'nt' else '/mnt/c/Users/Tno/OneDrive/Lipnet_nchu/LipNet'
DATA_DIR = os.path.join(BASE_DIR, 'data')
ALIGN_DIR = os.path.join(DATA_DIR, 'alignments')
TMP_DIR = os.path.join(BASE_DIR, '_grid_download_tmp')

ALL_SPEAKERS = [n for n in range(1, 35) if n != 21]  # GRID官方就沒有s21(已用curl -I確認404)
VIDEO_URL = 'https://spandh.dcs.shef.ac.uk/gridcorpus/s{n}/video/s{n}.mpg_vcd.zip'
ALIGN_URL = 'https://spandh.dcs.shef.ac.uk/gridcorpus/s{n}/align/s{n}.tar'

EST_MB_PER_SPEAKER = 410  # 影片zip實測約409MB，抓410當估計值
MIN_FREE_GB = 15  # 磁碟防呆門檻：剩餘空間低於這個數字就直接中止，不繼續下載


def get_missing_speakers():
    existing = set()
    for name in os.listdir(DATA_DIR):
        if name.startswith('s') and name[1:].isdigit():
            existing.add(int(name[1:]))
    return [n for n in ALL_SPEAKERS if n not in existing]


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def download(url, dest_path, label):
    print(f'    下載 {label} ...', flush=True)
    t0 = time.time()
    urllib.request.urlretrieve(url, dest_path)
    size_mb = os.path.getsize(dest_path) / 1e6
    dt = time.time() - t0
    print(f'    完成：{size_mb:.1f}MB，耗時{dt:.1f}秒', flush=True)


def process_speaker(n):
    spk = f's{n}'
    print(f'\n=== {spk} ===', flush=True)

    os.makedirs(TMP_DIR, exist_ok=True)
    align_tar = os.path.join(TMP_DIR, f'{spk}_align.tar')
    video_zip = os.path.join(TMP_DIR, f'{spk}_video.zip')

    # 對齊檔（小，先下載）
    download(ALIGN_URL.format(n=n), align_tar, f'{spk} 對齊檔')
    align_extract_tmp = os.path.join(TMP_DIR, f'{spk}_align_extract')
    with tarfile.open(align_tar) as tf:
        tf.extractall(align_extract_tmp)
    src_align_dir = os.path.join(align_extract_tmp, 'align')
    dest_align_dir = os.path.join(ALIGN_DIR, spk)
    if os.path.exists(src_align_dir):
        shutil.move(src_align_dir, dest_align_dir)
    os.remove(align_tar)
    shutil.rmtree(align_extract_tmp, ignore_errors=True)
    print(f'    對齊檔已歸位: {dest_align_dir}', flush=True)

    # 影片（大，後下載，下載前再檢查一次空間）
    if free_gb(BASE_DIR) < MIN_FREE_GB:
        print(f'    [中止] 剩餘空間低於 {MIN_FREE_GB}GB，停止下載影片', flush=True)
        return False
    download(VIDEO_URL.format(n=n), video_zip, f'{spk} 影片')
    video_extract_tmp = os.path.join(TMP_DIR, f'{spk}_video_extract')
    with zipfile.ZipFile(video_zip) as zf:
        zf.extractall(video_extract_tmp)
    # zip 內部結構：通常是 s{n}/*.mpg 或直接 *.mpg，兩種都處理
    dest_video_dir = os.path.join(DATA_DIR, spk)
    os.makedirs(dest_video_dir, exist_ok=True)
    moved = 0
    for root, dirs, files in os.walk(video_extract_tmp):
        for f in files:
            if f.endswith('.mpg'):
                shutil.move(os.path.join(root, f), os.path.join(dest_video_dir, f))
                moved += 1
    os.remove(video_zip)
    shutil.rmtree(video_extract_tmp, ignore_errors=True)
    print(f'    影片已歸位: {dest_video_dir}（{moved} 支.mpg）', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', type=int, nargs='*', default=None, help='只處理指定的說話者編號（測試用）')
    parser.add_argument('--dry-run', action='store_true', help='只列出缺哪些說話者+預估空間，不實際下載')
    args = parser.parse_args()

    missing = get_missing_speakers()
    if args.only:
        missing = [n for n in missing if n in args.only]

    est_total_gb = len(missing) * EST_MB_PER_SPEAKER / 1000
    print(f'缺少的說話者（共{len(missing)}位）: {missing}')
    print(f'預估下載總量: 約 {est_total_gb:.1f} GB')
    print(f'目前 {BASE_DIR} 所在磁碟剩餘空間: {free_gb(BASE_DIR):.1f} GB')

    if args.dry_run:
        print('\n[dry-run模式，未實際下載任何檔案]')
        return

    if free_gb(BASE_DIR) < MIN_FREE_GB:
        print(f'\n[中止] 剩餘空間({free_gb(BASE_DIR):.1f}GB) 低於安全門檻({MIN_FREE_GB}GB)，'
              f'請先執行 scripts\\cleanup_old_checkpoints.ps1 -Confirm 清出空間後再重新執行本腳本。')
        sys.exit(1)

    ok, fail = 0, 0
    for n in missing:
        try:
            if process_speaker(n):
                ok += 1
            else:
                fail += 1
                break  # 空間不夠中止時，不繼續嘗試下一位
        except Exception as e:
            print(f'  [失敗] s{n}: {type(e).__name__}: {e}', flush=True)
            fail += 1

    print(f'\n完成：成功 {ok} 位，失敗/中止 {fail} 位')
    print(f'目前剩餘空間: {free_gb(BASE_DIR):.1f} GB')
    print('\n提醒：下載完的影片還需要跑現有的嘴唇偵測前處理產生 _cached 快取才能用於訓練，'
          '這支腳本只負責下載歸位原始資料。')

    shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == '__main__':
    main()
