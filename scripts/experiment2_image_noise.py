"""
實驗二：影像雜訊對 LipNet 準確率的影響 + DSP 低通濾波補救
- 對已 cache 的 .npy 幀加入 Gaussian noise（模擬無線視訊傳輸雜訊）
- 比較：無雜訊 / 有雜訊 / 有雜訊+LPF 三種條件下的辨識準確率
- 對應 DSP 概念：AWGN → LPF → 訊號還原
"""

import os
import sys
import pathlib
import warnings
import numpy as np
import cv2
import tensorflow as tf
from scipy.ndimage import gaussian_filter

sys.path.insert(0, r'C:\Users\Tno\claude-code')
from vocab_correction import correct_sentence

pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore")

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model

BASE_DIR    = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_v8.h5')
LOG_PATH    = os.path.join(BASE_DIR, 'logs', 'experiment2_image_noise_log.txt')

os.chdir(BASE_DIR)

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def CTCLoss(y_true, y_pred):
    batch_len    = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def load_alignments(path: str) -> str:
    """與 benchmark 相同邏輯：過濾 sil/sp 及單字符，OOV 字符消失"""
    vocab_set = set("abcdefghijklmnopqrstuvwxyz'?!123456789 ")
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        word = parts[2]
        if word not in ('sil', 'sp') and len(word) > 1:
            tokens.extend([' ', word])
    raw = ''.join(tokens[1:])
    filtered = ''.join(c for c in raw if c in vocab_set)
    return ' '.join(filtered.split())


def add_gaussian_noise(frames: np.ndarray, sigma: float) -> np.ndarray:
    """在 normalized frame 上加 Gaussian noise（sigma 為 normalized 空間的標準差）"""
    noise = np.random.normal(0, sigma, frames.shape).astype(np.float32)
    return frames + noise


def apply_lpf(frames: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """對每一幀做 Gaussian LPF（spatial low-pass filter）"""
    result = np.zeros_like(frames)
    for t in range(frames.shape[0]):
        frame = frames[t, :, :, 0]
        smoothed = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        result[t, :, :, 0] = smoothed
    return result


def compute_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """計算 signal-to-noise ratio (dB)"""
    signal_power = np.mean(clean ** 2)
    noise_power  = np.mean((noisy - clean) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def normalize(frames: np.ndarray) -> np.ndarray:
    mean = np.mean(frames)
    std  = np.std(frames) + 1e-6
    return ((frames - mean) / std).astype(np.float32)


def predict_one(model, frames: np.ndarray) -> str:
    """逐一預測（beam search + correct_sentence），與 benchmark 完全一致"""
    inp  = tf.expand_dims(frames, axis=0)
    yhat = model.predict(inp, verbose=0)
    decoded = tf.keras.backend.ctc_decode(
        tf.cast(yhat, tf.float32), input_length=[75], greedy=False)[0][0].numpy()
    raw = tf.strings.reduce_join(
        [num_to_char(w) for w in decoded[0]]
    ).numpy().decode('utf-8').strip()
    raw_words = ' '.join(w for w in raw.split() if len(w) > 1)
    return ' '.join(correct_sentence(raw_words).split())


def test_speaker(model, speaker: str, align_folder: str, max_samples: int,
                 noise_sigmas: list, lpf_kernel: int = 3):
    """
    對某位說話者測試不同雜訊等級的辨識準確率（逐一預測，避免 GPU OOM）。
    回傳 dict: {sigma: {'correct_clean', 'correct_noisy', 'correct_lpf', 'total', 'snr_db_list'}}
    """
    cache_dir = os.path.join('data', f'{speaker}_cached')
    align_dir = os.path.join('data', 'alignments', align_folder)
    if not os.path.exists(cache_dir) or not os.path.exists(align_dir):
        print(f'  [跳過] 找不到 {cache_dir} 或 {align_dir}')
        return None

    files = sorted([f for f in os.listdir(cache_dir) if f.endswith('.npy')])[:max_samples]
    if not files:
        print(f'  [跳過] {cache_dir} 無 .npy 檔案')
        return None

    results = {sigma: {'correct_clean': 0, 'correct_noisy': 0, 'correct_lpf': 0,
                       'total': 0, 'snr_db_list': []}
               for sigma in noise_sigmas}

    for i, fname in enumerate(files):
        align_path = os.path.join(align_dir, fname.replace('.npy', '.align'))
        if not os.path.exists(align_path):
            continue

        frames = np.load(os.path.join(cache_dir, fname))
        frames = normalize(frames)
        if frames.shape[0] < 75:
            continue
        frames = frames[:75]
        gt = load_alignments(align_path)

        # 先預測一次 clean（所有 sigma 共用）
        pred_clean = predict_one(model, frames)

        for sigma in noise_sigmas:
            noisy = add_gaussian_noise(frames, sigma)
            lpf   = apply_lpf(noisy, kernel_size=lpf_kernel)
            snr   = compute_snr_db(frames, noisy)

            pred_noisy = predict_one(model, noisy)
            pred_lpf   = predict_one(model, lpf)

            results[sigma]['total'] += 1
            results[sigma]['snr_db_list'].append(snr)
            if pred_clean == gt: results[sigma]['correct_clean'] += 1
            if pred_noisy == gt: results[sigma]['correct_noisy'] += 1
            if pred_lpf   == gt: results[sigma]['correct_lpf']   += 1

        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{len(files)} 筆完成...', flush=True)

    return results


def format_results(speaker: str, results: dict) -> str:
    lines = []
    lines.append(f'\n說話者：{speaker}')
    lines.append(f'{"sigma":>8}  {"SNR(dB)":>8}  {"無雜訊":>8}  {"有雜訊":>8}  {"LPF後":>8}  {"LPF提升":>8}')
    lines.append('-' * 62)
    for sigma, r in sorted(results.items()):
        n = r['total']
        if n == 0:
            continue
        snr = np.mean(r['snr_db_list'])
        acc_clean = r['correct_clean'] / n * 100
        acc_noisy = r['correct_noisy'] / n * 100
        acc_lpf   = r['correct_lpf']   / n * 100
        delta     = acc_lpf - acc_noisy
        lines.append(
            f'{sigma:>8.2f}  {snr:>8.1f}  {acc_clean:>7.1f}%  '
            f'{acc_noisy:>7.1f}%  {acc_lpf:>7.1f}%  {delta:>+7.1f}%'
        )
    return '\n'.join(lines)


# ===== 設定 =====
NOISE_SIGMAS = [0.0, 0.2, 0.5, 1.0, 2.0]  # normalized 空間的 σ（對應 SNR ∞, ~14, ~6, ~0, ~-6 dB）
LPF_KERNEL   = 3                             # Gaussian blur kernel size
MAX_SAMPLES  = 100                           # 每位說話者最多測幾筆

TEST_SPEAKERS = [
    # (speaker_folder, alignment_folder, 顯示名稱)
    ('s99_1',  's99_1',  's99_1'),
    ('s34_3',  's34_3',  's34_3'),
    ('s1',     's1',     'GRID s1'),
    ('s5',     's5',     'GRID s5'),
    ('s6',     's6',     'GRID s6'),
]

if __name__ == '__main__':
    print('載入 v8 模型...')
    model = load_model(MODEL_PATH, custom_objects={'CTCLoss': CTCLoss})
    print('載入完成！\n')
    print(f'雜訊設定：sigma = {NOISE_SIGMAS}')
    print(f'LPF kernel size：{LPF_KERNEL}x{LPF_KERNEL} Gaussian')
    print(f'每位說話者最多 {MAX_SAMPLES} 筆\n')

    all_output = []
    all_output.append('=' * 70)
    all_output.append('實驗二：影像雜訊 vs LipNet 準確率（DSP LPF 補救）')
    all_output.append(f'模型：trained_model_grid_multi_v8.h5')
    all_output.append(f'雜訊：Gaussian noise σ = {NOISE_SIGMAS}（normalized 空間）')
    all_output.append(f'LPF：{LPF_KERNEL}x{LPF_KERNEL} Gaussian blur')
    all_output.append('=' * 70)

    for speaker_dir, align_dir, display_name in TEST_SPEAKERS:
        print(f'測試 {display_name}...', flush=True)
        results = test_speaker(
            model, speaker_dir, align_dir, MAX_SAMPLES,
            NOISE_SIGMAS, lpf_kernel=LPF_KERNEL
        )
        if results:
            text = format_results(display_name, results)
            print(text)
            all_output.append(text)

    summary = '\n'.join(all_output)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f'\n結果已儲存至：{LOG_PATH}')
