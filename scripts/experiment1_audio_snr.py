"""
實驗一：聲學通道 SNR vs 辨識準確率
- 對 GRID corpus .mpg 抽取音訊，加入 AWGN 雜訊（不同 SNR）
- 使用 Whisper 做音訊 ASR，計算 WER
- 與 LipNet 視覺辨識（不受音訊雜訊影響）比較
- 核心論點：聲學通道退化時，視覺通道維持穩定
"""

import os
import pathlib
import warnings
import subprocess
import tempfile
import numpy as np
import tensorflow as tf

pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model

# Whisper（需要：pip install openai-whisper）
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print('[警告] whisper 未安裝，音訊 ASR 部分將跳過。')
    print('       請執行：pip install openai-whisper')

# ffmpeg 用於從 .mpg 抽取音訊
import soundfile as sf

BASE_DIR   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_v8.h5')
LOG_PATH   = os.path.join(BASE_DIR, 'logs', 'experiment1_audio_snr_log.txt')

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


def load_alignments_str(path: str) -> str:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split()
        if parts[2] != 'sil':
            tokens.extend([' ', parts[2]])
    return ''.join(tokens[1:]).strip()


def extract_audio(mpg_path: str, wav_path: str, sample_rate: int = 16000):
    """使用 ffmpeg 從 .mpg 抽取 16kHz mono wav"""
    cmd = [
        'ffmpeg', '-y', '-i', mpg_path,
        '-ar', str(sample_rate), '-ac', '1',
        '-f', 'wav', wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def add_awgn(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """對音訊加入 AWGN，使訊噪比為 snr_db"""
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), audio.shape).astype(np.float32)
    return audio + noise


def word_error_rate(ref: str, hyp: str) -> float:
    """計算 WER（Word Error Rate）"""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    # 動態規劃計算編輯距離
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    mean = np.mean(frames)
    std  = np.std(frames) + 1e-6
    return ((frames - mean) / std).astype(np.float32)


def predict_lipnet(model, speaker: str, fname: str) -> str:
    """用 LipNet 對單個影片做預測（從 cache 讀取）"""
    for suffix in ['_cached', '_dlib_cached']:
        cache_path = os.path.join('data', f'{speaker}{suffix}', fname.replace('.mpg', '.npy'))
        if os.path.exists(cache_path):
            frames = np.load(cache_path)
            frames = normalize_frames(frames)
            if frames.shape[0] < 75:
                return ''
            frames = frames[:75]
            inp  = tf.expand_dims(frames, axis=0)
            yhat = model.predict(inp, verbose=0)
            decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
            result = tf.strings.reduce_join(
                [num_to_char(w) for w in decoded[0]]
            ).numpy().decode('utf-8')
            return result.strip()
    return ''


# ===== 設定 =====
SNR_LEVELS   = [20, 15, 10, 5, 0, -5]   # dB
MAX_SAMPLES  = 100                        # 每位說話者最多測幾筆
WHISPER_MODEL_SIZE = 'base'              # tiny / base / small / medium

# 測試說話者（只用有 alignment 的 GRID 說話者）
TEST_SPEAKERS = [
    ('s1',  's1',  'GRID s1'),
    ('s5',  's5',  'GRID s5'),
    ('s6',  's6',  'GRID s6'),
]


if __name__ == '__main__':
    print('載入 LipNet v8 模型...')
    lipnet = load_model(MODEL_PATH, custom_objects={'CTCLoss': CTCLoss})
    print('LipNet 載入完成！')

    whisper_model = None
    if WHISPER_AVAILABLE:
        print(f'載入 Whisper ({WHISPER_MODEL_SIZE})...')
        whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        print('Whisper 載入完成！\n')

    all_output = []
    all_output.append('=' * 72)
    all_output.append('實驗一：聲學通道 SNR vs 辨識準確率')
    all_output.append(f'LipNet 模型：trained_model_grid_multi_v8.h5')
    all_output.append(f'Whisper 模型：{WHISPER_MODEL_SIZE}')
    all_output.append(f'SNR 等級：{SNR_LEVELS} dB')
    all_output.append('=' * 72)

    for speaker_dir, align_dir, display_name in TEST_SPEAKERS:
        data_dir  = os.path.join('data', speaker_dir)
        align_base = os.path.join('data', 'alignments', align_dir)
        if not os.path.exists(data_dir):
            print(f'[跳過] {data_dir} 不存在')
            continue

        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mpg')])[:MAX_SAMPLES]
        print(f'\n測試 {display_name}（{len(files)} 筆）...')

        # 初始化計數器
        lipnet_correct = 0
        lipnet_total   = 0
        asr_wer = {snr: [] for snr in SNR_LEVELS}

        with tempfile.TemporaryDirectory() as tmp_dir:
            for fname in files:
                mpg_path   = os.path.join(data_dir, fname)
                align_path = os.path.join(align_base, fname.replace('.mpg', '.align'))
                if not os.path.exists(align_path):
                    continue

                ground_truth = load_alignments_str(align_path)

                # LipNet 預測（乾淨影像，不受音訊雜訊影響）
                pred_lip = predict_lipnet(lipnet, speaker_dir, fname)
                if pred_lip:
                    lipnet_total += 1
                    if pred_lip == ground_truth:
                        lipnet_correct += 1

                # Whisper 音訊 ASR（不同 SNR）
                if whisper_model:
                    wav_clean = os.path.join(tmp_dir, 'clean.wav')
                    extract_audio(mpg_path, wav_clean)
                    if not os.path.exists(wav_clean):
                        continue
                    audio_data, sr = sf.read(wav_clean, dtype='float32')

                    for snr in SNR_LEVELS:
                        if snr == float('inf'):
                            audio_noisy = audio_data
                        else:
                            audio_noisy = add_awgn(audio_data, snr)

                        wav_noisy = os.path.join(tmp_dir, f'noisy_{snr}.wav')
                        sf.write(wav_noisy, audio_noisy, sr)
                        try:
                            result = whisper_model.transcribe(wav_noisy, language='en')
                            hyp = result['text'].strip().lower()
                        except Exception:
                            hyp = ''
                        wer = word_error_rate(ground_truth, hyp)
                        asr_wer[snr].append(wer)

        # 輸出結果
        lipnet_acc = lipnet_correct / lipnet_total * 100 if lipnet_total > 0 else 0
        block = []
        block.append(f'\n說話者：{display_name}（{lipnet_total} 筆）')
        block.append(f'{"SNR (dB)":>10}  {"音訊 WER":>10}  {"音訊正確率":>10}  {"LipNet 正確率":>14}')
        block.append('-' * 52)

        for snr in SNR_LEVELS:
            if asr_wer[snr]:
                avg_wer = np.mean(asr_wer[snr])
                asr_acc = (1 - avg_wer) * 100
            else:
                avg_wer = -1
                asr_acc = -1
            lip_str = f'{lipnet_acc:.1f}%'
            asr_str = f'{asr_acc:.1f}%' if asr_acc >= 0 else '(無 Whisper)'
            block.append(f'{snr:>10}  {avg_wer:>10.3f}  {asr_str:>10}  {lip_str:>14}')

        text = '\n'.join(block)
        print(text)
        all_output.append(text)

    summary = '\n'.join(all_output)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f'\n結果已儲存至：{LOG_PATH}')
