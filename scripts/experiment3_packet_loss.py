"""
實驗三：UDP 封包遺失（幀遺失）對 LipNet 的影響
- 隨機丟棄一定比例的幀，模擬無線網路封包遺失
- 比較：零填充 vs 前幀複製（frame repetition）兩種補救策略
"""

import os
import sys
import pathlib
import warnings
import numpy as np
import tensorflow as tf

sys.path.insert(0, r'C:\Users\Tno\claude-code')
from vocab_correction import correct_sentence

pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore")

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model

BASE_DIR   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model_grid_multi_v8.h5')
LOG_PATH   = os.path.join(BASE_DIR, 'logs', 'experiment3_packet_loss_v2_log.txt')

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


VOCAB_SET = set("abcdefghijklmnopqrstuvwxyz'?!123456789 ")

def load_alignments_str(path: str) -> str:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[2] not in ('sil', 'sp') and len(parts[2]) > 1:
            tokens.extend([' ', parts[2]])
    raw = ''.join(tokens[1:])
    filtered = ''.join(c for c in raw if c in VOCAB_SET)
    return ' '.join(filtered.split())


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    mean = np.mean(frames)
    std  = np.std(frames) + 1e-6
    return ((frames - mean) / std).astype(np.float32)


def simulate_packet_loss_zero(frames: np.ndarray, loss_rate: float,
                               seed: int = 42) -> np.ndarray:
    """丟棄幀後用零填充"""
    rng = np.random.default_rng(seed)
    result = frames.copy()
    mask = rng.random(frames.shape[0]) < loss_rate
    result[mask] = 0.0
    return result


def simulate_packet_loss_repeat(frames: np.ndarray, loss_rate: float,
                                 seed: int = 42) -> np.ndarray:
    """丟棄幀後用前一幀複製（frame repetition）"""
    rng = np.random.default_rng(seed)
    result = frames.copy()
    mask = rng.random(frames.shape[0]) < loss_rate
    for i in range(len(mask)):
        if mask[i] and i > 0:
            result[i] = result[i - 1]
    return result


def predict(model, frames: np.ndarray) -> str:
    inp  = tf.expand_dims(frames, axis=0)
    yhat = model.predict(inp, verbose=0)
    decoded = tf.keras.backend.ctc_decode(
        tf.cast(yhat, tf.float32), input_length=[75], greedy=False)[0][0].numpy()
    raw = tf.strings.reduce_join(
        [num_to_char(w) for w in decoded[0]]
    ).numpy().decode('utf-8').strip()
    raw_words = ' '.join(w for w in raw.split() if len(w) > 1)
    return ' '.join(correct_sentence(raw_words).split())


# ===== 設定 =====
LOSS_RATES  = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
MAX_SAMPLES = 100

TEST_SPEAKERS = [
    ('s1',    's1',    'GRID s1'),
    ('s2',    's2',    'GRID s2'),
    ('s3',    's3',    'GRID s3'),
    ('s4',    's4',    'GRID s4'),
    ('s5',    's5',    'GRID s5'),
    ('s6',    's6',    'GRID s6'),
    ('s7',    's7',    'GRID s7'),
    ('s8',    's8',    'GRID s8'),
    ('s13',   's13',   'GRID s13'),
    ('s34_3', 's34_3', 's34_3'),
]


if __name__ == '__main__':
    print('載入 v8 模型...')
    model = load_model(MODEL_PATH, custom_objects={'CTCLoss': CTCLoss})
    print('載入完成！\n')

    all_output = []
    all_output.append('=' * 68)
    all_output.append('實驗三：UDP 封包遺失（幀遺失）vs LipNet 準確率')
    all_output.append(f'模型：trained_model_grid_multi_v8.h5')
    all_output.append(f'遺失率：{[f"{r*100:.0f}%" for r in LOSS_RATES]}')
    all_output.append('=' * 68)

    for speaker_dir, align_dir, display_name in TEST_SPEAKERS:
        cache_dir = os.path.join('data', f'{speaker_dir}_cached')
        align_base = os.path.join('data', 'alignments', align_dir)
        if not os.path.exists(cache_dir) or not os.path.exists(align_base):
            continue

        files = sorted([f for f in os.listdir(cache_dir) if f.endswith('.npy')])[:MAX_SAMPLES]
        print(f'測試 {display_name}（{len(files)} 筆）...')

        counters = {r: {'zero': 0, 'repeat': 0, 'total': 0} for r in LOSS_RATES}

        for fname in files:
            align_path = os.path.join(align_base, fname.replace('.npy', '.align'))
            if not os.path.exists(align_path):
                continue
            frames = np.load(os.path.join(cache_dir, fname))
            frames = normalize_frames(frames)
            if frames.shape[0] < 75:
                continue
            frames = frames[:75]
            gt = load_alignments_str(align_path)

            for rate in LOSS_RATES:
                f_zero   = simulate_packet_loss_zero(frames, rate)
                f_repeat = simulate_packet_loss_repeat(frames, rate)
                counters[rate]['total'] += 1
                if predict(model, f_zero)   == gt: counters[rate]['zero']   += 1
                if predict(model, f_repeat) == gt: counters[rate]['repeat'] += 1

        block = []
        block.append(f'\n說話者：{display_name}')
        block.append(f'{"遺失率":>8}  {"零填充":>10}  {"前幀複製":>10}  {"提升":>8}')
        block.append('-' * 44)
        for rate in LOSS_RATES:
            n = counters[rate]['total']
            if n == 0: continue
            acc_z = counters[rate]['zero']   / n * 100
            acc_r = counters[rate]['repeat'] / n * 100
            block.append(f'{rate*100:>7.0f}%  {acc_z:>9.1f}%  {acc_r:>9.1f}%  {acc_r-acc_z:>+7.1f}%')
        text = '\n'.join(block)
        print(text)
        all_output.append(text)

    summary = '\n'.join(all_output)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f'\n結果已儲存至：{LOG_PATH}')
