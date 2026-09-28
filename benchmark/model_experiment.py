"""
模型架構小實驗：比較四種 LipNet 變體的 loss 下降曲線

比較的模型：
  A. Original  — 原版 LipNet，輸入 (75,46,140,1)
  B. VShape    — 前/當/後 三幀合一，輸入 (75,46,140,3)
  C. BN        — 原版 + BatchNormalization（訓練更穩定）
  D. BN+Drop   — BN 版 + 更多 Dropout（防止過擬合）

使用方式：
  1. 確認 CACHE_DIR 指向你有 .npy 快取的資料夾
  2. 執行：python model_experiment.py
  3. 結果圖：experiment_loss.png
"""

import os
import glob
import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')   # 不需要開視窗，直接存圖
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional,
    MaxPool3D, Activation, TimeDistributed, Flatten,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam

# ============================================================
# ★ 請修改這裡 ★
# ============================================================
CACHE_DIR   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s1_cached'
ALIGN_DIR   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\alignments\s1'
N_SAMPLES   = 60       # 取幾筆資料做實驗（越少越快，建議 50~100）
EPOCHS      = 10       # 跑幾個 epoch
BATCH_SIZE  = 1        # 每批幾筆（記憶體不夠就改 1）
RESULT_IMG  = 'experiment_loss.png'
# ============================================================

VOCAB = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=VOCAB, oov_token='')
VOCAB_SIZE  = char_to_num.vocabulary_size()


# ── CTC Loss ────────────────────────────────────────────────
def CTCLoss(y_true, y_pred):
    batch_len    = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


# ── 四種模型 ─────────────────────────────────────────────────
def model_original():
    """A. 原版 LipNet（1 通道輸入）"""
    m = Sequential([
        Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'), Activation('relu'), MaxPool3D((1,2,2)),
        Conv3D(256, 3, padding='same'), Activation('relu'), MaxPool3D((1,2,2)),
        Conv3D(75,  3, padding='same'), Activation('relu'), MaxPool3D((1,2,2)),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Dense(VOCAB_SIZE + 1, kernel_initializer='he_normal', activation='softmax'),
    ], name='A_Original')
    return m

def model_vshape():
    """B. V-shape（前/當/後 三幀，3 通道輸入）"""
    m = Sequential([
        Conv3D(128, 3, input_shape=(75, 46, 140, 3), padding='same'), Activation('relu'), MaxPool3D((1,2,2)),
        Conv3D(256, 3, padding='same'), Activation('relu'), MaxPool3D((1,2,2)),
        Conv3D(75,  3, padding='same'), Activation('relu'), MaxPool3D((1,2,2)),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Dense(VOCAB_SIZE + 1, kernel_initializer='he_normal', activation='softmax'),
    ], name='B_VShape')
    return m

def model_bn():
    """C. 原版 + BatchNormalization（訓練更穩定）"""
    m = Sequential([
        Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)),
        Conv3D(256, 3, padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)),
        Conv3D(75,  3, padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Dense(VOCAB_SIZE + 1, kernel_initializer='he_normal', activation='softmax'),
    ], name='C_BN')
    return m

def model_bn_drop():
    """D. BN + 更多 Dropout（防過擬合）"""
    m = Sequential([
        Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)), Dropout(0.2),
        Conv3D(256, 3, padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)), Dropout(0.2),
        Conv3D(75,  3, padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Dense(VOCAB_SIZE + 1, kernel_initializer='he_normal', activation='softmax'),
    ], name='D_BN_Drop')
    return m


# ── 載入對齊檔 ───────────────────────────────────────────────
def load_alignment(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3 and parts[2] != 'sil':
            tokens.append(parts[2])
    text = ' '.join(tokens)
    nums = char_to_num(tf.strings.unicode_split(text, 'UTF-8'))
    # 固定長度 40，不足補 0，超過截斷
    nums = nums[:40]
    pad = tf.zeros([40 - tf.shape(nums)[0]], dtype=tf.int64)
    return tf.concat([nums, pad], axis=0).numpy()


# ── 載入資料 ─────────────────────────────────────────────────
def load_dataset(n_samples, use_vshape=False):
    npy_files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.npy')))
    random.shuffle(npy_files)
    npy_files = npy_files[:n_samples]

    X_list, Y_list = [], []
    skipped = 0
    for p in npy_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        align_path = os.path.join(ALIGN_DIR, stem + '.align')
        if not os.path.exists(align_path):
            skipped += 1
            continue
        try:
            frames = np.load(p).astype(np.float32)   # 預期 (75,46,140) 或 (75,46,140,1/3)

            # 統一正規化
            mean, std = frames.mean(), frames.std() + 1e-6
            frames = (frames - mean) / std

            # 調整通道維度
            if frames.ndim == 3:
                frames = frames[..., np.newaxis]      # → (75,46,140,1)

            if use_vshape:
                ch = frames[..., 0]                   # (75,46,140)
                prev_f = np.concatenate([ch[:1], ch[:-1]], axis=0)
                next_f = np.concatenate([ch[1:], ch[-1:]], axis=0)
                frames = np.stack([prev_f, ch, next_f], axis=-1)  # (75,46,140,3)
            else:
                frames = frames[..., :1]              # 只取第 1 通道

            label = load_alignment(align_path)
            X_list.append(frames)
            Y_list.append(label)
        except Exception as e:
            skipped += 1

    print(f"  載入 {len(X_list)} 筆，跳過 {skipped} 筆")
    return np.array(X_list), np.array(Y_list)


# ── 訓練單一模型並回傳 loss 歷史 ────────────────────────────
def run_experiment(model_fn, X, Y):
    tf.keras.backend.clear_session()   # 清空上一個模型的 GPU 記憶體
    model = model_fn()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

    history = model.fit(
        X, Y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    del model   # 明確釋放記憶體
    return history.history['loss']


# ── 主程式 ───────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("載入資料（1 通道，A/C/D 用）...")
    X1, Y = load_dataset(N_SAMPLES, use_vshape=False)
    print(f"  X shape: {X1.shape}, Y shape: {Y.shape}")

    print("\n載入資料（3 通道，B V-shape 用）...")
    X3, _ = load_dataset(N_SAMPLES, use_vshape=True)
    print(f"  X shape: {X3.shape}")

    results = {}
    configs = [
        ('A_Original', model_original, X1),
        ('B_VShape',   model_vshape,   X3),
        ('C_BN',       model_bn,       X1),
        ('D_BN_Drop',  model_bn_drop,  X1),
    ]

    for name, fn, X in configs:
        print(f"\n{'='*60}")
        print(f"訓練模型：{name}")
        print(f"{'='*60}")
        losses = run_experiment(fn, X, Y)
        results[name] = losses
        print(f"  最終 loss：{losses[-1]:.4f}  最低 loss：{min(losses):.4f}")

    # ── 畫圖 ──────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    styles = {'A_Original': 'b-o', 'B_VShape': 'r-s', 'C_BN': 'g-^', 'D_BN_Drop': 'm-D'}
    for name, losses in results.items():
        plt.plot(range(1, EPOCHS+1), losses, styles.get(name, '-'), label=name, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('CTC Loss')
    plt.title('四種模型架構 Loss 比較')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_IMG, dpi=150)
    print(f"\n結果圖已儲存：{RESULT_IMG}")

    # ── 文字結論 ──────────────────────────────────────────────
    print("\n【結果總覽】")
    for name, losses in results.items():
        print(f"  {name}: 起始={losses[0]:.2f} → 最低={min(losses):.2f} （第{losses.index(min(losses))+1}epoch）")

    best = min(results, key=lambda k: min(results[k]))
    print(f"\n★ loss 下降最多的模型：{best}")
