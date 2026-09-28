"""
模型架構小實驗：比較六種 LipNet 變體的 loss 下降曲線、參數量與訓練速度

比較的模型：
  A. Original   — 原版 LipNet，輸入 (75,46,140,1)
  B. VShape     — 前/當/後 三幀合一，輸入 (75,46,140,3)
  C. BN         — 原版 + BatchNormalization（訓練更穩定，目前確認有效的基準版本）
  D. BN+Drop    — BN 版 + 更多 Dropout（防止過擬合）
  E. DSConv     — BN 版 + 深度可分離卷積（輕量化，減少參數量與運算量）
  F. Width0.5   — BN 版 + 通道數縮減為一半（輕量化，減少參數量與運算量）

E、F 兩個模型是為了「效能輕量化」這個專題方向新增的，做法參考自：
  Ma, P., Martinez, B., Petridis, S., & Pantic, M. (2021).
  "Towards Practical Lipreading with Distilled and Efficient Models."
  ICASSP 2021. Imperial College London / Samsung AI Center Cambridge.
  https://arxiv.org/abs/2007.06504
  （論文核心發現：用深度可分離卷積替換一般卷積，在 LRW 資料集上參數量可縮小
    4~17 倍、運算量(FLOPs)可縮小 8 倍以上，準確度幾乎不掉。）

完整實驗原由、方法說明、結果記錄表，請見同資料夾的 experiment_dsconv_width_notes.md。

使用方式：
  1. 確認 CACHE_DIR 指向你有 .npy 快取的資料夾
  2. 執行：python model_experiment.py
  3. 結果圖：experiment_loss.png
  4. 結果數據表（參數量、訓練時間、loss）：experiment_results.csv
     → 訓練完後，把這個 CSV 的數據填進 experiment_dsconv_width_notes.md 的表格裡
"""

import os
import csv
import gc
import glob
import random
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')   # 不需要開視窗，直接存圖
import matplotlib.pyplot as plt

# GPU 記憶體「用多少要多少」，不要一開始就把整張卡的記憶體全部鎖住。
# 這台筆電 GPU 只有 6GB，還有其他軟體在用，一定要開這個才不容易 OOM。
for _gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

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
N_SAMPLES   = 80       # 取幾筆資料做實驗（越少越快，建議 50~100）
EPOCHS      = 15       # 跑幾個 epoch
BATCH_SIZE  = 1        # 每批幾筆（GPU 只剩約3.5GB可用，2會OOM，改成1）
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

def model_bn_dsconv():
    """E. BN 版 + 深度可分離卷積（Depthwise Separable Conv3D，輕量化）

    原理：把一個「一次看完所有輸入通道」的卷積，拆成兩個步驟：
      1. 深度卷積（depthwise）：每個輸入通道「各自」做空間+時間濾波，
         用 Conv3D 的 groups 參數設成等於輸入通道數就能達成，不需要自訂新的層。
      2. 逐點卷積（pointwise）：用 1x1x1 卷積把剛剛各自處理完的通道「混合」，
         同時把通道數變換成想要的輸出數量。

    第一層維持標準卷積：因為輸入只有 1 個通道（灰階影格），拆成深度可分離
    卷積沒有意義（1 個通道沒什麼好「分開處理」的），所以只拆第二、三層。
    """
    m = Sequential([
        # 第一層：維持標準卷積（輸入通道數=1）
        Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)),

        # 第二層：深度可分離卷積（128 → 256）
        Conv3D(128, 3, padding='same', groups=128), BatchNormalization(), Activation('relu'),   # 深度卷積
        Conv3D(256, 1, padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)),  # 逐點卷積

        # 第三層：深度可分離卷積（256 → 75）
        Conv3D(256, 3, padding='same', groups=256), BatchNormalization(), Activation('relu'),   # 深度卷積
        Conv3D(75, 1, padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)),   # 逐點卷積

        TimeDistributed(Flatten()),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Dense(VOCAB_SIZE + 1, kernel_initializer='he_normal', activation='softmax'),
    ], name='E_DSConv')
    return m

def model_bn_width(width_mult=0.5):
    """F. BN 版 + 寬度縮放（Width Multiplier，輕量化）

    原理：把每一層卷積的「輸出通道數」統一乘上一個倍率（例如0.5倍）。
    通道數越少，代表每一層要學習、要運算的東西越少，模型就更小更快，
    但學習能力也會跟著下降，所以要實際跑過才知道準確度掉多少。
    """
    c1 = max(8, int(128 * width_mult))
    c2 = max(8, int(256 * width_mult))
    c3 = max(8, int(75 * width_mult))
    m = Sequential([
        Conv3D(c1, 3, input_shape=(75, 46, 140, 1), padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)),
        Conv3D(c2, 3, padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)),
        Conv3D(c3, 3, padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Dense(VOCAB_SIZE + 1, kernel_initializer='he_normal', activation='softmax'),
    ], name=f'F_Width{width_mult}')
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


# ── 訓練單一模型並回傳 loss 歷史 + 參數量 + 訓練時間 ─────────
def run_experiment(model_fn, X, Y, use_vshape=False):
    # 清掉上一個模型留在 GPU 上的東西（權重、優化器狀態、計算圖），
    # 不然訓練到第 3、4 個模型時記憶體會被前面的模型卡住，導致 OOM。
    tf.keras.backend.clear_session()
    gc.collect()

    model = model_fn()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)
    model.summary(line_length=80)

    total_params = model.count_params()   # 模型總參數量，越少代表模型越小

    start_time = time.time()
    history = model.fit(
        X, Y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    elapsed_sec = time.time() - start_time   # 總訓練耗時（秒）

    return {
        'loss_history': history.history['loss'],
        'total_params': total_params,
        'train_time_sec': elapsed_sec,
        'time_per_epoch_sec': elapsed_sec / EPOCHS,
    }


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
        ('E_DSConv',   model_bn_dsconv, X1),
        ('F_Width0.5', lambda: model_bn_width(0.5), X1),
    ]

    for name, fn, X in configs:
        print(f"\n{'='*60}")
        print(f"訓練模型：{name}")
        print(f"{'='*60}")
        result = run_experiment(fn, X, Y)
        results[name] = result
        losses = result['loss_history']
        print(f"  最終 loss：{losses[-1]:.4f}  最低 loss：{min(losses):.4f}")
        print(f"  參數量：{result['total_params']:,}  每 epoch 耗時：{result['time_per_epoch_sec']:.1f} 秒")

    # ── 畫圖 ──────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    styles = {
        'A_Original': 'b-o', 'B_VShape': 'r-s', 'C_BN': 'g-^', 'D_BN_Drop': 'm-D',
        'E_DSConv': 'c-v', 'F_Width0.5': 'y-*',
    }
    for name, result in results.items():
        plt.plot(range(1, EPOCHS+1), result['loss_history'], styles.get(name, '-'), label=name, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('CTC Loss')
    plt.title('模型架構 Loss 比較')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_IMG, dpi=150)
    print(f"\n結果圖已儲存：{RESULT_IMG}")

    # ── 文字結論 ──────────────────────────────────────────────
    print("\n【結果總覽】")
    for name, result in results.items():
        losses = result['loss_history']
        print(f"  {name}: 起始={losses[0]:.2f} → 最低={min(losses):.2f} （第{losses.index(min(losses))+1}epoch）"
              f"  參數量={result['total_params']:,}  每epoch={result['time_per_epoch_sec']:.1f}秒")

    best = min(results, key=lambda k: min(results[k]['loss_history']))
    print(f"\n★ loss 下降最多的模型：{best}")

    # ── 存成 CSV，方便報告時直接引用或畫表格 ────────────────────
    csv_path = 'experiment_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            '模型', '參數量', '每epoch耗時(秒)', '總訓練耗時(秒)',
            '起始loss', '最終loss', '最低loss', '最低loss所在epoch',
        ])
        for name, result in results.items():
            losses = result['loss_history']
            writer.writerow([
                name,
                result['total_params'],
                round(result['time_per_epoch_sec'], 2),
                round(result['train_time_sec'], 2),
                round(losses[0], 4),
                round(losses[-1], 4),
                round(min(losses), 4),
                losses.index(min(losses)) + 1,
            ])
    print(f"結果數據表已儲存：{csv_path}")

    # ── 針對輕量化實驗的 before/after 對照（以 C_BN 為基準）────
    if 'C_BN' in results:
        baseline = results['C_BN']
        print("\n【輕量化前後對照，基準 = C_BN】")
        for name in ('E_DSConv', 'F_Width0.5'):
            if name not in results:
                continue
            r = results[name]
            param_ratio = r['total_params'] / baseline['total_params']
            speed_ratio = baseline['time_per_epoch_sec'] / r['time_per_epoch_sec']
            loss_diff = min(r['loss_history']) - min(baseline['loss_history'])
            print(f"  {name}: 參數量為基準的 {param_ratio:.2%}"
                  f"　訓練速度為基準的 {speed_ratio:.2f} 倍"
                  f"　最低loss差異 {loss_diff:+.4f}（正值代表比基準差）")
