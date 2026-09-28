"""
小規模架構先導實驗：Transformer 後端 vs 現有 C_BN(LSTM) 基準

比較：
  C_BN         — 現有基準（Conv3D+BatchNorm 前端 ＋ BiLSTM 後端），沿用 model_experiment.py 的版本
  E_Transformer— 同樣的 Conv3D+BatchNorm 前端，後端把 BiLSTM 換成 Transformer Encoder（Self-Attention）

方法論比照既有 model_experiment.py：60 筆 s1 資料、10 epoch、batch_size=1、從頭訓練，
比較 loss 下降曲線 + 記錄每個 epoch 平均耗時，用來推算完整訓練規模所需時間。
"""

import os
import glob
import random
import time
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, LSTM, Dense, Dropout, Bidirectional,
    MaxPool3D, Activation, TimeDistributed, Flatten,
    BatchNormalization, MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.optimizers import Adam

CACHE_DIR   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s1_cached'
ALIGN_DIR   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\alignments\s1'
N_SAMPLES   = 60
EPOCHS      = 10
BATCH_SIZE  = 1

VOCAB = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=VOCAB, oov_token='')
VOCAB_SIZE  = char_to_num.vocabulary_size()


def CTCLoss(y_true, y_pred):
    batch_len    = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def model_bn():
    """C. 現有基準：Conv3D+BN 前端 + BiLSTM 後端（與 model_experiment.py 一致）"""
    m = tf.keras.Sequential([
        Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)),
        Conv3D(256, 3, padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)),
        Conv3D(75,  3, padding='same'), BatchNormalization(), Activation('relu'), MaxPool3D((1,2,2)),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)), Dropout(0.5),
        Dense(VOCAB_SIZE + 1, kernel_initializer='he_normal', activation='softmax'),
    ], name='C_BN_baseline')
    return m


def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)


def transformer_block(x, d_model=256, num_heads=4, ff_dim=512, dropout=0.1, name=''):
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                                   name=f'{name}_mha')(x, x)
    attn_out = Dropout(dropout)(attn_out)
    x = Add()([x, attn_out])
    x = LayerNormalization(epsilon=1e-6)(x)

    ffn = Dense(ff_dim, activation='relu')(x)
    ffn = Dense(d_model)(ffn)
    ffn = Dropout(dropout)(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def model_transformer():
    """E. 先導版：同樣 Conv3D+BN 前端，後端把 BiLSTM 換成 2 層 Transformer Encoder"""
    d_model = 256
    inp = Input(shape=(75, 46, 140, 1))
    x = Conv3D(128, 3, padding='same')(inp); x = BatchNormalization()(x); x = Activation('relu')(x); x = MaxPool3D((1,2,2))(x)
    x = Conv3D(256, 3, padding='same')(x);   x = BatchNormalization()(x); x = Activation('relu')(x); x = MaxPool3D((1,2,2))(x)
    x = Conv3D(75,  3, padding='same')(x);   x = BatchNormalization()(x); x = Activation('relu')(x); x = MaxPool3D((1,2,2))(x)
    x = TimeDistributed(Flatten())(x)                 # (batch, 75, feat_dim)
    x = Dense(d_model)(x)                              # 投影到 d_model
    pe = positional_encoding(75, d_model)
    x = x + pe                                          # 加上位置編碼(Transformer本身無序列順序概念)
    x = transformer_block(x, d_model=d_model, num_heads=4, ff_dim=512, dropout=0.1, name='blk1')
    x = transformer_block(x, d_model=d_model, num_heads=4, ff_dim=512, dropout=0.1, name='blk2')
    out = Dense(VOCAB_SIZE + 1, kernel_initializer='he_normal', activation='softmax')(x)
    return Model(inp, out, name='E_Transformer_pilot')


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
    nums = nums[:40]
    pad = tf.zeros([40 - tf.shape(nums)[0]], dtype=tf.int64)
    return tf.concat([nums, pad], axis=0).numpy()


def load_dataset(n_samples):
    npy_files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.npy')))
    random.seed(42)
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
            frames = np.load(p).astype(np.float32)
            mean, std = frames.mean(), frames.std() + 1e-6
            frames = (frames - mean) / std
            if frames.ndim == 3:
                frames = frames[..., np.newaxis]
            frames = frames[..., :1]
            label = load_alignment(align_path)
            X_list.append(frames)
            Y_list.append(label)
        except Exception:
            skipped += 1

    print(f"  載入 {len(X_list)} 筆，跳過 {skipped} 筆")
    return np.array(X_list), np.array(Y_list)


def run_experiment(model_fn, X, Y):
    tf.keras.backend.clear_session()
    model = model_fn()
    n_params = model.count_params()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

    epoch_times = []

    class TimingCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self._t0 = time.time()
        def on_epoch_end(self, epoch, logs=None):
            epoch_times.append(time.time() - self._t0)

    history = model.fit(
        X, Y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[TimingCallback()],
    )
    del model
    return history.history['loss'], n_params, epoch_times


if __name__ == '__main__':
    print("=" * 60)
    print(f"載入資料（{N_SAMPLES} 筆 s1）...")
    X, Y = load_dataset(N_SAMPLES)
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")

    results = {}
    configs = [
        ('C_BN_baseline',     model_bn),
        ('E_Transformer_pilot', model_transformer),
    ]

    for name, fn in configs:
        print(f"\n{'='*60}")
        print(f"訓練模型：{name}")
        print(f"{'='*60}")
        t_start = time.time()
        losses, n_params, epoch_times = run_experiment(fn, X, Y)
        total_time = time.time() - t_start
        results[name] = dict(losses=losses, n_params=n_params, epoch_times=epoch_times, total_time=total_time)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        print(f"  參數量：{n_params:,}")
        print(f"  最終 loss：{losses[-1]:.4f}  最低 loss：{min(losses):.4f}")
        print(f"  平均每 epoch 耗時：{avg_epoch:.2f} 秒（{N_SAMPLES}筆/batch={BATCH_SIZE}）")

    print("\n" + "=" * 60)
    print("【結果總覽】")
    for name, r in results.items():
        avg_epoch = sum(r['epoch_times']) / len(r['epoch_times'])
        print(f"  {name}: 參數={r['n_params']:,}  起始loss={r['losses'][0]:.2f} → 最低loss={min(r['losses']):.2f}"
              f"（第{r['losses'].index(min(r['losses']))+1}epoch）  平均每epoch={avg_epoch:.2f}秒")

    # ── 推算完整訓練規模所需時間 ──────────────────────────────
    # 完整訓練規模參考：width05_yolov8 用 16 位說話者、10861支影片(約10800筆訓練)，
    # 60輪訓練約9.5小時 → 平均每輪約9.5*3600/60 ≈ 570秒(batch_size=2，資料量約180倍於這次pilot的60筆)
    FULL_DATASET_SIZE = 10800
    FULL_EPOCHS = 60
    print("\n【推算：若用完整16位說話者資料集(約10800筆)、60輪，需要多久？】")
    for name, r in results.items():
        avg_epoch_pilot = sum(r['epoch_times']) / len(r['epoch_times'])
        # pilot 用 batch_size=1、60筆 → 每筆耗時
        per_sample_time = avg_epoch_pilot / N_SAMPLES
        full_epoch_time = per_sample_time * FULL_DATASET_SIZE
        full_total_time = full_epoch_time * FULL_EPOCHS
        print(f"  {name}: 推算每輪約 {full_epoch_time/60:.1f} 分鐘，"
              f"{FULL_EPOCHS}輪總計約 {full_total_time/3600:.1f} 小時"
              f"（{full_total_time/3600/24:.1f} 天，24小時不間斷計算）")
