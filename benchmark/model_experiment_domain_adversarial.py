"""
小規模驗證實驗：對抗式訓練（Domain-Adversarial Training）能不能改善「換新說話者」辨識率

背景：
  目前現役模型 F_Width0.5（正式benchmark 76.6%）對完全沒訓練過的新說話者
  （s99_3+5）正確率是 0%，對有訓練過但學不好的說話者（GRID s2）也只有 9~13%。
  查文獻後找到「對抗式訓練 / 說話者不變訓練（Domain-Adversarial / Speaker-Invariant
  Training）」是專門對付這個問題的技術。

原理（白話）：
  正常訓練只有一個學生（辨識老師）在學「這段嘴型在講什麼字」。
  這個技術多加一個學生（抓臉老師），負責學「這是誰在講話」。
  然後對共用的中間特徵動手腳：訓練時故意讓這個中間特徵沒辦法讓抓臉老師猜出是誰
  （用「梯度反轉層 Gradient Reversal Layer, GRL」做到這件事——抓臉老師自己
  正常學習，但往回傳給共用特徵層的梯度方向被反過來，等於在教共用層「变得更難被
  猜出身分」）。逼模型被迫學到「不管是誰講、嘴型一樣就辨識出同一個字」的通用
  特徵，而不是偷懶記住訓練時那幾個人的臉/嘴型。

  重要：抓臉老師這個分支只有訓練時存在，訓練完就丟掉不用，部署的模型完全不變、
  推論速度不受影響——跟目前拚效能的方向不衝突。

參考文獻：
  Wand, M., & Schmidhuber, J. (2017). "Improving Speaker-Independent Lipreading
  with Domain-Adversarial Training." Interspeech 2017.
  https://arxiv.org/abs/1708.01565
  （論文報告：換一個新說話者，只用15~20秒「不用標籤」的新資料去適應，
    正確率可提升約40%）

實驗設計：
  - 訓練資料：s1 + s6 + s7 三位說話者，各取 60 筆（共 180 筆），標上說話者ID(0/1/2)
  - s13（40筆）從頭到尾都不出現在訓練資料裡，只在訓練完後拿來測「換新的人」表現
  - 比較兩個模型（骨幹架構完全相同，都是 F_Width0.5 那套 BN+寬度縮放骨幹）：
      A. Baseline    ：只有 CTC 辨識這一個輸出，正常訓練
      B. Adversarial ：多一個「猜說話者」分支 + 梯度反轉層(GRL)
    兩者的 epoch 數、batch size、訓練資料完全相同，唯一差異是有沒有加對抗式訓練
  - 訓練完後，兩個模型都對 s13 做辨識測試，比較正確率誰高

這是小規模驗證（180筆訓練、25epoch），目的是快速確認這個方向有沒有潛力，
不是正式訓練。如果 Adversarial 版本在 s13 上明顯贏過 Baseline，才值得投入
完整資料集的正式訓練。

使用方式：
  python model_experiment_domain_adversarial.py
  結果：印在畫面上 + 存成 experiment_domain_adversarial_results.csv
"""

import os
import csv
import gc
import glob
import random
import time
import numpy as np
import tensorflow as tf

# GPU 記憶體「用多少要多少」，不要一開始就把整張卡的記憶體全部鎖住。
# 這台筆電 GPU 只有 6GB，還有其他軟體在用，一定要開這個才不容易 OOM。
for _gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

import sys
sys.path.append(r'C:\Users\Tno\claude-code')
from vocab_correction import correct_sentence

from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional,
    MaxPool3D, Activation, TimeDistributed, Flatten,
    BatchNormalization, GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam

# ============================================================
# ★ 設定 ★
# ============================================================
DATA_ROOT           = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data'
TRAIN_SPEAKERS       = [('s1', 0), ('s6', 1), ('s7', 2)]   # (資料夾名稱, 說話者ID)
HELDOUT_SPEAKER      = 's13'                                # 全程不出現在訓練資料裡
N_TRAIN_PER_SPEAKER  = 60
N_HELDOUT            = 40
EPOCHS               = 25
BATCH_SIZE           = 1     # GPU 只剩約3.5GB可用，2會OOM
WIDTH_MULT           = 0.5   # 跟現役 F_Width0.5 同樣的通道縮減倍率
SEED                 = 42
RESULT_CSV           = 'experiment_domain_adversarial_results.csv'
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

VOCAB = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=VOCAB, oov_token='')
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token='', invert=True)
VOCAB_SIZE = char_to_num.vocabulary_size()


# ── CTC Loss（跟 model_experiment.py / benchmark 系列一致）───────
def CTCLoss(y_true, y_pred):
    batch_len    = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


# ── 梯度反轉層（Gradient Reversal Layer, GRL）─────────────────
# 前向傳播：什麼都不做，原封不動把輸入傳出去。
# 反向傳播：把梯度乘上 -lamb 再往回傳，等於教前面的層「往讓抓臉老師更猜不出來」的
#           方向走，而不是「讓抓臉老師更容易猜對」的方向。
# lamb 用 tf.Variable 存，讓訓練過程中可以透過 callback 動態調整（見下方 LambdaScheduler）。
class GradReversal(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lamb = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='grl_lambda')

    def call(self, x):
        @tf.custom_gradient
        def _reverse(x):
            def grad(dy):
                return -self.lamb * dy
            return x, grad
        return _reverse(x)


# lamb 排程：訓練一開始 lamb=0（先讓辨識學一下基本功），之後逐漸升到 1。
# 這是 Ganin & Lempitsky (2015) domain-adversarial 論文提出的標準排程公式，
# 用意是避免一開始就用很強的反向梯度把還沒學好的骨幹弄亂。
class LambdaScheduler(tf.keras.callbacks.Callback):
    def __init__(self, grl_layer, total_epochs):
        super().__init__()
        self.grl_layer = grl_layer
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        p = epoch / max(1, self.total_epochs - 1)
        lamb = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        self.grl_layer.lamb.assign(lamb)
        print(f"    [epoch {epoch+1}] GRL lambda = {lamb:.3f}")


# ── 共用骨幹（跟現役 F_Width0.5 相同架構：BN + 通道數縮減0.5倍）───
def build_backbone(width_mult=WIDTH_MULT):
    c1 = max(8, int(128 * width_mult))
    c2 = max(8, int(256 * width_mult))
    c3 = max(8, int(75 * width_mult))
    inp = tf.keras.Input(shape=(75, 46, 140, 1))
    x = Conv3D(c1, 3, padding='same')(inp); x = BatchNormalization()(x); x = Activation('relu')(x); x = MaxPool3D((1, 2, 2))(x)
    x = Conv3D(c2, 3, padding='same')(x);   x = BatchNormalization()(x); x = Activation('relu')(x); x = MaxPool3D((1, 2, 2))(x)
    x = Conv3D(c3, 3, padding='same')(x);   x = BatchNormalization()(x); x = Activation('relu')(x); x = MaxPool3D((1, 2, 2))(x)
    x = TimeDistributed(Flatten())(x)
    x = Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True))(x)
    feat = Dropout(0.5)(x)   # (75, 256)，這是共用特徵，兩個模型的辨識輸出都從這裡接出去
    return inp, feat


def build_baseline_model():
    """A. Baseline：只有 CTC 辨識輸出，正常訓練，沒有任何對抗式訓練的東西"""
    inp, feat = build_backbone()
    ctc_out = Dense(VOCAB_SIZE + 1, kernel_initializer='he_normal', activation='softmax', name='ctc_out')(feat)
    model = tf.keras.Model(inp, ctc_out, name='Baseline')
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)
    return model, None


def build_adversarial_model(num_speakers):
    """B. Adversarial：CTC 辨識輸出 + 猜說話者輸出(經過GRL)，兩個loss一起訓練"""
    inp, feat = build_backbone()
    ctc_out = Dense(VOCAB_SIZE + 1, kernel_initializer='he_normal', activation='softmax', name='ctc_out')(feat)

    pooled = GlobalAveragePooling1D()(feat)          # (75,256) → (256,)，把時間軸壓掉，抓整段的「是誰」
    grl = GradReversal(name='grl')
    reversed_feat = grl(pooled)
    s = Dense(64, activation='relu')(reversed_feat)
    speaker_out = Dense(num_speakers, activation='softmax', name='speaker_out')(s)

    model = tf.keras.Model(inp, [ctc_out, speaker_out], name='Adversarial')
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={'ctc_out': CTCLoss, 'speaker_out': 'sparse_categorical_crossentropy'},
        loss_weights={'ctc_out': 1.0, 'speaker_out': 1.0},
    )
    return model, grl


# ── 載入對齊檔：固定長度版（訓練用）+ 原始文字版（評估用）──────
def load_alignment_fixed(path, max_len=40):
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3 and parts[2] != 'sil':
            tokens.append(parts[2])
    text = ' '.join(tokens)
    nums = char_to_num(tf.strings.unicode_split(text, 'UTF-8'))
    nums = nums[:max_len]
    pad = tf.zeros([max_len - tf.shape(nums)[0]], dtype=tf.int64)
    return tf.concat([nums, pad], axis=0).numpy(), text


# ── 載入單一說話者的資料 ─────────────────────────────────────
def load_speaker_dataset(speaker_name, n_samples, speaker_id):
    cache_dir = os.path.join(DATA_ROOT, f'{speaker_name}_cached')
    align_dir = os.path.join(DATA_ROOT, 'alignments', speaker_name)
    npy_files = sorted(glob.glob(os.path.join(cache_dir, '*.npy')))
    random.shuffle(npy_files)
    npy_files = npy_files[:n_samples]

    X_list, Y_list, TXT_list = [], [], []
    skipped = 0
    for p in npy_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        align_path = os.path.join(align_dir, stem + '.align')
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

            label, text = load_alignment_fixed(align_path)
            X_list.append(frames)
            Y_list.append(label)
            TXT_list.append(text)
        except Exception:
            skipped += 1

    print(f"  {speaker_name}：載入 {len(X_list)} 筆，跳過 {skipped} 筆")
    X = np.array(X_list)
    Y = np.array(Y_list)
    SID = np.full(len(X_list), speaker_id, dtype=np.int64)
    return X, Y, SID, TXT_list


# ── 用 CTC decode 算辨識正確率（貼合 GRID 詞彙校正，跟 benchmark 腳本一致）
def evaluate_accuracy(model, X, TXT, is_adversarial):
    ok = 0
    n = 0
    for i in range(len(X)):
        x = X[i:i + 1]
        pred = model.predict(x, verbose=0)
        yhat = pred[0] if is_adversarial else pred   # adversarial模型回傳[ctc_out, speaker_out]
        dec = tf.keras.backend.ctc_decode(
            tf.cast(yhat, tf.float32), [yhat.shape[1]], greedy=True)[0][0].numpy()
        raw = ' '.join(
            w for w in tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split()
            if len(w) > 0)
        corr = ' '.join(correct_sentence(raw).split())
        orig = ' '.join(TXT[i].split())
        if orig == corr:
            ok += 1
        n += 1
    return ok, n


# ── 主程式 ───────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("載入訓練資料（s1 + s6 + s7，各取樣，標上說話者ID）...")
    print("=" * 70)
    Xs, Ys, SIDs = [], [], []
    for name, sid in TRAIN_SPEAKERS:
        X, Y, SID, _ = load_speaker_dataset(name, N_TRAIN_PER_SPEAKER, sid)
        Xs.append(X); Ys.append(Y); SIDs.append(SID)
    X_train = np.concatenate(Xs, axis=0)
    Y_train = np.concatenate(Ys, axis=0)
    SID_train = np.concatenate(SIDs, axis=0)

    perm = np.random.permutation(len(X_train))
    X_train, Y_train, SID_train = X_train[perm], Y_train[perm], SID_train[perm]
    print(f"訓練集總計：{len(X_train)} 筆，說話者ID分布：{np.bincount(SID_train)}")

    print("\n" + "=" * 70)
    print(f"載入保留測試說話者（{HELDOUT_SPEAKER}，全程不參與訓練）...")
    print("=" * 70)
    X_held, Y_held, _, TXT_held = load_speaker_dataset(HELDOUT_SPEAKER, N_HELDOUT, speaker_id=-1)
    print(f"保留測試集：{len(X_held)} 筆")

    results = {}

    # ── A. Baseline ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("訓練 A. Baseline（正常訓練，無對抗式訓練）")
    print("=" * 70)
    tf.keras.backend.clear_session(); gc.collect()
    baseline_model, _ = build_baseline_model()
    baseline_model.summary(line_length=80)
    t0 = time.time()
    hist_base = baseline_model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    base_train_time = time.time() - t0

    print(f"\n測試 Baseline 對 {HELDOUT_SPEAKER}（沒看過的人）的辨識率...")
    base_ok, base_n = evaluate_accuracy(baseline_model, X_held, TXT_held, is_adversarial=False)
    base_acc = base_ok / base_n * 100 if base_n else 0
    print(f"  → Baseline 對 {HELDOUT_SPEAKER}：{base_ok}/{base_n} = {base_acc:.1f}%")

    results['Baseline'] = {
        'loss_history': hist_base.history['loss'],
        'heldout_ok': base_ok, 'heldout_n': base_n, 'heldout_acc': base_acc,
        'train_time_sec': base_train_time,
    }

    # ── B. Adversarial ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("訓練 B. Adversarial（加對抗式訓練 / GRL）")
    print("=" * 70)
    tf.keras.backend.clear_session(); gc.collect()
    adv_model, grl_layer = build_adversarial_model(num_speakers=len(TRAIN_SPEAKERS))
    adv_model.summary(line_length=80)
    t0 = time.time()
    hist_adv = adv_model.fit(
        X_train,
        {'ctc_out': Y_train, 'speaker_out': SID_train},
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
        callbacks=[LambdaScheduler(grl_layer, EPOCHS)],
    )
    adv_train_time = time.time() - t0

    print(f"\n測試 Adversarial 對 {HELDOUT_SPEAKER}（沒看過的人）的辨識率...")
    adv_ok, adv_n = evaluate_accuracy(adv_model, X_held, TXT_held, is_adversarial=True)
    adv_acc = adv_ok / adv_n * 100 if adv_n else 0
    print(f"  → Adversarial 對 {HELDOUT_SPEAKER}：{adv_ok}/{adv_n} = {adv_acc:.1f}%")

    results['Adversarial'] = {
        'loss_history': hist_adv.history['ctc_out_loss'],
        'heldout_ok': adv_ok, 'heldout_n': adv_n, 'heldout_acc': adv_acc,
        'train_time_sec': adv_train_time,
    }

    # ── 結果總覽 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("【結果總覽：對完全沒看過的新說話者（{}）的辨識率】".format(HELDOUT_SPEAKER))
    print("=" * 70)
    for name, r in results.items():
        losses = r['loss_history']
        print(f"  {name:12s}: 最低CTC loss={min(losses):.2f}  "
              f"{HELDOUT_SPEAKER}正確率={r['heldout_acc']:.1f}%（{r['heldout_ok']}/{r['heldout_n']}）  "
              f"訓練耗時={r['train_time_sec']:.0f}秒")

    diff = results['Adversarial']['heldout_acc'] - results['Baseline']['heldout_acc']
    diff_s = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
    print(f"\n★ Adversarial 相較 Baseline，在完全沒看過的 {HELDOUT_SPEAKER} 上：{diff_s}")
    if diff > 0:
        print("  → 有潛力！對抗式訓練方向值得投入正式的完整資料集訓練。")
    else:
        print("  → 這次小規模測試沒看到優勢，可能需要更多訓練說話者、調整lamb排程，或資料量太小訊號不足。")

    # ── 存成 CSV ─────────────────────────────────────────────
    with open(RESULT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['模型', '最低CTC_loss', f'{HELDOUT_SPEAKER}_正確率(%)',
                          f'{HELDOUT_SPEAKER}_答對筆數', f'{HELDOUT_SPEAKER}_總筆數', '訓練耗時(秒)'])
        for name, r in results.items():
            writer.writerow([
                name, round(min(r['loss_history']), 4), round(r['heldout_acc'], 1),
                r['heldout_ok'], r['heldout_n'], round(r['train_time_sec'], 1),
            ])
    print(f"\n結果數據表已儲存：{RESULT_CSV}")
