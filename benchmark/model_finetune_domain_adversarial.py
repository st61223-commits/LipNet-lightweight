"""
修正版小規模驗證實驗：對抗式訓練（Domain-Adversarial Training）能不能救回
「完全沒看過的新說話者＝0%」這個問題

跟第一版（model_experiment_domain_adversarial.py）的差別：
  第一版從零開始訓練，180筆資料練25epoch，loss只降到59左右——遠遠沒到
  「模型開始有辨識能力」的門檻（對照F_Width0.5正式訓練：用全部10861筆練15epoch
  loss才到25.7，那時正確率也才3.8%）。從零練的小實驗量級，沒辦法拿來測
  「辨識正確率」，只能測loss曲線。

  這一版改成「暖啟動」：直接載入現役 F_Width0.5 模型（trained_model_grid_multi_width05.h5，
  正式benchmark 76.6%）的權重當起點，只做少量步數的微調（fine-tune），
  而不是從零訓練。這樣才能真正測「加了對抗式訓練，能不能在已經堪用的基礎上，
  把新使用者的辨識率從0%往上拉」。

三方比較（同一顆起始模型，只有「有沒有微調」「微調時有沒有加對抗式訓練」不同）：
  ① Original     ：完全不動的原始模型（基準）
  ② FT-Plain     ：用小樣本繼續微調，但沒有對抗式訓練分支
  ③ FT-Adversarial：一樣的小樣本、一樣的epoch數，但多加「猜說話者」分支+梯度反轉層(GRL)

  ①②的差異：純粹微調本身有沒有效果
  ②③的差異：加對抗式訓練有沒有比單純微調更好 ← 這是我們真正想知道的答案

測試對象：
  - s99_3 + s99_5：現役模型正式benchmark紀錄是「完全沒訓練過、正確率0%」的兩位
    真正的新說話者，這才是「換新使用者」的真實測試
  - s6（額外抽一批「微調時沒用過」的樣本）：安全檢查，確認微調有沒有把
    模型原本學得很好的東西（s6現役97.2%）弄壞

參考文獻：Wand & Schmidhuber (2017), Interspeech.
  "Improving Speaker-Independent Lipreading with Domain-Adversarial Training."
  https://arxiv.org/abs/1708.01565

使用方式：
  python model_finetune_domain_adversarial.py
  結果存成 experiment_finetune_domain_adversarial_results.csv
"""

import os
import csv
import gc
import glob
import random
import time
import numpy as np
import tensorflow as tf

for _gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

import sys
sys.path.append(r'C:\Users\Tno\claude-code')
from vocab_correction import correct_sentence

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam

# ============================================================
# ★ 設定 ★
# ============================================================
DATA_ROOT       = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data'
MODEL_PATH      = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\trained_model_grid_multi_width05.h5'

# 微調用的資料：6位已訓練過的說話者，各抽一批，標上說話者ID（給對抗式分支用）
# 第一次小規模驗證（50筆/8epoch/lr=1e-5）對抗式訓練沒有效果，但地板效應+訓練量太保守
# 都有可能是原因，這次放大規模再給一次公平機會：樣本數x3、epoch數x2.5、學習率x3
FT_SPEAKERS         = [('s1', 0), ('s2', 1), ('s5', 2), ('s6', 3), ('s7', 4), ('s13', 5)]
N_FT_PER_SPEAKER    = 150

# 真正的測試對象：完全沒訓練過的新說話者
HELDOUT_DIRS         = ['s99_3', 's99_5']
N_HELDOUT_PER_DIR    = 20

# 安全檢查：s6 額外抽一批「微調沒用過」的樣本，確認沒有把已經學會的東西弄壞
REGRESSION_SPEAKER   = 's6'
N_REGRESSION         = 30   # 從 s6 抽 N_FT_PER_SPEAKER + N_REGRESSION 筆，前面拿去微調，後面這些純粹拿來檢查

EPOCHS      = 20        # 暖啟動微調，第一次8epoch可能太少，放大到20
BATCH_SIZE  = 1
LR          = 3e-5      # 第一次1e-5可能太保守幾乎沒移動，略為提高但仍遠低於從零訓練的1e-4
SEED        = 42
RESULT_CSV  = 'experiment_finetune_domain_adversarial_results.csv'
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

VOCAB = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=VOCAB, oov_token='')
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token='', invert=True)
VOCAB_SIZE = char_to_num.vocabulary_size()


def CTCLoss(y_true, y_pred):
    batch_len    = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


class GradReversal(tf.keras.layers.Layer):
    """前向：原封不動。反向：梯度乘上 -lamb 再往回傳。"""
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


class LambdaScheduler(tf.keras.callbacks.Callback):
    """lamb 從 0 緩緩升到 1（Ganin & Lempitsky 2015 的標準排程）"""
    def __init__(self, grl_layer, total_epochs):
        super().__init__()
        self.grl_layer = grl_layer
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        p = epoch / max(1, self.total_epochs - 1)
        lamb = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        self.grl_layer.lamb.assign(lamb)
        print(f"    [epoch {epoch+1}] GRL lambda = {lamb:.3f}")


# ── 對齊檔載入（固定長度訓練版 + 原始文字評估版）──────────────
def load_alignment_fixed(path, max_len=40):
    """訓練用固定長度標籤——照訓練腳本(train_grid_multi_width05.py)的規則，
    只濾掉'sil'，單一字母(例如GRID句子裡的字母格e/a/b...)照樣保留，
    因為模型訓練時就是照這個規則學的。"""
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


def load_alignment_eval_text(path):
    """評分用答案文字——照官方 benchmark_width05.py 的規則，濾掉'sil'/'sp'跟
    單一字母(len<=1)，因為模型對GRID句子裡那個單一字母格本來就辨識不穩，
    官方公佈的76.6%/97.2%等數字都是不計較這個字母對不對算出來的。
    評分要跟官方數字站在同一把尺上比較，不能自己另外訂更嚴格的標準。"""
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.strip().split()
        word = parts[2]
        if word not in ('sil', 'sp') and len(word) > 1:
            tokens.append(word)
    return ' '.join(tokens)


def _load_one(cache_dir, align_dir, stem):
    frames = np.load(os.path.join(cache_dir, stem + '.npy')).astype(np.float32)
    mean, std = frames.mean(), frames.std() + 1e-6
    frames = (frames - mean) / std
    if frames.ndim == 3:
        frames = frames[..., np.newaxis]
    frames = frames[..., :1]
    align_path = os.path.join(align_dir, stem + '.align')
    label, _ = load_alignment_fixed(align_path)          # 訓練用固定長度標籤（含字母）
    eval_text = load_alignment_eval_text(align_path)      # 評分用答案文字（濾掉字母，跟官方benchmark同標準）
    return frames, label, eval_text


def load_speaker_samples(speaker_name, n_samples, speaker_id, skip_first=0):
    """載入某位說話者的 n_samples 筆資料。skip_first 用來跟其他用途的樣本錯開，避免重複用到同一筆。"""
    cache_dir = os.path.join(DATA_ROOT, f'{speaker_name}_cached')
    align_dir = os.path.join(DATA_ROOT, 'alignments', speaker_name)
    npy_files = sorted(glob.glob(os.path.join(cache_dir, '*.npy')))
    random.Random(SEED).shuffle(npy_files)   # 固定 shuffle，這樣 skip_first 才能穩定錯開
    npy_files = npy_files[skip_first: skip_first + n_samples]

    X_list, Y_list, TXT_list = [], [], []
    skipped = 0
    for p in npy_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        align_path = os.path.join(align_dir, stem + '.align')
        if not os.path.exists(align_path):
            skipped += 1
            continue
        try:
            frames, label, text = _load_one(cache_dir, align_dir, stem)
            X_list.append(frames); Y_list.append(label); TXT_list.append(text)
        except Exception:
            skipped += 1
    print(f"  {speaker_name}（第{skip_first}~{skip_first+n_samples}筆）：載入 {len(X_list)} 筆，跳過 {skipped} 筆")
    X = np.array(X_list)
    Y = np.array(Y_list)
    SID = np.full(len(X_list), speaker_id, dtype=np.int64)
    return X, Y, SID, TXT_list


def evaluate_accuracy(model, X, TXT, is_adversarial):
    ok = 0
    n = 0
    for i in range(len(X)):
        x = X[i:i + 1]
        pred = model.predict(x, verbose=0)
        yhat = pred[0] if is_adversarial else pred
        dec = tf.keras.backend.ctc_decode(
            tf.cast(yhat, tf.float32), [yhat.shape[1]], greedy=True)[0][0].numpy()
        raw = ' '.join(
            w for w in tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split()
            if len(w) > 1)   # 濾掉單一字母，跟官方benchmark_width05.py同標準
        corr = ' '.join(correct_sentence(raw).split())
        orig = ' '.join(TXT[i].split())
        if orig == corr:
            ok += 1
        n += 1
    return ok, n


def freeze_batchnorm(model):
    """把模型裡所有BatchNormalization層設成trainable=False。

    原因：這次微調batch_size被迫只能設1（GPU只有6GB），但BatchNorm本來的設計是
    「用整批資料的平均值/變異數做正規化」，batch_size=1時這個「一批的統計量」
    其實就是「這一筆資料自己的統計量」，每一步都在劇烈跳動，會讓整個微調過程
    很不穩定。凍結成暖啟動模型原本（用正常batch size訓練出來）的統計量，
    等於是說「正規化方式維持原本學好的樣子，只微調其他權重」，減少雜訊。"""
    count = 0
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
            count += 1
    return count


def build_adversarial_from_loaded(loaded_model, num_speakers):
    """重用已載入模型的每一層（含訓練好的權重），接出 GRL+猜說話者分支"""
    inp = tf.keras.Input(shape=(75, 46, 140, 1))
    x = inp
    for layer in loaded_model.layers[:-1]:   # 到 dropout_1（index 16）為止，共用特徵
        x = layer(x)
    feat = x
    ctc_out = loaded_model.layers[-1](feat)  # 重用原本訓練好的最後一層 Dense

    # 改用「最大值池化」而非「平均池化」：平均會把75幀裡每一幀的說話者線索
    # 稀釋掉（尤其嘴巴在講不同字時形狀差很多，平均起來可能什麼都不像），
    # 改成取每個特徵維度上「整段影片裡最強烈的那個值」，比較容易保留
    # 「這個人嘴型/臉部特徵」這種在少數幾幀就很明顯的線索。
    pooled = tf.keras.layers.GlobalMaxPooling1D(name='speaker_gmp')(feat)
    grl = GradReversal(name='grl')
    rev = grl(pooled)
    # 明確給名字，避免跟重用的舊層(裡面已經有一層固定叫"dense")撞名——
    # clear_session()會重置Keras的自動命名計數器，但重用的舊層名字是存檔時就固定的，
    # 不會被重置，所以這裡新建的層一定要手動取名，不能依賴自動命名。
    s = Dense(64, activation='relu', name='speaker_hidden')(rev)
    speaker_out = Dense(num_speakers, activation='softmax', name='speaker_out')(s)

    model = tf.keras.Model(inp, [ctc_out, speaker_out], name='FT_Adversarial')
    # 重用的舊層名字是存檔當下就固定的（這個模型存的是"dense"，不是"ctc_out"），
    # clear_session()不會幫忙改名，所以compile/fit時必須用「這一層真正的名字」，
    # 不能假設它叫ctc_out——用變數帶著這個真實名字，避免寫死字串又對不上。
    ctc_out_name = loaded_model.layers[-1].name
    return model, grl, ctc_out_name


# ── 主程式 ───────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("載入微調用資料（6位已訓練過的說話者）...")
    print("=" * 70)
    Xs, Ys, SIDs = [], [], []
    for name, sid in FT_SPEAKERS:
        X, Y, SID, _ = load_speaker_samples(name, N_FT_PER_SPEAKER, sid, skip_first=0)
        Xs.append(X); Ys.append(Y); SIDs.append(SID)
    X_ft = np.concatenate(Xs, axis=0)
    Y_ft = np.concatenate(Ys, axis=0)
    SID_ft = np.concatenate(SIDs, axis=0)
    perm = np.random.permutation(len(X_ft))
    X_ft, Y_ft, SID_ft = X_ft[perm], Y_ft[perm], SID_ft[perm]
    print(f"微調資料總計：{len(X_ft)} 筆")

    print("\n" + "=" * 70)
    print("載入測試資料：s99_3+s99_5（真正沒訓練過的新說話者）...")
    print("=" * 70)
    Xh, Yh, _, TXTh = [], [], [], []
    for name in HELDOUT_DIRS:
        X, Y, SID, TXT = load_speaker_samples(name, N_HELDOUT_PER_DIR, speaker_id=-1, skip_first=0)
        Xh.append(X); TXTh.extend(TXT)
    X_held = np.concatenate(Xh, axis=0)
    print(f"held-out 測試集：{len(X_held)} 筆")

    print("\n" + "=" * 70)
    print(f"載入安全檢查資料：{REGRESSION_SPEAKER}（微調沒用過的另一批，確認沒被練壞）...")
    print("=" * 70)
    X_reg, Y_reg, _, TXT_reg = load_speaker_samples(
        REGRESSION_SPEAKER, N_REGRESSION, speaker_id=-1, skip_first=N_FT_PER_SPEAKER)
    print(f"安全檢查測試集：{len(X_reg)} 筆")

    results = {}

    # ── ① Original：完全不動的原始模型 ─────────────────────────
    print("\n" + "=" * 70)
    print("① 測試 Original（現役模型，完全不微調）")
    print("=" * 70)
    tf.keras.backend.clear_session(); gc.collect()
    m_orig = load_model(MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)
    ok, n = evaluate_accuracy(m_orig, X_held, TXTh, is_adversarial=False)
    reg_ok, reg_n = evaluate_accuracy(m_orig, X_reg, TXT_reg, is_adversarial=False)
    results['Original'] = {'held_ok': ok, 'held_n': n, 'reg_ok': reg_ok, 'reg_n': reg_n, 'train_time': 0}
    print(f"  held-out(s99_3+5)：{ok}/{n} = {ok/n*100:.1f}%　{REGRESSION_SPEAKER}安全檢查：{reg_ok}/{reg_n} = {reg_ok/reg_n*100:.1f}%")

    # ── ② FT-Plain：一般微調，不加對抗式訓練 ────────────────────
    print("\n" + "=" * 70)
    print("② 訓練 FT-Plain（暖啟動，微調但不加對抗式訓練）")
    print("=" * 70)
    tf.keras.backend.clear_session(); gc.collect()
    m_plain = load_model(MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)
    n_bn = freeze_batchnorm(m_plain)
    print(f"  已凍結 {n_bn} 層 BatchNormalization（batch_size=1時統計量不穩定，維持暖啟動時學好的統計量）")
    m_plain.compile(optimizer=Adam(learning_rate=LR), loss=CTCLoss)
    t0 = time.time()
    m_plain.fit(X_ft, Y_ft, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    t_plain = time.time() - t0
    ok, n = evaluate_accuracy(m_plain, X_held, TXTh, is_adversarial=False)
    reg_ok, reg_n = evaluate_accuracy(m_plain, X_reg, TXT_reg, is_adversarial=False)
    results['FT-Plain'] = {'held_ok': ok, 'held_n': n, 'reg_ok': reg_ok, 'reg_n': reg_n, 'train_time': t_plain}
    print(f"  held-out(s99_3+5)：{ok}/{n} = {ok/n*100:.1f}%　{REGRESSION_SPEAKER}安全檢查：{reg_ok}/{reg_n} = {reg_ok/reg_n*100:.1f}%")

    # ── ③ FT-Adversarial：微調 + 對抗式訓練 ─────────────────────
    print("\n" + "=" * 70)
    print("③ 訓練 FT-Adversarial（暖啟動，微調 + 對抗式訓練/GRL）")
    print("=" * 70)
    tf.keras.backend.clear_session(); gc.collect()
    m_base_for_adv = load_model(MODEL_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)
    n_bn = freeze_batchnorm(m_base_for_adv)
    print(f"  已凍結 {n_bn} 層 BatchNormalization（跟FT-Plain同樣處理，公平比較）")
    m_adv, grl_layer, ctc_out_name = build_adversarial_from_loaded(m_base_for_adv, num_speakers=len(FT_SPEAKERS))
    print(f"  （重用的舊層真實名稱：{ctc_out_name}，compile/fit要用這個名字對應）")
    # speaker_out權重提高到3.0：第一次1.0時抓臉老師學得太慢太弱(loss只降到1.55左右)，
    # 加重這條loss的比重，讓抓臉老師有更強的訊號趕快變厲害，GRL反轉的梯度才有意義。
    m_adv.compile(
        optimizer=Adam(learning_rate=LR),
        loss={ctc_out_name: CTCLoss, 'speaker_out': 'sparse_categorical_crossentropy'},
        loss_weights={ctc_out_name: 1.0, 'speaker_out': 3.0},
    )
    t0 = time.time()
    m_adv.fit(
        X_ft, {ctc_out_name: Y_ft, 'speaker_out': SID_ft},
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
        callbacks=[LambdaScheduler(grl_layer, EPOCHS)],
    )
    t_adv = time.time() - t0
    ok, n = evaluate_accuracy(m_adv, X_held, TXTh, is_adversarial=True)
    reg_ok, reg_n = evaluate_accuracy(m_adv, X_reg, TXT_reg, is_adversarial=True)
    results['FT-Adversarial'] = {'held_ok': ok, 'held_n': n, 'reg_ok': reg_ok, 'reg_n': reg_n, 'train_time': t_adv}
    print(f"  held-out(s99_3+5)：{ok}/{n} = {ok/n*100:.1f}%　{REGRESSION_SPEAKER}安全檢查：{reg_ok}/{reg_n} = {reg_ok/reg_n*100:.1f}%")

    # ── 結果總覽 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("【結果總覽】")
    print("=" * 70)
    print(f"{'模型':<16}{'held-out(s99_3+5)':>20}{f'{REGRESSION_SPEAKER}安全檢查':>18}{'耗時(秒)':>10}")
    for name, r in results.items():
        held_pct = r['held_ok'] / r['held_n'] * 100 if r['held_n'] else 0
        reg_pct = r['reg_ok'] / r['reg_n'] * 100 if r['reg_n'] else 0
        print(f"{name:<16}{held_pct:>16.1f}% ({r['held_ok']}/{r['held_n']}){reg_pct:>12.1f}% ({r['reg_ok']}/{r['reg_n']}){r['train_time']:>10.0f}")

    orig_held = results['Original']['held_ok'] / results['Original']['held_n'] * 100
    plain_held = results['FT-Plain']['held_ok'] / results['FT-Plain']['held_n'] * 100
    adv_held = results['FT-Adversarial']['held_ok'] / results['FT-Adversarial']['held_n'] * 100
    print(f"\n★ Original → FT-Plain：{plain_held - orig_held:+.1f}%（單純微調的效果）")
    print(f"★ FT-Plain → FT-Adversarial：{adv_held - plain_held:+.1f}%（加對抗式訓練的額外效果，這是我們要看的重點）")

    orig_reg = results['Original']['reg_ok'] / results['Original']['reg_n'] * 100
    adv_reg = results['FT-Adversarial']['reg_ok'] / results['FT-Adversarial']['reg_n'] * 100
    if adv_reg < orig_reg - 15:
        print(f"⚠ 警告：FT-Adversarial 在 {REGRESSION_SPEAKER} 上的安全檢查掉了 {orig_reg-adv_reg:.1f}%，可能微調把原本學會的東西弄壞了，結果需謹慎解讀")

    with open(RESULT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['模型', 'heldout_ok', 'heldout_n', 'heldout_%', 'regression_ok', 'regression_n', 'regression_%', '訓練耗時(秒)'])
        for name, r in results.items():
            writer.writerow([
                name, r['held_ok'], r['held_n'], round(r['held_ok']/r['held_n']*100, 1),
                r['reg_ok'], r['reg_n'], round(r['reg_ok']/r['reg_n']*100, 1),
                round(r['train_time'], 1),
            ])
    print(f"\n結果數據表已儲存：{RESULT_CSV}")
