"""
LipNet V-shape 三幀輸入改進版
靈感來自 ECCV 2020: "Acquiring Dynamic Light Fields through Coded Aperture Camera"

核心改動：
  原版：每個時間步只看「當前幀」→ 輸入形狀 (75, 46, 140, 1)
  改版：每個時間步同時看「前一幀、當前幀、後一幀」→ 輸入形狀 (75, 46, 140, 3)

這讓模型能感知嘴唇的「動作方向」，而不只是靜態形狀。
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional,
    MaxPool3D, Activation, TimeDistributed, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import glob
from typing import List


# ============================================================
# 1. 全局統計量：計算與載入
# ============================================================

def compute_and_save_global_stats(cache_dir: str, stats_path: str = 'data/global_stats.npz'):
    """
    掃描 cache_dir 內所有 .npy 快取檔案，計算全資料集的 mean 與 std，並存檔。

    使用兩輪掃描（two-pass），避免一次把全部資料載入記憶體：
      第一輪：累加所有像素值 → 得到 global_mean
      第二輪：累加所有 (pixel - mean)^2 → 得到 global_std
    """
    npy_files = sorted([
        os.path.join(cache_dir, f)
        for f in os.listdir(cache_dir)
        if f.endswith('.npy')
    ])

    if not npy_files:
        raise ValueError(f"在 {cache_dir} 找不到任何 .npy 快取檔案，請先完成影片預處理。")

    print(f"找到 {len(npy_files)} 個快取檔案，開始計算全局統計量...")

    # 第一輪：計算全局平均值
    total_sum = 0.0
    total_count = 0
    for i, p in enumerate(npy_files):
        arr = np.load(p).astype(np.float64)
        total_sum += arr.sum()
        total_count += arr.size
        if (i + 1) % 100 == 0:
            print(f"  第一輪進度：{i + 1}/{len(npy_files)}")
    global_mean = total_sum / total_count

    # 第二輪：計算全局標準差
    total_sq_diff = 0.0
    for i, p in enumerate(npy_files):
        arr = np.load(p).astype(np.float64)
        total_sq_diff += ((arr - global_mean) ** 2).sum()
        if (i + 1) % 100 == 0:
            print(f"  第二輪進度：{i + 1}/{len(npy_files)}")
    global_std = float(np.sqrt(total_sq_diff / total_count)) + 1e-6

    os.makedirs(os.path.dirname(stats_path) if os.path.dirname(stats_path) else '.', exist_ok=True)
    np.savez(stats_path, mean=global_mean, std=global_std)
    print(f"全局統計量已儲存 → {stats_path}")
    print(f"  global_mean = {global_mean:.4f}")
    print(f"  global_std  = {global_std:.4f}")
    return float(global_mean), float(global_std)


def load_global_stats(stats_path: str = 'data/global_stats.npz'):
    """載入已儲存的全局統計量，回傳 (mean, std)。"""
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"找不到全局統計量檔案：{stats_path}\n"
            "請先呼叫 compute_and_save_global_stats() 產生此檔案。"
        )
    data = np.load(stats_path)
    return float(data['mean']), float(data['std'])


# ============================================================
# 2. 修改後的 load_video：回傳 3 通道（前幀、當前幀、後幀）
# ============================================================

def load_video_vshape(path: str, yolov5s, input_shape=(75, 46, 140),
                      global_mean=None, global_std=None) -> np.ndarray:
    """
    讀取影片並產生 V-shape 三幀輸入。

    原版 load_video 回傳：(75, 46, 140, 1)
    本函式回傳：         (75, 46, 140, 3)
      通道 0 = 前一幀 (t-1)
      通道 1 = 當前幀 (t)
      通道 2 = 後一幀 (t+1)

    正規化優先順序：
      1. 若有提供 global_mean / global_std → 使用全局統計量（推薦）
      2. 否則 → 退回使用該影片自身的 mean/std（舊行為）
    """
    project_name = os.path.basename(os.path.dirname(path))
    cache_dir = os.path.join('data', f'{project_name}_cached_vshape')
    file_name = os.path.splitext(os.path.basename(path))[0]
    cache_path = os.path.join(cache_dir, f'{file_name}.npy')

    if os.path.exists(cache_path):
        frames_3ch = np.load(cache_path)
        return _normalize(frames_3ch, global_mean, global_std)

    cap = cv2.VideoCapture(path)
    raw_frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        detections = yolov5s(frame)
        for det in detections.pred:
            for *xyxy, conf, cls in det:
                if int(cls) == 0 and conf >= 0.5:
                    x1, y1, x2, y2 = map(int, xyxy)
                    lip = frame[y1:y2, x1:x2]
                    lip = cv2.resize(lip, (input_shape[2], input_shape[1]))
                    lip = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
                    raw_frames.append(lip.astype(np.float32))

    cap.release()

    if len(raw_frames) == 0:
        return np.zeros((input_shape[0], input_shape[1], input_shape[2], 3), dtype=np.float32)

    # 截取或補齊到 75 幀
    T = input_shape[0]
    raw_frames = raw_frames[:T]
    while len(raw_frames) < T:
        raw_frames.append(raw_frames[-1])
    raw_frames = np.array(raw_frames)  # (75, 46, 140)

    # ── V-shape 三幀堆疊 ──────────────────────────────────────
    prev_frames = np.concatenate([raw_frames[:1], raw_frames[:-1]], axis=0)
    next_frames = np.concatenate([raw_frames[1:], raw_frames[-1:]], axis=0)
    frames_3ch = np.stack([prev_frames, raw_frames, next_frames], axis=-1)  # (75, 46, 140, 3)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, frames_3ch)  # 儲存未正規化的原始資料

    return _normalize(frames_3ch, global_mean, global_std)


def _normalize(frames_3ch: np.ndarray, global_mean, global_std) -> np.ndarray:
    """統一的正規化邏輯。"""
    if global_mean is not None and global_std is not None:
        return ((frames_3ch - global_mean) / global_std).astype(np.float32)
    mean = np.mean(frames_3ch)
    std = np.std(frames_3ch) + 1e-6
    return ((frames_3ch - mean) / std).astype(np.float32)


# ============================================================
# 3. 修改後的模型：輸入最後一維從 1 改為 3
# ============================================================

def build_lipnet_vshape(vocab_size: int) -> Sequential:
    """
    V-shape LipNet 模型。
    唯一與原版的差異：input_shape 的最後一維從 1 改成 3。
    """
    model = Sequential([
        # ── 3D 卷積區塊 ──
        Conv3D(128, 3, input_shape=(75, 46, 140, 3), padding='same'),
        Activation('relu'),
        MaxPool3D((1, 2, 2)),

        Conv3D(256, 3, padding='same'),
        Activation('relu'),
        MaxPool3D((1, 2, 2)),

        Conv3D(75, 3, padding='same'),
        Activation('relu'),
        MaxPool3D((1, 2, 2)),

        # ── 時序區塊 ──
        TimeDistributed(Flatten()),

        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
        Dropout(0.5),

        Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
        Dropout(0.5),

        Dense(vocab_size + 1, kernel_initializer='he_normal', activation='softmax'),
    ])
    return model


# ============================================================
# 4. 修改後的 load_data 與 mappable_function
# ============================================================

def load_data_vshape(path, yolov5s, char_to_num, load_alignments_fn,
                     global_mean=None, global_std=None):
    try:
        path = bytes.decode(path.numpy())
        file_name = os.path.splitext(os.path.basename(path))[0]

        parts = path.replace('\\', '/').split('/')
        parts = [p for p in parts if p]
        project_name = parts[-2]

        video_path = os.path.join('data', project_name, f'{file_name}.mpg')
        align_name = project_name.replace('_new', '')
        alignment_path = os.path.join('data', 'alignments', align_name, f'{file_name}.align')

        frames = load_video_vshape(video_path, yolov5s,
                                   global_mean=global_mean, global_std=global_std)
        alignments = load_alignments_fn(alignment_path)

    except Exception as e:
        print(f"[SKIPPED] {path} → {e}")
        with open("bad_files.txt", "a") as f:
            f.write(f"{path} → {e}\n")
        frames = np.zeros((75, 46, 140, 3), dtype=np.float32)
        alignments = np.zeros((1,), dtype=np.int32)

    return frames, alignments


def mappable_function_vshape(path, yolov5s, char_to_num, load_alignments_fn,
                              global_mean=None, global_std=None):
    features, labels = tf.py_function(
        lambda p: load_data_vshape(p, yolov5s, char_to_num, load_alignments_fn,
                                   global_mean, global_std),
        [path],
        (tf.float32, tf.int64)
    )
    features.set_shape([75, None, None, 3])
    labels.set_shape([40])
    return features, labels


# ============================================================
# 5. CTC Loss（與原版相同，無需修改）
# ============================================================

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


# ============================================================
# 6. 學習率排程（與原版相同）
# ============================================================

def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    return lr * tf.math.exp(-0.1)


# ============================================================
# 7. Checkpoint：自動儲存 + 從中途繼續訓練
# ============================================================

def make_checkpoint_callback(checkpoint_dir: str = 'checkpoints', save_every: int = 1) -> ModelCheckpoint:
    """
    建立自動儲存 callback。
    每 save_every 個 epoch 存一次，檔名包含 epoch 編號方便辨識。
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    return ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'lipnet_epoch{epoch:03d}.h5'),
        save_weights_only=False,
        save_freq='epoch',
        period=save_every,
        verbose=1,
    )


def find_latest_checkpoint(checkpoint_dir: str = 'checkpoints'):
    """
    在 checkpoint_dir 裡找最新的 .h5 檔案。
    找到就回傳路徑和已訓練的 epoch 數；找不到就回傳 (None, 0)。
    """
    files = sorted(glob.glob(os.path.join(checkpoint_dir, 'lipnet_epoch*.h5')))
    if not files:
        print(f"[Checkpoint] 在 {checkpoint_dir} 找不到任何存檔，將從頭開始訓練。")
        return None, 0

    latest = files[-1]
    # 從檔名解析 epoch 數，例如 lipnet_epoch025.h5 → 25
    basename = os.path.basename(latest)
    epoch_str = basename.replace('lipnet_epoch', '').replace('.h5', '')
    try:
        epoch_num = int(epoch_str)
    except ValueError:
        epoch_num = 0

    print(f"[Checkpoint] 找到最新存檔：{latest}（第 {epoch_num} epoch）")
    return latest, epoch_num


def load_model_from_checkpoint(checkpoint_path: str):
    """載入存檔的模型，包含權重和優化器狀態。"""
    from tensorflow.keras.models import load_model
    model = load_model(checkpoint_path, custom_objects={'CTCLoss': CTCLoss})
    print(f"[Checkpoint] 模型已從 {checkpoint_path} 載入。")
    return model


# ============================================================
# 使用說明（複製貼上到您的 Final.ipynb）
# ============================================================
"""
在您的 notebook 中，依照以下步驟操作：

【步驟一：第一次執行，產生全局統計量（只需跑一次）】

from lipnet_vshape import compute_and_save_global_stats

compute_and_save_global_stats(
    cache_dir='data/s99_7_new_cached_vshape',
    stats_path='data/global_stats.npz'
)

──────────────────────────────────────────────────

【步驟二：之後每次訓練，先載入統計量再建立 dataset】

from lipnet_vshape import (
    load_global_stats, mappable_function_vshape,
    build_lipnet_vshape, CTCLoss, scheduler,
    make_checkpoint_callback, find_latest_checkpoint, load_model_from_checkpoint
)

# 載入全局統計量
global_mean, global_std = load_global_stats('data/global_stats.npz')

# 建立 dataset
data = tf.data.Dataset.list_files('data/s99_7_new\\*.mpg')
data = data.shuffle(500, reshuffle_each_iteration=False)
data = data.map(lambda p: mappable_function_vshape(
    p, yolov5s, char_to_num, load_alignments,
    global_mean=global_mean, global_std=global_std
))
data = data.padded_batch(1, padded_shapes=([75, None, None, 3], [40]))
data = data.prefetch(tf.data.AUTOTUNE)

# ── 自動判斷：從頭訓練 or 從 checkpoint 繼續 ──────────────
checkpoint_path, initial_epoch = find_latest_checkpoint('checkpoints')

if checkpoint_path:
    # 有存檔 → 載入模型繼續訓練
    model = load_model_from_checkpoint(checkpoint_path)
else:
    # 沒有存檔 → 從頭建立模型
    model = build_lipnet_vshape(vocab_size=char_to_num.vocabulary_size())
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

# ── 開始訓練（自動每 5 個 epoch 存一次）────────────────────
checkpoint_cb = make_checkpoint_callback(checkpoint_dir='checkpoints', save_every=1)
scheduler_cb  = LearningRateScheduler(scheduler)

model.fit(
    data,
    epochs=100,
    initial_epoch=initial_epoch,   # ← 從上次中斷的地方繼續
    callbacks=[checkpoint_cb, scheduler_cb]
)
"""
