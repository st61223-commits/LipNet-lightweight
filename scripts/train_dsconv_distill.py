"""
E_DSConv 知識蒸餾訓練 — 用 F_Width0.5（老師，76.6%）教 DSConv（學生）

背景：DSConv 單純從頭訓練三次都失敗（benchmark最好只有0.8%），查文獻發現
靈感來源論文(Ma et al. 2021)的方法其實是「輕量架構＋知識蒸餾」合一才有效，
DSConv之前只做了架構那一半。這次補上蒸餾：學生不只學CTC硬標籤，也同時學
老師(F_Width0.5)每個時間點輸出的完整機率分布(軟標籤)。

Loss = alpha * CTC(硬標籤) + (1-alpha) * KL(老師機率 || 學生機率)
（不加溫度縮放，因為兩個模型最後一層都已經是softmax輸出，只有機率可用，
沒有logits，用溫度縮放需要另外重建無softmax版本，先求簡單可靠）

學生從已存的 DSConv 最佳權重(val_loss 42.42)熱啟動，不從亂數初始化重來。
沿用同一套斷電接續機制（獨立檔名，不動到舊的dsconv訓練檔案）。
"""

import os
import json
import time
import pathlib
import warnings
import numpy as np
import tensorflow as tf

pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

BASE_DIR = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet'
os.chdir(BASE_DIR)

TEACHER_PATH = os.path.join('models', 'trained_model_grid_multi_width05.h5')
STUDENT_INIT_PATH = os.path.join('models', 'trained_model_grid_multi_dsconv.h5')  # 熱啟動起點(val_loss 42.42)
SAVE_PATH = os.path.join('models', 'trained_model_grid_multi_dsconv_distill.h5')

# 獨立檔名，不動到舊的(已放棄的)DSConv單純訓練那一批 resume 檔案
RESUME_WEIGHTS = os.path.join('models', 'checkpoint_dsconv_distill_resume.weights.h5')
STATE_FILE = os.path.join('models', 'train_dsconv_distill_state.json')
DONE_FLAG = os.path.join('models', 'train_dsconv_distill_DONE.flag')

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
VOCAB_SIZE = char_to_num.vocabulary_size()

ALPHA = 0.5  # 硬標籤(CTC) vs 軟標籤(KD) 各半，之後可依訓練狀況調整
BATCH_SIZE = 1  # DSConv 架構跟基準版一樣寬，batch=2會OOM(已知教訓)，維持1


def load_video(path: str) -> np.ndarray:
    project_name = os.path.basename(os.path.dirname(path))
    file_name = os.path.splitext(os.path.basename(path))[0]
    for suffix in ['_cached', '_dlib_cached']:
        cache_path = os.path.join('data', f'{project_name}{suffix}', f'{file_name}.npy')
        if os.path.exists(cache_path):
            frames = np.load(cache_path)
            mean = np.mean(frames)
            std = np.std(frames) + 1e-6
            return ((frames - mean) / std).astype(np.float32)
    raise FileNotFoundError(f'找不到快取：{project_name}/{file_name}')


def load_alignments(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens.extend([' ', line[2]])
    tokens_flat = tokens[1:]
    return char_to_num(tf.reshape(
        tf.strings.unicode_split(tokens_flat, input_encoding='UTF-8'), (-1)
    ))


def load_data(path):
    try:
        path = bytes.decode(path.numpy())
        file_name = os.path.splitext(os.path.basename(path))[0]
        parts = path.replace('\\\\', '/').replace('\\', '/').split('/')
        parts = [p for p in parts if p]
        project_name = parts[-2]

        video_path = os.path.join('data', project_name, f'{file_name}.mpg')
        align_name = project_name.replace('_new', '')
        alignment_path = os.path.join('data', 'alignments', align_name, f'{file_name}.align')

        frames = load_video(video_path)
        alignments = load_alignments(alignment_path)
    except Exception as e:
        print(f'[SKIPPED] {path} -> {e}')
        frames = np.zeros((75, 46, 140, 1), dtype=np.float32)
        alignments = np.zeros((1,), dtype=np.int32)
    return frames, alignments


def mappable_function(path):
    features, labels = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    features.set_shape([75, None, None, 1])
    labels.set_shape([40])
    return features, labels


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


kld = tf.keras.losses.KLDivergence()


def kd_loss(teacher_probs, student_probs):
    eps = 1e-8
    t = tf.clip_by_value(teacher_probs, eps, 1.0)
    s = tf.clip_by_value(student_probs, eps, 1.0)
    return kld(t, s)


class DistillModel(tf.keras.Model):
    def __init__(self, student, teacher, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        self.alpha = alpha

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def train_step(self, data):
        x, y = data
        teacher_pred = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)
            ctc = tf.reduce_mean(CTCLoss(y, student_pred))
            kd = kd_loss(teacher_pred, student_pred)
            loss = self.alpha * ctc + (1.0 - self.alpha) * kd
        trainable_vars = self.student.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        return {'loss': loss, 'ctc_loss': ctc, 'kd_loss': kd}

    def test_step(self, data):
        x, y = data
        student_pred = self.student(x, training=False)
        ctc = tf.reduce_mean(CTCLoss(y, student_pred))
        return {'loss': ctc}


class ResumeCheckpoint(tf.keras.callbacks.Callback):
    """每隔一段時間存一次最新權重，並記錄目前跑到第幾輪，讓電腦意外重開機後可以接續訓練。"""

    def __init__(self, weights_path, state_path, save_every_sec=180):
        super().__init__()
        self.weights_path = weights_path
        self.state_path = state_path
        self.save_every_sec = save_every_sec
        self.last_save = time.time()

    def on_train_batch_end(self, batch, logs=None):
        now = time.time()
        if now - self.last_save >= self.save_every_sec:
            self.model.student.save_weights(self.weights_path)
            self.last_save = now

    def on_epoch_end(self, epoch, logs=None):
        self.model.student.save_weights(self.weights_path)
        raw_val_loss = logs.get('val_loss') if logs else None
        val_loss = float(np.array(raw_val_loss)) if raw_val_loss is not None else None
        with open(self.state_path, 'w') as f:
            json.dump({'next_epoch': epoch + 1, 'val_loss': val_loss}, f)


if __name__ == '__main__':
    print('載入老師模型 F_Width0.5（凍結，不訓練）...')
    teacher = load_model(TEACHER_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)
    teacher.trainable = False

    print('載入學生模型 DSConv（從 val_loss 42.42 的權重熱啟動）...')
    student = load_model(STUDENT_INIT_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)

    initial_epoch = 0
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        initial_epoch = state.get('next_epoch', 0)

    if os.path.exists(RESUME_WEIGHTS):
        print(f'偵測到先前中斷的蒸餾訓練進度，接續讀取學生權重（從第 {initial_epoch + 1} 輪繼續）')
        student.load_weights(RESUME_WEIGHTS)
    else:
        print('沒有中斷紀錄，學生從 DSConv 最佳權重(val_loss 42.42)開始蒸餾訓練')

    distill_model = DistillModel(student, teacher, alpha=ALPHA)
    distill_model.compile(optimizer=Adam(learning_rate=1e-4, clipnorm=1.0))

    all_patterns = [
        r'data\s1\*.mpg', r'data\s2\*.mpg', r'data\s3\*.mpg', r'data\s4\*.mpg',
        r'data\s5\*.mpg', r'data\s6\*.mpg', r'data\s7\*.mpg', r'data\s8\*.mpg',
        r'data\s13\*.mpg', r'data\s99_1\*.mpg', r'data\s99_6\*.mpg', r'data\s99_6_new\*.mpg',
        r'data\s99_7\*.mpg', r'data\s99_7_new\*.mpg', r'data\s99_8\*.mpg', r'data\s34_3\*.mpg',
    ]
    data = tf.data.Dataset.list_files(all_patterns)
    data = data.shuffle(1000, reshuffle_each_iteration=False)
    data = data.map(mappable_function)
    data = data.padded_batch(BATCH_SIZE, padded_shapes=([75, None, None, 1], [40]))
    data = data.prefetch(tf.data.AUTOTUNE)

    total = tf.data.experimental.cardinality(data).numpy()
    train_size = int(total * 0.8)
    train = data.take(train_size)
    val = data.skip(train_size)
    print(f'訓練集：{train_size} batch，驗證集：{total - train_size} batch（batch_size={BATCH_SIZE}）')

    MAX_EPOCHS = 65

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ResumeCheckpoint(RESUME_WEIGHTS, STATE_FILE, save_every_sec=180),
    ]

    print(f'\n開始知識蒸餾訓練（alpha={ALPHA}, lr=1e-4，最多 {MAX_EPOCHS} 輪）...\n')
    history = distill_model.fit(
        train,
        validation_data=val,
        epochs=MAX_EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    student.save(SAVE_PATH)
    print(f'\n學生模型已儲存：{SAVE_PATH}')

    with open(DONE_FLAG, 'w') as f:
        f.write('done')
    print('已寫入完成標記。')
