import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 強制不使用 GPU，模擬純 CPU（無獨立顯卡）環境
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

print('可見的裝置:', tf.config.list_physical_devices())

def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)

MODELS = [
    ('v11（原始架構，目前通話系統換掉前用的舊模型）', 'models/trained_model_grid_multi_v11.h5'),
    ('F_Width0.5（輕量化，現役模型）', 'models/trained_model_grid_multi_width05.h5'),
]

rng = np.random.default_rng(0)
dummy = rng.standard_normal((1, 75, 46, 140, 1)).astype('float32')
N = 20

results = []
for label, path in MODELS:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    model = load_model(path, custom_objects={'CTCLoss': CTCLoss}, compile=False)

    for _ in range(3):  # 暖機，排除第一次呼叫的 tracing/初始化開銷
        _ = model.predict(dummy, verbose=0)

    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        out = model.predict(dummy, verbose=0)
        tf.keras.backend.ctc_decode(out, input_length=[75], greedy=True)[0][0].numpy()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times = np.array(times)
    results.append((label, size_mb, times))
    del model
    tf.keras.backend.clear_session()

print(f'\n===== CPU 推論耗時比較（含 CTC decode，各測 {N} 次）=====')
print(f'{"模型":<45}{"檔案大小(MB)":>14}{"平均(ms)":>12}{"中位數(ms)":>12}{"最慢(ms)":>10}')
for label, size_mb, times in results:
    print(f'{label:<45}{size_mb:>14.1f}{times.mean():>12.1f}{np.median(times):>12.1f}{times.max():>10.1f}')

print()
for label, size_mb, times in results:
    print(f'{label}: 每秒約可跑 {1000/times.mean():.2f} 次辨識')
