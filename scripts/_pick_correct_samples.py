import os, sys, io, glob
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.path.append(r'C:\Users\Tno\claude-code')
import numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
from vocab_correction import correct_sentence

def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype='int64')
    il = tf.cast(tf.shape(y_pred)[1], dtype='int64') * tf.ones((bl,1), dtype='int64')
    ll = tf.cast(tf.shape(y_true)[1], dtype='int64') * tf.ones((bl,1), dtype='int64')
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token='')
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token='', invert=True)

model = load_model('models/trained_model_grid_multi_width05_yolov8.h5', custom_objects={'CTCLoss': CTCLoss}, compile=False)

def load_gt(name):
    words = []
    with open(f'data/alignments/s6/{name}.align') as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and p[2] not in ('sil', 'sp') and len(p[2]) > 1:
                words.append(p[2])
    return ' '.join(words)

all_files = sorted(glob.glob('data/s6_cached_yolov8/*.npy'))
correct_names, wrong_names = [], []
for fp in all_files[:60]:
    name = os.path.splitext(os.path.basename(fp))[0]
    frames = np.load(fp)
    mean, std = np.mean(frames), np.std(frames) + 1e-6
    frames = ((frames - mean) / std).astype(np.float32)
    x = np.expand_dims(frames, 0)
    yhat = model.predict(x, verbose=0)
    dec = tf.keras.backend.ctc_decode(tf.cast(yhat, tf.float32), [75], greedy=False)[0][0].numpy()
    raw = ' '.join(w for w in tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split() if len(w) > 1)
    pred = correct_sentence(raw)
    gt = load_gt(name)
    if pred.strip() == gt.strip():
        correct_names.append((name, gt))
    else:
        wrong_names.append((name, gt, pred))
    if len(correct_names) >= 8 and len(wrong_names) >= 2:
        break

print('=== 答對(官方快取) ===')
for n, gt in correct_names:
    print(f'  {n}: "{gt}"')
print('=== 答錯(官方快取) ===')
for n, gt, pred in wrong_names:
    print(f'  {n}: gt="{gt}" pred="{pred}"')
