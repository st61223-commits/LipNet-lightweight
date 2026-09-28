import os, sys, pathlib, warnings, io
pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
sys.path.append(r'C:\Users\Tno\claude-code')
import numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
from vocab_correction import correct_sentence

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token='')
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token='', invert=True)

def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype='int64')
    il = tf.cast(tf.shape(y_pred)[1], dtype='int64') * tf.ones((bl,1), dtype='int64')
    ll = tf.cast(tf.shape(y_true)[1], dtype='int64') * tf.ones((bl,1), dtype='int64')
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)

def _load_frames(path):
    project_name = os.path.basename(os.path.dirname(path))
    file_name = os.path.splitext(os.path.basename(path))[0]
    cache_path = os.path.join('data', f'{project_name}_cached_yolov8', f'{file_name}.npy')
    if os.path.exists(cache_path):
        frames = np.load(cache_path)
        mean, std = np.mean(frames), np.std(frames) + 1e-6
        return ((frames - mean) / std).astype(np.float32)
    raise FileNotFoundError(f'not found: {project_name}/{file_name}')

def load_alignments(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split(); word = parts[2]
        if word not in ('sil', 'sp') and len(word) > 1:
            tokens.extend([' ', word])
    tokens_flat = tokens[1:]
    if not tokens_flat:
        return tf.zeros([1], dtype=tf.int64)
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens_flat, input_encoding='UTF-8'), (-1)))

def load_data(path):
    try:
        p = bytes.decode(path.numpy())
        file_name = os.path.splitext(os.path.basename(p))[0]
        parts = [x for x in p.replace('\\\\', '/').replace('\\', '/').split('/') if x]
        project_name = parts[-2]
        frames = _load_frames(os.path.join('data', project_name, f'{file_name}.mpg'))
        alignments = load_alignments(os.path.join('data', 'alignments', project_name.replace('_new', ''), f'{file_name}.align'))
    except Exception as e:
        frames = np.zeros((75, 46, 140, 1), dtype=np.float32)
        alignments = np.zeros((1,), dtype=np.int32)
    return frames, alignments

def mappable(path):
    f, l = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    f.set_shape([75, None, None, 1]); l.set_shape([40])
    return f, l

model = load_model('models/trained_model_grid_multi_width05_yolov8.h5', custom_objects={'CTCLoss': CTCLoss}, compile=False)

ds = tf.data.Dataset.list_files([r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s6\*.mpg'], shuffle=False).map(mappable)
ds = ds.apply(tf.data.experimental.ignore_errors())
ds = ds.padded_batch(1, padded_shapes=([75, None, None, 1], [40])).prefetch(tf.data.AUTOTUNE).take(20)

ok = n = 0
for frames, labels in ds.as_numpy_iterator():
    yhat = model.predict(frames, verbose=0)
    dec = tf.keras.backend.ctc_decode(tf.cast(yhat, tf.float32), [75] * len(yhat), greedy=False)[0][0].numpy()
    orig = ' '.join(tf.strings.reduce_join(num_to_char(labels[0])).numpy().decode().strip().split())
    raw = ' '.join(w for w in tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split() if len(w) > 1)
    corr = ' '.join(correct_sentence(raw).split())
    tag = 'MATCH' if orig == corr else 'wrong'
    print(f'orig="{orig}" corr="{corr}" [{tag}]')
    if orig == corr:
        ok += 1
    n += 1
print(f'\n{ok}/{n}')
