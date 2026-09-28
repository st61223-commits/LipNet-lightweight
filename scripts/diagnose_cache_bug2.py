import os, glob, traceback
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
import numpy as np
import tensorflow as tf

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

def _load_frames(path):
    project_name = os.path.basename(os.path.dirname(path))
    file_name = os.path.splitext(os.path.basename(path))[0]
    for suffix in ['_cached', '_dlib_cached']:
        cache_path = os.path.join('data', f'{project_name}{suffix}', f'{file_name}.npy')
        if os.path.exists(cache_path):
            frames = np.load(cache_path)
            mean, std = np.mean(frames), np.std(frames) + 1e-6
            return ((frames - mean) / std).astype(np.float32)
    raise FileNotFoundError(f'找不到快取：{project_name}/{file_name}')

def load_alignments(path):
    with open(path, 'r') as f: lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split(); word = parts[2]
        if word not in ('sil', 'sp') and len(word) > 1:
            tokens.extend([' ', word])
    tokens_flat = tokens[1:]
    if not tokens_flat: return tf.zeros([1], dtype=tf.int64)
    return char_to_num(tf.reshape(
        tf.strings.unicode_split(tokens_flat, input_encoding='UTF-8'), (-1)))

def load_data_for_path(p):
    file_name = os.path.splitext(os.path.basename(p))[0]
    parts = [x for x in p.replace('\\\\', '/').replace('\\', '/').split('/') if x]
    project_name = parts[-2]
    frames = _load_frames(os.path.join('data', project_name, f'{file_name}.mpg'))
    alignments = load_alignments(os.path.join(
        'data', 'alignments', project_name.replace('_new', ''), f'{file_name}.align'))
    return frames, alignments

SPEAKERS = ['s2', 's6', 's7']

for spk in SPEAKERS:
    print(f'=== {spk} ===', flush=True)
    files = sorted(glob.glob(os.path.join('data', spk, '*.mpg')))
    fail_count = 0
    for fp in files:
        try:
            frames, alignments = load_data_for_path(fp)
            f_shape = frames.shape
            a_shape = alignments.shape.as_list() if hasattr(alignments, 'shape') else None
            if f_shape != (75, 46, 140, 1):
                print(f'  [frames shape異常] {fp}: {f_shape}', flush=True)
                fail_count += 1
        except Exception as e:
            fail_count += 1
            print(f'  [例外] {fp}: {type(e).__name__}: {e}', flush=True)
            print(traceback.format_exc(), flush=True)
    print(f'  {spk} 總共有問題: {fail_count} 筆', flush=True)
