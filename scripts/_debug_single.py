import os, sys, numpy as np, tensorflow as tf
sys.path.insert(0, r'C:\Users\Tno\claude-code')
from vocab_correction import correct_sentence
from tensorflow.keras.models import load_model
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0],'int64')
    il = tf.cast(tf.shape(y_pred)[1],'int64') * tf.ones((bl,1),dtype='int64')
    ll = tf.cast(tf.shape(y_true)[1],'int64') * tf.ones((bl,1),dtype='int64')
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token='')
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token='', invert=True)

model = load_model(r'models\trained_model_grid_multi_v8.h5', custom_objects={'CTCLoss':CTCLoss})

def predict_one(arr):
    mean = np.mean(arr); std = np.std(arr)+1e-6
    arr = ((arr-mean)/std).astype(np.float32)[:75]
    yhat = model.predict(tf.expand_dims(arr,0), verbose=0)
    decoded = tf.keras.backend.ctc_decode(tf.cast(yhat,tf.float32), input_length=[75], greedy=False)[0][0].numpy()
    raw = tf.strings.reduce_join([num_to_char(w) for w in decoded[0]]).numpy().decode('utf-8').strip()
    raw_words = ' '.join(w for w in raw.split() if len(w) > 1)
    return ' '.join(correct_sentence(raw_words).split())

VOCAB_SET = set("abcdefghijklmnopqrstuvwxyz'?!123456789 ")
def load_gt(align_path):
    with open(align_path) as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[2] not in ('sil','sp') and len(parts[2]) > 1:
            tokens.extend([' ', parts[2]])
    raw = ''.join(tokens[1:])
    filtered = ''.join(c for c in raw if c in VOCAB_SET)
    return ' '.join(filtered.split())

# 測試 s1 前 10 個檔案
print('=== GRID s1 ===')
files = sorted(os.listdir('data/s1_cached'))[:10]
correct = 0
for fname in files:
    align = fname.replace('.npy', '.align')
    align_path = f'data/alignments/s1/{align}'
    if not os.path.exists(align_path):
        continue
    arr = np.load(f'data/s1_cached/{fname}')
    pred = predict_one(arr)
    gt   = load_gt(align_path)
    match = (pred == gt)
    if match: correct += 1
    print(f'{fname}: pred={repr(pred)} | gt={repr(gt)} | match={match}')
print(f'Correct: {correct}/{len(files)}')

# 測試 s99_1 前 5 個
print()
print('=== s99_1 ===')
files99 = sorted(os.listdir('data/s99_1_cached'))[:5]
for fname in files99:
    align = fname.replace('.npy', '.align')
    align_path = f'data/alignments/s99_1/{align}'
    if not os.path.exists(align_path):
        print(f'{fname}: no align'); continue
    arr = np.load(f'data/s99_1_cached/{fname}')
    pred = predict_one(arr)
    gt   = load_gt(align_path)
    print(f'{fname}: pred={repr(pred)} | gt={repr(gt)} | match={pred==gt} | lower={pred.lower()==gt.lower()}')
