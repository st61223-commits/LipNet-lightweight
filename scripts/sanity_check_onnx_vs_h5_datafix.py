"""
在 py39（正式環境）跑同一組 dummy 輸入，跟 tf215 那邊存的
width05_datafix ONNX 輸出比較 argmax / CTC解碼文字是否一致。
"""
import os
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)

model = load_model('models/trained_model_grid_multi_width05_datafix.h5', custom_objects={'CTCLoss': CTCLoss}, compile=False)

rng = np.random.default_rng(0)
dummy = rng.standard_normal((1, 75, 46, 140, 1)).astype('float32')

h5_out = model.predict(dummy, verbose=0)
h5_argmax = h5_out[0].argmax(axis=-1)

onnx_out = np.load('models/_onnx_datafix_dummy_output.npy')
onnx_argmax = onnx_out[0].argmax(axis=-1)

print('h5   逐幀 argmax（前30個）:', h5_argmax[:30])
print('onnx 逐幀 argmax（前30個）:', onnx_argmax[:30])

match = (h5_argmax == onnx_argmax)
print(f'\n75幀中 argmax 完全一致的幀數: {match.sum()}/75 ({match.mean()*100:.1f}%)')

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

def decode(yhat):
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
    return " ".join(
        tf.strings.reduce_join([num_to_char(w) for w in sentence]).numpy().decode('utf-8')
        for sentence in decoded
    )

print('\nh5   CTC解碼文字:', repr(decode(h5_out)))
print('onnx CTC解碼文字:', repr(decode(onnx_out)))
