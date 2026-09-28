import os
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import tensorflow as tf
from tensorflow.keras.models import load_model

def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)

model = load_model('models/trained_model_grid_multi_width05.h5', custom_objects={'CTCLoss': CTCLoss}, compile=False)
print('輸入形狀:', model.input_shape)
print('輸出形狀:', model.output_shape)

import numpy as np
dummy = np.zeros((1, 75, 46, 140, 1), dtype='float32')
out = model.predict(dummy, verbose=0)
print('實際跑一次 predict 成功，輸出 shape:', out.shape)
decoded = tf.keras.backend.ctc_decode(out, input_length=[75], greedy=True)[0][0].numpy()
print('CTC decode 成功，decoded shape:', decoded.shape)
