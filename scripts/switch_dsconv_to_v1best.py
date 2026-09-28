import os, json
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import tensorflow as tf
from tensorflow.keras.models import load_model

def CTCLoss(y_true, y_pred):
    bl = tf.cast(tf.shape(y_true)[0], dtype="int64")
    il = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    ll = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((bl, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, il, ll)

SAVE_PATH = os.path.join('models', 'trained_model_grid_multi_dsconv.h5')
RESUME_WEIGHTS = os.path.join('models', 'checkpoint_grid_multi_dsconv_resume.weights.h5')
STATE_FILE = os.path.join('models', 'train_dsconv_state.json')

model = load_model(SAVE_PATH, custom_objects={'CTCLoss': CTCLoss}, compile=False)
model.save_weights(RESUME_WEIGHTS)

with open(STATE_FILE, 'w') as f:
    json.dump({'next_epoch': 0, 'val_loss': 56.2438}, f)

print('已將第一次訓練最佳權重(val_loss 56.24)轉存為 resume 權重，並重設 state.json，next_epoch=0')
