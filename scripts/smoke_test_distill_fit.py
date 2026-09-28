import os, sys, json
sys.path.insert(0, r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\scripts')
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import tensorflow as tf
import train_dsconv_distill as m

teacher = tf.keras.models.load_model(m.TEACHER_PATH, custom_objects={'CTCLoss': m.CTCLoss}, compile=False)
teacher.trainable = False
student = tf.keras.models.load_model(m.STUDENT_INIT_PATH, custom_objects={'CTCLoss': m.CTCLoss}, compile=False)

distill = m.DistillModel(student, teacher, alpha=m.ALPHA)
distill.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0))

data = tf.data.Dataset.list_files([r'data\s1\*.mpg'], shuffle=False).take(6)
data = data.map(m.mappable_function)
data = data.padded_batch(1, padded_shapes=([75, None, None, 1], [40]))
train = data.take(4)
val = data.skip(4)

TEST_STATE = 'models/_smoke_test_state.json'
TEST_WEIGHTS = 'models/_smoke_test_resume.weights.h5'
cb = m.ResumeCheckpoint(TEST_WEIGHTS, TEST_STATE, save_every_sec=999999)

print('用 model.fit() 真正跑 1 epoch（4 train batch, 2 val batch）...')
distill.fit(train, validation_data=val, epochs=1, callbacks=[cb], verbose=2)

print('\n檢查 state.json 是否為合法 JSON...')
with open(TEST_STATE, 'r') as f:
    state = json.load(f)
print('讀取成功:', state)
print('\n測試通過！ResumeCheckpoint.on_epoch_end 沒有再出現 ndarray not JSON serializable 的問題。')
