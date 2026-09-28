import os, sys, time
sys.path.insert(0, r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\scripts')
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import tensorflow as tf
import train_dsconv_distill as m

print('載入老師...')
teacher = tf.keras.models.load_model(m.TEACHER_PATH, custom_objects={'CTCLoss': m.CTCLoss}, compile=False)
teacher.trainable = False
print('載入學生（熱啟動權重）...')
student = tf.keras.models.load_model(m.STUDENT_INIT_PATH, custom_objects={'CTCLoss': m.CTCLoss}, compile=False)

distill = m.DistillModel(student, teacher, alpha=m.ALPHA)
distill.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0))

data = tf.data.Dataset.list_files([r'data\s1\*.mpg'], shuffle=False).take(4)
data = data.map(m.mappable_function)
data = data.padded_batch(1, padded_shapes=([75, None, None, 1], [40]))

print('\n===== 跑 4 個 train_step，觀察 loss 是否正常、有沒有變化 =====')
for i, (x, y) in enumerate(data):
    t0 = time.time()
    logs = distill.train_step((x, y))
    print(f'step {i}: loss={float(logs["loss"]):.4f}  ctc={float(logs["ctc_loss"]):.4f}  '
          f'kd={float(logs["kd_loss"]):.4f}  耗時={time.time()-t0:.1f}s')

print('\n===== 測試 save/load 權重 round-trip =====')
distill.student.save_weights('models/_smoke_test_weights.weights.h5')
student2 = tf.keras.models.load_model(m.STUDENT_INIT_PATH, custom_objects={'CTCLoss': m.CTCLoss}, compile=False)
student2.load_weights('models/_smoke_test_weights.weights.h5')
print('save/load 成功，沒有拋出例外')

print('\n===== 測試 test_step（驗證用）=====')
logs = distill.test_step((x, y))
print(f'val loss(純CTC): {float(logs["loss"]):.4f}')

print('\n全部測試通過！')
