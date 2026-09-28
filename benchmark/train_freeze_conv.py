"""
凍結 Conv 層 + 重訓 LSTM 多說話者訓練
策略：
  Conv3D 層已從 s99_6_14 學到通用的嘴唇特徵抽取能力 → 凍結保留
  Bidirectional LSTM + Dense → 解凍，在全部 1939 支影片上重新訓練
  這讓模型保留「看嘴」的能力，同時學會辨識不同說話者
"""
import os, sys, pathlib, warnings, time
pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)

os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
sys.path.append(r'C:\Users\Tno\claude-code')

import cv2, numpy as np, tensorflow as tf, torch
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv3D, MaxPool3D, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from vocab_correction import correct_sentence

# ── GPU ──
physical_devices = tf.config.list_physical_devices('GPU')
try: tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: pass
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print(f'GPU: {physical_devices}')

# ── YOLOv5 ──
yolov5s = torch.hub.load(
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5', 'custom',
    path=r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5\runs\v5s\weights\best.pt',
    source='local')
print('YOLOv5 載入完成')

# ── 字元對照表 ──
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

input_shape = (75, 46, 140, 1)

def load_video(path):
    project_name = os.path.basename(os.path.dirname(path))
    cache_dir = os.path.join('data', f'{project_name}_cached')
    file_name = os.path.splitext(os.path.basename(path))[0]
    cache_path = os.path.join(cache_dir, f'{file_name}.npy')
    if os.path.exists(cache_path):
        frames = np.load(cache_path)
        mean, std = np.mean(frames), np.std(frames) + 1e-6
        return ((frames - mean) / std).astype(np.float32)
    cap = cv2.VideoCapture(path); frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret: break
        for det in yolov5s(frame).pred:
            for *xyxy, conf, cls in det:
                if int(cls)==0 and conf>=0.5:
                    x1,y1,x2,y2=map(int,xyxy)
                    lip=cv2.resize(frame[y1:y2,x1:x2],(input_shape[2],input_shape[1]))
                    frames.append(cv2.cvtColor(lip,cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    T=75; frames=frames[:T]
    while len(frames)<T: frames.append(frames[-1] if frames else np.zeros((input_shape[1],input_shape[2]),dtype=np.float32))
    frames=np.expand_dims(np.array(frames),-1)
    os.makedirs(cache_dir,exist_ok=True); np.save(cache_path,frames)
    mean,std=np.mean(frames),np.std(frames)+1e-6
    return ((frames-mean)/std).astype(np.float32)

def load_alignments(path):
    with open(path,'r') as f: lines=f.readlines()
    tokens=[]
    for line in lines:
        parts=line.split(); word=parts[2]
        if word not in ('sil','sp') and len(word)>1:
            tokens.extend([' ',word])
    tokens_flat=tokens[1:]
    if not tokens_flat: return tf.zeros([1],dtype=tf.int64)
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens_flat,input_encoding='UTF-8'),(-1)))

def load_data(path):
    try:
        path=bytes.decode(path.numpy())
        file_name=os.path.splitext(os.path.basename(path))[0]
        parts=[p for p in path.replace('\\\\','/').replace('\\','/').split('/') if p]
        project_name=parts[-2]
        frames=load_video(os.path.join('data',project_name,f'{file_name}.mpg'))
        alignments=load_alignments(os.path.join('data','alignments',project_name.replace('_new',''),f'{file_name}.align'))
    except Exception as e:
        frames=np.zeros((75,46,140,1),dtype=np.float32); alignments=np.zeros((1,),dtype=np.int32)
    return frames, alignments

def mappable_function(path):
    features,labels=tf.py_function(load_data,[path],(tf.float32,tf.int64))
    features.set_shape([75,None,None,1]); labels.set_shape([40])
    return features,labels

def CTCLoss(y_true,y_pred):
    bl=tf.cast(tf.shape(y_true)[0],dtype="int64")
    il=tf.cast(tf.shape(y_pred)[1],dtype="int64")*tf.ones((bl,1),dtype="int64")
    ll=tf.cast(tf.shape(y_true)[1],dtype="int64")*tf.ones((bl,1),dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true,y_pred,il,ll)

def scheduler(epoch, lr):
    return lr if epoch < 30 else lr * float(tf.math.exp(-0.1))

def make_dataset(patterns, batch_size=2, buf=2000):
    ds = tf.data.Dataset.list_files(patterns)
    ds = ds.shuffle(buf, reshuffle_each_iteration=False)
    ds = ds.map(mappable_function)
    ds = ds.padded_batch(batch_size, padded_shapes=([75,None,None,1],[40]))
    return ds.prefetch(tf.data.AUTOTUNE)

def measure_all(mdl, ds, label):
    raw_ok=corr_ok=total=0
    for frames,labels in ds.as_numpy_iterator():
        yhat=mdl.predict(frames,verbose=0)
        dec=tf.keras.backend.ctc_decode(tf.cast(yhat,tf.float32),[75]*len(yhat),greedy=False)[0][0].numpy()
        orig=' '.join(tf.strings.reduce_join(num_to_char(labels[0])).numpy().decode().strip().split())
        raw =' '.join(tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split())
        corr=' '.join(correct_sentence(raw).split())
        if orig==raw:  raw_ok+=1
        if orig==corr: corr_ok+=1
        total+=1
    if total==0: return 0,0
    r=raw_ok/total*100; c=corr_ok/total*100
    print(f'  {label:30s}: 原始 {raw_ok:3d}/{total} = {r:5.1f}%   校正後 {corr_ok:3d}/{total} = {c:5.1f}%')
    return r, c

# ════════════════════════════════════════
# ■ 載入 s99_6_14，凍結 Conv 層
# ════════════════════════════════════════
print('\n' + '='*60)
print('策略：凍結 Conv 層，重訓 LSTM 於全部 1939 支影片')
print('='*60)

model = load_model(
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\trained_model_s99_6_14.h5',
    custom_objects={'CTCLoss': CTCLoss})

# 凍結 Conv3D / MaxPool3D / Activation 層
frozen_count = trainable_count = 0
for layer in model.layers:
    if isinstance(layer, (Conv3D, MaxPool3D, Activation)):
        layer.trainable = False
        frozen_count += 1
    else:
        layer.trainable = True
        trainable_count += 1

model.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1.0), loss=CTCLoss)

# 顯示哪些層被凍結
print('\n層的可訓練狀態：')
for layer in model.layers:
    params = layer.count_params()
    status = '❄ 凍結' if not layer.trainable else '▶ 訓練'
    if params > 0:
        print(f'  {status}  {layer.name:35s} 參數: {params:,}')

total_params = model.count_params()
trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
print(f'\n總參數: {total_params:,}  可訓練: {trainable_params:,}  凍結: {total_params-trainable_params:,}')

# ════════════════════════════════════════
# ■ 建立全資料集
# ════════════════════════════════════════
ALL_PATTERNS = [
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_1\*.mpg',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_3\*.mpg',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_5\*.mpg',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6\*.mpg',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6_new\*.mpg',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7\*.mpg',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7_new\*.mpg',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_8\*.mpg',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s34_3\*.mpg',
]

data_all = make_dataset(ALL_PATTERNS, batch_size=2)
total_batches = tf.data.experimental.cardinality(data_all).numpy()
tr_b = int(total_batches * 0.7)
va_b = int(total_batches * 0.2)
train_ds = data_all.take(tr_b)
val_ds   = data_all.skip(tr_b).take(va_b)
test_ds  = data_all.skip(tr_b + va_b)
print(f'\n全資料集（batch_size=2）: 總={total_batches}  訓練={tr_b}  驗證={va_b}  測試={total_batches-tr_b-va_b}')

# 各說話者專用測試集（batch_size=1）
test_s6  = make_dataset([
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6\*.mpg',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6_new\*.mpg',
], batch_size=1, buf=300)
test_s35 = make_dataset([
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_3\*.mpg',
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_5\*.mpg',
], batch_size=1, buf=50)
test_s7  = make_dataset([
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7\*.mpg',
], batch_size=1, buf=300)

# ════════════════════════════════════════
# ■ Phase 1：只訓練 LSTM + Dense
# ════════════════════════════════════════
print('\n' + '-'*60)
print('Phase 1：LSTM + Dense 訓練（最多 40 epochs）')
print('-'*60)

SAVE_PATH = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\trained_model_freeze_conv.h5'

class ShowExample(tf.keras.callbacks.Callback):
    def __init__(self, ds): self._ds=ds; self._it=ds.as_numpy_iterator()
    def on_epoch_end(self, epoch, logs=None):
        try: b=self._it.next()
        except StopIteration: self._it=self._ds.as_numpy_iterator(); b=self._it.next()
        yhat=self.model.predict(b[0],verbose=0)
        dec=tf.keras.backend.ctc_decode(tf.cast(yhat,tf.float32),[75]*len(yhat),greedy=False)[0][0].numpy()
        raw=tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip()
        orig=tf.strings.reduce_join(num_to_char(b[1][0])).numpy().decode().strip()
        print(f'  [{epoch+1}] 原：{orig}  預：{raw}  校正：{correct_sentence(raw)}')

t0 = time.time()
h1 = model.fit(train_ds, validation_data=val_ds, epochs=40, callbacks=[
    ModelCheckpoint(SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1),
    LearningRateScheduler(scheduler),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ShowExample(test_ds),
])
print(f'Phase 1 完成，耗時 {(time.time()-t0)/60:.1f} 分鐘')

# ════════════════════════════════════════
# ■ Phase 2：解凍全部，低學習率微調
# ════════════════════════════════════════
best_val_p1 = min(h1.history['val_loss'])
print(f'\nPhase 1 最佳 val_loss: {best_val_p1:.4f}')

if best_val_p1 < 5.0:
    print('\n' + '-'*60)
    print('Phase 2：解凍全部層，低學習率微調（最多 20 epochs）')
    print('-'*60)
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=0.00003, clipnorm=1.0), loss=CTCLoss)

    SAVE_PATH2 = r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\trained_model_freeze_conv_p2.h5'
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[
        ModelCheckpoint(SAVE_PATH2, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ShowExample(test_ds),
    ])
    best_val_p2 = min(h2.history['val_loss'])
    if best_val_p2 < best_val_p1:
        SAVE_PATH = SAVE_PATH2
        print(f'Phase 2 改善：{best_val_p1:.4f} → {best_val_p2:.4f}，使用 Phase 2 模型')
    else:
        print(f'Phase 2 未改善（{best_val_p2:.4f} > {best_val_p1:.4f}），保留 Phase 1 模型')
else:
    print('Phase 1 val_loss > 5.0，跳過 Phase 2')

print(f'\n總耗時：{(time.time()-t0)/60:.1f} 分鐘')

# ════════════════════════════════════════
# ■ 最終測試
# ════════════════════════════════════════
print('\n' + '='*60)
print('最終準確率（載入最佳模型 + 詞彙校正）')
print('='*60)

best_model = load_model(SAVE_PATH, custom_objects={'CTCLoss': CTCLoss})

measure_all(best_model, test_s6,  's99_6（訓練過的人，200支）')
measure_all(best_model, test_s35, 's99_3+s99_5（其他人，40支）')
measure_all(best_model, test_s7,  's99_7（其他人，200支）')

# 基準線比較
model_orig = load_model(
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\trained_model_s99_6_14.h5',
    custom_objects={'CTCLoss': CTCLoss})
print('\n基準線（s99_6_14）：')
measure_all(model_orig, test_s6,  's99_6')
measure_all(model_orig, test_s35, 's99_3+s99_5')
measure_all(model_orig, test_s7,  's99_7')
