import os, sys, pathlib, warnings
pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore", category=FutureWarning)
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
sys.path.append(r'C:\Users\Tno\claude-code')

import numpy as np, tensorflow as tf, torch
from tensorflow.keras.models import load_model
from vocab_correction import correct_sentence

physical_devices = tf.config.list_physical_devices('GPU')
try: tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: pass
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

yolov5s = torch.hub.load(
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5','custom',
    path=r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5\runs\v5s\weights\best.pt',source='local')

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(),oov_token="",invert=True)

input_shape = (75,46,140,1)

def load_video(path):
    project_name=os.path.basename(os.path.dirname(path))
    cache_dir=os.path.join('data',f'{project_name}_cached')
    file_name=os.path.splitext(os.path.basename(path))[0]
    cache_path=os.path.join(cache_dir,f'{file_name}.npy')
    if os.path.exists(cache_path):
        frames=np.load(cache_path); mean,std=np.mean(frames),np.std(frames)+1e-6
        return ((frames-mean)/std).astype(np.float32)
    cap=__import__('cv2').VideoCapture(path); frames=[]
    for _ in range(int(cap.get(2))):
        ret,frame=cap.read()
        if not ret: break
        for det in yolov5s(frame).pred:
            for *xyxy,conf,cls in det:
                if int(cls)==0 and conf>=0.5:
                    import cv2; x1,y1,x2,y2=map(int,xyxy)
                    lip=cv2.resize(frame[y1:y2,x1:x2],(input_shape[2],input_shape[1]))
                    frames.append(cv2.cvtColor(lip,cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release(); T=75; frames=frames[:T]
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
        if word not in ('sil','sp') and len(word)>1: tokens.extend([' ',word])
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
    except:
        frames=np.zeros((75,46,140,1),dtype=np.float32); alignments=np.zeros((1,),dtype=np.int32)
    return frames,alignments

def mappable_function(path):
    f,l=tf.py_function(load_data,[path],(tf.float32,tf.int64))
    f.set_shape([75,None,None,1]); l.set_shape([40]); return f,l

def CTCLoss(y_true,y_pred):
    bl=tf.cast(tf.shape(y_true)[0],dtype="int64")
    il=tf.cast(tf.shape(y_pred)[1],dtype="int64")*tf.ones((bl,1),dtype="int64")
    ll=tf.cast(tf.shape(y_true)[1],dtype="int64")*tf.ones((bl,1),dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true,y_pred,il,ll)

def make_ds(patterns):
    ds=tf.data.Dataset.list_files(patterns).map(mappable_function)
    return ds.padded_batch(1,padded_shapes=([75,None,None,1],[40])).prefetch(tf.data.AUTOTUNE)

def test(mdl, ds, label):
    raw_ok=corr_ok=n=0; rows=[]
    for frames,labels in ds.as_numpy_iterator():
        yhat=mdl.predict(frames,verbose=0)
        dec=tf.keras.backend.ctc_decode(tf.cast(yhat,tf.float32),[75]*len(yhat),greedy=False)[0][0].numpy()
        orig=' '.join(tf.strings.reduce_join(num_to_char(labels[0])).numpy().decode().strip().split())
        raw =' '.join(tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split())
        corr=' '.join(correct_sentence(raw).split())
        r_ok=(orig==raw); c_ok=(orig==corr)
        if r_ok: raw_ok+=1
        if c_ok: corr_ok+=1
        n+=1; rows.append((orig,raw,corr,r_ok,c_ok))
    r=raw_ok/n*100; c=corr_ok/n*100
    print(f'\n【{label}】 {n} 筆')
    print(f'  原始: {raw_ok}/{n} = {r:.1f}%   校正後: {corr_ok}/{n} = {c:.1f}%')
    print(f'  {"狀態":4} {"原始答案":28} {"模型輸出":28} {"校正後":20}')
    print('  ' + '-'*83)
    for orig,raw,corr,r_ok,c_ok in rows:
        sym=('✓' if r_ok else '✗')+'→'+('✓' if c_ok else '✗')
        changed=' ←校正' if (not r_ok and c_ok) else ''
        print(f'  {sym:5} {orig:28} {raw:28} {corr:20}{changed}')
    return r,c

# ── 載入模型 ──
print('載入模型...')
new_model = load_model(
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\trained_model_freeze_conv.h5',
    custom_objects={'CTCLoss':CTCLoss})
orig_model = load_model(
    r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\trained_model_s99_6_14.h5',
    custom_objects={'CTCLoss':CTCLoss})
print('載入完成')

# ── 測試各說話者 ──
ds_s6  = make_ds([r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6\*.mpg',
                  r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_6_new\*.mpg'])
ds_s35 = make_ds([r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_3\*.mpg',
                  r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_5\*.mpg'])
ds_s7  = make_ds([r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7\*.mpg',
                  r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_7_new\*.mpg'])
ds_s8  = make_ds([r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_8\*.mpg'])
ds_s1  = make_ds([r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s99_1\*.mpg'])
ds_s34 = make_ds([r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s34_3\*.mpg'])

print('\n' + '='*60)
print('凍結Conv模型 — 全說話者測試')
print('='*60)

r1,c1 = test(new_model, ds_s6,  's99_6（200支）')
r2,c2 = test(new_model, ds_s7,  's99_7+s99_7_new（400支）')
r3,c3 = test(new_model, ds_s8,  's99_8（100支）')
r4,c4 = test(new_model, ds_s1,  's99_1（1000支）')
r5,c5 = test(new_model, ds_s34, 's34_3（199支）')
r6,c6 = test(new_model, ds_s35, 's99_3+s99_5（各20支）')

print(f'\n{"="*65}')
print(f'{"說話者":<28} {"影片數":>6} {"原始準確率":>12} {"加詞彙校正":>12}')
print('-'*65)
rows = [
    ('s99_6', 200, r1, c1),
    ('s99_7+s99_7_new', 400, r2, c2),
    ('s99_8', 100, r3, c3),
    ('s99_1', 1000, r4, c4),
    ('s34_3', 199, r5, c5),
    ('s99_3+s99_5', 40, r6, c6),
]
for name, n, r, c in rows:
    print(f'{name:<28} {n:>6}   {r:>10.1f}%   {c:>10.1f}%')
total_vids = sum(n for _,n,_,_ in rows)
avg_raw  = sum(n*r for _,n,r,_ in rows) / total_vids
avg_corr = sum(n*c for _,n,_,c in rows) / total_vids
print('-'*65)
print(f'{"加權平均":<28} {total_vids:>6}   {avg_raw:>10.1f}%   {avg_corr:>10.1f}%')
