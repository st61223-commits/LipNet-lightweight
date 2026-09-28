import os, time, glob
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
import numpy as np
import tensorflow as tf

def _load_frames(path):
    project_name = os.path.basename(os.path.dirname(path))
    file_name = os.path.splitext(os.path.basename(path))[0]
    for suffix in ['_cached', '_dlib_cached']:
        cache_path = os.path.join('data', f'{project_name}{suffix}', f'{file_name}.npy')
        if os.path.exists(cache_path):
            frames = np.load(cache_path)
            mean, std = np.mean(frames), np.std(frames) + 1e-6
            return ((frames - mean) / std).astype(np.float32)
    raise FileNotFoundError(cache_path)

TFLITE_PATH = 'models/width05_dynamic_int8.tflite'

def predict_one(frames):
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(in_idx, frames.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(out_idx)

files = sorted(glob.glob('data/s99_1/*.mpg'))[:5]
for i, fp in enumerate(files):
    x = np.expand_dims(_load_frames(fp), 0)
    t0 = time.time()
    yhat = predict_one(x)
    t1 = time.time()
    print(f'樣本{i+1}: {t1-t0:.2f}秒  argmax前5={yhat[0].argmax(axis=-1)[:5]}')
