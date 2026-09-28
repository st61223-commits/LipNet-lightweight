import os, sys, pathlib, warnings
pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
import numpy as np, tensorflow as tf

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# load a real cached sample
frames = np.load('data/s1_cached/bbaf2n.npy')
mean, std = np.mean(frames), np.std(frames) + 1e-6
frames = ((frames - mean) / std).astype(np.float32)
frames = frames[None, ...]
print("frames shape:", frames.shape)

def run_tflite(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    interp.set_tensor(in_d['index'], frames)
    interp.invoke()
    return interp.get_tensor(out_d['index'])

for path in ['models/width05_dynamic_int8.tflite', 'models/width05_float32_baseline.tflite']:
    out = run_tflite(path)
    dec = tf.keras.backend.ctc_decode(tf.cast(out, tf.float32), [75]*len(out), greedy=False)[0][0].numpy()
    txt = ' '.join(w for w in tf.strings.reduce_join(num_to_char(dec[0])).numpy().decode().strip().split() if len(w) > 1)
    print(f"{path} -> argmax first10: {out[0,:5].argmax(axis=-1)} decoded: '{txt}'")

# true answer for bbaf2n should be something like "bin blue at f two now"
