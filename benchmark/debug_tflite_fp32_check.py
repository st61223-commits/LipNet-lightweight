import os, sys, pathlib, warnings
pathlib.PosixPath = pathlib.WindowsPath
warnings.simplefilter("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')
import numpy as np, tensorflow as tf

for path in ['models/width05_dynamic_int8.tflite', 'models/width05_float32_baseline.tflite']:
    print(f"=== {path} ===")
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    print("input:", in_d['shape'], in_d['dtype'])
    print("output:", out_d['shape'], out_d['dtype'])
    dummy = np.random.randn(*in_d['shape']).astype(np.float32)
    interp.set_tensor(in_d['index'], dummy)
    interp.invoke()
    out = interp.get_tensor(out_d['index'])
    print("out stats: shape=", out.shape, "min=", out.min(), "max=", out.max(), "mean=", out.mean())
    print()
