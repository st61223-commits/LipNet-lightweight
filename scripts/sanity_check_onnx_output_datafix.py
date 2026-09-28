"""
驗證 width05_datafix.onnx 輸出是否跟 .h5 一致（在 tf215 環境跑，存輸出給 py39 那邊比對）。
"""
import os
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import numpy as np
import onnxruntime as ort

rng = np.random.default_rng(0)
dummy = rng.standard_normal((1, 75, 46, 140, 1)).astype('float32')

sess = ort.InferenceSession('models/width05_datafix.onnx', providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
onnx_out = sess.run([output_name], {input_name: dummy})[0]

print('ONNX 輸出 shape:', onnx_out.shape)
onnx_argmax = onnx_out[0].argmax(axis=-1)
print('ONNX 逐幀 argmax（前30個）:', onnx_argmax[:30])
np.save('models/_onnx_datafix_dummy_output.npy', onnx_out)
print('已存 ONNX 輸出，供另一支腳本在 py39 讀取比對')
