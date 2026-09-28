"""
驗證 width05.onnx 的輸出，是不是真的跟原本的 .h5 模型一致（不是轉換壞掉的垃圾輸出）。
用同一組固定亂數種子的 dummy 輸入，比較兩邊 argmax 逐幀是否相同。
在 tf215 環境跑（同時裝了 tf 跟 onnxruntime，方便一次比較兩邊）。

注意：這裡的 .h5 模型是用 tf215（TF2.20/Keras3）去讀，跟正式跑在 py39（TF2.10）
可能有些微數值差異是正常的（不同TF版本的浮點運算順序），重點是看 argmax 決策
（也就是最後解碼出的文字）是否一致，而不是看小數點後幾位的機率是否完全相同。
"""
import os
os.chdir(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet')

import numpy as np
import onnxruntime as ort

rng = np.random.default_rng(0)
dummy = rng.standard_normal((1, 75, 46, 140, 1)).astype('float32')

sess = ort.InferenceSession('models/width05.onnx', providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
onnx_out = sess.run([output_name], {input_name: dummy})[0]

print('ONNX 輸出 shape:', onnx_out.shape)
onnx_argmax = onnx_out[0].argmax(axis=-1)
print('ONNX 逐幀 argmax（前30個）:', onnx_argmax[:30])
np.save('models/_onnx_dummy_output.npy', onnx_out)
print('已存 ONNX 輸出，供另一支腳本在 py39 讀取比對')
