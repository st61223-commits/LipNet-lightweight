import onnxruntime as ort
import numpy as np
for f in ['models/width05_yolov8_int8.onnx', 'models/width05_yolov8_int8_uint8w.onnx']:
    try:
        sess = ort.InferenceSession(f, providers=['CPUExecutionProvider'])
        in_name = sess.get_inputs()[0].name
        dummy = np.random.rand(1,75,46,140,1).astype(np.float32)
        out = sess.run(None, {in_name: dummy})[0]
        print(f'{f}: OK, shape={out.shape}')
    except Exception as e:
        print(f'{f}: FAILED - {type(e).__name__}: {e}')
