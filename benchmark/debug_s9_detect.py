import pathlib, cv2, torch
pathlib.PosixPath = pathlib.WindowsPath

model = torch.hub.load(r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5', 'custom',
                        path=r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov5\runs\v5s\weights\best.pt', source='local')
model.conf = 0.001

cap = cv2.VideoCapture(r'C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\data\s9\bbak9s.mpg')
frame_idx = 0
any_detect = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    det = model(frame)
    n = len(det.pred[0])
    if n > 0:
        any_detect += 1
        confs = [float(c) for c in det.pred[0][:,4]]
        print(f"frame {frame_idx}: {n} detections, confs={confs}", flush=True)
    else:
        print(f"frame {frame_idx}: NO detection at all", flush=True)
    frame_idx += 1
cap.release()
print(f"\nTotal frames={frame_idx}, frames_with_any_detection={any_detect}", flush=True)
