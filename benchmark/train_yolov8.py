"""
第二步：用自動標記資料訓練 YOLOv8n（最小最快版本）
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

DATASET_YAML = r'C:\Users\Tno\OneDrive\Lipnet_nchu\lip_dataset\dataset.yaml'
OUTPUT_DIR   = r'C:\Users\Tno\OneDrive\Lipnet_nchu\yolov8_lip'

if __name__ == '__main__':
    model = YOLO(r'C:\Users\Tno\yolov8n.pt')  # 本機已有的 YOLOv8n 預訓練權重

    results = model.train(
        data=DATASET_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,          # 10 epoch 沒進步就停止
        device=0,             # GPU
        amp=False,            # 關閉自動混合精度（避免需要下載額外檔案）
        workers=0,            # Windows 多進程問題，設 0 用主進程載入資料
        project=OUTPUT_DIR,
        name='lip_detect',
        exist_ok=True,
        verbose=True,
    )

    print('\n訓練完成！')
    print(f'最佳權重：{OUTPUT_DIR}\\lip_detect\\weights\\best.pt')
