@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet

echo =============================================
echo  第一步：從影片自動產生標記資料
echo =============================================
C:\Users\Tno\miniconda3\envs\py39\python.exe -u C:\Users\Tno\claude-code\generate_lip_dataset.py
if %errorlevel% neq 0 (
    echo [錯誤] 標記資料產生失敗，請檢查錯誤訊息
    pause
    exit /b 1
)

echo.
echo =============================================
echo  第二步：訓練 YOLOv8n
echo =============================================
C:\Users\Tno\miniconda3\envs\py39\python.exe -u C:\Users\Tno\claude-code\train_yolov8.py
if %errorlevel% neq 0 (
    echo [錯誤] YOLOv8 訓練失敗，請檢查錯誤訊息
    pause
    exit /b 1
)

echo.
echo =============================================
echo  全部完成！
echo  最佳權重存放在：
echo  C:\Users\Tno\OneDrive\Lipnet_nchu\yolov8_lip\lip_detect\weights\best.pt
echo =============================================
pause
