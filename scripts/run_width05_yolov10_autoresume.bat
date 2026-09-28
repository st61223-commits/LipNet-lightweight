@echo off
rem Auto-run at Windows login. If width05_yolov10 training is not finished yet,
rem resume it (matches run_width05_yolov8-style autoresume convention).

set BASE_DIR=C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet
set DONE_FLAG=%BASE_DIR%\models\train_width05_yolov10_DONE.flag
set PYTHON=C:\Users\Tno\miniconda3\envs\py39\python.exe
set SCRIPT=%BASE_DIR%\scripts\train_grid_multi_width05_yolov10.py
set LOG=%BASE_DIR%\logs\train_grid_multi_width05_yolov10_out.txt

if exist "%DONE_FLAG%" (
    echo width05_yolov10 training already done, not running again.
    exit /b 0
)

echo. >> "%LOG%"
echo ===== %date% %time% auto-resume start width05_yolov10 training ===== >> "%LOG%"
"%PYTHON%" "%SCRIPT%" >> "%LOG%" 2>&1

if not exist "%DONE_FLAG%" (
    echo width05_yolov10 training did not finish this run, will resume next login. >> "%LOG%"
)
