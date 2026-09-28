@echo off
rem 這個檔案會在每次登入 Windows 時自動執行。
rem 如果 v12 訓練還沒完成，就自動接續訓練；已完成的話就什麼都不做。

set BASE_DIR=C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet
set DONE_FLAG=%BASE_DIR%\models\train_v12_DONE.flag
set PYTHON=C:\Users\Tno\miniconda3\envs\py39\python.exe
set SCRIPT=%BASE_DIR%\scripts\train_grid_multi_v12.py
set LOG=%BASE_DIR%\logs\train_grid_multi_v12_log.txt

if exist "%DONE_FLAG%" (
    echo v12 訓練已完成，不再自動啟動。
    exit /b 0
)

echo. >> "%LOG%"
echo ===== %date% %time% 自動接續啟動 v12 訓練 ===== >> "%LOG%"
"%PYTHON%" "%SCRIPT%" >> "%LOG%" 2>&1
