@echo off
rem 這個檔案如果放進 Windows 啟動資料夾，每次登入 Windows 時會自動執行。
rem 如果 bn_v3 續訓還沒完成，就自動接續訓練；已完成的話就什麼都不做。

set BASE_DIR=C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet
set DONE_FLAG=%BASE_DIR%\models\train_bn_v3_p2r_DONE.flag
set PYTHON=C:\Users\Tno\miniconda3\envs\py39\python.exe
set SCRIPT=%BASE_DIR%\scripts\train_bn_v3_p2_resume.py
set LOG=%BASE_DIR%\logs\train_bn_v3_p2_resume_log.txt

if exist "%DONE_FLAG%" (
    echo bn_v3 續訓已完成，不再自動啟動。
    exit /b 0
)

echo. >> "%LOG%"
echo ===== %date% %time% 自動接續啟動 bn_v3 續訓 ===== >> "%LOG%"
"%PYTHON%" "%SCRIPT%" >> "%LOG%" 2>&1
