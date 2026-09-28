@echo off
rem Auto-run at Windows login. If distillation training is not finished yet,
rem resume it. Comments kept English-only (cmd.exe reads this file as Big5,
rem Chinese comments saved as UTF-8 get mis-parsed and break the whole script).

set BASE_DIR=C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet
set DONE_FLAG=%BASE_DIR%\models\train_dsconv_distill_DONE.flag
set PYTHON=C:\Users\Tno\miniconda3\envs\py39\python.exe
set SCRIPT=%BASE_DIR%\scripts\train_dsconv_distill.py
set LOG=%BASE_DIR%\logs\train_dsconv_distill_log.txt

if exist "%DONE_FLAG%" exit /b 0

echo. >> "%LOG%"
echo ===== %date% %time% auto-resume start distill training ===== >> "%LOG%"
"%PYTHON%" "%SCRIPT%" >> "%LOG%" 2>&1
