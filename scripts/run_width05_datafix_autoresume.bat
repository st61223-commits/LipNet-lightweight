@echo off
REM Autoresume script for F_Width0.5 datafix retraining.
REM English comments only to avoid Big5/UTF-8 encoding corruption (see project notes).
REM If DONE flag exists, training already finished, do nothing.

if exist "C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\train_width05_datafix_DONE.flag" (
    exit /b 0
)

cd /d "C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet"
"C:\Users\Tno\miniconda3\envs\py39\python.exe" scripts\train_grid_multi_width05_datafix.py >> logs\train_width05_datafix_log.txt 2>&1
