@echo off
REM Autoresume script for Transformer training (English comments only, avoid Big5 encoding corruption)
REM Checks DONE flag first, skips if training already finished
if exist "C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\models\train_transformer_DONE.flag" (
    exit /b 0
)
cd /d "C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet"
"C:\Users\Tno\miniconda3\envs\py39\python.exe" "scripts\train_grid_multi_transformer.py" >> "logs\train_transformer_log.txt" 2>> "logs\train_transformer_err.txt"
