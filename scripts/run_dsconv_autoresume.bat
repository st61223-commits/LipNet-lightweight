@echo off
rem Auto-run at Windows login. If dsconv training is not finished yet,
rem resume it. Once training is truly done (DONE flag written), run the
rem benchmark exactly once and record the result.
rem NOTE: keep comments English-only in this file (cmd.exe reads it as Big5,
rem Chinese comments saved as UTF-8 get mis-parsed and break the whole script).

set BASE_DIR=C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet
set DONE_FLAG=%BASE_DIR%\models\train_dsconv_DONE.flag
set BENCH_DONE_FLAG=%BASE_DIR%\models\dsconv_benchmark_DONE.flag
set PYTHON=C:\Users\Tno\miniconda3\envs\py39\python.exe
set SCRIPT=%BASE_DIR%\scripts\train_grid_multi_dsconv.py
set LOG=%BASE_DIR%\logs\train_grid_multi_dsconv_log.txt
set BENCH_SCRIPT=C:\Users\Tno\claude-code\benchmark_dsconv.py
set BENCH_LOG=%BASE_DIR%\logs\benchmark_dsconv_log.txt

if exist "%DONE_FLAG%" goto :maybe_benchmark

echo. >> "%LOG%"
echo ===== %date% %time% auto-resume start dsconv training ===== >> "%LOG%"
"%PYTHON%" "%SCRIPT%" >> "%LOG%" 2>&1

if not exist "%DONE_FLAG%" (
    echo dsconv training did not finish this run, will resume next login. >> "%LOG%"
    exit /b 0
)

:maybe_benchmark
if exist "%BENCH_DONE_FLAG%" (
    echo dsconv benchmark already done, not running again.
    exit /b 0
)

echo. >> "%BENCH_LOG%"
echo ===== %date% %time% dsconv training finished, running benchmark ===== >> "%BENCH_LOG%"
"%PYTHON%" "%BENCH_SCRIPT%" >> "%BENCH_LOG%" 2>&1
echo done > "%BENCH_DONE_FLAG%"
echo ===== %date% %time% benchmark finished ===== >> "%BENCH_LOG%"
