@echo off
rem Auto-run at Windows login. If width05 training is not finished yet,
rem resume it. Once training is truly done (DONE flag written), run the
rem benchmark exactly once and record the result.

set BASE_DIR=C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet
set DONE_FLAG=%BASE_DIR%\models\train_width05_DONE.flag
set BENCH_DONE_FLAG=%BASE_DIR%\models\width05_benchmark_DONE.flag
set PYTHON=C:\Users\Tno\miniconda3\envs\py39\python.exe
set SCRIPT=%BASE_DIR%\scripts\train_grid_multi_width05.py
set LOG=%BASE_DIR%\logs\train_grid_multi_width05_log.txt
set BENCH_SCRIPT=C:\Users\Tno\claude-code\benchmark_width05.py
set BENCH_LOG=%BASE_DIR%\logs\benchmark_width05_log.txt

if exist "%DONE_FLAG%" goto :maybe_benchmark

echo. >> "%LOG%"
echo ===== %date% %time% auto-resume start width05 training ===== >> "%LOG%"
"%PYTHON%" "%SCRIPT%" >> "%LOG%" 2>&1

if not exist "%DONE_FLAG%" (
    echo width05 training did not finish this run, will resume next login. >> "%LOG%"
    exit /b 0
)

:maybe_benchmark
if exist "%BENCH_DONE_FLAG%" (
    echo width05 benchmark already done, not running again.
    exit /b 0
)

echo. >> "%BENCH_LOG%"
echo ===== %date% %time% width05 training finished, running benchmark ===== >> "%BENCH_LOG%"
"%PYTHON%" "%BENCH_SCRIPT%" >> "%BENCH_LOG%" 2>&1
echo done > "%BENCH_DONE_FLAG%"
echo ===== %date% %time% benchmark finished ===== >> "%BENCH_LOG%"
