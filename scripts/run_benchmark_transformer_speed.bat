@echo off
cd /d "C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet"
set CUDA_VISIBLE_DEVICES=-1
"C:\Users\Tno\miniconda3\envs\py39\python.exe" scripts\benchmark_cpu_speed_yolov10_vs_transformer.py > scripts\benchmark_cpu_speed_yolov10_vs_transformer_out.txt 2>&1
