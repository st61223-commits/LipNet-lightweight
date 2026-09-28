@echo off
cd /d "C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet"
"C:\Users\Tno\miniconda3\envs\py39\python.exe" scripts\benchmark_yolo_detector_speed.py > scripts\benchmark_yolo_detector_speed_out.txt 2>&1
