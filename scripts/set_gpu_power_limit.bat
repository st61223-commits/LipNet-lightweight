@echo off
rem Lock GPU clock at a lower ceiling to reduce heat, since power-limit
rem (nvidia-smi -pl) is locked out on this laptop's GPU firmware.
rem NOTE: keep comments English-only (see run_dsconv_autoresume.bat note).

set LOG=C:\Users\Tno\OneDrive\Lipnet_nchu\LipNet\logs\gpu_power_limit_log.txt

echo ===== %date% %time% ===== >> "%LOG%"
nvidia-smi -lgc 300,1400 >> "%LOG%" 2>&1
nvidia-smi --query-gpu=clocks.gr,clocks.max.gr,power.limit,temperature.gpu --format=csv >> "%LOG%" 2>&1
