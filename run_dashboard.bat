@echo off
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title Resilient Microgrid Digital Twin
cd /d "%~dp0"
echo ========================================================
echo Launching Microgrid Digital Twin Interactive Dashboard...
echo ========================================================
"C:\Users\hanwi\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app/app.py --server.port 8501
pause
