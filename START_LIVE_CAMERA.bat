@echo off
echo ====================================================
echo  Aravind Eye Hospital - OCT Analysis System
echo ====================================================
echo.
echo Installing required AI libraries...
pip install -r requirements.txt
echo.
echo Starting OCT Scanner...
python scan.py
pause
