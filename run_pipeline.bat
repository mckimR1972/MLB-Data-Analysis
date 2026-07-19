@echo off
chcp 65001 >null
"C:\Users\mirmi\AppData\Local\Programs\Python\Python313\python.exe" "C:\Users\mirmi\MLB_Analysis\MLB-Data-Analysis\statcast_pipeline.py"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Pipeline failed
    pause
)