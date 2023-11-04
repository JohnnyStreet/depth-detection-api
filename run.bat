@echo off

setlocal

set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1

if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    python.exe -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install torch --index-url https://download.pytorch.org/whl/cu121
) else (
    call venv\Scripts\activate.bat
)

uvicorn run:app --reload --log-level info

call venv\Scripts\deactivate.bat

endlocal