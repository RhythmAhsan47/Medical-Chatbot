@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating project-local virtual environment...
  python -m venv .venv
)
".venv\Scripts\python.exe" -c "import sys; print('Using Python:', sys.executable); print(sys.version)"
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" app.py
pause
