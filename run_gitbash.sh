#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Always invoke the venv's interpreter explicitly. This avoids Conda/base Python
# leaking into imports when Git Bash shows both (base) and (.venv).
if [[ ! -x .venv/Scripts/python.exe ]]; then
  echo "Creating project-local virtual environment using the current Python..."
  python -m venv .venv
fi

PY=".venv/Scripts/python.exe"
"$PY" -c 'import sys; print("Using Python:", sys.executable); print(sys.version)'
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt
exec "$PY" app.py
