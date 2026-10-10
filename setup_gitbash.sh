#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Remove only this project's venv so it is recreated cleanly without inherited
# Conda packages. Does not touch any global Python/Conda installation.
rm -rf .venv
python -m venv .venv
PY=".venv/Scripts/python.exe"
"$PY" -c 'import sys; print("Clean project interpreter:", sys.executable); print(sys.version)'
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt
