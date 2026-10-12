#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
printf '%s\n' 'Installed. Run: sh start_linux.sh (a graphical display is required).'
