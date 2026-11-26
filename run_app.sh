#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  python3.12 -m venv .venv --system-site-packages
  .venv/bin/python -m pip install -r requirements.txt -r requirements-api.txt
fi
if [ ! -d frontend/node_modules ]; then
  (cd frontend && npm install)
fi

.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill $API_PID' EXIT
(cd frontend && npm run dev)
