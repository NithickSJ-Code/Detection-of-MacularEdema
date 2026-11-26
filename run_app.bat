@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python 3.12 virtual environment...
  py -3.12 -m venv .venv --system-site-packages
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt -r requirements-api.txt
)

if not exist "frontend\node_modules" (
  echo Installing frontend dependencies...
  pushd frontend
  npm install
  popd
)

echo Starting API on http://127.0.0.1:8000
start "OCT API" cmd /k "cd /d "%~dp0" && .venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul

echo Starting UI on http://localhost:5173
pushd frontend
start "OCT UI" cmd /k "npm run dev"
popd

echo.
echo Open http://localhost:5173 in Chrome.
echo Close the OCT API and OCT UI windows to stop the app.
endlocal
