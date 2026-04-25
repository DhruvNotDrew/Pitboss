@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
  echo Expected virtual environment at .venv\Scripts\python.exe
  echo Create one first: py -m venv .venv
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt

if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

pyinstaller --clean "Pitboss.spec"

echo.
echo Build complete.
echo Executable: dist\Pitboss.exe
endlocal
