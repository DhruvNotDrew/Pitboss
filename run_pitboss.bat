@echo off
setlocal

if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" "pitboss.pyw"
) else (
  start "" pyw "pitboss.pyw"
)

endlocal
