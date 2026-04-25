# Pitboss

Pitboss is a desktop FRC match analyzer for WPILog CSV exports. It detects robot events (power, CAN, vision, drive, loop timing, etc.), compares matches, and provides timeline/report views.

## Run Locally (dev)

1. Create a virtual environment:
   - `py -m venv .venv`
2. Install dependencies:
   - `.venv\Scripts\activate`
   - `pip install -r requirements.txt`
3. Launch the app:
   - `python frc_analyzer.py`
   - or double-click `run_pitboss.bat`

## Key Mapping

- Use the **KEY MAPPING** tab to map signals to exact keys in your logs.
- `Vision has-target` supports **multiple camera keys**.
- Save mappings to `frc_key_config.json` so they persist across sessions.

## Build a Clickable Windows App (.exe)

1. Ensure `.venv` exists.
2. Double-click `build_app.bat` (or run from terminal).
3. Output executable:
   - `dist\Pitboss.exe`

You can zip and share `Pitboss.exe` for others to run.

## GitHub Release Workflow

This repo includes a workflow at `.github/workflows/build-windows.yml` that:
- installs dependencies
- builds `Pitboss.exe` with PyInstaller
- uploads the executable as a workflow artifact

Run the workflow from GitHub Actions after pushing your repo.
