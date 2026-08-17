@echo off
:: Crea un venv, installa le dipendenze ed esegue entrambi i test del modulo one/
setlocal

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%.venv

if not exist "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
)

"%VENV_DIR%\Scripts\pip" install --upgrade pip -q
"%VENV_DIR%\Scripts\pip" install -r "%SCRIPT_DIR%requirements.txt" -q

"%VENV_DIR%\Scripts\python" -m pytest ^
    "%SCRIPT_DIR%tests\test_trading_env.py" ^
    "%SCRIPT_DIR%tests\test_metrics_and_backtest.py" ^
    -q
