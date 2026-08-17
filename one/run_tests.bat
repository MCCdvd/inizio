@echo off
:: Crea un venv, installa le dipendenze ed esegue entrambi i test del modulo one/
setlocal

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%.venv

:: Individua il comando Python disponibile (python oppure py)
set PYTHON_CMD=
where python >nul 2>&1 && set PYTHON_CMD=python
if "%PYTHON_CMD%"=="" (
    where py >nul 2>&1 && set PYTHON_CMD=py
)
if "%PYTHON_CMD%"=="" (
    echo ERRORE: Python non trovato nel PATH. Installa Python da https://www.python.org/
    exit /b 1
)

:: Crea il venv se non esiste
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creazione venv in %VENV_DIR% ...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERRORE: impossibile creare il venv.
        exit /b 1
    )
)

echo Installazione dipendenze...
"%VENV_DIR%\Scripts\pip.exe" install --upgrade pip -q
"%VENV_DIR%\Scripts\pip.exe" install -r "%SCRIPT_DIR%requirements.txt" -q

echo Esecuzione test...
"%VENV_DIR%\Scripts\python.exe" -m pytest ^
    "%SCRIPT_DIR%tests\test_trading_env.py" ^
    "%SCRIPT_DIR%tests\test_metrics_and_backtest.py" ^
    -q
