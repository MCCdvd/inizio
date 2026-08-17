#!/usr/bin/env bash
# Crea un venv, installa le dipendenze ed esegue entrambi i test del modulo one/
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" -q

"$VENV_DIR/bin/python" -m pytest \
    "$SCRIPT_DIR/tests/test_trading_env.py" \
    "$SCRIPT_DIR/tests/test_metrics_and_backtest.py" \
    -q
