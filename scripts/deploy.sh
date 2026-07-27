#!/usr/bin/env bash
# Build the web client and restart the uvicorn server.
# Run on the remote GPU server (not locally).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
VENV_UVICORN="$REPO_ROOT/.venv/bin/uvicorn"

echo "==> Build web client"
if command -v npm &>/dev/null && [[ -d "$REPO_ROOT/client" ]]; then
    cd "$REPO_ROOT/client"
    npm install --silent
    npm run build
    echo "    Web client built → client/dist/"
    cd "$REPO_ROOT"
else
    echo "    npm not found — skipping client build"
fi

echo "==> Restart server"
# If running under systemd:
if systemctl is-active --quiet nepali-translator 2>/dev/null; then
    sudo systemctl restart nepali-translator
    echo "    Restarted via systemd"
elif [[ -f /tmp/translator.pid ]]; then
    OLD_PID=$(cat /tmp/translator.pid)
    kill -TERM "$OLD_PID" 2>/dev/null || true
    sleep 1
fi

# Start server in background
nohup "$VENV_PYTHON" -m server.main \
    --config "$REPO_ROOT/configs/default.yaml" \
    > "$REPO_ROOT/translator.log" 2>&1 &

echo $! > /tmp/translator.pid
echo "    Server PID: $!"
echo "    Logs: $REPO_ROOT/translator.log"
echo ""
echo "==> Deploy complete. Access at http://$(hostname -I | awk '{print $1}'):8000"
