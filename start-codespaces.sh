#!/usr/bin/env bash
set -euo pipefail

echo "Starting PlexIE OCC Demo for Codespaces..."
cd "$(dirname "${BASH_SOURCE[0]}")"

# Start Vite in background and capture logs
nohup npm run dev > /tmp/vite-dev.log 2>&1 &
echo "Dev server started in background (logs → /tmp/vite-dev.log)"
gh codespace ports visibility 5173:public -c "$CODESPACE_NAME"