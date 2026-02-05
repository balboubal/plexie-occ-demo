#!/usr/bin/env bash
set -euo pipefail

echo "Starting PlexIE OCC Demo for Codespaces..."
cd "$(dirname "${BASH_SOURCE[0]}")"

# Start Vite in background
npm run dev > /tmp/vite-dev.log 2>&1 &
VITE_PID=$!

echo "Vite started (PID=$VITE_PID), logs → /tmp/vite-dev.log"

# Give Vite time to bind the port
sleep 2

# Make port public (this works in your environment)
gh codespace ports visibility 5173:public -c "$CODESPACE_NAME"

# IMPORTANT: keep script alive briefly so Codespaces doesn't reap the process
sleep 3

echo "Startup complete"
