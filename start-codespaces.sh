#!/bin/bash

# Codespaces-specific startup script
# Runs the dev server in the background without interactive menus

echo "Starting PlexIE OCC Demo for Codespaces..."

# Change to script directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# Start Vite dev server in background, redirect all output to log file
nohup npm run dev > /tmp/vite-dev.log 2>&1 &

echo "Dev server started in background"
echo "Check logs: tail -f /tmp/vite-dev.log"
echo "Port 5173 will be auto-forwarded by Codespaces"


# Wait until the port is actually listening
until nc -z localhost 5173; do
  sleep 0.5
done

# Force port to public
gh codespace ports visibility 5173:public -c "$CODESPACE_NAME"