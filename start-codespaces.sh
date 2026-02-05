#!/usr/bin/env bash
set -euo pipefail

echo "Starting PlexIE OCC Demo for Codespaces..."
cd "$(dirname "${BASH_SOURCE[0]}")"

# Start Vite in background and capture logs
nohup npm run dev > /tmp/vite-dev.log 2>&1 &
gh codespace ports visibility 5173:public -c "$CODESPACE_NAME"
echo "Dev server started in background (logs → /tmp/vite-dev.log)"

# Wait until the port is listening (timeout after 60s)
MAX_TRIES=60
TRY=0
echo -n "Waiting for localhost:5173 to be ready"
while ! (command -v nc >/dev/null 2>&1 && nc -z 127.0.0.1 5173); do
  TRY=$((TRY+1))
  if [ "$TRY" -ge "$MAX_TRIES" ]; then
    echo
    echo "Timed out waiting for port 5173 after ${MAX_TRIES} seconds. Check /tmp/vite-dev.log"
    break
  fi
  echo -n "."
  sleep 1
done
echo
if [ "$TRY" -lt "$MAX_TRIES" ]; then
  echo "Port 5173 is listening (after $TRY seconds). Attempting to set visibility..."
  if command -v gh >/dev/null 2>&1; then
    # Try a few times in case of transient errors
    for attempt in 1 2 3 4 5; do
      if [ -n "${CODESPACE_NAME:-}" ]; then
        gh codespace ports visibility 5173:public -c "$CODESPACE_NAME" && { echo "Port 5173 set to public."; break; }
      else
        gh codespace ports visibility 5173:public && { echo "Port 5173 set to public."; break; }
      fi
      echo "gh attempt $attempt failed. Retrying in 1s..."
      sleep 1
    done
  else
    echo "gh CLI not found in the Codespace. You can install it or set the port to Public manually via the Ports panel."
  fi
else
  echo "Skipping visibility step because the dev server never appeared to start."
fi

echo "Done. Tail logs with: tail -f /tmp/vite-dev.log"
