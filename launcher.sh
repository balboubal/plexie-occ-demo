#!/bin/bash

echo ""
echo " PlexIE OCC Demo Launcher"
echo " ========================"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo " ERROR: Node.js not found!"
    echo " Install from https://nodejs.org/"
    echo ""
    exit 1
fi

echo " Starting launcher..."
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Automatically send "3" to launch the demo
# Using Codespaces-specific launcher (no auto-browser opening)
echo "3" | node launcher-codespaces.cjs
