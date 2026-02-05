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
    read -p "Press Enter to continue..."
    exit 1
fi

echo " Starting launcher..."
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

node launcher.cjs

echo ""
read -p "Press Enter to continue..."