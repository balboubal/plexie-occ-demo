#!/bin/bash

# PlexIE OCC Demo Launcher
# Works on macOS and Linux

# Change to script directory
cd "$(dirname "$0")" || exit 1

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Check Node.js
if ! command -v node &> /dev/null; then
    echo ""
    echo -e "${RED}ERROR: Node.js is not installed!${NC}"
    echo ""
    echo "Install Node.js:"
    echo "  macOS:   brew install node"
    echo "  Ubuntu:  sudo apt install nodejs npm"
    echo "  Or:      https://nodejs.org/"
    echo ""
    exit 1
fi

# Run the launcher
node launcher.cjs
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo -e "${RED}Launcher exited with error code $EXIT_CODE${NC}"
    read -p "Press Enter to continue..."
fi

exit $EXIT_CODE
