#!/usr/bin/env bash

# This script starts VS Code with flags that disable service worker sandbox
# which should fix the webview service worker registration error

echo "Starting VS Code with service worker sandbox disabled..."

code --disable-features=DesktopPWAs,WebOTPService --no-sandbox --disable-gpu-sandbox "$@"
