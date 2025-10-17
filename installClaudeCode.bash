#!/usr/bin/env bash

set -e

echo "=================================="
echo "Claude Code Auto-Installer"
echo "=================================="
echo ""

# Check if Claude Code is already installed
if command -v claude &> /dev/null; then
    echo "✓ Claude Code is already installed"
    claude --version
    echo ""
    read -p "Do you want to reinstall? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
fi

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    OS="windows"
fi

echo "Detected OS: $OS"
echo ""

# Install based on OS
case $OS in
    "linux")
        echo "Installing Claude Code for Linux..."
        if command -v curl &> /dev/null; then
            curl -fsSL https://claude.ai/install.sh | bash
        else
            echo "Error: curl is required but not installed."
            echo "Please install curl and try again."
            exit 1
        fi
        ;;
    "macos")
        echo "Installing Claude Code for macOS..."
        if command -v brew &> /dev/null; then
            echo "Using Homebrew installation..."
            brew install --cask claude-code
        elif command -v curl &> /dev/null; then
            echo "Using curl installation..."
            curl -fsSL https://claude.ai/install.sh | bash
        else
            echo "Error: Neither Homebrew nor curl found."
            echo "Please install Homebrew or curl and try again."
            exit 1
        fi
        ;;
    "windows")
        echo "For Windows, please run one of the following commands:"
        echo ""
        echo "PowerShell:"
        echo "  irm https://claude.ai/install.ps1 | iex"
        echo ""
        echo "CMD:"
        echo "  curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd"
        echo ""
        exit 1
        ;;
    *)
        echo "Unsupported OS: $OSTYPE"
        echo ""
        echo "Please install Claude Code manually using one of these methods:"
        echo ""
        echo "NPM (Node.js 18+):"
        echo "  npm install -g @anthropic-ai/claude-code"
        echo ""
        echo "Or visit: https://docs.claude.com/en/docs/claude-code/quickstart"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "Installation Complete!"
echo "=================================="
echo ""

# Verify installation
if command -v claude &> /dev/null; then
    echo "✓ Claude Code successfully installed"
    claude --version
    echo ""
    echo "To get started, run: claude"
else
    echo "⚠ Installation may not have completed successfully."
    echo "Please check the output above for errors."
    echo ""
    echo "You may need to restart your terminal or add Claude Code to your PATH."
    exit 1
fi
