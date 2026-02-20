#!/bin/bash
# install.sh — Install all dependencies for TokenPrint
#
# Sets up: ccusage (Claude), @ccusage/codex (Codex), Gemini telemetry
# Safe to run multiple times (idempotent).

set -e

echo "=== TokenPrint Setup ==="
echo ""

# --- 1. Check Python ---
if ! command -v python3 &>/dev/null; then
    echo "[error] python3 is required but not found."
    exit 1
fi
echo "[ok] python3 found: $(python3 --version 2>&1)"

# --- 2. Check Node/npm ---
if ! command -v npm &>/dev/null; then
    echo "[error] npm is required but not found. Install Node.js: https://nodejs.org"
    exit 1
fi
echo "[ok] npm found: $(npm --version 2>&1)"

# --- 3. Install ccusage (Claude Code usage tracking) ---
echo ""
echo "--- Claude Code (ccusage) ---"
if command -v ccusage &>/dev/null; then
    echo "[ok] ccusage already installed: $(ccusage --version 2>&1 || echo 'installed')"
else
    echo "Installing ccusage..."
    npm install -g ccusage@18
    echo "[ok] ccusage installed"
fi

# --- 4. Install @ccusage/codex (Codex CLI usage tracking) ---
echo ""
echo "--- Codex CLI (@ccusage/codex) ---"
# @ccusage/codex runs via npx, so just verify it's reachable
echo "Verifying @ccusage/codex is accessible via npx..."
if npx @ccusage/codex@18 --help &>/dev/null 2>&1; then
    echo "[ok] @ccusage/codex accessible via npx"
else
    echo "[warn] @ccusage/codex may not be available. It runs via npx and requires Codex CLI usage logs to exist."
fi

# --- 5. Setup Gemini CLI telemetry ---
echo ""
echo "--- Gemini CLI (telemetry) ---"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/setup-gemini-telemetry.sh" ]; then
    bash "$SCRIPT_DIR/setup-gemini-telemetry.sh"
else
    echo "[warn] setup-gemini-telemetry.sh not found. Skipping Gemini setup."
fi

# --- 6. Verify GitHub CLI (optional, for share image username) ---
echo ""
echo "--- GitHub CLI (optional) ---"
if command -v gh &>/dev/null; then
    GH_USER=$(gh api user --jq .login 2>/dev/null || echo "")
    if [ -n "$GH_USER" ]; then
        echo "[ok] gh CLI authenticated as @$GH_USER"
    else
        echo "[warn] gh CLI installed but not authenticated. Run: gh auth login"
    fi
else
    echo "[info] gh CLI not found. Share image will use git config user.name or prompt for name."
fi

# --- Done ---
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Run TokenPrint:"
echo "  pipx install -e .   # one-time CLI install"
echo "  tokenprint           # run dashboard"
echo ""
echo "Options:"
echo "  --since YYYYMMDD    Start date filter"
echo "  --until YYYYMMDD    End date filter"
echo "  --output PATH       Custom output path"
echo "  --no-open           Don't open in browser"
echo ""
echo "Note: Gemini telemetry only tracks future sessions — no historical backfill."
