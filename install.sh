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
    npm install -g ccusage
    echo "[ok] ccusage installed"
fi

# --- 4. Install @ccusage/codex (Codex CLI usage tracking) ---
echo ""
echo "--- Codex CLI (@ccusage/codex) ---"
# @ccusage/codex runs via npx, so just verify it's reachable
echo "Verifying @ccusage/codex is accessible via npx..."
if npx @ccusage/codex@latest --help &>/dev/null 2>&1; then
    echo "[ok] @ccusage/codex accessible via npx"
else
    echo "[warn] @ccusage/codex may not be available. It runs via npx and requires Codex CLI usage logs to exist."
fi

# --- 5. Setup Gemini CLI telemetry ---
echo ""
echo "--- Gemini CLI (telemetry) ---"
GEMINI_DIR="$HOME/.gemini"
SETTINGS_FILE="$GEMINI_DIR/settings.json"
TELEMETRY_LOG="$GEMINI_DIR/telemetry.log"

mkdir -p "$GEMINI_DIR"

if [ ! -f "$SETTINGS_FILE" ]; then
    cat > "$SETTINGS_FILE" << 'GEMEOF'
{
  "telemetry": {
    "enabled": true,
    "exporters": {
      "file": {
        "enabled": true,
        "path": "~/.gemini/telemetry.log"
      }
    }
  }
}
GEMEOF
    echo "[ok] Created $SETTINGS_FILE with telemetry config"
else
    if python3 -c "
import json, sys
with open('$SETTINGS_FILE') as f:
    data = json.load(f)
tel = data.get('telemetry', {})
exp = tel.get('exporters', {}).get('file', {})
sys.exit(0 if exp.get('enabled') and exp.get('path') else 1)
" 2>/dev/null; then
        echo "[ok] Gemini telemetry already configured"
    else
        python3 -c "
import json
with open('$SETTINGS_FILE') as f:
    data = json.load(f)
data.setdefault('telemetry', {})
data['telemetry']['enabled'] = True
data['telemetry'].setdefault('exporters', {})
data['telemetry']['exporters']['file'] = {'enabled': True, 'path': '~/.gemini/telemetry.log'}
with open('$SETTINGS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"
        echo "[ok] Updated $SETTINGS_FILE with telemetry config"
    fi
fi
touch "$TELEMETRY_LOG"

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
echo "  python3 tokenprint.py"
echo ""
echo "Options:"
echo "  --since YYYYMMDD    Start date filter"
echo "  --until YYYYMMDD    End date filter"
echo "  --output PATH       Custom output path"
echo "  --no-open           Don't open in browser"
echo ""
echo "Note: Gemini telemetry only tracks future sessions — no historical backfill."
