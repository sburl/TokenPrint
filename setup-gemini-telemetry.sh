#!/bin/bash
# setup-gemini-telemetry.sh
# Enables OpenTelemetry local file logging for Gemini CLI
# so that tokenprint can read token usage data.
#
# Safe to run multiple times (idempotent).

set -e

GEMINI_DIR="$HOME/.gemini"
SETTINGS_FILE="$GEMINI_DIR/settings.json"
TELEMETRY_LOG="$GEMINI_DIR/telemetry.log"

echo "Setting up Gemini CLI telemetry logging..."

# Create .gemini directory if needed
mkdir -p "$GEMINI_DIR"

# If settings.json doesn't exist, create it with telemetry config
if [ ! -f "$SETTINGS_FILE" ]; then
    python3 - "$SETTINGS_FILE" "$TELEMETRY_LOG" <<'PYEOF'
import json, sys
settings_file, log_path = sys.argv[1], sys.argv[2]
data = {"telemetry": {"enabled": True, "exporters": {"file": {"enabled": True, "path": log_path}}}}
with open(settings_file, "w") as f:
    json.dump(data, f, indent=2)
print(f"Created {settings_file} with telemetry config.")
PYEOF
else
    # Check if telemetry config already exists, update if needed
    python3 - "$SETTINGS_FILE" "$TELEMETRY_LOG" <<'PYEOF'
import json, sys
settings_file, log_path = sys.argv[1], sys.argv[2]
with open(settings_file) as f:
    data = json.load(f)
tel = data.get("telemetry", {})
exporters = tel.get("exporters", {})
file_exp = exporters.get("file", {})
if file_exp.get("enabled") and file_exp.get("path"):
    print(f"Telemetry already configured in {settings_file}. No changes needed.")
    sys.exit(0)
data.setdefault("telemetry", {})
data["telemetry"]["enabled"] = True
data["telemetry"].setdefault("exporters", {})
data["telemetry"]["exporters"]["file"] = {"enabled": True, "path": log_path}
with open(settings_file, "w") as f:
    json.dump(data, f, indent=2)
print(f"Updated {settings_file} with telemetry config.")
PYEOF
fi

# Touch the telemetry log so it exists
touch "$TELEMETRY_LOG"

echo ""
echo "Gemini CLI telemetry setup complete."
echo "  Config: $SETTINGS_FILE"
echo "  Log:    $TELEMETRY_LOG"
echo ""
echo "Future Gemini CLI sessions will log token usage to the log file."
echo "Note: Historical data cannot be backfilled - only new sessions will be tracked."
