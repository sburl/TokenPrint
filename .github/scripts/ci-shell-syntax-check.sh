#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

scripts=()
while IFS= read -r script; do
  scripts+=("$script")
done < <(git ls-files '*.sh')
if (( ${#scripts[@]} == 0 )); then
  echo "No shell scripts found; skipping shell syntax check."
  exit 0
fi

status=0
for script in "${scripts[@]}"; do
  if ! bash -n "$script"; then
    echo "Shell syntax check failed for: $script"
    status=1
  else
    echo "Shell syntax check passed: $script"
  fi
done

exit $status
