#!/usr/bin/env bash
set -euo pipefail

# Lightweight in-repo secret pattern check used in nightly CI.
cd "$(dirname "$0")/../.."

if rg -n --hidden --glob '!.git' --glob '!.github/scripts/ci-secret-scan.sh' \
  -e 'AKIA[0-9A-Z]{16}' \
  -e 'ghp_[A-Za-z0-9]{36}' \
  -e 'sk-[A-Za-z0-9]{48}' \
  -e '-----BEGIN PRIVATE KEY-----' \
  -e 'BEGIN RSA PRIVATE KEY' \
  -e 'BEGIN OPENSSH PRIVATE KEY'; then
  echo "Potential secrets detected in repository"
  exit 1
fi

echo "Secret scan complete (no obvious secrets found)"
