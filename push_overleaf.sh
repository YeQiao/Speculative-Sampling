#!/usr/bin/env bash
# Push all paper changes to Overleaf.
# Usage: ./push_overleaf.sh [commit message]
#
# Examples:
#   ./push_overleaf.sh                     # uses default message with timestamp
#   ./push_overleaf.sh "add results table"  # custom message

set -euo pipefail
cd "$(dirname "$0")"

MSG="${1:-Update paper $(date '+%Y-%m-%d %H:%M')}"

# Stage everything (new + modified + deleted)
git add -A

# Check if there's anything to commit
if git diff --cached --quiet; then
    echo "Nothing to commit — working tree clean."
    exit 0
fi

echo "=== Changes to push ==="
git diff --cached --stat
echo ""

git commit -m "$MSG"
git push overleaf main:master

echo ""
echo "✓ Pushed to Overleaf."
