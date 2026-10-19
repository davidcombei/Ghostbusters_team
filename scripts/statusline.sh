#!/usr/bin/env bash
# Claude Code status line for this repo: the exploop wave view (scripts/exploop/view.py line, unchanged)
# plus one rdloop row while a research cycle is live or waiting for the owner (scripts/rdloop/view.py line).
# Reads the statusLine JSON from stdin once and feeds it to both; never fails.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
in="$(cat 2>/dev/null || true)"
printf '%s' "$in" | python3 "$REPO/scripts/exploop/view.py" line 2>/dev/null || true
printf '%s' "$in" | python3 "$REPO/scripts/rdloop/view.py" line 2>/dev/null || true
exit 0
