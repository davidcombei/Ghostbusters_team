#!/usr/bin/env bash
@@HEADER@@
#
#   bash @@SELF@@                                   # fresh run (the exploop supervisor launches it)
#   bash @@SELF@@ --resume exp/<run dir name>       # after a container restart
#   EXPLOOP_PRINT_ARGV=1 bash @@SELF@@              # print the trainer argv, launch nothing
#
# Rendered from scripts/exploop/launcher_template.sh (same shape as scripts/run_B3a.sh). Extra
# arguments are appended to the trainer command line. Training runs under nohup; stdout/stderr go to
# <workdir>/<track>_<timestamp>.log and the per-epoch summary to <run dir>/logs/train.log.
set -euo pipefail

PY="${PY:-@@PY@@}"
TRACK="@@TRACK@@"
WORKDIR="@@WORKDIR@@"

ARGS=(
@@ARGS@@
)

if [ "${EXPLOOP_PRINT_ARGV:-0}" = "1" ]; then
    printf '%s\n' "${ARGS[@]}"
    exit 0
fi

if [ ! -x "$PY" ]; then
    echo "python not found: $PY (set PY=... to override)" >&2
    exit 1
fi
@@PREFLIGHT@@

cd "$WORKDIR"
LOG="${TRACK}_$(date +%Y%m%d%H%M%S).log"

nohup "$PY" -u @@ENTRY@@ "${ARGS[@]}" "$@" > "$LOG" 2>&1 &

PID=$!
echo "Started $TRACK (pid $PID)"
echo "Log:     $WORKDIR/$LOG"
echo "Follow:  tail -f $WORKDIR/$LOG"
echo "Stop:    kill $PID"
