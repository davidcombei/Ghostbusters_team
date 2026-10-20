#!/usr/bin/env bash
# loopctl.sh - control an exploop wave (unattended training-exploration loop).
#
#   scripts/exploop/loopctl.sh init   <wave>               # loop_state.json from exploop/<wave>/wave.json
#   scripts/exploop/loopctl.sh plan   <wave>               # what runs next (expanded commands); changes nothing
#   scripts/exploop/loopctl.sh start  <wave>               # start the supervisor in the background (setsid nohup)
#   scripts/exploop/loopctl.sh status <wave>               # exit 0 running, 1 down, 2 blocked, 3 done
#   scripts/exploop/loopctl.sh tail   <wave> [n]           # recent events (epochs, launches, verdicts, ...)
#   scripts/exploop/loopctl.sh note   <wave> "text"        # steer the next DESIGN step (user_notes.md)
#   scripts/exploop/loopctl.sh pause  <wave> | unpause <wave>   # no new GPU job while paused
#   scripts/exploop/loopctl.sh stop   <wave>               # the supervisor exits at its next tick; jobs keep running
#   scripts/exploop/loopctl.sh resume <wave> [--reset-attempts]  # after BLOCKED / DOWN / a container restart
#
# <wave> is a name under exploop/ (e.g. B3) or a path to a wave dir. The supervisor survives a
# closed terminal, a dropped SSH connection and Claude exits; after a container restart run `resume`.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${EXPLOOP_PY:-/root/.conda/envs/rtc-sdd/bin/python}"
SUP="$REPO/scripts/exploop/supervisor.py"

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

cmd="${1:-}"; [ -n "$cmd" ] || usage
wave="${2:-}"; [ -n "$wave" ] || usage
shift 2 || true
if [ -d "$wave" ]; then WD="$(cd "$wave" && pwd)"; else WD="$REPO/exploop/$wave"; fi
[ -f "$WD/wave.json" ] || { echo "no wave.json in $WD" >&2; exit 1; }
[ -x "$PY" ] || PY=python3

start() {
    if "$PY" "$SUP" status --wave "$WD" >/dev/null 2>&1; then
        echo "a supervisor is already running for $(basename "$WD")"; "$PY" "$SUP" status --wave "$WD" || true; return 0
    fi
    rm -f "$WD/STOP"
    mkdir -p "$WD/logs"
    cd "$REPO"
    # Started from inside a Claude Code session (`! loopctl.sh start`, or Claude running it), the
    # supervisor would inherit that session's ids and sockets and hand them to every `claude -p` step,
    # long after the session is gone. Drop them; auth / provider variables are kept.
    local v
    for v in CLAUDECODE CLAUDE_PID CLAUDE_EFFORT AI_AGENT CLAUDE_CODE_SESSION_ID CLAUDE_CODE_CHILD_SESSION \
             CLAUDE_CODE_SESSION_ATTENDED CLAUDE_CODE_MESSAGING_SOCKET CLAUDE_CODE_MESSAGING_TOKEN \
             CLAUDE_CODE_SSE_PORT CLAUDE_CODE_ENTRYPOINT CLAUDE_CODE_EXECPATH; do
        unset "$v"
    done
    setsid nohup "$PY" -u "$SUP" run --wave "$WD" </dev/null >>"$WD/logs/supervisor.out" 2>&1 &
    sleep 3
    "$PY" "$SUP" status --wave "$WD" || true
    echo "supervisor log: $WD/logs/supervisor.out"
    echo "status page:    $WD/STATUS.md"
}

case "$cmd" in
    init)    "$PY" "$SUP" init --wave "$WD" "$@" ;;
    plan)    "$PY" "$SUP" plan --wave "$WD" ;;
    start)   start ;;
    status)  "$PY" "$SUP" status --wave "$WD" ;;
    tail)    "$PY" "$SUP" tail --wave "$WD" -n "${1:-30}" ;;
    note)    [ -n "${1:-}" ] || { echo "note needs a text" >&2; exit 1; }
             printf '\n- %s: %s\n' "$(date '+%Y-%m-%d %H:%M')" "$*" >> "$WD/user_notes.md"
             echo "noted in $WD/user_notes.md (read by the next DESIGN step)" ;;
    pause)   touch "$WD/PAUSE"; echo "paused: no new GPU job starts (running jobs continue)" ;;
    unpause) rm -f "$WD/PAUSE"; echo "unpaused" ;;
    stop)    touch "$WD/STOP"; echo "stop requested: the supervisor exits at its next tick; running jobs continue" ;;
    resume)  "$PY" "$SUP" unblock --wave "$WD" "$@"; rm -f "$WD/STOP"; start ;;
    *)       usage ;;
esac
