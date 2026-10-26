#!/usr/bin/env bash
# rdctl.sh - control an rdloop research & development cycle (it feeds exploop waves with new levers).
#
#   scripts/rdloop/rdctl.sh new     <C> --brief "..." [--auto-gate] [--from-wave W] [--levers-max N] [--focus ..] [--exclude ..]
#   scripts/rdloop/rdctl.sh init    <C>                  # state.json + isolation manifest (nothing runs yet)
#   scripts/rdloop/rdctl.sh plan    <C>                  # what runs next; changes nothing
#   scripts/rdloop/rdctl.sh start   <C>                  # start the supervisor in the background (setsid nohup)
#   scripts/rdloop/rdctl.sh status  <C>                  # exit 0 running, 1 down, 2 blocked / needs you, 3 waiting for your review
#   scripts/rdloop/rdctl.sh tail    <C> [n]              # recent events
#   scripts/rdloop/rdctl.sh note    <C> "text"           # steer every later judgement step (user_notes.md)
#   scripts/rdloop/rdctl.sh approve <C> [--levers a,b]   # plan gate: approve the agreed plan (or a subset)
#   scripts/rdloop/rdctl.sh gate    <C> owner|auto       # switch the plan gate
#   scripts/rdloop/rdctl.sh reopen  <C> --at propose|implement ["note"]   # send the plan / code back, then start
#   scripts/rdloop/rdctl.sh pause   <C> | unpause <C>    # no new GPU job while paused
#   scripts/rdloop/rdctl.sh stop    <C>                  # the supervisor exits at its next tick; jobs keep running
#   scripts/rdloop/rdctl.sh resume  <C> [--reset-attempts] [--accept-manifest]  # after BLOCKED / NEEDS_USER / a restart
#   scripts/rdloop/rdctl.sh park    <C>                  # shelve the cycle
#   scripts/rdloop/rdctl.sh verify  <C>                  # the module is exactly the code that passed regression + smoke
#   scripts/rdloop/rdctl.sh handed-off <C> <wave>        # mark the cycle as handed to an exploop wave
#
# <C> is a cycle name under rdloop/ (e.g. B4) or a path to a cycle dir.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${RDLOOP_PY:-/root/.conda/envs/rtc-sdd/bin/python}"
SUP="$REPO/scripts/rdloop/supervisor.py"
export PYTHONDONTWRITEBYTECODE=1

usage() { sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

cmd="${1:-}"; [ -n "$cmd" ] || usage
cyc="${2:-}"; [ -n "$cyc" ] || usage
shift 2 || true
if [ -d "$cyc" ] || [[ "$cyc" == */* ]]; then CD="$(mkdir -p "$cyc" && cd "$cyc" && pwd)"; else CD="$REPO/rdloop/$cyc"; fi
[ -x "$PY" ] || PY=python3
if [ "$cmd" != "new" ]; then
    [ -f "$CD/cycle.json" ] || { echo "no cycle.json in $CD (create it with: rdctl.sh new $(basename "$CD") --brief \"...\")" >&2; exit 1; }
fi

start() {
    if "$PY" "$SUP" alive --cycle "$CD" >/dev/null 2>&1; then
        echo "a supervisor is already running for $(basename "$CD")"; "$PY" "$SUP" status --cycle "$CD" || true; return 0
    fi
    rm -f "$CD/STOP"
    mkdir -p "$CD/logs"
    cd "$REPO"
    # Started from inside a Claude Code session, the supervisor would inherit that session's ids and
    # sockets and hand them to every `claude -p` step. Drop them; auth / provider variables are kept.
    local v
    for v in CLAUDECODE CLAUDE_PID CLAUDE_EFFORT AI_AGENT CLAUDE_CODE_SESSION_ID CLAUDE_CODE_CHILD_SESSION \
             CLAUDE_CODE_SESSION_ATTENDED CLAUDE_CODE_MESSAGING_SOCKET CLAUDE_CODE_MESSAGING_TOKEN \
             CLAUDE_CODE_SSE_PORT CLAUDE_CODE_ENTRYPOINT CLAUDE_CODE_EXECPATH; do
        unset "$v"
    done
    setsid nohup "$PY" -u "$SUP" run --cycle "$CD" </dev/null >>"$CD/logs/supervisor.out" 2>&1 &
    sleep 3
    "$PY" "$SUP" status --cycle "$CD" || true
    echo "supervisor log: $CD/logs/supervisor.out"
    echo "status page:    $CD/STATUS.md   (live view: /rdloop in Claude Code)"
}

case "$cmd" in
    new)     "$PY" "$SUP" new --cycle "$CD" "$@" ;;
    init)    "$PY" "$SUP" init --cycle "$CD" "$@" ;;
    plan)    "$PY" "$SUP" plan --cycle "$CD" ;;
    start)   start ;;
    status)  "$PY" "$SUP" status --cycle "$CD" ;;
    tail)    "$PY" "$SUP" tail --cycle "$CD" -n "${1:-30}" ;;
    note)    [ -n "${1:-}" ] || { echo "note needs a text" >&2; exit 1; }
             printf '\n- %s: %s\n' "$(date '+%Y-%m-%d %H:%M')" "$*" >> "$CD/user_notes.md"
             echo "noted in $CD/user_notes.md (read first by every later judgement step)" ;;
    approve) "$PY" "$SUP" approve --cycle "$CD" "$@"
             "$PY" "$SUP" alive --cycle "$CD" >/dev/null 2>&1 || start ;;
    gate)    [ -n "${1:-}" ] || { echo "gate owner|auto" >&2; exit 1; }
             "$PY" "$SUP" gate "$1" --cycle "$CD" ;;
    reopen)  at=""; note=""
             while [ $# -gt 0 ]; do case "$1" in --at) at="$2"; shift 2 ;; *) note="$note $1"; shift ;; esac; done
             [ -n "$at" ] || { echo "reopen --at propose|implement [\"note\"]" >&2; exit 1; }
             if [ -n "${note// /}" ]; then printf '\n- %s: %s\n' "$(date '+%Y-%m-%d %H:%M')" "${note# }" >> "$CD/user_notes.md"; fi
             "$PY" "$SUP" reopen --cycle "$CD" --at "$at" --note "${note# }"
             "$PY" "$SUP" alive --cycle "$CD" >/dev/null 2>&1 || start ;;
    pause)   touch "$CD/PAUSE"; echo "paused: no new GPU job starts (running jobs continue)" ;;
    unpause) rm -f "$CD/PAUSE"; echo "unpaused" ;;
    stop)    touch "$CD/STOP"; echo "stop requested: the supervisor exits at its next tick; running jobs continue" ;;
    resume)  "$PY" "$SUP" unblock --cycle "$CD" "$@"; rm -f "$CD/STOP"; start ;;
    park)    "$PY" "$SUP" park --cycle "$CD" ;;
    verify)  "$PY" "$SUP" verify --cycle "$CD" ;;
    handed-off) [ -n "${1:-}" ] || { echo "handed-off <C> <wave>" >&2; exit 1; }
             "$PY" "$SUP" handed-off --cycle "$CD" --wave "$1" ;;
    *)       usage ;;
esac
