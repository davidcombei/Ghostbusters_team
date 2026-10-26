#!/usr/bin/env python3
"""PreToolUse guard for the headless exploop steps (wired in scripts/exploop/claude_settings.json).

Active only when EXPLOOP_STEP is set (the supervisor sets it for its claude -p steps), so it is a
no-op anywhere else. Rules:
  - Edit / Write / MultiEdit / NotebookEdit: the target must be one of EXPLOOP_ALLOWED_WRITES
    (colon-separated absolute paths; the analyst's assigned report, nothing for the designer).
  - Bash: no deleting / moving files, no git writes, no installs, no killing, no GPU jobs
    (training / evaluation scripts, launchers), no file writes through redirection or python, no
    reading of progress-split scores.
Denials go back to Claude as a reason; the supervisor additionally diffs protected files around
every step and blocks the wave on any change.
"""
import json
import os
import re
import shlex
import sys

BASH_DENY = [
    (r"(^|[;&|`(]\s*|\s)(/usr)?(/bin/)?(rm|rmdir|unlink|shred|truncate|mv)\s", "deleting / moving / truncating files"),
    (r"\bgit\b[^;&|]*\s(commit|push|add|reset|checkout|restore|clean|stash|rm|mv|rebase|merge|tag)\b", "git write"),
    (r"(^|[;&|`(]\s*|\s)(sudo|pip3?|conda|mamba|micromamba|apt|apt-get|dpkg)\s", "installs / sudo"),
    (r"(^|[;&|`(]\s*)(kill|pkill|killall|nohup|setsid|claude)\s", "process control"),
    (r"(python3?|\$PY|\$\{PY\}|/bin/python\S*)\s+(-\w+\s+)*(-m\s+b3wave\.train|\S*(main_train|main_eval|eval_checkpoints|local_eval|b3_local_eval|select_topk|make_submission|log_experiment|b3_log_ci)\.py)",
     "running a GPU / registry job (only the supervisor runs those)"),
    (r"(^|[;&|`(]\s*)(bash|sh|source|\.)\s+\S*run_\w+\.sh", "launching a training run"),
    (r"model_merging_[^ /]*/scores\.txt|RTCFake/data/progress\.txt|wav/progress", "progress-split scores (never analysed)"),
    (r"\bsed\b[^;&|]*\s-i", "in-place edit"),
    (r"(^|[;&|`(]\s*)(tee|cp|install|rsync|ln|dd)\s", "writing files from Bash"),
    (r"\.(write_text|write_bytes|to_csv|to_json|to_parquet|savefig|unlink|rmdir|touch)\(|os\.(remove|unlink|rename|replace|rmdir|makedirs|mkdir)|shutil\.(rmtree|move|copy\w*)|np\.save|torch\.save",
     "writing / deleting files from python"),
    (r"open\([^)]*,\s*(mode\s*=\s*)?\\?['\"](w|a|x|r\+|w\+|wb|ab)", "writing files from python"),
]
SAFE_DST = ("/dev/null", "/dev/stderr", "/dev/stdout")


def redirect_targets(cmd):
    """Files a shell command redirects output into (quotes respected, so `print(a > b)` inside a
    quoted python snippet is not a redirect). Unparsable commands return [] (the supervisor's
    before/after file diff still catches writes)."""
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError:
        return []
    out = []
    for i, t in enumerate(toks):
        if t in (">", ">>", ">|", "&>", "&>>") and i + 1 < len(toks):
            out.append(toks[i + 1])
    return out


def deny(reason):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                             "permissionDecisionReason": f"exploop guard: {reason}"}}))
    sys.exit(0)


def main():
    if not os.environ.get("EXPLOOP_STEP"):
        sys.exit(0)
    try:
        ev = json.load(sys.stdin)
    except ValueError:
        sys.exit(0)
    tool = ev.get("tool_name", "")
    ti = ev.get("tool_input") or {}
    allowed = {os.path.realpath(p) for p in os.environ.get("EXPLOOP_ALLOWED_WRITES", "").split(":") if p}
    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        target = ti.get("file_path") or ti.get("notebook_path") or ""
        if os.path.realpath(target) not in allowed:
            deny(f"this step may write only {sorted(allowed) or 'nothing'}; {target} is not allowed")
        sys.exit(0)
    if tool == "Bash":
        cmd = ti.get("command", "")
        for rx, why in BASH_DENY:
            if re.search(rx, cmd):
                deny(f"{why} is not allowed in an unattended step (command: {cmd[:160]})")
        for dst in redirect_targets(cmd):
            if dst in SAFE_DST or dst.startswith("/tmp/") or dst.startswith("&"):
                continue
            deny(f"redirecting output into a file ({dst}) is not allowed; use the Write tool on your assigned file only")
    sys.exit(0)


if __name__ == "__main__":
    main()
