#!/usr/bin/env python3
"""PreToolUse guard for the headless rdloop steps (wired in rdloop/<cycle>/settings/<policy>.json).

A no-op unless RDLOOP_STEP is set (the supervisor sets it for its `claude -p` steps only). Runs under
the system python3 (3.8): stdlib only, no 3.9+ syntax.

Policies (RDLOOP_POLICY):
  readonly  auditor / architect / summary: Write/Edit only to RDLOOP_WRITE_FILES; Bash may read and
            run python, never write files (except under /tmp), install, kill, launch or write git.
  critic    readonly + WebFetch / WebSearch (WebFetch only on the RDLOOP_WEB_DOMAINS allowlist).
  scout     no Bash at all; WebFetch on the allowlist; writes only its report.
  engineer  Write/Edit and Bash file operations only inside RDLOOP_WRITE_FILES / RDLOOP_WRITE_PREFIXES
            (module dir, scripts prefix, its impl/ + scratch/ dirs, its TMPDIR); no GPU (CUDA is hidden
            by the supervisor; re-enabling it is denied), no training / eval / launches, read-only git,
            no installs, no network tools, no background jobs.
Every policy: progress-split scores are never read.

Commands are checked per simple command (first word after VAR= prefixes and wrappers such as env /
nice / timeout), so reading files that merely mention a script name (grep, cat) is not blocked. The
hook only sees command text; the supervisor's isolation manifest (sha256 of every file outside the
scope, checked after each engineer step) is the real boundary.
"""
import json
import os
import re
import shlex
import sys

try:
    from urllib.parse import urlparse
except ImportError:  # pragma: no cover
    urlparse = None

WRITE_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
PROGRESS_RX = r"model_merging_[^ /]*/scores\.txt|RTCFake/data/progress\.txt|/wav/progress\b"
SAFE_DST = ("/dev/null", "/dev/stderr", "/dev/stdout")

FIRST_WORD_DENY = {}
for _w in ("sudo", "pip", "pip3", "conda", "mamba", "micromamba", "apt", "apt-get", "dpkg", "npm", "npx", "yarn",
           "pnpm", "uv", "uvx", "cargo", "gem", "brew", "easy_install"):
    FIRST_WORD_DENY[_w] = "installs / sudo (report a missing dependency as needs_user)"
for _w in ("curl", "wget", "ssh", "scp", "sftp", "rsync", "nc", "ncat", "telnet", "ftp", "huggingface-cli"):
    FIRST_WORD_DENY[_w] = "network tools / remote copies"
for _w in ("kill", "pkill", "killall", "nohup", "setsid", "disown", "crontab", "claude", "screen", "tmux", "at"):
    FIRST_WORD_DENY[_w] = "process control / launching sessions"
for _w in ("ln", "dd", "install", "chmod", "chown", "chgrp", "shred", "mkfifo"):
    FIRST_WORD_DENY[_w] = "links / raw writes / permission changes"
for _w in ("loopctl.sh", "rdctl.sh", "supervisor.py"):
    FIRST_WORD_DENY[_w] = "controlling the exploop / rdloop supervisors"

GPU_SCRIPTS = re.compile(r"^(main_train|main_eval|eval_checkpoints|local_eval|b\d+_local_eval|select_topk|make_submission|"
                         r"log_experiment|b\d+_log_ci|probe_report|b\d+_compare_evals|b\d+_build_unseen_set)\.py$")
LAUNCH_SH = re.compile(r"(^|/)(run_\w+|b\d+_smoke\w*|loopctl|rdctl)\.sh$")
TRAIN_MOD = re.compile(r"^[A-Za-z_][\w.]*\.train$|^main_train$|^main_eval$")
CHECKS_MOD = re.compile(r"^[A-Za-z_][\w.]*\.checks$")
GPU_CHECK_TARGETS = {"model", "optim", "gpu", "overfit"}
# assignments only (a leading `CUDA_VISIBLE_DEVICES=...` word is caught per command in unwrap); printing it is fine
GPU_ENV_RX = (r"\bexport\s+CUDA_VISIBLE_DEVICES\s*=|environ\[\s*['\"]CUDA_VISIBLE_DEVICES['\"]\s*\]\s*=|putenv\(\s*['\"]CUDA|"
              r"environ\.(pop|update|setdefault)\([^)]*CUDA|unset\s+CUDA_VISIBLE_DEVICES|\benv\s+(-\w+\s+)*-[ui]\b")
INPLACE_RX = r"(^|[;&|(\s])(sed|perl)\s+([^;&|]*\s)?-[a-zA-Z]*i"
PY_INLINE_RX = r"(python\S*|\$\{?PY\}?)\s+(-\S+\s+)*(-c\s|-\s*<<|<<)"
PY_WRITE_RX = (r"\.(write_text|write_bytes|to_csv|to_json|to_parquet|savefig|unlink|rmdir|touch|mkdir)\(|"
               r"os\.(remove|unlink|rename|replace|rmdir|makedirs|mkdir)|shutil\.(rmtree|move|copy\w*)|np\.save|torch\.save|"
               r"open\([^)]*,\s*(mode\s*=\s*)?\\?['\"](w|a|x|r\+|w\+|wb|ab)")
PY_DELETE_RX = r"\.(unlink|rmdir)\(|os\.(remove|unlink|rename|replace|rmdir)|shutil\.(rmtree|move)"
GIT_READ = {"status", "diff", "log", "show", "ls-files", "grep", "rev-parse", "blame", "cat-file", "describe",
            "shortlog", "ls-tree", "merge-base", "rev-list", "diff-tree", "name-rev", "check-ignore", "--version", "help"}
FILE_OPS = {"rm", "rmdir", "unlink", "mv", "mkdir", "touch", "tee", "cp", "truncate"}
WRAPPERS = {"env", "nice", "time", "timeout", "command", "builtin", "exec", "stdbuf", "ionice", "xargs"}


def deny(reason):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                             "permissionDecisionReason": f"rdloop guard: {reason}"}}))
    sys.exit(0)


def env_list(name):
    return [x for x in os.environ.get(name, "").split(":") if x]


def in_scope(path, exact, prefixes, cwd):
    path = os.path.expanduser(os.path.expandvars(path))
    p = path if os.path.isabs(path) else os.path.join(cwd, path)
    rp = os.path.realpath(p)
    if rp in {os.path.realpath(x) for x in exact}:
        return True
    for pre in prefixes:
        if pre.endswith("/"):
            base = os.path.realpath(pre[:-1])
            if rp == base or rp.startswith(base + "/"):
                return True
        else:
            parent, stem = os.path.split(pre)
            if rp.startswith(os.path.join(os.path.realpath(parent), stem)):
                return True
    return False


def strip_heredocs(cmd):
    """Drop heredoc bodies (python fed to `python - <<EOF`) before shell parsing; the inline-python
    checks still see the full text."""
    out, lines, i = [], cmd.split("\n"), 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", line)
        i += 1
        if m:
            term = m.group(1)
            while i < len(lines) and lines[i].strip() != term:
                i += 1
            i += 1
    return "\n".join(out)


def tokens(cmd):
    text = strip_heredocs(cmd).replace("\n", " ; ").replace("`", " ; ")
    lex = shlex.shlex(text, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    return list(lex)


def simple_commands(toks):
    cmds, cur = [], []
    for t in toks:
        if t in (";", "&&", "||", "|", "&", "(", ")", ";;", "|&", "$(", "$") or set(t) <= set(";&|()"):
            if cur:
                cmds.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        cmds.append(cur)
    return cmds


def redirect_targets(toks):
    out = []
    for i, t in enumerate(toks):
        if t in (">", ">>", ">|", "&>", "&>>", "2>", "1>") and i + 1 < len(toks):
            out.append(toks[i + 1])
    return out


def unwrap(toks):
    """Drop VAR=value prefixes and wrappers (env, nice -n 10, timeout 60, ...)."""
    i = 0
    while i < len(toks):
        t = toks[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t):
            if t.startswith("CUDA_VISIBLE_DEVICES="):
                deny("setting CUDA_VISIBLE_DEVICES (steps are CPU only; the GPU belongs to the training waves)")
            i += 1
            continue
        if os.path.basename(t) in WRAPPERS:
            i += 1
            while i < len(toks) and (toks[i].startswith("-") or re.match(r"^\d+(\.\d+)?[smhd]?$", toks[i])):
                i += 1
            continue
        break
    return toks[i:]


def is_python(word):
    b = os.path.basename(word)
    return b.startswith("python") or word in ("$PY", "${PY}")


def check_python(toks):
    """python -m <mod> / python <script>: no training, no GPU checks, no GPU / registry scripts."""
    args = toks[1:]
    helpish = any(a in ("--help", "-h") for a in args)
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-m" and i + 1 < len(args):
            mod = args[i + 1]
            if TRAIN_MOD.match(mod) and not helpish:
                deny(f"training ({mod}) runs only in the supervisor's GPU session; `--help` is allowed")
            if CHECKS_MOD.match(mod) and i + 2 < len(args) and args[i + 2] in GPU_CHECK_TARGETS:
                deny(f"`{mod} {args[i + 2]}` is a GPU check: the supervisor runs it in its GPU session")
            return
        if a in ("-c", "-") or a.startswith("<<"):
            return
        if a.startswith("-"):
            i += 1
            continue
        if GPU_SCRIPTS.match(os.path.basename(a)) and not helpish:
            deny(f"{os.path.basename(a)} is a GPU / registry job: the supervisor runs those")
        return


def check_bash(cmd, policy, exact, prefixes, cwd):
    if re.search(PROGRESS_RX, cmd):
        deny("progress-split scores are never analysed")
    if re.search(GPU_ENV_RX, cmd):
        deny("re-enabling CUDA (steps are CPU only; the GPU belongs to the training waves)")
    if re.search(INPLACE_RX, cmd):
        deny("in-place edits from Bash (use the Edit tool)")
    if re.search(r"-m\s+(pip|ensurepip)\b", cmd):
        deny("pip installs (report a missing dependency as needs_user)")
    try:
        toks = tokens(cmd)
    except ValueError:
        deny("could not parse the command (unbalanced quotes?); split it into simpler commands")
    cmds = [unwrap(c) for c in simple_commands(toks)]
    inline_py = re.search(PY_INLINE_RX, cmd) is not None
    for c in cmds:
        if not c:
            continue
        word = c[0]
        name = os.path.basename(word)
        if name in FIRST_WORD_DENY:
            deny(f"{FIRST_WORD_DENY[name]} ({name})")
        if name == "git":
            i = 1
            while i < len(c) and c[i].startswith("-") and c[i] not in ("--version",):
                i += 2 if c[i] in ("-C", "-c", "--git-dir", "--work-tree") else 1
            sub = c[i] if i < len(c) else ""
            if sub and sub not in GIT_READ:
                deny(f"git {sub} writes; only read-only git is allowed")
        if name in ("bash", "sh", "source", ".", "zsh") and len(c) > 1 and LAUNCH_SH.search(c[1]):
            deny(f"launching {c[1]} (training / smoke / supervisor control runs only in the supervisor)")
        if LAUNCH_SH.search(word):
            deny(f"launching {word} (training / smoke / supervisor control runs only in the supervisor)")
        if is_python(word):
            check_python(c)
            if any(os.path.basename(a) == "supervisor.py" for a in c[1:2]):
                deny("controlling the exploop / rdloop supervisors")
        if name in FILE_OPS:
            args = [t for t in c[1:] if not t.startswith("-")]
            if name == "truncate":
                args = [t for t in args if not re.match(r"^[+\-<>/%]?\d", t)]
            targets = (args[-1:] if len(args) >= 2 else args) if name == "cp" else args
            for a in targets:
                if policy == "engineer":
                    ok = in_scope(a, exact, prefixes, cwd)
                else:
                    ok = os.path.realpath(os.path.join(cwd, os.path.expandvars(a))).startswith("/tmp/")
                if not ok:
                    deny(f"{name} {a}: outside the write scope "
                         f"{prefixes if policy == 'engineer' else '(this step writes only its assigned file, with the Write tool)'}")
    if inline_py:
        rx = PY_DELETE_RX if policy == "engineer" else PY_WRITE_RX
        if re.search(rx, cmd):
            deny("writing / deleting files from inline python (write files with the Write tool)")
    for dst in redirect_targets(toks):
        if dst in SAFE_DST or dst.startswith("&"):
            continue
        if policy == "engineer":
            if not in_scope(dst, exact, prefixes, cwd):
                deny(f"redirect into {dst}: outside the write scope {prefixes}")
        elif not os.path.realpath(os.path.join(cwd, os.path.expandvars(dst))).startswith("/tmp/"):
            deny(f"redirect into {dst}: this step writes only its assigned file, with the Write tool")


def main():
    if not os.environ.get("RDLOOP_STEP"):
        sys.exit(0)
    try:
        ev = json.load(sys.stdin)
    except ValueError:
        sys.exit(0)
    tool = ev.get("tool_name", "")
    ti = ev.get("tool_input") or {}
    cwd = ev.get("cwd") or os.environ.get("RDLOOP_REPO") or os.getcwd()
    policy = os.environ.get("RDLOOP_POLICY", "readonly")
    exact = env_list("RDLOOP_WRITE_FILES")
    prefixes = env_list("RDLOOP_WRITE_PREFIXES") if policy == "engineer" else []
    if tool in WRITE_TOOLS:
        target = ti.get("file_path") or ti.get("notebook_path") or ""
        if not target or not in_scope(target, exact, prefixes, cwd):
            deny(f"this step may write only {exact + prefixes or 'nothing'}; {target} is not allowed")
        sys.exit(0)
    if tool in ("Read", "Grep", "Glob"):
        blob = " ".join(str(ti.get(k, "")) for k in ("file_path", "path", "glob", "pattern"))
        if re.search(PROGRESS_RX, blob):
            deny("progress-split scores are never analysed")
        sys.exit(0)
    if tool in ("WebFetch", "WebSearch"):
        if policy not in ("scout", "critic"):
            deny(f"{tool} is for the scout and the critic only")
        if tool == "WebFetch":
            url = str(ti.get("url", ""))
            host = (urlparse(url).hostname or "").lower() if urlparse else ""
            domains = [d.strip().lower() for d in os.environ.get("RDLOOP_WEB_DOMAINS", "").split(",") if d.strip()]
            if domains and not any(host == d or host.endswith("." + d) for d in domains):
                deny(f"{host or url} is not on the source allowlist {domains}; cite the paper's arXiv / ISCA / "
                     f"OpenReview / ACL / IEEE / GitHub / Hugging Face page")
        sys.exit(0)
    if tool == "Bash":
        if policy == "scout":
            deny("the scout has no shell; use WebSearch / WebFetch / Read / Grep / Glob")
        if ti.get("run_in_background"):
            deny("background commands are not allowed in an unattended step")
        check_bash(str(ti.get("command", "")), policy, exact, prefixes, cwd)
    sys.exit(0)


if __name__ == "__main__":
    main()
