"""Helpers for the exploop supervisor: state I/O, argv model and one-factor validator, log parsers,
process / GPU checks, verdict rules and the STATUS.md renderer.

Stdlib only, except `yaml` (PyYAML, present in the rtc-sdd env) for the trainers' config.yaml.
Every number the supervisor shows or writes is read from run files (train.log, metrics.jsonl,
local_eval/*/metrics.json, compare_ci*.json); nothing is typed in by hand.
"""
import csv
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - the rtc-sdd env has PyYAML
    yaml = None

PHASES = ["design", "render", "smoke", "train", "eval", "verdict", "log", "analyze", "decide"]

# ----------------------------------------------------------------------------------------- basics

def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path, default=None):
    p = Path(path)
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    """Atomic write (tmp + replace) so a kill never leaves half a state file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def write_text(path, text):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tail_bytes(path, n=65536):
    """Last n bytes of a (possibly huge, tqdm-heavy) log as text, with \\r turned into \\n."""
    p = Path(path)
    if not p.is_file():
        return ""
    with open(p, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n))
        data = f.read()
    return data.decode("utf-8", "replace").replace("\r", "\n")


def head_text(path, n=200000):
    p = Path(path)
    if not p.is_file():
        return ""
    with open(p, "rb") as f:
        data = f.read(n)
    return data.decode("utf-8", "replace").replace("\r", "\n")


def get_path(obj, dotted):
    """get_path(d, 'half_B.wf1.wf1') -> value or None."""
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# ----------------------------------------------------------------------------------- config.yaml

def read_config(run_dir):
    p = Path(run_dir) / "config.yaml"
    if not p.is_file() or yaml is None:
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def config_argv(run_dir):
    """The trainer argv recorded in config.yaml `command` (tokens after the script path)."""
    cmd = read_config(run_dir).get("command") or ""
    toks = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    return toks[1:] if toks else []


def model_flags(run_dir):
    """Eval flags the post-train-eval skill derives from config.yaml (step 0.1)."""
    args = read_config(run_dir).get("args") or {}
    flags = ["--ssl_name", str(args.get("ssl_name")), "--n_layers", str(args.get("n_layers"))]
    if args.get("loudness_norm"):
        flags += ["--loudness_norm", "--loudness_norm_lufs", fmt_num(args.get("loudness_norm_lufs", -23.0))]
    return flags


def fmt_num(x):
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


# ------------------------------------------------------------------------------------ argv model

def _is_flag(tok):
    return tok.startswith("-") and not re.match(r"^-\d|^-\.\d", tok) and len(tok) > 1


def parse_argv(tokens):
    """Ordered list of (flag, [values]). A token starting with '-' followed by a digit is a value
    (e.g. --loudness_norm_lufs -23). Repeated flags: argparse keeps the last one, so do we."""
    out = []
    cur = None
    for tok in tokens:
        if _is_flag(tok):
            cur = [tok, []]
            out.append(cur)
        elif cur is not None:
            cur[1].append(tok)
        else:
            raise ValueError(f"positional token before any flag: {tok!r}")
    merged = {}
    order = []
    for flag, vals in out:
        if flag not in merged:
            order.append(flag)
        merged[flag] = vals
    return [(f, merged[f]) for f in order]


def argv_dict(tokens):
    return dict(parse_argv(tokens))


def render_argv(pairs):
    toks = []
    for flag, vals in pairs:
        toks.append(flag)
        toks.extend(str(v) for v in vals)
    return toks


def apply_change(base_tokens, change, overrides):
    """base argv + design change {set: {flag: [vals]}, remove: [flags]} + forced overrides."""
    pairs = parse_argv(base_tokens)
    d = dict(pairs)
    order = [f for f, _ in pairs]
    for flag in change.get("remove") or []:
        if flag in d:
            del d[flag]
            order.remove(flag)
    for flag, vals in list((change.get("set") or {}).items()) + list(overrides.items()):
        vals = [str(v) for v in (vals if isinstance(vals, list) else [vals])]
        if flag not in d:
            order.append(flag)
        d[flag] = vals
    return render_argv([(f, d[f]) for f in order])


def changed_flags(base_tokens, cand_tokens, ignore=("--track", "--keep_epochs")):
    a, b = argv_dict(base_tokens), argv_dict(cand_tokens)
    out = []
    for flag in sorted(set(a) | set(b)):
        if flag in ignore:
            continue
        if a.get(flag) != b.get(flag):
            out.append(flag)
    return out


def trainer_flags(source_files):
    """Flags the trainer's argparse knows (grep of add_argument in its source files)."""
    flags = set()
    for src in source_files:
        p = Path(src)
        if p.is_file():
            flags |= set(re.findall(r"add_argument\(\s*[\"'](--?[A-Za-z0-9_\-]+)[\"']", p.read_text(encoding="utf-8")))
    return flags


def validate_design(wave, base_tokens, cand_tokens, design, used_tracks, known_flags):
    """One-factor validator. Returns a list of error strings (empty = ok)."""
    ss = wave["search_space"]
    errs = []
    kind = design.get("kind", "config")
    changed = changed_flags(base_tokens, cand_tokens)
    cand = argv_dict(cand_tokens)
    if not changed:
        errs.append("the candidate argv equals the base argv: nothing changes")
    if kind == "seed_repeat":
        if changed != ["--seed"]:
            errs.append(f"kind=seed_repeat may change only --seed; changed: {changed}")
    else:
        if "--seed" in changed:
            errs.append("--seed may change only with kind=seed_repeat")
        groups = [set(g) for g in ss.get("coupled", [])]
        if len(changed) > 1 and not any(set(changed) <= g for g in groups):
            errs.append(f"more than one factor: {changed} is not a single flag and not inside one coupled group "
                        f"{[sorted(g) for g in groups]}")
    allowed = set(ss.get("allowed", []))
    for flag in changed:
        if flag != "--seed" and flag not in allowed:
            errs.append(f"{flag} is not in search_space.allowed")
        if flag in set(ss.get("forbidden", [])):
            errs.append(f"{flag} is forbidden in this wave")
    for flag in cand:          # forbidden = may not be added or changed (checked above on `changed`)
        if known_flags and flag not in known_flags:
            errs.append(f"{flag} is not a flag of the trainer (typo?)")
    for flag, vals in (ss.get("fixed") or {}).items():
        if cand.get(flag) != [str(v) for v in vals]:
            errs.append(f"fixed flag {flag} must stay {vals}, got {cand.get(flag)}")
    try:
        ne = int((cand.get("--num_epochs") or ["0"])[0])
    except ValueError:
        ne = 0
    if not 1 <= ne <= int(ss.get("max_epochs", 10)):
        errs.append(f"--num_epochs must be in 1..{ss.get('max_epochs', 10)}, got {cand.get('--num_epochs')}")
    if cand.get("--keep_topk_by_wf1") != [str(ss.get("keep_topk_by_wf1", 7))]:
        errs.append(f"--keep_topk_by_wf1 must be {ss.get('keep_topk_by_wf1', 7)}")
    if cand.get("--keep_epochs") != [str(ne)]:
        errs.append(f"--keep_epochs must be the final epoch {ne}")
    if kind != "seed_repeat" and cand.get("--seed") != [str(ss.get("seed", 1234))]:
        errs.append(f"--seed must be {ss.get('seed', 1234)} for a config change")
    track = design.get("track", "")
    pat = rf"^{re.escape(wave['track_prefix'])}{re.escape(design.get('letter', ''))}_[A-Za-z0-9.]+(_[A-Za-z0-9.\-]+)*_s\d+$"
    if not re.match(pat, track):
        errs.append(f"track {track!r} must match {pat}")
    if track in used_tracks:
        errs.append(f"track {track!r} was already used in this wave")
    if cand.get("--track") != [track]:
        errs.append("--track in the argv must equal design.track")
    return errs


# ---------------------------------------------------------------------------------- log parsers

EPOCH_LOSS_RE = re.compile(r"^(?:\[(?P<ts>[\d\- :]+)\] )?Epoch (?P<ep>\d+)/(?P<n>\d+) TrainLoss=(?P<tl>[\d.eE+\-]+) "
                           r"TrainAcc=(?P<ta>[\d.]+)% DevLoss=(?P<dl>[\d.eE+\-]+) DevAcc=(?P<da>[\d.]+)%"
                           r"(?: LR=(?P<lr>[\d.eE+\-]+))?")
EPOCH_WF1_RE = re.compile(r"^(?:\[(?P<ts>[\d\- :]+)\] )?Epoch (?P<ep>\d+) LocalWF1=(?P<wf1>[\d.]+) F1clean=(?P<f1c>[\d.]+) "
                          r"F1noisy=(?P<f1n>[\d.]+) WF1oracle=(?P<wo>[\d.]+) EERnoisy=(?P<eer>[\d.]+)")
# tqdm: "Training:  53%|#####2    | 4999/9474 [32:14<27:17,  2.73batch/s]" (unit batch/s or s/batch; also it/s)
TQDM_RE = re.compile(r"Training:\s+(?P<pct>\d+)%\|[^|]*\|\s*(?P<i>\d+)/(?P<n>\d+) \[(?P<el>[\d:]+)<(?P<rem>[\d:?]+),\s*"
                     r"(?P<rate>[\d.?]+)\s*(?P<unit>batch/s|s/batch|it/s|s/it)")


def parse_train_log(path):
    """{epoch: {...loss fields..., ...wf1 fields...}} from logs/train.log (or a stdout log)."""
    rows = {}
    p = Path(path)
    if not p.is_file():
        return rows
    for line in p.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines():
        m = EPOCH_LOSS_RE.match(line)
        if m:
            r = rows.setdefault(int(m["ep"]), {})
            r.update(epoch=int(m["ep"]), num_epochs=int(m["n"]), train_loss=float(m["tl"]), train_acc=float(m["ta"]),
                     dev_loss=float(m["dl"]), dev_acc=float(m["da"]), lr=m["lr"], ts_loss=m["ts"])
            continue
        m = EPOCH_WF1_RE.match(line)
        if m:
            r = rows.setdefault(int(m["ep"]), {})
            r.update(epoch=int(m["ep"]), wf1=float(m["wf1"]), f1_clean=float(m["f1c"]), f1_noisy=float(m["f1n"]),
                     wf1_oracle=float(m["wo"]), eer_noisy=float(m["eer"]), ts_wf1=m["ts"])
    return rows


def hms_to_s(s):
    if not s or "?" in s:
        return None
    parts = [int(x) for x in s.split(":")]
    sec = 0
    for x in parts:
        sec = sec * 60 + x
    return sec


def parse_tqdm(text):
    """Last 'Training:' progress of a stdout log tail -> dict or None."""
    last = None
    for m in TQDM_RE.finditer(text):
        last = m
    if not last:
        return None
    el, rem = hms_to_s(last["el"]), hms_to_s(last["rem"])
    rate = None if "?" in last["rate"] else float(last["rate"])
    if rate and last["unit"] in ("s/batch", "s/it"):
        rate = 1.0 / rate
    return {"i": int(last["i"]), "n": int(last["n"]), "pct": int(last["pct"]), "elapsed_s": el,
            "remaining_s": rem, "rate_per_s": rate,
            "epoch_s": (el + rem) if (el is not None and rem is not None) else None}


def metrics_epochs(run_dir):
    p = Path(run_dir) / "metrics.jsonl"
    eps = []
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    eps.append(json.loads(line))
                except ValueError:
                    pass
    return eps


def final_epoch(run_dir):
    eps = [r.get("epoch") for r in metrics_epochs(run_dir) if isinstance(r.get("epoch"), int)]
    return max(eps) if eps else None


# ------------------------------------------------------------------------------ processes / GPU

def proc_start_ticks(pid):
    """starttime (field 22 of /proc/<pid>/stat) or None. Used with the pid to re-identify a job."""
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except (OSError, ValueError):
        return None
    rest = data[data.rfind(")") + 2:].split()
    try:
        return int(rest[19])
    except (IndexError, ValueError):
        return None


def proc_alive(pid, start_ticks):
    if not pid:
        return False
    st = proc_start_ticks(pid)
    if st is None:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text()
        state = state[state.rfind(")") + 2:].split()[0]
        if state == "Z":
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            return False
    except OSError:
        return False
    return start_ticks is None or st == start_ticks


def proc_cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def pid1_start():
    """Identity of the container boot: boot time + PID 1 starttime. Changes on a container restart."""
    btime = None
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                btime = int(line.split()[1])
    except OSError:
        pass
    return f"{btime}:{proc_start_ticks(1)}"


def busy_processes(regex, exclude_pids=()):
    me = {os.getpid(), *exclude_pids}
    out = []
    rx = re.compile(regex)
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in me:
            continue
        cmd = proc_cmdline(int(d.name))
        if not cmd or "pgrep" in cmd.split(" ")[0]:
            continue
        # only python processes count (an editor or grep mentioning local_eval does not hold the GPU)
        first = cmd.split(" ")[0]
        if "python" not in os.path.basename(first):
            continue
        if rx.search(cmd):
            out.append((int(d.name), cmd[:200]))
    return out


def gpu_memory_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    vals = []
    for line in out.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            vals.append(int(parts[1]))
    return vals


def disk_free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return None


# --------------------------------------------------------------------------------- verdict rules

def readout_numbers(metrics_path, key, clean_key):
    m = read_json(metrics_path)
    if not m:
        return None
    v, c = get_path(m, key), get_path(m, clean_key)
    if v is None:
        return None
    return {"value": 100 * v, "clean": None if c is None else 100 * c}


def ci_numbers(ci_path):
    c = read_json(ci_path)
    if not c or not c.get("wf1"):
        return None
    w = c["wf1"]
    return {"delta": 100 * w["delta"], "ci_low": 100 * w["ci_low"], "ci_high": 100 * w["ci_high"]}


def decide(run, base, ci, rules, higher_is_better=True):
    """Encodes the approved B3 plan (thresholds in wave.json rules):
    new best = improvement > new_best_delta; registry decision promote / revisit / drop;
    pre-registered verdict positive / null / negative (+-band).
    rules.keep_best (screen waves): the best never moves (every arm vs the start best), but the
    decision still reads as if it could, so a screen arm is labelled on its own merits."""
    imp = (run["value"] - base["value"]) if higher_is_better else (base["value"] - run["value"])
    clean_drop = None
    if run.get("clean") is not None and base.get("clean") is not None:
        clean_drop = base["clean"] - run["clean"]
    ci_low = ci["ci_low"] if ci else None
    if not higher_is_better and ci:
        ci_low = -ci["ci_high"]
    improved = imp > float(rules.get("new_best_delta", 0.0))
    new_best = improved and not rules.get("keep_best", False)
    pr = rules.get("promote", {})
    ok_ci = ci_low is not None and ci_low > float(pr.get("ci_low_gt", 0.0))
    ok_clean = clean_drop is None or clean_drop <= float(pr.get("clean_drop_le", 0.5))
    if improved and ok_ci and ok_clean:
        decision = "promote"
    elif improved:
        decision = "revisit"
    else:
        decision = "drop"
    band = float(rules.get("band", 0.5))
    pos = rules.get("positive", {})
    if imp > float(pos.get("delta_gt", band)) and ok_ci and ok_clean:
        verdict = "positive"
    elif imp > band:
        verdict = "above band, " + ("CI includes 0" if not ok_ci else "clean F1 dropped")
    elif imp < -band:
        verdict = "negative"
    else:
        verdict = "null (inside +-%g)" % band
    return {"run": run["value"], "base": base["value"], "delta": imp, "ci_low": ci_low,
            "ci_high": (ci["ci_high"] if ci and higher_is_better else (-ci["ci_low"] if ci else None)),
            "clean_drop": clean_drop, "new_best": new_best, "decision": decision, "verdict": verdict}


# ---------------------------------------------------------------------------------- registry

def registry_rows(path):
    p = Path(path)
    if not p.is_file():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def registry_row(path, rid):
    for r in registry_rows(path):
        if r.get("id") == rid:
            return r
    return None


# ------------------------------------------------------------------------------- STATUS.md

def _f(x, nd=2, sign=False):
    if x is None:
        return "–"
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")


def render_status(wave, st, health, epoch_rows, progress, best_epoch_wf1, wave_dir):
    w = wave["wave"]
    lines = []
    cur = st.get("current")
    it = next((i for i in st["iterations"] if i["k"] == cur), None)
    phase = (it or {}).get("phase", "-")
    head = f"# exploop wave {w} - {st['status']}"
    if it and st["status"] not in ("DONE",):
        head += f" (iteration {it['k']}/{wave['max_iterations']} {it.get('label', '')}, phase {phase.upper()})"
    lines += [head, ""]
    hb = st.get("heartbeat_ts")
    lines.append(f"Updated {now_ts()} - supervisor pid {st.get('supervisor_pid')} - heartbeat {hb} - "
                 f"container {health.get('container', '?')} - Claude spend ${st.get('cost_usd', 0):.2f} "
                 f"of ${wave['claude'].get('max_total_usd', 0)}")
    if st.get("blocked"):
        b = st["blocked"]
        lines += ["", f"> **BLOCKED in {b.get('phase')}** ({b.get('ts')}): {b.get('reason')}",
                  f"> Hint: {b.get('hint', '')}",
                  f"> After fixing: `scripts/exploop/loopctl.sh resume {w}`"]
    if st.get("waiting"):
        lines += ["", f"> Waiting: {st['waiting']}"]
    if health.get("warnings"):
        lines += [""] + [f"> {x}" for x in health["warnings"]]
    b = st.get("best") or {}
    lines += ["", "## Best so far", "",
              "| label | run | merge (objective) | final epoch | registry ids | submission |",
              "|---|---|---|---|---|---|"]
    lines.append(f"| {b.get('label')} | `{Path(b.get('run_dir', '')).name}` | {_f(b.get('merge_value'))} | "
                 f"{_f(b.get('final_value'))} | {', '.join((b.get('registry_ids') or {}).values())} | "
                 f"{b.get('submission') or '–'} |")
    lines += ["", "## Iterations", "",
              "| # | track | change | status | merge | delta | CI | final | delta | decision | verdict | report |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i in st["iterations"]:
        v = i.get("verdict") or {}
        m, f = v.get("merge") or {}, v.get("final") or {}
        ci = "–" if m.get("ci_low") is None else f"[{m['ci_low']:+.2f}, {m['ci_high']:+.2f}]"
        rep = f"[report]({os.path.relpath(i['report'], wave_dir)})" if i.get("report") and Path(i["report"]).exists() else "–"
        lines.append(f"| {i['k']} | `{i.get('track', '?')}` | {i.get('change_vs_ref', '')} | "
                     f"{'done' if i.get('done') else i.get('phase', '').upper()} | {_f(m.get('run'))} | "
                     f"{_f(m.get('delta'), sign=True)} | {ci} | {_f(f.get('run'))} | {_f(f.get('delta'), sign=True)} | "
                     f"{m.get('decision', '–')} | {m.get('verdict', '–')} | {rep} |")
    if it and epoch_rows:
        lines += ["", f"## Current run {it.get('label')}: per epoch (logs/train.log; delta = vs best {b.get('label')} at the same epoch)", "",
                  "| ep | TrainLoss | DevLoss | LocalWF1 | delta | F1clean | F1noisy | WF1oracle | EERnoisy | time |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        best_so_far = None
        for ep in sorted(epoch_rows):
            r = epoch_rows[ep]
            ref = best_epoch_wf1.get(ep)
            d = (r["wf1"] - ref) if (r.get("wf1") is not None and ref is not None) else None
            if r.get("wf1") is not None and (best_so_far is None or r["wf1"] > best_so_far[1]):
                best_so_far = (ep, r["wf1"])
            lines.append(f"| {ep} | {_f(r.get('train_loss'), 4)} | {_f(r.get('dev_loss'), 4)} | {_f(r.get('wf1'))} | "
                         f"{_f(d, sign=True)} | {_f(r.get('f1_clean'))} | {_f(r.get('f1_noisy'))} | "
                         f"{_f(r.get('wf1_oracle'))} | {_f(r.get('eer_noisy'))} | {r.get('ts_wf1') or r.get('ts_loss') or ''} |")
        if best_so_far:
            lines.append(f"\nBest epoch so far: {best_so_far[0]} (LocalWF1 {best_so_far[1]:.2f}).")
    if progress:
        eta = ""
        if progress.get("eta_h") is not None:
            eta = f", run ETA ~{progress['eta_h']:.1f} h"
        rate = progress.get("rate_per_s")
        lines.append(f"\nProgress: epoch {progress.get('epoch', '?')}, batch {progress['i']}/{progress['n']} "
                     f"({progress['pct']}%), {('%.2f batch/s' % rate) if rate else '? batch/s'}{eta}.")
    if st.get("next"):
        lines += ["", "## Next", "", st["next"]]
    steps = [(i["k"], s) for i in st["iterations"] for s in i.get("claude_sessions", [])] + \
            [(0, s) for s in st.get("claude_sessions", [])]
    if steps:
        lines += ["", "## Claude steps (inspect with `claude --resume <session>`)", ""]
        for k, s in steps[-12:]:
            lines.append(f"- iter {k} {s['step']} attempt {s['attempt']}: `{s['session_id']}` "
                         f"({s.get('status', '?')}, ${s.get('cost', 0):.2f})")
    lines += ["", "## Commands", "", "```",
              f"scripts/exploop/loopctl.sh status {w}      # health + exit code",
              f"scripts/exploop/loopctl.sh tail {w}        # recent events",
              f"scripts/exploop/loopctl.sh note {w} \"...\"  # steer the next design step",
              f"scripts/exploop/loopctl.sh pause {w}       # no new GPU job (running ones continue)",
              f"scripts/exploop/loopctl.sh stop {w}        # supervisor exits; running jobs continue",
              f"scripts/exploop/loopctl.sh resume {w}      # after BLOCKED / DOWN / container restart",
              "```", ""]
    return "\n".join(lines)
