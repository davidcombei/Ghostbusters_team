#!/usr/bin/env python3
"""exploop view: a read-only snapshot of a wave, for Claude Code's status line and the /exploop pane.

    view.py snapshot [--wave B3]   # JSON: phase, progress, current-vs-best epochs, eval steps, Claude step, events
    view.py line     [--wave B3]   # 2 coloured lines for Claude Code's statusLine (reads its JSON on stdin)

Wave: --wave, else $EXPLOOP_WAVE, else the wave with the newest heartbeat that is not DONE (or one that
finished in the last 24 h); with none, `line` prints nothing.

Read-only by construction: it reads loop_state.json, wave.json, events.jsonl, the trainer logs, the best
run's train.log and the headless Claude transcripts. It never takes .supervisor.lock (a test-lock could
race `loopctl start`), never writes, and never instantiates Supervisor (whose save() writes).
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exploop_lib as L  # noqa: E402

REPO = HERE.parents[1]
PHASES = L.PHASES
CLAUDE_PROJECTS = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / "projects"
SPARK = "▁▂▃▄▅▆▇█"
# any tqdm bar ("Training:  40%|###9 | 3753/9474 [24:59<33:41,  2.83batch/s]") and the per-epoch log lines
ANY_TQDM_RE = re.compile(r"(?P<label>Training|Validating):\s+(?P<pct>\d+)%\|[^|]*\|\s*(?P<i>\d+)/(?P<n>\d+)")
EPOCH_END_RE = re.compile(r"Epoch \d+/\d+ TrainLoss=")
EPOCH_EVAL_RE = re.compile(r"Epoch \d+ LocalWF1=")
TOAST_KINDS = ("phase", "epoch", "new_best", "iteration_done", "early_check_failed", "train_died", "blocked",
               "done", "eval_failed", "design_rejected", "redesign", "container_restart", "supervisor_exit")


# ------------------------------------------------------------------------------------- helpers

def _abs(p, repo=REPO):
    p = Path(str(p))
    return p if p.is_absolute() else repo / p


def _ts_epoch(ts):
    """'2026-10-10 14:56:08' (local, as the supervisor and trainer write it) -> epoch seconds or None."""
    if not ts:
        return None
    try:
        return datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError:
        return None


def sparkline(values, lo=None, hi=None):
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    span = (hi - lo) or 1.0
    return "".join(" " if v is None else SPARK[max(0, min(7, int((v - lo) / span * 7 + 0.5)))] for v in values)


def wave_dir_of(name):
    p = Path(name)
    if p.is_dir() and (p / "wave.json").is_file():
        return p.resolve()
    return REPO / "exploop" / name


def pick_wave(name=None, root=None):
    """The wave dir to show, or None."""
    name = name or os.environ.get("EXPLOOP_WAVE")
    if name:
        return wave_dir_of(name)
    root = Path(root) if root else REPO / "exploop"
    live, done = [], []
    for sp in root.glob("*/loop_state.json"):
        st = L.read_json(sp) or {}
        hb = st.get("heartbeat_epoch") or sp.stat().st_mtime
        (done if st.get("status") == "DONE" else live).append((hb, sp.parent))
    if live:
        return max(live)[1]
    recent = [x for x in done if time.time() - x[0] < 24 * 3600]
    return max(recent)[1] if recent else None


def read_events(path, n=20):
    """(total line count, last n events each with its 1-based line number as `seq`)."""
    p = Path(path)
    if not p.is_file():
        return 0, []
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for seq in range(max(0, len(lines) - n), len(lines)):
        try:
            e = json.loads(lines[seq])
        except ValueError:
            continue
        e["seq"] = seq + 1
        out.append(e)
    return len(lines), out


def supervisor_liveness(st, wave, wave_dir):
    pid = st.get("supervisor_pid")
    cmd = L.proc_cmdline(pid) if pid else ""
    pid_alive = bool(pid) and "supervisor.py" in cmd and L.proc_start_ticks(pid) is not None
    hb = st.get("heartbeat_epoch")
    age = (time.time() - hb) if hb else None
    poll = float((wave.get("gpu") or {}).get("poll_s", 60))
    if pid_alive:
        state = "alive" if (age is not None and age <= 3 * poll + 30) else "stale"
    else:
        state = "exited" if st.get("status") in ("DONE", "STOPPED", "INITIALISED") else "down"
    return {"state": state, "pid": pid, "heartbeat_age_s": None if age is None else round(age),
            "container_restarted": bool(st.get("pid1_start") and st["pid1_start"] != L.pid1_start()),
            "paused": (wave_dir / "PAUSE").exists(), "stop_requested": (wave_dir / "STOP").exists()}


def train_progress(it, rows, num_epochs):
    """Live position inside the run from the trainer's stdout tail (fresher than st.progress)."""
    if not it.get("logs"):
        return None
    tail = L.tail_bytes(it["logs"][-1], 65536)
    prog = L.parse_tqdm(tail) or {}
    marks = {"train": -1, "validate": -1, "local eval": -1}
    vbar = None
    for m in ANY_TQDM_RE.finditer(tail):
        if m["label"] == "Training":
            marks["train"] = m.start()
        else:
            marks["validate"] = m.start()
            vbar = m
    for m in EPOCH_END_RE.finditer(tail):
        marks["local eval"] = m.start()
    last_eval = max((m.start() for m in EPOCH_EVAL_RE.finditer(tail)), default=-1)
    sub = max(marks, key=marks.get) if max(marks.values()) >= 0 else None
    if sub == "local eval" and last_eval > marks["local eval"]:
        sub = "train"            # the epoch's local eval landed; the next epoch's first batches are not logged yet
    done_eps = max([ep for ep, r in rows.items() if r.get("wf1") is not None], default=0)
    out = {"epoch": min(done_eps + 1, num_epochs), "num_epochs": num_epochs, "epochs_done": done_eps, "substep": sub,
           "i": prog.get("i"), "n": prog.get("n"), "pct": prog.get("pct"), "rate_per_s": prog.get("rate_per_s"),
           "epoch_s": prog.get("epoch_s"), "remaining_s": prog.get("remaining_s"), "eta_h": None,
           "log_age_s": None}
    if sub == "validate" and vbar:
        out["validate"] = {"i": int(vbar["i"]), "n": int(vbar["n"]), "pct": int(vbar["pct"])}
    if prog.get("epoch_s"):     # supervisor.monitor_train's formula
        rem = prog.get("remaining_s") or 0
        if sub != "train":
            rem = 0
        out["eta_h"] = rem / 3600 + max(0, num_epochs - done_eps - 1) * prog["epoch_s"] / 3600 * 1.03
    try:
        out["log_age_s"] = round(time.time() - Path(it["logs"][-1]).stat().st_mtime)
    except OSError:
        pass
    return out


def epoch_pace_s(rows):
    """Mean wall time between consecutive per-epoch local-eval lines (train + validate + local eval)."""
    ts = [(ep, _ts_epoch(r.get("ts_wf1") or r.get("ts_loss"))) for ep, r in sorted(rows.items())]
    ts = [(ep, t) for ep, t in ts if t is not None]
    if len(ts) < 2:
        return None
    return (ts[-1][1] - ts[0][1]) / (ts[-1][0] - ts[0][0])


def compare_block(rows, best_rows, best, prog, num_epochs):
    """Current run vs the wave's best at the same epoch (per-epoch LocalWF1 on the dev mini sets)."""
    keys = ("wf1", "f1_clean", "f1_noisy", "eer_noisy", "wf1_oracle", "dev_loss", "train_loss")
    table = []
    for ep in sorted(rows):
        r, b = rows[ep], best_rows.get(ep, {})
        if r.get("wf1") is None:
            continue
        row = {"epoch": ep}
        for k in keys:
            row[k] = r.get(k)
            row["best_" + k] = b.get(k)
            row["d_" + k] = (r[k] - b[k]) if (r.get(k) is not None and b.get(k) is not None) else None
        row["ts"] = r.get("ts_wf1")
        table.append(row)
    ahead = sum(1 for r in table if r["d_wf1"] is not None and r["d_wf1"] > 0)
    judged = sum(1 for r in table if r["d_wf1"] is not None)
    best_wf1 = {ep: r.get("wf1") for ep, r in best_rows.items() if r.get("wf1") is not None}
    peak = max(best_wf1.items(), key=lambda x: x[1]) if best_wf1 else None
    cur_peak = max(((r["epoch"], r["wf1"]) for r in table), key=lambda x: x[1]) if table else None
    epoch_now = (prog or {}).get("epoch") if prog else None
    lo_hi = [v for v in [r["wf1"] for r in table] + list(best_wf1.values()) if v is not None]
    lo, hi = (min(lo_hi), max(lo_hi)) if lo_hi else (None, None)
    n_best = max(best_wf1) if best_wf1 else 0
    return {
        "best_label": best.get("label"),
        "best_run": Path(best.get("run_dir") or "").name or None,
        "best_merge": best.get("merge_value"), "best_final": best.get("final_value"),
        "best_registry_ids": best.get("registry_ids") or {},
        "rows": table, "last": table[-1] if table else None,
        "epoch_now": epoch_now, "num_epochs": num_epochs,
        "target_now": best_wf1.get(epoch_now) if epoch_now else None,
        "best_peak": {"epoch": peak[0], "wf1": peak[1]} if peak else None,
        "current_peak": {"epoch": cur_peak[0], "wf1": cur_peak[1]} if cur_peak else None,
        "ahead": ahead, "judged": judged,
        "pace_s": epoch_pace_s(rows) or (prog or {}).get("epoch_s"), "best_pace_s": epoch_pace_s(best_rows),
        "spark": sparkline([r["wf1"] for r in table], lo, hi),
        "best_spark": sparkline([best_wf1.get(ep) for ep in range(1, n_best + 1)], lo, hi),
    }


def profile_steps(wave):
    ev = wave.get("eval") or {}
    if ev.get("pipeline"):
        prof = ev
    else:
        name = ev.get("profile", "sls_local17")
        p = Path(name) if name.endswith(".json") else HERE / "profiles" / f"{name}.json"
        prof = L.read_json(_abs(p)) or {}
    return [{"name": s["name"], "gpu": bool(s.get("gpu")), "optional": bool(s.get("optional"))}
            for s in prof.get("steps", [])]


def eval_progress(it, steps, events):
    """The supervisor runs eval steps in profile order and stops at the first one not done on disk,
    so the running step's index is the number done; without a running job, the last eval_* event says."""
    names = [s["name"] for s in steps]
    running = None
    for name, job in (it.get("jobs") or {}).items():
        if name.startswith("eval:") and L.proc_alive(job.get("pid"), job.get("start_ticks")):
            running = {"name": name[5:], "pid": job.get("pid"), "elapsed_s": round(time.time() - job.get("started", time.time())),
                       "log": job.get("log"), "attempt": job.get("attempt")}
    if running and running["name"] in names:
        n_done = names.index(running["name"])
    else:
        n_done = 0
        for e in events:
            if e.get("iter") == it.get("k") and e.get("kind") in ("eval_step", "eval_skipped", "eval_cache"):
                nm = e.get("msg", "").split()[0].rstrip(":")
                nm = nm if nm in names else e.get("msg", "").replace("optional step ", "").split()[0]
                if nm in names:
                    n_done = max(n_done, names.index(nm) + (0 if e["kind"] == "eval_cache" else 1))
            if e.get("iter") == it.get("k") and e.get("kind") == "eval_done":
                n_done = len(names)
    skipped = set(it.get("eval_skipped") or [])
    out = []
    for i, s in enumerate(steps):
        state = "done" if i < n_done else ("running" if running and s["name"] == running["name"] else "todo")
        if s["name"] in skipped:
            state = "skipped"
        out.append(dict(s, state=state))
    return {"steps": out, "done": n_done, "total": len(names), "running": running}


def _short_input(name, inp):
    inp = inp or {}
    if name == "Bash":
        return inp.get("description") or (inp.get("command") or "")[:80]
    for k in ("file_path", "path", "notebook_path"):
        if inp.get(k):
            return os.path.relpath(inp[k], REPO) if os.path.isabs(inp[k]) else inp[k]
    for k in ("pattern", "query", "url", "skill"):
        if inp.get(k):
            return str(inp[k])[:80]
    return ""


def claude_activity(step, job):
    """What a running headless claude step is doing, from its live transcript."""
    sid = job.get("session_id")
    out = {"step": step, "agent": None, "session_id": sid, "pid": job.get("pid"), "attempt": job.get("attempt"),
           "elapsed_s": round(time.time() - job.get("started", time.time())), "tool_uses": 0, "last_tool": None,
           "last_target": None, "last_ts": None}
    m = re.search(r"--agent\s+(\S+)", " ".join(map(str, job.get("cmd") or [])))
    out["agent"] = m.group(1) if m else {"design": "exp-designer"}.get(step, "exp-analyst")
    if not sid:
        return out
    paths = list(CLAUDE_PROJECTS.glob(f"*/{sid}.jsonl"))
    if not paths:
        return out
    p = paths[0]
    text = p.read_text(encoding="utf-8", errors="replace") if p.stat().st_size < 8 << 20 else L.tail_bytes(p, 4 << 20)
    for line in text.splitlines():
        if '"tool_use"' not in line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") != "assistant":
            continue
        for b in (d.get("message") or {}).get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                out["tool_uses"] += 1
                out["last_tool"], out["last_target"] = b.get("name"), _short_input(b.get("name"), b.get("input"))
                out["last_ts"] = d.get("timestamp")
    return out


# ------------------------------------------------------------------------------------ snapshot

def snapshot(wave_dir):
    wave_dir = Path(wave_dir)
    snap = {"wave": wave_dir.name, "wave_dir": str(wave_dir), "ok": False, "error": None, "taken": L.now_ts()}
    try:
        _fill(snap, wave_dir)
        snap["ok"] = True
    except Exception as exc:  # a viewer must never take the status line down
        snap["error"] = f"{type(exc).__name__}: {exc}"
    return snap


def _fill(snap, wave_dir):
    wave = L.read_json(wave_dir / "wave.json")
    if not wave:
        raise FileNotFoundError(f"no readable wave.json in {wave_dir}")
    st = L.read_json(wave_dir / "loop_state.json")
    if not st:
        raise FileNotFoundError(f"no readable loop_state.json in {wave_dir} (not initialised?)")
    repo = Path(wave.get("repo") or REPO)
    snap["wave"] = wave.get("wave", wave_dir.name)
    best = st.get("best") or {}
    n_iter = int(wave.get("max_iterations", 0))
    smoke_on = wave.get("smoke", "off") not in ("off", False, None)
    snap.update(status=st.get("status"), waiting=st.get("waiting"), blocked=st.get("blocked"), next=st.get("next"),
                warnings=list(st.get("warnings") or [])[-3:], max_iterations=n_iter,
                phases=[p for p in PHASES if smoke_on or p != "smoke"],
                cost_usd=round(float(st.get("cost_usd") or 0), 2),
                max_total_usd=(wave.get("claude") or {}).get("max_total_usd"),
                best={k: best.get(k) for k in ("label", "iter", "run_dir", "merge_value", "final_value", "registry_ids",
                                               "submission")},
                supervisor=supervisor_liveness(st, wave, wave_dir),
                status_page=os.path.relpath(wave_dir / "STATUS.md", repo),
                summary_report=st.get("summary_report"))
    total, events = read_events(wave_dir / "events.jsonl", 200)
    snap["events_total"] = total
    snap["events"] = []
    for e in events[-20:]:
        rec = {k: e.get(k) for k in ("seq", "ts", "iter", "phase", "kind", "msg")}
        if e.get("kind") == "epoch" and e.get("data"):
            d = e["data"]
            rec["data"] = {"epoch": d.get("epoch"), "delta_vs_best": d.get("delta_vs_best"),
                           "wf1": (d.get("row") or {}).get("wf1")}
        snap["events"].append(rec)
    snap["iterations"] = []
    for i in st.get("iterations", []):
        v = i.get("verdict") or {}
        m, f = v.get("merge") or {}, v.get("final") or {}
        snap["iterations"].append({
            "k": i["k"], "label": i.get("label"), "track": i.get("track"), "change": i.get("change_vs_ref"),
            "phase": i.get("phase"), "done": bool(i.get("done")), "adopted": bool(i.get("adopted")),
            "merge": m.get("run"), "merge_delta": m.get("delta"), "ci_low": m.get("ci_low"), "ci_high": m.get("ci_high"),
            "final": f.get("run"), "final_delta": f.get("delta"), "decision": m.get("decision"),
            "new_best": bool(m.get("new_best") or f.get("new_best")), "report": i.get("report")})
    cur = st.get("current")
    it = next((i for i in st.get("iterations", []) if i["k"] == cur), None)
    snap["current"] = None
    if cur is None and st.get("status") != "INITIALISED":
        job = (st.get("jobs") or {}).get("summary")
        snap["wave_phase"] = "done" if st.get("status") == "DONE" else "summary"
        if job and L.proc_alive(job.get("pid"), job.get("start_ticks")):
            snap["claude"] = claude_activity("summary", job)
        return
    if not it:
        snap["current"] = {"k": cur, "label": f"{wave.get('track_prefix', '')}{wave.get('letters', 'abcdefghij')[cur - 1]}"
                           if cur else None, "phase": "design", "phase_index": 0}
        return
    num_epochs = int((L.argv_dict(it.get("argv") or []).get("--num_epochs") or ["8"])[0]) if it.get("argv") else 8
    phase = it.get("phase", "design")
    c = {"k": it["k"], "label": it.get("label"), "track": it.get("track"), "phase": phase,
         "phase_index": snap["phases"].index(phase) if phase in snap["phases"] else None,
         "change": it.get("change_vs_ref"), "kind": it.get("kind"), "lever": it.get("lever_slug"),
         "run_dir": it.get("run_dir"), "run_name": Path(it["run_dir"]).name if it.get("run_dir") else None,
         "launcher": it.get("launcher"), "early_ok": it.get("early_ok"), "num_epochs": num_epochs,
         "jobs": {n: {"pid": j.get("pid"), "alive": L.proc_alive(j.get("pid"), j.get("start_ticks")),
                      "started": j.get("started_ts")} for n, j in (it.get("jobs") or {}).items()}}
    snap["current"] = c
    rows = L.parse_train_log(_abs(it["run_dir"], repo) / "logs" / "train.log") if it.get("run_dir") else {}
    best_rows = L.parse_train_log(_abs(best["run_dir"], repo) / "logs" / "train.log") if best.get("run_dir") else {}
    prog = train_progress(it, rows, num_epochs) if phase == "train" and not it.get("train_done") else None
    c["progress"] = prog
    if rows or best_rows:
        snap["compare"] = compare_block(rows, best_rows, best, prog, num_epochs)
    if phase in ("eval", "verdict", "log", "analyze", "decide") or any(n.startswith("eval:") for n in it.get("jobs", {})):
        c["eval"] = eval_progress(it, profile_steps(wave), events)
    for step in ("design", "analyze"):
        job = (it.get("jobs") or {}).get(step)
        if job and L.proc_alive(job.get("pid"), job.get("start_ticks")):
            snap["claude"] = claude_activity(step, job)


# --------------------------------------------------------------------------------- status line

C = {"red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36", "dim": "2", "bold": "1"}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def col(text, *styles):
    codes = ";".join(C[s] for s in styles if s in C)
    return f"\x1b[{codes}m{text}\x1b[0m" if codes else text


def visible_len(s):
    return len(_ANSI_RE.sub("", s))


def fit(s, width):
    """Cut an ANSI-coloured string to `width` visible cells."""
    if visible_len(s) <= width:
        return s
    out, n, i = [], 0, 0
    while i < len(s) and n < width - 1:
        m = _ANSI_RE.match(s, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        out.append(s[i])
        n += 1
        i += 1
    return "".join(out) + "…\x1b[0m"


def _dur(s):
    if s is None:
        return "?"
    s = int(s)
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m"
    return f"{s / 3600:.1f}h"


def _num(x, nd=2, sign=False):
    return "–" if x is None else (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")


def bar(pct, width=10):
    pct = max(0, min(100, pct or 0))
    full = int(pct / 100 * width + 0.5)
    return "▕" + "█" * full + "░" * (width - full) + "▏"


def phase_strip(snap):
    c = snap.get("current") or {}
    idx = c.get("phase_index")
    parts = []
    for i, p in enumerate(snap.get("phases", [])):
        if idx is None:
            parts.append(col(p, "dim"))
        elif i < idx:
            parts.append(col(p, "green", "dim"))
        elif i == idx:
            parts.append(col(p.upper(), "cyan", "bold"))
        else:
            parts.append(col(p, "dim"))
    return col(" › ", "dim").join(parts)


def _delta_col(d, nd=2, higher_better=True):
    if d is None:
        return col("–", "dim")
    good = (d > 0) == higher_better
    return col(f"{d:+.{nd}f}", "green" if good and d != 0 else ("red" if d != 0 else "dim"))


def render_line(snap, width=120, claude=None):
    """Two lines for the status bar (wave/iteration/phase; then the phase-specific detail + best + spend)."""
    w = snap.get("wave", "?")
    if not snap.get("ok"):
        return [col(f"exploop {w}: view error ({snap.get('error', '?')[:120]})", "dim")]
    sup = snap.get("supervisor") or {}
    status = snap.get("status") or "?"
    c = snap.get("current") or {}
    best = snap.get("best") or {}
    cmp_ = snap.get("compare") or {}
    st_col = {"RUNNING": "green", "WAITING": "yellow", "PAUSED": "yellow", "BLOCKED": "red", "STOPPED": "yellow",
              "DONE": "green", "RESUMING": "yellow", "INITIALISED": "dim"}.get(status, "dim")
    dot = {"alive": col("●", "green"), "stale": col("●", "yellow"), "down": col("●", "red"), "exited": col("○", "dim")}
    head = f"{dot.get(sup.get('state'), '?')} {col(w, 'bold')} {col(status, st_col, 'bold')}"
    line1 = head
    if c.get("k"):
        head += f" · it {c['k']}/{snap.get('max_iterations', '?')} {col(c.get('label', ''), 'bold')}"
        line1 = head
        if status != "DONE":
            line1 = head + " · " + phase_strip(snap)
            if visible_len(line1) > width and c.get("phase"):     # narrow terminal: just the phase and its place
                idx, n = c.get("phase_index"), len(snap.get("phases", []))
                line1 = head + " · " + col(c["phase"].upper(), "cyan", "bold") + (col(f" ({idx + 1}/{n})", "dim") if idx is not None else "")
    elif snap.get("wave_phase") == "summary":
        line1 = head + " · " + col("SUMMARY", "cyan", "bold")

    alerts = []
    if sup.get("container_restarted") and status != "DONE":
        alerts.append(col(f"container restarted - loopctl.sh resume {w}", "red", "bold"))
    if sup.get("state") == "down":
        alerts.append(col(f"supervisor DOWN - loopctl.sh resume {w}", "red", "bold"))
    elif sup.get("state") == "stale":
        alerts.append(col(f"supervisor heartbeat {_dur(sup.get('heartbeat_age_s'))} old", "yellow", "bold"))
    if status == "BLOCKED" and snap.get("blocked"):
        b = snap["blocked"]
        alerts.append(col(f"BLOCKED in {b.get('phase')}: {b.get('reason', '')[:90]} - loopctl.sh resume {w}", "red", "bold"))
    if sup.get("paused"):
        alerts.append(col("PAUSED (no new GPU job)", "yellow"))
    if sup.get("stop_requested") and status not in ("STOPPED", "DONE"):
        alerts.append(col("stop requested", "yellow"))
    if snap.get("waiting"):
        alerts.append(col(f"waiting: {snap['waiting'][:80]}", "yellow"))

    detail = []
    phase = c.get("phase")
    prog = c.get("progress")
    if status == "DONE":
        detail.append(col("wave done", "green") + (f" · summary {snap['summary_report']}" if snap.get("summary_report") else ""))
    elif snap.get("claude"):
        a = snap["claude"]
        detail.append(f"{col(a['agent'] or a['step'], 'magenta')} {_dur(a['elapsed_s'])} · {a['tool_uses']} tools"
                      + (f" · last: {a['last_tool']} {a['last_target'] or ''}".rstrip() if a.get("last_tool") else ""))
    elif phase == "train" and prog:
        sub = prog.get("substep")
        if sub == "validate" and prog.get("validate"):
            pos = f"validating {prog['validate']['pct']}%"
        elif sub == "local eval":
            pos = "local eval"
        else:
            pos = f"{bar(prog.get('pct'))}{prog.get('pct') or 0}%"
            if prog.get("rate_per_s"):
                pos += f" {prog['rate_per_s']:.2f} b/s"
        detail.append(f"ep {prog['epoch']}/{prog['num_epochs']} {pos}"
                      + (f" · ETA {_dur(prog['eta_h'] * 3600)}" if prog.get("eta_h") is not None else ""))
    elif phase == "eval" and c.get("eval"):
        ev = c["eval"]
        r = ev.get("running")
        detail.append(f"eval {ev['done']}/{ev['total']}" + (f" ▶ {r['name']} {_dur(r['elapsed_s'])}" if r else ""))
    elif phase:
        detail.append(col(phase, "cyan"))

    if cmp_ and status != "DONE":
        bl = cmp_.get("best_label") or best.get("label")
        last = cmp_.get("last")
        if last:
            detail.append(f"ep{last['epoch']} LocalWF1 {last['wf1']:.2f} ({_delta_col(last.get('d_wf1'))} vs {bl}@{last['epoch']})"
                          + (f" {col(cmp_['spark'], 'cyan')}" if cmp_.get("spark") else ""))
        if phase == "train" and cmp_.get("target_now") is not None and cmp_.get("epoch_now"):
            detail.append(f"to beat: {bl}@ep{cmp_['epoch_now']} {cmp_['target_now']:.2f}")
    if best.get("label"):
        detail.append(f"best {col(best['label'], 'bold')} half_B {_num(best.get('merge_value'))}")
    spend = f"${snap.get('cost_usd', 0):.2f}"
    if snap.get("max_total_usd"):
        spend += f"/{snap['max_total_usd']}"
    detail.append(col(spend, "dim"))
    if claude:
        detail.append(col(claude, "dim"))
    sep = col(" · ", "dim")
    line2 = sep.join(alerts + detail) if alerts else sep.join(detail)
    return [fit(line1, width), fit(line2, width)]


def claude_segment(stdin_text):
    """'Opus ctx 34%' from the statusLine JSON Claude Code pipes in, or ''."""
    try:
        d = json.loads(stdin_text) if stdin_text.strip() else {}
    except ValueError:
        return ""
    name = ((d.get("model") or {}).get("display_name") or "").split(" (")[0]
    pct = (d.get("context_window") or {}).get("used_percentage")
    return " ".join(x for x in (name, f"ctx {pct:.0f}%" if isinstance(pct, (int, float)) else "") if x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["snapshot", "line"])
    ap.add_argument("--wave", help="wave name under exploop/ or a wave dir (default: $EXPLOOP_WAVE / the live one)")
    ap.add_argument("--width", type=int, help="status line width (default: $COLUMNS or the terminal's)")
    a = ap.parse_args(argv)
    if a.cmd == "line":
        try:
            stdin_text = "" if sys.stdin is None or sys.stdin.isatty() else sys.stdin.read()
            seg = claude_segment(stdin_text)
            wd = pick_wave(a.wave)
            if wd is None:
                if seg:
                    print(col(seg, "dim"))
                return 0
            width = a.width or int(os.environ.get("COLUMNS") or 0) or shutil.get_terminal_size((140, 24)).columns
            for line in render_line(snapshot(wd), width, seg):
                print(line)
        except Exception as exc:  # never break the status bar
            print(col(f"exploop: view error ({type(exc).__name__}: {str(exc)[:80]})", "dim"))
        return 0
    wd = pick_wave(a.wave)
    if wd is None:
        print(json.dumps({"ok": False, "wave": None, "error": "no live exploop wave (pass --wave)"}))
        return 1
    print(json.dumps(snapshot(wd), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
