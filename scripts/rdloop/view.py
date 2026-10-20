#!/usr/bin/env python3
"""Read-only view of an rdloop cycle for humans: the /rdloop pane (rdloop-view mod) and the status line.

    view.py snapshot [--cycle C]     JSON: status, phase strip, owner action, the team (what every agent is
                                     doing, from its live session transcript), debate rounds, levers, tests, events
    view.py line [--cycle C]         one status-line row while a cycle is live or waiting for the owner, else nothing

Read-only by construction: it reads state.json, cycle.json, events.jsonl, the debate / plan / test JSON and
the Claude session transcripts; it never takes the supervisor's lock and never writes. Python 3.8 (it runs
under the system python3 from the status line and the mod).
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rdlib as R  # noqa: E402

L = R.L
ROOT = R.REPO / "rdloop"
AGENT_LABEL = {"research-auditor": "auditor", "research-scout": "scout", "research-architect": "architect",
               "research-critic": "critic", "research-engineer": "engineer", "supervisor": "supervisor (tests)"}
TEAM_ORDER = ["research-auditor", "research-scout", "research-architect", "research-critic", "research-engineer", "supervisor"]
ROLE = {"research-auditor": "gap audit + final summary", "research-scout": "SOTA search, idea cards",
        "research-architect": "design plan, answers the critic", "research-critic": "grills the plan, reviews the code",
        "research-engineer": "implements the module (CPU)", "supervisor": "static / regression / smoke / exploop dry run"}


def cycle_dir_of(name):
    p = Path(name)
    if p.is_dir() and (p / "cycle.json").is_file():
        return p.resolve()
    return ROOT / name


def pick_cycle(name=None, root=None):
    """The named cycle, $RDLOOP_CYCLE, else the most recently updated one that is not parked / handed off."""
    name = name or os.environ.get("RDLOOP_CYCLE")
    if name:
        d = cycle_dir_of(name)
        return d if (d / "cycle.json").is_file() else None
    root = Path(root or ROOT)
    if not root.is_dir():
        return None
    best = None
    for d in root.iterdir():
        sp = d / "state.json"
        if not sp.is_file():
            continue
        st = R.read_json(sp) or {}
        if st.get("status") in R.FINAL_STATUSES:
            continue
        mt = sp.stat().st_mtime
        if best is None or mt > best[0]:
            best = (mt, d)
    return best[1] if best else None


def read_events(path, n=40):
    out, total = [], 0
    p = Path(path)
    if not p.is_file():
        return 0, out
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    for i, line in enumerate(lines[-n * 3:]):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == "job_start":
            continue
        e["seq"] = total - len(lines[-n * 3:]) + i + 1
        e.pop("data", None)
        out.append(e)
    return total, out[-n:]


def _short(name, inp):
    inp = inp or {}
    if name == "Bash":
        return inp.get("description") or (inp.get("command") or "")[:90]
    for k in ("file_path", "path", "notebook_path"):
        if inp.get(k):
            return os.path.relpath(inp[k], str(R.REPO)) if os.path.isabs(inp[k]) else inp[k]
    for k in ("query", "url", "pattern", "skill"):
        if inp.get(k):
            return str(inp[k])[:90]
    return ""


def activity(sid, max_bytes=3 << 20):
    """What a step did / is doing, from its session transcript (tail only)."""
    out = {"tool_uses": 0, "last_tool": None, "last_target": None, "last_text": None, "last_ts": None, "recent": [],
           "queries": [], "fetched": [], "files_written": []}
    p = R.transcript_path(sid)
    if not p:
        return out
    try:
        text = p.read_text(encoding="utf-8", errors="replace") if p.stat().st_size < max_bytes else R.tail_bytes(p, max_bytes)
    except OSError:
        return out
    for line in text.splitlines():
        if '"assistant"' not in line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") != "assistant":
            continue
        for b in (d.get("message") or {}).get("content") or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                name, inp = b.get("name"), b.get("input") or {}
                tgt = _short(name, inp)
                out["tool_uses"] += 1
                out.update(last_tool=name, last_target=tgt, last_ts=d.get("timestamp"))
                out["recent"].append({"tool": name, "target": tgt, "ts": d.get("timestamp")})
                if name == "WebSearch" and inp.get("query"):
                    out["queries"].append(str(inp["query"])[:120])
                if name == "WebFetch" and inp.get("url"):
                    out["fetched"].append(str(inp["url"])[:160])
                if name in ("Write", "Edit", "MultiEdit") and inp.get("file_path"):
                    fp = _short(name, inp)
                    if fp not in out["files_written"]:
                        out["files_written"].append(fp)
            elif b.get("type") == "text":
                t = " ".join(str(b.get("text", "")).split())
                if t:
                    out["last_text"] = t[:240]
    out["recent"] = out["recent"][-12:]
    out["queries"] = out["queries"][-5:]
    out["fetched"] = out["fetched"][-5:]
    out["files_written"] = out["files_written"][-12:]
    return out


def supervisor_liveness(st):
    hb = st.get("heartbeat_epoch") or 0
    age = round(time.time() - hb) if hb else None
    pid = st.get("supervisor_pid")
    alive = bool(pid) and L.proc_alive(pid, None)
    if st.get("status") in ("AWAITING_USER", "NEEDS_USER", "BLOCKED", "PARKED", "HANDED_OFF", "STOPPED", "INITIALISED") and not alive:
        state = "exited"
    elif alive and age is not None and age < 300:
        state = "alive"
    elif alive:
        state = "stale"
    else:
        state = "down"
    return {"state": state, "pid": pid, "heartbeat_age_s": age,
            "container_restarted": bool(st.get("pid1_start") and st["pid1_start"] != L.pid1_start())}


def owner_action(name, st):
    s = st.get("status")
    if s == "AWAITING_APPROVAL":
        pl = st.get("plan") or {}
        return {"kind": "approve", "text": f"plan v{pl.get('version', '?')} ready: levers {', '.join(pl.get('levers', []))}"
                + (f" (contested: {', '.join(pl.get('contested'))})" if pl.get("contested") else ""),
                "command": f"/rd-cycle approve {name}  (or rdctl.sh approve {name} [--levers a,b]; send back: "
                           f"rdctl.sh reopen {name} --at propose \"why\")", "file": f"rdloop/{name}/plan_agreed.md"}
    if s == "AWAITING_USER":
        return {"kind": "review", "text": "cycle finished: implemented, regression + smoke passed",
                "command": f"/rd-cycle review {name}", "file": f"rdloop/{name}/FINAL_SUMMARY.md"}
    if s == "NEEDS_USER":
        return {"kind": "needs_user", "text": "; ".join(st.get("needs_user") or [])[:300], "command": st.get("next") or "",
                "file": f"rdloop/{name}/STATUS.md"}
    if s == "BLOCKED":
        b = st.get("blocked") or {}
        return {"kind": "blocked", "text": f"{b.get('phase')}: {b.get('reason', '')[:300]}",
                "command": f"{b.get('hint', '')} -> rdctl.sh resume {name}", "file": f"rdloop/{name}/STATUS.md"}
    return None


def team_rows(cd, st):
    team = st.get("team") or {}
    rows = []
    for agent in TEAM_ORDER:
        t = team.get(agent) or {}
        row = {"agent": agent, "label": AGENT_LABEL[agent], "role": ROLE[agent], "state": t.get("state", "idle"),
               "step": t.get("step"), "headline": t.get("headline"), "cost": round(float(t.get("cost", 0) or 0), 2),
               "attempt": t.get("attempt"), "session_id": t.get("session_id"), "elapsed_s": None}
        if t.get("state") == "running" and t.get("started"):
            row["elapsed_s"] = round(time.time() - float(t["started"]))
        elif t.get("started") and t.get("finished"):
            row["elapsed_s"] = round(float(t["finished"]) - float(t["started"]))
        if agent != "supervisor" and t.get("session_id"):
            row.update(activity(t["session_id"]))
        if agent == "supervisor":
            for key in ("static", "gpu", "compat"):
                seq = (st.get("tests") or {}).get(key) or {}
                cur = next((j for j in seq.get("jobs", []) if j.get("state") == "running"), None)
                if cur:
                    row.update(state="running", step=f"{key}: {cur['name']}", last_target=" ".join(cur.get("argv", []))[-120:])
        rows.append(row)
    return rows


def debate_block(cd, st):
    d = st.get("debate") or {}
    led = d.get("ledger") or {}
    open_ = [{"id": o["id"], "severity": o["severity"], "lever_id": o.get("lever_id") or "plan", "claim": o.get("claim", "")[:200],
              "required_change": o.get("required_change", "")[:200]}
             for o in led.values() if o.get("status") == "open"]
    v = d.get("version") or 0
    prop = R.read_json(Path(cd) / "debate" / f"proposal_v{v}.json") or {}
    return {"version": v, "rounds": d.get("rounds") or [], "outcome": d.get("outcome"), "open": open_,
            "counts": R.debate_counts(led) if led else {}, "lint_failures": d.get("lint_failures", 0),
            "responses": [{"id": r.get("objection_id"), "action": r.get("action"), "detail": str(r.get("detail", ""))[:200]}
                          for r in prop.get("responses") or []],
            "levers_in_play": [lv.get("id") for lv in prop.get("levers") or []]}


def lever_rows(cd, st):
    plan = R.read_json(Path(cd) / "plan_agreed.json") or {}
    v = (st.get("debate") or {}).get("version") or 0
    in_plan = {lv.get("id"): lv for lv in plan.get("levers") or []}
    prop = (R.read_json(Path(cd) / "debate" / f"proposal_v{v}.json") or {}).get("levers") or []
    src = [in_plan.get(lv.get("id"), lv) for lv in prop] or list(in_plan.values())
    approved = st.get("approved_levers") or []
    dropped = st.get("dropped_levers") or {}
    parts = {p["id"]: p for p in (st.get("impl") or {}).get("parts", [])}
    gpu = ((st.get("tests") or {}).get("gpu") or {}).get("jobs", [])
    rows = []
    for lv in src:
        lid = lv.get("id")
        if lid in dropped:
            status = "dropped"
        elif lid in parts:
            status = {"pending": "approved", "running": "implementing", "done": "implemented", "dropped": "dropped"}.get(parts[lid]["status"], "approved")
        elif lid in approved:
            status = "approved"
        else:
            status = "contested" if lv.get("contested") else "proposed"
        sm = next((j for j in gpu if j.get("lever") == lid and j.get("check") == "train"), {})
        rows.append({"id": lid, "title": lv.get("title"), "category": lv.get("category"), "status": status,
                     "flags": [f"{f.get('name')}={f.get('default')}" for f in lv.get("flags") or []],
                     "gap_ids": lv.get("gap_ids") or [], "dropped_reason": dropped.get(lid),
                     "smoke_state": sm.get("state"), "smoke_note": sm.get("note"), "peak_gib": sm.get("peak_gib"),
                     "time_frac": sm.get("time_frac"), "eta_full_h": sm.get("eta_full_h")})
    return rows


def tests_block(cycle, st):
    out = {}
    for key in ("static", "gpu", "compat"):
        seq = (st.get("tests") or {}).get(key)
        if not seq:
            continue
        jobs = []
        for j in seq.get("jobs", []):
            el = j.get("elapsed_s")
            if j.get("state") == "running" and j.get("started"):
                el = round(time.time() - float(j["started"]))
            jobs.append({"name": j["name"], "kind": j.get("kind", key), "state": j.get("state"), "elapsed_s": el,
                         "note": str(j.get("note") or "")[:220], "lever": j.get("lever"), "peak_gib": j.get("peak_gib"),
                         "budget_s": j.get("timeout_s")})
        spent = sum(float(j.get("elapsed_s") or 0) for j in jobs if j["kind"] == "smoke")
        out[key] = {"jobs": jobs, "done": seq.get("done"), "ok": seq.get("ok"), "code_sha": (seq.get("code_sha") or "")[:12],
                    "smoke_spent_s": round(spent), "smoke_budget_s": float(R.get(cycle, "smoke", "max_minutes", default=30)) * 60}
    return out


def snapshot(cd):
    cd = Path(cd)
    snap = {"cycle": cd.name, "cycle_dir": str(cd), "ok": False, "error": None, "taken": R.now_ts()}
    try:
        cycle = R.read_json(cd / "cycle.json")
        st = R.read_json(cd / "state.json")
        if not cycle:
            snap["error"] = f"no cycle.json in {cd}"
            return snap
        if not st:
            snap.update(ok=True, status="NEW", phase=None, phases=[lbl for _, lbl in R.DISPLAY], phase_index=None,
                        error=None, next=f"rdctl.sh init {cd.name}", team=[], levers=[], tests={}, events=[], events_total=0)
            return snap
        name = cycle["cycle"]
        key, idx = R.phase_display(st.get("phase"))
        d = st.get("debate") or {}
        n_rounds = len(d.get("rounds") or []) - int(d.get("rounds_offset", 0) or 0)
        total, events = read_events(cd / "events.jsonl", 40)
        base = st.get("base") or st.get("base_provisional") or {}
        snap.update(ok=True, status=st.get("status"), phase=st.get("phase"), phase_key=key, phase_index=idx,
                    phases=[lbl for _, lbl in R.DISPLAY], gate_mode=st.get("gate_mode"), module=cycle.get("module"),
                    base={"label": base.get("label"), "run_dir": base.get("run_dir"), "from_wave": base.get("from_wave")},
                    cost_usd=round(float(st.get("cost_usd", 0) or 0), 2), max_total_usd=R.get(cycle, "claude", "max_total_usd", default=150),
                    supervisor=dict(supervisor_liveness(st), paused=(cd / "PAUSE").exists(), stop_requested=(cd / "STOP").exists()),
                    action=owner_action(name, st), waiting=st.get("waiting"), blocked=st.get("blocked"),
                    needs_user=st.get("needs_user") or [], next=st.get("next"),
                    debate_round=n_rounds, debate_max=R.get(cycle, "debate", "max_rounds", default=3),
                    team=team_rows(cd, st), debate=debate_block(cd, st), levers=lever_rows(cd, st),
                    tests=tests_block(cycle, st), review={"n": (st.get("review") or {}).get("n", 0),
                                                          "open": [f for f in ((st.get("review") or {}).get("findings") or {}).values()
                                                                   if f.get("status") == "open"][:12]},
                    fix_rounds=(st.get("fix") or {}).get("n", 0), events_total=total, events=events,
                    files={"status": f"rdloop/{name}/STATUS.md", "plan": f"rdloop/{name}/plan_agreed.md",
                           "summary": f"rdloop/{name}/FINAL_SUMMARY.md", "audit": f"rdloop/{name}/01_audit.md",
                           "sota": f"rdloop/{name}/02_sota.md"})
    except Exception as exc:     # the view never breaks
        snap.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return snap


# ------------------------------------------------------------------------------------- status line

def col(text, *styles):
    codes = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36"}
    return "".join(f"\033[{codes[s]}m" for s in styles) + text + "\033[0m"


def _dur(s):
    if s is None:
        return "?"
    s = float(s)
    return f"{s:.0f}s" if s < 90 else (f"{s / 60:.0f}m" if s < 5400 else f"{s / 3600:.1f}h")


def render_line(snap):
    if not snap.get("ok") or snap.get("status") in R.FINAL_STATUSES or snap.get("status") in ("NEW", "INITIALISED"):
        return None
    name = snap["cycle"]
    st = snap.get("status")
    head = col(f"rdloop {name}", "bold", "cyan")
    if snap.get("action"):
        a = snap["action"]
        colr = "red" if a["kind"] in ("blocked", "needs_user") else "yellow"
        return f"{head} {col('⚠ ' + st, 'bold', colr)} · {a['text'][:80]} · {col(a['command'].split('  ')[0], 'bold')}"
    sup = snap.get("supervisor") or {}
    if sup.get("state") in ("down", "stale") and st not in ("INITIALISED",):
        return f"{head} {col('supervisor ' + sup['state'].upper(), 'bold', 'red')} · scripts/rdloop/rdctl.sh resume {name}"
    ph = (snap.get("phase_key") or "").upper()
    if ph == "DEBATE":
        ph += f" r{max(1, snap.get('debate_round', 0) + (0 if snap.get('phase') == 'critique' else 1))}/{snap.get('debate_max')}"
    parts = [f"{head} {st}", col(ph, "bold")]
    run = next((t for t in snap.get("team", []) if t.get("state") == "running"), None)
    if run:
        seg = f"{col(run['label'], 'magenta')} {_dur(run.get('elapsed_s'))}"
        if run.get("tool_uses"):
            seg += f" · {run['tool_uses']} tools"
        if run.get("last_tool"):
            seg += f" · last: {run['last_tool']} {str(run.get('last_target') or '')[:50]}"
        elif run.get("last_target"):
            seg += f" · {str(run['last_target'])[-50:]}"
        parts.append(seg)
    if snap.get("waiting"):
        parts.append(col(str(snap["waiting"])[:70], "yellow"))
    parts.append(col(f"${snap.get('cost_usd', 0):.0f}/{snap.get('max_total_usd')}", "dim"))
    return " · ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["snapshot", "line"])
    ap.add_argument("--cycle", help="cycle name under rdloop/ or a cycle dir (default: $RDLOOP_CYCLE / the live one)")
    a = ap.parse_args(argv)
    if a.cmd == "line":
        try:
            cd = pick_cycle(a.cycle)
            if cd is not None:
                line = render_line(snapshot(cd))
                if line:
                    width = int(os.environ.get("COLUMNS") or 0) or shutil.get_terminal_size((160, 24)).columns
                    vis = re.sub(r"\033\[[0-9;]*m", "", line)
                    if len(vis) > width:
                        line = vis[: max(10, width - 1)] + "…"
                    print(line)
        except Exception as exc:  # never break the status bar
            print(col(f"rdloop: view error ({type(exc).__name__}: {str(exc)[:60]})", "dim"))
        return 0
    cd = pick_cycle(a.cycle)
    if cd is None:
        print(json.dumps({"ok": False, "cycle": None, "error": "no live rdloop cycle (pass --cycle, or /rdloop <cycle>)"}))
        return 1
    print(json.dumps(snapshot(cd), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
