"""Helpers for the rdloop supervisor (research & development cycles that feed exploop waves).

Stdlib only (plus PyYAML through exploop_lib for config.yaml); python 3.8 compatible because
view.py (status line / mod) imports it with the system python3.

What lives here: phase names, write scopes, the isolation manifest (detection of any change
outside a cycle's write scope), the code fingerprint, the proposal LINT, the debate ledger and its
convergence rule, transcript readers (what a headless step did), the exploop yield rule, score
comparison for the regression check, the wave seed for the exploop handoff and STATUS.md.

The generic pieces (JSON / argv / log parsers / process / GPU) are imported read-only from
scripts/exploop/exploop_lib.py; nothing in scripts/exploop is modified by rdloop.
"""
import copy
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE.parent / "exploop") not in sys.path:
    sys.path.append(str(HERE.parent / "exploop"))     # appended: never shadow rdloop's own view / supervisor
import exploop_lib as L  # noqa: E402  (read-only reuse)

read_json = L.read_json
write_json = L.write_json
write_text = L.write_text
now_ts = L.now_ts
stamp = L.stamp
sha256_file = L.sha256_file
sha256_text = L.sha256_text
tail_bytes = L.tail_bytes

CLAUDE_PROJECTS = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))) / "projects"

# Phase order of a cycle. The debate (propose -> lint -> critique) and the fix loop
# (fix -> static -> review -> gpu) go back in this list; `phase_index` is for display.
PHASES = ["wait_waves", "audit", "scout", "propose", "lint", "critique", "gate", "scaffold", "implement",
          "static", "review", "fix", "gpu", "compat", "summary", "awaiting_user"]
DISPLAY = [("wait_waves", "WAIT"), ("audit", "AUDIT"), ("scout", "SCOUT"), ("debate", "DEBATE"), ("gate", "GATE"),
           ("implement", "IMPLEMENT"), ("static", "STATIC"), ("review", "REVIEW"), ("gpu", "GPU"),
           ("compat", "COMPAT"), ("summary", "SUMMARY"), ("awaiting_user", "REVIEW-BY-OWNER")]
DISPLAY_OF = {"propose": "debate", "lint": "debate", "critique": "debate", "scaffold": "implement",
              "fix": "review"}
OWNER_STATUSES = ("AWAITING_APPROVAL", "AWAITING_USER", "NEEDS_USER", "BLOCKED")
FINAL_STATUSES = ("PARKED", "HANDED_OFF")

# step -> (agent, policy, schema, skill)
STEPS = {
    "audit": ("research-auditor", "readonly", "audit", "rd-audit"),
    "scout": ("research-scout", "scout", "scout", "rd-scout"),
    "propose": ("research-architect", "readonly", "proposal", "rd-propose"),
    "critique": ("research-critic", "critic", "critique", "rd-critique"),
    "implement": ("research-engineer", "engineer", "implement", "rd-implement"),
    "review": ("research-critic", "critic", "code_review", "rd-critique"),
    "fix": ("research-engineer", "engineer", "implement", "rd-implement"),
    "summary": ("research-auditor", "readonly", "summary", "rd-audit"),
}
AGENTS = ["research-auditor", "research-scout", "research-architect", "research-critic", "research-engineer"]

PROGRESS_RX = r"model_merging_[^ /]*/scores\.txt|RTCFake/data/progress\.txt|/wav/progress\b"
DEFAULT_WEB_DOMAINS = ["arxiv.org", "openreview.net", "isca-archive.org", "isca-speech.org", "aclanthology.org",
                       "github.com", "githubusercontent.com", "huggingface.co", "semanticscholar.org",
                       "ieeexplore.ieee.org", "paperswithcode.com", "asvspoof.org", "dl.acm.org",
                       "proceedings.neurips.cc", "proceedings.mlr.press", "researchgate.net", "sciencedirect.com",
                       "springer.com", "mdpi.com", "codabench.org"]


# ------------------------------------------------------------------------------------- basics

def get(obj, *keys, default=None):
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def deep_merge(base, over):
    """dict merge: `over` wins, nested dicts merge, lists are replaced."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def rel(repo, p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(Path(repo).resolve()))
    except ValueError:
        return str(p)


def phase_display(phase):
    """('debate', index) for the phase strip."""
    key = DISPLAY_OF.get(phase, phase)
    names = [k for k, _ in DISPLAY]
    return key, (names.index(key) if key in names else None)


# -------------------------------------------------------------------------------- cycle paths

def module_layout(cycle):
    """Repo-relative paths that belong to a cycle's new code."""
    mod = cycle["module"]
    prefix = cycle.get("script_prefix", mod[:2] + "_")
    return {
        "module_dir": f"SLS_setup/{mod}/",
        "shim": f"SLS_setup/main_train_{mod[:2]}.py",
        "scripts_prefix": f"scripts/{prefix}",
        "profile_prefix": f"scripts/exploop/profiles/{prefix}",
    }


def write_scope(repo, cycle, cycle_dir, policy, files=()):
    """(exact files, prefixes) a step of this policy may write (absolute paths)."""
    repo = Path(repo).resolve()
    cycle_dir = Path(cycle_dir).resolve()
    exact = [str(Path(f).resolve()) for f in files]
    prefixes = []
    if policy == "engineer":
        lay = module_layout(cycle)
        prefixes = [str(repo / lay["module_dir"]) + "/", str(repo / lay["scripts_prefix"]), str(repo / lay["shim"]),
                    str(cycle_dir / "impl") + "/", str(cycle_dir / "scratch") + "/", tmpdir(cycle) + "/"]
        if cycle.get("_state_dict_change"):
            prefixes.append(str(repo / lay["profile_prefix"]))
    return exact, prefixes


def tmpdir(cycle):
    return f"/tmp/rdloop_{cycle['cycle']}"


def in_scope(path, exact, prefixes, cwd=None):
    """True when path (relative to cwd) resolves to an allowed file or under an allowed prefix."""
    p = path if os.path.isabs(path) else os.path.join(cwd or os.getcwd(), path)
    rp = os.path.realpath(p)
    if rp in {os.path.realpath(x) for x in exact}:
        return True
    for pre in prefixes:
        if pre.endswith("/"):
            base = os.path.realpath(pre[:-1])
            if rp == base or rp.startswith(base + "/"):
                return True
        else:
            # file-name prefix such as scripts/b4_  -> realpath of the parent + the stem
            parent, stem = os.path.split(pre)
            full = os.path.join(os.path.realpath(parent), stem)
            if rp.startswith(full):
                return True
    return False


# ------------------------------------------------------------------------- isolation manifest

def manifest_excludes(cycle, cycle_rel):
    """Repo-relative prefixes the manifest ignores: the cycle's own write scope and paths that a
    running exploop wave or the owner legitimately writes."""
    lay = module_layout(cycle)
    return sorted({cycle_rel.rstrip("/") + "/", "rdloop/", "exploop/", ".claude/worktrees/", "docs/experiment_log.csv",
                   "docs/experiments-exploration/", "docs/research/", "scripts/run_", lay["module_dir"],
                   lay["scripts_prefix"], lay["shim"], lay["profile_prefix"], "SLS_setup/exp_smoke/"}
                  | set(cycle.get("manifest_extra_excludes", [])))


PROTECTED_ROOTS = ("SLS_setup/", "scripts/", "configs/", ".claude/", "docs/")


def git_files(repo):
    """Tracked + untracked-not-ignored files (repo-relative)."""
    def ls(*args):
        try:
            out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", *args], capture_output=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            return []
        return [x for x in out.stdout.decode("utf-8", "replace").split("\0") if x]
    tracked = ls()
    untracked = ls("-o", "--exclude-standard")
    return tracked, untracked


def _excluded(path, excludes):
    return any(path.startswith(e) for e in excludes)


def build_manifest(repo, cycle, cycle_rel, exp_root="SLS_setup/exp"):
    repo = Path(repo)
    excludes = manifest_excludes(cycle, cycle_rel)
    tracked, untracked = git_files(repo)
    files = {}
    for p in sorted(set(tracked) | set(untracked)):
        if _excluded(p, excludes):
            continue
        fp = repo / p
        if fp.is_file() and not fp.is_symlink():
            files[p] = sha256_file(fp)
    runs = {}
    root = repo / exp_root
    if root.is_dir():
        for fp in root.rglob("*"):
            try:
                if fp.is_file() and not fp.is_symlink():
                    st = fp.stat()
                    runs[str(fp.relative_to(repo))] = [st.st_size, st.st_mtime_ns]
            except OSError:
                continue
    return {"created": now_ts(), "excludes": excludes, "files": files, "runs": runs, "untracked": sorted(untracked)}


def active_run_dirs(repo):
    """Run dirs an exploop wave is writing right now (current, unfinished iterations)."""
    out = []
    for w in exploop_waves(repo):
        for it in (w["state"].get("iterations") or []):
            if not it.get("done") and it.get("run_dir"):
                out.append(it["run_dir"].rstrip("/") + "/")
    return out


def check_manifest(repo, manifest, cycle, cycle_rel, exp_root="SLS_setup/exp"):
    """List of violations: protected files changed / deleted, new files under protected roots
    outside the scope, existing run files changed / deleted (new run files are fine)."""
    repo = Path(repo)
    excludes = manifest.get("excludes") or manifest_excludes(cycle, cycle_rel)
    out = []
    for p, sha in manifest.get("files", {}).items():
        fp = repo / p
        if not fp.is_file():
            out.append(f"deleted: {p}")
        elif sha256_file(fp) != sha:
            out.append(f"modified: {p}")
    tracked, untracked = git_files(repo)
    known = set(manifest.get("files", {}))
    for p in sorted(set(tracked) | set(untracked)):
        if p in known or _excluded(p, excludes):
            continue
        if p.startswith(PROTECTED_ROOTS):
            out.append(f"new file outside the cycle scope: {p}")
    skip = active_run_dirs(repo)
    for p, (size, mt) in manifest.get("runs", {}).items():
        if any(p.startswith(s) for s in skip):
            continue
        fp = repo / p
        try:
            st = fp.stat()
        except OSError:
            out.append(f"run file deleted: {p}")
            continue
        if st.st_size != size or st.st_mtime_ns != mt:
            out.append(f"run file modified: {p}")
    return out


def freeze_untracked(repo, manifest, out_tar):
    """Tarball of the untracked files the manifest covers (they have no git backup)."""
    repo = Path(repo)
    paths = [p for p in manifest.get("untracked", []) if p in manifest.get("files", {})]
    Path(out_tar).parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, "w:gz") as tf:
        for p in paths:
            fp = repo / p
            if fp.is_file() and fp.stat().st_size < 20 << 20:
                tf.add(str(fp), arcname=p)
    return len(paths)


def code_files(repo, cycle):
    repo = Path(repo)
    lay = module_layout(cycle)
    out = []
    mdir = repo / lay["module_dir"]
    if mdir.is_dir():
        out += [p for p in mdir.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    sp = repo / lay["scripts_prefix"]
    out += [p for p in sp.parent.glob(sp.name + "*") if p.is_file()]
    sh = repo / lay["shim"]
    if sh.is_file():
        out.append(sh)
    pp = repo / lay["profile_prefix"]
    out += [p for p in pp.parent.glob(pp.name + "*") if p.is_file()]
    return sorted(out)


def code_sha(repo, cycle):
    """Fingerprint of the cycle's code: sha256 over (path, sha256) of every module / script file."""
    h = hashlib.sha256()
    for p in code_files(repo, cycle):
        h.update(rel(repo, p).encode())
        h.update(b"\0")
        h.update(sha256_file(p).encode())
        h.update(b"\n")
    return h.hexdigest()


# ------------------------------------------------------------------------------- exploop waves

def exploop_waves(repo):
    out = []
    root = Path(repo) / "exploop"
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        st = read_json(d / "loop_state.json")
        if isinstance(st, dict):
            out.append({"wave": d.name, "dir": d, "state": st, "spec": read_json(d / "wave.json") or {}})
    return out


def _wave_live(st, stale_s):
    hb = st.get("heartbeat_epoch") or 0
    return st.get("status") in ("RUNNING", "WAITING", "RESUMING") and hb and time.time() - hb < stale_s


def _current_iter(st):
    return next((i for i in (st.get("iterations") or []) if i.get("k") == st.get("current")), None)


def exploop_gpu_busy(repo, stale_s=900):
    """Reason string when a live exploop wave is in (or about to enter) a GPU phase, else None."""
    for w in exploop_waves(repo):
        st = w["state"]
        if not _wave_live(st, stale_s):
            continue
        it = _current_iter(st)
        if it and it.get("phase") in ("design", "render", "smoke", "train", "eval"):
            return f"yielding to exploop {w['wave']} ({it.get('label')} {it.get('phase', '').upper()})"
    return None


def exploop_claude_busy(repo, stale_s=900):
    """Reason string while an exploop headless Claude step is running (its integrity snapshot would
    flag our module writes as protected-file changes), else None."""
    for w in exploop_waves(repo):
        st = w["state"]
        if not _wave_live(st, stale_s):
            continue
        holders = [st] + ([_current_iter(st)] if _current_iter(st) else [])
        for h in holders:
            for name, job in (h.get("jobs") or {}).items():
                if name in ("design", "analyze", "summary") and L.proc_alive(job.get("pid"), job.get("start_ticks")):
                    return f"exploop {w['wave']} {name.upper()} Claude step is running"
    return None


def wave_status(repo, wave):
    st = read_json(Path(repo) / "exploop" / wave / "loop_state.json")
    return (st or {}).get("status")


def latest_wave(repo):
    waves = exploop_waves(repo)
    if not waves:
        return None
    return max(waves, key=lambda w: (w["dir"] / "loop_state.json").stat().st_mtime)


def resolve_base(repo, base_cfg):
    """The cycle's base (current best): explicit run or the `best` block of an exploop wave."""
    repo = Path(repo)
    if base_cfg.get("run_dir"):
        b = dict(base_cfg)
        run = repo / b["run_dir"]
        b.setdefault("label", run.name.split("_")[0])
        b.setdefault("readouts", {})
        b["argv"] = b.get("argv") or L.config_argv(run)
        b["from_wave"] = None
    else:
        wave = base_cfg.get("from_wave") or (latest_wave(repo) or {}).get("wave")
        st = read_json(repo / "exploop" / str(wave) / "loop_state.json") if wave else None
        if not st or not st.get("best"):
            return None
        best = st["best"]
        b = {"label": best["label"], "run_dir": best["run_dir"], "readouts": best.get("readouts", {}),
             "registry_ids": best.get("registry_ids", {}), "argv": best.get("argv") or L.config_argv(repo / best["run_dir"]),
             "merge_value": best.get("merge_value"), "final_value": best.get("final_value"), "from_wave": wave}
    run = repo / b["run_dir"]
    b["entry_module"] = entry_module(run)
    b["model_flags"] = L.model_flags(run)
    fin = b.get("readouts", {}).get("final") or ""
    m = re.search(r"epoch_(\d+)$", fin.rstrip("/"))
    ep = int(m.group(1)) if m else L.final_epoch(run)
    b["final_epoch"] = ep
    ck = sorted((run / "ckpt").glob(f"epoch_{ep}_*.pth")) if ep else []
    b["ckpt_final"] = rel(repo, ck[0]) if ck else None
    if not b.get("readouts", {}).get("final") and ep:
        b.setdefault("readouts", {})["final"] = rel(repo, run / "local_eval" / f"epoch_{ep}")
    return b


def entry_module(run_dir):
    """'b3wave' from config.yaml command '<...>/SLS_setup/b3wave/train.py ...' (None for main_train.py)."""
    cmd = L.read_config(run_dir).get("command") or ""
    first = (shlex.split(cmd) or [""])[0] if isinstance(cmd, str) else (cmd[0] if cmd else "")
    m = re.search(r"SLS_setup/([A-Za-z0-9_]+)/train\.py$", first)
    return m.group(1) if m else None


def run_hours(run_dir):
    """Wall hours of a finished run: config.yaml `created` -> last epoch line of train.log."""
    run = Path(run_dir)
    created = L.read_config(run).get("created")
    rows = L.parse_train_log(run / "logs" / "train.log")
    if not created or not rows:
        return None
    last = rows[max(rows)]
    ts = last.get("ts_wf1") or last.get("ts_loss")
    if not ts:
        return None
    try:
        from datetime import datetime
        t0 = datetime.fromisoformat(str(created))
        t1 = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S")
        return max(0.0, (t1 - t0).total_seconds() / 3600)
    except ValueError:
        return None


# ------------------------------------------------------------------------------------ LINT

FLAG_RX = re.compile(r"^--[a-z0-9][a-z0-9_]*$")
LEVER_RX = re.compile(r"^[a-z][a-z0-9_]{0,23}$")


def lint_proposal(prop, cycle, base_flags, prev_open_ids=(), check_import=None):
    """Mechanical checks of an architect proposal. Returns (errors, needs_user) where needs_user maps
    lever id -> [missing dependencies]. `check_import(name) -> bool` tests a dependency."""
    errs = []
    needs = {}
    levers = prop.get("levers") or []
    lim = cycle.get("levers", {})
    lo, hi = int(lim.get("min", 2)), int(lim.get("max", 4))
    if not lo <= len(levers) <= hi:
        errs.append(f"{len(levers)} levers; the cycle asks for {lo}-{hi}")
    ids = [lv.get("id", "") for lv in levers]
    for i in ids:
        if not LEVER_RX.match(i or ""):
            errs.append(f"lever id {i!r} must match {LEVER_RX.pattern}")
    if len(set(ids)) != len(ids):
        errs.append(f"duplicate lever ids: {ids}")
    seen_flags = {}
    limits = cycle.get("limits", {})
    smoke_total = float(get(cycle, "smoke", "s0_minutes", default=4.5)) + float(get(cycle, "smoke", "evalpath_minutes", default=2.0))
    n_sd = 0
    for lv in levers:
        lid = lv.get("id", "?")
        flags = lv.get("flags") or []
        if not flags:
            errs.append(f"{lid}: no flags (every lever is one or more new default-off flags)")
        names = []
        for fl in flags:
            n = fl.get("name", "")
            names.append(n)
            if not FLAG_RX.match(n):
                errs.append(f"{lid}: flag {n!r} must match {FLAG_RX.pattern}")
            if n in base_flags:
                errs.append(f"{lid}: {n} already exists in the base trainer; a lever adds NEW flags")
            if n in seen_flags:
                errs.append(f"{lid}: {n} is also a flag of lever {seen_flags[n]}")
            seen_flags[n] = lid
            if "default" not in fl or str(fl.get("default", "")).strip() == "":
                errs.append(f"{lid}: {n} needs a default that reproduces the old behaviour")
        if not str(lv.get("default_equivalence", "")).strip():
            errs.append(f"{lid}: default_equivalence must say why the defaults reproduce the base exactly")
        wl = lv.get("wave_lever") or {}
        sets = wl.get("set") or {}
        if not sets:
            errs.append(f"{lid}: wave_lever.set is empty (what the wave should screen)")
        for f, vals in sets.items():
            if f not in names:
                errs.append(f"{lid}: wave_lever.set flag {f} is not one of the lever's own flags {names}")
            if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
                errs.append(f"{lid}: wave_lever.set[{f}] must be a list of strings ([] for a store_true flag)")
        for rx in (lv.get("expect_regex") or []):
            try:
                re.compile(rx)
            except re.error as exc:
                errs.append(f"{lid}: expect_regex {rx!r} does not compile: {exc}")
        if not lv.get("expect_regex"):
            errs.append(f"{lid}: expect_regex is empty (header line(s) proving the lever is on)")
        if lv.get("state_dict_change"):
            n_sd += 1
        cost = lv.get("cost") or {}
        try:
            etf = float(cost.get("epoch_time_frac", 1.0))
            if etf > float(limits.get("max_epoch_time_frac", 1.3)):
                errs.append(f"{lid}: epoch_time_frac {etf} > limit {limits.get('max_epoch_time_frac', 1.3)}")
            vram = float(cost.get("extra_vram_gib", 0.0))
            if vram > float(limits.get("max_extra_vram_gib", 4.0)):
                errs.append(f"{lid}: extra_vram_gib {vram} > limit {limits.get('max_extra_vram_gib', 4.0)}")
            loc = int(cost.get("est_loc", 0))
            if loc > int(limits.get("max_loc_per_lever", 400)):
                errs.append(f"{lid}: est_loc {loc} > limit {limits.get('max_loc_per_lever', 400)}")
        except (TypeError, ValueError):
            errs.append(f"{lid}: cost fields must be numbers")
        try:
            smoke_total += float((lv.get("smoke") or {}).get("minutes_est", 4.5))
            if lv.get("params_added"):
                smoke_total += float(get(cycle, "smoke", "overfit_minutes", default=2.0))
        except (TypeError, ValueError):
            errs.append(f"{lid}: smoke.minutes_est must be a number")
        deps = [d for d in (lv.get("dependencies") or []) if str(d).strip()]
        if deps and check_import:
            missing = [d for d in deps if not check_import(d)]
            if missing:
                needs[lid] = missing
        for k in ("hypothesis", "predicted_signature", "falsifier", "mechanism"):
            if not str(lv.get(k, "")).strip():
                errs.append(f"{lid}: {k} is empty")
        if not (lv.get("gap_ids") or []):
            errs.append(f"{lid}: gap_ids is empty (which audit gap does it target?)")
    if n_sd > 1:
        errs.append(f"{n_sd} levers change the state_dict; at most one per cycle")
    budget = float(get(cycle, "smoke", "max_minutes", default=30)) - float(get(cycle, "smoke", "margin_minutes", default=5))
    if smoke_total > budget:
        errs.append(f"smoke estimate {smoke_total:.1f} min (S0 + levers + overfit + eval path) > {budget:.0f} min")
    answered = {r.get("objection_id") for r in (prop.get("responses") or [])}
    missing = [o for o in prev_open_ids if o not in answered]
    if missing:
        errs.append(f"no response to open objection(s) {missing}: answer each with accept / modify / rebut")
    return errs, needs


# ---------------------------------------------------------------------------- debate ledger

SEVERITIES = ("blocker", "major", "minor")
BLOCKING = ("blocker", "major")


def apply_critique(ledger, critique, version, lever_ids):
    """Update the objection ledger {id: {...}} with critique round `version`. Returns notes
    (invalid blockers downgraded, unknown ids) for the event log."""
    notes = []
    for o in critique.get("objections") or []:
        oid = str(o.get("id") or "").strip() or f"O{version}.{len(ledger) + 1}"
        sev = o.get("severity", "minor")
        if sev not in SEVERITIES:
            sev = "minor"
        evidence = [e for e in (o.get("evidence") or []) if str(e).strip()]
        if sev == "blocker" and not evidence:
            notes.append(f"{oid}: blocker without evidence -> minor")
            sev = "minor"
        lever = (o.get("lever_id") or "").strip()
        prev = ledger.get(oid)
        if prev:
            prev.update(severity=sev, status="open", claim=o.get("claim", prev.get("claim")),
                        required_change=o.get("required_change", prev.get("required_change")),
                        evidence=evidence or prev.get("evidence", []))
            prev["reopened"] = prev.get("reopened", 0) + 1
            prev["last_v"] = version
        else:
            ledger[oid] = {"id": oid, "lever_id": lever, "severity": sev, "category": o.get("category", "other"),
                           "claim": o.get("claim", ""), "required_change": o.get("required_change", ""),
                           "evidence": evidence, "status": "open", "opened_v": version, "last_v": version, "reopened": 0}
    for oid in critique.get("resolved") or []:
        if oid in ledger:
            ledger[oid]["status"] = "resolved"
            ledger[oid]["resolved_v"] = version
        else:
            notes.append(f"resolved id {oid} is unknown")
    for so in critique.get("still_open") or []:
        oid = so.get("id") if isinstance(so, dict) else so
        if oid in ledger and ledger[oid]["status"] != "open":
            ledger[oid]["status"] = "open"
            ledger[oid]["reopened"] = ledger[oid].get("reopened", 0) + 1
    for oid, o in ledger.items():               # objections on levers that no longer exist are moot
        if o["status"] == "open" and o.get("lever_id") and o["lever_id"] not in lever_ids:
            o["status"] = "moot"
    return notes


def open_objections(ledger, severities=BLOCKING):
    return [o for o in ledger.values() if o["status"] == "open" and o["severity"] in severities]


def debate_counts(ledger):
    c = {s: 0 for s in SEVERITIES}
    for o in ledger.values():
        if o["status"] == "open":
            c[o["severity"]] += 1
    c["resolved"] = sum(1 for o in ledger.values() if o["status"] == "resolved")
    return c


def resolve_exhausted(ledger, lever_ids):
    """After the last round: levers with an open blocker are dropped, with an open major contested;
    plan-level (no lever) blockers / majors need the owner."""
    dropped, contested, plan_level = [], [], []
    for o in open_objections(ledger):
        lid = o.get("lever_id")
        if not lid:
            plan_level.append(o["id"])
        elif o["severity"] == "blocker":
            if lid not in dropped:
                dropped.append(lid)
        elif lid not in contested:
            contested.append(lid)
    contested = [c for c in contested if c not in dropped and c in lever_ids]
    dropped = [d for d in dropped if d in lever_ids]
    return dropped, contested, plan_level


# ------------------------------------------------------------------------------- transcripts

def transcript_path(sid):
    if not sid:
        return None
    paths = list(CLAUDE_PROJECTS.glob(f"*/{sid}.jsonl"))
    return paths[0] if paths else None


def transcript_events(sid, max_bytes=16 << 20):
    """Assistant tool_use blocks and text lines of a session transcript, in order."""
    p = transcript_path(sid)
    out = []
    if not p:
        return out
    text = p.read_text(encoding="utf-8", errors="replace") if p.stat().st_size < max_bytes else tail_bytes(p, max_bytes)
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
                out.append({"kind": "tool", "name": b.get("name"), "input": b.get("input") or {}, "ts": d.get("timestamp")})
            elif b.get("type") == "text" and str(b.get("text", "")).strip():
                out.append({"kind": "text", "text": b["text"], "ts": d.get("timestamp")})
    return out


def fetched_urls(sid):
    return [e["input"].get("url", "") for e in transcript_events(sid) if e["kind"] == "tool" and e["name"] == "WebFetch"]


def bash_commands(sid):
    return [e["input"].get("command", "") for e in transcript_events(sid) if e["kind"] == "tool" and e["name"] == "Bash"]


ARXIV_RX = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def norm_url(u):
    """Comparable form of a paper URL: arXiv id when there is one, else host+path without
    scheme / www / trailing slash / query."""
    u = str(u or "").strip().lower()
    if "arxiv" in u:
        m = ARXIV_RX.search(u)
        if m:
            return "arxiv:" + m.group(1)
    u = re.sub(r"^[a-z]+://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = u.split("#")[0].split("?")[0].rstrip("/")
    u = re.sub(r"\.pdf$", "", u)
    return u


def verify_sources(ideas, fetched):
    """Mark every idea source verified when its URL was fetched by the step. Returns counts."""
    got = {norm_url(u) for u in fetched}
    n = v = 0
    for idea in ideas:
        for s in idea.get("sources") or []:
            n += 1
            s["verified"] = norm_url(s.get("url")) in got
            v += int(s["verified"])
        idea["verified_sources"] = sum(1 for s in idea.get("sources") or [] if s.get("verified"))
    return v, n


# ------------------------------------------------------------------------------- argv / smoke

def set_flags(argv, sets, removes=()):
    """Base argv with flags set (flag -> list of values; [] for store_true) / removed."""
    sets = {k: [str(x) for x in (v if isinstance(v, list) else [v])] for k, v in (sets or {}).items()}
    return L.apply_change(argv, {"set": sets, "remove": list(removes)}, {})


def smoke_argv(base_argv, cycle, cycle_dir, track, extra_sets=None):
    sm = cycle.get("smoke", {})
    sets = {"--train_subset_frac": [str(sm.get("subset_frac", 0.01))], "--num_epochs": ["1"], "--keep_epochs": ["1"],
            "--local_eval_clean_n": [str(sm.get("local_eval_clean_n", 200))],
            "--dev_protocol": [str(Path(cycle_dir).resolve() / "tests" / "smoke" / "dev_small.txt")],
            "--out_path": [str(Path(cycle_dir).resolve() / "runs")], "--track": [track]}
    sets.update(extra_sets or {})
    return set_flags(base_argv, sets, removes=("--resume",))


def find_run(out_root, track):
    cands = sorted(Path(out_root).glob(f"{track}_epoch*_bs*_*"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


# --------------------------------------------------------------------------- score comparison

def read_score_file(path):
    out = {}
    p = Path(path)
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return out


def compare_scores(old_dir, new_dir, names, tol):
    """Per set: n, max |logit diff|, decisions flipped at p=0.5 (logit 0), missing ids."""
    res = {"ok": True, "sets": {}}
    for name in names:
        old = read_score_file(Path(old_dir) / f"{name}.logits")
        new = read_score_file(Path(new_dir) / f"{name}.logits")
        if not old or not new:
            res["sets"][name] = {"ok": False, "error": f"missing scores (old {len(old)}, new {len(new)})"}
            res["ok"] = False
            continue
        common = set(old) & set(new)
        diffs = [abs(old[k] - new[k]) for k in common]
        flips = sum(1 for k in common if (old[k] > 0) != (new[k] > 0))
        mx = max(diffs) if diffs else float("inf")
        ok = len(common) == len(old) == len(new) and mx <= tol and flips == 0
        res["sets"][name] = {"ok": ok, "n": len(common), "n_old": len(old), "n_new": len(new),
                             "max_abs_logit_diff": mx, "flips_at_0.5": flips}
        res["ok"] = res["ok"] and ok
    return res


def compare_metrics(old_metrics, new_metrics, names, tol=1e-6):
    """Set-level f1 / eer / recall_bona of two local_eval metrics.json files."""
    a, b = read_json(old_metrics) or {}, read_json(new_metrics) or {}
    out = {"ok": True, "sets": {}}
    for n in names:
        sa, sb = get(a, "full", "sets", n) or {}, get(b, "full", "sets", n) or {}
        row = {}
        for k in ("f1", "eer", "recall_bona"):
            if sa.get(k) is None or sb.get(k) is None:
                row[k] = None
                continue
            row[k] = abs(float(sa[k]) - float(sb[k]))
        ok = all(v is not None and v <= tol for v in row.values()) if row else False
        out["sets"][n] = {"ok": ok, "abs_diff": row}
        out["ok"] = out["ok"] and ok
    return out


# --------------------------------------------------------------------------------- headlines

def headline(step, payload, extra=None):
    p = payload or {}
    extra = extra or {}
    if step == "audit":
        g = p.get("gaps") or []
        return f"{len(g)} gaps" + (f" · top: {g[0].get('title', '')[:60]}" if g else "")
    if step == "scout":
        ideas = p.get("ideas") or []
        v, n = extra.get("verified", 0), extra.get("sources", 0)
        return f"{len(ideas)} ideas · {v}/{n} sources fetched"
    if step == "propose":
        lv = [x.get("id") for x in p.get("levers") or []]
        return f"proposal v{p.get('version', '?')}: {len(lv)} levers ({', '.join(lv)})"
    if step == "critique":
        c = extra.get("counts") or {}
        return (f"v{p.get('version', '?')}: {c.get('blocker', 0)} blocker / {c.get('major', 0)} major / "
                f"{c.get('minor', 0)} minor open · {'agreed' if extra.get('agreed') else 'revise'}")
    if step in ("implement", "fix"):
        lv = p.get("levers") or []
        done = sum(1 for x in lv if x.get("implemented"))
        return (f"{p.get('part', step)}: {done}/{len(lv)} levers implemented · {len(p.get('files') or [])} files"
                if lv else f"{p.get('part', step)}: {len(p.get('files') or [])} files")
    if step == "review":
        f = [x for x in p.get("findings") or [] if x.get("severity") in BLOCKING]
        return f"review r{p.get('round', '?')}: {len(f)} blocking findings"
    if step == "summary":
        return str(p.get("headline", ""))[:120]
    return ""


# ------------------------------------------------------------------------------ wave seed

def build_seed(cycle, st, plan, repo):
    """Partial wave.json for the exploop handoff (merged by /exp-loop new --from-research)."""
    mod = cycle["module"]
    lay = module_layout(cycle)
    approved = [lv for lv in plan.get("levers", []) if lv["id"] in (st.get("approved_levers") or [])
                and lv["id"] not in (st.get("dropped_levers") or {})]
    allowed, coupled, queue = [], [], []
    for lv in approved:
        names = [f["name"] for f in lv.get("flags") or []]
        allowed += names
        for g in (lv.get("wave_lever") or {}).get("coupled") or []:
            if len(g) > 1:
                coupled.append(list(g))
        sets = (lv.get("wave_lever") or {}).get("set") or {}
        if len(sets) > 1 and sorted(sets) not in [sorted(g) for g in coupled]:
            coupled.append(sorted(sets))
        queue.append({"id": lv["id"], "set": sets, "targets": (lv.get("wave_lever") or {}).get("targets", ""),
                      "expect": lv.get("expect_regex") or [], "hypothesis": lv.get("hypothesis", ""),
                      "predicted_signature": lv.get("predicted_signature", ""), "falsifier": lv.get("falsifier", ""),
                      "sources": lv.get("sources") or []})
    sd = any(lv.get("state_dict_change") for lv in approved)
    prefix = cycle.get("script_prefix", mod[:2] + "_")
    base = st.get("base") or {}
    return {
        "_research": {"cycle": cycle["cycle"], "module": mod, "module_sha256": st.get("code_sha"),
                      "plan": f"rdloop/{cycle['cycle']}/plan_agreed.md", "summary": f"rdloop/{cycle['cycle']}/FINAL_SUMMARY.md",
                      "created": now_ts()},
        "trainer": {"entry": ["-m", f"{mod}.train"], "flag_sources": [f"{lay['module_dir']}train.py"]},
        "search_space": {"allowed_add": sorted(set(allowed)), "coupled_add": coupled},
        "lever_queue": queue,
        "lever_notes": [f"{lv['id']}: {lv.get('mechanism', '')[:300]}" for lv in approved],
        "early_check": {"expect_add": [f"^{mod}: "]},
        "eval": {"profile": f"{prefix}local17" if sd else "sls_local17"},
        "gpu": {"busy_regex_add": "b\\d+wave\\.(train|checks)"},
        "smoke": "off",
        "start": {"best": {"label": base.get("label"), "run_dir": base.get("run_dir"), "readouts": base.get("readouts", {}),
                           "registry_ids": base.get("registry_ids", {}), "argv_from": "config"}},
    }


def merge_seed(wave, seed, name, report_dir):
    """Full wave.json = the latest wave's spec + the seed (used for the COMPAT dry run and by the
    exp-loop handoff)."""
    w = copy.deepcopy(wave)
    w["wave"] = name
    w["description"] = f"{name}: exploration of the levers of rdloop cycle {seed['_research']['cycle']} ({seed['_research']['module']})"
    w["report_dir"] = report_dir
    w["track_prefix"] = name
    w["trainer"] = deep_merge(w.get("trainer", {}), {k: v for k, v in seed["trainer"].items()})
    ss = w.setdefault("search_space", {})
    ss["allowed"] = sorted(set(ss.get("allowed", [])) | set(seed["search_space"]["allowed_add"]))
    ss["coupled"] = list(ss.get("coupled", [])) + [g for g in seed["search_space"]["coupled_add"] if g not in ss.get("coupled", [])]
    w["lever_queue"] = seed["lever_queue"]
    w["lever_notes"] = seed["lever_notes"]
    ec = w.setdefault("early_check", {})
    expect = list(ec.get("expect", []))
    for x in seed["early_check"]["expect_add"]:
        if x not in expect:
            expect.append(x)
    ec["expect"] = expect
    w.setdefault("eval", {})["profile"] = seed["eval"]["profile"]
    g = w.setdefault("gpu", {})
    br = g.get("busy_regex", "main_train|b3wave\\.train|local_eval|main_eval|eval_checkpoints")
    if seed["gpu"]["busy_regex_add"] not in br:
        g["busy_regex"] = br + "|" + seed["gpu"]["busy_regex_add"]
    w["smoke"] = seed["smoke"]
    w["start"] = {"best": seed["start"]["best"]}
    w.pop("context", None)
    return w


# ---------------------------------------------------------------------------- STATUS.md

def _dur(s):
    if s is None:
        return "?"
    s = float(s)
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.0f}m"
    return f"{s / 3600:.1f}h"


def render_status(cycle, st, cycle_dir):
    c = cycle["cycle"]
    lines = [f"# rdloop cycle {c}: {st.get('status')} - phase {str(st.get('phase', '')).upper()}", "",
             f"_updated {now_ts()} by the supervisor; read-only view of state.json_", ""]
    b = st.get("base") or {}
    lines += [f"- **module:** `SLS_setup/{cycle['module']}/` (from `{b.get('entry_module') or '?'}`)",
              f"- **base:** {b.get('label', '?')} `{b.get('run_dir', '?')}`",
              f"- **plan gate:** {get(cycle, 'gates', 'plan', default='owner')}",
              f"- **Claude spend:** ${float(st.get('cost_usd', 0)):.2f} / ${get(cycle, 'claude', 'max_total_usd', default=150)}"]
    if st.get("blocked"):
        bl = st["blocked"]
        lines += ["", f"## BLOCKED in {bl.get('phase')}", "", f"{bl.get('reason')}", "", f"hint: {bl.get('hint')}"]
    if st.get("needs_user"):
        lines += ["", "## Needs the owner", ""] + [f"- {x}" for x in st["needs_user"]]
    if st.get("waiting"):
        lines += ["", f"waiting: {st['waiting']}"]
    if st.get("next"):
        lines += ["", f"**next:** {st['next']}"]
    team = st.get("team") or {}
    if team:
        lines += ["", "## Team", "", "| agent | state | step | headline | cost |", "|---|---|---|---|---|"]
        for a in AGENTS:
            t = team.get(a)
            if t:
                lines.append(f"| {a} | {t.get('state')} | {t.get('step', '')} | {t.get('headline', '')} | "
                             f"${float(t.get('cost', 0)):.2f} |")
    d = st.get("debate") or {}
    if d.get("rounds"):
        lines += ["", "## Debate", "", "| round | levers | blocker | major | minor | resolved |", "|---|---|---|---|---|---|"]
        for r in d["rounds"]:
            cnt = r.get("counts", {})
            lines.append(f"| v{r['v']} | {', '.join(r.get('levers', []))} | {cnt.get('blocker', 0)} | {cnt.get('major', 0)} | "
                         f"{cnt.get('minor', 0)} | {cnt.get('resolved', 0)} |")
    if st.get("approved_levers"):
        lines += ["", f"approved levers: {', '.join(st['approved_levers'])}"]
    if st.get("dropped_levers"):
        lines += ["", "dropped: " + "; ".join(f"{k} ({v})" for k, v in st["dropped_levers"].items())]
    tests = st.get("tests") or {}
    for key in ("static", "gpu", "compat"):
        jobs = (tests.get(key) or {}).get("jobs") or []
        if jobs:
            lines += ["", f"## {key.upper()}", ""]
            for j in jobs:
                mark = {"pass": "PASS", "fail": "FAIL", "running": "RUN ", "pending": "....", "skipped": "SKIP"}.get(j.get("state"), j.get("state"))
                lines.append(f"- {mark} {j['name']} {_dur(j.get('elapsed_s'))} {j.get('note', '')}".rstrip())
    lines += ["", f"events: `{rel(REPO, Path(cycle_dir) / 'events.jsonl')}` · supervisor log: "
                  f"`{rel(REPO, Path(cycle_dir) / 'logs' / 'supervisor.out')}`"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- step settings

DENY_TOOLS = ["Agent", "CronCreate", "CronDelete", "RemoteTrigger", "ScheduleWakeup", "PushNotification", "Workflow",
              "EnterWorktree", "ExitWorktree", "SendMessage", "TaskStop", "Monitor", "NotebookEdit", "ShareOnboardingGuide",
              "ListAgents", "ReportFindings", "DesignSync", "CronList", "Artifact", "ArtifactComments", "ArtifactData"]
GIT_WRITES = ["commit", "push", "add", "reset", "checkout", "restore", "clean", "stash", "rm", "mv", "rebase", "merge",
              "tag", "switch", "cherry-pick", "revert", "apply", "am", "worktree", "pull", "fetch"]
READ_CMDS = ["ls", "cat", "head", "tail", "grep", "rg", "find", "wc", "diff", "sort", "uniq", "cut", "awk", "sed -n",
             "stat", "file", "du", "jq", "nvidia-smi", "which", "pwd", "echo", "date", "basename", "dirname", "realpath",
             "git status", "git diff", "git log", "git show", "git ls-files", "git rev-parse", "git blame", "git grep"]


def step_settings(policy, repo, cycle, cycle_dir, write_files, write_prefixes, py, hook_py="python3"):
    """--settings JSON for one headless step: deny rules (not a boundary; merged with project
    settings and never overridable), allow rules for what the step must do, and the guard hook."""
    repo = str(Path(repo).resolve())
    cyc = str(Path(cycle_dir).resolve())

    def ab(p):
        return "/" + p if p.startswith("/") else p

    deny = [f"Read({ab(repo)}/SLS_setup/exp/**/model_merging_*/scores.txt)", "Read(//mnt/paml-research/RTCFake/data/progress.txt)",
            "Read(//mnt/paml-research/RTCFake/data/wav/progress/**)"]
    deny += [f"Bash(git {g}:*)" for g in GIT_WRITES]
    deny += [f"Bash({c}:*)" for c in ("pip", "pip3", "conda", "mamba", "micromamba", "apt", "apt-get", "dpkg", "sudo", "kill",
                                       "pkill", "killall", "claude", "nohup", "setsid", "curl", "wget", "ssh", "scp", "rsync",
                                       "ln", "dd", "chmod", "chown", "shred")]
    deny += list(DENY_TOOLS)
    if policy not in ("scout", "critic"):
        deny += ["WebFetch", "WebSearch"]
    protected_cycle = ["settings/**", "isolation/**", "tests/**", "handoff/**", "debate/**", "review/**", "steps/**",
                       "state.json", "cycle.json", "approval.json", "plan_agreed.json", "plan_agreed.md", "events.jsonl",
                       "brief.md", "user_notes.md"]
    allowed_abs = [str(Path(f).resolve()) for f in write_files]
    for p in protected_cycle:
        full = f"{cyc}/{p}"
        base = full.rstrip("*").rstrip("/")
        if base in allowed_abs:
            continue
        if any(a.startswith(base + "/") for a in allowed_abs):
            # deny beats allow, so a glob over the step's own file (debate/proposal_vN.md) would block it: deny the
            # other files already there instead; the guard hook still limits the step to its assigned files
            deny += [f"Edit({ab(str(f))})" for f in sorted(Path(base).rglob("*")) if f.is_file() and str(f) not in allowed_abs]
            continue
        deny.append(f"Edit({ab(full)})")
    outside = [f"{repo}/configs/**", f"{repo}/data/**", f"{repo}/.claude/**", f"{repo}/exploop/**", f"{repo}/docs/**",
               "/root/.claude/**", "/mnt/**"]
    if policy == "engineer":
        lay = module_layout(cycle)
        keep_sls = {lay["module_dir"].split("/")[1], Path(lay["shim"]).name}
        sls = Path(repo) / "SLS_setup"
        for ch in sorted(sls.iterdir()) if sls.is_dir() else []:
            if ch.name in keep_sls:
                continue
            outside += [f"{repo}/SLS_setup/{ch.name}"] + ([f"{repo}/SLS_setup/{ch.name}/**"] if ch.is_dir() else [])
        sp = Path(lay["scripts_prefix"]).name
        scr = Path(repo) / "scripts"
        for ch in sorted(scr.iterdir()) if scr.is_dir() else []:
            if ch.name.startswith(sp) or (lay["profile_prefix"].startswith("scripts/exploop/") and ch.name == "exploop"
                                          and cycle.get("_state_dict_change")):
                continue
            outside += [f"{repo}/scripts/{ch.name}"] + ([f"{repo}/scripts/{ch.name}/**"] if ch.is_dir() else [])
        if cycle.get("_state_dict_change"):
            exp_dir = scr / "exploop"
            for ch in sorted(exp_dir.iterdir()) if exp_dir.is_dir() else []:
                if ch.name != "profiles":
                    outside += [f"{repo}/scripts/exploop/{ch.name}"] + ([f"{repo}/scripts/exploop/{ch.name}/**"] if ch.is_dir() else [])
    else:
        outside += [f"{repo}/SLS_setup/**", f"{repo}/scripts/**"]
    deny += [f"Edit({ab(p)})" for p in outside]
    allow = [f"Bash({c}:*)" for c in READ_CMDS] + [f"Bash({py}:*)", "Bash(python:*)", "Bash(python3:*)", "Read", "Grep", "Glob"]
    allow += [f"Edit({ab(f)})" for f in allowed_abs]
    if policy == "engineer":
        allow += [f"Edit({ab(p.rstrip('/'))}/**)" if p.endswith("/") else f"Edit({ab(p)}*)" for p in write_prefixes]
        allow += [f"Bash({c}:*)" for c in ("mkdir", "touch", "cp", "mv", "rm", "tee")]
    if policy in ("scout", "critic"):
        allow += ["WebSearch"] + [f"WebFetch(domain:{d})" for d in get(cycle, "web", "domains", default=DEFAULT_WEB_DOMAINS)]
    hook = f"{hook_py} -I {HERE / 'guard_hook.py'}"
    return {
        "$comment": f"Generated by scripts/rdloop/supervisor.py for one headless {policy} step of cycle {cycle['cycle']}; "
                    "deny rules + the guard hook are seatbelts, the isolation manifest is the boundary.",
        "permissions": {"deny": deny, "allow": allow},
        "hooks": {"PreToolUse": [{"matcher": "Edit|Write|MultiEdit|NotebookEdit|Bash|Read|Grep|Glob|WebFetch|WebSearch",
                                  "hooks": [{"type": "command", "command": hook, "timeout": 20}]}]},
    }
