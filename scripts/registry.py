"""
Experiment registry (docs/experiment_log.csv): one row per evaluated model.

Torch-free on purpose, so it can be used without a GPU env: scripts/local_eval.py (--registry)
and scripts/log_experiment.py (the /log-experiment skill) both build rows here, from a
local_eval metrics.json plus the training run dir (config.yaml, logs/train.log).
"""

import csv
import re
import subprocess
from datetime import date, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SLS_DIR = REPO_ROOT / "SLS_setup"
DEFAULT_REGISTRY = REPO_ROOT / "docs/experiment_log.csv"

REGISTRY_COLUMNS = [
    "id", "date", "owner", "git_sha", "config_hash", "model", "change_vs_ref",
    "clean_f1", "matched_f1", "heldout_f1", "echo_f1", "wf1@0.5", "wf1@oracle",
    "eer_clean", "eer_matched", "eer_heldout", "eer_echo",
    "recall_spoof_noisy", "recall_bona_noisy", "progress_score", "gpu_h", "decision", "notes",
    "seed", "half_b_wf1", "ci_low", "ci_high",
]

# The comparable report (docs/dev_splits.md): pass exactly these to local_eval.py --names.
REPORT_SETS = [
    "dev_online_clean", "sim_matched_v1", "sim_heldout_v1", "sim_echo_v1",
    "probe_orig", "probe_sil_only", "probe_noise_only", "probe_trimmed", "probe_padded",
    "probe_gain_up", "probe_gain_down", "probe_gain_m20", "probe_gain_m6", "probe_gain_p6",
    "probe_gain_p10_limit", "probe_agc_dynaudnorm", "probe_loudnorm",
]

LOG_TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*)$")
# Lines main_train.py writes when --resume starts: the gap that ends on one of them is downtime.
RESUME_RE = re.compile(r"^(Resumed |Dropped metrics rows of unfinished epochs|Removed .*\(unfinished epoch\))")


def git(*cmd):
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *cmd],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def owner_handle():
    """Local part of git user.email, e.g. 'tmarchitan'."""
    return git("config", "user.email").split("@")[0]


def run_dir_for(ckpt_path):
    """exp/<run>/ for a checkpoint anywhere below it (paths may be relative to SLS_setup/)."""
    p = Path(ckpt_path)
    if not p.is_absolute():
        p = SLS_DIR / p
    for parent in p.parents:
        if parent.parent.name == "exp":
            return parent
    return None


def read_run_config(run_dir):
    """Training-time facts from exp/<run>/config.yaml; {} for legacy runs without one."""
    if run_dir is None or not (Path(run_dir) / "config.yaml").is_file():
        return {}
    import yaml
    with open(Path(run_dir) / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    args = cfg.get("args") or {}
    sha = (cfg.get("git_sha") or "")[:7]
    if sha and cfg.get("git_dirty"):
        sha += "+dirty"
    return {"git_sha": sha, "seed": args.get("seed"), "devices": args.get("devices"),
            "created": cfg.get("created")}


def gpu_hours(run_dir):
    """GPU-hours of a run: train.log wall-clock (from config.yaml 'created'), minus the downtime
    before each --resume, times the number of GPUs. '' when it cannot be derived."""
    cfg = read_run_config(run_dir)
    log = Path(run_dir) / "logs/train.log" if run_dir else None
    if not cfg.get("created") or log is None or not log.is_file():
        return ""
    created = cfg["created"]
    prev = created if isinstance(created, datetime) else datetime.fromisoformat(str(created))
    seconds = 0.0
    with open(log, encoding="utf-8") as f:
        for line in f:
            m = LOG_TS_RE.match(line.rstrip("\n"))
            if not m:
                continue
            ts = datetime.fromisoformat(m.group(1))
            if not RESUME_RE.match(m.group(2)):
                seconds += max((ts - prev).total_seconds(), 0.0)
            prev = ts
    n_gpus = len(cfg["devices"]) if cfg.get("devices") else 1
    return f"{seconds / 3600 * n_gpus:.2f}" if seconds else ""


def pct(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.2f}"


def build_row(report, *, id="", owner="", change_vs_ref="", notes="", decision="",
              progress_score="", ci_low="", ci_high="", run_dir=None):
    """One registry row from a local_eval report (metrics.json). Training facts (git_sha, seed,
    gpu_h) come from run_dir, which defaults to the run of the first checkpoint."""
    sets = report["full"]["sets"]

    def by_role(role, key):
        vals = [s[key] for s in sets.values() if s["role"] == role]
        return pct(vals[0]) if vals else ""

    if run_dir is None and report.get("checkpoints"):
        run_dir = run_dir_for(report["checkpoints"][0])
    cfg = read_run_config(run_dir)
    noisy = [s for s in sets.values() if s["role"] in ("matched", "heldout")]
    w = report["full"]["wf1"]
    wb = (report.get("half_B") or {}).get("wf1")
    seed = cfg.get("seed")
    return {
        "id": id, "date": date.today().isoformat(), "owner": owner,
        "git_sha": cfg.get("git_sha") or git("rev-parse", "--short", "HEAD"),
        "config_hash": report["config_hash"], "model": report["model"],
        "change_vs_ref": change_vs_ref,
        "clean_f1": by_role("clean", "f1"), "matched_f1": by_role("matched", "f1"),
        "heldout_f1": by_role("heldout", "f1"), "echo_f1": by_role("echo", "f1"),
        "wf1@0.5": pct(w["wf1"]) if w else "",
        "wf1@oracle": pct(w["wf1_oracle"]) if w else "",
        "eer_clean": by_role("clean", "eer"), "eer_matched": by_role("matched", "eer"),
        "eer_heldout": by_role("heldout", "eer"), "eer_echo": by_role("echo", "eer"),
        "recall_spoof_noisy": pct(np.mean([s["recall_spoof"] for s in noisy])) if noisy else "",
        "recall_bona_noisy": pct(np.mean([s["recall_bona"] for s in noisy])) if noisy else "",
        "progress_score": progress_score, "gpu_h": gpu_hours(run_dir), "decision": decision,
        "notes": notes, "seed": "" if seed is None else str(seed),
        "half_b_wf1": pct(wb["wf1"]) if wb else "", "ci_low": ci_low, "ci_high": ci_high,
    }


def read_registry(path):
    """Rows as dicts with every REGISTRY_COLUMNS key (missing columns of an old header -> '')."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{c: (r.get(c) or "") for c in REGISTRY_COLUMNS} for r in csv.DictReader(f)]


def write_registry(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_row(path, row, replace=False):
    """Append row (or replace the row with the same id when replace=True). An existing id is
    refused otherwise. A file with an older header is rewritten with the current columns."""
    path = Path(path)
    rows = read_registry(path)
    same = [i for i, r in enumerate(rows) if row["id"] and r["id"] == row["id"]]
    if same and not replace:
        raise ValueError(f"id '{row['id']}' already in {path} (use --replace or another id)")
    if same:
        rows[same[0]] = row
    else:
        rows.append(row)
    write_registry(path, rows)
    return "replaced" if same else "appended"
