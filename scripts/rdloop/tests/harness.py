"""Scratch repo for the rdloop integration tests: a git repo with a fake b3wave trainer / checks, a fake
scripts/local_eval.py, the REAL exploop supervisor (for the handoff dry run), a finished base run with
stored readouts, a DONE exploop wave W1 whose best is that run, and an rdloop cycle T1 driven by
fake_claude.py. The real repo is never touched."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
RDLOOP = TESTS.parent
REPO = RDLOOP.parents[1]
PY = "/root/.conda/envs/rtc-sdd/bin/python" if Path("/root/.conda/envs/rtc-sdd/bin/python").exists() else sys.executable
sys.path.insert(0, str(RDLOOP))
import rdlib as R  # noqa: E402

L = R.L
FAKES = TESTS / "fakes"


def sh(*cmd, cwd=None, env=None):
    return subprocess.run([str(c) for c in cmd], cwd=cwd, env={**os.environ, **(env or {})}, check=True,
                          capture_output=True, text=True)


def make_repo(root, gate="auto", cycle_over=None):
    root = Path(root)
    repo = root / "repo"
    sls = repo / "SLS_setup"
    for d in ("b3wave", "utils", "model", "exp"):
        (sls / d).mkdir(parents=True, exist_ok=True)
    (sls / "b3wave" / "__init__.py").write_text('__version__ = "0.1.0"\n')
    shutil.copy(FAKES / "b3wave_train.py", sls / "b3wave" / "train.py")
    shutil.copy(FAKES / "b3wave_checks.py", sls / "b3wave" / "checks.py")
    (sls / "utils" / "__init__.py").write_text("")
    (sls / "utils" / "rtc_augment.py").write_text("# original augmenter\n")
    (sls / "model" / "sls_model.py").write_text("# original model\n")
    (repo / "scripts" / "exploop").mkdir(parents=True)
    shutil.copy(FAKES / "local_eval.py", repo / "scripts" / "local_eval.py")
    (repo / "scripts" / "b3_smoke.sh").write_text("# old smoke\n")
    for f in ("supervisor.py", "exploop_lib.py", "launcher_template.sh"):
        shutil.copy(REPO / "scripts" / "exploop" / f, repo / "scripts" / "exploop" / f)
    shutil.copytree(REPO / "scripts" / "exploop" / "profiles", repo / "scripts" / "exploop" / "profiles")
    (repo / "configs").mkdir()
    (repo / "configs" / "local_eval_sets.yaml").write_text("sets: {}\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.md").write_text("# notes\n")
    (repo / "docs" / "experiment_log.csv").write_text("id,date\n")
    (repo / ".gitignore").write_text("SLS_setup/exp\n__pycache__\n*.log\n")
    sh("git", "init", "-q", cwd=repo)
    sh("git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A", cwd=repo)
    sh("git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base", cwd=repo)
    # the base run (W1a) + its readouts
    sh(PY, "-m", "b3wave.train", "--track", "W1a_ref_s1234", "--num_epochs", "3", "--batch_size", "16", "--loudness_norm",
       "--loudness_norm_lufs", "-23", "--keep_topk_by_wf1", "7", "--keep_epochs", "3", "--seed", "1234", "--out_path", "./exp",
       cwd=sls, env={"RDLOOP_FAKE_EPOCH_S": "0.4"})
    run = next((sls / "exp").glob("W1a_ref_s1234_epoch3_bs16_*"))
    ck = next((run / "ckpt").glob("epoch_3_*.pth"))
    for tag in ("epoch_3", "merge_top5"):
        sh(PY, repo / "scripts" / "local_eval.py", "--model_path", ck, "--names", "s1", "probe_orig", "--out", run / "local_eval" / tag)
    rel_run = str(run.relative_to(repo))
    best = {"label": "W1a", "iter": 1, "run_dir": rel_run, "track": "W1a_ref_s1234",
            "readouts": {"merge": rel_run + "/local_eval/merge_top5", "final": rel_run + "/local_eval/epoch_3"},
            "registry_ids": {}, "argv": L.config_argv(run), "merge_value": 93.0, "final_value": 93.0}
    wd = repo / "exploop" / "W1"
    wd.mkdir(parents=True)
    R.write_json(wd / "loop_state.json", {"wave": "W1", "status": "DONE", "best": best, "iterations": [], "current": None,
                                          "heartbeat_epoch": time.time() - 7200})
    R.write_json(wd / "wave.json", {
        "wave": "W1", "repo": str(repo), "python": PY, "report_dir": "docs/experiments-exploration/w1", "registry": "docs/experiment_log.csv",
        "max_iterations": 3, "letters": "abc", "track_prefix": "W1",
        "trainer": {"entry": ["-m", "b3wave.train"], "cwd": "SLS_setup", "exp_root": "SLS_setup/exp",
                    "launcher": "scripts/run_{prefix}{letter}.sh", "flag_sources": ["SLS_setup/b3wave/train.py"], "preflight": [],
                    "smoke_args": ["--num_epochs", "1"]},
        "search_space": {"allowed": ["--lr", "--lr_min"], "coupled": [["--lr", "--lr_min"]], "forbidden": ["--out_path"],
                         "fixed": {"--loudness_norm": [], "--loudness_norm_lufs": ["-23"]}, "max_epochs": 10,
                         "keep_topk_by_wf1": 7, "seed": 1234},
        "lever_queue": [], "lever_notes": [], "smoke": "off",
        "early_check": {"minutes": 1, "expect": ["^b3wave: "], "forbid": ["Traceback"]},
        "eval": {"profile": "sls_local17", "k": 5, "ci": 200, "names": ["s1", "probe_orig"], "fixed_comparators": []},
        "objective": {"key": "half_B.wf1.wf1", "clean_key": "half_B.wf1.f1_clean", "higher_is_better": True},
        "rules": {"new_best_delta": 0.0, "band": 0.5, "promote": {"ci_low_gt": 0.0, "clean_drop_le": 0.5},
                  "positive": {"delta_gt": 0.5, "ci_low_gt": 0.0, "clean_drop_le": 0.5}},
        "gpu": {"enabled": False, "busy_regex": "main_train|b3wave\\.train"}, "disk_gb": {}, "timeouts": {}, "retries": {},
        "integrity": {"protected": []}, "claude": {"bin": "claude", "steps": {}},
        "start": {"best": {"label": "W1a", "run_dir": rel_run, "readouts": best["readouts"], "registry_ids": {}, "argv_from": "config"}}})
    # the cycle
    cyc = json.loads((RDLOOP / "templates" / "cycle.json").read_text().replace("@CYCLE@", "T1").replace("@MODULE@", "t1wave")
                     .replace("@PREFIX@", "t1_").replace("@FROM_WAVE@", "W1").replace("@DESCRIPTION@", "test cycle"))
    cyc.update(repo=str(repo), python=PY, wait_for_waves=["W1"])
    cyc["gates"]["plan"] = gate
    cyc["gpu"].update(enabled=False, yield_to_exploop=False, poll_s=1, debounce_s=0)
    cyc["disk_gb"]["gpu"] = 0
    cyc["smoke"].update(s0_minutes=1, evalpath_minutes=1, overfit_minutes=1)
    cyc["regression"].update(names=["s1", "probe_orig"], base_checks=[["model", "--ckpt", "{base_ckpt}"]])
    cyc["static"].update(exploop_tests=[])
    cyc["claude"]["bin"] = str(TESTS / "fake_claude.py")
    for s in cyc["claude"]["steps"].values():
        s.update(timeout_min=2, attempts=2)
    cyc["context"]["docs"] = ["docs/notes.md"]
    for k, v in (cycle_over or {}).items():
        cyc[k] = R.deep_merge(cyc.get(k, {}), v) if isinstance(v, dict) else v
    cd = repo / "rdloop" / "T1"
    cd.mkdir(parents=True)
    R.write_json(cd / "cycle.json", cyc)
    (cd / "brief.md").write_text("# brief\n\nfind two levers\n")
    return repo, cd


def sup(cd, *args, env=None, timeout=600):
    return subprocess.run([PY, str(RDLOOP / "supervisor.py"), *args, "--cycle", str(cd)], capture_output=True, text=True,
                          env={**os.environ, **(env or {})}, timeout=timeout)


def start(cd, env=None):
    log = open(Path(cd) / "supervisor.test.out", "a")
    return subprocess.Popen([PY, "-u", str(RDLOOP / "supervisor.py"), "run", "--cycle", str(cd)], stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, **(env or {})})


def state(cd):
    return R.read_json(Path(cd) / "state.json") or {}


def wait_for(pred, timeout=240, poll=0.5, what="condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if pred():
                return True
        except (KeyError, IndexError, TypeError, ValueError, StopIteration):
            pass
        time.sleep(poll)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")
