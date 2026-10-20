"""Build a scratch test wave (fake trainer / eval / claude, real supervisor, real log_experiment.py
on a scratch copy of the registry) and helpers to drive it."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
EXPLOOP = TESTS.parent
REPO = EXPLOOP.parents[1]
PY = "/root/.conda/envs/rtc-sdd/bin/python" if Path("/root/.conda/envs/rtc-sdd/bin/python").exists() else sys.executable
B2F = REPO / "SLS_setup/exp/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623"

sys.path.insert(0, str(EXPLOOP))
import exploop_lib as L  # noqa: E402

BASE_ARGV = ["--track", "T0_ref_s1234", "--num_epochs", "3", "--batch_size", "16", "--lr", "2.5e-6",
             "--lr_min", "5e-7", "--loudness_norm", "--loudness_norm_lufs", "-23", "--keep_topk_by_wf1", "7",
             "--keep_epochs", "3", "--seed", "1234"]


def make_wave(root, max_iterations=2, slow_s=6, extra_env=None):
    root = Path(root)
    work, launchers, reports, wave_dir = root / "work", root / "launchers", root / "reports", root / "wave"
    for d in (work / "exp", launchers, reports, wave_dir, root / "bin"):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "docs/experiment_log.csv", root / "registry.csv")
    # reference run (the starting best) produced by the fake trainer + fake readouts
    env = {**os.environ, "EXPLOOP_FAKE_EPOCH_S": "0"}
    subprocess.run([PY, str(TESTS / "fake_trainer.py"), *BASE_ARGV], cwd=work, env=env, check=True, capture_output=True)
    ref = next((work / "exp").glob("T0_ref_s1234_epoch3_bs16_*"))
    for kind, src, tag in (("merge", B2F / "local_eval/merge_top5/metrics.json", "merge_top5"),
                           ("final", B2F / "local_eval/epoch_8/metrics.json", "epoch_3")):
        subprocess.run([PY, str(TESTS / "fake_eval.py"), "readout", "--src", str(src), "--out", str(ref / "local_eval" / tag),
                        "--label", "T0", "--kind", kind], check=True, capture_output=True,
                       env={**os.environ, "EXPLOOP_FAKE_SHIFTS": "T0:0"})
    prof = (TESTS / "profile_fake.json").read_text()
    prof = prof.replace("@SRC_MERGE@", str(B2F / "local_eval/merge_top5/metrics.json")).replace(
        "@SRC_FINAL@", str(B2F / "local_eval/epoch_8/metrics.json")).replace("@SLOW_S@", str(slow_s))
    (root / "profile.json").write_text(prof)
    wave = {
        "wave": "T1", "repo": str(REPO), "python": PY, "report_dir": str(reports), "registry": str(root / "registry.csv"),
        "report_template": "docs/experiments-exploration/20261009_175355_B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234.md",
        "max_iterations": max_iterations, "letters": "abcd", "track_prefix": "T1",
        "trainer": {"entry": [str(TESTS / "fake_trainer.py")], "cwd": str(work), "exp_root": str(work / "exp"),
                    "launcher": str(launchers / "run_{prefix}{letter}.sh"), "flag_sources": [],
                    "preflight": ["ffmpeg"],
                    "ffmpeg_install_cmd": f"ln -sf $(command -v ffmpeg) {root / 'bin' / 'ffmpeg-fake'}",
                    "smoke_args": ["--num_epochs", "1", "--keep_epochs", "1", "--out_path", "./exp_smoke"]},
        "search_space": {"allowed": ["--lr", "--lr_min", "--aug_codec_profiles", "--specaug", "--aug_snr_db", "--num_epochs"],
                         "coupled": [["--lr", "--lr_min"]], "forbidden": ["--out_path"],
                         "fixed": {"--loudness_norm": [], "--loudness_norm_lufs": ["-23"]},
                         "max_epochs": 10, "keep_topk_by_wf1": 7, "seed": 1234},
        "lever_queue": [{"id": "codec_random", "set": {"--aug_codec_profiles": ["random"]}},
                        {"id": "specaug", "set": {"--specaug": []}},
                        {"id": "snr_0_20", "set": {"--aug_snr_db": ["0", "20"]}}],
        "lever_notes": ["test"],
        "smoke": "off",
        "early_check": {"minutes": 0.05, "expect": ["^b3wave: fake", "^Loudness norm: loudness_norm"],
                        "forbid": ["Traceback \\(most recent call last\\)", "out of memory", "failed \\("]},
        "eval": {"profile": str(root / "profile.json"), "k": 5, "ci": 200, "names": [], "fixed_comparators": []},
        "objective": {"key": "half_B.wf1.wf1", "clean_key": "half_B.wf1.f1_clean", "higher_is_better": True},
        "rules": {"new_best_delta": 0.0, "band": 0.5, "promote": {"ci_low_gt": 0.0, "clean_drop_le": 0.5},
                  "positive": {"delta_gt": 0.5, "ci_low_gt": 0.0, "clean_drop_le": 0.5}},
        "gpu": {"enabled": False, "poll_s": 1},
        "disk_gb": {"train": 0, "eval": 0, "smoke": 0},
        "timeouts": {"stall_min": 5, "max_run_hours": 10, "eval_step_h": 1, "smoke_min": 5},
        "retries": {"train_resume": 3, "eval_step": 2, "early_check": 2},
        "integrity": {"protected": ["SLS_setup/", "scripts/", "configs/", ".claude/", "docs/experiment_log.csv"]},
        "claude": {"bin": str(TESTS / "fake_claude.py"), "settings": "scripts/exploop/claude_settings.json",
                   "permission_mode": "auto", "model": None, "max_total_usd": 5,
                   "steps": {"design": {"agent": "exp-designer", "usd": 1, "timeout_min": 2, "attempts": 3},
                             "analyze": {"agent": "exp-analyst", "usd": 1, "timeout_min": 2, "attempts": 3},
                             "summary": {"agent": "exp-analyst", "usd": 1, "timeout_min": 2, "attempts": 2}}},
        "start": {"best": {"label": "T0", "run_dir": str(ref),
                           "readouts": {"merge": str(ref / "local_eval/merge_top5"), "final": str(ref / "local_eval/epoch_3")},
                           "registry_ids": {}, "argv_from": "config"}},
    }
    (wave_dir / "wave.json").write_text(json.dumps(wave, indent=2))
    (wave_dir / "plan_snapshot.md").write_text("# test plan\n")
    return wave_dir


def sup(wave_dir, *args, env=None, check=False):
    return subprocess.run([PY, str(EXPLOOP / "supervisor.py"), *args, "--wave", str(wave_dir)], cwd=REPO,
                          capture_output=True, text=True, env={**os.environ, **(env or {})}, check=check)


def start(wave_dir, env=None):
    """Start the supervisor detached, like loopctl.sh start."""
    log = open(Path(wave_dir) / "supervisor.test.out", "a")
    p = subprocess.Popen([PY, "-u", str(EXPLOOP / "supervisor.py"), "run", "--wave", str(wave_dir)], cwd=REPO,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                         env={**os.environ, **(env or {})})
    return p


def state(wave_dir):
    return L.read_json(Path(wave_dir) / "loop_state.json") or {}


def wait_for(pred, timeout=120, poll=0.5, what="condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if pred():
                return True
        except (KeyError, IndexError, TypeError, ValueError, StopIteration):
            pass
        time.sleep(poll)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def procs(pattern):
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            cmd = L.proc_cmdline(int(d.name))
            if pattern in cmd and "pytest" not in cmd and "unittest" not in cmd:
                out.append(int(d.name))
    return out
