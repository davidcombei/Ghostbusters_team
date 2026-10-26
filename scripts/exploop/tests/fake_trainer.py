#!/usr/bin/env python3
"""Stand-in trainer for the exploop tests: same files and log lines as b3wave.train / main_train.py
(config.yaml with `command`, logs/train.log, metrics.jsonl, ckpt/epoch_N_*.pth, last_state.pt, tqdm
'Training:' lines, 'Experiment saved to:'), and the same --resume semantics, in seconds instead of hours.

env: EXPLOOP_FAKE_EPOCH_S (seconds per epoch, default 2), EXPLOOP_FAKE_DIE_AT=<epoch> (os._exit in
that epoch once, no traceback = a container restart / kill -9), EXPLOOP_FAKE_BAD_HEADER=1 (omit the
header lines), EXPLOOP_FAKE_CRASH_AT=<epoch> (raise -> Traceback).
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exploop_lib as L  # noqa: E402


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    argv = sys.argv[1:]
    a = L.argv_dict(argv)
    track = a["--track"][0]
    n = int(a.get("--num_epochs", ["3"])[0])
    bs = a.get("--batch_size", ["16"])[0]
    out = Path(a.get("--out_path", ["./exp"])[0])
    per = float(os.environ.get("EXPLOOP_FAKE_EPOCH_S", "2"))
    resume = a.get("--resume", [None])[0]
    if not os.environ.get("EXPLOOP_FAKE_BAD_HEADER"):
        print("Device: in=cpu out=cpu")
        print("b3wave: fake 0.0")
        print("Loudness norm: loudness_norm target=-23 LUFS" if "--loudness_norm" in a else "Loudness norm: off")
        print("Local eval: dev_online_clean(20), sim_matched_mini(10), sim_heldout_mini(10)")
        print(f"Warmup-cosine: 1 warmup / {n * 100} steps, floor 5.0e-07")
        for flag, vals in L.parse_argv(argv):
            print(f"FLAG {flag}={' '.join(vals)}")
    sys.stdout.flush()
    if resume:
        run = Path(resume).resolve()
        rows = L.metrics_epochs(run)
        start = max([r["epoch"] for r in rows] or [0]) + 1
        with open(run / "logs" / "train.log", "a") as f:
            f.write(f"[{ts()}] Resumed {run} after epoch {start - 1}: {' '.join(argv)}\n")
    else:
        run = (out / f"{track}_epoch{n}_bs{bs}_{datetime.now().strftime('%Y%m%d%H%M%S')}").resolve()
        (run / "logs").mkdir(parents=True)
        (run / "ckpt").mkdir()
        cfg = {"created": datetime.now().isoformat(timespec="seconds"), "command": " ".join([__file__] + argv),
               "git_sha": "0" * 40, "git_dirty": False, "host": "test",
               "args": {"track": track, "num_epochs": n, "seed": int(a.get("--seed", ["1234"])[0]), "devices": None,
                        "ssl_name": "fake/ssl", "n_layers": 2, "loudness_norm": "--loudness_norm" in a,
                        "loudness_norm_lufs": float(a.get("--loudness_norm_lufs", ["-23"])[0])}}
        import yaml
        (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        with open(run / "logs" / "train.log", "a") as f:
            f.write(f"[{ts()}] start {track}\n")
        start = 1
    die_at = int(os.environ.get("EXPLOOP_FAKE_DIE_AT", "0") or 0)
    crash_at = int(os.environ.get("EXPLOOP_FAKE_CRASH_AT", "0") or 0)
    for ep in range(start, n + 1):
        steps = 100
        for i in (25, 50, 75, 100):
            time.sleep(per / 4)
            print(f"\rTraining:  {i}%|##        | {i * steps // 100}/{steps} [00:0{i // 25}<00:0{4 - i // 25}, "
                  f"{steps / max(per, 0.01):.2f}batch/s]", end="", flush=True)
            if ep == die_at and i == 50 and not (run / "died_once").exists():
                (run / "died_once").write_text("x")
                os._exit(137)
            if ep == crash_at and i == 50:
                raise RuntimeError("fake crash")
        print()
        wf1 = 0.90 + 0.002 * ep
        ck = run / "ckpt" / f"epoch_{ep}_dev_loss_0.{100 + ep}.pth"
        ck.write_text("fake")
        with open(run / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"epoch": ep, "train_loss": 0.1 / ep, "dev_loss": 0.1, "ckpt": str(ck),
                                "wf1": {"wf1": wf1}}) + "\n")
        with open(run / "logs" / "train.log", "a") as f:
            f.write(f"[{ts()}] Epoch {ep}/{n} TrainLoss={0.1 / ep:.6f} TrainAcc=99.00% DevLoss=0.100000 DevAcc=98.00% "
                    f"LR=1.000e-06 HeadLR=1.000e-05\n")
            f.write(f"[{ts()}] Epoch {ep} LocalWF1={100 * wf1:.2f} F1clean=97.00 F1noisy=90.00 WF1oracle=93.00 EERnoisy=10.00\n")
        print(f"Epoch {ep}/{n} TrainLoss={0.1 / ep:.6f} TrainAcc=99.00% DevLoss=0.100000 DevAcc=98.00%", flush=True)
        (run / "last_state.pt").write_text(str(ep))
    rel = os.path.relpath(run, Path.cwd())
    print(f"Experiment saved to: ./{rel}" if not rel.startswith("..") else f"Experiment saved to: {run}", flush=True)


if __name__ == "__main__":
    main()
