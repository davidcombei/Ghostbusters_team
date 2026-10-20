"""Fake b3wave trainer for the rdloop tests: same files / log lines as SLS_setup/b3wave/train.py
(config.yaml with `command`, logs/train.log, metrics.jsonl, ckpt/epoch_N_*.pth, last_state.pt, tqdm
'Training:' lines, 'Experiment saved to:'), in seconds. Copied into a scratch repo as
SLS_setup/b3wave/train.py; the rdloop scaffold copies it again into the cycle's module."""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from b3wave import __version__ as B3_VERSION


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.5e-6)
    parser.add_argument("--lr_min", type=float, default=5e-7)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--keep_topk_by_wf1", type=int, default=7)
    parser.add_argument("--keep_epochs", type=int, default=0)
    parser.add_argument("--out_path", type=str, default="./exp")
    parser.add_argument("--dev_protocol", type=str, default="dev.txt")
    parser.add_argument("--local_eval_clean_n", type=int, default=2000)
    parser.add_argument("--ssl_name", type=str, default="fake/ssl")
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--loudness_norm", action="store_true")
    parser.add_argument("--loudness_norm_lufs", type=float, default=-23.0)
    parser.add_argument("--resume", type=str, default=None)
    b3 = parser.add_argument_group("B3 wave")
    b3.add_argument("--train_subset_frac", type=float, default=1.0)
    # @LEVER_ARGS@
    args = parser.parse_args()
    print("Device: in=cpu out=cpu")
    print(f"Loudness norm: {'loudness_norm target=-23 LUFS' if args.loudness_norm else 'off'}")
    print(f"b3wave: {B3_VERSION}")
    # @LEVER_PRINTS@
    sys.stdout.flush()
    per = float(os.environ.get("RDLOOP_FAKE_EPOCH_S", "0.4"))
    run = (Path(args.out_path) / f"{args.track}_epoch{args.num_epochs}_bs{args.batch_size}_{datetime.now().strftime('%Y%m%d%H%M%S')}").resolve()
    (run / "logs").mkdir(parents=True)
    (run / "ckpt").mkdir()
    import yaml
    cfg = {"created": datetime.now().isoformat(timespec="seconds"), "command": " ".join(sys.argv), "git_sha": "0" * 40,
           "args": {"track": args.track, "num_epochs": args.num_epochs, "seed": args.seed, "ssl_name": args.ssl_name,
                    "n_layers": args.n_layers, "loudness_norm": args.loudness_norm, "loudness_norm_lufs": args.loudness_norm_lufs}}
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    bump = float(os.environ.get("RDLOOP_FAKE_LEVER_BUMP", "0.5"))
    extra = bump if getattr(args, "aug_x", False) else 0.0
    if getattr(args, "head_y", 0.0):
        extra += 0.1 * args.head_y
    for ep in range(1, args.num_epochs + 1):
        steps = 100
        for i in (25, 50, 75, 100):
            time.sleep(per / 4)
            print(f"\rTraining:  {i}%|##        | {i * steps // 100}/{steps} [00:0{i // 25}<00:0{4 - i // 25}, "
                  f"{steps / max(per, 0.01):.2f}batch/s]", end="", flush=True)
        print()
        tl = 0.1 / ep * (1 + extra)
        ck = run / "ckpt" / f"epoch_{ep}_dev_loss_0.{100 + ep}.pth"
        ck.write_text(f"weights {args.track} {ep} {extra}")
        with open(run / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"epoch": ep, "train_loss": tl, "dev_loss": 0.1, "ckpt": str(ck), "wf1": {"wf1": 0.9}}) + "\n")
        with open(run / "logs" / "train.log", "a") as f:
            f.write(f"[{ts()}] Epoch {ep}/{args.num_epochs} TrainLoss={tl:.6f} TrainAcc=99.00% DevLoss=0.100000 DevAcc=98.00% "
                    f"LR=1.000e-06 HeadLR=1.000e-05\n")
            f.write(f"[{ts()}] Epoch {ep} LocalWF1=90.00 F1clean=97.00 F1noisy=90.00 WF1oracle=93.00 EERnoisy=10.00\n")
        (run / "last_state.pt").write_text(str(ep))
    print(f"Experiment saved to: {run}", flush=True)


if __name__ == "__main__":
    main()
