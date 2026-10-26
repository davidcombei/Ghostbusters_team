#!/usr/bin/env python3
"""Stand-in eval commands for the exploop tests.

  fake_eval.py readout --src <real metrics.json> --out <eval dir> --label <iter label> --kind merge|final
  fake_eval.py ci --out <eval dir>/compare_ci.json --label <iter label> --kind merge|final
  fake_eval.py sleep <seconds>

The readout copies a real metrics.json (so scripts/log_experiment.py accepts it) and shifts
half_B.wf1.wf1 by EXPLOOP_FAKE_SHIFTS ("T1a:+0.006,T1b:-0.002", fractions; default +0.003).
"""
import json
import os
import sys
import time
from pathlib import Path


def shift(label):
    for item in os.environ.get("EXPLOOP_FAKE_SHIFTS", "").split(","):
        if ":" in item:
            k, v = item.split(":")
            if k == label:
                return float(v)
    return 0.003


def main():
    cmd = sys.argv[1]
    args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
    if cmd == "sleep":
        time.sleep(float(sys.argv[2]))
        print("slept")
        return
    if cmd == "readout":
        m = json.loads(Path(args["--src"]).read_text())
        m["half_B"]["wf1"]["wf1"] += shift(args["--label"]) + (0.001 if args["--kind"] == "final" else 0.0)
        out = Path(args["--out"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(m))
        print(f"[half_B] WF1_local@0.5 = {100 * m['half_B']['wf1']['wf1']:.2f}")
        return
    if cmd == "ci":
        d = shift(args["--label"])
        res = {"wf1": {"delta": d, "ci_low": d - 0.002, "ci_high": d + 0.002}, "labels": {"a": "base", "b": args["--label"]}}
        Path(args["--out"]).write_text(json.dumps(res))
        print("ci written")
        return
    sys.exit(f"unknown {cmd}")


if __name__ == "__main__":
    main()
