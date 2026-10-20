"""Fake scripts/local_eval.py: deterministic per-clip logits from the checkpoint's content."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
src = p.add_mutually_exclusive_group(required=True)
src.add_argument("--model_path")
src.add_argument("--merge_list")
p.add_argument("--names", nargs="*", default=None)
p.add_argument("--out", required=True)
p.add_argument("--force", action="store_true")
p.add_argument("--run_config", default=None)
a, _ = p.parse_known_args()
if a.merge_list:
    paths = [x.strip() for x in Path(a.merge_list).read_text().splitlines() if x.strip() and not x.startswith("#")]
else:
    paths = [a.model_path]
content = Path(paths[0]).read_text()
names = a.names or ["s1", "probe_orig"]
out = Path(a.out)
(out / "scores").mkdir(parents=True, exist_ok=True)
sets = {}
for n in names:
    rows_l, rows_p, pos = [], [], 0
    for i in range(20):
        u = f"wav/{n}_{i:03d}.flac"
        h = int(hashlib.sha256((content + u).encode()).hexdigest()[:8], 16) / 2 ** 32
        z = h * 20 - 10
        pos += z > 0
        rows_l.append(f"{u} {z:.10f}")
        rows_p.append(f"{u} {1 / (1 + 2.718281828 ** (-z)):.10f}")
    (out / "scores" / f"{n}.logits").write_text("\n".join(rows_l) + "\n")
    (out / "scores" / f"{n}.txt").write_text("\n".join(rows_p) + "\n")
    sets[n] = {"n": 20, "f1": pos / 20, "eer": 1 - pos / 20, "recall_bona": 0.5}
m = {"model": paths[0], "full": {"sets": sets}, "half_B": {"wf1": {"wf1": 0.93, "f1_clean": 0.97}, "sets": sets}}
(out / "metrics.json").write_text(json.dumps(m, indent=2))
print(f"Metrics saved to: {out / 'metrics.json'}")
