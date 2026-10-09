"""
Probe report: how much does P(spoof) move when only non-speech cues change?

    python scripts/probe_report.py <local_eval out dir>/scores [--json out.json]

Reads the probe_*.txt score files written by scripts/local_eval.py, pairs every probe clip
with its probe_orig twin and reports, per probe x label: mean dP(spoof), decision flip rate at
0.5 and predicted-spoof rate. Headline: `level_flip_bona` = mean bonafide flip rate over the
level probes (the level shortcut; 0.30 at +10 dB before level augmentation).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROBE_ROOT = REPO / "data" / "dev_noisy_sim" / "probes_v1"
LEVEL_PROBES = ("gain_up", "gain_down", "gain_m20", "gain_m6", "gain_p6", "gain_p10_limit",
                "agc_dynaudnorm", "loudnorm")


def load(scores_dir, probe):
    path = Path(scores_dir) / f"probe_{probe}.txt"
    if not path.is_file():
        return None
    scores = dict(line.split() for line in open(path, encoding="utf-8") if line.strip())
    with open(PROBE_ROOT / probe / "meta.csv", newline="", encoding="utf-8") as f:
        return {r["source_utt"]: (r["label"], float(scores[r["path"]])) for r in csv.DictReader(f)}


def report(scores_dir):
    orig = load(scores_dir, "orig")
    if orig is None:
        raise FileNotFoundError(f"{scores_dir}/probe_orig.txt (score probe_orig with local_eval.py)")
    probes = sorted(p.name for p in PROBE_ROOT.iterdir() if p.is_dir() and p.name != "orig")
    rows = []
    for probe in probes:
        cur = load(scores_dir, probe)
        if cur is None:
            continue
        for label in ("bonafide", "spoof"):
            pairs = [(orig[u][1], p) for u, (lab, p) in cur.items() if lab == label]
            p0, p1 = np.array(pairs).T
            rows.append({"probe": probe, "label": label, "n": len(pairs),
                         "mean_dP": float(np.mean(p1 - p0)),
                         "flip_rate": float(np.mean((p1 >= 0.5) != (p0 >= 0.5))),
                         "pred_spoof": float(np.mean(p1 >= 0.5)), "pred_spoof_orig": float(np.mean(p0 >= 0.5))})
    level = [r["flip_rate"] for r in rows if r["probe"] in LEVEL_PROBES and r["label"] == "bonafide"]
    level_spoof = [r["flip_rate"] for r in rows if r["probe"] in LEVEL_PROBES and r["label"] == "spoof"]
    summary = {"level_flip_bona": float(np.mean(level)) if level else None,
               "level_flip_spoof": float(np.mean(level_spoof)) if level_spoof else None,
               "level_flip_bona_max": float(np.max(level)) if level else None}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores_dir")
    parser.add_argument("--json", default=None, help="default: <scores_dir>/../probe_report.json")
    args = parser.parse_args()
    rows, summary = report(args.scores_dir)
    print(f"{'probe':<16}{'label':<10}{'n':>5}{'mean dP':>9}{'flip':>7}{'P>=.5':>7}{'orig':>7}")
    for r in rows:
        mark = "  <- level" if r["probe"] in LEVEL_PROBES else ""
        print(f"{r['probe']:<16}{r['label']:<10}{r['n']:>5}{r['mean_dP']:>+9.3f}{r['flip_rate']:>7.3f}"
              f"{r['pred_spoof']:>7.3f}{r['pred_spoof_orig']:>7.3f}{mark}")
    print(f"\nlevel_flip_bona = {summary['level_flip_bona']:.3f} (max {summary['level_flip_bona_max']:.3f})   "
          f"level_flip_spoof = {summary['level_flip_spoof']:.3f}")
    out = Path(args.json) if args.json else Path(args.scores_dir).parent / "probe_report.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
