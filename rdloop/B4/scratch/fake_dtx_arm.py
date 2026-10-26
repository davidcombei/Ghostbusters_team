"""F1.1 probe: a fabricated dtx_gate arm dir (B2f's readouts symlinked + a metrics.jsonl with b4 rows, incl. a
levers-off row and a repeated epoch) for scripts/b4_dsp_slices.py --lever dtx_gate.
    python -B rdloop/B4/scratch/fake_dtx_arm.py <out dir>      (from the repo root)"""
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
B2F = REPO / "SLS_setup/exp/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623"


def row(epoch, bona, spoof, n_bona=1000, n_spoof=4000):
    return {"epoch": epoch, "b3": {}, "b4": {"levers": ["dtx_gate"], "aug_by_class": {
        "bona": {"n": n_bona, "counts": bona}, "spoof": {"n": n_spoof, "counts": spoof}}}}


ROWS = [
    {"epoch": 1, "b3": {}},                                         # replaced by the later epoch-1 row
    row(1, {"dtx_g0": 30, "dtx_g1_20": 260, "dtx_g20_40": 8, "dtx_g40p": 2, "wrtc_dtx_zero": 150, "wrtc_dtx_cng": 150},
        {"dtx_g0": 120, "dtx_g1_20": 1060, "dtx_g20_40": 15, "dtx_g40p": 4, "fail_wrtc_dtx": 1}),
    row(2, {"dtx_g0": 25, "dtx_g1_20": 265, "dtx_g20_40": 9, "dtx_g40p": 1},
        {"dtx_g0": 110, "dtx_g1_20": 1075, "dtx_g20_40": 12, "dtx_g40p": 3}),
]


def build(out):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "local_eval").mkdir(parents=True)
    for name in ("epoch_8", "merge_top5"):
        (out / "local_eval" / name).symlink_to(B2F / "local_eval" / name)
    with open(out / "metrics.jsonl", "w", encoding="utf-8") as f:
        for r in ROWS:
            f.write(json.dumps(r) + "\n")
    return out


if __name__ == "__main__":
    print(build(sys.argv[1]))
