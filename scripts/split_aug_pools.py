"""
Split the augmentation pools into training pools and the pools of the local proxy sets.

The original tree (data/augmentation/, built by prepare_rtc_aug_data.py and used by every
earlier run) is only read, never changed, so old runs stay reproducible. Two new trees:

    data/augmentation_train/                 for new training runs (hardlinks, no extra disk)
        rnnoise/{office,coffee}/             ~85% of the files (CRC32 hash split)
        esc50/{rain,footsteps,keyboard}/     ESC-50 folds 1-4
        rirs/                                every room except "office"
    data/augmentation_eval/                  never used in training (utils/heldout_reserve.py)
        rnnoise/{office,coffee}/             the other ~15%            -> sim-matched
        esc50/{rain,footsteps,keyboard}/     ESC-50 fold 5             -> sim-matched
        rirs/                                the "office" room (8 RIRs) -> sim-matched / sim-echo
        esc50_reserved/<class>/              RESERVED_ESC50_CLASSES    -> sim-heldout
        holdout_files.txt                    basenames held out from the training pools
        MANIFEST.json

MUSAN is not split (no proxy set uses it); keep pointing --musan_dir at data/augmentation/musan.
New training runs:

    --use_rtc_aug --aug_noise_dirs data/augmentation_train/rnnoise data/augmentation_train/esc50 \
        --aug_rir_dirs data/augmentation_train/rirs

    python scripts/split_aug_pools.py --esc50_dir /mnt/paml-datasets/ESC-50-master
"""

import argparse
import csv
import json
import os
import shutil
import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "SLS_setup"))

from utils.heldout_reserve import RESERVED_ESC50_CLASSES  # noqa: E402

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".f32")
RNNOISE_HOLDOUT_PCT = 15
ESC50_HOLDOUT_FOLD = "5"
RIR_HOLDOUT_ROOM = "office"
MARKER = ".complete"


def place(src, dest):
    """Hardlink (same filesystem, no extra disk), falling back to a copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def audio_files(directory):
    return sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def is_heldout(source, path):
    if source == "rnnoise":
        return zlib.crc32(path.name.encode()) % 100 < RNNOISE_HOLDOUT_PCT
    if source == "esc50":
        return path.name.split("-")[0] == ESC50_HOLDOUT_FOLD
    if source == "rirs":
        return path.name.split("_")[0] == RIR_HOLDOUT_ROOM
    raise ValueError(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=REPO / "data" / "augmentation")
    parser.add_argument("--train_out", type=Path, default=REPO / "data" / "augmentation_train")
    parser.add_argument("--eval_out", type=Path, default=REPO / "data" / "augmentation_eval")
    parser.add_argument("--esc50_dir", type=Path, required=True,
                        help="local ESC-50 root (audio/ and meta/esc50.csv); read only")
    args = parser.parse_args()

    if (args.eval_out / MARKER).is_file() and (args.train_out / MARKER).is_file():
        print(f"{args.train_out} and {args.eval_out} already built; delete them to rebuild")
        return

    counts, holdout = {}, []
    for source, stages in (("rnnoise", ("office", "coffee")),
                           ("esc50", ("rain", "footsteps", "keyboard")),
                           ("rirs", (None,))):
        for stage in stages:
            src_dir = args.src / source / stage if stage else args.src / source
            rel = Path(source) / stage if stage else Path(source)
            n_train = n_eval = 0
            for path in audio_files(src_dir):
                if is_heldout(source, path):
                    place(path, args.eval_out / rel / path.name)
                    holdout.append(path.name)
                    n_eval += 1
                else:
                    place(path, args.train_out / rel / path.name)
                    n_train += 1
            counts[str(rel)] = {"train": n_train, "eval": n_eval}
            print(f" {str(rel):<20} train {n_train:>5}   eval {n_eval:>4}")

    with open(args.esc50_dir / "meta" / "esc50.csv", newline="") as f:
        for row in csv.DictReader(f):
            cls = RESERVED_ESC50_CLASSES.get(int(row["target"]))
            if cls is not None:
                place(args.esc50_dir / "audio" / row["filename"],
                      args.eval_out / "esc50_reserved" / cls / row["filename"])
                counts.setdefault(f"esc50_reserved/{cls}", {"eval": 0})["eval"] += 1
    n_reserved = sum(v["eval"] for k, v in counts.items() if k.startswith("esc50_reserved"))
    print(f" esc50_reserved       eval {n_reserved:>4} ({len(RESERVED_ESC50_CLASSES)} classes)")

    (args.eval_out / "holdout_files.txt").write_text("\n".join(sorted(holdout)) + "\n", encoding="utf-8")
    manifest = {
        "src": str(args.src), "esc50_dir": str(args.esc50_dir),
        "rnnoise_holdout_pct": RNNOISE_HOLDOUT_PCT, "esc50_holdout_fold": ESC50_HOLDOUT_FOLD,
        "rir_holdout_room": RIR_HOLDOUT_ROOM, "reserved_esc50_classes": RESERVED_ESC50_CLASSES,
        "counts": counts,
    }
    (args.eval_out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for out in (args.train_out, args.eval_out):
        (out / MARKER).touch()
    print(f"\nwrote {args.train_out} and {args.eval_out}")


if __name__ == "__main__":
    main()
