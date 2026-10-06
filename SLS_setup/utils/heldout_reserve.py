"""
Components reserved for the local `sim-heldout` / `sim-matched` proxy sets.

sim-heldout only measures generalisation to unseen platforms if training never sees its
noise classes, DSP and codecs. Everything listed here is therefore off-limits for training
augmentation, now and in later phases; RTCAugmenter calls check_pools() on start-up.

    reserved for sim-heldout   codecs   G.722, G.726, GSM
                               filters  ffmpeg anlmdn, acompressor
                               noise    the ESC-50 classes in RESERVED_ESC50_CLASSES
    held out for sim-matched   files listed in data/augmentation_eval/holdout_files.txt
                               (written by scripts/split_aug_pools.py)

Pools under data/augmentation_eval/ or with reserved ESC-50 classes raise. Pools that merely
contain held-out sim-matched files (the original data/augmentation/ tree, which older runs
used) only warn, so old command lines keep working.
"""

import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_POOL_DIRNAME = "augmentation_eval"
HOLDOUT_LIST = REPO_ROOT / "data" / EVAL_POOL_DIRNAME / "holdout_files.txt"

RESERVED_CODECS = {"g722", "adpcm_g722", "g726", "adpcm_g726", "libgsm", "gsm"}
RESERVED_FILTERS = {"anlmdn", "acompressor"}

# ESC-50 target id -> class. Indoor / RTC-plausible, non-human; the whole "human, non-speech"
# category (targets 20-29) is excluded for rule safety, and the training classes (rain 10,
# footsteps 25, keyboard_typing 32) are of course not reserved.
RESERVED_ESC50_CLASSES = {
    16: "wind",
    17: "pouring_water",
    30: "door_wood_knock",
    31: "mouse_click",
    33: "door_wood_creaks",
    34: "can_opening",
    35: "washing_machine",
    36: "vacuum_cleaner",
    37: "clock_alarm",
    38: "clock_tick",
    39: "glass_breaking",
}


def esc50_target(path):
    """Target id from an ESC-50 file name (<fold>-<src>-<take>-<target>.wav), else None."""
    parts = Path(path).stem.split("-")
    if len(parts) == 4 and parts[0].isdigit() and parts[3].isdigit():
        return int(parts[3])
    return None


def _holdout_names():
    if not HOLDOUT_LIST.is_file():
        return set()
    with open(HOLDOUT_LIST, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def check_pools(paths=(), codecs=(), filters=()):
    """Raise on reserved components; warn on overlap with the sim-matched held-out files."""
    reserved_codecs = sorted(set(map(str.lower, codecs)) & RESERVED_CODECS)
    reserved_filters = sorted(set(map(str.lower, filters)) & RESERVED_FILTERS)
    if reserved_codecs or reserved_filters:
        raise ValueError(f"reserved for sim-heldout, not allowed in training: "
                         f"codecs={reserved_codecs} filters={reserved_filters}")

    holdout = _holdout_names()
    overlap = 0
    for path in paths:
        path = Path(path)
        if EVAL_POOL_DIRNAME in path.parts:
            raise ValueError(f"training pool file under {EVAL_POOL_DIRNAME}/: {path}")
        if esc50_target(path) in RESERVED_ESC50_CLASSES:
            raise ValueError(f"ESC-50 class reserved for sim-heldout in a training pool: {path}")
        if path.name in holdout:
            overlap += 1
    if overlap:
        warnings.warn(f"{overlap} training pool files are held out for sim-matched; use the "
                      f"data/augmentation_train/ pools for new runs (sim-matched scores of this "
                      f"run will be optimistic)", stacklevel=2)
    return overlap
