"""
Extended training reserve for the B3 wave.

utils/heldout_reserve.py reserves what `sim_heldout_v1` uses (G.722 / G.726 / GSM, anlmdn,
acompressor, 11 ESC-50 classes, the data/augmentation_eval/ pools). The B3 wave adds the recipe of
the frozen *unseen* set (`sim_unseen_v1`, scripts/b3_build_unseen_set.py) and the other
`utils/sim_ops.py` stages, so that no training run of the wave can learn them:

    codecs    speex, G.711 mu-law / A-law            (unseen set)
    filters   afftdn, dynaudnorm                     (sim_heldout_v1 via sim_ops, never enforced before)
              compand, agate, crystalizer            (unseen set; speechnorm kept for newer ffmpeg)
              vad_gate, packet_loss, bandlimit       (sim_ops stages used by heldout / unseen)
    dirs      data/augmentation/musan/music          (unseen-set noise)
              data/augmentation_eval                 (heldout / matched pools; already reserved)

RTCAugmenterB3 calls `assert_training_ops` on start-up with the codecs, filters and pool
directories it is about to use; anything reserved raises before the first training step.
"""

from pathlib import Path

from utils.heldout_reserve import (  # noqa: F401  (re-exported for callers)
    REPO_ROOT,
    RESERVED_CODECS,
    RESERVED_ESC50_CLASSES,
    RESERVED_FILTERS,
    check_pools,
)

RESERVED_CODECS_B3 = frozenset(RESERVED_CODECS) | {
    "libspeex", "speex", "pcm_mulaw", "pcm_alaw", "g711", "g711_mulaw", "g711_alaw",
}
RESERVED_FILTERS_B3 = frozenset(RESERVED_FILTERS) | {
    "afftdn", "dynaudnorm", "vad_gate", "packet_loss", "bandlimit", "resample8k",     # sim_heldout_v1 (sim_ops)
    "compand", "agate", "crystalizer", "speechnorm",                                  # sim_unseen_v1
}
RESERVED_DIRS_B3 = (
    "data/augmentation/musan/music",
    "data/augmentation_eval",
)


def _resolve(path):
    p = Path(path)
    if not p.is_absolute():
        # relative paths in launchers are given from SLS_setup/ ("../data/...") or the repo root
        candidates = [Path.cwd() / p, REPO_ROOT / p]
        for c in candidates:
            if c.exists():
                p = c
                break
        else:
            p = candidates[0]
    return p.resolve()


def reserved_dir_hits(dirs):
    """The reserved directory each of `dirs` falls under (or is), as (dir, reserved) pairs."""
    hits = []
    reserved = [(REPO_ROOT / r).resolve() for r in RESERVED_DIRS_B3]
    for d in dirs:
        rd = _resolve(d)
        for r in reserved:
            if rd == r or r in rd.parents:
                hits.append((str(d), str(r.relative_to(REPO_ROOT))))
    return hits


def assert_training_ops(codecs=(), filters=(), dirs=(), paths=()):
    """
    Raise ValueError when training is about to use anything reserved for the proxy sets.
    `paths` go through the original check_pools (eval pools, reserved ESC-50 classes, held-out
    files); `codecs` / `filters` are encoder / ffmpeg-filter names; `dirs` are pool roots.
    """
    check_pools(paths)
    bad_codecs = sorted({str(c).lower() for c in codecs} & RESERVED_CODECS_B3)
    bad_filters = sorted({str(f).lower() for f in filters} & RESERVED_FILTERS_B3)
    bad_dirs = reserved_dir_hits(dirs)
    if bad_codecs or bad_filters or bad_dirs:
        raise ValueError(
            "reserved for the local proxy sets (sim_heldout_v1 / sim_unseen_v1), not allowed in "
            f"training: codecs={bad_codecs} filters={bad_filters} dirs={bad_dirs}")
    return True
