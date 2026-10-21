"""
Build the RTC-SDD augmentation pools under data/augm/.

The challenge eval set carries a noisy subset whose scenarios come from three
upstream sources:

  S02 office, S03 coffee   RNNoise background noise (Xiph)
                           https://media.xiph.org/rnnoise/data/
  S04 echo                 CLAD (algorithmic, no assets needed)
                           https://github.com/CLAD23/CLAD
  S05 rain, S06 footsteps,
  S07 keyboard             ESC-50
                           https://github.com/karolpiczak/ESC-50

plus the measured RIRs that ship with the RNNoise training data
(data/measured_rirs-v3.tar.gz) for the reverberant condition.

Everything is written as 16 kHz mono float32 WAV, which is what
utils/rtc_augment.py expects:

  data/augm/noise/office/*.wav       data/augm/noise/rain/*.wav
  data/augm/noise/coffee/*.wav       data/augm/noise/footsteps/*.wav
  data/augm/measured_rirs/*.f32      data/augm/noise/keyboard/*.wav

Note on office vs coffee: background_noise_v2.sw is one unlabelled
concatenation of many recordings, so the split between the two pools is by
byte offset in that file, not by a true scene label. The scenarios are told
apart at train time by their SNR ranges (see rtc_augment.DEFAULT_SNR_DB).

Usage:
    python scripts/prepare_rtc_aug_data.py --out_dir data/augm
    python scripts/prepare_rtc_aug_data.py --out_dir data/augm --only esc50
"""

import argparse
import csv
import io
import os
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000

ESC50_RAW = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
ESC50_CLASSES = {          # ESC-50 category -> our scenario tag
    "rain": "rain",
    "footsteps": "footsteps",
    "keyboard_typing": "keyboard",
}

RNNOISE_BASE = "https://media.xiph.org/rnnoise/data"
RNNOISE_NOISE = "background_noise_v2.sw"   # raw s16le, 48 kHz, mono
RNNOISE_SR = 48000
RNNOISE_TAGS = ("office", "coffee")

RIR_TARBALL = "data/measured_rirs-v3.tar.gz"


# --------------------------------------------------------------------------- #
def fetch(url, byte_range=None, retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": "rtc-sdd-prep/1"})
    if byte_range is not None:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            print(f"  retry {attempt + 1}/{retries}: {exc}", file=sys.stderr)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def content_length(url):
    req = urllib.request.Request(url, method="HEAD",
                                headers={"User-Agent": "rtc-sdd-prep/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def to_16k_mono(audio, sr):
    """audio: (n,) or (n, ch) float32 -> (n,) float32 at 16 kHz."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        audio = resample_poly(audio, SR // g, sr // g).astype(np.float32)
    return audio


def write_wav(path, audio):
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 1.0:                      # keep the pools inside [-1, 1]
        audio = audio / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SR, subtype="FLOAT")


# --------------------------------------------------------------------------- #
def prepare_esc50(out_dir, force=False):
    """Download only the three ESC-50 classes we need (40 clips each)."""
    print(f"[esc50] metadata from {ESC50_RAW}/meta/esc50.csv")
    meta = fetch(f"{ESC50_RAW}/meta/esc50.csv").decode("utf-8")
    rows = [r for r in csv.DictReader(io.StringIO(meta))
            if r["category"] in ESC50_CLASSES]
    print(f"[esc50] {len(rows)} clips over {sorted(ESC50_CLASSES)}")

    written = 0
    for i, row in enumerate(rows, start=1):
        tag = ESC50_CLASSES[row["category"]]
        dst = out_dir / "noise" / tag / row["filename"]
        if dst.exists() and not force:
            continue
        raw = fetch(f"{ESC50_RAW}/audio/{row['filename']}")
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        write_wav(dst, to_16k_mono(audio, sr))
        written += 1
        if i % 20 == 0 or i == len(rows):
            print(f"[esc50] {i}/{len(rows)}")
    print(f"[esc50] wrote {written} clip(s)")


def prepare_rnnoise(out_dir, n_chunks=20, chunk_mb=12, seg_seconds=10.0, force=False):
    """
    Sample `n_chunks` evenly spread chunks out of the RNNoise background-noise
    blob with HTTP range requests and slice them into fixed-length segments.
    Reading the whole 1.3 GB file is unnecessary; spreading the chunks keeps the
    pool varied because the blob is a concatenation of many recordings.
    """
    url = f"{RNNOISE_BASE}/{RNNOISE_NOISE}"
    total = content_length(url)
    chunk_bytes = int(chunk_mb * 1024 * 1024) & ~1          # even: s16 frames
    seg_len = int(seg_seconds * SR)
    print(f"[rnnoise] {RNNOISE_NOISE}: {total / 1e9:.2f} GB "
          f"({total / 2 / RNNOISE_SR / 3600:.2f} h) -> "
          f"{n_chunks} x {chunk_mb} MB")

    existing = {tag: sorted((out_dir / "noise" / tag).glob("rnnoise_*.wav"))
                for tag in RNNOISE_TAGS}
    if all(existing[tag] for tag in RNNOISE_TAGS) and not force:
        have = {t: len(v) for t, v in existing.items()}
        print(f"[rnnoise] pools already present {have}; use --force to rebuild")
        return

    span = max(1, (total - chunk_bytes) // max(1, n_chunks - 1))
    counts = dict.fromkeys(RNNOISE_TAGS, 0)
    for k in range(n_chunks):
        start = min(k * span, total - chunk_bytes) & ~1
        raw = fetch(url, byte_range=(start, start + chunk_bytes - 1))
        pcm = np.frombuffer(raw[: len(raw) & ~1], dtype="<i2").astype(np.float32) / 32768.0
        audio = to_16k_mono(pcm, RNNOISE_SR)
        # chunks alternate between the two pools so both see the whole file
        tag = RNNOISE_TAGS[k % len(RNNOISE_TAGS)]
        n_seg = len(audio) // seg_len
        for s in range(n_seg):
            seg = audio[s * seg_len:(s + 1) * seg_len]
            if float(np.sqrt((seg ** 2).mean())) < 1e-4:     # drop silent slices
                continue
            write_wav(out_dir / "noise" / tag / f"rnnoise_{k:03d}_{s:03d}.wav", seg)
            counts[tag] += 1
        print(f"[rnnoise] chunk {k + 1}/{n_chunks} -> {tag} "
              f"({counts[tag]} segments so far)")
    total_seg = sum(counts.values())
    print(f"[rnnoise] wrote {total_seg} segments "
          f"({total_seg * seg_seconds / 60:.1f} min) {counts}")


def prepare_rirs(out_dir, repo_root, force=False):
    """Extract the measured RIRs shipped with the challenge data, if needed."""
    dst = out_dir / "measured_rirs"
    if dst.is_dir() and any(dst.iterdir()) and not force:
        print(f"[rirs] {dst} already holds {len(list(dst.glob('*.f32')))} RIR(s)")
        return
    tarball = repo_root / RIR_TARBALL
    if not tarball.exists():
        print(f"[rirs] {tarball} not found; skipping", file=sys.stderr)
        return
    print(f"[rirs] extracting {tarball} -> {out_dir}")
    with tarfile.open(tarball) as tf:
        tf.extractall(out_dir)
    print(f"[rirs] {len(list(dst.glob('*.f32')))} RIR(s)")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="data/augm")
    p.add_argument("--repo_root", default=".",
                   help="where data/measured_rirs-v3.tar.gz lives")
    p.add_argument("--only", nargs="*", default=["esc50", "rnnoise", "rirs"],
                   choices=["esc50", "rnnoise", "rirs"])
    p.add_argument("--n_chunks", type=int, default=20,
                   help="[rnnoise] number of range-request chunks")
    p.add_argument("--chunk_mb", type=int, default=12, help="[rnnoise] MB per chunk")
    p.add_argument("--seg_seconds", type=float, default=10.0,
                   help="[rnnoise] length of one noise segment")
    p.add_argument("--force", action="store_true", help="redo work already on disk")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_dir}")

    if "rirs" in args.only:
        prepare_rirs(out_dir, Path(args.repo_root).resolve(), args.force)
    if "esc50" in args.only:
        prepare_esc50(out_dir, args.force)
    if "rnnoise" in args.only:
        prepare_rnnoise(out_dir, args.n_chunks, args.chunk_mb,
                        args.seg_seconds, args.force)

    print("\nPools:")
    for d in sorted((out_dir / "noise").glob("*")):
        if d.is_dir():
            wavs = list(d.glob("*.wav"))
            dur = sum(sf.info(str(w)).duration for w in wavs)
            print(f"  {d.relative_to(out_dir)}: {len(wavs)} files, {dur / 60:.1f} min")
    rirs = list((out_dir / "measured_rirs").glob("*.f32"))
    print(f"  measured_rirs: {len(rirs)} RIRs")


if __name__ == "__main__":
    main()
