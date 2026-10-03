"""
Build the noise / RIR pools that SLS_setup/utils/rtc_augment.py draws from.

One folder per source under --out (default <repo>/data/augmentation):

    downloads/                          archives fetched from xiph.org (kept for reruns)
    rnnoise/{office,coffee}/*.wav       S02 / S03  RNNoise contributions, labelled uploads
    esc50/{rain,footsteps,keyboard}/    S05-S07    ESC-50 clips of those three categories
    rirs/*.f32                          reverb     RNNoise measured_rirs-v3 (raw float32, 48 kHz)
    musan/{noise,music}/                optional   MUSAN without speech/

Every first-level folder under rnnoise/ and esc50/ is one augmentation stage,
so train with

    --use_rtc_aug --aug_noise_dirs $A/rnnoise $A/esc50 --aug_rir_dirs $A/rirs \
    [--musan --musan_dir $A/musan]

ESC-50 and MUSAN are copied from local copies (--esc50_dir, --musan_dir), which
are only ever read. Each step drops a .complete marker when done and is skipped
on the next run; delete the step's folder to rebuild it.

    python scripts/prepare_rtc_aug_data.py \
        --esc50_dir /mnt/paml-datasets/ESC-50-master --musan_dir /mnt/paml-datasets/musan
"""

import argparse
import csv
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
RNNOISE_URL = "https://media.xiph.org/rnnoise/rnnoise_contributions.tar.gz"
RIRS_URL = "https://media.xiph.org/rnnoise/data/measured_rirs-v3.tar.gz"
RNNOISE_SR = 48000                          # contributions are headerless int16 LE, 48 kHz, mono
RNNOISE_LABELS = ("office", "coffee")       # <timestamp>-<label>.raw; "other"/"train" are dropped
ESC50_STAGES = {"rain": "rain", "footsteps": "footsteps", "keyboard_typing": "keyboard"}
MUSAN_PARTS = ("noise", "music")            # never speech/: no extra speech data allowed
AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".f32")
STEPS = ("rnnoise", "rirs", "esc50", "musan")
MARKER = ".complete"


def download(url, dest):
    """Fetch `url` to `dest` through a .part file, unless it is already there."""
    if dest.is_file():
        print(f" {dest.name}: already downloaded")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f" downloading {url}")
    with urllib.request.urlopen(url) as resp, open(part, "wb") as fout:
        total = int(resp.headers.get("Content-Length", 0))
        done, next_report = 0, 0.0
        while chunk := resp.read(1 << 20):
            fout.write(chunk)
            done += len(chunk)
            if total and done / total >= next_report:
                print(f"   {done / 2**30:.2f} / {total / 2**30:.2f} GiB", flush=True)
                next_report += 0.1
    part.rename(dest)
    return dest


def is_complete(directory):
    if (directory / MARKER).is_file():
        print(f" {directory}: already built, skipping")
        return True
    return False


def mark_complete(directory):
    (directory / MARKER).touch()


def prepare_rnnoise(out):
    """Keep the office / coffee uploads and write each one as a 48 kHz PCM16 wav."""
    dest = out / "rnnoise"
    if is_complete(dest):
        return
    archive = download(RNNOISE_URL, out / "downloads" / Path(RNNOISE_URL).name)
    for label in RNNOISE_LABELS:
        (dest / label).mkdir(parents=True, exist_ok=True)

    counts = dict.fromkeys(RNNOISE_LABELS, 0)
    with tarfile.open(archive, "r|gz") as tar:              # streamed: one pass, no extraction
        for member in tar:
            name = Path(member.name).name
            stem, _, label = name.removesuffix(".raw").rpartition("-")
            if not member.isfile() or not name.endswith(".raw") or label not in counts:
                continue
            data = tar.extractfile(member).read()
            pcm = np.frombuffer(data[:len(data) // 2 * 2], dtype="<i2")
            if pcm.size == 0:
                continue
            sf.write(str(dest / label / f"{stem}.wav"), pcm, RNNOISE_SR, subtype="PCM_16")
            counts[label] += 1
    print(f" rnnoise: {counts}")
    mark_complete(dest)


def prepare_rirs(out):
    """Extract the measured RIRs as shipped (headerless float32, 48 kHz)."""
    dest = out / "rirs"
    if is_complete(dest):
        return
    archive = download(RIRS_URL, out / "downloads" / Path(RIRS_URL).name)
    dest.mkdir(parents=True, exist_ok=True)

    count = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".f32"):
                (dest / Path(member.name).name).write_bytes(tar.extractfile(member).read())
                count += 1
    print(f" rirs: {count}")
    mark_complete(dest)


def prepare_esc50(out, src):
    """Copy the rain / footsteps / keyboard_typing clips, one folder per stage."""
    dest = out / "esc50"
    if is_complete(dest):
        return
    src = Path(src)
    counts = dict.fromkeys(ESC50_STAGES.values(), 0)
    with open(src / "meta" / "esc50.csv", newline="") as fin:
        for row in csv.DictReader(fin):
            stage = ESC50_STAGES.get(row["category"])
            if stage is None:
                continue
            (dest / stage).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "audio" / row["filename"], dest / stage / row["filename"])
            counts[stage] += 1
    print(f" esc50: {counts}")
    mark_complete(dest)


def prepare_musan(out, src):
    """Copy MUSAN's noise/ and music/ trees; speech/ is left out on purpose."""
    dest = out / "musan"
    if is_complete(dest):
        return
    src = Path(src)
    for part in MUSAN_PARTS:
        print(f" copying {src / part}", flush=True)
        shutil.copytree(src / part, dest / part, dirs_exist_ok=True)
    mark_complete(dest)


def summarize(out):
    """Audio files per stage, i.e. what RTCAugmenter will find."""
    print(f"\n pools under {out}:")
    for source in ("rnnoise", "esc50"):
        for stage in sorted(p for p in (out / source).glob("*") if p.is_dir()):
            n = sum(1 for p in stage.iterdir() if p.suffix.lower() in AUDIO_EXTS)
            print(f"   {source}/{stage.name}: {n}")
    for source in ("rirs", "musan"):
        root = out / source
        if root.is_dir():
            n = sum(1 for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
            print(f"   {source}: {n}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "augmentation")
    parser.add_argument("--esc50_dir", type=str, default=None,
                        help="local ESC-50 root (holds audio/ and meta/esc50.csv); read only")
    parser.add_argument("--musan_dir", type=str, default=None,
                        help="local MUSAN root (holds noise/ and music/); read only")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    args = parser.parse_args()

    out = args.out.resolve()
    for step in args.steps:
        print(f"[{step}]", flush=True)
        if step == "rnnoise":
            prepare_rnnoise(out)
        elif step == "rirs":
            prepare_rirs(out)
        elif step == "esc50":
            if args.esc50_dir:
                prepare_esc50(out, args.esc50_dir)
            else:
                print(" no --esc50_dir given, skipping")
        elif step == "musan":
            if args.musan_dir:
                prepare_musan(out, args.musan_dir)
            else:
                print(" no --musan_dir given, skipping")
    summarize(out)


if __name__ == "__main__":
    main()
