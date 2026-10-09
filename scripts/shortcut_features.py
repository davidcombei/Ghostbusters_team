"""
Per-file signal statistics for the shortcut audit (roadmap E0.5), train + dev only.

    python scripts/shortcut_features.py --workers 64

Writes data/audit/features_v1.csv, one row per file. Resumable: files already in the CSV are
skipped. The analysis (plots, AUCs, conclusions) lives in notebooks/E0.5_shortcut_audit.ipynb,
which only reads this CSV.

If bonafide and spoof differ in edge silence, level or bandwidth, a model can lean on cues that
noise suppression, VAD and AGC destroy under noise (Mueller et al. 2021, "Speech is Silver,
Silence is Golden"). The eval reads only the first 4.04 s, so leading silence matters most.
"""

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import welch

REPO = Path(__file__).resolve().parents[1]
SR = 16000
CROP = 64600                                # the 4.04 s the model actually sees (pad_audio)
FRAME = 320                                 # 20 ms
COLUMNS = ["path", "split", "src", "lang", "label", "sr", "duration",
           "lead_sil_35", "trail_sil_35", "lead_sil_45", "trail_sil_45",
           "nonspeech_frac", "crop_nonspeech_frac", "rms_db", "rms_active_db", "peak_db",
           "clip_ratio", "dc_offset", "rolloff99_hz", "noise_floor_db"]


def db(x):
    return float(10 * np.log10(max(float(x), 1e-12)))


def frame_db(x):
    n = len(x) // FRAME
    if n == 0:
        return np.array([db(np.mean(x ** 2))])
    return 10 * np.log10(np.mean(x[:n * FRAME].reshape(n, FRAME) ** 2, axis=1) + 1e-12)


def features(item):
    data_root, split, path, label = item
    audio, sr = sf.read(str(data_root / "wav" / split / path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    src, lang = path.split("/")[:2]
    row = {"path": path, "split": split, "src": src, "lang": lang, "label": label, "sr": sr,
           "duration": len(audio) / SR}
    for top_db in (35, 45):
        _, (start, end) = librosa.effects.trim(audio, top_db=top_db, frame_length=640, hop_length=160)
        row[f"lead_sil_{top_db}"] = start / SR
        row[f"trail_sil_{top_db}"] = (len(audio) - end) / SR

    fdb = frame_db(audio)
    active = fdb > fdb.max() - 35
    crop = audio if len(audio) >= CROP else np.tile(audio, CROP // max(len(audio), 1) + 1)
    cdb = frame_db(crop[:CROP])
    _, psd = welch(audio, fs=SR, nperseg=1024)
    cum = np.cumsum(psd) / max(psd.sum(), 1e-20)
    row.update({
        "nonspeech_frac": float(1 - active.mean()),
        "crop_nonspeech_frac": float(np.mean(cdb <= fdb.max() - 35)),
        "rms_db": db(np.mean(audio ** 2)),
        "rms_active_db": float(10 * np.log10(np.mean(10 ** (fdb[active] / 10)))),
        "peak_db": float(20 * np.log10(max(np.abs(audio).max(), 1e-6))),
        "clip_ratio": float(np.mean(np.abs(audio) >= 0.999)),
        "dc_offset": float(audio.mean()),
        "rolloff99_hz": float(np.searchsorted(cum, 0.99) * SR / 1024),
        "noise_floor_db": float(np.percentile(fdb, 10)),
    })
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=Path, default=Path("/mnt/paml-research/RTCFake/data"))
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--out", type=Path, default=REPO / "data" / "audit" / "features_v1.csv")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.out.is_file():
        with open(args.out, newline="", encoding="utf-8") as f:
            done = {(r["split"], r["path"]) for r in csv.DictReader(f)}
    items = []
    for split in args.splits:
        with open(args.data_root / f"{split}_label.txt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    path, label = line.split()
                    if (split, path) not in done:
                        items.append((args.data_root, split, path, label))
    print(f"{len(done)} files already done, {len(items)} to go")

    new = not args.out.is_file() or args.out.stat().st_size == 0
    with open(args.out, "a", newline="", encoding="utf-8") as f, Pool(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        for i, row in enumerate(pool.imap_unordered(features, items, chunksize=32), 1):
            writer.writerow(row)
            if i % 5000 == 0:
                f.flush()
                print(f"  {i}/{len(items)}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
