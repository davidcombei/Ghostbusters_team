"""
Probe sets for the shortcut audit (roadmap E0.5, part 2): does P(spoof) depend on silence,
level or non-speech content rather than on the speech itself?

    python scripts/build_probe_sets.py
    cd SLS_setup && python ../scripts/local_eval.py --model_path <ckpt> ... \
        --names probe_orig probe_sil_only probe_noise_only probe_trimmed probe_padded \
                probe_gain_up probe_gain_down --out <dir>

Takes 1000 dev-online clips (stratified by label, seeded) and writes, under
data/dev_noisy_sim/probes_v1/<probe>/ (same layout as the sim sets):

    orig        the clip as is (FLAC round trip, the reference for the deltas)
    sil_only    speech region (librosa trim, top_db 35) zeroed, edge silence kept
    noise_only  the whole clip replaced by a held-out noise at the clip's RMS
    trimmed     leading / trailing silence removed
    padded      1 s of digital silence added at both ends
    gain_up     +10 dB (hard-limited at 0.99)
    gain_down   -10 dB
  level probes (added for the level-augmentation screen; built only if missing):
    gain_m20        -20 dB
    gain_m6         -6 dB
    gain_p6         +6 dB (hard-limited at 0.99)
    gain_p10_limit  +10 dB through a soft look-ahead limiter (loud, but not clipped)
    agc_dynaudnorm  ffmpeg dynaudnorm (an AGC not used in training)
    loudnorm        ffmpeg loudnorm I=-16 LUFS

notebooks/E0.5_shortcut_audit.ipynb compares each probe with `orig` per source clip.
"""

import argparse
import csv
import sys
import zlib
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "SLS_setup"))

from utils.level_augment import limit  # noqa: E402
from utils.rtc_augment import finalize, list_audio, read_window  # noqa: E402
from utils.sim_ops import ffmpeg_filter  # noqa: E402

SR = 16000
PROBES = ("orig", "sil_only", "noise_only", "trimmed", "padded", "gain_up", "gain_down",
          "gain_m20", "gain_m6", "gain_p6", "gain_p10_limit", "agc_dynaudnorm", "loudnorm")
LEVEL_PROBES = ("gain_up", "gain_down", "gain_m20", "gain_m6", "gain_p6", "gain_p10_limit",
                "agc_dynaudnorm", "loudnorm")


def make_probe(probe, audio, rng, noise_files):
    _, (start, end) = librosa.effects.trim(audio, top_db=35, frame_length=640, hop_length=160)
    if probe == "orig":
        return audio
    if probe == "sil_only":
        out = audio.copy()
        out[start:end] = 0.0
        return out
    if probe == "noise_only":
        noise = read_window(noise_files[int(rng.integers(len(noise_files)))], len(audio), SR, rng)
        rms_a, rms_n = np.sqrt(np.mean(audio ** 2)), np.sqrt(np.mean(noise ** 2))
        return noise * np.float32(rms_a / max(rms_n, 1e-8))
    if probe == "trimmed":
        return audio[start:end] if end - start > SR // 10 else audio
    if probe == "padded":
        pad = np.zeros(SR, dtype=np.float32)
        return np.concatenate([pad, audio, pad])
    if probe == "gain_up":
        return np.clip(audio * np.float32(10 ** 0.5), -0.99, 0.99)
    if probe == "gain_down":
        return audio * np.float32(10 ** -0.5)
    if probe == "gain_m20":
        return audio * np.float32(0.1)
    if probe == "gain_m6":
        return audio * np.float32(10 ** -0.3)
    if probe == "gain_p6":
        return np.clip(audio * np.float32(10 ** 0.3), -0.99, 0.99)
    if probe == "gain_p10_limit":
        return limit(audio * np.float32(10 ** 0.5), SR, rng, ceiling_db=(-1.0, -1.0), release_ms=(60.0, 60.0))[0]
    if probe == "agc_dynaudnorm":
        return finalize(ffmpeg_filter(audio, SR, "dynaudnorm=f=150:g=15:m=20"), audio)
    if probe == "loudnorm":
        return finalize(ffmpeg_filter(audio, SR, "loudnorm=I=-16:TP=-1:LRA=11"), audio)
    raise ValueError(probe)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev_root", type=Path, default=Path("/mnt/paml-research/RTCFake/data"))
    parser.add_argument("--noise_dir", type=Path, default=REPO / "data" / "augmentation_eval" / "esc50_reserved")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "dev_noisy_sim" / "probes_v1")
    args = parser.parse_args()

    rows = [line.split() for line in open(args.dev_root / "dev_label.txt", encoding="utf-8")
            if line.startswith("online/")]
    rng = np.random.default_rng([args.seed, zlib.crc32(b"probes")])
    chosen = []
    for label in ("bonafide", "spoof"):
        group = [r for r in rows if r[1] == label]
        k = int(round(args.n * len(group) / len(rows)))
        chosen += [group[i] for i in sorted(rng.choice(len(group), size=k, replace=False))]
    noise_files = [p for d in sorted(args.noise_dir.iterdir()) if d.is_dir() for p in list_audio(d)]

    for probe in PROBES:
        out_dir = args.out / probe
        if (out_dir / "meta.csv").is_file():
            print(f"{probe}: exists, kept as is")
            continue
        (out_dir / "wav").mkdir(parents=True, exist_ok=True)
        metas = []
        for utt, label in chosen:
            frng = np.random.default_rng([args.seed, zlib.crc32(probe.encode()), zlib.crc32(utt.encode())])
            audio, _ = sf.read(str(args.dev_root / "wav" / "dev" / utt), dtype="float32")
            name = utt.removesuffix(".wav").replace("/", "_") + ".flac"
            sf.write(str(out_dir / "wav" / name), make_probe(probe, audio, frng, noise_files), SR, subtype="PCM_16")
            metas.append({"path": f"wav/{name}", "source_utt": utt, "label": label, "probe": probe})
        with open(out_dir / "protocol.txt", "w", encoding="utf-8") as f:
            f.writelines(f"{m['path']} {m['label']}\n" for m in metas)
        with open(out_dir / "meta.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metas[0]))
            writer.writeheader()
            writer.writerows(metas)
        print(f"{probe}: {len(metas)} files -> {out_dir}")


if __name__ == "__main__":
    main()
