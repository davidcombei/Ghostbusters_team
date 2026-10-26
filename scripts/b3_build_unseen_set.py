#!/usr/bin/env python
"""
[B3 wave] Build the frozen *unseen* proxy set `sim_unseen_v1` (anti-overfitting check).

    python scripts/b3_build_unseen_set.py                       # data/dev_noisy_sim/unseen_v1
    python scripts/b3_build_unseen_set.py --limit 24 --version smoke --out /tmp/b3_unseen

A recipe disjoint from training AND from sim_heldout_v1, so that it measures generalisation to
degradations nobody tuned on (every component is reserved from training by
SLS_setup/b3wave/heldout_reserve.py):

    sources   the half-B source utterances of the v1 proxy sets (same seed / n as
              scripts/build_dev_noisy_sim.py, so the same 4,000 dev clips and halves)
    noise     MUSAN *music* (data/augmentation/musan/music; training only uses MUSAN noise)
              at an SNR drawn from {-5, 0, 5, 10, 15, 20, 25} dB
    DSP       ffmpeg compand (p 0.5), agate (p 0.3), crystalizer (p 0.3), 8 kHz round trip (p 0.5)
    codec     speex (libspeex in Ogg, random quality / VAD / DTX) or G.711 mu-law; never used in training
    network   sim_ops.packet_loss (p 0.3)

Outputs <out>/unseen_<version>/{wav/*.flac, protocol.txt, meta.csv, MANIFEST.json} with the
same meta columns as the v1 sets (half fixed to "B", family = "music") and, unless --no_yaml,
adds a `sim_unseen_<version>` entry (role: unseen) to configs/local_eval_sets.yaml. The set
never enters WF1 (utils.local_eval only averages the matched / heldout roles); it is reported
on its own by scripts/local_eval.py / b3_local_eval.py --names ... sim_unseen_v1.
A version is never rebuilt in place: bump --version instead.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "SLS_setup"))

import build_dev_noisy_sim as base  # noqa: E402  (read_dev, select_sources, load, CTX, ...)
from utils import sim_ops  # noqa: E402
from utils.rtc_augment import _ffmpeg, add_noise, finalize, list_audio, resample  # noqa: E402

SR = base.SR
VARIANT = "unseen"
SNR_GRID = (-5, 0, 5, 10, 15, 20, 25)
CODECS = ("speex", "g711_mulaw")
META_COLUMNS = base.META_COLUMNS


def mix_music(audio, files, rng, tries=20):
    snr = int(base.pick(rng, SNR_GRID))
    noisy, path, noise = audio, files[0], np.zeros_like(audio)
    for _ in range(tries):
        path = base.pick(rng, files)
        noisy = add_noise(audio, [path], rng, snr_db=(snr, snr), sr=SR)
        noise = noisy - audio
        if np.mean(noise ** 2) > 1e-10:
            break
    realised = 10 * np.log10(np.mean(audio ** 2) / max(np.mean(noise ** 2), 1e-12))
    rel = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
    return noisy, rel, snr, round(float(realised), 2)


def dsp_compand(audio, sr, rng):
    # soft knee compressor/expander with a random amount of gain reduction above -20 dB
    knee = float(rng.uniform(-30, -15))
    ratio = float(rng.uniform(2, 6))
    top = knee + (0 - knee) / ratio
    graph = (f"compand=attacks=0.02:decays=0.3:points=-80/-80|{knee:.1f}/{knee:.1f}|0/{top:.1f}"
             f":soft-knee=6:gain=0")
    return sim_ops.ffmpeg_filter(audio, sr, graph), {"compand_knee_db": round(knee, 1), "ratio": round(ratio, 2)}


def dsp_agate(audio, sr, rng):
    thr = float(10 ** rng.uniform(-2.3, -1.3))          # 0.005 .. 0.05 linear
    graph = f"agate=threshold={thr:.4f}:ratio={float(rng.uniform(2, 6)):.1f}:attack=10:release={int(rng.integers(100, 400))}"
    return sim_ops.ffmpeg_filter(audio, sr, graph), {"agate_thr": round(thr, 4)}


def dsp_crystalizer(audio, sr, rng):
    i = float(rng.uniform(1, 4))
    return sim_ops.ffmpeg_filter(audio, sr, f"crystalizer=i={i:.2f}"), {"crystalizer_i": round(i, 2)}


def dsp_8k(audio, sr, rng):
    return resample(resample(audio, sr, 8000), 8000, sr), {"bw": "resample8k"}


RAW16 = ["-f", "f32le", "-ac", "1", "-ar", str(SR)]


def codec_stage(audio, sr, codec, rng):
    """
    speex: encode + decode through libspeex in an OGG container (the WAV container that
    sim_ops.ffmpeg_codec uses corrupts speex streams after band-limiting), with a random
    quality 2..8 and VAD / DTX on or off. g711_mulaw: sim_ops.ffmpeg_codec (works in WAV).
    Returns (audio, params) like sim_ops.ffmpeg_codec.
    """
    if codec != "speex":
        return sim_ops.ffmpeg_codec(audio, sr, codec, rng)
    quality = int(rng.integers(2, 9))
    vad = int(rng.random() < 0.5)
    dtx = int(rng.random() < 0.5) if vad else 0
    enc = _ffmpeg([*RAW16, "-i", "pipe:0", "-c:a", "libspeex", "-cbr_quality", str(quality),
                   "-vad", str(vad), "-dtx", str(dtx), "-f", "ogg", "pipe:1"],
                  np.asarray(audio, dtype=np.float32).tobytes())
    dec = _ffmpeg(["-i", "pipe:0", *RAW16, "pipe:1"], enc)
    return np.frombuffer(dec, dtype=np.float32).copy(), {"codec": "speex", "codec_sr": sr, "quality": quality,
                                                         "vad": vad, "dtx": dtx, "container": "ogg"}


def process_unseen(row):
    seed = base.CTX["seed"]
    rng = base.file_rng(seed, VARIANT, row["utt"])
    clean = base.load(row["utt"])
    out = clean.copy()
    meta = {k: "" for k in META_COLUMNS}
    meta.update({"source_utt": row["utt"], "src": row["src"], "lang": row["lang"],
                 "label": row["label"], "half": row["half"], "seed": seed, "family": "music", "aec": ""})
    chain = []
    out, meta["noise_file"], meta["snr_db"], meta["snr_realised"] = mix_music(out, base.CTX["pools"]["music"], rng)
    out = finalize(out, clean)
    for p, fn in ((0.5, dsp_compand), (0.3, dsp_agate), (0.3, dsp_crystalizer), (0.5, dsp_8k)):
        if rng.random() < p:
            out, params = sim_ops.safe(fn, out, SR, rng)
            out = finalize(out, clean)
            chain.append(params)
    out, params = sim_ops.safe(codec_stage, out, SR, base.pick(rng, CODECS), rng)
    meta["codec"], meta["codec_params"] = params.get("codec", "failed"), json.dumps(params)
    out = finalize(out, clean)
    if rng.random() < 0.3:
        out, params = sim_ops.packet_loss(out, SR, rng)
        chain.append(params)
    out = finalize(out, clean)
    name = row["utt"].removesuffix(".wav").replace("/", "_") + ".flac"
    sf.write(str(base.CTX["out_dir"] / "wav" / name), out, SR, subtype="PCM_16")
    meta.update({"path": f"wav/{name}", "dsp_chain": json.dumps(chain)})
    return meta


def add_yaml_entry(yaml_path, name, out_dir):
    text = yaml_path.read_text(encoding="utf-8") if yaml_path.is_file() else ""
    if f"\n{name}:" in text or text.startswith(f"{name}:"):
        print(f"[yaml] {name} already in {yaml_path}")
        return
    rel = out_dir.relative_to(REPO) if out_dir.is_relative_to(REPO) else out_dir
    block = (f"\n# [B3 wave] frozen unseen-recipe set (scripts/b3_build_unseen_set.py): MUSAN music, compand/agate/"
             f"crystalizer/8k, speex|g711, packet loss. role unseen -> never part of WF1; read at wave end only.\n"
             f"{name}:\n  role: unseen\n  base_dir: {rel}\n  protocol: {rel}/protocol.txt\n  meta: {rel}/meta.csv\n")
    yaml_path.write_text(text.rstrip("\n") + "\n" + block, encoding="utf-8")
    print(f"[yaml] added {name} to {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev_root", type=Path, default=Path("/mnt/paml-research/RTCFake/data"))
    parser.add_argument("--music_dir", type=Path, default=REPO / "data" / "augmentation" / "musan" / "music")
    parser.add_argument("--n", type=int, default=4000, help="v1 source count (keep: reproduces the v1 halves)")
    parser.add_argument("--seed", type=int, default=20261003, help="v1 source seed (keep)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N half-B sources (smoke tests)")
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "dev_noisy_sim")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--no_yaml", action="store_true", help="do not touch configs/local_eval_sets.yaml")
    args = parser.parse_args()

    out_dir = args.out / f"{VARIANT}_{args.version}"
    if (out_dir / "MANIFEST.json").is_file():
        sys.exit(f"{out_dir} already built (frozen); bump --version to rebuild")
    music = list_audio(args.music_dir, recursive=True)
    if not music:
        sys.exit(f"no music files under {args.music_dir}")

    dev_rows = base.read_dev(args.dev_root)
    sources = [r for r in base.select_sources(dev_rows, args.n, args.seed) if r["half"] == "B"]
    if args.limit:
        sources = sources[:args.limit]
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    ctx = {"variant": VARIANT, "seed": args.seed, "out_dir": out_dir, "dev_wav": args.dev_root / "wav" / "dev",
           "pools": {"music": music}, "by_label": {}, "rirs": []}
    print(f"[{VARIANT}] {len(sources)} half-B sources -> {out_dir}; music pool {len(music)} files; "
          f"SNR grid {SNR_GRID}; codecs {CODECS}", flush=True)
    with Pool(args.workers, initializer=base.init_worker, initargs=(ctx,)) as pool:
        metas = []
        for i, m in enumerate(pool.imap(process_unseen, sources, chunksize=4), 1):
            metas.append(m)
            if i % 250 == 0 or i == len(sources):
                print(f"   {i}/{len(sources)}", flush=True)

    with open(out_dir / "protocol.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{m['path']} {m['label']}\n" for m in metas)
    with open(out_dir / "meta.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_COLUMNS)
        writer.writeheader()
        writer.writerows(metas)
    failed = sum("failed" in m["dsp_chain"] or "failed" in m["codec_params"] for m in metas)
    manifest = {
        "variant": VARIANT, "version": args.version, "created": datetime.now().isoformat(timespec="seconds"),
        "git_sha": base.git_sha(), "seed": args.seed, "n": len(metas), "snr_grid": SNR_GRID,
        "recipe": {"noise": "musan/music", "dsp": "compand p0.5, agate p0.3, crystalizer p0.3, resample8k p0.5",
                   "codecs": CODECS, "packet_loss_p": 0.3, "sources": "half B of the v1 sources"},
        "args": {k: str(v) for k, v in vars(args).items()},
        "counts_label": dict(Counter(m["label"] for m in metas)),
        "counts_snr": dict(sorted(Counter(str(m["snr_db"]) for m in metas).items())),
        "codec_counts": dict(Counter(m["codec"] for m in metas)),
        "failed_stages": failed,
        "sha256": {"protocol.txt": base.sha256(out_dir / "protocol.txt"), "meta.csv": base.sha256(out_dir / "meta.csv")},
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{VARIANT}] done: {len(metas)} files, failed stages {failed}, codecs {manifest['codec_counts']}")
    if failed:
        print(f"[{VARIANT}] WARNING: {failed} clips had a failed stage (see meta.csv dsp_chain / codec_params)")
    if not args.no_yaml:
        add_yaml_entry(REPO / "configs" / "local_eval_sets.yaml", f"sim_{VARIANT}_{args.version}", out_dir)


if __name__ == "__main__":
    main()
