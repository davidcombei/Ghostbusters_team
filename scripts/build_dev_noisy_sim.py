"""
Build the frozen local proxy sets for the noisy eval subset (roadmap E0.1).

    python scripts/build_dev_noisy_sim.py --variant all          # matched + heldout + echo, v1

Writes, under --out (default data/dev_noisy_sim):

    source_list.txt                  the 4000 dev sources shared by every variant (+ half A/B)
    dev_online_clean/protocol.txt    the clean proxy: dev online, unprocessed
    dev_label_excl_halfB.txt         dev protocol without the half-B sources and their twins
                                     (optional --dev_protocol for new runs)
    <variant>_<version>/
        wav/*.flac                   16 kHz PCM16
        protocol.txt                 "wav/<name>.flac <label>", readable by read_protocol
        protocol_mini.txt            1000 half-A rows, for per-epoch selection (matched/heldout)
        meta.csv                     per-file noise / SNR / DSP / codec metadata
        MANIFEST.json                arguments, git SHA, counts, sha256 of protocol and meta

Variants (see utils/heldout_reserve.py for what is reserved):

    matched   {office, coffee, rain, footsteps, keyboard} from data/augmentation_eval or
              self-echo -> Opus RTC-app profile (codecs_augm)            == today's training sim
    heldout   reserved ESC-50 classes -> [offline sources] NS -> AGC -> VAD -> band-limit
              -> G.722 / G.726 / GSM -> packet loss                      == never seen in training
    echo      self-echo | far-end echo (same-label dev utt x held-out RIR) | AEC residual
              -> Opus RTC-app profile

Every file gets its own RNG seeded from (seed, variant, utt), so the output does not depend
on worker scheduling. A version is never rebuilt in place: bump --version instead.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import zlib
from collections import Counter
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "SLS_setup"))

from utils import sim_ops  # noqa: E402
from utils.rtc_augment import add_echo, add_noise, codecs_augm, finalize, list_audio, read_rir, RIR_EXTS  # noqa: E402

SR = 16000
SNR_GRID = (0, 5, 10, 15, 20)
MATCHED_FAMILIES = ("office", "coffee", "rain", "footsteps", "keyboard", "echo")
HELDOUT_CODECS = ("g722", "g726", "gsm")
ECHO_FAMILIES = ("self_echo", "farend", "farend_aec")
MINI_N = 1000
META_COLUMNS = ["path", "source_utt", "src", "lang", "label", "half", "family", "noise_file",
                "snr_db", "snr_realised", "dsp_chain", "codec", "codec_params", "aec", "seed"]


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def read_dev(dev_root):
    rows = []
    with open(dev_root / "dev_label.txt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                path, label = line.split()
                src, lang = path.split("/")[:2]
                rows.append({"utt": path, "label": label, "src": src, "lang": lang})
    groups = {}
    with open(dev_root / "meta" / "dev_offline_online_pairs.csv", newline="") as f:
        for r in csv.DictReader(f):
            off = r["offline_id"].removeprefix("dev/")
            groups[off] = off
            if r["online_id"]:
                groups[r["online_id"].removeprefix("dev/")] = off
    for r in rows:
        r["group"] = groups.get(r["utt"], r["utt"])
    return rows


def select_sources(rows, n, seed):
    """Stratified (src x lang x label) sample at the natural ratio, with a half A/B per pair group."""
    rng = np.random.default_rng([seed, zlib.crc32(b"sources")])
    strata = {}
    for r in rows:
        strata.setdefault((r["src"], r["lang"], r["label"]), []).append(r)
    keys = sorted(strata)
    quota = {k: int(round(n * len(strata[k]) / len(rows))) for k in keys}
    quota[max(keys, key=lambda k: len(strata[k]))] += n - sum(quota.values())
    chosen = []
    for k in keys:
        idx = rng.choice(len(strata[k]), size=quota[k], replace=False)
        chosen += [strata[k][i] for i in sorted(idx)]
    for r in chosen:
        r["half"] = "AB"[zlib.crc32(f"{seed}:{r['group']}".encode()) % 2]
    return sorted(chosen, key=lambda r: r["utt"])


# --------------------------------------------------------------------------- #
# per-file processing (runs in worker processes)
# --------------------------------------------------------------------------- #
CTX = {}


def init_worker(ctx):
    CTX.update(ctx)


def file_rng(seed, variant, utt):
    return np.random.default_rng([seed, zlib.crc32(variant.encode()), zlib.crc32(utt.encode())])


def load(utt):
    audio, sr = sf.read(str(CTX["dev_wav"] / utt), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    assert sr == SR, f"{utt}: {sr} Hz"
    return audio


def pick(rng, items):
    return items[int(rng.integers(len(items)))]


def mix_noise(audio, files, rng, tries=20):
    snr = int(pick(rng, SNR_GRID))
    for _ in range(tries):
        # ESC-50 clips are zero-padded, so a window can be digital silence; add_noise then
        # returns the clip unchanged -- draw another window / file instead
        path = pick(rng, files)
        noisy = add_noise(audio, [path], rng, snr_db=(snr, snr), sr=SR)
        noise = noisy - audio
        if np.mean(noise ** 2) > 1e-10:
            break
    realised = 10 * np.log10(np.mean(audio ** 2) / max(np.mean(noise ** 2), 1e-12))
    return noisy, str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path), snr, round(float(realised), 2)


def far_end_echo(audio, row, rng, level_db):
    """Another dev utterance of the same label, through a held-out RIR, delayed and scaled."""
    candidates = CTX["by_label"][row["label"]]
    while True:
        other = pick(rng, candidates)
        if other["group"] != row["group"]:
            break
    far = load(other["utt"])
    rir = read_rir(pick(rng, CTX["rirs"]), SR)
    rir = rir[int(np.argmax(np.abs(rir))):][:SR]
    far = fftconvolve(far, rir / max(float(np.sqrt(np.sum(rir ** 2))), 1e-8))
    delay = int(rng.uniform(20, 300) * SR / 1000)
    echo = np.zeros(len(audio), dtype=np.float32)
    seg = far[:max(0, len(audio) - delay)]
    echo[delay:delay + len(seg)] = seg
    p_s, p_e = np.mean(audio ** 2), np.mean(echo ** 2)
    if p_e > 1e-12:
        echo *= np.float32(np.sqrt(p_s / p_e * 10 ** (level_db / 10)))
    return echo, {"far_utt": other["utt"], "delay_ms": round(delay * 1000 / SR), "level_db": round(level_db, 1)}


def process(row):
    variant, seed = CTX["variant"], CTX["seed"]
    rng = file_rng(seed, variant, row["utt"])
    clean = load(row["utt"])
    out = clean.copy()
    meta = {k: "" for k in META_COLUMNS}
    meta.update({"source_utt": row["utt"], "src": row["src"], "lang": row["lang"],
                 "label": row["label"], "half": row["half"], "seed": seed})
    chain = []

    if variant == "matched":
        family = pick(rng, MATCHED_FAMILIES)
        if family == "echo":
            out = add_echo(out, rng, (40.0, 160.0), (0.15, 0.5), SR)
        else:
            out, meta["noise_file"], meta["snr_db"], meta["snr_realised"] = mix_noise(out, CTX["pools"][family], rng)
        out = finalize(out, clean)
        out, app = codecs_augm(out, rng, SR)
        meta["codec"] = f"opus:{app}"

    elif variant == "heldout":
        family = pick(rng, sorted(CTX["pools"]))
        out, meta["noise_file"], meta["snr_db"], meta["snr_realised"] = mix_noise(out, CTX["pools"][family], rng)
        if row["src"] == "offline":                  # faithful order: noise -> platform DSP -> codec
            for p, fn in ((0.7, sim_ops.noise_suppress), (0.5, sim_ops.agc),
                          (0.3, sim_ops.vad_gate), (0.5, sim_ops.bandlimit)):
                if rng.random() < p:
                    out, params = sim_ops.safe(fn, out, SR, rng)
                    out = finalize(out, clean)
                    chain.append(params)
        out, params = sim_ops.safe(sim_ops.ffmpeg_codec, out, SR, pick(rng, HELDOUT_CODECS), rng)
        meta["codec"], meta["codec_params"] = params.get("codec", "failed"), json.dumps(params)
        out = finalize(out, clean)
        if rng.random() < 0.3:
            out, params = sim_ops.packet_loss(out, SR, rng)
            chain.append(params)

    elif variant == "echo":
        family = pick(rng, ECHO_FAMILIES)
        if family == "self_echo":
            taps = int(rng.integers(1, 4))
            for _ in range(taps):
                out = add_echo(out, rng, (30.0, 300.0), (0.1, 0.6), SR)
            chain.append({"taps": taps})
            meta["aec"] = "none"
        elif family == "farend":
            echo, params = far_end_echo(out, row, rng, float(rng.uniform(-25, -5)))
            out = out + echo
            chain.append(params)
            meta["aec"] = "off"
        else:
            echo, params = far_end_echo(out, row, rng, float(rng.uniform(-10, 5)))
            residual, aec_params = sim_ops.aec_residual(echo, SR, rng)
            out = out + residual
            chain += [params, aec_params]
            meta["aec"] = "on"
        out = finalize(out, clean)
        out, app = codecs_augm(out, rng, SR)
        meta["codec"] = f"opus:{app}"
    else:
        raise ValueError(variant)

    out = finalize(out, clean)
    name = row["utt"].removesuffix(".wav").replace("/", "_") + ".flac"
    sf.write(str(CTX["out_dir"] / "wav" / name), out, SR, subtype="PCM_16")
    meta.update({"path": f"wav/{name}", "family": family, "dsp_chain": json.dumps(chain)})
    return meta


# --------------------------------------------------------------------------- #
# writing a variant
# --------------------------------------------------------------------------- #
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_sha():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def variant_pools(variant, eval_root):
    if variant == "matched":
        pools = {f: list_audio(eval_root / ("rnnoise" if f in ("office", "coffee") else "esc50") / f)
                 for f in MATCHED_FAMILIES if f != "echo"}
    elif variant == "heldout":
        pools = {d.name: list_audio(d) for d in sorted((eval_root / "esc50_reserved").iterdir()) if d.is_dir()}
    else:
        pools = {}
    empty = [k for k, v in pools.items() if not v]
    if empty:
        raise FileNotFoundError(f"empty pools {empty} under {eval_root}; run scripts/split_aug_pools.py")
    return pools


def build_variant(variant, sources, dev_rows, args):
    out_dir = args.out / f"{variant}_{args.version}"
    if (out_dir / "MANIFEST.json").is_file():
        print(f"[{variant}] {out_dir} already built (frozen); bump --version to rebuild")
        return
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    by_label = {}
    for r in dev_rows:
        by_label.setdefault(r["label"], []).append(r)
    ctx = {
        "variant": variant, "seed": args.seed, "out_dir": out_dir, "dev_wav": args.dev_root / "wav" / "dev",
        "pools": variant_pools(variant, args.eval_pools), "by_label": by_label,
        "rirs": list_audio(args.eval_pools / "rirs", exts=RIR_EXTS),
    }
    print(f"[{variant}] {len(sources)} files -> {out_dir}  pools: "
          f"{ {k: len(v) for k, v in ctx['pools'].items()} } rirs: {len(ctx['rirs'])}", flush=True)
    with Pool(args.workers, initializer=init_worker, initargs=(ctx,)) as pool:
        metas = []
        for i, m in enumerate(pool.imap(process, sources, chunksize=4), 1):
            metas.append(m)
            if i % 500 == 0 or i == len(sources):
                print(f"   {i}/{len(sources)}", flush=True)

    with open(out_dir / "protocol.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{m['path']} {m['label']}\n" for m in metas)
    with open(out_dir / "meta.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_COLUMNS)
        writer.writeheader()
        writer.writerows(metas)
    if variant in ("matched", "heldout"):
        half_a = [m for m in metas if m["half"] == "A"]
        rng = np.random.default_rng([args.seed, zlib.crc32(f"mini:{variant}".encode())])
        idx = sorted(rng.choice(len(half_a), size=min(MINI_N, len(half_a)), replace=False))
        with open(out_dir / "protocol_mini.txt", "w", encoding="utf-8") as f:
            f.writelines(f"{half_a[i]['path']} {half_a[i]['label']}\n" for i in idx)

    counts = Counter(f"{m['family']}|{m['label']}" for m in metas)
    manifest = {
        "variant": variant, "version": args.version, "created": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(), "seed": args.seed, "n": len(metas), "snr_grid": SNR_GRID,
        "args": {k: str(v) for k, v in vars(args).items()},
        "counts_family_label": dict(sorted(counts.items())),
        "codec_counts": dict(Counter(m["codec"] for m in metas)),
        "failed_stages": sum("failed" in m["dsp_chain"] or "failed" in m["codec_params"] for m in metas),
        "sha256": {"protocol.txt": sha256(out_dir / "protocol.txt"), "meta.csv": sha256(out_dir / "meta.csv")},
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{variant}] done; failed stages: {manifest['failed_stages']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev_root", type=Path, default=Path("/mnt/paml-research/RTCFake/data"))
    parser.add_argument("--eval_pools", type=Path, default=REPO / "data" / "augmentation_eval")
    parser.add_argument("--variant", choices=["matched", "heldout", "echo", "all"], default="all")
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=None, help="only the first N sources (smoke tests)")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "dev_noisy_sim")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev_rows = read_dev(args.dev_root)
    sources = select_sources(dev_rows, args.n, args.seed)
    if args.limit:
        sources = sources[:args.limit]

    with open(args.out / "source_list.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{r['utt']} {r['label']} {r['half']}\n" for r in sources)
    (args.out / "dev_online_clean").mkdir(exist_ok=True)
    with open(args.out / "dev_online_clean" / "protocol.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{r['utt']} {r['label']}\n" for r in dev_rows if r["src"] == "online")
    half_b_groups = {r["group"] for r in sources if r["half"] == "B"}
    with open(args.out / "dev_label_excl_halfB.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{r['utt']} {r['label']}\n" for r in dev_rows if r["group"] not in half_b_groups)
    print(f"sources: {len(sources)} ({Counter(r['half'] for r in sources)}), "
          f"labels {Counter(r['label'] for r in sources)}")

    variants = ["matched", "heldout", "echo"] if args.variant == "all" else [args.variant]
    for variant in variants:
        build_variant(variant, sources, dev_rows, args)


if __name__ == "__main__":
    main()
