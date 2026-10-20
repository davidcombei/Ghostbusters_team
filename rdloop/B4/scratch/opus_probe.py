"""
Probe of b4wave.opus_plc (implement_opus_plc a0). Run from SLS_setup/:  python -B ../rdloop/B4/scratch/opus_probe.py
1. selftest values; 2. in-process packets vs the packets ffmpeg writes into its Ogg stream (same system libopus);
3. alignment (best lag) and SNR of the in-process 48 kHz decode vs ffmpeg, and of ffmpeg vs the input;
4. CPU-s per call in-process vs ffmpeg (self + children).
"""
import resource
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from utils import rtc_augment as ra            # noqa: E402
from b4wave import opus_plc as op              # noqa: E402


def ogg_packets(data):
    pkts, cur, pos = [], b"", 0
    while pos < len(data):
        assert data[pos:pos + 4] == b"OggS", pos
        nseg = data[pos + 26]
        lac = data[pos + 27:pos + 27 + nseg]
        body = pos + 27 + nseg
        for L in lac:
            cur += data[body:body + L]
            body += L
            if L < 255:
                pkts.append(cur)
                cur = b""
        pos = body
    return pkts


def ffmpeg_encode(x, p, sr=16000):
    raw = ["-f", "f32le", "-ac", "1", "-ar", str(sr)]
    return ra._ffmpeg([*raw, "-i", "pipe:0", "-c:a", "libopus", "-b:a", f"{p['kbps']}k", "-cutoff", str(p["cutoff"]),
                       "-frame_duration", str(p["frame_ms"]), "-application", p["application"], "-f", "ogg", "pipe:1"],
                      np.asarray(x, np.float32).tobytes())


def snr_aligned(ref, test, max_lag=64):
    best = None
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = ref[lag:], test[:len(test) - lag]
        else:
            a, b = ref[:lag], test[-lag:]
        m = min(len(a), len(b))
        a, b = a[:m].astype(np.float64), b[:m].astype(np.float64)
        s = 10 * np.log10(np.sum(a ** 2) / max(np.sum((a - b) ** 2), 1e-20))
        if best is None or s > best[0]:
            best = (s, lag)
    return best


def main():
    t = time.process_time()
    vals = op.selftest()
    print("selftest", op.summary(vals), f"{time.process_time() - t:.2f} CPU-s")
    import librosa
    from utils.data_utils import read_protocol
    files, labels = read_protocol("/mnt/paml-research/RTCFake/data/train_label.txt", require_label=True)
    rng = np.random.default_rng(0)
    pick = [files[i] for i in rng.choice(len(files), 3, replace=False)]
    for u in pick:
        x, _ = librosa.load(f"/mnt/paml-research/RTCFake/data/wav/train/{u}", sr=16000)
        x = x.astype(np.float32)
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 10 ** (-23 / 20)
        print(f"clip {u} len {len(x)}")
        for app in ra.CODEC_PROFILES:
            p = op.draw_codec_params(np.random.default_rng(hash(app) % 1000), [app])
            info = {}
            y_in, lost = op.roundtrip_mask(x, lambda n: np.zeros(n, bool), p["kbps"], p["cutoff"], p["frame_ms"],
                                           p["application"], info=info)
            ogg = ffmpeg_encode(x, p)
            fp = ogg_packets(ogg)[2:]
            ip = info["packets"]
            same = sum(a == b for a, b in zip(fp, ip))
            y_ff, _ = op.ffmpeg_opus(x, p)
            s_ff_in, lag1 = snr_aligned(y_ff, y_in)
            s_ff_x, lag2 = snr_aligned(y_ff, x)
            print(f"  {app:8s} {p}  la {info['lookahead']} packets ffmpeg {len(fp)} in-proc {len(ip)} identical {same}"
                  f" | SNR(ffmpeg,inproc) {s_ff_in:.2f} dB lag {lag1} | SNR(ffmpeg,input) {s_ff_x:.2f} dB lag {lag2}"
                  f" | margin {s_ff_in - max(15, s_ff_x + 5):+.2f}")
    # cost per call
    x, _ = librosa.load(f"/mnt/paml-research/RTCFake/data/wav/train/{pick[0]}", sr=16000)
    x = x.astype(np.float32)
    ps = [op.draw_codec_params(np.random.default_rng(i)) for i in range(24)]
    c0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.process_time()
    for p in ps:
        op.opus_loss_roundtrip(x, rng, p["kbps"], p["cutoff"], p["frame_ms"], p["application"], (0.01, 0.2), (1, 4))
    t1 = time.process_time()
    for p in ps:
        op.ffmpeg_opus(x, p)
    t2 = time.process_time()
    c1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    ch = (c1.ru_utime + c1.ru_stime) - (c0.ru_utime + c0.ru_stime)
    print(f"cost per call ({len(x) / 16000:.2f} s clip): in-process {(t1 - t0) / 24:.4f} CPU-s; ffmpeg self "
          f"{(t2 - t1) / 24:.4f} + children {ch / 24:.4f} CPU-s")


main()
