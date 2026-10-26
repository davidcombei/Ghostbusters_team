# --------------------------------------------------------------------------- #
# lever opus_plc (plan 2.3)
OPUS_HEADER_RE = r"^B4 Opus PLC: on p="          # plan expect_regex
OPUS_HEADER = "B4 Opus PLC: on p=0.3 loss=0.01..0.20 burst=1..4 decode=48k+resample_poly libopus=libopus 1.3.1"
OPUS_KEYS = ("opus_plc", "plc_l0_5", "plc_l5_10", "plc_l10p", "fail_opus_plc")
OPUS_SNR_FLOOR_DB, OPUS_SNR_MARGIN_DB = 15.0, 5.0       # zero-loss attribution: SNR(ffmpeg, in-process) >= max(15, SNR(ffmpeg, input) + 5)
OPUS_MAX_LAG = 200                                      # samples searched for the aligned SNR (ffmpeg 4.2 decodes ~69 samples late)


def _opus_parent_counts(counts):
    return {k: v for k, v in counts.items() if k not in OPUS_KEYS}


def _opus_counts(counts):
    return {k: counts.get(k, 0) for k in OPUS_KEYS}


def _opus_lib_available():
    try:
        from b4wave import opus_plc
        opus_plc._lib()
        return True, ""
    except OSError as exc:
        return False, str(exc)


@contextlib.contextmanager
def ffmpeg_stub_roundtrip():
    """opus_plc.opus_loss_roundtrip replaced by: the GE parameter draws from rng (uniform, uniform) and ffmpeg_opus on
    the pre-drawn profile, i.e. the lever path with the ffmpeg codec instead of libopus (records the profiles)."""
    from b4wave import opus_plc
    saved = opus_plc.opus_loss_roundtrip
    calls = []

    def stub(audio, rng, kbps, cutoff, frame_ms, application, loss_range, burst_range, sr=16000, dec_sr=48000, info=None):
        rng.uniform(*loss_range)
        rng.uniform(*burst_range)
        calls.append((kbps, cutoff, frame_ms, application))
        y, _ = opus_plc.ffmpeg_opus(audio, {"app": "stub", "kbps": kbps, "cutoff": cutoff, "frame_ms": frame_ms,
                                            "application": application}, sr)
        return y, 0.0

    opus_plc.opus_loss_roundtrip = stub
    try:
        yield calls
    finally:
        opus_plc.opus_loss_roundtrip = saved


OPUS_IMPORT_PROBE = r"""
import contextlib, io, json, sys
import numpy as np
sys.path.insert(0, ".")
from b4wave.checks import base_args, _synthetic_clips
from b4wave.data import build_augmenter_b4
with contextlib.redirect_stdout(io.StringIO()):
    args = base_args(no_codec_aug=False)
    aug, rtc = build_augmenter_b4(args)
    for c in _synthetic_clips(4, np.random.default_rng(3), seconds=0.5):
        aug(c, 16000)
maps = open("/proc/self/maps").read()
print("PROBE " + json.dumps({"opus_module": "b4wave.opus_plc" in sys.modules, "libopus_mapped": "libopus" in maps,
                             "type": type(rtc).__name__, "codec_calls": sum(v for k, v in rtc.stage_counts.items() if k.startswith("codec:"))}))
"""


def _equiv_opus_plc(n_clips=200, n_codec=10):
    """Plan 2.3 equivalence: (4) levers off with the codec ON -> no b4wave.opus_plc import and no libopus mapped
    (fresh process; ffmpeg runs in a child); (3) opus_plc_p = 1.0 with the codec on -> the parent stage:* AND
    codec:<app> counts equal the B3 augmenter's: (a) with the libopus round trip replaced by ffmpeg on the pre-drawn
    profile, the outputs are bit-identical to B3 (the lever's parent draws = codecs_augm's); (b) the real libopus
    path, every codec call in-process, finite, input length."""
    from b3wave.augment import build_rtc_augmenter_b3
    from b4wave.augment import RTCAugmenterB4, build_rtc_augmenter_b4
    ok = True
    r = subprocess.run([sys.executable, "-B", "-c", OPUS_IMPORT_PROBE], cwd=str(SLS_DIR), capture_output=True, text=True, timeout=600)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("PROBE ")), None)
    res = json.loads(line[6:]) if r.returncode == 0 and line else None
    ok &= report("opus_levers_off_codec_on_no_libopus_handle", res is not None and not res["opus_module"]
                 and not res["libopus_mapped"] and res["type"] == "RTCAugmenterB3" and res["codec_calls"] == 4,
                 f"probe rc {r.returncode}: {res or r.stderr[-300:]}", [f"levers off, codec on, 4 clips (fresh process): {res}"])
    avail, why = _opus_lib_available()
    # (a) ffmpeg stub: the lever path is bit-identical to B3 when its codec is ffmpeg
    a = base_args(no_codec_aug=False, aug_opus_plc_p=1.0)
    with quiet():
        ref = build_rtc_augmenter_b3(a)
    with ffmpeg_stub_roundtrip() as calls:
        from b4wave import opus_plc
        saved = opus_plc.selftest, opus_plc.summary, opus_plc.version
        opus_plc.selftest, opus_plc.summary = (lambda: {}), (lambda vals: "stub")
        try:
            with quiet():
                aug = build_rtc_augmenter_b4(a)
            clips = _synthetic_clips(n_codec, np.random.default_rng(53))
            same = sum(int(np.array_equal(ref(c.copy(), 16000), aug(c.copy(), 16000))) for c in clips)
        finally:
            opus_plc.selftest, opus_plc.summary, opus_plc.version = saved
    oc = _opus_counts(aug.stage_counts)
    parent_eq = _opus_parent_counts(aug.stage_counts) == ref.stage_counts
    n_long = sum(1 for c in clips if len(c) >= 16)
    ok &= report("opus_plc_ffmpeg_stub_bit_identical_to_b3", type(aug) is RTCAugmenterB4 and same == n_codec and parent_eq
                 and oc["opus_plc"] == len(calls) == n_long and oc["fail_opus_plc"] == 0,
                 f"{same}/{n_codec} identical; parent counts equal {parent_eq}; opus counts {oc}; stub calls {len(calls)}",
                 [f"opus_plc_p=1.0, round trip = ffmpeg on the pre-drawn profile: {same}/{n_codec} clips identical to B3; "
                  f"parent counts {dict(sorted(_opus_parent_counts(aug.stage_counts).items()))} == B3"])
    if not avail:
        skip("opus_plc_real_parent_stream_untouched", f"libopus missing ({why})")
        return ok
    # (b) the real libopus path, p = 1.0
    n = 40
    with quiet():
        ref = build_rtc_augmenter_b3(a)
        aug = build_rtc_augmenter_b4(a)
    clips = _synthetic_clips(n, np.random.default_rng(59))
    shape_ok, changed = True, 0
    for c in clips:
        x, y = ref(c.copy(), 16000), aug(c.copy(), 16000)
        shape_ok &= len(y) == len(c) and bool(np.all(np.isfinite(y)))
        changed += int(not np.array_equal(x, y))
    oc = _opus_counts(aug.stage_counts)
    parent_eq = _opus_parent_counts(aug.stage_counts) == ref.stage_counts
    codec = {k: v for k, v in aug.stage_counts.items() if k.startswith("codec:")}
    stage = {k: v for k, v in aug.stage_counts.items() if not k.startswith("codec:") and k not in OPUS_KEYS}
    n_codec_calls = sum(codec.values())
    ok &= report("opus_plc_real_parent_stream_untouched", parent_eq and shape_ok and oc["opus_plc"] == n_codec_calls
                 and oc["fail_opus_plc"] == 0 and sum(oc[b] for b in OPUS_KEYS[1:4]) == n_codec_calls,
                 f"parent counts equal {parent_eq}; shape ok {shape_ok}; opus counts {oc}; codec calls {n_codec_calls}",
                 [f"opus_plc_p=1.0, libopus: stage counts {dict(sorted(stage.items()))} and codec counts "
                  f"{dict(sorted(codec.items()))} == B3 ({parent_eq}); {changed}/{n} clips differ from the ffmpeg output; "
                  f"opus counts {oc}"])
    return ok


def check_codec_draw_order(n_seeds=500, n_real=14):
    """
    Plan 2.3: from the same start state, draw_codec_params leaves rng.bit_generator.state exactly where a real
    codecs_augm call leaves it, and ffmpeg_opus on the pre-drawn parameters runs the same ffmpeg commands and is
    bit-identical to codecs_augm. (i) n_seeds start states x (all apps, a 3-app subset), with _ffmpeg recorded (not
    run): same end state, same commands; (ii) n_real real ffmpeg calls: same end state and bit-identical output.
    """
    from utils import rtc_augment as ra
    from b4wave import opus_plc
    ok = True
    saved = ra._ffmpeg
    log = []

    def fake(args, stdin):
        log.append(list(args))
        return np.zeros(8, np.float32).tobytes()

    audio = _synthetic_clips(1, np.random.default_rng(61))[0]
    bad_state, bad_cmd, seen = 0, 0, set()
    ra._ffmpeg = fake
    try:
        for apps in (None, ["wechat", "dingtalk", "telegram"]):
            for s in range(n_seeds):
                r1, r2 = np.random.default_rng([s, 7]), np.random.default_rng([s, 7])
                r1.random(int(s % 5))
                r2.random(int(s % 5))                      # varied start states
                log.clear()
                with quiet():
                    ra.codecs_augm(audio, r1, 16000, apps)
                cmd1 = [list(c) for c in log]
                log.clear()
                p = opus_plc.draw_codec_params(r2, apps)
                with quiet():
                    opus_plc.ffmpeg_opus(audio, p, 16000)
                cmd2 = [list(c) for c in log]
                bad_state += int(r1.bit_generator.state != r2.bit_generator.state)
                bad_cmd += int(cmd1 != cmd2)
                seen.add((p["app"], p["cutoff"], p["frame_ms"]))
    finally:
        ra._ffmpeg = saved
    ok &= report("codec_draw_order_state_and_commands", bad_state == 0 and bad_cmd == 0 and len({a for a, _, _ in seen}) == 7,
                 f"state mismatches {bad_state}, command mismatches {bad_cmd}, apps seen {sorted({a for a, _, _ in seen})}",
                 [f"{2 * n_seeds} start states (all apps / 3-app subset): rng state after draw_codec_params == after codecs_augm "
                  f"in every case; ffmpeg commands identical; {len(seen)} (app, cutoff, frame) combinations seen"])
    same_state, same_out = 0, 0
    clips = _synthetic_clips(n_real, np.random.default_rng(67), seconds=1.0)
    for s, c in enumerate(clips):
        r1, r2 = np.random.default_rng([s, 11]), np.random.default_rng([s, 11])
        with quiet():
            y1, a1 = ra.codecs_augm(c, r1, 16000, None)
            p = opus_plc.draw_codec_params(r2, None)
            y2, a2 = opus_plc.ffmpeg_opus(c, p, 16000)
        same_state += int(r1.bit_generator.state == r2.bit_generator.state)
        same_out += int(a1 == a2 and np.array_equal(y1, y2))
    ok &= report("codec_draw_order_real_ffmpeg_bit_identical", same_state == n_real and same_out == n_real,
                 f"state equal {same_state}/{n_real}, output identical {same_out}/{n_real}",
                 [f"{n_real} real codecs_augm calls: end state equal {same_state}/{n_real}; ffmpeg_opus(pre-drawn) output "
                  f"bit-identical {same_out}/{n_real}"])
    return ok


def _snr_aligned(ref, test, max_lag=OPUS_MAX_LAG):
    """(SNR dB, lag): max over integer lags |lag| <= max_lag of 10 log10(sum ref^2 / sum (ref - test)^2) on the
    overlap (lag > 0: test leads ref by lag samples)."""
    ref = np.asarray(ref, np.float64)
    test = np.asarray(test, np.float64)
    best = (-np.inf, 0)
    for lag in range(-max_lag, max_lag + 1):
        a, b = (ref[lag:], test[:len(test) - lag]) if lag >= 0 else (ref[:lag], test[-lag:])
        m = min(len(a), len(b))
        if m <= 0:
            continue
        a, b = a[:m], b[:m]
        s = 10 * np.log10(np.sum(a ** 2) / max(np.sum((a - b) ** 2), 1e-30))
        if s > best[0]:
            best = (float(s), lag)
    return best


def _ogg_packets(data):
    """Opus packets of an Ogg stream (the OpusHead / OpusTags header packets dropped)."""
    pkts, cur, pos = [], b"", 0
    while pos + 27 <= len(data) and data[pos:pos + 4] == b"OggS":
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
    return pkts[2:]


def _training_clips(k, seed=0):
    """k training clips (seeded pick from the training protocol), loaded at 16 kHz + the trainer's loudness norm."""
    import librosa
    from utils.data_utils import build_loudness, read_protocol
    args = base_args()
    files, _ = read_protocol(args.train_protocol, require_label=True)
    pick = [files[int(i)] for i in sorted(np.random.default_rng(seed).choice(len(files), size=k, replace=False))]
    loud = build_loudness(args)
    out = []
    for u in pick:
        audio, sr = librosa.load(str(Path(args.train_data_path) / u), sr=16000)
        audio = audio.astype(np.float32)
        if loud is not None:
            audio = loud(audio, sr)
        out.append((u, audio))
    return out


def check_opus_plc(n_clips=100):
    """opus_plc: validation, header / describe / meta keys, the selftest's named asserts with their values, the GE
    mask statistics, PLC only where packets are lost, the lever with the real libopus path (rate, bins, unpicked clips =
    B3 bit for bit, determinism), the fallback, chains with apm_ns / dtx_gate, and the zero-loss attribution on
    3 training clips x 7 profiles (+ encoder parity: the packets ffmpeg writes)."""
    from utils import rtc_augment as ra
    from b3wave.augment import RTCAugmenterB3, build_rtc_augmenter_b3
    from b4wave import opus_plc
    from b4wave.augment import RTCAugConfigB4, RTCAugmenterB4, b4_config_from_args, b4_header_lines, build_rtc_augmenter_b4
    from b4wave.train import build_parser, validate_b4_args

    ok = True
    parser = build_parser()
    rej = {}
    with quiet():
        for name, extra in (("loss_hi_0.5", ["--aug_opus_loss", "0.1", "0.5"]), ("loss_lo_gt_hi", ["--aug_opus_loss", "0.2", "0.1"]),
                            ("loss_neg", ["--aug_opus_loss", "-0.01", "0.1"]), ("burst_lt_1", ["--aug_opus_burst", "0.5", "2"]),
                            ("burst_lo_gt_hi", ["--aug_opus_burst", "3", "2"]), ("p_gt_1", []), ("loss_1_value", ["--aug_opus_loss", "0.1"])):
            argv = BASE_ARGV + ["--aug_opus_plc_p", "1.5" if name == "p_gt_1" else "0.3"] + extra
            try:
                validate_b4_args(parser, parser.parse_args(argv))
                rej[name] = False
            except SystemExit:
                rej[name] = True
        try:                                            # ranges unread at p = 0 (plan: "Read only when p > 0")
            validate_b4_args(parser, parser.parse_args(BASE_ARGV + ["--aug_opus_loss", "0.6", "0.7"]))
            rej["p0_ranges_unread"] = True
        except SystemExit:
            rej["p0_ranges_unread"] = False
        try:
            validate_b4_args(parser, parser.parse_args(BASE_ARGV + ["--aug_opus_plc_p", "0.3", "--aug_opus_loss", "0", "0"]))
            rej["control_0_0_accepted"] = True
        except SystemExit:
            rej["control_0_0_accepted"] = False
    for name, lr, br in (("check_ranges_loss", (0.3, 0.5), (1, 4)), ("check_ranges_burst", (0.01, 0.2), (0.9, 4))):
        try:
            opus_plc.check_ranges(lr, br)
            rej[name] = False
        except ValueError:
            rej[name] = True
    base4 = _b4_fields(b4_config_from_args(base_args()))
    for name, over in (("aug_codec_off", {"with_codecs": False}), ("aug_codec_random", {"codec_profiles": "random"}),
                       ("aug_bad_loss", {"opus_loss": (0.3, 0.2)})):
        try:
            with quiet():
                RTCAugmenterB4(RTCAugConfigB4(**dict(base4, opus_plc_p=0.3, **over)))
            rej[name] = False
        except ValueError:
            rej[name] = True
    ok &= report("opus_rejects_bad_ranges_and_profiles", all(rej.values()), f"{rej}", [f"{rej}"])
    with quiet():
        off = build_rtc_augmenter_b4(base_args(aug_opus_loss=[0.6, 0.7], aug_opus_burst=[0.1, 0.2]))
    ok &= report("opus_ranges_unread_when_p0", type(off) is RTCAugmenterB3, f"{type(off).__name__}")

    avail, why = _opus_lib_available()
    if not avail:
        skip("opus_plc (header / selftest / lever)", f"libopus missing ({why})")
        return ok
    a = base_args(aug_opus_plc_p=0.3)
    head = b4_header_lines(a)
    ok &= report("opus_header_matches_expect_regex", head == [OPUS_HEADER] and re.match(OPUS_HEADER_RE, head[0]) is not None,
                 f"{head}", head)
    with quiet():
        ref = build_rtc_augmenter_b3(a)
        aug = build_rtc_augmenter_b4(a)
    ok &= report("opus_describe_and_meta_keys", type(aug) is RTCAugmenterB4 and aug.codec_lever == "opus_plc"
                 and aug.describe() == ref.describe() + " | opus_plc p=0.3 loss=0.01..0.20 burst=1..4 decode=48k"
                 and aug.meta_keys == ref.meta_keys + list(OPUS_KEYS),
                 f"describe ...{aug.describe()[-60:]}; meta tail {aug.meta_keys[-6:]}",
                 [f"describe suffix: {aug.describe()[len(ref.describe()):]}", f"meta keys + {aug.meta_keys[len(ref.meta_keys):]}"])

    # the selftest: every named assert with its measured values
    t0 = time.process_time()
    vals = opus_plc.measure()
    t_self = time.process_time() - t0
    ctl = vals["ctl"]
    p = vals["plc"]
    sh = vals["shapes"]
    lines = {
        "loads": [f"{vals['version']} via ctypes from {vals['path']}"],
        "ctl_roundtrip": [f"{len(ctl)} encoder configs (7 apps x min/max kbps x cutoffs): set == get for bitrate "
                          f"{sum(r['bitrate'] == r['kbps'] * 1000 for r in ctl)}/{len(ctl)}, max bandwidth "
                          f"{sum(r['max_bw'] == opus_plc.BANDWIDTH[r['cutoff']] for r in ctl)}/{len(ctl)}; vbr "
                          f"{sorted({r['vbr'] for r in ctl})} constraint {sorted({r['vbr_constraint'] for r in ctl})} complexity "
                          f"{sorted({r['complexity'] for r in ctl})} loss% {sorted({r['loss_perc'] for r in ctl})} fec "
                          f"{sorted({r['fec'] for r in ctl})} dtx {sorted({r['dtx'] for r in ctl})} lookahead "
                          f"{sorted({r['lookahead'] for r in ctl})}"],
        "plc_not_zero": [f"tone {p['rms_tone_db']:.2f} dBFS, packets {opus_plc.GAP[0]}-{opus_plc.GAP[1] - 1} lost ({p['lost']}): "
                         f"first concealed 20 ms max |x| {p['max_abs_first']:.4f} RMS {p['rms_first_db']:.2f} dBFS "
                         f"(no-loss decode {p['rms_noloss_db']:.2f}); whole 100 ms gap RMS {p['rms_gap_db']:.2f} dBFS"],
        "shape_finite": [f"{sum(s['len'] == s['n'] and s['finite'] for s in sh)}/{len(sh)} runs (7 apps x frames x loss 0 / 0.2, "
                         f"{sorted({s['n'] for s in sh})} samples): length == input and finite; lost packets in the 0.2 runs "
                         f"{sum(s['lost'] for s in sh if s['loss'] > 0)}/{sum(s['packets'] for s in sh if s['loss'] > 0)}"],
    }
    for name in opus_plc.ASSERTS:
        try:
            opus_plc.verify(vals, (name,))
            ok &= report(f"opus_selftest[{name}]", True, values=lines[name])
        except AssertionError as exc:
            ok &= report(f"opus_selftest[{name}]", False, str(exc), lines[name])
    print(f"    selftest CPU {t_self:.2f} s", flush=True)

    # Gilbert-Elliott mask statistics
    stats = {}
    r = np.random.default_rng(71)
    for L, B in ((0.10, 2.5), (0.20, 4.0), (0.01, 1.0), (0.05, 1.0)):
        m = opus_plc.ge_mask(r, 400_000, L, B)
        edges = np.flatnonzero(np.diff(np.r_[0, m.astype(np.int8), 0]))
        runs = edges[1::2] - edges[0::2]
        stats[(L, B)] = (float(m.mean()), float(runs.mean()) if len(runs) else 0.0, int(runs.max()) if len(runs) else 0)
    first = np.mean([opus_plc.ge_mask(r, 3, 0.2, 3.0)[0] for _ in range(20000)])
    zero = opus_plc.ge_mask(r, 10000, 0.0, 2.0)
    st_ok = (all(abs(v[0] - L) <= 0.01 * max(1.0, L / 0.05) and abs(v[1] - B) <= 0.05 * B for (L, B), v in stats.items())
             and stats[(0.01, 1.0)][2] == 1 and stats[(0.05, 1.0)][2] == 1 and abs(first - 0.2) < 0.015 and not zero.any())
    ok &= report("opus_ge_mask_statistics", st_ok, f"{stats}; P(first lost) {first:.4f}; any loss at L=0 {bool(zero.any())}",
                 [f"(L, B) -> (realised loss, mean burst, max burst) over 400k packets: "
                  + "; ".join(f"({L:g}, {B:g}) -> ({v[0]:.4f}, {v[1]:.3f}, {v[2]})" for (L, B), v in stats.items()),
                  f"stationary start: P(first packet lost | L 0.2) {first:.4f} over 20k; L = 0 -> {int(zero.sum())} lost"])

    # PLC only where packets are lost: the 48 kHz decode equals the no-loss decode up to the first lost packet
    x = opus_plc._tone(16000, -26.0)
    i0, i1 = {}, {}
    mask = np.zeros(64, bool)
    mask[[17, 30, 31]] = True
    y0, _ = opus_plc.roundtrip_mask(x, lambda n: np.zeros(n, bool), 16, 6000, 20, "voip", info=i0)
    y1, lost = opus_plc.roundtrip_mask(x, lambda n: mask[:n], 16, 6000, 20, "voip", info=i1)
    k = 17 * i1["frame_dec"]
    pre_same = np.array_equal(i0["decoded"][:k], i1["decoded"][:k])
    post_diff = not np.array_equal(i0["decoded"][k:k + i1["frame_dec"]], i1["decoded"][k:k + i1["frame_dec"]])
    ok &= report("opus_plc_only_from_the_first_lost_packet", pre_same and post_diff and int(lost.sum()) == 3
                 and i0["packets"] == i1["packets"],
                 f"pre identical {pre_same}; lost frame differs {post_diff}; lost {int(lost.sum())}",
                 [f"packets 17, 30, 31 lost: the 48 kHz decode is bit-identical to the no-loss decode before sample {k} "
                  f"({pre_same}) and differs in the concealed frame ({post_diff}); encoded packets identical (loss is "
                  f"applied at the receiver only)"])

    # the lever with the real path, p = 0.3, codec on: rate, bins, unpicked clips = B3 bit for bit, the rng4 draws
    # replayed draw by draw (pick random(); picked: uniform, uniform, random(n_packets)), determinism
    a = base_args(aug_opus_plc_p=0.3)
    with quiet():
        ref = build_rtc_augmenter_b3(a)
        aug = build_rtc_augmenter_b4(a)
        aug2 = build_rtc_augmenter_b4(a)
    clips = _synthetic_clips(n_clips, np.random.default_rng(73))
    drawn = []
    real_draw = opus_plc.draw_codec_params

    def spy(rng, apps=None):
        out = real_draw(rng, apps)
        drawn.append(out)
        return out

    outs, same_flags, shape_ok = [], [], True
    opus_plc.draw_codec_params = spy
    try:
        for c in clips:
            x0, y = ref(c.copy(), 16000), aug(c.copy(), 16000)
            outs.append(y)
            same_flags.append(bool(np.array_equal(x0, y)))
            shape_ok &= len(y) == len(c) and bool(np.all(np.isfinite(y)))
    finally:
        opus_plc.draw_codec_params = real_draw
    det = sum(int(np.array_equal(y, aug2(c.copy(), 16000))) for c, y in zip(clips, outs))
    oc = _opus_counts(aug.stage_counts)
    n_codec_calls = sum(v for k2, v in aug.stage_counts.items() if k2.startswith("codec:"))
    same_ref = sum(same_flags)
    lo_hi = (0.3 * n_codec_calls - 3 * math.sqrt(n_codec_calls * 0.21), 0.3 * n_codec_calls + 3 * math.sqrt(n_codec_calls * 0.21))
    cond = (_opus_parent_counts(aug.stage_counts) == ref.stage_counts and shape_ok and det == n_clips
            and same_ref == n_codec_calls - oc["opus_plc"] + (n_clips - n_codec_calls) and oc["fail_opus_plc"] == 0
            and lo_hi[0] <= oc["opus_plc"] <= lo_hi[1] and sum(oc[b] for b in OPUS_KEYS[1:4]) == oc["opus_plc"])
    ok &= report("opus_lever_real_path", cond,
                 f"parent counts equal {_opus_parent_counts(aug.stage_counts) == ref.stage_counts}; counts {oc}; codec calls "
                 f"{n_codec_calls}; identical to B3 {same_ref}; deterministic {det}/{n_clips}; shape ok {shape_ok}",
                 [f"p=0.3 on {n_clips} synthetic clips: {oc['opus_plc']}/{n_codec_calls} codec calls in-process (rate "
                  f"{oc['opus_plc'] / max(1, n_codec_calls):.3f}, 3-sigma range {lo_hi[0]:.1f}..{lo_hi[1]:.1f}); loss bins "
                  f"{ {b: oc[b] for b in OPUS_KEYS[1:4]} }; the {same_ref} other clips bit-identical to B3 (ffmpeg, same "
                  f"parent draws); a second augmenter gives identical outputs {det}/{n_clips}; parent counts == B3"])
    la = sorted({row["lookahead"] for row in vals["ctl"]})
    r4 = np.random.default_rng([1234, os.getpid(), 0xB4])
    replay_picked, it = [], iter(drawn)
    for c in clips:
        if len(c) < 16:
            replay_picked.append(False)
            continue
        if r4.random() < 0.3:
            prm = next(it)
            r4.uniform(0.01, 0.20)
            r4.uniform(1.0, 4.0)
            frame = 16 * int(prm["frame_ms"])
            r4.random(-(-(len(c) + la[0]) // frame))
            replay_picked.append(True)
        else:
            replay_picked.append(False)
    picked = [not f for f in same_flags]
    ok &= report("opus_plc_pick_draws_replay", len(la) == 1 and replay_picked == picked and len(drawn) == sum(picked)
                 and r4.bit_generator.state == aug.rng4.bit_generator.state,
                 f"lookahead {la}; replay picks {sum(replay_picked)} vs {sum(picked)} (equal {replay_picked == picked}); "
                 f"rng4 end state equal {r4.bit_generator.state == aug.rng4.bit_generator.state}",
                 [f"rng4 = default_rng([1234, pid, 0xB4]) replayed per codec call (random() < p; picked: uniform(loss), "
                  f"uniform(burst), random(ceil((len + {la[0] if la else '?'}) / frame))): the picked clips are exactly the clips "
                  f"that differ from B3 ({sum(picked)}), and the stream ends in the augmenter's rng4 state"])

    # loss bins under the default ranges: every bin populated over 300 draws of the real round trip on 1 s clips
    rb = np.random.default_rng([1234, 0xB4, 1])
    bins = {b: 0 for b in opus_plc.BINS}
    realised = []
    cx = opus_plc._tone(16000, -26.0)
    for _ in range(300):
        _, loss = opus_plc.opus_loss_roundtrip(cx, rb, 24, 8000, 20, "voip", (0.01, 0.20), (1.0, 4.0))
        bins[opus_plc.loss_bin(loss)] += 1
        realised.append(loss)
    zero_ctl = [opus_plc.opus_loss_roundtrip(cx, rb, 24, 8000, 20, "voip", (0.0, 0.0), (1.0, 4.0))[1] for _ in range(20)]
    ok &= report("opus_loss_bins_and_zero_loss_control", all(v > 0 for v in bins.values()) and max(zero_ctl) == 0.0,
                 f"bins {bins}; zero-loss control realised {max(zero_ctl)}",
                 [f"default ranges, 300 round trips of 1 s (55 packets): bins {bins}, realised loss mean "
                  f"{np.mean(realised):.4f} (target mean 0.105); --aug_opus_loss 0 0: realised loss {max(zero_ctl):g} in 20/20"])

    # fallback: an unsupported rate (22050 Hz) fails in libopus -> ffmpeg with the same draws = B3 bit for bit
    a = base_args(aug_opus_plc_p=1.0)
    with quiet():
        ref = build_rtc_augmenter_b3(a)
        aug = build_rtc_augmenter_b4(a)
        clips = _synthetic_clips(4, np.random.default_rng(79), sr=22050)
        same = sum(int(np.array_equal(ref(c.copy(), 22050), aug(c.copy(), 22050))) for c in clips)
    oc = _opus_counts(aug.stage_counts)
    ok &= report("opus_fallback_ffmpeg_same_draws", same == 4 and oc["fail_opus_plc"] == 4 and oc["opus_plc"] == 0,
                 f"identical {same}/4; counts {oc}", [f"22050 Hz clips, p=1: fail_opus_plc {oc['fail_opus_plc']}, outputs = B3 {same}/4"])

    # chain: apm_ns (identity) -> dtx_gate (identity) -> opus_plc on one rng4 stream; parent counts unchanged
    a = base_args(aug_apm_p=1.0, aug_dtx_p=1.0, aug_opus_plc_p=1.0)
    with identity_ns(), identity_dtx():
        with quiet():
            ref = build_rtc_augmenter_b3(a)
            aug = build_rtc_augmenter_b4(a)
        clips = _synthetic_clips(10, np.random.default_rng(83))
        for c in clips:
            ref(c.copy(), 16000)
            aug(c.copy(), 16000)
    n_ns = sum(aug.stage_counts.get(f"wrtc_ns{d}", 0) for d in APM_LEVELS)
    n_dtx = aug.stage_counts.get("wrtc_dtx_zero", 0) + aug.stage_counts.get("wrtc_dtx_cng", 0)
    parent = {k2: v for k2, v in aug.stage_counts.items() if k2 not in OPUS_KEYS and k2 not in DTX_KEYS and not k2.startswith("wrtc_ns")}
    n_long = sum(1 for c in clips if len(c) >= 16)
    ok &= report("opus_chain_after_apm_and_dtx", aug.levers_on == ["apm_ns", "dtx_gate", "opus_plc"] and n_ns == n_long
                 and n_dtx == n_long and aug.stage_counts.get("opus_plc", 0) == sum(v for k2, v in parent.items() if k2.startswith("codec:"))
                 and parent == ref.stage_counts,
                 f"levers {aug.levers_on}; {aug.stage_counts}",
                 [f"apm_ns + dtx_gate (identity) + opus_plc, p=1: levers {aug.levers_on}; wrtc_ns {n_ns}, dtx {n_dtx}, "
                  f"opus_plc {aug.stage_counts.get('opus_plc', 0)}; parent counts == B3"])

    # zero-loss attribution on 3 training clips x 7 profiles (+ encoder parity with the ffmpeg packets)
    args = base_args()
    if not Path(args.train_protocol).is_file() or not Path(args.train_data_path).is_dir():
        skip("opus_zero_loss_attribution", f"training data missing ({args.train_protocol})")
        return ok
    rows, bad, parity_bad = [], [], []
    for u, audio in _training_clips(3, seed=0):
        for app, (kbps, cutoffs, frames, application) in ra.CODEC_PROFILES.items():
            prof = {"app": app, "kbps": kbps[0], "cutoff": min(cutoffs), "frame_ms": max(frames), "application": application}
            info = {}
            y_in, _ = opus_plc.roundtrip_mask(audio, lambda n: np.zeros(n, bool), prof["kbps"], prof["cutoff"],
                                              prof["frame_ms"], application, info=info)
            with quiet():
                y_ff, _ = opus_plc.ffmpeg_opus(audio, prof)
            raw = ["-f", "f32le", "-ac", "1", "-ar", "16000"]
            ogg = ra._ffmpeg([*raw, "-i", "pipe:0", "-c:a", "libopus", "-b:a", f"{prof['kbps']}k", "-cutoff", str(prof["cutoff"]),
                              "-frame_duration", str(prof["frame_ms"]), "-application", application, "-f", "ogg", "pipe:1"],
                             np.asarray(audio, np.float32).tobytes())
            fp = _ogg_packets(ogg)
            same_pk = sum(x1 == x2 for x1, x2 in zip(fp, info["packets"]))
            s_pair, lag_pair = _snr_aligned(y_ff, y_in)
            s_in, lag_in = _snr_aligned(y_ff, audio)
            need = max(OPUS_SNR_FLOOR_DB, s_in + OPUS_SNR_MARGIN_DB)
            rows.append((u, app, prof, s_pair, lag_pair, s_in, lag_in, need, same_pk, len(fp), len(info["packets"])))
            if s_pair < need:
                bad.append((u, app, round(s_pair, 2), round(need, 2)))
            if not (len(fp) == len(info["packets"]) and same_pk == len(fp)):
                parity_bad.append((u, app, same_pk, len(fp), len(info["packets"])))
    margins = [r[3] - r[7] for r in rows]
    ok &= report("opus_zero_loss_attribution", not bad, f"profiles below the bar {bad}",
                 [f"3 training clips (loudness -23 LUFS) x 7 app profiles at their lowest kbps / narrowest cutoff / longest frame; "
                  f"SNR aligned over |lag| <= {OPUS_MAX_LAG}; bar max({OPUS_SNR_FLOOR_DB:g}, SNR(ffmpeg, input) + {OPUS_SNR_MARGIN_DB:g}); "
                  f"min margin {min(margins):+.2f} dB ({rows[int(np.argmin(margins))][1]})"]
                 + [f"{u.split('/')[-1]} {app:8s} {pr['kbps']}k cut {pr['cutoff']} {pr['frame_ms']} ms {pr['application']}: "
                    f"SNR(ffmpeg, in-process) {sp:.2f} dB (lag {lp:+d}) | SNR(ffmpeg, input) {si:.2f} dB (lag {li:+d}) | bar {nd:.2f}"
                    f" | margin {sp - nd:+.2f}" for u, app, pr, sp, lp, si, li, nd, *_ in rows])
    ok &= report("opus_encoder_parity_with_ffmpeg_packets", not parity_bad, f"{parity_bad}",
                 [f"21 runs: in-process packets byte-identical to the packets in ffmpeg's Ogg stream in "
                  f"{sum(r[8] == r[9] == r[10] for r in rows)}/21 (packets per run {sorted({r[10] for r in rows})}); "
                  f"ffmpeg's decode lags the input by {sorted({r[6] for r in rows})} samples (its 4.2 pre-skip), the in-process "
                  f"decode by {sorted({r[6] - r[4] for r in rows})} relative to the input (3 x look-ahead trimmed)"])
    return ok
