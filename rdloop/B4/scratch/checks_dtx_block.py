# --------------------------------------------------------------------------- #
# lever dtx_gate (plan 2.2)
DTX_HEADER_RE = r"^B4 DTX gate: on p="          # plan expect_regex
DTX_HEADER = "B4 DTX gate: on p=0.3 fill=mix modes=[1, 2, 3] frames_ms=[10, 20, 30] hangover_ms=40..200"
DTX_KEYS = ("wrtc_dtx_zero", "wrtc_dtx_cng", "dtx_g0", "dtx_g1_20", "dtx_g20_40", "dtx_g40p", "fail_wrtc_dtx")
DTX_BALANCE_WARN = 0.05             # |bona - spoof| of the mean realised gated fraction (plan 2.2)


@contextlib.contextmanager
def identity_dtx():
    """dsp_dtx with an identity gate (no webrtcvad): wrtc_dtx draws the documented per-clip parameters from rng
    and returns its input (gated fraction 0); records the drawn parameters."""
    from b4wave import dsp_dtx
    saved = (dsp_dtx.wrtc_dtx, dsp_dtx.selftest, dsp_dtx.summary)
    calls = []

    def ident(audio, rng, sr=16000, fill="mix"):
        p = dsp_dtx.draw_params(rng, fill)
        calls.append(p)
        return np.asarray(audio, dtype=np.float32).reshape(-1).copy(), 0.0, p["fill"]

    dsp_dtx.wrtc_dtx = ident
    dsp_dtx.selftest = lambda: {}
    dsp_dtx.summary = lambda vals: "identity stub"
    try:
        yield calls
    finally:
        dsp_dtx.wrtc_dtx, dsp_dtx.selftest, dsp_dtx.summary = saved


def _dtx_parent_counts(counts):
    return {k: v for k, v in counts.items() if k not in DTX_KEYS}


def _dtx_counts(counts):
    return {k: counts.get(k, 0) for k in DTX_KEYS}


def _equiv_dtx_gate(n_clips=200, n_codec=10):
    """Plan 2.2 equivalence item 3: dtx_p = 1.0 -> the parent stage:* / codec counts equal the B3 augmenter's
    (rng4 leaves the parent stream untouched). (a) identity gate (no webrtcvad): outputs bit-identical to B3 and
    the per-clip parameters = default_rng([1234, pid, 0xB4]) draws; (b) the real webrtcvad gate: parent counts
    equal, every clip processed, lengths unchanged."""
    import importlib.util
    from b3wave.augment import build_rtc_augmenter_b3
    from b4wave import dsp_dtx
    from b4wave.augment import RTCAugmenterB4, build_rtc_augmenter_b4
    ok = True
    for with_codecs, n in ((False, n_clips), (True, n_codec)):
        a = base_args(no_codec_aug=not with_codecs, aug_dtx_p=1.0)
        with quiet():
            ref = build_rtc_augmenter_b3(a)
        with identity_dtx() as calls:
            with quiet():
                aug = build_rtc_augmenter_b4(a)
            clips = _synthetic_clips(n, np.random.default_rng(23))
            same = sum(int(np.array_equal(ref(c.copy(), 16000), aug(c.copy(), 16000))) for c in clips)
        r = np.random.default_rng([1234, os.getpid(), 0xB4])
        expect = []
        for _ in range(n):
            if r.random() < 1.0:
                expect.append(dsp_dtx.draw_params(r, "mix"))
        dc = _dtx_counts(aug.stage_counts)
        parent_eq = _dtx_parent_counts(aug.stage_counts) == ref.stage_counts
        cond = (type(aug) is RTCAugmenterB4 and same == n and parent_eq and calls == expect
                and dc["wrtc_dtx_zero"] + dc["wrtc_dtx_cng"] == n and dc["dtx_g0"] == n and dc["fail_wrtc_dtx"] == 0)
        ok &= report(f"dtx_identity_parent_stream_untouched(codecs={with_codecs})", cond,
                     f"{same}/{n} identical; parent counts equal {parent_eq}; dtx counts {dc}; params follow rng4 {calls == expect}",
                     [f"dtx_p=1.0, identity gate: {same}/{n} clips identical to B3; parent counts "
                      f"{dict(sorted(_dtx_parent_counts(aug.stage_counts).items()))} == B3; fills "
                      f"{ {k: dc[k] for k in ('wrtc_dtx_zero', 'wrtc_dtx_cng')} }, modes "
                      f"{ {m: sum(p['mode'] == m for p in calls) for m in (1, 2, 3)} }, frames "
                      f"{ {f: sum(p['frame_ms'] == f for p in calls) for f in (10, 20, 30)} } = rng4 draws"])
    if importlib.util.find_spec("webrtcvad") is None:
        skip("dtx_real_parent_stream_untouched", "package webrtcvad missing")
        return ok
    for with_codecs, n in ((False, n_clips), (True, n_codec)):
        a = base_args(no_codec_aug=not with_codecs, aug_dtx_p=1.0)
        with quiet():
            ref = build_rtc_augmenter_b3(a)
            aug = build_rtc_augmenter_b4(a)
        clips = _synthetic_clips(n, np.random.default_rng(29))
        shape_ok, changed = True, 0
        for c in clips:
            x, y = ref(c.copy(), 16000), aug(c.copy(), 16000)
            shape_ok &= len(y) == len(c) and bool(np.all(np.isfinite(y)))
            changed += int(not np.array_equal(x, y))
        dc = _dtx_counts(aug.stage_counts)
        parent_eq = _dtx_parent_counts(aug.stage_counts) == ref.stage_counts
        cond = (parent_eq and shape_ok and dc["wrtc_dtx_zero"] + dc["wrtc_dtx_cng"] == n
                and sum(dc[b] for b in DTX_KEYS[2:6]) == n and dc["fail_wrtc_dtx"] == 0)
        ok &= report(f"dtx_real_parent_stream_untouched(codecs={with_codecs})", cond,
                     f"parent counts equal {parent_eq}; shape ok {shape_ok}; dtx counts {dc}",
                     [f"dtx_p=1.0, webrtcvad: parent counts == B3 {parent_eq}; {changed}/{n} clips changed; dtx counts {dc}"])
    return ok


def _dtx_replay(clips, ref, p, fill, apm_first=False):
    """The lever's documented draws replayed on the B3 outputs (no codec): rng4 = default_rng([1234, pid, 0xB4]);
    per clip [apm: random(), integers(4) with an identity NS]; random() < p -> wrtc_dtx(x, rng4, 16000, fill),
    finalize. Returns (expected outputs, expected dtx counts)."""
    from utils.rtc_augment import finalize
    from b4wave import dsp_dtx
    r = np.random.default_rng([1234, os.getpid(), 0xB4])
    outs, counts = [], {k: 0 for k in DTX_KEYS}
    for c in clips:
        x = ref(c.copy(), 16000)
        if apm_first:
            if r.random() < 1.0:
                r.integers(len(APM_LEVELS))
        if r.random() < p:
            y, frac, used = dsp_dtx.wrtc_dtx(x, r, 16000, fill)
            counts[f"wrtc_dtx_{used}"] += 1
            counts[dsp_dtx.gated_bin(frac)] += 1
            x = finalize(y, x)
        outs.append(x)
    return outs, counts


def check_dtx(n_clips=200):
    """dtx_gate: validation, header / describe / meta keys, the selftest's named asserts with their values, the lever
    with the real webrtcvad gate (replayed draw order), the chain with the codec and with apm_ns, edge cases."""
    import importlib.util
    from b3wave.augment import RTCAugmenterB3, build_rtc_augmenter_b3
    from b4wave import dsp_dtx
    from b4wave.augment import RTCAugConfigB4, RTCAugmenterB4, b4_config_from_args, b4_header_lines, build_rtc_augmenter_b4
    from b4wave.train import build_parser, validate_b4_args

    ok = True
    parser = build_parser()
    with quiet():
        try:
            parser.parse_args(BASE_ARGV + ["--aug_dtx_p", "0.3", "--aug_dtx_fill", "white"])
            rej_parse = False
        except SystemExit:
            rej_parse = True
        try:
            validate_b4_args(parser, base_args(aug_dtx_p=1.5))
            rej_p = False
        except SystemExit:
            rej_p = True
    try:
        dsp_dtx.check_fill("white")
        rej_dsp = False
    except ValueError:
        rej_dsp = True
    with identity_dtx():
        try:
            with quiet():
                RTCAugmenterB4(RTCAugConfigB4(**dict(_b4_fields(b4_config_from_args(base_args())), dtx_p=0.3,
                                                     dtx_fill="white")))
            rej_aug = False
        except ValueError:
            rej_aug = True
    ok &= report("dtx_rejects_bad_fill_and_p", rej_parse and rej_p and rej_dsp and rej_aug,
                 f"argparse fill {rej_parse}, validate p {rej_p}, check_fill {rej_dsp}, augmenter {rej_aug}")
    with quiet():
        off = build_rtc_augmenter_b4(base_args(aug_dtx_fill="white"))
    ok &= report("dtx_fill_unread_when_p0", type(off) is RTCAugmenterB3, f"{type(off).__name__}")

    a = base_args(no_codec_aug=True, aug_dtx_p=0.3)
    head = b4_header_lines(a)
    ok &= report("dtx_header_matches_expect_regex", head == [DTX_HEADER] and re.match(DTX_HEADER_RE, head[0]) is not None,
                 f"{head}", head)
    with identity_dtx():
        with quiet():
            ref = build_rtc_augmenter_b3(a)
            aug = build_rtc_augmenter_b4(a)
    ok &= report("dtx_describe_and_meta_keys", type(aug) is RTCAugmenterB4
                 and aug.describe() == ref.describe() + " | wrtc_dtx p=0.3 fill=mix"
                 and aug.meta_keys == ref.meta_keys + list(DTX_KEYS),
                 f"describe ...{aug.describe()[-50:]}; meta tail {aug.meta_keys[-8:]}",
                 [f"describe suffix: {aug.describe()[len(ref.describe()):]}", f"meta keys + {aug.meta_keys[len(ref.meta_keys):]}"])

    if importlib.util.find_spec("webrtcvad") is None:
        skip("dtx_selftest / dtx_lever_real_gate", "package webrtcvad missing")
        return ok

    # the selftest (plan 2.2 equivalence test): every named assert reported with its measured values
    t0 = time.process_time()
    vals = dsp_dtx.measure()
    t_self = time.process_time() - t0
    runs = vals["runs"]
    grid = [f"mode {m}: " + " | ".join(f"f{f} " + " ".join(f"h{h:g} {r['gated_frac']:.3f}" for r in runs
                                                           if r["mode"] == m and r["frame"] == f for h in [r["hang"]])
                                       for f in dsp_dtx.FRAMES_MS) for m in dsp_dtx.MODES]
    lv = np.array([r["cng_level_db"] for r in runs], dtype=np.float64)
    col = [r["cng_colour_db"] for r in runs]
    lines = {
        "shape_finite": [f"{sum(r['shape_finite'] for r in runs)}/{len(runs)} runs: length == input and finite "
                         f"(burst zero / cng 0 dB / cng -6 dB / silent-gap cng / tone)"],
        "gated_frac": [f"burst clip (tone 220 Hz complex -26 dBFS, gaps one-pole noise -50 dBFS, gap share "
                       f"{vals['gap_share']:.4f}): gated fraction per frame (ms) and hangover (ms), bounds "
                       f"[{dsp_dtx.GATED_LO}, {vals['gap_share']:.4f}]"] + grid,
        "speech_kept": [f"changed tone samples per run (zero + cng fill): {[r['speech_changed'] for r in runs]}  (all 0)"],
        "zero_exact": [f"max |fully gated sample| {max(r['zero_max'] for r in runs):g} over "
                       f"{min(r['n_full'] for r in runs)}..{max(r['n_full'] for r in runs)} samples per run  (exactly 0)"],
        "cng_level": [f"CNG RMS - target (dB) over fully gated samples: level 0 dB {lv[:, 0].min():+.3f}..{lv[:, 0].max():+.3f}, "
                      f"level -6 dB {lv[:, 1].min():+.3f}..{lv[:, 1].max():+.3f}, digital-silence gaps (target "
                      f"{runs[0]['floor_target_dbfs']} dBFS floor) {lv[:, 2].min():+.3f}..{lv[:, 2].max():+.3f}  "
                      f"(within +-{dsp_dtx.CNG_TOL_DB})"],
        "cng_colour": [f"gap content (0.1-1 kHz)/(2-6 kHz) {vals['content_ratio_db']:+.2f} dB; fill - content "
                       f"{min(col):+.3f}..{max(col):+.3f} dB (within +-{dsp_dtx.COLOUR_TOL_DB}; white noise would be "
                       f"about -16 dB off); white fallback in {sum(r['white_fallback'] for r in runs)} runs"],
        "tone_unchanged": [f"4 s tone clip, changed samples per run: {[r['tone_changed'] for r in runs]}  (all 0)"],
    }
    for name in dsp_dtx.ASSERTS:
        try:
            dsp_dtx.verify(vals, (name,))
            ok &= report(f"dtx_selftest[{name}]", True, values=lines[name])
        except AssertionError as exc:
            ok &= report(f"dtx_selftest[{name}]", False, str(exc), lines[name])
    print(f"    selftest CPU {t_self:.2f} s (incl. the scipy import on first use)", flush=True)

    # the lever with the real gate, p = 0.3, fill mix, no codec: outputs = the documented draw order replayed on the
    # B3 outputs (bit for bit), parent counts = B3, rate in range
    a = base_args(no_codec_aug=True, aug_dtx_p=0.3)
    with quiet():
        ref = build_rtc_augmenter_b3(a)
        aug = build_rtc_augmenter_b4(a)
        ref2 = build_rtc_augmenter_b3(a)
    clips = _synthetic_clips(n_clips, np.random.default_rng(31))
    got = [aug(c.copy(), 16000) for c in clips]
    want, wc = _dtx_replay(clips, ref, 0.3, "mix")
    same_ref = sum(int(np.array_equal(ref2(c.copy(), 16000), g)) for c, g in zip(clips, got))
    match = sum(int(np.array_equal(g, w)) for g, w in zip(got, want))
    dc = _dtx_counts(aug.stage_counts)
    n_proc = dc["wrtc_dtx_zero"] + dc["wrtc_dtx_cng"]
    cond = (match == n_clips and dc == wc and _dtx_parent_counts(aug.stage_counts) == ref2.stage_counts
            and 35 <= n_proc <= 85 and same_ref >= n_clips - n_proc and dc["fail_wrtc_dtx"] == 0
            and all(len(g) == len(c) for g, c in zip(got, clips)))
    ok &= report("dtx_lever_real_gate_replay", cond,
                 f"replay match {match}/{n_clips}; counts {dc} vs replay {wc}; processed {n_proc}; identical to B3 {same_ref}",
                 [f"p=0.3 fill=mix on {n_clips} synthetic clips: {match}/{n_clips} outputs = the rng4 replay "
                  f"(random() < p, draw_params, gate, finalize); processed {n_proc} (rate {n_proc / n_clips:.3f}); "
                  f"counts {dc}; {same_ref} clips bit-identical to B3; parent counts == B3"])

    # each fixed fill, p = 1 on 20 clips: only that fill, replay match
    for fill in ("zero", "cng"):
        a = base_args(no_codec_aug=True, aug_dtx_p=1.0, aug_dtx_fill=fill)
        with quiet():
            ref = build_rtc_augmenter_b3(a)
            aug = build_rtc_augmenter_b4(a)
        clips = _synthetic_clips(20, np.random.default_rng(37))
        got = [aug(c.copy(), 16000) for c in clips]
        want, wc = _dtx_replay(clips, ref, 1.0, fill)
        match = sum(int(np.array_equal(g, w)) for g, w in zip(got, want))
        dc = _dtx_counts(aug.stage_counts)
        ok &= report(f"dtx_fill_{fill}_only", match == 20 and dc == wc and dc[f"wrtc_dtx_{fill}"] == 20,
                     f"match {match}/20; counts {dc}", [f"fill={fill}, p=1: {dc}"])

    # chain with the codec: DTX -> codec on 5 clips (fixed profiles), p = 1
    a = base_args(no_codec_aug=False, aug_dtx_p=1.0)
    with quiet():
        ref = build_rtc_augmenter_b3(a)
        aug = build_rtc_augmenter_b4(a)
    shape_ok = True
    for c in _synthetic_clips(5, np.random.default_rng(41)):
        ref(c.copy(), 16000)
        y = aug(c.copy(), 16000)
        shape_ok &= len(y) == len(c) and bool(np.all(np.isfinite(y)))
    codec = {k: v for k, v in aug.stage_counts.items() if k.startswith("codec:")}
    dc = _dtx_counts(aug.stage_counts)
    ok &= report("dtx_then_codec_chain", shape_ok and dc["wrtc_dtx_zero"] + dc["wrtc_dtx_cng"] == 5
                 and codec == {k: v for k, v in ref.stage_counts.items() if k.startswith("codec:")} and sum(codec.values()) == 5,
                 f"{aug.stage_counts}", [f"p=1.0 with the codec: {dict(sorted(aug.stage_counts.items()))}"])

    # chain with apm_ns (identity NS), both p = 1: NS draws first, then DTX, on the one rng4 stream
    a = base_args(no_codec_aug=True, aug_apm_p=1.0, aug_dtx_p=1.0)
    with identity_ns():
        with quiet():
            ref = build_rtc_augmenter_b3(a)
            aug = build_rtc_augmenter_b4(a)
        clips = _synthetic_clips(20, np.random.default_rng(43))
        got = [aug(c.copy(), 16000) for c in clips]
    want, wc = _dtx_replay(clips, ref, 1.0, "mix", apm_first=True)
    match = sum(int(np.array_equal(g, w)) for g, w in zip(got, want))
    n_ns = sum(aug.stage_counts.get(f"wrtc_ns{d}", 0) for d in APM_LEVELS)
    ok &= report("apm_then_dtx_chain_order", aug.levers_on == ["apm_ns", "dtx_gate"] and match == 20 and n_ns == 20
                 and _dtx_counts(aug.stage_counts) == wc,
                 f"levers {aug.levers_on}; replay match {match}/20; wrtc_ns {n_ns}; dtx {_dtx_counts(aug.stage_counts)} vs {wc}",
                 [f"apm_ns (identity) + dtx_gate, p=1: levers {aug.levers_on}; {match}/20 outputs = replay "
                  f"(apm random, integers, then dtx random, params, gate)"])

    # edge cases: digital silence, a clip shorter than a frame, 8 kHz, an unsupported rate -> fail_wrtc_dtx
    sil = np.zeros(16000, np.float32)
    yz, iz = dsp_dtx.gate(sil, 3, 20, 100.0, "zero")
    yc, ic = dsp_dtx.gate(sil, 3, 20, 100.0, "cng", 0.0, np.random.default_rng(0))
    sil_cng_db = 20 * np.log10(float(np.sqrt(np.mean(yc.astype(np.float64) ** 2))) + 1e-30)
    ys, _ = dsp_dtx.gate(np.full(100, 0.01, np.float32), 1, 30, 40.0, "zero")
    y8, _, _ = dsp_dtx.wrtc_dtx(_synthetic_clips(1, np.random.default_rng(1), sr=8000)[0], np.random.default_rng(2), 8000)
    a = base_args(no_codec_aug=True, aug_dtx_p=1.0)
    with quiet():
        ref = build_rtc_augmenter_b3(a)
        aug = build_rtc_augmenter_b4(a)
    clips = _synthetic_clips(5, np.random.default_rng(47), sr=22050)
    same = sum(int(np.array_equal(ref(c.copy(), 22050), aug(c.copy(), 22050))) for c in clips)
    edge = {"silence_zero_all_gated": bool(iz["gated_frac"] == 1.0 and not np.any(yz)),
            "silence_cng_floor_white": bool(ic["white_fallback"] and abs(sil_cng_db - dsp_dtx.CNG_FLOOR_DBFS) <= 0.1),
            "short_clip_len": len(ys) == 100, "sr8000_len": len(y8) == 16000,
            "sr22050_fail_passthrough": same == 5 and aug.stage_counts.get("fail_wrtc_dtx", 0) == 5}
    ok &= report("dtx_edge_cases", all(edge.values()), f"{edge}",
                 [f"{edge}; digital silence + cng -> {sil_cng_db:.2f} dBFS"])
    return ok


def check_dtx_balance(per_cell=50, seed=0):
    """
    Plan 2.2 (risk: class-dependent realised gating): realised gated fraction per class on 200 bona + 200 spoof
    training clips (50 per class x src x lang cell; whole clips after load + loudness norm) in two conditions --
    (none) the loaded clip, (stage) after the B3 augmenter's acoustic stage (its own seeded rng, p_apply as
    configured) -- with the lever's per-clip parameters (mode / frame / hangover, one draw per clip, shared by both
    conditions) from a seeded stream, plus each mode forced (frame 20 ms, hangover 120 ms). Prints mean / quartiles
    / share gated / bins per class and bona - spoof; `DTX_BALANCE WARN` when the class means of the drawn-parameter
    fraction differ by > 0.05 (either condition). A warning, not a failure.
    """
    import importlib.util
    import librosa
    from utils.data_utils import build_loudness, read_protocol
    from b3wave.augment import build_rtc_augmenter_b3
    from b3wave.data import utt_meta

    if importlib.util.find_spec("webrtcvad") is None:
        skip("dtx_balance", "package webrtcvad missing")
        return True
    args = base_args()
    if not Path(args.train_protocol).is_file() or not Path(args.train_data_path).is_dir():
        skip("dtx_balance", f"training data missing ({args.train_protocol})")
        return True
    from b4wave import dsp_dtx
    files, labels = read_protocol(args.train_protocol, require_label=True)
    cells = {}
    for u in files:
        m = utt_meta(u)
        cells.setdefault(("bona" if labels[u] == 1 else "spoof", m["src"], m["lang"]), []).append(u)
    rng = np.random.default_rng(seed)
    pick = {k: [cells[k][i] for i in sorted(rng.choice(len(cells[k]), size=min(per_cell, len(cells[k])), replace=False))]
            for k in sorted(cells)}
    with quiet():
        b3 = build_rtc_augmenter_b3(args)
    loud = build_loudness(args)
    srng = np.random.default_rng(seed + 1)          # the acoustic stage's draws
    prng = np.random.default_rng(seed + 2)          # the lever's per-clip parameters
    res = {}                                        # (cls, cond, key) -> [fraction]; key "drawn" or "mode<m>"
    t0 = time.time()
    for (cls, src, lang), utts in pick.items():
        for u in utts:
            audio, sr = librosa.load(str(Path(args.train_data_path) / u), sr=16000)
            audio = audio.astype(np.float32)
            if loud is not None:
                audio = loud(audio, sr)
            staged = b3._stage(audio, srng, 16000)
            p = dsp_dtx.draw_params(prng, "zero")
            for cond, x in (("none", audio), ("stage", staged)):
                res.setdefault((cls, cond, "drawn"), []).append(
                    float(dsp_dtx.gated_mask(x, p["mode"], p["frame_ms"], p["hang_ms"]).mean()))
                for m in dsp_dtx.MODES:
                    res.setdefault((cls, cond, f"mode{m}"), []).append(float(dsp_dtx.gated_mask(x, m, 20, 120.0).mean()))
    n_cls = {c: sum(len(v) for k, v in pick.items() if k[0] == c) for c in ("bona", "spoof")}
    lines = [f"{sum(n_cls.values())} training clips ({n_cls}; {per_cell} per class x src x lang cell), whole clips after "
             f"load + loudness norm; conditions none / stage (B3 acoustic stage {sorted(b3.stages)}); "
             f"{time.time() - t0:.0f}s"]
    warns = []
    for cond in ("none", "stage"):
        for key in ["drawn"] + [f"mode{m}" for m in dsp_dtx.MODES]:
            st = {}
            for cls in ("bona", "spoof"):
                v = np.asarray(res[(cls, cond, key)], np.float64)
                bins = {b: int(sum(dsp_dtx.gated_bin(x) == b for x in v)) for b in dsp_dtx.BINS}
                st[cls] = (float(v.mean()), np.percentile(v, [25, 50, 75]).round(3).tolist(), float((v > 0).mean()), bins)
            diff = st["bona"][0] - st["spoof"][0]
            what = "drawn params (mode 1-3, frame 10/20/30, hang 40-200)" if key == "drawn" else f"{key} frame 20 hang 120"
            lines.append(f"{cond:5s} {what}: bona mean {st['bona'][0]:.3f} q {st['bona'][1]} gated>0 {st['bona'][2]:.2f} "
                         f"bins {st['bona'][3]} | spoof mean {st['spoof'][0]:.3f} q {st['spoof'][1]} gated>0 "
                         f"{st['spoof'][2]:.2f} bins {st['spoof'][3]} | bona - spoof {diff:+.3f}")
            if key == "drawn" and abs(diff) > DTX_BALANCE_WARN:
                warns.append(f"DTX_BALANCE WARN cond={cond}: bona - spoof mean realised gated fraction {diff:+.3f} "
                             f"(|diff| > {DTX_BALANCE_WARN:g})")
    ok = report("dtx_balance", bool(res), "no clip measured", lines)
    for w in warns:
        print(w, flush=True)
    if not warns:
        print("    no DTX_BALANCE WARN", flush=True)
    return ok


