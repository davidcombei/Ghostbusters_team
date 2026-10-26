# b4wave: the B4 research-cycle package (rdloop/B4)

Additive package for rdloop cycle B4 (focus: augmentation; plan `rdloop/B4/plan_agreed.md`, approved levers
`apm_ns`, `dtx_gate`, `opus_plc`; added after the cycle: `tel_codec` for run B4e, telephony codecs from
`origin/feature/model-updates`). **Nothing in `SLS_setup/b3wave/`, `main_*.py`, `utils/`, `model/` or the
existing `scripts/*` is changed**; old runs stay reproducible from their own entry points. No lever changes the
model: checkpoints are plain `ModelSLS` state_dicts, scored with the `sls_local17` profile and
`--loudness_norm --loudness_norm_lufs -23` as for B2f (no `b4_local_eval.py` / eval profile needed).

```
cd SLS_setup
python -u -m b4wave.train <b3wave / main_train.py arguments> [B4 flags]     # trainer
python -m b4wave.checks cpu                                                 # CPU self-checks (STATIC step)
python -m b4wave.checks gpu --ckpt <B2f epoch_8 checkpoint>                 # [GPU] regression checks
python -m b4wave.checks overfit --lever <id>                                # params_added levers (none: SKIP)
python ../scripts/b4_dsp_slices.py --run <arm local_eval/readout> --ref <B2f same readout> ...   # mechanism readout
```

## Reproduction statement

`python -u -m b4wave.train` with B2f's command line (`exp/B2f_*/config.yaml: command`, base entry module
`b3wave`) trains exactly what `python -u -m b3wave.train` trains:

- every B4 flag defaults to off; with all of them off `build_rtc_augmenter_b4(args)` returns
  `build_rtc_augmenter_b3(args)` (the B3 object itself), so the data path, the numpy / python / torch RNG draws,
  `describe()`, `meta_keys` and the augmenter keys of `config.yaml` are b3wave's (`checks augmenter`, `checks data`);
- a forced `RTCAugmenterB4` with no lever on is bit-identical to the B3 augmenter on 200 clips without and 10 with
  the codec (fixed and random profiles), and never creates its B4 stream (`checks augmenter`);
- no lever package is imported and no libopus handle is opened with the levers off (`checks imports`);
- the model is b3wave's `ModelSLSB3` (same state_dict keys, same logits as `ModelSLS`; `checks gpu`) and the
  optimizer / schedule / EMA / checkpointing / resume / pruning code is the b3wave copy, unchanged;
- the argparse result on the base argv equals b3wave's for all 100 b3wave arguments (`checks flags`).

What differs from a b3wave run with the levers off (bookkeeping only): two header lines after `b3wave: <version>`
(`b4wave: <version>` and `b4wave sha256: <first 12 of module_sha256>`) and three `config.yaml` keys
(`b4wave_version`, `module_sha256`, `b4_levers: []`). The `Epoch N/M TrainLoss=...`, `Epoch N LocalWF1=...` and
`Epoch N B3 ...` lines and the `metrics.jsonl` rows are unchanged.

## Flags (group "B4 wave" in `train.py`; default = B2f behaviour)

| flag | default | what it changes | file | check | status |
|---|---|---|---|---|---|
| `--aug_apm_p P` | 0.0 | per-clip probability (rng4) of one WebRTC NS pass (pywebrtc_audio standalone `NoiseSuppressor`, no HPF / AGC / AEC) after the acoustic stage, before DTX and the codec | `dsp_apm.py`, `augment.py` | `checks apm`, `apm_balance`, `augmenter`, `timing` | implemented (apm_ns) |
| `--aug_apm_ns_levels DB ...` | 6 12 18 21 | NS targets in dB, drawn uniformly per processed clip (pywebrtc_audio levels 0-3); read only when `--aug_apm_p > 0` | `dsp_apm.py` | `checks apm` | implemented (apm_ns) |
| `--aug_dtx_p P` | 0.0 | per-clip probability (rng4) of webrtcvad DTX gating (mode 1-3, 10/20/30 ms frames, hangover 40-200 ms), after the NS slot, before the codec | `dsp_dtx.py`, `augment.py` | `checks dtx`, `dtx_balance`, `augmenter`, `timing` | implemented (dtx_gate) |
| `--aug_dtx_fill {mix,zero,cng}` | mix | fill of gated frames: exact zeros, order-10 LPC comfort noise (−6..0 dB, floor −70 dBFS), or 50/50 per clip; read only when `--aug_dtx_p > 0` | `dsp_dtx.py` | `checks dtx` | implemented (dtx_gate) |
| `--aug_opus_plc_p P` | 0.0 | probability (rng4) that a codec call takes the in-process libopus path with Gilbert-Elliott loss + decoder PLC (48 kHz decode + `resample_poly`) instead of ffmpeg, with the parent's app / bitrate / cutoff / frame draws | `opus_plc.py`, `augment.py` | `checks opus_plc`, `codec_draw_order`, `augmenter`, `timing` | implemented (opus_plc) |
| `--aug_opus_loss LO HI` | 0.01 0.20 | per-clip target packet-loss range, 0 ≤ lo ≤ hi < 0.5; `0 0` = in-process codec without loss (attribution control); read only when `--aug_opus_plc_p > 0` | `opus_plc.py` | `checks opus_plc` | implemented (opus_plc) |
| `--aug_opus_burst LO HI` | 1.0 4.0 | mean loss-burst length range in packets, 1 ≤ lo ≤ hi; read only when `--aug_opus_plc_p > 0` | `opus_plc.py` | `checks opus_plc` | implemented (opus_plc) |
| `--aug_tel_p P` | 0.0 | probability (own stream `default_rng([seed, pid, 0xB4E])`) that a codec call goes through a telephony codec (PyAV in-memory round trip) instead of the Opus app profile; the parent's Opus draws are made either way; exclusive with `--aug_opus_plc_p` | `tel_codec.py`, `augment.py` | `checks tel`, `augmenter`, `timing` | implemented (tel_codec, B4e) |
| `--aug_tel_codecs CODEC ...` | g723_1 amr_nb | codecs drawn uniformly per picked call, bitrate drawn per call; `g711_ulaw` / `g711_alaw` / `g722` / `g726` are accepted by argparse but refused at start-up by the reserve; read only when `--aug_tel_p > 0` | `tel_codec.py` | `checks tel` | implemented (tel_codec, B4e) |

Wave arms (plan): `--aug_apm_p 0.3`, `--aug_dtx_p 0.3`, `--aug_opus_plc_p 0.3` (one flag off B2f each).
B4e (after the wave): B4a's flags + `--aug_tel_p 0.25 --aug_tel_codecs g723_1 amr_nb` (`scripts/run_B4e.sh`).
Dependencies (owner-installed at the gate, never by the module): `pywebrtc_audio` (apm_ns), `webrtcvad`
(dtx_gate), `av==15.1.0` (tel_codec: the 19.x wheels have no AMR-NB encoder); `opus_plc` uses the system libopus 1.3.1 through `ctypes` (`/usr/lib/x86_64-linux-gnu/libopus.so.0`;
a container restart can wipe it with ffmpeg: optional `conda install -n rtc-sdd -c conda-forge libopus`).

## Lever `apm_ns` (plan §2.1): WebRTC NS between the noise stage and the codec

- **Where**: `RTCAugmenterB4._lever_apm_ns` (`augment.py`, slot `post_dsp`, first in the chain) calls
  `dsp_apm.wrtc_ns(out, level_db)` with probability `--aug_apm_p` (`rng4.random() < p`, then the level
  `rng4.integers(len(levels))`; nothing from the parent `rng`), and the result goes through `finalize(·, out)`.
  Class-independent, independent of the stage drawn, also on clips without a noise stage.
- **Backend** (`dsp_apm.py`): `pywebrtc_audio.NoiseSuppressor(sample_rate=16000, num_channels=1, level=0..3)`, the
  standalone NS (HPF / AGC / AEC exist only on `AudioProcessor`, which is not used). Per clip: clip to [−1, 1],
  int16; a fresh suppressor primed on the first min(0.5 s, len) (output discarded); the whole clip from the start
  in 10 ms frames (`_process_frame`); back to float32. **Latency**: WebRTC NS delays its output by 96 samples
  (6 ms; 256-point FFT on 160-sample frames; measured at every level); `wrtc_ns` runs 96 extra zero samples and
  drops the first 96 output samples, so the output is sample-aligned with the input (selftest `[aligned]`).
- **Start-up selftest** (when on; `B4 APM NS selftest: PASS ...` line with the measured values; an
  `AssertionError("apm selftest [<name>] level=<dB>: ...")` stops the run): `[shape_finite]`, `[no_gain]`
  (one-sided, 150 Hz harmonic complex under a 4 Hz on / off envelope at −30 dBFS: out RMS ≤ in RMS + 0.5 dB),
  `[monotone]` (white noise −30 dBFS: removed dB non-decreasing in level within 1 dB, removed(21) > removed(6)),
  `[no_hpf]` (512-pt Hann Welch over the 5.5 s after priming: attenuation in 31.25-93.75 Hz within 3 dB of the
  1-4 kHz attenuation), `[aligned]` (output lag 0). Measured on pywebrtc_audio 0.2.0 (`checks apm`): no_gain
  −6.02 / −12.38 / −18.40 / −21.47 dB, removed 5.99 / 11.89 / 17.53 / 20.83 dB, no_hpf diff −0.09 / −0.49 /
  −0.83 / −0.92 dB, lag 0 (levels 6 / 12 / 18 / 21). The `[aligned]` lag search uses sums of elementwise
  products, not `np.dot` (review F1.2: the dot product woke numpy's multi-threaded OpenBLAS, whose spinning threads
  billed ~3× the wall time to the process and slowed the next calls; now `measure()` CPU = wall, ~0.5 s).
- **Counters** (`meta_keys`, in the `Epoch N B3 aug` line and the per-class B4 line): `wrtc_ns6`, `wrtc_ns12`,
  `wrtc_ns18`, `wrtc_ns21`, `fail_wrtc_ns` (an exception in the backend: the clip passes through unchanged; the
  first one per process prints `[b4wave] wrtc_ns failed (<exc>); passing through`, see Telemetry).
- **Header** (on only): `B4 APM NS: on p=0.3 levels_db=[6, 12, 18, 21] backend=pywebrtc_audio.NoiseSuppressor
  hpf=off agc=off`; `describe()` gains ` | wrtc_ns p=0.3 levels_db=[6, 12, 18, 21]`; reserve: `wrtc_ns`.
- **Off** (`--aug_apm_p 0.0`, the default): the B3 object, no rng4, `pywebrtc_audio` / `b4wave.dsp_apm` never
  imported, `--aug_apm_ns_levels` unread (`checks augmenter`, `imports`, `apm_levels_unread_when_p0`).

## Lever `dtx_gate` (plan §2.2): webrtcvad DTX gating with zero or LPC comfort-noise fill

- **Where**: `RTCAugmenterB4._lever_dtx_gate` (`augment.py`, slot `post_dsp`, after `apm_ns`) with probability
  `--aug_dtx_p` (`rng4.random() < p`) calls `dsp_dtx.wrtc_dtx(out, rng4, sr, fill)`; the result goes through
  `finalize(·, out)`. Class-independent; after the NS slot, before the codec; nothing from the parent `rng`.
- **Per-clip draws** (`dsp_dtx.draw_params`, all from rng4, in this order): mode `integers(1..3)`, frame
  `(10, 20, 30)[integers(3)]` ms, hangover `uniform(40, 200)` ms, fill (`mix`: zero if `random() < 0.5` else cng),
  CNG level `uniform(−6, 0)` dB (cng only); a CNG fill then draws its excitation `standard_normal(len + 512)` when a
  sample is gated. `checks dtx` replays this order bit for bit on the B3 outputs.
- **Gate** (`dsp_dtx.gate`): `webrtcvad.Vad(mode).is_speech` on consecutive int16 frames (the last partial frame
  zero-padded); the sample-level speech mask is dilated forward by round(hang · sr / 1000) samples; the other samples
  are gated. Fill weight: raised cosine over the first / last 5 ms (80 samples) of every gated run (distance to the
  nearest kept sample), 1 inside; clip edges are not faded. Kept samples are the input bit for bit; fully gated
  samples are exactly the fill (exact zeros for `zero`). Length unchanged; no gated sample → the input itself.
- **CNG** (fill `cng`): order-10 LPC of the gated content (autocorrelation lags 0-10 with kept samples zeroed,
  lag 0 × (1 + 1e-6), `scipy.linalg.solve_toeplitz`; the predictor's stability checked), synthesis
  `lfilter([1], A, white)` after 512 discarded samples, scaled so its RMS over the gated samples = gated-content
  RMS × 10^(level/20), floor −70 dBFS. Digital-silence content (or an unstable predictor) → flat spectrum at the
  target. Autocorrelation lags are elementwise sums, not `np.dot`: the dot product wakes numpy's multi-threaded
  OpenBLAS, whose spinning threads cost ~8× the CPU per clip in a loader process.
- **webrtcvad overhang**: webrtcvad keeps its own internal hangover after speech (about 100-330 ms by mode / frame
  in this build); the drawn hangover adds to it, so only pauses longer than ~0.2-0.5 s are gated.
- **Start-up selftest** (when on; `B4 DTX gate selftest: PASS ...` line; an `AssertionError("dtx selftest [<name>]
  mode=.. frame=..ms hang=..ms: ...")` stops the run). Grid: modes 1-3 × frames 10/20/30 × hangover 40 / 200 ms on
  a burst clip (0.96 s gap + 4 × (0.96 s tone + 0.96 s gap); 220 Hz 20-harmonic complex at −26 dBFS, gaps one-pole
  (0.9) low-passed noise at −50 dBFS; segments on the 60 ms grid) and a 4 s tone clip. Named asserts:
  `[shape_finite]`, `[gated_frac]` (burst clip gated fraction in [0.20, gap share 0.556]), `[speech_kept]` (every
  tone sample unchanged), `[zero_exact]` (fully gated samples exactly 0), `[cng_level]` (CNG RMS within ±3 dB of
  the target at level 0 / −6 dB and at the −70 dBFS floor on digital-silence gaps), `[cng_colour]` ((0.1-1 kHz) /
  (2-6 kHz) band ratio of the fill within 3 dB of the gap content's; white noise would be ~16 dB off),
  `[tone_unchanged]` (tone clip bit-identical). Measured (`checks dtx`, webrtcvad 2.0.14.post1): gated fraction
  0.343-0.491, 0 tone samples changed, zero max 0, CNG level −0.015..+0.020 dB, colour −0.14..+0.01 dB; ~1.2 CPU-s.
- **Counters** (`meta_keys`, in the `Epoch N B3 aug` line and the per-class B4 line), per processed clip: the fill
  used `wrtc_dtx_zero` / `wrtc_dtx_cng`, the realised gated-fraction bin `dtx_g0` (< 1 %) / `dtx_g1_20` (1-20 %) /
  `dtx_g20_40` (20-40 %) / `dtx_g40p` (≥ 40 %), and `fail_wrtc_dtx` (an exception, e.g. an unsupported sample
  rate: the clip passes through unchanged and is not counted as processed; the first one per process prints
  `[b4wave] wrtc_dtx failed (<exc>); passing through`).
- **Header** (on only): `B4 DTX gate: on p=0.3 fill=mix modes=[1, 2, 3] frames_ms=[10, 20, 30]
  hangover_ms=40..200`; `describe()` gains ` | wrtc_dtx p=0.3 fill=mix`; reserve: `wrtc_dtx`.
- **Off** (`--aug_dtx_p 0.0`, the default): the B3 object, no rng4, `webrtcvad` / `b4wave.dsp_dtx` never imported,
  `--aug_dtx_fill` unread (`checks augmenter`, `imports`, `dtx_fill_unread_when_p0`).
- **Balance** (`checks dtx_balance`, 200 bona + 200 spoof training clips): mean realised gated fraction with the
  drawn parameters bona 0.058 / spoof 0.046 (loaded clip) and 0.045 / 0.032 (after the B3 acoustic stage), bona −
  spoof +0.012 / +0.013, no `DTX_BALANCE WARN` (bar 0.05).
- **Dose (code review F1.1; open issue for the owner).** The realised dose is far below the reserved eval op's.
  Training clips with ≥ 20 % of their samples gated (`dtx_g20_40` + `dtx_g40p`, `checks dtx_balance`): 5 / 4 of 200
  (bona / spoof, loaded clip) and 3 / 1 of 200 (after the B3 stage). The held-out `vad_gate`
  (`data/dev_noisy_sim/heldout_v1/meta.csv`, n 629, median 0.125, mean 0.233): bins g0 / g1_20 / g20_40 / g40p
  1.3 / 66.6 / 15.7 / 16.4 %, so 32.1 % of clips are ≥ 20 % gated. Cause: webrtcvad's internal overhang plus the
  drawn 40-200 ms hangover (above). The arm is trained as planned; the readout measures the realised dose and a
  pre-registered dose rule keeps an under-dosed null from reading as "refuted" (verdict rules below). A
  higher-dose variant (e.g. hangover U(0, 200) ms, or a higher `p`) is a training change and the owner's call.
- **Readout**: with `--lever dtx_gate`, `scripts/b4_dsp_slices.py` also prints
  - the **dose** (JSON `dtx_dose`): the arm's `dtx_g*` bin shares of its processed clips per class and pooled,
    summed over epochs from `metrics.jsonl["b4"]["aug_by_class"]` (a later row for the same epoch wins), with
    the processed share and `fail_wrtc_dtx`, next to the held-out `vad_gate` shares computed from
    `heldout_v1/meta.csv` (per class and pooled); dose_ratio and the dose rule;
  - the O2.3 report "VAD by fill" (vad:zero / vad:cng DiD and DiD*; "near-in-distribution" when zero-fill DiD* ≤
    −σ_zero (−0.42) with CNG-fill DiD* ≥ 0, "transfer claimable" when CNG-fill DiD* < 0; JSON
    `vad_fill_reading`); a report, not a verdict rule;
  - the **decision lines** (plan §2.2 falsifiers, JSON `dtx_gate_reading`): (2) clean F1 Δ < −0.5 or a noisy /
    VAD spoof-miss guard broken on either readout → Drop; (3) the shared under-fit falsifier → Drop; (1) VAD DiD*
    ≥ 0 (verdict readout) **and** drop_high − drop_low ≤ the panel mean (−10.2; full set, final) **and** no
    outcome escape (WF1 Δ ≥ +0.5 on merge_top5 **and** final) → Drop only at an adequate dose, else "under-dosed
    / unknown: owner decides a dose variant".

## Lever `opus_plc` (plan §2.3): in-process libopus with Gilbert-Elliott packet loss and decoder PLC

- **Where**: `RTCAugmenterB4._lever_opus_plc` (`augment.py`, slot `codec`), called when the parent's codec draw fires
  (`rng.random() < codec_p`, p = 1.0 in B2f). With probability `--aug_opus_plc_p` (`rng4.random() < p`) the clip
  takes the in-process path: `opus_plc.draw_codec_params(rng, apps)` makes the parent's app / kbps / cutoff / frame
  draws in exactly `codecs_augm`'s order, then `opus_plc.opus_loss_roundtrip(out, rng4, ...)`. Otherwise the clip
  goes through the original `utils.rtc_augment.codecs_augm` with the same parent draws. Either way the parent's
  `__call__` counts `codec:<app>` and applies `finalize(·, audio)`. The parent stream is identical to B2f's whatever
  the lever picks (`checks codec_draw_order`, `augmenter`).
- **Round trip** (`opus_plc.py`, libopus 1.3.1 through `ctypes`, loaded on first use): a 16 kHz mono encoder with
  the drawn profile, configured like ffmpeg 4.2's libopusenc: bitrate, complexity 10, VBR on (unconstrained = ffmpeg
  `-vbr on`), expected loss 0, max bandwidth from the cutoff (4000 / 6000 / 8000 Hz → NB / MB / WB), the profile's
  application; in-band FEC and DTX stay off. The clip, zero-padded to whole frames covering len + look-ahead (104
  samples), is encoded frame by frame. **The packets are byte-identical to those ffmpeg writes into its Ogg stream**
  (21 / 21 runs, `checks opus_plc`), so the lever changes only the receiver. Gilbert-Elliott mask over the packets:
  L ~ U(`--aug_opus_loss`), B ~ U(`--aug_opus_burst`), p_bg = 1/B, p_gb = L·p_bg/(1 − L), start state bad with
  probability L; one `rng4.random()` per packet. A 48 kHz libopus decoder decodes the received packets and conceals
  each lost one with the NULL-payload PLC call. The first 3 × 104 output samples are dropped, then
  `resample_poly(·, 1, 3)` and cut / zero-pad to the input length. Encoder and decoder are created per call and
  destroyed in `finally`. rng4 draws per picked clip: pick `random()`, `uniform(loss)`, `uniform(burst)`,
  `random(n_packets)`; `checks opus_plc` replays them draw by draw.
- **Alignment note**: ffmpeg 4.2 writes the Ogg pre-skip at the encoder rate (104) instead of 48 kHz (312), so the
  ffmpeg path (B2f) decodes ~69 samples (4.3 ms) late and drops the clip's last ~69 samples in `finalize`. The
  in-process path trims 3 × the look-ahead (plan) and is aligned with the input (lag −4..+1 samples). The
  zero-loss check therefore aligns before comparing; the attribution control arm covers this difference too.
- **Fallback**: any exception in the in-process path (e.g. a sample rate libopus does not support) counts
  `fail_opus_plc`, and the clip goes through ffmpeg with the same pre-drawn profile (`ffmpeg_opus`, bit-identical to
  `codecs_augm` for those draws).
- **Start-up checks** (when on; `B4 Opus PLC selftest: PASS ...` line; `AssertionError("opus selftest [<name>]
  ...")` or an `OSError` for a missing libopus stops the run, no silent pass-through): `[loads]` (version string),
  `[ctl_roundtrip]` (20 encoder configs = 7 apps × min / max kbps × cutoffs: bitrate and max bandwidth read back as
  set; VBR 1, VBR constraint 0, complexity 10, expected loss 0, FEC 0, DTX 0; look-ahead > 0), `[plc_not_zero]`
  (−20 dBFS tone, WB 24 kbps 20 ms, packets 20-24 lost: the first concealed 20 ms are not all-zero), `[shape_finite]`
  (every profile × frame × loss 0 / 0.2 on 16123- and 17-sample clips: input length, finite). Measured: lookahead
  104; first concealed 20 ms −21.5 dBFS (tone −20.0); 0.37 CPU-s. `init` also raises when the codec stage is off,
  with `--aug_codec_profiles random`, or with a loss / burst range outside the plan bounds.
- **Counters** (`meta_keys`, in the `Epoch N B3 aug` line and the per-class B4 line): `opus_plc` (in-process
  calls), the realised-loss bin `plc_l0_5` (< 5 %) / `plc_l5_10` (5-10 %) / `plc_l10p` (≥ 10 %), `fail_opus_plc`.
- **Header** (on only): `B4 Opus PLC: on p=0.3 loss=0.01..0.20 burst=1..4 decode=48k+resample_poly
  libopus=libopus 1.3.1`; `describe()` gains ` | opus_plc p=0.3 loss=0.01..0.20 burst=1..4 decode=48k`; reserve:
  filter `opus_plc`, codec `libopus`.
- **Off** (`--aug_opus_plc_p 0.0`, the default): the B3 object, every codec call is `codecs_augm`, no rng4,
  `b4wave.opus_plc` never imported, no libopus mapped (also with the codec on), `--aug_opus_loss` /
  `--aug_opus_burst` unread (not even validated) (`checks augmenter`, `imports`, `opus_ranges_unread_when_p0`).
- **Zero-loss attribution** (plan equivalence test; `checks opus_plc`): 3 training clips (seeded pick, −23 LUFS) ×
  7 profiles at the lowest kbps / first cutoff / longest frame (v2's profiles); best-integer-lag SNR; per profile on
  the energies pooled over the 3 clips, SNR(ffmpeg, in-process) ≥ max(15 dB, SNR(ffmpeg, input) + 5 dB). Measured:
  pooled 16.22 (wechat, margin +1.22) to 23.81 dB (dingtalk). On its own, one clip (offline/zh Train_038736) is
  below the bar at qq 14.78 and wechat 12.96 dB. That is the SILK-layer decoder / resampler residual listed in the
  plan's risks; the check prints every clip × profile value.
- **Cost** (`checks timing`): 0.0999 CPU-s per in-process call vs 0.243 for ffmpeg (self + children) on a 4.8 s
  clip (probe); the timing group prints the per-call comparison (smoke bar: in-process below ffmpeg).
- **Readout**: with `--lever opus_plc`, `scripts/b4_dsp_slices.py` also prints the plan §2.3 decision lines (JSON
  `opus_plc_reading`). These are falsifier (1) (online PLC DiD* ≥ 0 and no outcome escape on merge_top5 **and**
  final) and falsifier (2) (WF1 Δ < −0.5 with the shared under-fit signature). Then the must-not-move guards per
  readout: matched F1 Δ ≥ −0.5; matched qq / wechat bona recall Δ ≥ −6 (metrics.json `breakdowns`); PLC spoof
  misses ≤ the Poisson bound (26 / 25); clean F1 within 0.5; ep1 TrainAcc within 2. Finally falsifier (3): with
  `--attribution_control <control merge> <control final>`, the control's WF1 gain vs the arm's per readout ("credited
  to the decoder path" if the control has ≥ half of the arm's gain). All of this is a report: the lever verdict
  stays capped at "directional".

## Lever `tel_codec` (B4e): telephony codecs instead of the Opus app profile

- **Origin**: David's `utils/rtc_augment_expert.py::telephony_augm` (`origin/feature/model-updates`, 2283dfd), ported
  to `tel_codec.py` with the same codec table (G.711 µ / A-law, G.722, G.723.1, G.726, AMR-NB) and the same path:
  resample to the codec rate (s16) → encode into an in-memory WAV with PyAV → decode → resample back. The port raises
  on a backend error (the original printed and passed the clip through) so the lever can count it.
- **Allowed codecs**: `g723_1` (6.3 kbps) and `amr_nb` (4.75-12.2 kbps, `libopencore_amrnb`), 8 kHz narrowband.
  G.722 / G.726 are in `sim_heldout_v1` and G.711 in `sim_unseen_v1`: `_init_tel_codec` passes the selected names
  **and** their encoder names (`pcm_mulaw`, `pcm_alaw`, `g722`, `g726`) to `assert_training_ops`, so they raise before
  the first step (`checks tel`: `tel_reserved_refused(*)`).
- **Where**: codec slot (`RTCAugmenterB4._lever_tel_codec`), so exclusive with `opus_plc` (augmenter and
  `validate_b4_args`); needs the codec stage and `--aug_codec_profiles fixed`. Per codec call: the parent's codec
  draws first (`opus_plc.draw_codec_params(rng)`, `codecs_augm`'s order; pure numpy, libopus is not loaded), so the
  parent rng stays in step with the run without the lever; then the tel stream `default_rng([seed, pid, 0xB4E])`
  decides (`random() < p`), picks the codec uniformly and its bitrate. Picked: the PyAV round trip, counted as
  `codec:<name>`. Not picked: `opus_plc.ffmpeg_opus` on the parent's draws (bit-identical to `codecs_augm`). rng4
  is not touched, so the apm_ns draws are B4a's (`tel_parent_and_rng4_streams_untouched`: parent and rng4 states
  equal B4a's after each of 200 clips at p = 1.0).
- **Start-up selftest** (when on; `B4 TEL codec selftest: PASS ...`; an `AssertionError("tel selftest [<name>] ...")`
  stops the run): `[encoder]` (the encoder is in this PyAV build), `[shape_finite]`, `[length]` (output ≥ input),
  `[level]` (RMS within ±3 dB), `[narrowband]` (share of the spectral magnitude above 4.2 kHz < 0.05 on a probe
  clip with 0.25 there) at the lowest and highest bitrate. Measured on PyAV 15.1.0 (libavcodec 61.19.101): G.723.1
  +0.43 dB, 0.027; AMR-NB 4.75 / 12.2 kbps −0.03 / −0.77 dB, 0.020 / 0.025; output 960-1120 samples longer per 2 s
  (frame padding + codec delay), trimmed by `finalize`.
- **Cost** (`checks tel`): 0.062 CPU-s per telephony call vs 0.242 for the ffmpeg Opus path (self + children) on
  20 training clips (mean 6.4 s): the lever makes the loader cheaper.
- **Counters** (`meta_keys`): `codec:g723_1`, `codec:amr_nb`, `fail_tel_codec` (an exception in the round trip: the
  clip goes through ffmpeg with the parent's draws and is counted under its Opus app; the first one per process prints
  `[b4wave] tel_codec failed (<exc>); ffmpeg Opus with the parent's profile`).
- **Header** (on only): `B4 TEL codec: on p=0.25 of codec calls codecs=['g723_1', 'amr_nb'] backend=PyAV <v>
  libavcodec <v> (in-memory WAV round trip)`; `describe()` gains ` | tel_codec p=0.25 codecs=[...]`.
- **Off** (`--aug_tel_p 0.0`, the default): the B3 object (or B4a's augmenter with only apm_ns on), no `av` /
  `b4wave.tel_codec` import, `--aug_tel_codecs` unread (`checks augmenter`, `imports`).
- **Expected readout** (B4e vs B4a): `sim_heldout_v1` bona recall up (narrowband CELP / ADPCM family seen; G.722 /
  G.726 / GSM implementations unseen); `sim_matched_v1` (Opus apps, now 25 % fewer Opus clips) within −0.5 F1. The
  heldout set also contains `bandlimit` / `resample8k`, so part of a heldout gain can be narrowband familiarity
  rather than codec transfer.

## How a lever plugs in (contract for the lever parts)

1. **Flags**: literal `b4.add_argument("--aug_...", ...)` calls in the "B4 wave" group of `train.py`
   (`build_parser()`; exploop reads the allowed flags by grepping `add_argument` in that file), plan names and
   defaults exactly; value checks in `validate_b4_args`.
2. **Config**: the lever's fields on `RTCAugConfigB4` (`augment.py`), defaults = off.
3. **Registry**: one `register(LeverSpec(id, slot, prob_arg, prob_field, arg_fields, meta_keys, modules, filters,
   codecs, describe, header))` block in `augment.py`, in chain order (apm_ns, dtx_gate, opus_plc, tel_codec). A lever is on
   when its probability is > 0; `active_levers(args)` drives the builder, the header lines, `config.yaml`
   `b4_levers` and the per-class telemetry.
4. **Code**: `RTCAugmenterB4._lever_<id>` — slot `post_dsp`: `(out, sr) -> out`, drawing only from
   `self._get_rng4()`; slot `codec`: `(out, rng, sr) -> (out, app)`, making the parent's codec draws in
   `codecs_augm`'s order. Optional `_init_<id>()` (lazy import, selftest) runs at construction when the lever is on.
   Counters via `self._count(key)` with every key listed in `meta_keys`; failures counted as `fail_<op>`, the clip
   passes through.
5. **Header**: `LeverSpec.header(args)` returns the lever's ONE startup line (plan `expect_regex`), printed only
   when the lever is on (plan §2.0: levers off print nothing new).
6. **Checks**: the generic checks cover every registered lever automatically (plan defaults, registry vs parser,
   imports with the levers off, timing projection, `gpu` forward / backward with the lever on). Add the lever's own
   groups (selftest, balance, equivalence item 3 with the real lever, codec draw order, ...) to `CPU_GROUPS`, update
   the flag table above (status → implemented).

## Telemetry

- Header (always): `b4wave: <version>`, `b4wave sha256: <12>`; per lever that is on: its header line
  (`^B4 APM NS: on p=`, `^B4 DTX gate: on p=`, `^B4 Opus PLC: on p=`, `^B4 TEL codec: on p=`).
- `config.yaml` (always): `b4wave_version`, `module_sha256` (sha256 over the module's `.py` files sorted by
  name: name, NUL, sha256 hex of the bytes, newline per file; `b4wave.module_sha256()`), `b4_levers`
  (ids of the levers that are on).
- Per epoch: lever counters appear in the existing `Epoch N B3 aug ...` line (they are `meta_keys` of the B4
  augmenter); when a lever is on, `Epoch N B4 aug-by-class bona: n=<clips> <key>=<count> ... | spoof: n=... ...`
  and `metrics.jsonl["b4"] = {"levers": [...], "aug_by_class": {"bona": {"n", "counts"}, "spoof": {...}}}`
  (the meta vector summed per label; consistency mode counts both views of a clip).
- Lever failures (review F1.3; on only): an exception in a lever's backend is counted on every clip (`fail_wrtc_ns`,
  `fail_wrtc_dtx`, `fail_opus_plc`, `fail_tel_codec`), and the first one per process and op prints
  `[b4wave] <op> failed (<ExcType>: <exc>); passing through` (`ffmpeg fallback with the same profile` for
  `opus_plc`, `ffmpeg Opus with the parent's profile` for `tel_codec`). This is the pattern of the parent's `[rtc_augment] codec ... failed (...)` line, so the wave's early
  check (`failed \(`) catches a systematic worker-side lever failure at the first failing clip instead of the
  epoch-1 counters (`checks lever_failures`). Levers off: no such line can occur (the B3 object).

## Reserve (never in training)

The B4 op names `wrtc_ns`, `wrtc_dtx`, `opus_plc`, `tel_codec` and the codecs `libopus`, `g723_1`, `amr_nb`,
`libopencore_amrnb` (plus, at tel_codec start-up, the selected telephony names and encoders) are checked against
`b3wave/heldout_reserve.py` at start-up (`assert_training_ops`); the reserved `vad_gate`, `packet_loss`,
`afftdn`, `anlmdn`, G.722 / G.726 / GSM, speex, G.711 still raise (`checks reserve_names`). How heldout slices must
be labelled after B4 (G6, plan §2.0):

| heldout slice | label |
|---|---|
| `ns:*` | NS family seen (WebRTC NS); implementation unseen (`afftdn` / `anlmdn`) |
| `vad:*` | gating family and zero fill seen; detector (energy VAD, p10 + margin, hangover) and white-noise CNG colour unseen (CNG level overlaps); read `vad:zero` as near-in-distribution |
| `plc` | loss-pattern family seen (Gilbert-Elliott); codec + concealment unseen (repeat-PLC after G.722 / G.726 / GSM) |
| no-DSP, `agc:*`, `bw`, noise families, codecs | fully unseen |

`sim_unseen_v1` stays unread (music leak, audit G6).

## Mechanism readout and verdict rules (`scripts/b4_dsp_slices.py`; plan §1.1 + owner notes 2026-10-17 13:48)

```
python scripts/b4_dsp_slices.py --run SLS_setup/exp/<arm>/local_eval/epoch_8 \
    --placebo SLS_setup/exp/<other arm>/local_eval/epoch_8 ... [--lever apm_ns|dtx_gate|opus_plc] [--summary] \
    [--json_out <file>]
```
Defaults: `--ref_pair` = B2f `merge_top5` / `epoch_8`, `--ref` = the B2f readout of `--run`'s kind, the pinned null
panel (final and merge), `--run_pair` = the arm's `merge_top5` and highest `epoch_N`, `--lever` from the arm's
`config.yaml` `b4_levers`. Analyst use: once per arm on the final readout (primary) with the other arms as
`--placebo`; the wave SUMMARY re-runs it with `--summary` (verdicts final, not provisional). It reads only
`scores/sim_{matched,heldout}_v1.{txt,logits}`, `metrics.json`, `probe_report.json`, the sim `meta.csv`, the dev
source wavs (tertiles) and the arm's `logs/train.log` / `config.yaml`. CPU, ~10 s.

Statistic: per arm R, contrast c, readout r: Δᵢ = clip(logitᵢ(R), ±8) − clip(logitᵢ(B2f, r), ±8);
**DiD = mean_T(Δ) − mean_C(Δ)** with a 2,000× clip-bootstrap 95 % CI (negative = the treated slice moved toward
bona relative to its same-source control). Re-centred on the pinned **null panel** (B2b, B2c, B2d `epoch_10`, B2e,
B3a-B3e at their final readouts; merge side on `merge_top5`; B2a printed and flagged "failed run, excluded"):
DiD* = DiD − μ, z = DiD* / σ (σ with ddof 1). Bar = min(μ − 2σ, panel min, μ − 0.4·G).

Contrast verdict (full set, final readout primary):
- **shown** — (a) DiD ≤ bar; (b) CI upper bound < μ; (c) clip DiD − panel clip μ ≥ 0.4·D_final; (d) half-B DiD* < 0
  on both readouts (each on its own readout's half-B panel mean); (e) the arm's DiD on its own contrast is below the
  DiD of every other B4 arm on that contrast (placebo arms), same readout;
- **directional** — DiD* < 0 and (e), not all of (a)-(d);
- **inconclusive** — DiD* < 0 without (e);
- **refuted** — DiD* ≥ 0.

Owner amendments (critic O3.1-O3.3, adopted at the gate; readout only, no training change):
- **O3.1 run anomaly.** The script prints the arm's z on **every** contrast. The run is flagged **anomalous** if
  |z| > 3 on a non-target primary contrast (NS, VAD, PLC-online), or if the shared under-fit falsifier (ep1
  TrainAcc < 85.36 or ep8 TrainLoss > 0.0752, with heldout bona recall down) or the clean-F1 guard (clean F1 down
  > 0.5) fires. While flagged, a lever verdict that would be "shown" is reported as **"confounded (run anomaly)"**.
  The `apm_ns` lower-p retry needs "shown" **without** the flag.
- **O3.2 outcome thresholds.** Next to every heldout-F1 threshold the script prints the panel SD and the number of
  panel runs that clear it. In falsifier (1) an arm escapes on outcome only with heldout F1 Δ ≥ +0.5 on **both**
  readouts (merge_top5 and the final epoch, also for `opus_plc`). For the plan §4 combined-run recommendation,
  "positive outcome" means the wave's WF1 CI rule (`exploop/B3/wave.json` `rules`), not heldout F1 alone.
- **O3.3 missing arms.** At SUMMARY, (e) is evaluated on the B4 arms that exist, plus the `opus_plc` zero-loss
  control if it ran. With **no** placebo arm a contrast is capped at "directional". The placebo count is printed
  next to every verdict; verdicts are "provisional" until the wave SUMMARY re-issues them.

Lever-level rules: `apm_ns` mechanism shown = NS contrast shown (refuted: NS DiD ≥ +0.32); `dtx_gate` = criterion
(ii) (saturated-FA drop_high − drop_low ≥ +5 on the full set, final, and ≥ 0 on half B for both readouts; pinned
tertiles k ≤ 32 / 33-50 / ≥ 51 of 201 frames) **and** the VAD contrast at least directional ("strong" if shown);
`opus_plc` capped at "directional" (refuted: online PLC DiD ≥ −0.57), decided on outcome + guards + its zero-loss
control.

**`dtx_gate` dose rule** (code review F1.1, pre-registered before the smoke run; readout only, no training change).
dose_ratio = the arm's share of processed clips with ≥ 20 % of the samples gated (`dtx_g20_40` + `dtx_g40p`, both
classes pooled, summed over epochs from `metrics.jsonl["b4"]`) / the held-out `vad_gate`'s share (0.321 from
`heldout_v1/meta.csv`); per-class ratios are printed too. If dose_ratio < **0.25** ("under-dosed"), or the dose
cannot be read (no `b4` rows: "dose unknown"), the `dtx_gate` lever verdict cannot be "refuted": the refuted case
reads **"inconclusive (under-dosed)"** / "inconclusive (dose unknown)", and falsifier (1) reports "under-dosed:
owner decides a dose variant" instead of Drop. A positive result ("mechanism shown") still counts at any dose.
Falsifiers (2) and (3) drop at any dose. In falsifier (1) the plan's "with WF1 Δ < +0.5" is read on both
readouts (O3.2): the arm escapes on outcome only with WF1 Δ ≥ +0.5 on merge_top5 **and** final.

## Checks (`python -m b4wave.checks <target>`)

| target | what |
|---|---|
| `cpu` / `all` | every group below except `gpu` / `model` / `optim` / `overfit` (~5 min on the CPU) |
| `module` | `module_sha256` stable and over all `.py` files; trainer prints version + sha and writes them to config.yaml |
| `flags` | B4 group exists; implemented flags have the plan defaults, no unplanned flag; registry ↔ parser ↔ plan; flags grep-able by exploop; base-argv parse equals b3wave's; levers off on the base argv; defaults validate |
| `augmenter` | levers off → the B3 object; B4 config = B3 config; forced B4 bit-identical (200 + 10 + 10 clips, fixed / random profiles); stub levers: parent stream untouched by rng4, rng4 = `default_rng([seed, pid, 0xB4])`, re-seeded on a pid change, meta keys / describe / header, codec-slot dispatch, lever without RTC augmenter raises; plan equivalence item 3 per lever (`_equiv_<id>`): apm_ns with `apm_p = 1.0` and an identity NS → 200 clips without and 10 with the codec bit-identical to B3, parent counts equal, levels = rng4 draws; dtx_gate with `dtx_p = 1.0` and an identity gate (bit-identical, parameters = rng4 draws) and with the real webrtcvad gate (parent stage / codec counts = B3, every clip processed); opus_plc: levers off with the codec on in a fresh process (no `b4wave.opus_plc`, no libopus mapped), `opus_plc_p = 1.0` with the libopus round trip replaced by ffmpeg on the pre-drawn profile (10 clips bit-identical to B3, parent counts equal) and with the real libopus path (40 clips: parent stage:* and codec:<app> counts = B3, every codec call in-process, finite, input length) |
| `reserve_names` | plan op names not reserved; registered levers' ops planned and allowed; reserved ops still raise |
| `imports` | fresh process, levers-off build + 20 clips: no lever package / module imported, no libopus mapped |
| `trainer` | per-class meta sums, `B4 aug-by-class` format (not parsed as an epoch row), `train_epoch` bit-identical with the class telemetry on / off |
| `data` | `build_dataset_b4` = `build_dataset_b3` with the levers off (train subset + dev), 3 real clips identical after load + augment + codec |
| `apm` | apm_ns: levels outside {6, 12, 18, 21} rejected (argparse, `validate_b4_args`, `wrtc_ns`, augmenter); levels unread at p = 0; the selftest's named asserts `shape_finite` / `no_gain` / `monotone` / `no_hpf` / `aligned`, each with its measured values; the lever with the real NS on 200 synthetic clips (header = plan `expect_regex`, describe / meta keys, levels = rng4 draws, parent counts = B3, unprocessed clips bit-identical, no gain, no failure); NS → codec chain (SKIP of the backend parts when `pywebrtc_audio` is missing) |
| `apm_balance` | apm_ns, plan §2.1 / critic O1.5: 200 training clips (25 per class × src × lang cell) under no stage and a forced training-pool noise stage at U(5, 20) dB (200 + 200 runs) × 4 levels: removed dB over the whole clip and over the input's low-energy 10 ms frames (< p95 − 30 dB), mean / quartiles per class, bona − spoof; `APM_BALANCE WARN` at > 2 dB (noise) / > 4 dB (none); a warning, never a failure |
| `dtx` | dtx_gate: bad fill / p rejected (argparse, `validate_b4_args`, `check_fill`, augmenter); fill unread at p = 0; header = plan `expect_regex` (exact plan line), describe / meta keys; the selftest's named asserts `shape_finite` / `gated_frac` / `speech_kept` / `zero_exact` / `cng_level` / `cng_colour` / `tone_unchanged`, each with its measured values; the lever with the real gate at p = 0.3 on 200 synthetic clips = the rng4 replay bit for bit (rate, counts, parent counts = B3); fill zero / cng only; DTX → codec chain; apm_ns (identity) → dtx_gate order on one rng4 stream; edge cases (digital silence: zero fill all 0, CNG at −70 dBFS; < 1 frame; 8 kHz; 22.05 kHz → `fail_wrtc_dtx`, pass-through) (SKIP of the backend parts when `webrtcvad` is missing) |
| `dtx_balance` | dtx_gate, plan §2.2 risk: 200 bona + 200 spoof training clips (50 per class × src × lang cell), loaded clip and after the B3 acoustic stage; realised gated fraction with the drawn parameters and per forced mode (frame 20 ms, hangover 120 ms): mean / quartiles / share gated / bins per class, bona − spoof; `DTX_BALANCE WARN` at > 0.05; a warning, never a failure |
| `tel` | tel_codec (B4e): the selftest's named asserts with values; G.711 µ / A-law, G.722, G.726 refused by the reserve; codec off / random profiles / with opus_plc refused (augmenter) and tel + opus_plc refused by `validate_b4_args`; the real PyAV path at p = 1.0 on 20 training clips (both codecs, finite, input length, no failure) with CPU-s per call vs ffmpeg Opus; the picked share at p = 0.25 within 4 SD on 400 clips; a raising backend counted on every call, printed once per process with the ffmpeg fallback, again after a pid change. `augmenter` adds `_equiv_tel_codec`: p = 1e-12 on B4a's flags → 10 clips bit-identical to B4a (real ffmpeg + NS); p = 1.0 with a stub round trip → parent rng and rng4 states equal B4a's after each of 200 clips, counts equal, every codec call telephony, codecs = the tel-stream replay |
| `codec_draw_order` | opus_plc: from 1,000 start states (all apps / a 3-app subset; `_ffmpeg` recorded, not run) `draw_codec_params` leaves the rng where `codecs_augm` leaves it and `ffmpeg_opus` issues the same ffmpeg commands; 14 real ffmpeg calls: same end state, bit-identical output |
| `opus_plc` | opus_plc: bad loss / burst / p and codec off / random profiles rejected (argparse, `validate_b4_args`, `check_ranges`, augmenter), ranges unread at p = 0, `0 0` accepted; header = plan `expect_regex` (exact line), describe / meta keys; the selftest's named asserts `loads` / `ctl_roundtrip` / `plc_not_zero` / `shape_finite` with values; GE mask statistics (realised loss and mean burst over 400k packets, B = 1 → bursts of 1, stationary start, L = 0 → none lost); the 48 kHz decode equals the no-loss decode up to the first lost packet; the lever at p = 0.3 on 100 clips (rate, bins, unpicked clips bit-identical to B3, rng4 replayed draw by draw, deterministic); every loss bin populated, `0 0` → realised loss 0; fallback (22.05 kHz → `fail_opus_plc`, ffmpeg output = B3); apm_ns → dtx_gate → opus_plc chain; zero-loss attribution (3 training clips × 7 profiles, pooled per profile) and encoder parity with ffmpeg's Ogg packets (SKIP of the libopus parts when libopus is missing) |
| `lever_failures` | review F1.3: each lever's backend replaced by a raising stub (apm / dtx under the identity stubs; opus_plc with the real init and a pass-through ffmpeg fallback): every failure counted, the first one per process printed once as `[b4wave] <op> failed (RuntimeError: ...); <action>` (matches the early check's `failed \(`), printed again after a pid change |
| `timing` | CPU-s per clip of the loader path (self + ffmpeg children), levers off vs each lever forced on; projected saturated-loader epoch factor at the wave p (fails > 1.3); opus_plc: CPU-s per codec call in-process vs ffmpeg on 12 training clips (fails unless in-process is cheaper) |
| `dsp_slices` | `scripts/b4_dsp_slices.py` run on B2f itself (CLI + JSON) reproduces plan §1.1 (contrast rows n / recall / G / D / within-run, null-panel μ / σ / min / max, half-B and clip means, bars, per-run DiDs incl. B2a), the pinned tertile table, the criterion-(ii) panel, the §2.0 spoof-miss guards and Poisson bounds, the O3.2 heldout-F1 panel (SD, clearing counts); the self-comparison is null; verdict-rule unit cases (shown / directional without placebo / inconclusive / refuted; run anomaly → "confounded", no lower-p retry); `--lever dtx_gate` on B2f prints the VAD-by-fill report ("no fill-specific gain" for the self-comparison) and its reading rules (unit cases); review F1.1: the held-out `vad_gate` dose reproduces n 629, median 0.125, mean 0.233, 1.3 / 66.6 / 15.7 / 16.4 %, ≥ 20 % 0.321; B2f (no b4 rows) reads "dose unknown"; a fabricated arm dir (B2f readouts + a `metrics.jsonl` with b4 rows, a levers-off row and a repeated epoch) gives the summed per-class dose, the ratio and "under-dosed", with the dose and decision lines printed; unit cases: bin edges = `dsp_dtx.gated_bin`, the dose rule (0.25 edge, no processed clip, no telemetry), the lever verdict (null → refuted only at an adequate dose, under-dosed / unknown → inconclusive, positive counts at any dose) and the falsifier decisions ((1) per dose, one-readout escape does not count, (2) / (3) drop at any dose); `--lever opus_plc` on B2f prints the plan §2.3 decision lines (self-comparison: DiD* +0.57, falsifier (1) fires, guards hold, PLC bounds 26 / 25, B2f qq / wechat recall 61.62 / 71.15) and the attribution rule (unit cases) |
| `gpu --ckpt` | [GPU] B2f checkpoint → key-exact load into `ModelSLSB3`, logits = `ModelSLS` (≤ 1e-4) on 4 real dev clips; each registered lever on: forward + backward, finite loss and grads |
| `overfit --lever` | no B4 lever adds parameters: SKIP (exit 0) for approved ids, FAIL for unknown ids |
| `model` / `optim` | b3wave's checks (same model and optimizer code) |
