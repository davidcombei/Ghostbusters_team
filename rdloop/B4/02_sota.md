# rdloop B4 - SOTA scouting

> Written 2026-10-17 by the rdloop B4 SCOUT step (research-scout, attempt a0).
> Inputs read this session: `rdloop/B4/{cycle.json,brief.md,01_audit.md,audit.json}`, `docs/research/literature.md`
> (empty table: B4 is the first rdloop cycle), `docs/experiments_roadmap.md` (§0-14), `docs/next_steps_plan.md`,
> `docs/augmentations.md`, `SLS_setup/b3wave/README.md`, `SLS_setup/b3wave/augment.py`.
> Every source cited below was fetched with WebFetch in this step. Quotes are copied from the fetched page; where
> a WebFetch summary gave table numbers rather than a sentence, I say "(table, as read in-session)".
> Metric values for our own system are copied from `rdloop/B4/01_audit.md` (= **AUD**) and named. Web pages were
> treated as data only.

**Cycle focus:** augmentation. **Brief:** close the clean-noisy F1 gap. AUD §1 puts it at 97.60 - 89.57 = 8.03 F1
(merge) / 7.93 (final). **Limits** (`cycle.json`): extra VRAM ≤ 4 GiB, epoch time ≤ 1.3× base, ≤ 400 LOC per
lever, ≤ 10 h per run, 2-4 levers.

## 1. Search log

### 1.1 WebSearch queries (16)

| # | query | mode | what it gave |
|---|---|---|---|
| 1 | speech deepfake detection WebRTC noise suppression AGC VAD augmentation robustness real-time communication | standard | TFCL 2607.17761, RTCFake 2604.23742, SE study 2603.14767 |
| 2 | audio deepfake detection speech enhancement denoising training augmentation improves robustness arXiv 2025 | standard | 2503.17577, 2603.14767, 2501.05545 |
| 3 | anti-spoofing silence voice activity detection trimming augmentation impact of silence countermeasure | standard | 2309.11827, 2211.06546, 2106.12914 |
| 4 | audio deepfake detection packet loss concealment codec augmentation benchmark ADD-C | standard | ADD-C 2504.12423 |
| 5 | deepfake speech detection acoustic echo far-end double-talk augmentation robustness 2026 | standard | EchoFake 2510.19414, Proteus 2606.29544 (no echo-aug paper) |
| 6 | ASVspoof 2021 data augmentation telephony codec transmission channel known-unknown augmentation countermeasure | standard | 2107.12018, RawBoost, 2110.10491 |
| 7 | github python bindings WebRTC audio processing noise suppression auto gain control pip webrtc-noise-gain | standard | webrtc-noise-gain, pywebrtc-audio |
| 8 | RTCFake speech deepfake detection real-time communication 2026 robustness noise augmentation new method | extended | RTCFake, TFCL, ADD-C (no RTC-SDD system papers yet) |
| 9 | opuslib python packet loss concealment decode lost frame Gilbert-Elliott simulation github | standard | libopus NULL-packet PLC convention, Gilbert-Elliott |
| 10 | AugMix ablation without Jensen-Shannon consistency mixing only CIFAR-10-C error | standard | AugMix 1912.02781 |
| 11 | rhythm perturbation data augmentation speech anti-spoofing silence trimming cross-dataset | standard | 2310.12014, 2406.17246 |
| 12 | noise-aware audio deepfake detection SNR benchmark survey practical recipes four-class 2512.13744 | standard | 2512.13744 (abstract-level only) |
| 13 | SILK codec AMR augmentation speech deepfake detection WeChat telephony narrowband robustness training | standard | 2107.12018, ADD-C (nothing WeChat/SILK-specific) |
| 14 | Opus DTX comfort noise generation VAD gating effect speech classifier robustness augmentation discontinuous transmission | standard | patents only, no ADD evidence |
| 15 | "echo" augmentation spoofing countermeasure robustness conference call residual echo deepfake detector false alarm | standard | Proteus 2606.29544 |
| 16 | audio deepfake detection training noise diversity DEMAND FSD50K unseen noise types generalization data augmentation study | standard | 2507.01750, 2407.07598 (no corpus-diversity evidence) |

### 1.2 Pages fetched (WebFetch) and cited

| short | title | URL(s) fetched | venue / year |
|---|---|---|---|
| TFCL | Time-Frequency Consistency Learning for Robust Speech Deepfake Detection | https://arxiv.org/abs/2607.17761 ; https://arxiv.org/html/2607.17761 | arXiv 2026 |
| RTCFake | RTCFake: Speech Deepfake Detection in Real-Time Communication | https://arxiv.org/html/2604.23742v1 | Findings of ACL 2026 |
| ADD-C | Benchmarking Audio Deepfake Detection Robustness in Real-world Communication Scenarios | https://arxiv.org/html/2504.12423v2 | arXiv 2025 |
| SIL | The Impact of Silence on Speech Anti-Spoofing | https://arxiv.org/abs/2309.11827 ; https://arxiv.org/html/2309.11827 | arXiv 2023 |
| SSG | Speech is Silver, Silence is Golden (Müller et al.) | https://arxiv.org/abs/2106.12914 | arXiv 2021 |
| BEY | Beyond Silence: Bias Analysis through Loss and Asymmetric Approach in Audio Anti-Spoofing | https://arxiv.org/abs/2406.17246 | arXiv 2024 |
| LPF | Low Pass Filtering and Bandwidth Extension for Robust Anti-spoofing Countermeasure Against Codec Variabilities | https://arxiv.org/abs/2211.06546 | ISCSLP 2022 |
| SE | Investigating the Impact of Speech Enhancement on Audio Deepfake Detection in Noisy Environments | https://arxiv.org/abs/2603.14767 | arXiv 2026 |
| CORR | Measuring the Robustness of Audio Deepfake Detection under Real-World Corruption | https://arxiv.org/abs/2503.17577 | arXiv 2025 |
| WCR | What Counts as Real? Speech Restoration and Voice Quality Conversion Pose New Challenges to Deepfake Detection | https://arxiv.org/abs/2603.14033 | arXiv 2026 |
| PROT | Proteus: Automated Adversarial Robustness Testing for Audio Deepfake Detectors | https://arxiv.org/abs/2606.29544 ; https://arxiv.org/html/2606.29544 | arXiv 2026 |
| CLAD | CLAD (contrastive learning against manipulation attacks) | https://arxiv.org/abs/2404.15854 | arXiv 2024 |
| ECHOF | EchoFake: A Replay-Aware Dataset for Practical Speech Deepfake Detection | https://arxiv.org/abs/2510.19414 | arXiv 2025 |
| UR21 | UR Channel-Robust Synthetic Speech Detection System for ASVspoof 2021 | https://arxiv.org/abs/2107.12018 | ASVspoof 2021 workshop (arXiv 2021) |
| UNM | Unmasking Deepfakes: Leveraging Augmentations and Features Variability for Deepfake Speech Detection | https://arxiv.org/html/2501.05545v1 | arXiv 2025 |
| AUGMIX | AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty | https://arxiv.org/abs/1912.02781 ; https://ar5iv.labs.arxiv.org/html/1912.02781 (the arxiv.org/pdf fetch was unreadable) | ICLR 2020 |
| DEEPAUG | The Many Faces of Robustness (DeepAugment) | https://arxiv.org/abs/2006.16241 ; https://ar5iv.labs.arxiv.org/html/2006.16241 (the arxiv.org/pdf fetch was unreadable) | ICCV 2021 |
| PYWRTC | pywebrtc-audio (GitHub README) | https://github.com/strands-labs/pywebrtc-audio | software |
| WNG | webrtc-noise-gain (GitHub README) | https://github.com/rhasspy/webrtc-noise-gain | software |
| WVAD | py-webrtcvad (GitHub README) | https://github.com/wiseman/py-webrtcvad | software |
| OPUSLIB | opuslib (GitHub repo + `opuslib/classes.py`) | https://github.com/orion-labs/opuslib ; https://github.com/orion-labs/opuslib/blob/master/opuslib/classes.py | software |
| DFN | DeepFilterNet (GitHub README) | https://github.com/Rikorose/DeepFilterNet | software |

## 2. Landscape per direction (what works under channel mismatch)

### 2.1 Platform front-end processing (AEC / NS / AGC / VAD) is a major error source, and training on it helps

- **TFCL** simulates the RTC "audio front-end" with WebRTC: "we adopt the standard implementations provided by WebRTC".
  On the cumulative chain, XLSR-AASIST EER goes clean 0.23 → echo 2.20 → AEC 2.34 → noisy 6.31 → NS 5.81 → AGC 12.70 →
  VAD 22.20 (Table 1, as read in-session): "while that of the SSL-based XLSR\_AASIST rises from 0.23% to 22.20%."
  The training-data baselines matter most for us (Table 2, as read in-session, XLSR-AASIST, VAD condition):
  clean-only 22.20, **Mix (clean + front-end-processed training) 17.39 with clean 0.52**, front-end-only 11.84 with
  clean 7.06. Quote: "The Mix strategy partially alleviates the degradation." Their consistency method gets 9.78, but
  two-view training breaks this cycle's 1.3× epoch cap (AUD §5, compute row). That leaves the *Mix* (augmentation)
  arm as the transferable part.
- **RTCFake** names the platform DSP as the distortion source. From the Fig. 1 caption: "key distortion sources,
  including noise suppression, echo cancellation, codec encoding and decoding, and packet loss". From §5.4:
  "built-in speech enhancement modules of communication platforms, such as noise suppression, echo cancellation, and
  automatic gain control". Echo (S04) is the hardest noise condition for PCL (13.40 EER), Mix (16.92) and On (17.34)
  (Table 3, as read in-session).
- **WCR**: "detectors can often identify that speech has been processed". **PROT** found that "all of the top-100
  chains (ranked by absolute score shift) target bonafide samples", and that "specific augmentation chains can
  reliably flip detection verdicts while preserving speech intelligibility and speaker identity". This is the same
  failure as AUD §3.3: about three quarters of noisy bona false alarms are clips the model gets right when clean.
- **CORR**: "Robustness to unseen corruptions can be improved through targeted data augmentation during training".
  **SE** applied enhancement only at test time ("while leaving the training data untouched"), and enhancer quality
  did not predict detection gain. Enhancement as an inference pre-processor is therefore not proposed (§4).

**Mapping to our gaps.** AUD §3.4 (half B, bona recall merge / final): heldout `afftdn` NS 55.4 / 58.1, `dynaudnorm`
AGC 64.2 / 62.3, VAD + CNG 55.9 / 61.8, loss + repeat-PLC 66.2 / 66.9, against 72.7 with no DSP. B2f has never
trained on any DSP stage (AUD G1).

### 2.2 Silence / VAD

- **SSG**: "EER increases from 3.6% (with silence) to 15.5% (trimmed silence)." Silence is a learnable shortcut.
- **SIL** (uses WebRTC VAD, 10 ms frames): "removing the silence of test speech through Voice Activity Detection
  (VAD) can severely degrade performance". Its training-side fix, "the silence frames are masked by zero
  (Sil-mask)", improves ASVspoof 2021 DF eval EER for all three models (Table VI, as read in-session: SENet
  22.95 → 22.14, LCNN 22.46 → 20.41, AASIST 16.92 → 16.06). The LA results are mixed, and on LA19 unseen attacks it
  helps LCNN ("relatively decreased by 31%") but hurts SENet and AASIST.
- **BEY**: "the distribution of silence differs between the two classes, which can serve as a shortcut."
- **LPF** reports that band-limiting also helps "under the codec condition when VAD is applied" ("reduced by up to
  25% relatively"). Band-limiting is reserved for us (§4).

**Mapping.** AUD §3.5: the `sil_only` and `noise_only` bona-source probes are 100 % spoof (mean dP +0.89 / +0.90).
AUD G3: 136 / 133 of the noisy bona false alarms sit at p ≥ 0.99. The heldout VAD + CNG slice is at 55.9 / 61.8. No
lever has ever targeted this.

### 2.3 Codec and packet loss

- **ADD-C** degrades six codecs (AMR-WB, EVS, IVAS, OPUS, Speex-WB, SILK) at packet-loss rates 0-20 %. Trained on
  clean data, "the baseline models experienced an average degradation of 5.30% in EER", and the authors' models
  showed "an average degradation of 2.33% (EER)". Trained on the codec + packet-loss augmented set: "The EER remains
  unchanged, with a slight improvement of 0.003% from C0 to C1."
- **UR21** used "an acoustic simulator to apply transmission codec, compression codec, and convolutional impulse
  responses". **UNM**: "the inclusion of compression augmentations consistently improves performance across all
  configurations" (XLS-R eval EER 8.34 → 6.15, as read in-session).
- **Our own counter-evidence.** B3d (random ffmpeg-Opus bitrate / frame / VBR / packet-loss *hint*) scored -2.20
  [-2.90, -1.57] (AUD §2 row 13). That lever never dropped a packet and never exercised decoder concealment: ffmpeg
  4.2 has no FEC / DTX options (`b3wave/README.md`; `augment.py` `codecs_augm_random` only sets `-packet_loss`, the
  encoder's expected-loss hint).

### 2.4 Echo / replay

- RTCFake Table 3 (above): echo is the hardest condition. TFCL Table 1: echo and AEC stages each add ~2 EER points
  on clean-trained XLSR-AASIST.
- **ECHOF**: "average accuracy dropping to 59.6% when evaluated on replayed audio". Including replay data in
  training is what fixes it there.
- **CLAD**: manipulations raise FAR: "with FAR rising to 36.69%, 31.23%, and 51.28%". Training against them helps:
  "reducing the FAR to 0.81% under noise injection".
- I found **no paper** that measures far-end echo / AEC-residual augmentation for ADD (queries 5 and 15). The echo
  card therefore rests on problem-size evidence and on our own slice data (AUD G4: self_echo bona recall
  79.40 / 79.83).

### 2.5 Compositional / randomised augmentation (vision, transferable as a method)

- **AUGMIX** Table 4 (CIFAR-10-C error, as read in-session): Standard 26.9; Random AutoAugment 17.0; AugmentAndMix
  (no JSD) 13.1; AugMix 11.2. Quote: "Mixing random augmentations without the Jenson-Shannon divergence loss gives
  us an error rate of 13.1%." Most of the gain comes **without** the consistency loss, so it fits a single-view
  budget.
- **DEEPAUG**: "DeepAugment distorts images by perturbing internal representations of deep networks." and "The
  image-to-image network's weights and feedforward activations are distorted with each pass." ImageNet-C mCE goes
  76.7 → 60.4, and → 53.6 with AugMix; ImageNet-R error 63.9 → 57.8 (as read in-session).

### 2.6 Tooling (all pip / conda installable; none is a reserved implementation)

- **PYWRTC**: "Python bindings for the WebRTC audio processing module." It exposes AEC3, NS (levels 0-3), AGC2,
  VAD and HPF, with pipeline "HP filter -> AEC -> NS -> AGC". Pre-built wheels for Linux and Python 3.10-3.14;
  Apache-2.0.
- **WNG** (fallback): "Tiny Python wrapper around webrtc-audio-processing for noise suppression and auto gain only".
  NS level 0-4, auto-gain 0-31 dBFS, 16 kHz mono int16, 10 ms blocks; `pip install webrtc-noise-gain`.
- **WVAD**: "A frame must be either 10, 20, or 30 ms in duration". Mode: "0 is the least aggressive about filtering
  out non-speech, 3 is the most aggressive."
- **OPUSLIB**: "Python bindings to the libopus, IETF low-delay audio codec". `Decoder.decode(opus_data, frame_size,
  decode_fec: bool = False)` gives in-band FEC decoding. It needs a shared libopus (conda-forge `libopus`).
  Signalling a lost packet with an empty payload is the libopus PLC convention (query 9). It must be verified in
  the smoke test.
- **DFN**: `pip install deepfilternet`, MIT / Apache-2.0. The repo already has a `suppress` stage that uses it
  (`--aug_use_deepfilternet`, never enabled per AUD G1).

**Rule check** (roadmap §2): local WebRTC AEC / NS / AGC / VAD simulation is "explicitly allowed". Software codecs
are allowed. Speech from our own training utterances (babble, far-end) is "likely allowed". Eval-only reserve (AUD
§5): training DSP must use implementations disjoint from `sim_ops` (`docs/next_steps_plan.md` §2.4 lists WebRTC APM
NS / AGC2, `webrtcvad`, libopus PLC, RNNoise, DeepFilterNet and speexdsp as acceptable).

## 3. Idea cards

### 3.0 Ranked overview

| rank | id | title | category | gaps | est LOC | extra time | deps | evidence strength |
|---|---|---|---|---|---|---|---|---|
| 1 | I1 | WebRTC-APM NS + AGC2 stage after the noise stage (noise → APM → codec) | augmentation | G1, G2 | ~220 | ~1.05× | pywebrtc-audio (or webrtc-noise-gain) | ADD/RTC-specific (TFCL Mix arm) |
| 2 | I2 | WebRTC-VAD DTX gating with comfort-noise / zero / attenuated gaps + pause insertion | augmentation | G3, G1 | ~200 | ~1.02× | webrtcvad | ADD-specific, mixed (SIL), problem-size (TFCL, SSG) |
| 3 | I3 | In-process libopus round trip with real packet loss + decoder PLC / FEC | augmentation | G1, G2 | ~230 | ≤ 1.0-1.05× | opuslib + libopus | ADD-specific (ADD-C) |
| 4 | I4 | AugMix-style single-view mixing of parallel augmentation chains | augmentation | G1, G2 | ~130 | ~1.1× | none | vision (AUGMIX), chain-flip evidence (PROT) |
| 5 | I5 | Echo suite: multi-tap self-echo + far-end talker echo with AEC3 residual | augmentation | G4, G1 | ~280 | ~1.08× | pywebrtc-audio | problem-size only (RTCFake, ECHOF, CLAD) |
| 6 | I6 | DeepAugment-audio: weight-perturbed DeepFilterNet as a randomised "unknown suppressor" | augmentation | G1 | ~180 | 1.1-1.3× (risk) | deepfilternet | vision (DEEPAUG) |

The ranking weighs gap size (AUD §3.1: heldout carries 4.23 / 4.11 of the missing WF1, matched 3.07 / 3.07, the
saturated core is 64 % of noisy FA), feasibility inside the limits, and evidence strength. With a 2-4 lever budget,
my recommended build is **I1, I2, I3** (+ I4 if a fourth slot is wanted). All six are CPU-side, change no
state_dict and add no VRAM.

**Cross-cutting recommendations for the architect** (they apply to every card):
1. **Order** noise / echo → platform DSP → codec, as in RTCFake and TFCL. Today's `RTCAugmenterB3.__call__` runs
   `_stage` then the codec, so new stages go between the two.
2. **Separate RNG stream** for the new stages (e.g. `default_rng(hash((seed, pid, "b4")))`). With a flag off,
   nothing is drawn and the B3/B2f path stays bit-identical (`b3wave.checks augmenter`). With a flag on, the
   parent's stage and codec draws are unchanged, so the lever really is "one component".
3. **Dose and the B3 lesson 4 guard.** AUD §6 warns that harder input changes cut the late fit. Default each new
   stage to p ≈ 0.3, applied class-independently on top of the existing draw, and pre-register the falsifier:
   ep8 train CE ≥ 1.5× B2f and heldout R_bona down.
4. **Reserve hygiene.** `assert_training_ops` matches names. New op names must not collide with or contain the
   reserved `vad_gate`, `packet_loss`, `bandlimit`, `afftdn`, `dynaudnorm`, etc. (use e.g. `wrtc_ns`, `wrtc_agc2`,
   `wrtc_dtx`, `opus_plc`).
5. **G6 readability.** After I1 / I2 / I3, the heldout NS / AGC / VAD / PLC slices become "family seen in training,
   implementation unseen". Report those slices separately from the "no DSP" and noise / codec slices, so a gain is
   not read as unseen-condition generalisation without the label.
6. **CPU budget.** Augmentation runs in the 64 loader workers. Smoke-time one epoch on the 1 % subset, with and
   without the stage, and make sure the clips/s headroom covers the 1.3× cap (≈ 40.6 min per epoch, AUD §5).

---

### I1. WebRTC-APM noise suppression + AGC2 after the noise stage

- **Category:** augmentation · **gaps:** G1 (priority 1), G2.
- **Mechanism.** Every RTC platform applies NS and AGC to (speech + noise) before the codec (RTCFake §5.4 quote
  above). B2f trains only on noise + Opus (AUD G1), so the residual-noise / musical-noise / gain-pumping
  signature of a suppressor is out of support, and the model reads it as "processing ⇒ spoof" (WCR, PROT). The
  worst heldout slices are exactly NS and AGC: `afftdn` 55.4 / 58.1, `dynaudnorm` 64.2 / 62.3 vs no-DSP 72.7 (AUD
  §3.4). Showing bona *and* spoof through a different suppressor family (WebRTC NS levels 1-3 + AGC2) removes
  "suppressor artefact" as a class cue. Running it **after** the noise stage also changes what in-distribution
  noise looks like (G2): the model sees the noise partly removed and the speech partly distorted, as on a real
  platform.
- **Sources:** TFCL (arxiv.org/abs + html/2607.17761): "we adopt the standard implementations provided by WebRTC";
  "The Mix strategy partially alleviates the degradation." RTCFake (arxiv.org/html/2604.23742v1): "built-in speech
  enhancement modules of communication platforms, such as noise suppression, echo cancellation, and automatic gain
  control". PYWRTC (github.com/strands-labs/pywebrtc-audio): "HP filter -> AEC -> NS -> AGC". WCR
  (arxiv.org/abs/2603.14033): "detectors can often identify that speech has been processed".
- **Reported effect:** TFCL Table 2 (XLSR-AASIST; WebRTC front-end chain, VAD = full-chain condition, EER %):
  clean-only training 22.20 → Mix training 17.39 (clean 0.23 → 0.52); front-end-only training 11.84 (clean 7.06).
  Quote: "The Mix strategy partially alleviates the degradation."
- **Transfer risk.** TFCL trains and tests on the *same* WebRTC chain, so cross-implementation transfer (WebRTC NS
  → `afftdn`) is not shown. The heldout NS slice may move less than the matched-style gain. Their Mix arm also cost
  clean EER (0.23 → 0.52), and our clean term is 30 % of WF1. Keep p ≈ 0.3 and watch clean bona recall. AGC2 on top
  of read-time loudness norm is a dynamic gain pattern, not a level shift, so it should not re-open the level
  question (AUD §6). B3 lesson 4 risk: a harder input distribution. Mitigate with the dose and falsifier above.
- **Sketch.** New `SLS_setup/b4wave/dsp.py`: `apm_process(audio, rng, ns_level, agc)`. It converts float32 16 kHz to
  int16 and runs `pywebrtc_audio` `AudioProcessor` / `NoiseSuppressor` + `GainController` in 10 ms frames, with a
  WNG fallback (`AudioProcessor(auto_gain_dbfs, noise_suppression_level).Process10ms`). `RTCAugConfigB4(RTCAugConfigB3)`
  gets `apm_p=0.0`, `apm_ns_levels=(1,2,3)` and `apm_agc_p=0.5`. `RTCAugmenterB4.__call__` runs `out =
  self._stage(audio)`, then with p = `apm_p` (b4 RNG) `out = finalize(apm_process(out), out)`, then the codec (same
  code path as B3). `assert_training_ops(filters=["wrtc_ns", "wrtc_agc2"])`. Telemetry keys `stage:wrtc_ns{1,2,3}`
  and `stage:wrtc_agc2`. Flags `--aug_apm_p`, `--aug_apm_ns_levels`, `--aug_apm_agc_p`. Checks: an off-path identity
  test, and an energy drop on a 1 s noise-only input.
- **Estimates:** ~220 LOC; `state_dict_change` false; extra VRAM 0; epoch ×~1.05 (C++ APM at ~1 call per 10 ms
  frame, in the workers). **Dependencies:** `pywebrtc-audio` (fallback `webrtc-noise-gain`).
- **Novelty vs tried:** no DSP stage has ever been in training (AUD G1 "already tried"). The existing DeepFilterNet
  `suppress` stage was never enabled. It is also drawn *instead of* a noise stage (one stage per clip,
  `docs/augmentations.md` §3.1), so it would suppress clean speech, not noisy speech. Not adjacent to any exhausted
  lever (AUD §6).

### I2. WebRTC-VAD DTX gating with comfort-noise / zero / attenuated gaps + pause insertion

- **Category:** augmentation · **gaps:** G3 (saturated core, "no speech evidence ⇒ spoof"), G1 (VAD + CNG slice).
- **Mechanism.** RTC endpoints gate non-speech: DTX, then comfort noise or digital silence. The model has never
  seen bona audio with gated pauses. It scores non-speech-dominated input as spoof with near certainty (`sil_only`
  / `noise_only` probes 100 % spoof, mean dP +0.89 / +0.90, AUD §3.5), and heldout VAD-gated bona drops to 55.9 /
  61.8 with CNG (AUD §3.4). Class-independent gating, with zeroed, attenuated or CNG-filled gaps and with pauses
  inserted at speech boundaries, makes the "silence content / duration" cue (SSG, SIL, BEY) useless as a label
  signal. It also teaches the head that gated stretches carry no class evidence. This is the only card that
  attacks G3 directly, and G3 is 64 % of noisy bona FA (AUD G3).
- **Sources:** SIL (arxiv.org/abs + html/2309.11827): "removing the silence of test speech through Voice Activity
  Detection (VAD) can severely degrade performance"; "the silence frames are masked by zero (Sil-mask)". SSG
  (arxiv.org/abs/2106.12914): "EER increases from 3.6% (with silence) to 15.5% (trimmed silence)." TFCL html: VAD
  is the last and largest step of the chain (AGC 12.70 → VAD 22.20). WVAD (github.com/wiseman/py-webrtcvad): "A
  frame must be either 10, 20, or 30 ms in duration".
- **Reported effect:** SIL Table VI, ASVspoof 2021 DF eval EER %, Sil-mask training vs unprocessed: AASIST 16.92 →
  16.06, LCNN 22.46 → 20.41, SENet 22.95 → 22.14. LA21 and LA19 results are mixed. Quote: "the EER of the LCNN
  model on the evaluation partition is relatively decreased by 31%".
- **Transfer risk.** The evidence is mixed (SIL hurts SENet / AASIST on LA19 unseen attacks). Sil-mask is a
  *deterministic* train-time transform, while I2 is a stochastic augmentation at p ≈ 0.3, which is milder. If
  RTCFake bona really does have different pause statistics, removing the cue may cost clean F1. Watch clean bona
  recall and spoof recall. Heldout's VAD uses `sim_ops` energy-VAD (reserved); webrtcvad gating differs in
  thresholds and hangover, so the transfer to that slice is plausible but not shown. Inserted pauses must stay
  short (≤ 1 s), so a 4.04 s crop is rarely speech-free.
- **Sketch.** `b4wave/dsp.py::dtx_gate(audio, rng)`: int16 → `webrtcvad.Vad(mode∈{0..3})` over 10 / 20 / 30 ms
  frames → speech mask → hangover dilation U(40, 200) ms → fill non-speech with one of {zeros, comfort noise
  (white or pink, level −45..−60 dB rel. clip RMS, 5 ms raised-cosine ramps), attenuation −15..−35 dB}. With
  probability `sil_insert_p`, insert 1-2 pauses of U(0.2, 1.0) s of the same fill at random speech → non-speech
  boundaries (the clip is lengthened, and the existing random crop then picks 4.04 s). Applied after the acoustic
  stage (and after I1 if both are on), before the codec. Flags `--aug_dtx_p`, `--aug_dtx_fill {zero,cng,atten,mix}`,
  `--aug_dtx_insert_p`. Op name `wrtc_dtx` (not `vad_gate`). Telemetry: fraction of frames gated. Checks: an off
  path, a speech-mask sanity check on a dev-free synthetic tone + silence, and length bookkeeping.
- **Estimates:** ~200 LOC; `state_dict_change` false; VRAM 0; ×~1.02. **Dependencies:** `webrtcvad`
  (py-webrtcvad, or its `webrtcvad-wheels` build).
- **Novelty vs tried:** A2.3 / A2.9 were never run (AUD §2 "never run"). Neither LR, sampler nor consistency moved
  the saturated core (AUD G3). The `padded` / `trimmed` probes only test digital silence at the clip ends, not
  gated pauses.

### I3. In-process libopus round trip with real packet loss + decoder PLC / FEC

- **Category:** augmentation · **gaps:** G1 (loss + PLC is the largest DSP slice, n = 130 bona), G2 (low-rate
  narrow-band Opus).
- **Mechanism.** Real RTC links drop packets, and the decoder conceals them: SILK / CELT PLC extrapolates the
  excitation, and in-band FEC decodes a low-rate copy. Both produce smeared, buzzy or over-smoothed segments that
  a detector can read as synthetic. Today's codec stage never drops a packet. B3d's `-packet_loss` was only the
  encoder's expected-loss hint (`b3wave/augment.py` `codecs_augm_random`). Heldout loss + repeat-PLC bona recall is
  66.2 / 66.9 against 72.7 without DSP (AUD §3.4). Doing the codec in-process with `opuslib` lets us drop packets
  with a Gilbert-Elliott burst model and let libopus conceal them. The fixed per-app profiles stay as they are
  (unlike B3d), so the codec distribution only gains loss + concealment.
- **Sources:** ADD-C (arxiv.org/html/2504.12423v2): "the baseline models experienced an average degradation of
  5.30% in EER"; "The EER remains unchanged, with a slight improvement of 0.003% from C0 to C1." RTCFake: "key
  distortion sources, including noise suppression, echo cancellation, codec encoding and decoding, and packet
  loss". OPUSLIB (github.com/orion-labs/opuslib + `classes.py`): "Python bindings to the libopus, IETF low-delay
  audio codec"; `decode_fec: bool = False`. UR21 (arxiv.org/abs/2107.12018): "an acoustic simulator to apply
  transmission codec, compression codec, and convolutional impulse responses".
- **Reported effect:** ADD-C (codec + packet loss 0-20 %, EER %): the authors' model trained on the original data
  degrades by 2.33 on average from C0 to C1. The same model trained on codec + packet-loss augmented data is
  unchanged (−0.003). Quote: "The EER remains unchanged, with a slight improvement of 0.003% from C0 to C1."
- **Transfer risk.** ADD-C's gain is in-distribution (same codecs and loss model in training and test). Heldout uses
  `sim_ops` repeat-PLC on non-Opus codecs (reserved), so transfer to that slice is unproven. B3d shows Opus-side
  changes can hurt (−2.20). Mitigations: keep the app profiles fixed, loss on p ≈ 0.3 of clips, loss rate
  U(2, 15) %. libopus must be a conda-env library, or a container restart can silently disable the stage (AUD §5
  environment row). Mirror the existing pass-through warning *and* count failures in telemetry.
- **Sketch.** New `b4wave/opus_plc.py::opus_loss_roundtrip(audio, rng, app)`. It takes bitrate / cutoff / frame /
  application from `CODEC_PROFILES[app]` (cutoff → Opus bandwidth NB / MB / WB), builds an
  `opuslib.Encoder(16000, 1, voip|audio)`, optionally sets in-band FEC + expected loss, and encodes frame by frame.
  A Gilbert-Elliott mask drops packets (loss U(0.02, 0.15), mean burst U(1, 4) frames). Decoding: a received packet
  is decoded normally; a lost packet with the next one received and FEC on uses `decode(next, frame,
  decode_fec=True)`; otherwise the empty-payload PLC call is used (verify in the smoke test that this triggers
  libopus PLC and does not raise). Hook: in `RTCAugmenterB4.__call__`, with p = `opus_plc_p` the codec step calls
  this instead of ffmpeg `codecs_augm`, with the same app draw, so the codec exposure (p = 1.0) and the app mix are
  unchanged. Op name `opus_plc` (not `packet_loss`). Flags `--aug_opus_plc_p`, `--aug_opus_loss`,
  `--aug_opus_burst`, `--aug_opus_fec_p`.
- **Estimates:** ~230 LOC; `state_dict_change` false; VRAM 0; ×1.0-1.05 (in-process encode / decode instead of two
  ffmpeg subprocesses per clip, on the share that takes this path). **Dependencies:** `opuslib`, conda-forge
  `libopus`.
- **Novelty vs tried:** B3d randomised encoder parameters and never dropped packets. "Packet loss + PLC" is listed
  as a never-run direction (AUD §2: A2.6 / D4).

### I4. AugMix-style single-view mixing of parallel augmentation chains

- **Category:** augmentation · **gaps:** G1 (unseen combinations; heldout chain depth 4 → 50.0 bona recall, AUD
  §3.4), G2 (multi-source noise, listed as untested in AUD G2).
- **Mechanism.** Unseen platform conditions are compositions: PROT's verdict-flipping chains are depth-2 chains,
  and every one of its top-100 targets bona. Instead of chaining (harder) or widening ranges (B3d / B3e, both
  negative), AugMix mixes k parallel chains with Dirichlet weights and keeps a Beta-weighted skip to the clean
  clip. That **raises diversity while lowering average strength**, the opposite of the B3d / B3e signature. Two
  noise chains mixed together also give two-source noise at a random effective SNR. The AUGMIX ablation shows that
  the mixing alone, without the JS loss, carries most of the corruption-robustness gain, which matters because
  the two-view JS loss breaks our time cap.
- **Sources:** AUGMIX (arxiv.org/abs/1912.02781; ar5iv.labs.arxiv.org/html/1912.02781): "Mixing random
  augmentations without the Jenson-Shannon divergence loss gives us an error rate of 13.1%." PROT
  (arxiv.org/abs + html/2606.29544): "all of the top-100 chains (ranked by absolute score shift) target bonafide
  samples".
- **Reported effect:** AUGMIX Table 4, CIFAR-10-C error %: Standard 26.9 → AugmentAndMix (no JSD) 13.1 (Random
  AutoAugment 17.0; AugMix with JSD 11.2). Quote as above.
- **Transfer risk.** Vision evidence. In audio, summing time-domain chains of the same clip (echo / reverb vs dry)
  causes comb filtering and partial cancellation, which is realistic for echo but new. The skip to the clean clip
  lowers effective noise exposure and could *reduce* the noisy gain (the B3e direction run in reverse). The codec
  still follows on 100 % of clips, so clean-codec exposure is unchanged.
- **Sketch.** `RTCAugmenterB4._stage` override, active when `mix_k > 0` and `rng < p_apply`: build k ∈ {2, 3}
  chains, each U{1, 2} distinct stages from `self.stages` via B3's `_apply_stage` (plus I1 / I2 stages if those
  flags are on). Draw w ~ Dir(α·1) and m ~ Beta(α, α), set `out = m·x + (1−m)·Σ wᵢ·chainᵢ`, then `finalize`.
  Telemetry counts per stage and a histogram of m. Flags `--aug_mix_k`, `--aug_mix_alpha` (default 1.0),
  `--aug_mix_p`.
- **Estimates:** ~130 LOC; `state_dict_change` false; VRAM 0; ×~1.1 (k noise reads / RIR convolutions; the codec
  runs once). **Dependencies:** none.
- **Novelty vs tried:** chaining (`--aug_max_stages`) has never run, and mixing has never been coded. Unlike the
  exhausted B3d / B3e it does not widen any range (AUD §6).

### I5. Echo suite: multi-tap self-echo + far-end talker echo with an AEC3 residual

- **Category:** augmentation · **gaps:** G4 (echo; in the brief's noisy measure, not in WF1), G1 (AEC-like
  processing).
- **Mechanism.** Training echo is one delayed copy (delay 40-160 ms, gain 0.15-0.5, `docs/augmentations.md` §3.2).
  Eval echo includes multi-tap self-echo (bona recall 79.40 / 79.83, the worst echo family) and far-end echo with or
  without AEC (AUD §3.4, G4). Two additions: (a) 2-4-tap self-echo with decaying gains, optionally through a
  measured RIR; (b) a far-end talker (a random **bona** training clip for both classes, so it is class-independent
  and leaks no label) convolved with an RIR, mixed at an echo-to-near ratio of −10..+5 dB (double-talk when both
  are active), then WebRTC AEC3 with the true far reference, which leaves realistic residual echo and
  suppression artefacts.
- **Sources:** RTCFake (arxiv.org/html/2604.23742v1): echo is the hardest condition (Table 3, as read in-session),
  and the caption quote "noise suppression, echo cancellation, codec encoding and decoding, and packet loss".
  ECHOF (arxiv.org/abs/2510.19414): "average accuracy dropping to 59.6% when evaluated on replayed audio". CLAD
  (arxiv.org/abs/2404.15854): "CLAD enhanced robustness, reducing the FAR to 0.81% under noise injection". PYWRTC:
  AEC3 with `process(near, far)`.
- **Reported effect:** none for echo augmentation in ADD (searched twice). Problem size: RTCFake Table 3, echo EER
  PCL 13.40 / Mix 16.92 / On 17.34, the hardest condition for each. Closest training-side evidence: CLAD
  (manipulation-robust training): FAR "rising to 36.69%, 31.23%, and 51.28%" without it, and "reducing the FAR to
  0.81% under noise injection" with it.
- **Transfer risk.** The weakest evidence of the six, and echo is not in local WF1, so a win shows only on
  `sim_echo_v1` and the progress echo subset. Far-end speech from our own training utterances is "likely allowed",
  not confirmed (roadmap §2). Reading a second clip per call adds I/O in the workers. AEC3 needs aligned far / near
  streams with a delay hint.
- **Sketch.** `b4wave/echo.py`: `self_echo_multi(audio, rng, rirs)` and `farend_aec(audio, far, rng, rirs, apm)`.
  The augmenter gets a `far_pool` (bona file list from the train protocol, passed via `RTCAugConfigB4`). It runs as
  a post-stage with p = `echo2_p` (the b4 RNG) so the 1/N stage shares stay as they are. Op names `echo_multitap`
  and `wrtc_aec3`. Flags `--aug_echo2_p`, `--aug_echo2_modes`, `--aug_echo2_aec_p`.
- **Estimates:** ~280 LOC; `state_dict_change` false; VRAM 0; ×~1.08. **Dependencies:** `pywebrtc-audio`.
- **Novelty vs tried:** A2.4 has never run. No lever has targeted G4 (AUD G4).

### I6. DeepAugment-audio: weight-perturbed DeepFilterNet as a randomised "unknown suppressor"

- **Category:** augmentation · **gaps:** G1 (unseen platform DSP).
- **Mechanism.** The five unseen eval platforms run suppressors we cannot reproduce. Perturbing the weights and
  activations of a real neural suppressor (DeepFilterNet, already wired as `suppress`) produces a *family* of
  plausible-but-novel NS distortions around a real one. This is domain randomisation of the DSP, the same
  mechanism that DEEPAUG uses for image corruptions. It covers "unknown suppressor" artefacts without touching any
  reserved implementation.
- **Sources:** DEEPAUG (arxiv.org/abs/2006.16241; ar5iv.labs.arxiv.org/html/2006.16241): "DeepAugment distorts
  images by perturbing internal representations of deep networks."; "The image-to-image network's weights and
  feedforward activations are distorted with each pass." DFN (github.com/Rikorose/DeepFilterNet): `pip install
  deepfilternet`.
- **Reported effect:** ImageNet-C mCE (ResNet-50) 76.7 → 60.4 with DeepAugment and → 53.6 with DeepAugment +
  AugMix; ImageNet-R error 63.9 → 57.8 %. Quote: "DeepAugment distorts images by perturbing internal
  representations of deep networks."
- **Transfer risk.** Vision-only evidence. DFN on CPU at 48 kHz inside 64 workers may break the 1.3× cap, and its
  RTF is not documented in the README, so it must be measured in the smoke test. Strongly perturbed masks can create
  vocoder-like artefacts on bona, which teaches "artefact ⇒ bona" and costs spoof recall. Neural processing of bona
  is a rule grey zone. It is not resynthesis, but check it against roadmap §2.
- **Sketch.** `b4wave/deepaug.py`: a per-worker DFN with a pristine state_dict. Every N = 50 calls, build a
  perturbed copy: multiplicative noise σ ~ U(0.05, 0.3) on 30-50 % of the weight tensors, 0-10 % zeroing, and sign
  flips of one random channel. Run `enhance`, then a wet / dry mix U(0.5, 1). Post-stage with p = `deepaug_p`
  after the noise stage; `torch.set_num_threads(1)` per worker. Op name `dfn_deepaug`.
- **Estimates:** ~180 LOC; `state_dict_change` false; VRAM 0 (CPU); ×1.1-1.3 (risk). **Dependencies:**
  `deepfilternet`.
- **Novelty vs tried:** roadmap X5.3, never built. DFN has never run in training (AUD G1).

## 4. Rejected ideas

| idea | why rejected |
|---|---|
| Two-view consistency variants (TFCL time-frequency consistency, RTCFake PCL, JS / embedding) | The strongest documented lever (TFCL VAD 9.78), but two views double the epoch (B3a 57.7-59.8 min vs the 1.3× cap of ≈ 40.6 min, AUD §5). Loss, not augmentation (cycle focus). JS dose above 0.3 is exhausted (AUD §6). |
| Codec zoo: real SILK (`pysilk`), AMR-NB / WB, EVS, or more Opus randomisation | Adjacent to the exhausted B3d (−2.20 [−2.90, −1.57]; g726 / gsm bona got worse, AUD §6). There is no local proxy for real SILK (matched qq / wechat are Opus-emulated, `docs/augmentations.md` §3.3). G.722 / G.726 / GSM / speex / G.711 are reserved. AMR encoder support in ffmpeg 4.2 is unknown. ADD-C / UR21 show in-distribution gains only. |
| Low-pass / band-limit / 8 kHz resampling augmentation (LPF 2211.06546; UNM LPF) | `bandlimit` and `resample8k` are reserved heldout filters (AUD §5). Training on them would erase the heldout band-limit slice (58.7) as an unseen test, whatever the implementation. |
| Speech enhancement as an inference pre-processor (SE 2603.14767, CORR 2503.17577) | Not augmentation. Rule status "ask" (roadmap §2). SE finds enhancer quality does not predict detection gain, and the roadmap lists it under "explicitly not recommended". |
| Babble from our own training speech | Rule status only "likely allowed" (roadmap §2). Mixed-label babble leaks label evidence into the noise. The coffee / office pools already hold babble-like content. Revisit after organiser confirmation. |
| Neural-codec round trip (EnCodec / DAC) as augmentation | Rule "ask" (resynthesis of bona). CORR notes that neural codecs cause large detection drops, so there is a label-noise risk. |
| Noise-pool expansion (DEMAND / FSD50K / DNS non-speech) for vacuum / washing-machine / wind-like noise | Query 16 found no ADD-specific evidence. It is a data-acquisition task, not a code lever, and must avoid the 11 reserved ESC-50 classes. Hand it to the owner as a data task. |
| SNR shape / range (`--aug_snr_dist beta`), chaining (`--aug_max_stages`), RawBoost SSI coloured noise, SpecAugment | Flags already exist in b3wave. Exploop can screen them without an rdloop build. Range widening is exhausted (B3e, AUD §6). |
| Rhythm-perturbation copy-synthesis (2310.12014) | Creates new spoofs by re-synthesising bona, which is forbidden ("no new TTS / VC spoofs", roadmap §2). |
| Proteus-style adversarial chain mining with the B2f model (worst-case augmentation search) | Needs an offline search with model forward passes over training bona, plus a policy loop. PROT reports no retraining numbers ("Model hardening" is described only). Hold for a later cycle, and I4 covers part of the compositional idea. |
| Speech-mask training (SIL "Speech-mask") | It keeps only silence, which is the shortcut we want to remove, and SIL's results for it are mixed. I2 takes the Sil-mask / gating direction instead. |

## 5. Notes for PROPOSE / CRITIQUE

- **Strongest RTC-specific evidence:** I1 (TFCL Mix arm: full-chain EER 22.20 → 17.39 by adding front-end-processed
  audio to training), then I3 (ADD-C), then I2 (SIL: positive on DF21 for 3 / 3 models, mixed elsewhere).
- **Largest addressable error mass:** heldout (I1, I3, I4, I6) and the saturated core (I2). G5 (English bona) has no
  augmentation answer in the literature I found: augmentation multiplies channels, not speakers (AUD G5).
- **One component per lever** (brief): I1 = APM NS + AGC as one platform module (they ship together, PYWRTC order
  "HP filter -> AEC -> NS -> AGC"). If CRITIQUE wants a stricter split, NS-only is the version to keep. I2 =
  gating + pause insertion, with insertion behind its own probability so it can be set to 0.
- **Interactions:** I1 + I2 + I3 sit at disjoint points of the chain (post-noise DSP, gating, codec / network), so
  a later combination run is well-defined. Each is screened alone against B2f (91.98 / 92.14, AUD §1).
