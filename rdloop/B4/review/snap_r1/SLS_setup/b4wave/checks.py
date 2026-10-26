"""
Self-checks of the b4wave package (rdloop/B4). Run from SLS_setup/:

    python -m b4wave.checks cpu                  every CPU check below (the supervisor's STATIC step)
    python -m b4wave.checks gpu --ckpt <path>    [GPU] base checkpoint in the b4wave model, flags off:
                                                 exact keys + identical logits to ModelSLS on a real
                                                 dev batch; each registered lever on: forward +
                                                 backward on lever-augmented training clips, finite
    python -m b4wave.checks overfit --lever <id> levers with params_added (none in B4: SKIP, exit 0)

    single CPU groups: module | flags | augmenter | reserve_names | imports | trainer | data |
                       apm | apm_balance | dtx | dtx_balance | codec_draw_order | opus_plc | timing |
                       dsp_slices
    as in b3wave:      model --ckpt <path> [GPU] | optim [GPU model build]
    all                = cpu

Every check prints `PASS <name>`, `FAIL <name>: <reason>` or `SKIP <name>: <reason>` (a lever whose
package is missing); the process exits non-zero on any FAIL. Measured values are printed on
indented lines below their check, also when it passes.

Lever parts plug in through the registry (`b4wave.augment.LEVERS`): the generic checks below
(plan defaults, registry vs parser, levers-off imports, timing, gpu forward / backward) cover every
registered lever automatically; lever-specific checks (selftests, balance, equivalence item 3 with
the real lever, codec draw order, ...) are added as their own groups and listed in CPU_GROUPS;
plan equivalence item 3 of a lever lives in `_equiv_<id>()`, which `checks augmenter` runs for every
registered lever.
"""

import argparse
import contextlib
import copy
import io
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

SLS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SLS_DIR.parent
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# the base trainer argv of rdloop/B4/cycle.json (B2f, entry module b3wave), relative to SLS_setup/
BASE_ARGV = shlex.split(
    "--train_data_path /mnt/paml-research/RTCFake/data/wav/train "
    "--train_protocol /mnt/paml-research/RTCFake/data/train_label.txt "
    "--dev_data_path /mnt/paml-research/RTCFake/data/wav/dev "
    "--dev_protocol ../data/dev_noisy_sim/dev_label_excl_halfB.txt "
    "--ssl_name Qwen/Qwen3-ASR-1.7B --n_layers 24 --batch_size 16 --num_epochs 8 --num_workers 64 "
    "--use_rtc_aug --aug_noise_dirs ../data/augmentation_train/rnnoise ../data/augmentation_train/esc50 "
    "--aug_rir_dirs ../data/augmentation_train/rirs --musan --musan_dir ../data/augmentation/musan "
    "--loudness_norm --loudness_norm_lufs -23 --optim adamw --lr 2.5e-6 --head_lr_mult 10 "
    "--weight_decay 0.01 --lr_scheduler warmup_cosine --warmup_frac 0.05 --lr_min 5e-7 --grad_clip 1.0 "
    "--ema_decay 0.9995 --local_eval_sets ../configs/local_eval_sets.yaml --earlystop_metric wf1 "
    "--earlystop_epoch 99 --keep_topk_by_wf1 7 --keep_epochs 8 --seed 1234 "
    "--track B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234")
BASE_CKPT = str(SLS_DIR / "exp/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623"
                         "/ckpt/epoch_8_dev_loss_0.144168.pth")

# rdloop/B4/plan_agreed.json: approved levers (chain order), their flags with the plan defaults (as
# argparse stores them), the wave arm's setting, the packages each may import, params_added
APPROVED = ("apm_ns", "dtx_gate", "opus_plc")
PLAN_FLAGS = {
    "--aug_apm_p": ("apm_ns", 0.0),
    "--aug_apm_ns_levels": ("apm_ns", [6, 12, 18, 21]),
    "--aug_dtx_p": ("dtx_gate", 0.0),
    "--aug_dtx_fill": ("dtx_gate", "mix"),
    "--aug_opus_plc_p": ("opus_plc", 0.0),
    "--aug_opus_loss": ("opus_plc", [0.01, 0.20]),
    "--aug_opus_burst": ("opus_plc", [1.0, 4.0]),
}
PLAN_WAVE_SET = {"apm_ns": {"--aug_apm_p": 0.3}, "dtx_gate": {"--aug_dtx_p": 0.3}, "opus_plc": {"--aug_opus_plc_p": 0.3}}
PLAN_PACKAGES = {"apm_ns": ("pywebrtc_audio",), "dtx_gate": ("webrtcvad",), "opus_plc": ()}
PLAN_PARAMS_ADDED = {"apm_ns": False, "dtx_gate": False, "opus_plc": False}
PLAN_OP_NAMES = {"filters": ("wrtc_ns", "wrtc_dtx", "opus_plc"), "codecs": ("libopus",)}
TIMING_LIMIT = 1.3                  # cycle.json limits.max_epoch_time_frac
TIMING_WARN = {"apm_ns": 1.15, "dtx_gate": 1.05}    # plan 1.2 stop rule / smoke bars: WARN, report to the owner

RESULTS = []


def report(name, ok, reason="", values=None):
    RESULTS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {reason}" if reason and not ok else ""), flush=True)
    for line in ([values] if isinstance(values, str) else (values or [])):
        print(f"    {line}", flush=True)
    return ok


def skip(name, reason):
    print(f"SKIP {name}: {reason}", flush=True)


@contextlib.contextmanager
def quiet():
    """Swallow the augmenter / dataset build prints (pool listings) of the objects under test."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def base_args(**over):
    """The base argv parsed by b4wave.train.build_parser (+ the trainer's ssl_path post-processing)."""
    from model.sls_model import ssl_path
    from b4wave.train import build_parser
    args = build_parser().parse_args(BASE_ARGV)
    args.ssl_name = ssl_path(args.ssl_name)
    for k, v in over.items():
        setattr(args, k, v)
    return args


class _Parsed(Exception):
    def __init__(self, ns):
        super().__init__("parsed")
        self.ns = ns


def b3wave_parse(argv):
    """b3wave.train's argparse result for `argv` (its parser lives inside main(): intercept parse_args)."""
    import b3wave.train as b3t
    orig = argparse.ArgumentParser.parse_args

    def fake(self, args=None, namespace=None):
        raise _Parsed(orig(self, argv, namespace))

    argparse.ArgumentParser.parse_args = fake
    try:
        b3t.main()
    except _Parsed as p:
        return p.ns
    finally:
        argparse.ArgumentParser.parse_args = orig
    raise RuntimeError("b3wave.train.main() did not reach parse_args")


def b4_group_actions(parser):
    groups = [g for g in parser._action_groups if g.title == "B4 wave"]
    return list(groups[0]._group_actions) if groups else None


def _synthetic_clips(n, rng, sr=16000, seconds=2.0):
    from b3wave.checks import _synthetic_clips as b3_clips
    return b3_clips(n, rng, sr, seconds)


# --------------------------------------------------------------------------- #
def check_module():
    import b4wave
    sha1, sha2 = b4wave.module_sha256(), b4wave.module_sha256()
    names = [p.name for p in b4wave.module_files()]
    on_disk = sorted(p.name for p in Path(b4wave.__file__).parent.glob("*.py"))
    ok = report("module_sha256_stable", sha1 == sha2 and re.fullmatch(r"[0-9a-f]{64}", sha1) is not None
                and names == on_disk and "checks.py" in names and "train.py" in names,
                f"{sha1} vs {sha2}; files {names}", [f"module_sha256 {sha1[:12]} over {names}"])
    ok &= report("module_version", isinstance(b4wave.__version__, str) and bool(b4wave.__version__),
                 f"{b4wave.__version__!r}", [f"b4wave {b4wave.__version__}"])
    src = (SLS_DIR / "b4wave" / "train.py").read_text(encoding="utf-8")
    ok &= report("trainer_prints_version_and_sha", 'print(f"b4wave: {B4WAVE_VERSION}")' in src
                 and 'print(f"b4wave sha256: {b4_sha[:12]}")' in src and '"module_sha256": b4_sha' in src
                 and '"b4wave_version": B4WAVE_VERSION' in src, "header / config.yaml lines missing in train.py")
    return ok


# --------------------------------------------------------------------------- #
def check_flags():
    from b4wave.augment import LEVERS, active_levers, b4_header_lines
    from b4wave.train import build_parser, validate_b4_args

    parser = build_parser()
    actions = b4_group_actions(parser)
    ok = report("b4_argument_group", actions is not None, "no 'B4 wave' argument group")
    if actions is None:
        return False
    opts = {a.option_strings[0]: a for a in actions}
    unplanned = sorted(set(opts) - set(PLAN_FLAGS))
    wrong = {o: (a.default, PLAN_FLAGS[o][1]) for o, a in opts.items() if o in PLAN_FLAGS and a.default != PLAN_FLAGS[o][1]}
    pending = sorted(set(PLAN_FLAGS) - set(opts))
    ok &= report("b4_flag_defaults_equal_plan", not unplanned and not wrong,
                 f"unplanned {unplanned}; default != plan {wrong}",
                 [f"implemented {sorted(opts)}", f"pending (lever not implemented yet) {pending}"])

    registered = list(LEVERS)
    bad_ids = [lid for lid in registered if lid not in APPROVED]
    order_ok = registered == [lid for lid in APPROVED if lid in registered]
    missing = {lid: [f for f, (l2, _) in PLAN_FLAGS.items() if l2 == lid and f not in opts] for lid in registered}
    missing = {k: v for k, v in missing.items() if v}
    dests = {a.dest for a in actions}
    unmapped = {lid: [d for d in [LEVERS[lid].prob_arg] + [a for a, _ in LEVERS[lid].arg_fields] if d not in dests]
                for lid in registered}
    unmapped = {k: v for k, v in unmapped.items() if v}
    ok &= report("lever_registry_vs_plan_and_parser", not bad_ids and order_ok and not missing and not unmapped,
                 f"not approved {bad_ids}; chain order ok {order_ok}; plan flags missing {missing}; "
                 f"registry args not in the B4 group {unmapped}", [f"registered levers {registered}"])

    # exploop reads the allowed flags by grepping add_argument in train.py (trainer.flag_sources)
    src = (SLS_DIR / "b4wave" / "train.py").read_text(encoding="utf-8")
    grepped = set(re.findall(r"add_argument\(\s*[\"'](--?[A-Za-z0-9_\-]+)[\"']", src))
    ok &= report("b4_flags_visible_to_exploop", set(opts) <= grepped, f"not grep-able: {sorted(set(opts) - grepped)}")

    b4ns = vars(base_args())
    b3ns = vars(b3wave_parse(BASE_ARGV))
    from model.sls_model import ssl_path
    b3ns["ssl_name"] = ssl_path(b3ns["ssl_name"])
    extra = sorted(set(b4ns) - set(b3ns))
    lost = sorted(set(b3ns) - set(b4ns))
    diff = {k: (b3ns[k], b4ns[k]) for k in b3ns if k in b4ns and b3ns[k] != b4ns[k]}
    ok &= report("parser_reproduces_b3wave_on_base_argv", not lost and not diff and set(extra) == dests,
                 f"lost {lost}; differ {diff}; extra {extra} vs B4 dests {sorted(dests)}",
                 [f"{len(b3ns)} b3wave args identical; B4 extras {extra}"])

    args = base_args()
    ok &= report("levers_off_on_base_argv", active_levers(args) == [] and b4_header_lines(args) == [],
                 f"active {active_levers(args)} headers {b4_header_lines(args)}")
    try:
        with quiet():
            validate_b4_args(parser, args)
        ok &= report("validate_b4_defaults", True)
    except SystemExit as exc:
        ok &= report("validate_b4_defaults", False, f"parser.error on the defaults ({exc})")
    return ok


# --------------------------------------------------------------------------- #
def _stub_classes():
    """A post-DSP stub lever and a codec-slot stub lever, to exercise the core hooks (not plan levers)."""
    from utils.rtc_augment import finalize
    from b4wave.augment import LeverSpec, RTCAugConfigB4, RTCAugmenterB4

    @dataclass
    class StubCfg(RTCAugConfigB4):
        stub_p: float = 0.0
        cstub_p: float = 0.0

    class StubAug(RTCAugmenterB4):
        draws = None

        def _lever__stub(self, out, sr):
            u = float(self._get_rng4().random())
            if self.draws is not None:
                self.draws.append(u)
            if u < self.cfg.stub_p:
                self._count("stub_hit")
                return finalize(out * np.float32(0.5), out)
            return out

        def _lever__cstub(self, out, rng, sr):
            self._count("cstub")
            return out.copy(), "stubapp"

    specs = [LeverSpec("_stub", "post_dsp", "stub_p", "stub_p", meta_keys=("stub_hit",),
                       filters=("wrtc_stub",), describe=lambda cfg: f"stub p={cfg.stub_p:g}",
                       header=lambda a: f"B4 stub: on p={a.stub_p:g}"),
             LeverSpec("_cstub", "codec", "cstub_p", "cstub_p", meta_keys=("cstub",))]
    return StubCfg, StubAug, specs


@contextlib.contextmanager
def registered(specs):
    from b4wave import augment as aug
    saved = dict(aug.LEVERS)
    try:
        for s in specs:
            aug.register(s)
        yield
    finally:
        aug.LEVERS.clear()
        aug.LEVERS.update(saved)


def _b4_fields(cfg):
    """{field: value} of an RTCAugConfigB4 (to rebuild it as a stub-lever config)."""
    from b4wave.augment import RTCAugConfigB4
    return {f.name: getattr(cfg, f.name) for f in fields(RTCAugConfigB4)}


def _cfg_dict(cfg):
    from b4wave.augment import b3_field_names
    d = {}
    for name in b3_field_names():
        v = getattr(cfg, name)
        d[name] = list(v) if isinstance(v, (list, tuple)) else v
    return d


def check_augmenter(n_clips=200, n_codec=10):
    from b3wave.augment import RTCAugmenterB3, build_rtc_augmenter_b3
    from b4wave.augment import RTCAugmenterB4, b4_config_from_args, build_rtc_augmenter_b4

    args = base_args()
    with quiet():
        b3 = build_rtc_augmenter_b3(args)
        b4 = build_rtc_augmenter_b4(args)
    ok = report("levers_off_returns_b3_object", type(b4) is RTCAugmenterB3 and b4.describe() == b3.describe()
                and b4.meta_keys == b3.meta_keys, f"type {type(b4).__name__}")
    cfg4 = b4_config_from_args(args)
    ok &= report("b4_config_equals_b3_config", _cfg_dict(cfg4) == _cfg_dict(b3.cfg),
                 f"{ {k: (v, _cfg_dict(b3.cfg)[k]) for k, v in _cfg_dict(cfg4).items() if v != _cfg_dict(b3.cfg)[k]} }")

    # (2) a forced RTCAugmenterB4 with no lever on is bit-identical to the B3 augmenter
    for profiles in ("fixed", "random"):
        for with_codecs, n in ((False, n_clips), (True, n_codec)):
            if profiles == "random" and not with_codecs:
                continue
            a = base_args(no_codec_aug=not with_codecs, aug_codec_profiles=profiles)
            with quiet():
                ref = build_rtc_augmenter_b3(a)
                forced = RTCAugmenterB4(b4_config_from_args(a))
            clips = _synthetic_clips(n, np.random.default_rng(7))
            same = sum(int(np.array_equal(ref(c.copy(), 16000), forced(c.copy(), 16000))) for c in clips)
            cond = (same == n and ref.stage_counts == forced.stage_counts and ref.describe() == forced.describe()
                    and ref.meta_keys == forced.meta_keys and forced.rng4 is None)
            ok &= report(f"forced_b4_bit_identical(codecs={with_codecs},profiles={profiles})", cond,
                         f"{same}/{n} identical; rng4 created {forced.rng4 is not None}; "
                         f"counts ref={ref.stage_counts} b4={forced.stage_counts}",
                         [f"{same}/{n} clips identical; stage_counts {dict(sorted(forced.stage_counts.items()))}"])

    # (3) core hook isolation with stub levers: rng4 leaves the parent stream untouched
    StubCfg, StubAug, specs = _stub_classes()
    with registered(specs):
        a = base_args(no_codec_aug=True)
        with quiet():
            ref = build_rtc_augmenter_b3(a)
        cfg = StubCfg(**_b4_fields(b4_config_from_args(a)), stub_p=0.5)
        with quiet():
            stub = StubAug(cfg)
        stub.draws = []
        clips = _synthetic_clips(n_clips, np.random.default_rng(11))
        same, hits = 0, 0
        for c in clips:
            x, y = ref(c.copy(), 16000), stub(c.copy(), 16000)
            same += int(np.array_equal(x, y))
        parent = {k: v for k, v in stub.stage_counts.items() if k != "stub_hit"}
        hits = stub.stage_counts.get("stub_hit", 0)
        expect = np.random.default_rng([1234, os.getpid(), 0xB4]).random(len(stub.draws)).tolist()
        n_short = sum(1 for c in clips if len(c) < 16)
        ok &= report("post_dsp_stub_parent_stream_untouched", parent == ref.stage_counts and 0 < hits < n_clips
                     and same == n_clips - hits and len(stub.draws) == n_clips - n_short,
                     f"parent counts equal {parent == ref.stage_counts}; hits {hits}; identical {same}",
                     [f"stub fired on {hits}/{n_clips} clips; the other {same} identical to B3; "
                      f"parent stage_counts equal to B3: {parent == ref.stage_counts}"])
        ok &= report("rng4_seed_is_seed_pid_0xB4", stub.draws == expect, "rng4 draws differ from default_rng([1234, pid, 0xB4])")
        r1 = stub._get_rng4()
        stub.rng4_pid = -1                       # a forked worker: re-created from the seed sequence
        r2 = stub._get_rng4()
        ok &= report("rng4_recreated_on_pid_change", r2 is not r1 and stub.rng4_pid == os.getpid())
        ok &= report("lever_meta_keys_and_describe", stub.meta_keys[-1] == "stub_hit"
                     and int(stub.counts_vector({})[-1]) == hits and stub.describe().endswith(" | stub p=0.5")
                     and stub.levers_on == ["_stub"], f"meta_keys tail {stub.meta_keys[-3:]} describe ...{stub.describe()[-40:]}")

        a2 = base_args(no_codec_aug=False)
        with quiet():
            cfg2 = StubCfg(**_b4_fields(b4_config_from_args(a2)), cstub_p=1.0)
            cst = StubAug(cfg2)
        for c in _synthetic_clips(5, np.random.default_rng(5)):
            cst(c, 16000)
        ok &= report("codec_slot_lever_dispatch", cst.stage_counts.get("codec:stubapp") == 5
                     and cst.stage_counts.get("cstub") == 5 and cst.codec_lever == "_cstub",
                     f"{cst.stage_counts}")

        # builder: a lever on -> RTCAugmenterB4 via the registry; header line; no RTC augmenter -> error
        from b4wave.augment import active_levers, b4_header_lines
        a3 = base_args(stub_p=0.25)
        ok &= report("registry_active_and_header", active_levers(a3) == ["_stub"]
                     and b4_header_lines(a3) == ["B4 stub: on p=0.25"], f"{active_levers(a3)} {b4_header_lines(a3)}")
        a4 = base_args(stub_p=0.25, use_rtc_aug=False, musan=False)
        try:
            build_rtc_augmenter_b4(a4)
            ok &= report("lever_without_rtc_augmenter_raises", False, "no error")
        except ValueError:
            ok &= report("lever_without_rtc_augmenter_raises", True)

    # plan equivalence item 3, per registered lever (the real lever with a stub backend)
    from b4wave.augment import LEVERS
    for lid in LEVERS:
        fn = globals().get(f"_equiv_{lid}")
        if fn is not None:
            ok &= fn(n_clips, n_codec)
    return ok


# --------------------------------------------------------------------------- #
def check_reserve_names():
    from b3wave.heldout_reserve import assert_training_ops
    from b4wave.augment import LEVERS
    ok = True
    try:
        assert_training_ops(codecs=list(PLAN_OP_NAMES["codecs"]), filters=list(PLAN_OP_NAMES["filters"]))
        ok &= report("plan_op_names_not_reserved", True, values=[f"{PLAN_OP_NAMES}"])
    except ValueError as exc:
        ok &= report("plan_op_names_not_reserved", False, str(exc))
    for lid, spec in LEVERS.items():
        try:
            assert_training_ops(codecs=list(spec.codecs), filters=list(spec.filters))
            ok &= report(f"reserve_ok({lid})", set(spec.filters) <= set(PLAN_OP_NAMES["filters"])
                         and set(spec.codecs) <= set(PLAN_OP_NAMES["codecs"]), f"unplanned ops {spec.filters} {spec.codecs}")
        except ValueError as exc:
            ok &= report(f"reserve_ok({lid})", False, str(exc))
    for name, kw in (("vad_gate", {"filters": ["vad_gate"]}), ("packet_loss", {"filters": ["packet_loss"]}),
                     ("afftdn", {"filters": ["afftdn"]}), ("anlmdn", {"filters": ["anlmdn"]}), ("g722", {"codecs": ["g722"]})):
        try:
            assert_training_ops(**kw)
            ok &= report(f"reserved_still_raises({name})", False, "no error")
        except ValueError:
            ok &= report(f"reserved_still_raises({name})", True)
    return ok


# --------------------------------------------------------------------------- #
IMPORT_PROBE = r"""
import contextlib, io, json, sys
import numpy as np
sys.path.insert(0, ".")
from b4wave.checks import base_args, _synthetic_clips
from b4wave.data import build_augmenter_b4
from b4wave.augment import build_rtc_augmenter_b4
with contextlib.redirect_stdout(io.StringIO()):
    args = base_args(no_codec_aug=True)
    aug, rtc = build_augmenter_b4(args)
    for c in _synthetic_clips(20, np.random.default_rng(3)):
        aug(c, 16000)
maps = open("/proc/self/maps").read()
mods = sorted(m for m in sys.modules if m.split(".")[0] in WATCH or m in WATCH)
print("PROBE " + json.dumps({"modules": mods, "libopus_mapped": "libopus" in maps, "type": type(rtc).__name__}))
"""


def check_imports():
    """Levers off: no lever package or lever module is imported and no libopus handle exists (fresh process)."""
    from b4wave.augment import LEVERS
    watch = sorted({m for pk in PLAN_PACKAGES.values() for m in pk}
                   | {"b4wave.dsp_apm", "b4wave.dsp_dtx", "b4wave.opus_plc"}
                   | {m for s in LEVERS.values() for m in s.modules})
    code = f"WATCH = {watch!r}\n" + IMPORT_PROBE
    r = subprocess.run([sys.executable, "-c", code], cwd=str(SLS_DIR), capture_output=True, text=True, timeout=600)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("PROBE ")), None)
    if r.returncode != 0 or line is None:
        return report("levers_off_imports_nothing", False, f"probe rc {r.returncode}: {r.stderr[-400:]}")
    res = json.loads(line[6:])
    return report("levers_off_imports_nothing", not res["modules"] and not res["libopus_mapped"]
                  and res["type"] == "RTCAugmenterB3", f"{res}", [f"watched {watch}: {res}"])


# --------------------------------------------------------------------------- #
def check_trainer():
    from torch import nn
    from b4wave.train import add_class_meta, b4_class_row, train_epoch

    meta = torch.tensor([[1, 0, 2], [0, 1, 0], [3, 0, 1], [0, 0, 1]], dtype=torch.int32)
    y = torch.tensor([1, 0, 1, 0])
    acc = {"bona": None, "spoof": None, "n": {"bona": 0, "spoof": 0}}
    add_class_meta(acc, meta, y)
    add_class_meta(acc, meta[:2], y[:2])
    ok = report("add_class_meta_sums", acc["bona"].tolist() == [5, 0, 5] and acc["spoof"].tolist() == [0, 2, 1]
                and acc["n"] == {"bona": 3, "spoof": 3}, f"{acc}")
    row, text = b4_class_row(acc, ["stage:musan", "codec:qq", "wrtc_x"], ["_stub"])
    ok &= report("b4_class_row_format", text == "B4 aug-by-class bona: n=3 stage:musan=5 wrtc_x=5 | spoof: n=3 codec:qq=2 wrtc_x=1"
                 and row["aug_by_class"]["bona"]["counts"] == {"stage:musan": 5, "wrtc_x": 5}
                 and json.loads(json.dumps(row)) == row, text, [text])
    for pat, line in ((r"^Epoch (\d+)/(\d+) TrainLoss=", "Epoch 3 " + text), (r"^Epoch (\d+) LocalWF1=", "Epoch 3 " + text)):
        ok &= report(f"b4_line_not_parsed_as_epoch_row({pat})", re.match(pat, line) is None)

    # telemetry on/off leaves the training step bit-identical
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(12, 2)

        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))

    g = torch.Generator().manual_seed(0)
    batches = [{"x": torch.randn(4, 3, 4, generator=g), "y": torch.tensor([1, 0, 0, 1]),
                "utt": [f"u{i}" for i in range(4)], "meta": torch.randint(0, 3, (4, 5), generator=g, dtype=torch.int32)}
               for _ in range(3)]
    outs = []
    for track in (False, True):
        torch.manual_seed(0)
        model = Toy()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        with quiet():
            loss, accu, stats = train_epoch(batches, model, opt, torch.device("cpu"),
                                            nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9])),
                                            track_by_class=track)
        outs.append((loss, accu, [p.detach().clone() for p in model.parameters()], stats))
    (l0, a0, p0, s0), (l1, a1, p1, s1) = outs
    tot = sum(b["meta"].sum(0) for b in batches)
    ok &= report("train_epoch_by_class_no_effect", l0 == l1 and a0 == a1 and all(torch.equal(x, y) for x, y in zip(p0, p1))
                 and s0["meta_by_class"] is None and s1["meta_by_class"] is not None
                 and torch.equal(s1["meta_by_class"]["bona"] + s1["meta_by_class"]["spoof"], tot.to(torch.int64))
                 and torch.equal(s0["meta"], s1["meta"]) and s1["meta_by_class"]["n"] == {"bona": 6, "spoof": 6},
                 f"loss {l0} vs {l1}; by_class {s1['meta_by_class']}")
    return ok


# --------------------------------------------------------------------------- #
def check_data(subset=0.002):
    from b3wave.data import SpoofAudioDatasetB3, build_dataset_b3
    from b3wave.augment import RTCAugmenterB3
    from utils.data_utils import SpoofAudioDataset
    from b4wave.data import build_dataset_b4

    args = base_args(train_subset_frac=subset)
    if not Path(args.train_protocol).is_file() or not Path(args.dev_protocol).is_file():
        skip("data_builders_levers_off", f"protocols missing ({args.train_protocol}, {args.dev_protocol})")
        return True
    try:
        with quiet():
            d3, f3, l3, i3 = build_dataset_b3(args.train_protocol, args.train_data_path, "train", args)
            d4, f4, l4, i4 = build_dataset_b4(args.train_protocol, args.train_data_path, "train", args)
            v3, g3, _, _ = build_dataset_b3(args.dev_protocol, args.dev_data_path, "dev", args)
            v4, g4, _, _ = build_dataset_b4(args.dev_protocol, args.dev_data_path, "dev", args)
    except OSError as exc:
        skip("data_builders_levers_off", f"cannot build the datasets offline ({exc})")
        return True
    ok = report("train_dataset_levers_off_equals_b3", type(d4) is SpoofAudioDatasetB3 and f4 == f3 and i4 == i3
                and l4 == l3 and type(d4.rtc_augmenter) is RTCAugmenterB3 and d4.meta_keys == d3.meta_keys
                and d4.augmenter.describe() == d3.augmenter.describe()
                and (d4.loudness.describe() if d4.loudness else None) == (d3.loudness.describe() if d3.loudness else None),
                f"files {len(f4)} vs {len(f3)}; info {i4} vs {i3}", [f"{len(f4)} training clips, info {i4}"])
    ok &= report("dev_dataset_equals_b3", type(v4) is SpoofAudioDataset and g4 == g3 and len(v4) == len(v3),
                 f"{type(v4).__name__} {len(g4)} vs {len(g3)}")
    # the same items through both training datasets: identical audio after load + augment
    same = 0
    n = min(3, len(f4))
    for i in range(n):
        a3, sr = d3._load(d3.base_dir / f3[i])
        a4, _ = d4._load(d4.base_dir / f4[i])
        same += int(np.array_equal(d3._augment(a3, sr), d4._augment(a4, sr)))
    ok &= report("train_items_bit_identical", same == n, f"{same}/{n}", [f"{same}/{n} real clips identical after load + augment (+codec)"])
    return ok


# --------------------------------------------------------------------------- #
# lever apm_ns (plan 2.1)
APM_HEADER_RE = r"^B4 APM NS: on p="          # plan expect_regex
APM_LEVELS = (6, 12, 18, 21)
APM_BALANCE_WARN_DB = {"noise": 2.0, "none": 4.0}      # |bona - spoof| of the mean removed dB (plan 2.1)


@contextlib.contextmanager
def identity_ns():
    """dsp_apm with an identity NS backend (no pywebrtc_audio): wrtc_ns returns its input; records levels."""
    from b4wave import dsp_apm
    saved = (dsp_apm.wrtc_ns, dsp_apm.selftest, dsp_apm.summary)
    calls = []

    def ident(audio, level_db, sr=16000):
        calls.append(dsp_apm.check_level(level_db))
        return np.asarray(audio, dtype=np.float32).reshape(-1).copy()

    dsp_apm.wrtc_ns = ident
    dsp_apm.selftest = lambda levels=APM_LEVELS: {}
    dsp_apm.summary = lambda vals: "identity stub"
    try:
        yield calls
    finally:
        dsp_apm.wrtc_ns, dsp_apm.selftest, dsp_apm.summary = saved


def _apm_expected_levels(n_clips, p, levels, seed=1234):
    """The levels rng4 = default_rng([seed, pid, 0xB4]) picks for n clips: random() < p, then integers(len)."""
    r = np.random.default_rng([seed, os.getpid(), 0xB4])
    out = []
    for _ in range(n_clips):
        if r.random() < p:
            out.append(int(levels[int(r.integers(len(levels)))]))
    return out


def _parent_counts(counts):
    return {k: v for k, v in counts.items() if not (k.startswith("wrtc_ns") or k == "fail_wrtc_ns")}


def _equiv_apm_ns(n_clips=200, n_codec=10):
    """Plan 2.1 equivalence item 3: apm_p = 1.0 with an identity stub NS -> the outputs and the parent stage:* /
    codec counts equal the B3 augmenter's (rng4 leaves the parent stream untouched); levels follow rng4."""
    from b3wave.augment import build_rtc_augmenter_b3
    from b4wave.augment import RTCAugmenterB4, build_rtc_augmenter_b4
    ok = True
    for with_codecs, n in ((False, n_clips), (True, n_codec)):
        a = base_args(no_codec_aug=not with_codecs, aug_apm_p=1.0)
        with quiet():
            ref = build_rtc_augmenter_b3(a)
        with identity_ns() as calls:
            with quiet():
                aug = build_rtc_augmenter_b4(a)
            clips = _synthetic_clips(n, np.random.default_rng(13))
            same = sum(int(np.array_equal(ref(c.copy(), 16000), aug(c.copy(), 16000))) for c in clips)
        n_ns = sum(aug.stage_counts.get(f"wrtc_ns{d}", 0) for d in APM_LEVELS)
        expect = _apm_expected_levels(n, 1.0, APM_LEVELS)
        cond = (type(aug) is RTCAugmenterB4 and same == n and _parent_counts(aug.stage_counts) == ref.stage_counts
                and n_ns == n and calls == expect and aug.stage_counts.get("fail_wrtc_ns", 0) == 0)
        ok &= report(f"apm_ns_identity_parent_stream_untouched(codecs={with_codecs})", cond,
                     f"{same}/{n} identical; parent counts equal {_parent_counts(aug.stage_counts) == ref.stage_counts}; "
                     f"wrtc_ns {n_ns}/{n}; levels follow rng4 {calls == expect}",
                     [f"apm_p=1.0, identity NS: {same}/{n} clips identical to B3; parent counts "
                      f"{dict(sorted(_parent_counts(aug.stage_counts).items()))} == B3; levels "
                      f"{ {d: calls.count(d) for d in APM_LEVELS} } = default_rng([1234, pid, 0xB4]) draws"])
    return ok


def check_apm(n_clips=200):
    """apm_ns: the selftest's named asserts with their values; level validation; the lever with the real NS."""
    import importlib.util
    from b3wave.augment import RTCAugmenterB3, build_rtc_augmenter_b3
    from b4wave import dsp_apm
    from b4wave.augment import RTCAugConfigB4, RTCAugmenterB4, b4_config_from_args, b4_header_lines, build_rtc_augmenter_b4
    from b4wave.train import build_parser, validate_b4_args

    ok = True
    # level validation (no backend needed)
    parser = build_parser()
    with quiet():
        try:
            parser.parse_args(BASE_ARGV + ["--aug_apm_p", "0.3", "--aug_apm_ns_levels", "6", "9"])
            rej_parse = False
        except SystemExit:
            rej_parse = True
        bad = base_args(aug_apm_p=0.3, aug_apm_ns_levels=[])
        try:
            validate_b4_args(parser, bad)
            rej_empty = False
        except SystemExit:
            rej_empty = True
        try:
            validate_b4_args(parser, base_args(aug_apm_p=0.3, aug_apm_ns_levels=[7]))
            rej_val = False
        except SystemExit:
            rej_val = True
    try:
        dsp_apm.wrtc_ns(np.zeros(1600, np.float32), 9)
        rej_dsp = False
    except ValueError:
        rej_dsp = True
    with identity_ns():
        try:
            with quiet():
                RTCAugmenterB4(RTCAugConfigB4(**dict(_b4_fields(b4_config_from_args(base_args())), apm_p=0.3,
                                                     apm_ns_levels=(6, 9))))
            rej_aug = False
        except ValueError:
            rej_aug = True
    ok &= report("apm_rejects_levels_outside_6_12_18_21", rej_parse and rej_empty and rej_val and rej_dsp and rej_aug,
                 f"argparse {rej_parse}, empty {rej_empty}, validate {rej_val}, wrtc_ns {rej_dsp}, augmenter {rej_aug}")
    with quiet():
        off = build_rtc_augmenter_b4(base_args(aug_apm_ns_levels=[9]))
    ok &= report("apm_levels_unread_when_p0", type(off) is RTCAugmenterB3, f"{type(off).__name__}")

    if importlib.util.find_spec("pywebrtc_audio") is None:
        skip("apm_selftest / apm_lever_real_ns", "package pywebrtc_audio missing")
        return ok

    # the selftest (plan O2.2): every named assert reported with its measured values
    t0 = time.process_time()
    vals = dsp_apm.measure()
    t_self = time.process_time() - t0
    lv = vals["levels"]
    lines = {
        "shape_finite": [f"level {l}: (length ok, finite) {vals['shape_finite'][l]}" for l in lv],
        "no_gain": [f"150 Hz harmonic complex, 4 Hz on/off, -30 dBFS: output - input RMS (dB) "
                    + " ".join(f"{l}:{vals['no_gain'][l]:+.3f}" for l in lv) + f"  (<= +{dsp_apm.NO_GAIN_TOL_DB})"],
        "monotone": [f"white noise -30 dBFS: removed (dB) " + " ".join(f"{l}:{vals['monotone'][l]:.3f}" for l in lv)
                     + f"  (non-decreasing within {dsp_apm.MONOTONE_TOL_DB}; removed(21) > removed(6))"],
        "no_hpf": [f"level {l}: attenuation 31.25-93.75 Hz {vals['no_hpf'][l]['att_31_94Hz']:.3f} dB, 1-4 kHz "
                   f"{vals['no_hpf'][l]['att_1_4kHz']:.3f} dB, diff {vals['no_hpf'][l]['diff']:+.3f}  (|diff| <= "
                   f"{dsp_apm.NO_HPF_TOL_DB})" for l in lv],
        "aligned": [f"output lag vs input (samples, latency {dsp_apm.LATENCY} compensated) "
                    + " ".join(f"{l}:{vals['aligned'][l]}" for l in lv)],
    }
    for name in dsp_apm.ASSERTS:
        try:
            dsp_apm.verify(vals, (name,))
            ok &= report(f"apm_selftest[{name}]", True, values=lines[name])
        except AssertionError as exc:
            ok &= report(f"apm_selftest[{name}]", False, str(exc), lines[name])
    print(f"    selftest CPU {t_self:.2f} s (incl. the scipy import on first use)", flush=True)

    # the lever with the real NS on synthetic clips (no codec): header, describe, meta keys, rng4 levels,
    # parent counts = B3, every processed clip changed and not louder, the others bit-identical
    a = base_args(no_codec_aug=True, aug_apm_p=0.3)
    with quiet():
        ref = build_rtc_augmenter_b3(a)
        aug = build_rtc_augmenter_b4(a)
    head = b4_header_lines(a)
    ok &= report("apm_header_matches_expect_regex", len(head) == 1 and re.match(APM_HEADER_RE, head[0]) is not None
                 and head[0] == "B4 APM NS: on p=0.3 levels_db=[6, 12, 18, 21] backend=pywebrtc_audio.NoiseSuppressor "
                                "hpf=off agc=off", f"{head}", head)
    ok &= report("apm_describe_and_meta_keys", type(aug) is RTCAugmenterB4
                 and aug.describe() == ref.describe() + " | wrtc_ns p=0.3 levels_db=[6, 12, 18, 21]"
                 and aug.meta_keys == ref.meta_keys + [f"wrtc_ns{d}" for d in APM_LEVELS] + ["fail_wrtc_ns"],
                 f"describe ...{aug.describe()[-60:]}; meta tail {aug.meta_keys[-6:]}",
                 [f"describe suffix: {aug.describe()[len(ref.describe()):]}", f"meta keys + {aug.meta_keys[len(ref.meta_keys):]}"])
    clips = _synthetic_clips(n_clips, np.random.default_rng(17))
    n_same, n_changed, gain_max = 0, 0, -np.inf
    for c in clips:
        x, y = ref(c.copy(), 16000), aug(c.copy(), 16000)
        if np.array_equal(x, y):
            n_same += 1
        else:
            n_changed += 1
            gain_max = max(gain_max, 10 * np.log10(float(np.mean(y.astype(np.float64) ** 2)) / float(np.mean(x.astype(np.float64) ** 2))))
    expect = _apm_expected_levels(n_clips, 0.3, APM_LEVELS)
    got = {d: aug.stage_counts.get(f"wrtc_ns{d}", 0) for d in APM_LEVELS}
    want = {d: expect.count(d) for d in APM_LEVELS}
    cond = (got == want and _parent_counts(aug.stage_counts) == ref.stage_counts and n_changed == len(expect)
            and n_same == n_clips - len(expect) and aug.stage_counts.get("fail_wrtc_ns", 0) == 0
            and 35 <= len(expect) <= 85 and gain_max <= 0.5)
    ok &= report("apm_lever_real_ns", cond,
                 f"levels {got} vs rng4 {want}; parent counts equal {_parent_counts(aug.stage_counts) == ref.stage_counts}; "
                 f"changed {n_changed}, identical {n_same}; fail {aug.stage_counts.get('fail_wrtc_ns', 0)}; max gain {gain_max:+.2f} dB",
                 [f"p=0.3 on {n_clips} synthetic clips: processed {n_changed} (rate {n_changed / n_clips:.3f}), levels {got} "
                  f"= rng4 draws; the other {n_same} bit-identical to B3; parent counts == B3; "
                  f"max output/input energy of a processed clip {gain_max:+.2f} dB; fail_wrtc_ns 0"])

    # chain with the codec: NS -> codec on 5 clips (fixed profiles), forced p = 1
    a = base_args(no_codec_aug=False, aug_apm_p=1.0)
    with quiet():
        ref = build_rtc_augmenter_b3(a)
        aug = build_rtc_augmenter_b4(a)
    for c in _synthetic_clips(5, np.random.default_rng(19)):
        ref(c.copy(), 16000)
        y = aug(c.copy(), 16000)
        assert len(y) == len(c) and np.all(np.isfinite(y))
    codec = {k: v for k, v in aug.stage_counts.items() if k.startswith("codec:")}
    ok &= report("apm_then_codec_chain", sum(aug.stage_counts.get(f"wrtc_ns{d}", 0) for d in APM_LEVELS) == 5
                 and codec == {k: v for k, v in ref.stage_counts.items() if k.startswith("codec:")} and sum(codec.values()) == 5,
                 f"{aug.stage_counts}", [f"p=1.0 with the codec: {dict(sorted(aug.stage_counts.items()))}"])
    return ok


def _frame_db(x, frame=160):
    n = len(x) // frame
    e = np.mean(np.asarray(x[:n * frame], np.float64).reshape(n, frame) ** 2, axis=1)
    return e, 10 * np.log10(e + 1e-12)


def _removed_db(e_in, e_out, mask=None):
    if mask is not None:
        e_in, e_out = e_in[mask], e_out[mask]
    s_in, s_out = float(e_in.sum()), float(e_out.sum())
    if s_in <= 1e-10 * max(len(e_in), 1):          # no frame selected, or digital silence: nothing to remove
        return None
    return 10 * np.log10(s_in / max(s_out, 1e-20))


def check_apm_balance(per_cell=25, seed=0):
    """
    Plan 2.1 (critic O1.5): realised removal per class on 200 training clips (25 per class x src x lang cell), in
    two conditions -- (none) the loaded clip, (noise) a forced training-pool noise stage at U(5, 20) dB SNR --
    i.e. 200 + 200 clip runs, at each NS level: removed dB over the whole clip and over the input's low-energy
    frames (10 ms frames < p95 - 30 dB). Prints mean / quartiles per class and the bona - spoof difference of
    the means; `APM_BALANCE WARN` when |diff| > 2 dB (noise) or > 4 dB (none). A warning, not a failure.
    """
    import importlib.util
    import librosa
    from utils.data_utils import build_loudness, read_protocol
    from utils.rtc_augment import add_noise, finalize
    from b3wave.augment import build_rtc_augmenter_b3
    from b3wave.data import utt_meta

    if importlib.util.find_spec("pywebrtc_audio") is None:
        skip("apm_balance", "package pywebrtc_audio missing")
        return True
    args = base_args()
    if not Path(args.train_protocol).is_file() or not Path(args.train_data_path).is_dir():
        skip("apm_balance", f"training data missing ({args.train_protocol})")
        return True
    from b4wave.dsp_apm import wrtc_ns
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
    noise_stages = sorted(b3.noise_files)
    loud = build_loudness(args)
    nrng = np.random.default_rng(seed + 1)
    res = {}                                       # (cls, cond, lvl, measure) -> [dB]
    n_undef = {}
    t0 = time.time()
    for (cls, src, lang), utts in pick.items():
        for u in utts:
            audio, sr = librosa.load(str(Path(args.train_data_path) / u), sr=16000)
            audio = audio.astype(np.float32)
            if loud is not None:
                audio = loud(audio, sr)
            stage = noise_stages[int(nrng.integers(len(noise_stages)))]
            noisy = finalize(add_noise(audio, b3.noise_files[stage], nrng, (5.0, 20.0), sr), audio)
            for cond, x in (("none", audio), ("noise", noisy)):
                e_in, db_in = _frame_db(x)
                low = db_in < (np.percentile(db_in, 95) - 30.0) if len(db_in) else None
                for lvl in APM_LEVELS:
                    y = finalize(wrtc_ns(x, lvl), x)
                    e_out, _ = _frame_db(y)
                    for meas, mask in (("whole", None), ("low", low)):
                        v = _removed_db(e_in, e_out, mask)
                        if v is None:
                            n_undef[(cls, cond, meas)] = n_undef.get((cls, cond, meas), 0) + (1 if lvl == APM_LEVELS[0] else 0)
                        else:
                            res.setdefault((cls, cond, lvl, meas), []).append(v)
    n_cls = {c: sum(len(v) for k, v in pick.items() if k[0] == c) for c in ("bona", "spoof")}
    lines = [f"{sum(n_cls.values())} training clips ({n_cls}; {per_cell} per class x src x lang cell), conditions none / "
             f"noise (forced pool stage {noise_stages}, SNR U(5, 20) dB): {2 * sum(n_cls.values())} clip runs x "
             f"{len(APM_LEVELS)} levels in {time.time() - t0:.0f}s"]
    warns = []
    for cond in ("none", "noise"):
        for meas in ("whole", "low"):
            for lvl in APM_LEVELS:
                stats = {}
                for cls in ("bona", "spoof"):
                    v = np.asarray(res.get((cls, cond, lvl, meas), []), np.float64)
                    stats[cls] = (float(v.mean()) if len(v) else float("nan"),
                                  np.percentile(v, [25, 50, 75]).round(2).tolist() if len(v) else [], len(v))
                diff = stats["bona"][0] - stats["spoof"][0]
                lines.append(f"{cond:5s} {meas:5s} {lvl:2d} dB: bona mean {stats['bona'][0]:6.2f} q {stats['bona'][1]} "
                             f"(n {stats['bona'][2]}) | spoof mean {stats['spoof'][0]:6.2f} q {stats['spoof'][1]} "
                             f"(n {stats['spoof'][2]}) | bona - spoof {diff:+.2f} dB")
                if math.isfinite(diff) and abs(diff) > APM_BALANCE_WARN_DB[cond]:
                    warns.append(f"APM_BALANCE WARN cond={cond} measure={meas} level={lvl} dB: bona - spoof removed "
                                 f"{diff:+.2f} dB (|diff| > {APM_BALANCE_WARN_DB[cond]:g})")
    if n_undef:
        lines.append(f"low-energy measure undefined (no 10 ms frame below p95 - 30 dB, or digital silence only) "
                     f"per (class, cond, measure): "
                     f"{ {'/'.join(k): v for k, v in sorted(n_undef.items())} }")
    ok = report("apm_balance", bool(res), "no clip measured", lines)
    for w in warns:
        print(w, flush=True)
    if not warns:
        print("    no APM_BALANCE WARN", flush=True)
    return ok


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
    grid = [f"mode {m}: " + " | ".join(f"f{f} " + " ".join(f"h{r['hang']:g} {r['gated_frac']:.3f}" for r in runs
                                                           if r["mode"] == m and r["frame"] == f)
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
    """(SNR dB, lag, sum ref^2, sum (ref - test)^2, share of the error energy above 3 kHz): the integer lag
    |lag| <= max_lag with the highest 10 log10(sum ref^2 / sum (ref - test)^2) on the overlap (lag > 0: ref is
    delayed by lag samples relative to test)."""
    ref = np.asarray(ref, np.float64)
    test = np.asarray(test, np.float64)
    best = (-np.inf, 0, 0.0, 0.0)
    for lag in range(-max_lag, max_lag + 1):
        a, b = (ref[lag:], test[:len(test) - lag]) if lag >= 0 else (ref[:lag], test[-lag:])
        m = min(len(a), len(b))
        if m <= 0:
            continue
        a, b = a[:m], b[:m]
        e_ref, e_err = float(np.sum(a ** 2)), float(np.sum((a - b) ** 2))
        s = 10 * np.log10(e_ref / max(e_err, 1e-30))
        if s > best[0]:
            best = (float(s), lag, e_ref, e_err)
    lag = best[1]
    a, b = (ref[lag:], test[:len(test) - lag]) if lag >= 0 else (ref[:lag], test[-lag:])
    m = min(len(a), len(b))
    spec = np.abs(np.fft.rfft(a[:m] - b[:m])) ** 2
    hi = float(spec[np.fft.rfftfreq(m, 1 / 16000) >= 3000].sum() / max(spec.sum(), 1e-30))
    return best + (hi,)


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
    rows, parity_bad = [], []
    profiles = {}
    for u, audio in _training_clips(3, seed=0):
        for app, (kbps, cutoffs, frames, application) in ra.CODEC_PROFILES.items():
            prof = {"app": app, "kbps": kbps[0], "cutoff": cutoffs[0], "frame_ms": max(frames), "application": application}
            profiles[app] = prof
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
            pair = _snr_aligned(y_ff, y_in)
            vs_in = _snr_aligned(y_ff, audio)
            rows.append({"utt": u, "app": app, "pair": pair, "in": vs_in, "same_pk": same_pk, "n_ff": len(fp),
                         "n_in": len(info["packets"])})
            if not (len(fp) == len(info["packets"]) and same_pk == len(fp)):
                parity_bad.append((u, app, same_pk, len(fp), len(info["packets"])))
    # the bar per profile, on the energies pooled over the 3 clips (plan 2.3 "on every profile"; v2's table has one
    # value per profile); every clip x profile value is printed, and a clip below the bar on its own is flagged
    pooled, bad, clip_below = {}, [], []
    for app in ra.CODEC_PROFILES:
        rs = [r for r in rows if r["app"] == app]
        s_pair = 10 * np.log10(sum(r["pair"][2] for r in rs) / max(sum(r["pair"][3] for r in rs), 1e-30))
        s_in = 10 * np.log10(sum(r["in"][2] for r in rs) / max(sum(r["in"][3] for r in rs), 1e-30))
        need = max(OPUS_SNR_FLOOR_DB, s_in + OPUS_SNR_MARGIN_DB)
        pooled[app] = (s_pair, s_in, need)
        if s_pair < need:
            bad.append((app, round(s_pair, 2), round(need, 2)))
        for r in rs:
            if r["pair"][0] < max(OPUS_SNR_FLOOR_DB, r["in"][0] + OPUS_SNR_MARGIN_DB):
                clip_below.append(f"{r['utt'].split('/')[-1]}:{app}")
    margins = {app: v[0] - v[2] for app, v in pooled.items()}
    worst = min(margins, key=margins.get)
    ok &= report("opus_zero_loss_attribution", not bad, f"profiles below the bar {bad}",
                 [f"3 training clips (seeded pick, loudness -23 LUFS) x 7 app profiles at the lowest kbps / first cutoff / longest "
                  f"frame (v2's profiles); SNR at the best integer lag |lag| <= {OPUS_MAX_LAG}; per profile on the energies pooled "
                  f"over the 3 clips: SNR(ffmpeg, in-process) >= max({OPUS_SNR_FLOOR_DB:g}, SNR(ffmpeg, input) + "
                  f"{OPUS_SNR_MARGIN_DB:g}); min margin {margins[worst]:+.2f} dB ({worst})"]
                 + [f"{app:8s} {pr['kbps']}k cut {pr['cutoff']} {pr['frame_ms']} ms {pr['application']}: pooled SNR(ffmpeg, "
                    f"in-process) {pooled[app][0]:.2f} | SNR(ffmpeg, input) {pooled[app][1]:.2f} | bar {pooled[app][2]:.2f} | "
                    f"margin {margins[app]:+.2f} | per clip " + ", ".join(
                        f"{r['pair'][0]:.2f} (lag {r['pair'][1]:+d}, err >3 kHz {100 * r['pair'][4]:.0f}%; vs input {r['in'][0]:.2f})"
                        for r in rows if r["app"] == app) for app, pr in profiles.items()]
                 + [f"note: clip x profile values below the per-clip bar: {clip_below or 'none'} (the SILK-layer decoder / "
                    f"resampler residual of plan 2.3's risks; reported, the bar is per profile)"])
    ok &= report("opus_encoder_parity_with_ffmpeg_packets", not parity_bad, f"{parity_bad}",
                 [f"{len(rows)} runs: in-process packets byte-identical to the packets in ffmpeg's Ogg stream in "
                  f"{sum(r['same_pk'] == r['n_ff'] == r['n_in'] for r in rows)}/{len(rows)} (packets per run "
                  f"{sorted({r['n_in'] for r in rows})}); ffmpeg's decode lags the input by {sorted({r['in'][1] for r in rows})} "
                  f"samples (its 4.2 Ogg pre-skip), the in-process decode by {sorted({r['in'][1] - r['pair'][1] for r in rows})} "
                  f"(3 x look-ahead trimmed)"])
    return ok


# --------------------------------------------------------------------------- #
def _cpu_per_clip(dataset, idx):
    """(self CPU-s, children CPU-s) per item of dataset[idx] (load + augment + featurize)."""
    import resource
    c0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.process_time()
    for i in idx:
        dataset[i]
    t1 = time.process_time()
    c1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (t1 - t0) / len(idx), ((c1.ru_utime + c1.ru_stime) - (c0.ru_utime + c0.ru_stime)) / len(idx)


def check_timing(n=40, subset=0.01):
    """
    CPU-s per training clip of the loader path, one process (torch / OMP threads 1), levers off vs
    each registered lever forced on (p = 1) on the same clips; projected saturated-loader epoch
    factor = 1 + p_wave * (cost_on - cost_off) / cost_off at the plan's wave probability.
    """
    from b4wave.augment import LEVERS
    from b4wave.data import build_dataset_b4

    torch.set_num_threads(1)
    args = base_args(train_subset_frac=subset)
    if not Path(args.train_protocol).is_file():
        skip("timing", f"training protocol missing ({args.train_protocol})")
        return True
    with quiet():
        base, files, _, _ = build_dataset_b4(args.train_protocol, args.train_data_path, "train", args)
    idx = np.random.default_rng(0).choice(len(files), size=min(n, len(files)), replace=False).tolist()
    base[idx[0]]                                  # warm-up (feature extractor, pools)
    s0, c0 = _cpu_per_clip(base, idx)
    cost0 = s0 + c0
    ok = report("timing_levers_off", cost0 > 0, f"{cost0}", [f"levers off: {cost0:.4f} CPU-s/clip (self {s0:.4f} + children {c0:.4f}) over {len(idx)} clips"])
    for lid, spec in LEVERS.items():
        a = base_args(train_subset_frac=subset, **{spec.prob_arg: 1.0})
        try:
            with quiet():
                ds, _, _, _ = build_dataset_b4(a.train_protocol, a.train_data_path, "train", a)
        except ImportError as exc:
            skip(f"timing({lid})", f"package missing ({exc})")
            continue
        ds[idx[0]]
        s1, c1 = _cpu_per_clip(ds, idx)
        p_wave = float(next(iter(PLAN_WAVE_SET[lid].values())))
        factor = 1.0 + p_wave * ((s1 + c1) - cost0) / cost0
        ok &= report(f"timing({lid})", factor <= TIMING_LIMIT, f"projected epoch factor {factor:.3f} > {TIMING_LIMIT}",
                     [f"{lid} forced on: {s1 + c1:.4f} CPU-s/clip (self {s1:.4f} + children {c1:.4f}); "
                      f"projected saturated-loader epoch factor at p={p_wave:g}: {factor:.3f} (limit {TIMING_LIMIT})"])
        if lid in TIMING_WARN and factor > TIMING_WARN[lid]:
            print(f"WARN timing({lid}): projected factor {factor:.3f} > {TIMING_WARN[lid]} (plan stop rule / smoke bar: "
                  f"report to the owner before smoke)", flush=True)
    if "opus_plc" in LEVERS:
        ok &= _opus_call_timing()
    return ok


def _opus_call_timing(n=12):
    """opus_plc smoke bar (plan 2.3): CPU-s per codec call, in-process libopus (default loss / burst) vs the ffmpeg path
    (self + children), same training clips and pre-drawn profiles, one process."""
    import resource
    from b4wave import opus_plc
    avail, why = _opus_lib_available()
    if not avail:
        skip("timing_opus_call", f"libopus missing ({why})")
        return True
    clips = [x for _, x in _training_clips(n, seed=1)]
    prms = [opus_plc.draw_codec_params(np.random.default_rng([i, 13])) for i in range(n)]
    r = np.random.default_rng(0)
    opus_plc.opus_loss_roundtrip(clips[0], r, 24, 8000, 20, "voip", (0.01, 0.20), (1.0, 4.0))     # warm-up
    t0 = time.process_time()
    for x, pr in zip(clips, prms):
        opus_plc.opus_loss_roundtrip(x, r, pr["kbps"], pr["cutoff"], pr["frame_ms"], pr["application"], (0.01, 0.20), (1.0, 4.0))
    t1 = time.process_time()
    c0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    with quiet():
        for x, pr in zip(clips, prms):
            opus_plc.ffmpeg_opus(x, pr)
    t2 = time.process_time()
    c1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    inproc = (t1 - t0) / n
    ff_self, ff_child = (t2 - t1) / n, ((c1.ru_utime + c1.ru_stime) - (c0.ru_utime + c0.ru_stime)) / n
    secs = np.mean([len(x) for x in clips]) / 16000
    return report("timing_opus_call_inprocess_below_ffmpeg", inproc < ff_self + ff_child,
                  f"in-process {inproc:.4f} >= ffmpeg {ff_self + ff_child:.4f} CPU-s per call",
                  [f"{n} training clips (mean {secs:.2f} s), CPU-s per codec call: in-process {inproc:.4f} vs ffmpeg "
                   f"{ff_self + ff_child:.4f} (self {ff_self:.4f} + children {ff_child:.4f}); plan measured 0.109 vs 0.258"])


# --------------------------------------------------------------------------- #
def check_gpu(ckpt, n_clips=4):
    """
    The model b4wave.train builds (b3wave ModelSLSB3; no B4 lever touches the model) loads the base
    checkpoint key-exact and gives the base ModelSLS logits on a real dev batch; every registered
    lever on (p = 1): forward + backward on lever-augmented training clips, finite loss and grads.
    """
    from torch import nn
    from model.sls_model import ModelSLS
    from b3wave.model import ModelSLSB3
    from b4wave.augment import LEVERS
    from b4wave.data import build_dataset_b4

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    if not cuda:
        print("    note: no CUDA device, running the gpu checks on the CPU")
    args = base_args()
    with quiet():
        dev, files, _, _ = build_dataset_b4(args.dev_protocol, args.dev_data_path, "dev", args)
    x = torch.stack([dev[i][0] for i in range(min(n_clips, len(files)))])
    state = torch.load(ckpt, map_location="cpu")

    ref = ModelSLS(args, device)
    ref.load_state_dict(state)
    ref.eval()
    with torch.no_grad():
        y_ref = ref(x.to(ref.input_device)).float().cpu()
    del ref
    if cuda:
        torch.cuda.empty_cache()

    model = ModelSLSB3(args, device)
    res = model.load_state_dict(state, strict=False)
    ok = report("ckpt_keys_exact", not res.missing_keys and not res.unexpected_keys
                and list(model.state_dict().keys()) == list(state.keys()),
                f"missing {res.missing_keys[:3]} unexpected {res.unexpected_keys[:3]}")
    model.eval()
    with torch.no_grad():
        y = model(x.to(model.input_device)).float().cpu()
    diff = float((y - y_ref).abs().max())
    ok &= report("logits_identical_levers_off", diff <= 1e-4, f"max |dlogit| {diff:.2e}",
                 [f"{len(x)} dev clips, max |dlogit| vs ModelSLS {diff:.2e}"])

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(args.ce_weights).to(model.output_device))
    if not LEVERS:
        print("    no B4 lever registered yet: nothing to run with a lever on")
    for lid, spec in LEVERS.items():
        a = base_args(train_subset_frac=1e-4, **{spec.prob_arg: 1.0})
        try:
            with quiet():
                ds, tf, _, _ = build_dataset_b4(a.train_protocol, a.train_data_path, "train", a)
        except ImportError as exc:
            skip(f"lever_forward_backward({lid})", f"package missing ({exc})")
            continue
        items = [ds[i] for i in range(min(n_clips, len(tf)))]
        xb = torch.stack([it["x"] for it in items]).to(model.input_device)
        yb = torch.tensor([it["y"] for it in items]).long().to(model.output_device)
        model.train()
        model.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        lv = float(loss.detach())
        finite = math.isfinite(lv) and all(bool(torch.isfinite(g).all()) for g in grads)
        counts = {k: int(v) for k, v in zip(ds.meta_keys, sum(it["meta"] for it in items).tolist()) if v}
        ok &= report(f"lever_forward_backward({lid})", finite and len(grads) > 0,
                     f"loss {lv}; {len(grads)} grads", [f"{lid}: loss {lv:.4f}, aug counts {counts}"])
        model.zero_grad(set_to_none=True)
        model.eval()
    del model
    return ok


def check_overfit(lever):
    if lever not in APPROVED:
        return report(f"overfit({lever})", False, f"unknown lever (approved: {list(APPROVED)})")
    if not PLAN_PARAMS_ADDED[lever]:
        skip(f"overfit({lever})", "not applicable: the lever adds no parameters (plan params_added=false)")
        return True
    return report(f"overfit({lever})", False, "params_added lever without an overfit check")


def run_b3(name, *a):
    """b3wave's model / optim checks (b4wave trains b3wave's ModelSLSB3 and optimizer unchanged)."""
    import b3wave.checks as b3c
    n0 = len(b3c.RESULTS)
    getattr(b3c, f"check_{name}")(*a)
    RESULTS.extend(b3c.RESULTS[n0:])


# --------------------------------------------------------------------------- #
# scripts/b4_dsp_slices.py must reproduce rdloop/B4/plan_agreed.md section 1.1 (B2f contrast rows, null-panel
# stats and bars, per-run DiDs, pinned tertile table, criterion-(ii) panel), section 2.0 (spoof-miss guards and
# their Poisson bounds) and critic O3.2 (debate/critique_v3.md: heldout-F1 panel, clearing counts). Values copied
# from those files: (n_T, n_C), recall T / C, G, D merge / final, within-run, panel mu / sd / min / max,
# half-B panel mu final / merge, panel clip mu, bar.
DSP_EXPECTED = {
    "NS": ((276, 106), (69.2, 75.5), 0.85, (30.1, 17.3), 0.22, (0.32, 0.27, -0.03, 0.68), (0.57, 0.23), -8.2, -0.22),
    "NS_afftdn_vs_anlmdn": ((139, 137), (62.6, 75.9), 2.01, (21.5, 18.5), 0.02, (-0.84, 0.43, -1.66, -0.23), (-0.61, None), 8.4, -1.70),
    "VAD": ((136, 246), (66.2, 73.6), 1.05, (13.6, 10.1), -0.05, (-0.57, 0.24, -0.81, -0.15), (-0.23, -0.34), 5.4, -1.05),
    "VAD_zero": ((75, 246), (68.0, 73.6), 0.87, (5.5, 4.2), -0.26, (-0.72, 0.42, -1.14, 0.28), (-0.34, None), 3.5, None),
    "VAD_cng": ((61, 246), (63.9, 73.6), 1.27, (8.1, 5.9), 0.21, (-0.38, 0.35, -0.75, 0.45), (-0.11, None), 1.9, None),
    "PLC_online": ((114, 254), (67.5, 75.6), 0.95, (11.1, 9.2), 0.08, (-0.57, 0.41, -1.11, 0.03), (-0.67, -0.59), 6.2, -1.39),
}
DSP_PER_RUN = {  # B2a first (excluded), then B2b, B2c, B2d, B2e, B3a, B3b, B3c, B3d, B3e
    "NS": (-1.42, [-0.02, -0.03, 0.39, 0.36, 0.49, 0.41, 0.62, 0.68, 0.01]),
    "VAD": (0.67, [-0.15, -0.17, -0.68, -0.74, -0.51, -0.70, -0.67, -0.65, -0.81]),
    "PLC_online": (-0.79, [-0.42, -1.08, -1.11, -0.73, -0.54, -0.09, -0.33, -0.88, 0.03]),
}
DSP_TERTILES = {"sources": [259, 250, 241], "n": [518, 500, 482], "fa_merge": [102, 130, 159], "fa_final": [97, 126, 154],
                "sat_merge": [63, 73, 109], "sat_final": [66, 76, 107], "halfB_sat_merge": [43, 47, 46], "halfB_sat_final": [44, 47, 42]}
DSP_CRIT2 = {"full_final": (-10.2, 5.8, 0), "halfB_final": (-13.4, 3.5, -5), "halfB_merge": (-6.8, 3.6, -2),
             "per_run_full_final": (14, [-12, -12, -21, -5, -12, -12, -8, -10, 0])}
DSP_GUARDS = {"merge": ({"NS_offline": 18, "VAD_offline": 15, "PLC": 16, "halfB_noisy": 24}, {"NS_offline": 29, "VAD_offline": 25, "PLC": 26, "halfB_noisy": 36}),
              "final": ({"NS_offline": 18, "VAD_offline": 13, "PLC": 15, "halfB_noisy": 23}, {"NS_offline": 29, "VAD_offline": 22, "PLC": 25, "halfB_noisy": 35})}
DSP_OUTCOME = {"final": (88.27, 88.50, 1.35, {"0.5": 5, "0.8": 4}), "merge": (87.92, 88.25, None, {"0.5": 6})}


def _close(x, e, nd):
    return e is None or (x is not None and abs(x - e) <= 0.5 * 10 ** -nd + 1e-9)


def _load_dsp_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("b4_dsp_slices", REPO_ROOT / "scripts" / "b4_dsp_slices.py")
    mod = importlib.util.module_from_spec(spec)
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True     # no scripts/__pycache__ from the checks
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = saved
    return mod


def check_dsp_slices():
    """scripts/b4_dsp_slices.py on B2f itself (CLI + JSON) reproduces the plan tables; verdict-rule unit cases."""
    import tempfile
    script = REPO_ROOT / "scripts" / "b4_dsp_slices.py"
    b2f_final = SLS_DIR / "exp/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623/local_eval/epoch_8"
    if not (b2f_final / "scores").is_dir() or not Path("/mnt/paml-research/RTCFake/data/wav/dev").is_dir():
        skip("dsp_slices", "B2f readout or dev source wavs missing")
        return True
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "b2f.json"
        r = subprocess.run([sys.executable, str(script), "--run", str(b2f_final), "--lever", "none", "--json_out", str(out)],
                           cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
        if r.returncode != 0 or not out.is_file():
            return report("dsp_slices_runs", False, f"rc {r.returncode}: {r.stderr[-400:]}")
        res = json.loads(out.read_text(encoding="utf-8"))
        out_dtx = Path(tmp) / "b2f_dtx.json"
        r_dtx = subprocess.run([sys.executable, "-B", str(script), "--run", str(b2f_final), "--lever", "dtx_gate",
                                "--json_out", str(out_dtx)], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
        res_dtx = json.loads(out_dtx.read_text(encoding="utf-8")) if r_dtx.returncode == 0 and out_dtx.is_file() else None
        out_opus = Path(tmp) / "b2f_opus.json"
        r_opus = subprocess.run([sys.executable, "-B", str(script), "--run", str(b2f_final), "--lever", "opus_plc",
                                 "--json_out", str(out_opus)], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900)
        res_opus = json.loads(out_opus.read_text(encoding="utf-8")) if r_opus.returncode == 0 and out_opus.is_file() else None
    ok = report("dsp_slices_runs", True)
    orr = (res_opus or {}).get("opus_plc_reading") or {}
    olines = [ln for ln in r_opus.stdout.splitlines() if ln.startswith("## opus_plc decision") or ln.startswith("  falsifier")
              or ln.startswith("  must not move") or ln.startswith("  decision")]
    g = orr.get("guards", {})
    ok &= report("dsp_slices_opus_plc_decision_lines", res_opus is not None and orr.get("falsifier1") is True
                 and orr.get("falsifier2") is False and _close(orr.get("plc_did_star"), 0.57, 2) and orr.get("guards_hold") is True
                 and set(g) == {"merge", "final"} and g["merge"]["plc_bound"] == 26 and g["final"]["plc_bound"] == 25
                 and _close(g["merge"]["app_recall"]["opus:qq"]["ref"], 61.62, 2) and _close(g["final"]["app_recall"]["opus:wechat"]["ref"], 71.15, 2)
                 and res_opus["lever_verdict"]["verdict"] == "refuted" and len(olines) >= 5,
                 f"rc {r_opus.returncode}; reading {orr}; verdict {(res_opus or {}).get('lever_verdict')}",
                 [f"B2f vs itself, --lever opus_plc: lever verdict {(res_opus or {}).get('lever_verdict', {}).get('verdict')}"] + olines)
    fr = (res_dtx or {}).get("vad_fill_reading") or {}
    line = next((ln for ln in r_dtx.stdout.splitlines() if ln.startswith("## Report (O2.3")), None)
    ok &= report("dsp_slices_dtx_gate_vad_fill_report", res_dtx is not None and fr.get("reading") == "no fill-specific gain"
                 and _close(fr.get("zero_bar"), -0.42, 2) and abs(fr.get("zero_did", 1.0)) < 1e-12 and line is not None
                 and res_dtx["lever_verdict"]["verdict"] == "not shown (criterion ii fails, VAD refuted)",
                 f"rc {r_dtx.returncode}; reading {fr}; verdict {(res_dtx or {}).get('lever_verdict')}",
                 [f"B2f vs itself, --lever dtx_gate: {line}", f"lever verdict: {(res_dtx or {}).get('lever_verdict', {}).get('verdict')}"])
    P = res["panel"]
    bad = []
    for c, (n, rec, g, d, within, pan, hb, clip, bar) in DSP_EXPECTED.items():
        x, p = res["contrasts"][c], P["contrasts"][c]
        got = [((x["ref_final"]["n_T"], x["ref_final"]["n_C"]) == n, "n"),
               (_close(x["ref_final"]["recall_T"], rec[0], 1) and _close(x["ref_final"]["recall_C"], rec[1], 1), "recall"),
               (_close(x["ref_final"]["G"], g, 2), "G"),
               (_close(x["ref_merge"]["D"], d[0], 1) and _close(x["ref_final"]["D"], d[1], 1), "D"),
               (_close(x["within_run"], within, 2), "within"),
               (all(_close(p[k], e, 2) for k, e in zip(("mu", "sd", "min", "max"), pan)), "panel"),
               (_close(p["halfB_mu_final"], hb[0], 2) and _close(p["halfB_mu_merge"], hb[1], 2), "halfB_mu"),
               (_close(p["clip_mu"], clip, 1), "clip_mu"), (_close(p["bar"], bar, 2), "bar")]
        bad += [f"{c}:{what}" for good, what in got if not good]
    ok &= report("dsp_contrast_rows_and_panel_reproduce_plan", not bad, f"mismatch {bad}",
                 [f"{c}: n {res['contrasts'][c]['ref_final']['n_T']}/{res['contrasts'][c]['ref_final']['n_C']} "
                  f"G {res['contrasts'][c]['ref_final']['G']:+.2f} panel mu/sd {P['contrasts'][c]['mu']:+.2f}/{P['contrasts'][c]['sd']:.2f} "
                  f"bar {P['contrasts'][c]['bar']:+.2f}" for c in DSP_EXPECTED])
    ex = res.get("excluded") or {}
    bad = [c for c, (b2a, runs) in DSP_PER_RUN.items()
           if not (all(_close(v, e, 2) for v, e in zip(P["contrasts"][c]["per_run"], runs))
                   and _close((ex.get("dids") or {}).get(c), b2a, 2))]
    ok &= report("dsp_per_run_dids_reproduce_plan", not bad and P["runs"] == ["B2b", "B2c", "B2d", "B2e", "B3a", "B3b", "B3c", "B3d", "B3e"],
                 f"mismatch {bad}; runs {P['runs']}")
    t = res["tertiles"]
    tc = t["counts"]

    def col(k, part, field):
        return [tc[k][part][x][field] for x in ("low", "mid", "high")]
    got = {"sources": [t["sources"][x] for x in ("low", "mid", "high")], "n": col("ref_final", "full", "n"),
           "fa_merge": col("ref_merge", "full", "fa"), "fa_final": col("ref_final", "full", "fa"),
           "sat_merge": col("ref_merge", "full", "sat"), "sat_final": col("ref_final", "full", "sat"),
           "halfB_sat_merge": col("ref_merge", "halfB", "sat"), "halfB_sat_final": col("ref_final", "halfB", "sat")}
    ok &= report("dsp_pinned_tertile_table_reproduces_plan", got == DSP_TERTILES,
                 f"{ {k: (got[k], v) for k, v in DSP_TERTILES.items() if got[k] != v} }", [f"{got}"])
    c2 = P["crit2"]
    bad = [k for k, v in DSP_CRIT2.items() if k != "per_run_full_final"
           and not (_close(c2[k]["mu"], v[0], 1) and _close(c2[k]["sd"], v[1], 1) and c2[k]["max"] == v[2])]
    per = DSP_CRIT2["per_run_full_final"]
    if c2["full_final"]["per_run"] != per[1] or ex.get("crit2_full_final") != per[0]:
        bad.append("per_run_full_final")
    ok &= report("dsp_criterion_ii_panel_reproduces_plan", not bad, f"mismatch {bad}: { {k: c2[k] for k in c2} }")
    g = res["spoof_guards"]
    ok &= report("dsp_spoof_guards_and_bounds_reproduce_plan",
                 all(g[k]["ref"] == base and g[k]["bound"] == bound for k, (base, bound) in DSP_GUARDS.items()),
                 f"{g}")
    o = res["outcome"]["panel"]
    bad = [k for k, (ref, mean, sd, clear) in DSP_OUTCOME.items()
           if not (_close(o[k]["ref"], ref, 2) and _close(o[k]["mean"], mean, 2) and _close(o[k]["sd"], sd, 2)
                   and all(o[k]["clearing"][t] == n for t, n in clear.items()))]
    ok &= report("dsp_heldout_f1_panel_reproduces_O3_2", not bad, f"mismatch {bad}: {o}")
    ok &= report("dsp_self_comparison_is_null", all(abs(x["arm"]["did"]) < 1e-12 for x in res["contrasts"].values())
                 and not res["anomaly"]["anomalous"])

    # verdict-rule unit cases (plan section 1.1 + O3.1 / O3.3) on a fabricated arm
    m = _load_dsp_script()
    panel = {"contrasts": {"NS": {"mu": 0.32, "sd": 0.27, "bar": -0.22, "clip_mu": -8.2}}}
    arm = {"did": -0.40, "ci": [-0.6, -0.2], "clip_did": 2.0, "D_ref_final": 17.3}
    v_shown = m.contrast_verdict("NS", arm, panel, [-0.1, -0.2], [0.1, 0.3], summary=True)
    v_nopl = m.contrast_verdict("NS", arm, panel, [-0.1, -0.2], [], summary=True)
    v_beaten = m.contrast_verdict("NS", arm, panel, [-0.1, -0.2], [-0.5], summary=False)
    v_ref = m.contrast_verdict("NS", dict(arm, did=0.40), panel, [-0.1, -0.2], [0.5], summary=True)
    v_dir = m.contrast_verdict("NS", dict(arm, did=0.0), panel, [-0.1, -0.2], [0.5], summary=True)
    cases = [(v_shown["verdict"], "shown"), (v_nopl["verdict"], "directional"), (v_beaten["verdict"], "inconclusive"),
             (v_ref["verdict"], "refuted"), (v_dir["verdict"], "directional")]
    ok &= report("dsp_verdict_rules", all(a == b for a, b in cases) and v_beaten["provisional"] and not v_shown["provisional"]
                 and v_nopl["n_placebo"] == 0, f"{cases}")
    fake = {"contrasts": {"NS": dict(v_shown), "VAD": {"verdict": "refuted"}, "PLC_online": {"verdict": "refuted"}},
            "anomaly": {"anomalous": True}, "tertiles": {"crit2_arm": {}}}
    lv_anom = m.lever_verdict("apm_ns", fake, {"crit2": {"full_final": {"mu": -10.2}}})
    fake["anomaly"]["anomalous"] = False
    lv_ok = m.lever_verdict("apm_ns", fake, {"crit2": {"full_final": {"mu": -10.2}}})
    ok &= report("dsp_run_anomaly_confounds_shown", lv_anom["verdict"] == "confounded (run anomaly)"
                 and not lv_anom["lower_p_retry_allowed"] and lv_ok["verdict"] == "mechanism shown" and lv_ok["lower_p_retry_allowed"],
                 f"{lv_anom} / {lv_ok}")

    # dtx_gate report (plan 1.1 / 2.2, O2.3): the VAD contrast split by fill
    pz = {"contrasts": {"VAD_zero": {"sd": 0.42}}}

    def fill_case(z, c):
        return m.vad_fill_reading({"VAD_zero": {"did_star": z, "arm": {"did": z - 0.72}},
                                   "VAD_cng": {"did_star": c, "arm": {"did": c - 0.38}}}, pz)["reading"]
    cases = [(fill_case(-0.50, 0.10), "near-in-distribution (gain confined to vad:zero)"),
             (fill_case(-0.50, -0.01), "transfer claimable (CNG-fill DiD* < 0)"),
             (fill_case(-0.10, 0.10), "no fill-specific gain"), (fill_case(-0.42, 0.0), "near-in-distribution (gain confined to vad:zero)")]
    ok &= report("dsp_vad_fill_reading", all(a == b for a, b in cases), f"{cases}")

    # opus_plc falsifier (3) (plan 2.3): the zero-loss control's WF1 gain vs the arm's
    cases = [(m.attribution(0.4, 0.2), "credited to the decoder path"), (m.attribution(0.4, 0.19), "credited to PLC"),
             (m.attribution(0.0, 0.3), "n/a (arm WF1 gain <= 0)"), (m.attribution(None, 0.1), "n/a"),
             (m.attribution(0.4, -0.1), "credited to PLC")]
    ok &= report("dsp_opus_attribution_rule", all(a == b for a, b in cases), f"{cases}")
    return ok


# --------------------------------------------------------------------------- #
CPU_GROUPS = ["module", "flags", "augmenter", "reserve_names", "imports", "trainer", "data", "apm", "apm_balance",
              "dtx", "dtx_balance", "codec_draw_order", "opus_plc", "timing", "dsp_slices"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["cpu", "all", "gpu", "overfit", "model", "optim"] + CPU_GROUPS)
    parser.add_argument("--ckpt", default=BASE_CKPT)
    parser.add_argument("--lever", default=None, help="[overfit] lever id")
    args = parser.parse_args()
    os.chdir(SLS_DIR)
    sys.path.insert(0, str(SLS_DIR))
    t0 = time.time()
    todo = CPU_GROUPS if args.what in ("cpu", "all") else [args.what]
    if args.what == "overfit" and not args.lever:
        parser.error("overfit needs --lever <id>")
    for what in todo:
        print(f"== {what}", flush=True)
        try:
            if what == "gpu":
                check_gpu(args.ckpt)
            elif what == "overfit":
                check_overfit(args.lever)
            elif what == "model":
                run_b3("model", args.ckpt)
            elif what == "optim":
                run_b3("optim")
            else:
                globals()[f"check_{what}"]()
        except Exception as exc:                    # one broken group must not hide the others
            import traceback
            traceback.print_exc(file=sys.stdout)
            report(f"{what}_group", False, f"{type(exc).__name__}: {exc}")
    failed = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed in {time.time() - t0:.0f}s"
          + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
