"""
Scratch (implement_apm_ns, copied from toy_trainer_run.py with an `apm` mode = --aug_apm_p 0.3): drive b4wave.train.main() end to end on the CPU with a toy model (no SSL backbone)
to exercise the trainer wiring: header lines, config.yaml keys, per-class telemetry, metrics.jsonl rows,
checkpoint / last_state, 'Experiment saved to:'. Not a training run: 1 epoch, ~40 clips, toy linear head.

    python rdloop/B4/scratch/toy_trainer_apm.py <out_dir> off|apm
"""
import os
import sys
from pathlib import Path

SLS = Path(__file__).resolve().parents[3] / "SLS_setup"
os.chdir(SLS)
sys.path.insert(0, str(SLS))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import b4wave.train as T  # noqa: E402
from b4wave import augment as A  # noqa: E402
from b4wave.checks import BASE_ARGV  # noqa: E402
from utils.rtc_augment import finalize  # noqa: E402

out_dir, mode = Path(sys.argv[1]), sys.argv[2]


class Toy(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.ssl_model = nn.Linear(1, 1)
        self.ssl_model.devices, self.ssl_model.name, self.ssl_model.n_layers = None, "toy", 0
        self.fc = nn.Linear(128, 2)
        self.input_device = self.output_device = device

    def forward(self, x, return_emb=False):
        return self.fc(x.float().mean(-1))


T.ModelSLSB3 = Toy
if mode == "stub":
    orig = T.build_parser

    def build_parser():
        p = orig()
        p.add_argument("--aug_stub_p", type=float, default=0.0)
        return p

    T.build_parser = build_parser

    def _lever__stub(self, out, sr):
        if self._get_rng4().random() < self.cfg.stub_p:
            self._count("stub_hit")
            return finalize(out * np.float32(0.5), out)
        return out

    A.RTCAugmenterB4._lever__stub = _lever__stub
    A.register(A.LeverSpec("_stub", "post_dsp", "aug_stub_p", "stub_p", meta_keys=("stub_hit",),
                           filters=("wrtc_stub",), describe=lambda c: f"stub p={c.stub_p:g}",
                           header=lambda a: f"B4 stub: on p={a.aug_stub_p:g}"))

dev = out_dir / "dev_small.txt"
dev.parent.mkdir(parents=True, exist_ok=True)
dev.write_text("".join(open("../data/dev_noisy_sim/dev_label_excl_halfB.txt").readlines()[:24]))
argv = list(BASE_ARGV)
sets = {"--train_subset_frac": "0.0005", "--num_epochs": "1", "--num_workers": "0", "--batch_size": "8",
        "--dev_protocol": str(dev), "--out_path": str(out_dir / "runs"), "--track": f"toyB4_{mode}",
        "--local_eval_clean_n": "50", "--keep_epochs": "1"}
for k, v in sets.items():
    i = argv.index(k) if k in argv else None
    if i is None:
        argv += [k, v]
    else:
        argv[i + 1] = v
if mode == "stub":
    argv += ["--aug_stub_p", "0.5"]
if mode == "apm":
    argv += ["--aug_apm_p", "0.3"]
sys.argv = ["b4wave/train.py", *argv]
T.main()
