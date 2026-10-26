"""
theta_init providers for ties_merging and helpers around the run directories.

    pretrained   the hub weights the run started from (Qwen3-ASR audio tower): read straight from
                 the cached safetensors (CPU, seconds, no model build) and mapped to ModelSLS keys
                 `ssl_model.model.*`; covers the backbone only, so the head is plain-averaged.
    checkpoint   any plain ModelSLS state dict; covers every float parameter (full-model TIES).

A base's `state` decides which keys get TIES (merge.partition_keys): every float, non-buffer
tensor present in it.
"""

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

SLS_DIR = Path(__file__).resolve().parents[1]
if str(SLS_DIR) not in sys.path:
    sys.path.insert(0, str(SLS_DIR))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from model.sls_model import ssl_kind, ssl_path  # noqa: E402

from .merge import load_state  # noqa: E402

HUB_PREFIX = "model.audio_tower."          # Qwen3-ASR-hf safetensors
SLS_PREFIX = "ssl_model.model."            # ModelSLS state dict
LAYER_RE = re.compile(r"^ssl_model\.model\.layers\.(\d+)\.")


@dataclass
class BaseWeights:
    kind: str                 # "pretrained" | "checkpoint"
    state: dict               # key -> tensor (ModelSLS key names)
    label: str
    meta: dict = field(default_factory=dict)

    @property
    def keys(self):
        return self.state.keys()


# --------------------------------------------------------------------------- #
# pretrained backbone
# --------------------------------------------------------------------------- #
def hub_snapshot_dir(name):
    """Local directory of the hub repo (a path, or the cached snapshot; never downloads)."""
    if os.path.isdir(name):
        return Path(name)
    # try_to_load_from_cache: cache lookup only, no network and no snapshot-completeness check
    # (snapshot_download(local_files_only=True) rejects our cache: README etc. were never fetched)
    from huggingface_hub import try_to_load_from_cache
    hit = try_to_load_from_cache(name, "config.json")
    if not isinstance(hit, str):
        raise FileNotFoundError(f"{name}: config.json is not in the local hub cache")
    return Path(hit).parent


def _layer_index(key):
    m = LAYER_RE.match(key)
    return int(m[1]) if m else None


def pretrained_backbone_from_safetensors(ssl_name, n_layers):
    """Qwen3-ASR audio tower from the cached safetensors, float32, layers >= n_layers dropped."""
    from safetensors import safe_open
    name = ssl_path(ssl_name)
    if ssl_kind(name) != "qwen3_asr":
        raise ValueError(f"{name}: only the Qwen3-ASR tower is mapped from safetensors; "
                         f"use pretrained_backbone_from_model for {ssl_kind(name)}")
    snap = hub_snapshot_dir(name)
    files = sorted(snap.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {snap}")
    state = {}
    for f in files:
        with safe_open(str(f), "pt") as sf:
            for key in sf.keys():
                if not key.startswith(HUB_PREFIX):
                    continue
                new = SLS_PREFIX + key[len(HUB_PREFIX):]
                li = _layer_index(new)
                if li is not None and n_layers is not None and li >= n_layers:
                    continue
                state[new] = sf.get_tensor(key).to(torch.float32).contiguous()
    if not state:
        raise ValueError(f"no {HUB_PREFIX}* tensors in {snap}")
    n_found = 1 + max((_layer_index(k) for k in state if _layer_index(k) is not None), default=-1)
    if n_layers is not None and n_found < n_layers:
        raise ValueError(f"{name} has {n_found} layers, run uses n_layers={n_layers}")
    return BaseWeights("pretrained", state, f"pretrained:{name}",
                       {"ssl_name": name, "n_layers": n_found, "snapshot": snap.name, "source": "safetensors"})


def pretrained_backbone_from_model(ssl_name, n_layers):
    """Fallback for any SSL front-end: build ModelSLS on CPU and take its ssl_model.* tensors."""
    from model.sls_model import ModelSLS
    name = ssl_path(ssl_name)
    args = SimpleNamespace(ssl_name=name, n_layers=n_layers)
    model = ModelSLS(args, torch.device("cpu"))
    state = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith("ssl_model.")}
    del model
    return BaseWeights("pretrained", state, f"pretrained:{name}",
                       {"ssl_name": name, "n_layers": n_layers, "snapshot": hub_snapshot_dir(name).name,
                        "source": "model"})


def pretrained_backbone(ssl_name, n_layers, via_model=False):
    if not via_model and ssl_kind(ssl_path(ssl_name)) == "qwen3_asr":
        return pretrained_backbone_from_safetensors(ssl_name, n_layers)
    return pretrained_backbone_from_model(ssl_name, n_layers)


# --------------------------------------------------------------------------- #
# checkpoint base
# --------------------------------------------------------------------------- #
def checkpoint_base(path):
    path = Path(path)
    return BaseWeights("checkpoint", load_state(str(path)), f"checkpoint:{path}",
                       {"path": str(path.resolve())})


def load_base(spec, ssl_name=None, n_layers=None, via_model=False):
    """`spec` is "pretrained" (needs ssl_name / n_layers) or a checkpoint path."""
    if spec == "pretrained":
        if ssl_name is None or n_layers is None:
            raise ValueError("--base pretrained needs ssl_name and n_layers (run config.yaml or flags)")
        return pretrained_backbone(ssl_name, n_layers, via_model)
    if Path(spec).is_file():
        return checkpoint_base(spec)
    raise FileNotFoundError(f"--base {spec}: not 'pretrained' and not a checkpoint file")


# --------------------------------------------------------------------------- #
# run directories
# --------------------------------------------------------------------------- #
def run_dir_of(ckpt):
    """exp/<run> of an epoch checkpoint (parent of its ckpt/ folder), else None."""
    p = Path(ckpt).resolve()
    for parent in (p.parent, p.parent.parent):
        if (parent / "config.yaml").is_file():
            return parent
    return None


def read_run_args(run_dir):
    cfg_path = Path(run_dir) / "config.yaml"
    if not cfg_path.is_file():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("args", {}) or {}


def runs_of(paths):
    """{run_dir: args} for the distinct runs behind a merge list (checkpoints without a run: skipped)."""
    runs = {}
    for p in paths:
        r = run_dir_of(p)
        if r is not None and r not in runs:
            runs[r] = read_run_args(r)
    return runs


def model_args_from_runs(runs, ssl_name=None, n_layers=None):
    """
    (ssl_name, n_layers, loudness_norm, loudness_norm_lufs, warnings): CLI values win, else the
    runs' config.yaml; warnings list the fields the runs disagree on (seed included: heads of
    runs with different seeds have no shared init lineage, averaging them is questionable).
    """
    warnings = []
    for key in ("ssl_name", "n_layers", "loudness_norm", "loudness_norm_lufs", "seed"):
        vals = {str(a.get(key)) for a in runs.values() if key in a}
        if len(vals) > 1:
            warnings.append(f"runs disagree on {key}: {sorted(vals)}")
    first = next(iter(runs.values()), {})
    ssl = ssl_name or first.get("ssl_name")
    nl = n_layers if n_layers is not None else first.get("n_layers")
    return (ssl_path(ssl) if ssl else None, nl, bool(first.get("loudness_norm", False)),
            float(first.get("loudness_norm_lufs", -23.0)), warnings)
