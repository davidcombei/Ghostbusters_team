"""
Scoring a model on the local proxy sets (dev-online-clean, sim-matched, sim-heldout, sim-echo).

Shared by scripts/local_eval.py (full report) and main_train.py --local_eval_sets (per-epoch
checkpoint selection). Read-only: the sets are scored in eval mode, without augmentation and
without gradients, so they never influence the weights.

The sets are listed in a YAML file (configs/local_eval_sets.yaml):

    sim_matched_v1:
      role: matched                 # clean | matched | heldout | echo | probe
      base_dir: data/dev_noisy_sim/matched_v1
      protocol: data/dev_noisy_sim/matched_v1/protocol.txt
      meta: data/dev_noisy_sim/matched_v1/meta.csv      # optional; needs a `path` column

Relative paths are resolved against the repository root.
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from . import metrics
from .data_utils import SpoofAudioDataset, read_protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
NOISY_ROLES = ("matched", "heldout")        # the two sets averaged into WF1_local


def _resolve(path):
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def read_meta(meta_path):
    """meta.csv rows keyed by their `path` column (the protocol relpath)."""
    if meta_path is None or not Path(meta_path).is_file():
        return {}
    with open(meta_path, newline="", encoding="utf-8") as f:
        return {row["path"]: row for row in csv.DictReader(f)}


def load_sets(config_path, names=None, ssl_name=None, clean_n=None, seed=0, loudness=None):
    """
    {name: {"role", "dataset", "file_list", "y_spoof", "meta_rows"}} for the requested sets.
    `clean_n` subsamples the role=clean set to a fixed, seeded subset (for per-epoch speed).
    `loudness` (utils/loudness.LoudnessNormalizer or None) is applied at read time, as in training.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    names = list(names) if names else list(cfg)
    missing = [n for n in names if n not in cfg]
    if missing:
        raise KeyError(f"sets {missing} not in {config_path} (have: {list(cfg)})")

    sets = {}
    for name in names:
        entry = cfg[name]
        file_list, labels = read_protocol(_resolve(entry["protocol"]), require_label=True)
        role = entry.get("role", "noisy")
        if role == "clean" and clean_n and clean_n < len(file_list):
            file_list = sorted(random.Random(seed).sample(file_list, clean_n))
        meta = read_meta(_resolve(entry.get("meta")))
        dataset = SpoofAudioDataset(file_list=file_list, base_dir=_resolve(entry["base_dir"]),
                                    labels=labels, ssl_name=ssl_name, train=False, loudness=loudness)
        sets[name] = {
            "role": role,
            "dataset": dataset,
            "file_list": file_list,
            # LABEL_TO_ID: spoof = 0, bonafide = 1  ->  y_spoof = 1 - id
            "y_spoof": np.array([1 - labels[u] for u in file_list], dtype=int),
            "meta_rows": [meta.get(u, {}) for u in file_list],
        }
    return sets


def score_dataset(model, dataset, device, batch_size=32, num_workers=4, desc=None):
    """(utt_ids, p_spoof, d) with d = z_spoof - z_bona, in dataset order."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                        pin_memory=torch.cuda.is_available())
    was_training = model.training
    model.eval()
    ids, p_all, d_all = [], [], []
    try:
        with torch.no_grad():
            for batch in loader:
                batch_x, utt_ids = batch[0], batch[-1]
                logits = model(batch_x.to(device)).float()
                p_all.append(torch.softmax(logits, dim=1)[:, 0].cpu().numpy())
                d_all.append((logits[:, 0] - logits[:, 1]).cpu().numpy())
                ids.extend(utt_ids)
    finally:
        model.train(was_training)
    return ids, np.concatenate(p_all), np.concatenate(d_all)


def evaluate_sets(sets, scores, thr=0.5, half=None):
    """
    Metrics from precomputed scores {name: p_spoof}. `half` ("A"/"B") restricts the sets that
    carry a `half` meta column to that half. Returns {"sets": {...}, "wf1": {...}}.
    """
    out, clean, noisy, clean_o, noisy_o = {}, None, [], None, []
    for name, s in sets.items():
        y, p = s["y_spoof"], np.asarray(scores[name])
        if half is not None:
            halves = np.array([r.get("half", "") for r in s["meta_rows"]])
            if (halves != "").any():
                keep = halves == half
                y, p = y[keep], p[keep]
        summary = metrics.summarize(y, p, thr)
        summary["role"] = s["role"]
        out[name] = summary
        if s["role"] == "clean":
            clean, clean_o = (y, p), summary["f1_oracle"]
        elif s["role"] in NOISY_ROLES:
            noisy.append((y, p))
            noisy_o.append(summary["f1_oracle"])

    wf1 = {}
    if clean is not None and noisy:
        w, f_clean, f_noisy = metrics.wf1_local(clean, noisy, thr)
        w_o = metrics.CLEAN_WEIGHT * clean_o + (1 - metrics.CLEAN_WEIGHT) * float(np.mean(noisy_o))
        wf1 = {
            "wf1": w, "f1_clean": f_clean, "f1_noisy": f_noisy,
            "wf1_oracle": w_o,
            # F1@oracle - F1@0.5: what calibration alone could recover
            "calibration_gap_noisy": float(np.mean(noisy_o)) - f_noisy,
            # F1@oracle(clean) - F1@oracle(noisy): what better separation has to recover
            "separability_gap": clean_o - float(np.mean(noisy_o)),
        }
    noisy_eers = [out[n]["eer"] for n, s in sets.items() if s["role"] in NOISY_ROLES]
    if noisy_eers:
        wf1["eer_noisy"] = float(np.mean(noisy_eers))
    return {"sets": out, "wf1": wf1}


def quick_eval(model, sets, device, batch_size=32, num_workers=4, thr=0.5):
    """
    Score every set and return evaluate_sets(...) (used per epoch by main_train).
    RNG-neutral: the global torch / numpy / python RNG states are restored afterwards, so
    turning per-epoch evaluation on does not change the training trajectory.
    """
    py_state, np_state = random.getstate(), np.random.get_state()
    cuda = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda):
        scores = {name: score_dataset(model, s["dataset"], device, batch_size, num_workers)[1]
                  for name, s in sets.items()}
    random.setstate(py_state)
    np.random.set_state(np_state)
    return evaluate_sets(sets, scores, thr)
