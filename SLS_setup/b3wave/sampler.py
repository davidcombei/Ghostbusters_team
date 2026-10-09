"""
Class- and language-balanced sampling for the B3 wave (--sampler balanced).

Train has 14,353 bonafide of 75,785 clips (18.9 %) and English is the scarcer language
(2,922 en bona). With --sampler balanced every epoch draws `len(dataset)` indices with
replacement, so that a share `bona_frac` of the draws are bonafide and, with
`lang_balance`, each class is drawn half from en and half from zh. The epoch length (steps)
is therefore unchanged and the warmup-cosine step count stays identical to B2e.

Weights are the per-item probabilities; `expected_shares` is what the sampler aims at and
`realised_shares` what one epoch actually drew (logged per epoch by b3wave.train).
"""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def sampling_weights(labels, langs, bona_frac=0.5, lang_balance=False):
    """
    labels: 1 = bonafide, 0 = spoof (LABEL_TO_ID); langs: "en" / "zh" / "?" per item.
    Returns a float64 array of per-item probabilities summing to 1.
    """
    labels = np.asarray(labels, dtype=int)
    langs = np.asarray(langs)
    if not 0.0 < bona_frac < 1.0:
        raise ValueError(f"bona_frac must be in (0, 1), got {bona_frac}")
    weights = np.zeros(len(labels), dtype=np.float64)
    for cls, mass in ((1, bona_frac), (0, 1.0 - bona_frac)):
        members = np.flatnonzero(labels == cls)
        if members.size == 0:
            continue
        if lang_balance:
            cells = [members[langs[members] == lg] for lg in sorted(set(langs[members]))]
            cells = [c for c in cells if c.size]
            for cell in cells:
                weights[cell] = mass / len(cells) / cell.size
        else:
            weights[members] = mass / members.size
    total = weights.sum()
    if total <= 0:
        raise ValueError("no items to sample")
    return weights / total


class RecordingWeightedSampler(WeightedRandomSampler):
    """WeightedRandomSampler that keeps the indices of the last epoch it drew (`last_indices`),
    so the trainer can log the realised bona / language shares of that epoch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_indices = []

    def __iter__(self):
        self.last_indices = list(super().__iter__())
        return iter(self.last_indices)


def make_balanced_sampler(labels, langs, bona_frac=0.5, lang_balance=False, seed=0, num_samples=None):
    weights = sampling_weights(labels, langs, bona_frac, lang_balance)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return RecordingWeightedSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=num_samples or len(weights),
                                    replacement=True, generator=generator)


def expected_shares(weights, labels, langs):
    labels = np.asarray(labels, dtype=int)
    langs = np.asarray(langs)
    w = np.asarray(weights, dtype=np.float64)
    return {"bona": float(w[labels == 1].sum()), "en": float(w[langs == "en"].sum())}


def realised_shares(indices, labels, langs):
    idx = np.asarray(list(indices), dtype=int)
    labels = np.asarray(labels, dtype=int)
    langs = np.asarray(langs)
    if idx.size == 0:
        return {"bona": float("nan"), "en": float("nan")}
    return {"bona": float((labels[idx] == 1).mean()), "en": float((langs[idx] == "en").mean())}
