"""
TIES-Merging (Yadav et al. 2023, arXiv 2306.01708) of plain ModelSLS state dicts.

    tau_t   = theta_t - theta_init                              task vectors
    trim    keep the top-k% of |tau_t| (global over the flattened vector), zero the rest
    elect   gamma = sign(sum_t tau_hat_t)
    merge   tau_m^p = sum_{t in A^p} tau_hat_t^p / |A^p|,  A^p = {t : sign(tau_hat_t^p) == gamma^p}
    theta_m = theta_init + lambda * tau_m

This follows the official code (prateeky2806/ties-merging): the trim threshold is the
(n - k_keep + 1)-th smallest |tau| and entries >= it are kept, exact zeros never count as kept,
the election is the sign of the plain sum, and the disjoint mean divides by the number of
selected non-zero entries, clamped to >= 1 (positions nobody keeps stay at theta_init).

Which keys get TIES is decided by the base (ties_merging.base): the pretrained backbone covers
the `ssl_model.*` tensors only, a checkpoint base covers every float parameter. The remaining
float tensors (head in pretrained mode) and the BatchNorm running statistics are plain-averaged
with the exact arithmetic of main_eval.merge_checkpoints (float64 sum in list order, / T, cast),
and non-float tensors (`first_bn.num_batches_tracked`) go through that same mean-then-cast.

Memory: the state dicts stay memory-mapped; per model only one int8 signed-mask vector
(0.3 GiB for 314M backbone floats) is kept, plus float32 / float64 accumulators of that length,
so T = 7 checkpoints need ~12 GiB whatever the base. Task vectors are re-read per key from the
mapped files (page cache) in float64 for the final accumulation, so T = 1, k = 100, lambda = 1
reproduces the checkpoint bit for bit.
"""

import math
from dataclasses import dataclass, field

import torch

BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")
MERGE_MODES = ("dis-mean", "mean_over_T")
TRIM_SCOPES = ("global", "tensor")


# --------------------------------------------------------------------------- #
# loading and key partition
# --------------------------------------------------------------------------- #
def load_state(path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def load_states(paths):
    """Memory-mapped state dicts; every one must have the keys, shapes and dtypes of the first."""
    states = [load_state(p) for p in paths]
    first = states[0]
    for p, s in zip(paths[1:], states[1:]):
        if list(s.keys()) != list(first.keys()):
            raise ValueError(f"{p} has different keys than {paths[0]}")
        for k, v in first.items():
            if s[k].shape != v.shape or s[k].dtype != v.dtype:
                raise ValueError(f"{p}: {k} is {tuple(s[k].shape)} {s[k].dtype}, "
                                 f"{paths[0]} has {tuple(v.shape)} {v.dtype}")
    return states


def is_bn_buffer(key):
    return key.rsplit(".", 1)[-1] in BN_BUFFER_SUFFIXES


def partition_keys(state, base_keys):
    """
    ties_keys: float tensors covered by the base and not BN buffers (in state order);
    mean_keys: the other float tensors; copy_keys: non-float tensors (mean-then-cast).
    """
    ties, mean, copy = [], [], []
    for k, v in state.items():
        if not v.is_floating_point():
            copy.append(k)
        elif is_bn_buffer(k) or k not in base_keys:
            mean.append(k)
        else:
            ties.append(k)
    return ties, mean, copy


@dataclass
class Layout:
    """Flat index space over the TIES keys: key -> (offset, numel), shapes, total length."""
    keys: list
    shapes: dict
    offsets: dict
    n: int

    @classmethod
    def build(cls, state, keys):
        shapes, offsets, o = {}, {}, 0
        for k in keys:
            shapes[k] = tuple(state[k].shape)
            offsets[k] = (o, state[k].numel())
            o += state[k].numel()
        return cls(list(keys), shapes, offsets, o)

    def slices(self):
        for k in self.keys:
            o, m = self.offsets[k]
            yield k, o, m


class Ingredients:
    """The T checkpoints of a merge list plus their base, with the key partition and flat layout."""

    def __init__(self, paths, base):
        self.paths = [str(p) for p in paths]
        if not self.paths:
            raise ValueError("empty merge list")
        self.base = base
        self.states = load_states(self.paths)
        self.T = len(self.states)
        self.ties_keys, self.mean_keys, self.copy_keys = partition_keys(self.states[0], base.state.keys())
        if not self.ties_keys:
            raise ValueError("no float tensor is covered by the base weights; nothing to merge with TIES")
        for k in self.ties_keys:
            if tuple(base.state[k].shape) != tuple(self.states[0][k].shape):
                raise ValueError(f"base {base.label}: {k} is {tuple(base.state[k].shape)}, "
                                 f"checkpoint has {tuple(self.states[0][k].shape)}")
        self.layout = Layout.build(self.states[0], self.ties_keys)
        self.n = self.layout.n

    def tau(self, t, key, dtype=torch.float32):
        """Flat task vector of model t for one key (theta_t - theta_init), in `dtype`."""
        return (self.states[t][key].to(dtype) - self.base.state[key].to(dtype)).reshape(-1)

    def flat_tau(self, t, dtype=torch.float32):
        out = torch.empty(self.n, dtype=dtype)
        for k, o, m in self.layout.slices():
            out[o:o + m] = self.tau(t, k, dtype)
        return out

    def flat_base(self, dtype=torch.float64):
        out = torch.empty(self.n, dtype=dtype)
        for k, o, m in self.layout.slices():
            out[o:o + m] = self.base.state[k].to(dtype).reshape(-1)
        return out


# --------------------------------------------------------------------------- #
# trim
# --------------------------------------------------------------------------- #
def keep_count(n, k_pct):
    return min(n, max(0, int(round(n * k_pct / 100.0))))


def global_threshold(magnitudes, k_pct):
    """
    The smallest |tau| kept when the top-k% entries survive: the (n - k_keep + 1)-th smallest
    value (official `topk_values_mask` convention, entries >= threshold are kept). k = 100 keeps
    every entry (threshold 0), k = 0 keeps none (threshold +inf).
    """
    n = magnitudes.numel()
    k_keep = keep_count(n, k_pct)
    if k_keep >= n:
        return 0.0
    if k_keep <= 0:
        return math.inf
    return magnitudes.kthvalue(n - k_keep + 1).values.item()


def trim_mask(flat, k_pct, scope="global", layout=None):
    """
    Bool mask of the entries of one flat task vector that survive trimming. Exact zeros never
    survive (they carry no sign). scope="tensor" applies the same rule per tensor of `layout`.
    Returns (mask, thresholds) with thresholds a float (global) or {key: float} (tensor).
    """
    if scope not in TRIM_SCOPES:
        raise ValueError(f"trim scope {scope!r} not in {TRIM_SCOPES}")
    mag = flat.abs()
    if scope == "global":
        thr = global_threshold(mag, k_pct)
        return (mag >= thr) & (flat != 0), thr
    if layout is None:
        raise ValueError("scope='tensor' needs the layout")
    mask, thrs = torch.zeros_like(flat, dtype=torch.bool), {}
    for k, o, m in layout.slices():
        thrs[k] = global_threshold(mag[o:o + m], k_pct)
        mask[o:o + m] = (mag[o:o + m] >= thrs[k]) & (flat[o:o + m] != 0)
    return mask, thrs


# --------------------------------------------------------------------------- #
# elect + disjoint merge
# --------------------------------------------------------------------------- #
@dataclass
class Election:
    """Signed trim masks of every model (int8 in {-1, 0, 1}), the elected sign and counters."""
    k_pct: float
    trim_scope: str
    trim: bool
    elect: bool
    signed: list                         # per model: int8 flat, sign(tau) where kept else 0
    gamma: torch.Tensor                  # int8 flat; zeros where nothing is kept / exact cancellation
    pos: torch.Tensor                    # int16 flat: number of models keeping a positive entry
    neg: torch.Tensor
    thresholds: list                     # per model (float or {key: float})
    kept: list = field(default_factory=list)      # per model: kept entries

    @property
    def T(self):
        return len(self.signed)

    def kept_count(self):
        return (self.pos + self.neg)

    def summary(self):
        n = self.gamma.numel()
        cnt = self.kept_count()
        shared = cnt >= 2
        conflict = shared & (self.pos > 0) & (self.neg > 0)
        n_shared = int(shared.sum())
        return {
            "n_params": n,
            "kept_frac": [k / n for k in self.kept],
            "kept_any_frac": int((cnt > 0).sum()) / n,
            "shared_frac": n_shared / n,
            "conflict_frac": (int(conflict.sum()) / n_shared) if n_shared else 0.0,
            "n_conflict": int(conflict.sum()),
            "n_gamma_zero_cancel": int(((self.gamma == 0) & (cnt > 0)).sum()),
            "thresholds": [t if isinstance(t, float) else None for t in self.thresholds],
        }


def elect_signs(ing, k_pct=20.0, trim_scope="global", trim=True, elect=True):
    """
    Trim every task vector and elect the sign per position. With trim=False every non-zero entry
    survives (same as k = 100); with elect=False gamma is all-ones-like "no sign filter" (the
    disjoint merge then averages every kept entry), used for the ablation of the Elect step.
    """
    n = ing.n
    s = torch.zeros(n, dtype=torch.float32)
    pos = torch.zeros(n, dtype=torch.int16)
    neg = torch.zeros(n, dtype=torch.int16)
    signed, thresholds, kept = [], [], []
    for t in range(ing.T):
        flat = ing.flat_tau(t)
        mask, thr = trim_mask(flat, 100.0 if not trim else k_pct, trim_scope, ing.layout)
        sm = flat.sign().to(torch.int8)
        sm.mul_(mask.to(torch.int8))
        signed.append(sm)
        thresholds.append(thr)
        kept.append(int(mask.sum()))
        pos += (sm > 0).to(torch.int16)
        neg += (sm < 0).to(torch.int16)
        s += torch.where(mask, flat, torch.zeros((), dtype=flat.dtype))
        del flat, mask
    gamma = s.sign().to(torch.int8)
    del s
    return Election(k_pct if trim else 100.0, trim_scope, trim, elect, signed, gamma, pos, neg,
                    thresholds, kept)


def disjoint_merge(ing, el, mode="dis-mean"):
    """
    tau_m (float64 flat). "dis-mean": sum of the kept entries agreeing with gamma / their count
    (paper); "mean_over_T": the same sum / T (ablation: sparsify without renormalising).
    Entries are re-read per key in float64 from the mapped checkpoints.
    """
    if mode not in MERGE_MODES:
        raise ValueError(f"merge mode {mode!r} not in {MERGE_MODES}")
    n = ing.n
    acc = torch.zeros(n, dtype=torch.float64)
    cnt = torch.zeros(n, dtype=torch.int16)
    sel_all = []
    for t in range(ing.T):
        sm = el.signed[t]
        sel = (sm != 0) & (sm == el.gamma) if el.elect else (sm != 0)
        sel_all.append(sel)
        cnt += sel.to(torch.int16)
    for k, o, m in ing.layout.slices():
        for t in range(ing.T):
            sel = sel_all[t][o:o + m]
            if not bool(sel.any()):
                continue
            d = ing.tau(t, k, torch.float64)
            acc[o:o + m] += torch.where(sel, d, torch.zeros((), dtype=torch.float64))
    if mode == "dis-mean":
        acc /= cnt.clamp(min=1).to(torch.float64)
    else:
        acc /= float(ing.T)
    return acc, cnt


def mean_tensor(states, key):
    """Exactly main_eval.merge_checkpoints for one key: float64 sum in list order / T, cast back."""
    v0 = states[0][key]
    acc = v0.to(torch.float64).clone()
    for s in states[1:]:
        acc += s[key].to(torch.float64)
    return (acc / len(states)).to(v0.dtype)


def assemble(ing, tau_m, lam=1.0, mean_cache=None):
    """
    The merged state dict, in the key order of the first checkpoint: TIES keys are
    (theta_init + lambda * tau_m) cast to the checkpoint dtype, the other keys the plain mean.
    `mean_cache` (dict) avoids recomputing the mean keys across lambda / k cells of a sweep.
    """
    out = {}
    for k, v in ing.states[0].items():
        if k in ing.layout.offsets:
            o, m = ing.layout.offsets[k]
            merged = ing.base.state[k].to(torch.float64) + lam * tau_m[o:o + m].view(ing.layout.shapes[k])
            out[k] = merged.to(v.dtype).contiguous()
        else:
            if mean_cache is not None and k in mean_cache:
                out[k] = mean_cache[k]
            else:
                out[k] = mean_tensor(ing.states, k)
                if mean_cache is not None:
                    mean_cache[k] = out[k]
    return out


# --------------------------------------------------------------------------- #
# one-call API
# --------------------------------------------------------------------------- #
class Merger:
    """
    Reusable merge of one ingredient list: `tau(k, ...)` caches the merged task vector per
    setting so a lambda sweep only re-assembles; `plain_mean()` is the merge_checkpoints result.
    """

    def __init__(self, paths, base):
        self.ing = Ingredients(paths, base)
        self._mean_cache = {}
        self._tau_cache = {}

    def election(self, k_pct=20.0, trim_scope="global", trim=True, elect=True):
        return elect_signs(self.ing, k_pct, trim_scope, trim, elect)

    def tau(self, k_pct=20.0, trim_scope="global", merge="dis-mean", trim=True, elect=True, cache=True):
        key = (float(k_pct), trim_scope, merge, bool(trim), bool(elect))
        if cache and key in self._tau_cache:
            return self._tau_cache[key]
        el = self.election(k_pct, trim_scope, trim, elect)
        tau_m, cnt = disjoint_merge(self.ing, el, merge)
        summary = el.summary()
        summary.update({"k": key[0], "trim_scope": trim_scope, "merge": merge, "trim": bool(trim),
                        "elect": bool(elect), "T": self.ing.T,
                        "merged_frac": int((cnt > 0).sum()) / self.ing.n,
                        "tau_m_norm": float(tau_m.norm())})
        res = (tau_m, summary)
        if cache:
            self._tau_cache[key] = res
        return res

    def state(self, tau_m, lam=1.0):
        return assemble(self.ing, tau_m, lam, self._mean_cache)

    def clear_cache(self):
        """Drop the cached merged task vectors (2.5 GiB each for the backbone); the mean keys stay."""
        self._tau_cache.clear()

    def plain_mean(self):
        """The existing merge (main_eval.merge_checkpoints) over every key, same arithmetic."""
        return {k: mean_tensor(self.ing.states, k) for k in self.ing.states[0]}


def ties_merge(paths, base, k_pct=20.0, lam=1.0, trim_scope="global", merge="dis-mean",
               trim=True, elect=True):
    """merge list + base -> (state_dict, summary); see the module docstring."""
    m = Merger(paths, base)
    tau_m, summary = m.tau(k_pct, trim_scope, merge, trim, elect, cache=False)
    summary["lambda"] = float(lam)
    summary["n_ties_keys"] = len(m.ing.ties_keys)
    summary["n_mean_keys"] = len(m.ing.mean_keys)
    summary["n_copy_keys"] = len(m.ing.copy_keys)
    return m.state(tau_m, lam), summary


def cell_tag(k_pct=20.0, lam=1.0, merge="dis-mean", trim=True, elect=True, trim_scope="global"):
    """Short name of one merge setting, used for output folders: k20_lam1.0[_meanT][_notrim][_noelect][_tensor]."""
    k = int(k_pct) if float(k_pct).is_integer() else k_pct
    tag = f"k{k}_lam{lam:g}"
    if merge == "mean_over_T":
        tag += "_meanT"
    if not trim:
        tag += "_notrim"
    if not elect:
        tag += "_noelect"
    if trim_scope != "global":
        tag += f"_{trim_scope}"
    return tag
