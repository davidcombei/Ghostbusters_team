"""
Datasets for the B3 wave, built on utils.data_utils.SpoofAudioDataset without changing it.

    utt_meta             src / lang of a protocol id ("offline/en/Train_000000.wav")
    load_pair_map        offline -> online twins from the RTCFake meta CSV (train/ prefix stripped)
    stratified_subset    deterministic label x lang x src subset (--train_subset_frac, smoke tests)
    SpoofAudioDatasetB3  training items as dicts: x [, x_b], y, utt, meta (stage-count vector)
    build_dataset_b3     the train / dev datasets from argparse args, like build_dataset_from_protocol

The audio path is the parent's: _load (16 kHz + optional loudness normalisation) -> _augment
(RTC augmenter -> level augmenter -> RawBoost, training only) -> _featurize (random crop in
training, first crop otherwise, then the backbone's feature extractor). Only the augmenter
object differs (RTCAugmenterB3) and only when a B3 flag asks for it.
"""

import csv
from pathlib import Path

import numpy as np
import torch

from utils.data_utils import (
    LABEL_TO_ID,
    SpoofAudioDataset,
    build_loudness,
    read_protocol,
)
from utils.level_augment import ChainAugmenter, LevelAugConfig, LevelAugmenter

from .augment import build_rtc_augmenter_b3

SRCS = ("offline", "online")
LANGS = ("en", "zh")


def utt_meta(utt_id):
    parts = str(utt_id).replace("\\", "/").split("/")
    src = parts[0] if parts and parts[0] in SRCS else "?"
    lang = parts[1] if len(parts) > 2 and parts[1] in LANGS else "?"
    return {"src": src, "lang": lang}


def composition(file_list, labels):
    """{(src, lang, label_name): count} over the protocol."""
    out = {}
    for u in file_list:
        m = utt_meta(u)
        name = "bonafide" if labels[u] == 1 else "spoof"
        key = (m["src"], m["lang"], name)
        out[key] = out.get(key, 0) + 1
    return out


def load_pair_map(csv_path, file_list, prefix="train/"):
    """
    {offline_id: online_id} for the ids in `file_list`, from a CSV with columns
    offline_id,online_id whose ids carry a `train/` (or `dev/`) prefix. Returns (map, stats).
    """
    wanted = set(file_list)
    pairs, n_rows, n_blank = {}, 0, 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            off = (row.get("offline_id") or "").strip()
            on = (row.get("online_id") or "").strip()
            if off.startswith(prefix):
                off = off[len(prefix):]
            if on.startswith(prefix):
                on = on[len(prefix):]
            if not on:
                n_blank += 1
                continue
            if off in wanted and on in wanted:
                pairs[off] = on
    return pairs, {"rows": n_rows, "blank_online": n_blank, "found": len(pairs)}


def stratified_subset(file_list, labels, frac, seed=0):
    """Keep round(frac * n) items per (label, lang, src) stratum (>= 1 when the stratum is not empty)."""
    if frac >= 1.0:
        return list(file_list)
    if frac <= 0.0:
        raise ValueError("--train_subset_frac must be in (0, 1]")
    rng = np.random.default_rng(int(seed))
    strata = {}
    for i, u in enumerate(file_list):
        m = utt_meta(u)
        strata.setdefault((labels[u], m["lang"], m["src"]), []).append(i)
    keep = []
    for key in sorted(strata):
        idx = strata[key]
        n = max(1, int(round(frac * len(idx))))
        keep += [idx[j] for j in rng.choice(len(idx), size=n, replace=False)]
    keep.sort()
    return [file_list[i] for i in keep]


def build_augmenter_b3(args):
    """(augmenter chain for the dataset, the RTCAugmenterB3 inside it or None), like build_augmenter."""
    rtc = build_rtc_augmenter_b3(args)
    if not getattr(args, "use_level_aug", False):
        return rtc, rtc
    level = LevelAugmenter(LevelAugConfig(
        p_apply=getattr(args, "level_p_apply", 0.8),
        gain_mode=getattr(args, "level_gain_mode", "absolute"),
        target_db=tuple(getattr(args, "level_target_db", (-38.0, -12.0))),
        gain_db=tuple(getattr(args, "level_gain_db", (-10.0, 10.0))),
        p_agc=getattr(args, "level_agc_p", 0.30),
        p_comp=getattr(args, "level_comp_p", 0.15),
        p_limit=getattr(args, "level_limit_p", 0.15),
        clip_p=getattr(args, "level_clip_p", 0.3),
        seed=getattr(args, "seed", None),
    ))
    return (ChainAugmenter([rtc, level]) if rtc is not None else level), rtc


class SpoofAudioDatasetB3(SpoofAudioDataset):
    """
    Training dataset of the B3 wave. Items are dicts so the default collate handles every mode:
        {"x": feat, "y": label, "utt": utt_id, "meta": int32 vector}            plain
        {"x": feat_a, "x_b": feat_b, "y": label, "utt": utt_id, "meta": ...}   consistency
    `meta` is the RTC augmenter's stage-count delta for this clip (all views), zeros when no
    RTC augmenter is set. Eval / dev datasets keep using the parent class (tuple items).
    """

    def __init__(self, *args, rtc_augmenter=None, pair_mode="both", clean_view=False, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.train:
            raise ValueError("SpoofAudioDatasetB3 is the training dataset; use SpoofAudioDataset for dev/eval")
        self.rtc_augmenter = rtc_augmenter
        self.meta_keys = list(rtc_augmenter.meta_keys) if rtc_augmenter is not None else []
        if pair_mode not in ("twin", "aug2", "both"):
            raise ValueError(f"pair_mode must be twin, aug2 or both, got {pair_mode}")
        self.pair_mode = pair_mode
        self.clean_view = clean_view
        if self.pair_map is not None and pair_mode == "twin":
            before = len(self.file_list)
            self.file_list = [u for u in self.file_list if self.pair_map.get(u)]
            self.dropped_without_twin = before - len(self.file_list)
        else:
            self.dropped_without_twin = 0
        self.langs = [utt_meta(u)["lang"] for u in self.file_list]
        self.srcs = [utt_meta(u)["src"] for u in self.file_list]
        self.label_ids = [self.labels[u] for u in self.file_list]

    def _snapshot(self):
        if self.rtc_augmenter is None:
            return None
        return dict(self.rtc_augmenter.stage_counts)

    def _meta_vector(self, before):
        if self.rtc_augmenter is None or before is None:
            return torch.zeros(len(self.meta_keys), dtype=torch.int32)
        return torch.from_numpy(self.rtc_augmenter.counts_vector(before))

    def __getitem__(self, index):
        utt_id = self.file_list[index]
        audio, sr = self._load(self.base_dir / utt_id)
        before = self._snapshot()
        item = {"y": self.labels[utt_id], "utt": utt_id}
        if self.pair_map is None:
            item["x"] = self._featurize(self._augment(audio, sr))
        else:
            if self.clean_view:
                item["x"] = self._featurize(audio)                  # un-augmented teacher view
            else:
                item["x"] = self._featurize(self._augment(audio, sr))
            pair_id = self.pair_map.get(utt_id) if self.pair_mode != "aug2" else None
            if pair_id is not None:
                audio_b, _ = self._load(self.pair_base_dir / pair_id)
            else:
                audio_b = audio                                     # same clip, independent augmentation
            item["x_b"] = self._featurize(self._augment(audio_b, sr))
        item["meta"] = self._meta_vector(before)
        return item


def build_dataset_b3(protocol_path, base_dir, mode, args):
    """
    mode "train": SpoofAudioDatasetB3 with the B3 augmenter, --train_subset_frac and (when
    --consistency != none) the offline/online pair map. mode "dev": exactly the original
    SpoofAudioDataset as build_dataset_from_protocol builds it.
    Returns (dataset, file_list, labels, info).
    """
    file_list, labels = read_protocol(protocol_path, require_label=True)
    info = {"n_protocol": len(file_list)}
    if mode == "dev":
        dataset = SpoofAudioDataset(file_list=file_list, base_dir=base_dir, labels=labels, args=args,
                                    algo=getattr(args, "algo", 0), use_rawboost=False,
                                    ssl_name=getattr(args, "ssl_name", None), augmenter=None,
                                    train=False, loudness=build_loudness(args))
        return dataset, file_list, labels, info

    frac = float(getattr(args, "train_subset_frac", 1.0))
    if frac < 1.0:
        file_list = stratified_subset(file_list, labels, frac, seed=getattr(args, "seed", 0))
    info["n_train"] = len(file_list)
    info["composition"] = {"/".join(k): v for k, v in sorted(composition(file_list, labels).items())}

    pair_map = None
    if getattr(args, "consistency", "none") != "none":
        pair_mode = getattr(args, "consistency_pairs", "both")
        if pair_mode == "aug2":
            pair_map = {}                                   # consistency on, no twins
        else:
            pair_map, stats = load_pair_map(getattr(args, "pair_csv"), file_list)
            info["pairs"] = stats
    augmenter, rtc = build_augmenter_b3(args)
    dataset = SpoofAudioDatasetB3(
        file_list=file_list, base_dir=base_dir, labels=labels, args=args,
        algo=getattr(args, "algo", 0), use_rawboost=getattr(args, "use_rawboost", False),
        ssl_name=getattr(args, "ssl_name", None), augmenter=augmenter, train=True,
        pair_map=pair_map, pair_base_dir=base_dir, loudness=build_loudness(args),
        rtc_augmenter=rtc,
        pair_mode=getattr(args, "consistency_pairs", "both"),
        clean_view=getattr(args, "consistency_clean_view", False),
    )
    if pair_map is not None:
        info["dropped_without_twin"] = dataset.dropped_without_twin
        info["n_train"] = len(dataset)
    return dataset, list(dataset.file_list), labels, info
