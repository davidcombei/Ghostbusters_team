"""
Datasets for the B4 cycle: b3wave.data with the B4 augmenter factory, nothing else changed.

    build_augmenter_b4   (augmenter chain, the RTC augmenter inside it), like build_augmenter_b3
    build_dataset_b4     the train / dev datasets from argparse args, like build_dataset_b3

Both are copies of the b3wave builders with `build_rtc_augmenter_b3` replaced by
`build_rtc_augmenter_b4`, which returns the B3 augmenter itself when every B4 flag is at its
default; so with the levers off the datasets are the b3wave ones (`checks data`). Dev / eval keep the
original SpoofAudioDataset; training reuses SpoofAudioDatasetB3 (dict items with the per-clip
`meta` count vector the trainer sums per class).
"""

from utils.data_utils import SpoofAudioDataset, build_loudness, read_protocol
from utils.level_augment import ChainAugmenter, LevelAugConfig, LevelAugmenter

from b3wave.data import SpoofAudioDatasetB3, composition, load_pair_map, stratified_subset

from .augment import build_rtc_augmenter_b4


def build_augmenter_b4(args):
    """(augmenter chain for the dataset, the RTC augmenter inside it or None), like build_augmenter_b3."""
    rtc = build_rtc_augmenter_b4(args)
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


def build_dataset_b4(protocol_path, base_dir, mode, args):
    """
    mode "train": SpoofAudioDatasetB3 with the B4 augmenter, --train_subset_frac and (when
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
    augmenter, rtc = build_augmenter_b4(args)
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
