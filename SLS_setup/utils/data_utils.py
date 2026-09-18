import os
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

from .RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav
from .rtc_augment import RTCAugConfig, RTCAugmenter, make_deepfilternet_suppressor


LABEL_TO_ID = {
    "spoof": 0,
    "fake": 0,
    "bonafide": 1,
    "bona-fide": 1,
    "real": 1,
}


def set_random_seed(random_seed, args=None):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    cudnn_deterministic = True if args is None else args.cudnn_deterministic_toggle
    cudnn_benchmark = False if args is None else args.cudnn_benchmark_toggle
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


def normalize_label(label):
    key = label.strip().lower()
    if key not in LABEL_TO_ID:
        raise ValueError(f"Unsupported label: {label}")
    return LABEL_TO_ID[key]


def read_protocol(protocol_path, require_label=None):
    file_list = []
    labels = {}

    with open(protocol_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 1:
                if require_label is True:
                    raise ValueError(f"Missing label at {protocol_path}:{line_no}")
                file_list.append(parts[0])
                continue
            if len(parts) == 2:
                utt_id, label = parts
            elif len(parts) >= 5:
                utt_id, label = parts[1], parts[4]
            else:
                raise ValueError(f"Invalid protocol line at {protocol_path}:{line_no}: {line.rstrip()}")

            file_list.append(utt_id)
            labels[utt_id] = normalize_label(label)

    has_labels = bool(labels)
    if require_label is False and has_labels:
        raise ValueError(f"Expected an unlabeled protocol file: {protocol_path}")
    return file_list, labels if has_labels else None


def pad_audio(audio, max_len=64600, random_start=False):
    """Tile short clips; crop long ones. random_start=True -> random crop (use for training)."""
    audio_len = audio.shape[0]
    if audio_len >= max_len:
        if random_start:
            s = random.randint(0, audio_len - max_len)
            return audio[s:s + max_len]
        return audio[:max_len]
    num_repeats = int(max_len / audio_len) + 1
    return np.tile(audio, (1, num_repeats))[:, :max_len][0]


def build_augmenter(args):
    """Build an RTCAugmenter from argparse-style args (all optional)."""
    if not getattr(args, "use_rtc_aug", False):
        return None
    cfg = RTCAugConfig(
        noise_dirs=getattr(args, "aug_noise_dirs", []) or [],
        music_dirs=getattr(args, "aug_music_dirs", []) or [],
        rir_dirs=getattr(args, "aug_rir_dirs", []) or [],
        p_apply=getattr(args, "aug_p_apply", 0.8),
        max_stages=getattr(args, "aug_max_stages", 4),
        seed=getattr(args, "seed", None),
    )
    if getattr(args, "aug_use_deepfilternet", False):
        cfg.suppress_fn = make_deepfilternet_suppressor()
        if cfg.suppress_fn is None:
            print("[data_utils] DeepFilterNet not available; suppression stage disabled")
    return RTCAugmenter(cfg)


class SpoofAudioDataset(Dataset):
    """
    Modes:
      - default: returns (feat, label, utt_id)  (or (feat, utt_id) when unlabeled)
      - paired : returns (feat_offline, feat_online, label, utt_id) for consistency losses.
                 Requires `pair_map` {utt_id -> online_utt_id}; both are loaded and
                 *independently* augmented, so the model must agree across RTC conditions.
    """

    def __init__(self, file_list, base_dir, labels=None, args=None, algo=0, use_rawboost=False,
                 ssl_name=None, augmenter=None, train=False, pair_map=None, pair_base_dir=None):
        self.file_list = file_list
        self.base_dir = Path(base_dir)
        self.labels = labels
        self.args = args
        self.algo = algo
        self.use_rawboost = use_rawboost
        self.augmenter = augmenter
        self.train = train
        self.cut = 64600
        self.pair_map = pair_map
        self.pair_base_dir = Path(pair_base_dir) if pair_base_dir else self.base_dir

        self.is_w2v_bert = ssl_name is not None and "w2v-bert" in ssl_name.lower()
        self.is_qwen3_asr = ssl_name is not None and "qwen3-asr" in ssl_name.lower()
        self.feature_extractor = None
        if self.is_w2v_bert or self.is_qwen3_asr:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(ssl_name)

    def __len__(self):
        return len(self.file_list)

    # ------------------------------------------------------------------ #
    def _load(self, wav_path):
        audio, sr = librosa.load(str(wav_path), sr=16000)
        return audio.astype(np.float32), sr

    def _augment(self, audio, sr):
        if not self.train:
            return audio
        if self.augmenter is not None:
            audio = self.augmenter(audio, sr)       # RTC chain first (codec, noise, PLC ...)
        if self.use_rawboost:
            audio = process_rawboost_feature(audio, sr, self.args, self.algo)
        return audio

    def _featurize(self, audio):
        audio = pad_audio(audio, self.cut, random_start=self.train)
        if self.is_w2v_bert:
            inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            return inputs.input_features[0]
        if self.is_qwen3_asr:
            # (n_mel, frames) log-mel without the default 30 s padding; the model pads the frame
            # axis to whole encoder chunks itself
            inputs = self.feature_extractor(audio, sampling_rate=16000, padding="longest", n_window=0,
                                            return_tensors="pt")
            return inputs.input_features[0]
        return torch.tensor(audio, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    def __getitem__(self, index):
        utt_id = self.file_list[index]
        audio, sr = self._load(self.base_dir / utt_id)
        feat = self._featurize(self._augment(audio, sr))

        if self.pair_map is not None:
            pair_id = self.pair_map.get(utt_id)
            if pair_id is not None:
                audio_b, _ = self._load(self.pair_base_dir / pair_id)
            else:
                audio_b = audio                     # no online twin -> augment same clip twice
            feat_b = self._featurize(self._augment(audio_b, sr))
            return feat, feat_b, self.labels[utt_id], utt_id

        if self.labels is None:
            return feat, utt_id
        return feat, self.labels[utt_id], utt_id


def build_dataset_from_protocol(protocol_path, base_dir, mode, args=None, algo=0,
                                pair_map=None, pair_base_dir=None):
    require_label = mode in {"train", "dev"}
    file_list, labels = read_protocol(protocol_path, require_label=require_label)
    is_train = mode == "train"
    dataset = SpoofAudioDataset(
        file_list=file_list,
        base_dir=base_dir,
        labels=labels,
        args=args,
        algo=algo,
        use_rawboost=is_train and getattr(args, "use_rawboost", True),
        ssl_name=getattr(args, "ssl_name", None),
        augmenter=build_augmenter(args) if is_train else None,
        train=is_train,
        pair_map=pair_map if is_train else None,
        pair_base_dir=pair_base_dir,
    )
    return dataset, file_list, labels


def process_rawboost_feature(feature, sr, args, algo):
    if algo == 1:
        return LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW,
            args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr,
        )
    if algo == 2:
        return ISD_additive_noise(feature, args.P, args.g_sd)
    if algo == 3:
        return SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG,
            args.maxG, sr,
        )
    if algo == 4:
        feature = process_rawboost_feature(feature, sr, args, 1)
        feature = process_rawboost_feature(feature, sr, args, 2)
        return process_rawboost_feature(feature, sr, args, 3)
    if algo == 5:
        feature = process_rawboost_feature(feature, sr, args, 1)
        return process_rawboost_feature(feature, sr, args, 2)
    if algo == 6:
        feature = process_rawboost_feature(feature, sr, args, 1)
        return process_rawboost_feature(feature, sr, args, 3)
    if algo == 7:
        feature = process_rawboost_feature(feature, sr, args, 2)
        return process_rawboost_feature(feature, sr, args, 3)
    if algo == 8:
        feature_1 = process_rawboost_feature(feature, sr, args, 1)
        feature_2 = process_rawboost_feature(feature, sr, args, 2)
        return normWav(feature_1 + feature_2, 0)
    return feature