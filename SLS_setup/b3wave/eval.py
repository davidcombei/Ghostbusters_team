"""
Multi-crop scoring for the B3 wave (zero-training lever, roadmap E1.3).

The SLS head is fixed-length (4.04 s = 64,600 samples); today every evaluation reads the first
crop only (utils.data_utils.pad_audio(random_start=False)). Here a clip is scored on all
4.04 s windows at a 50 % hop (capped at `max_crops`) and the per-crop LOGITS are aggregated
(mean or median) before the softmax, so p_spoof and d = z_spoof - z_bona keep their meaning.

    score_dataset_b3(model, dataset, device, ..., multicrop="first")   == utils.local_eval.score_dataset
    score_dataset_b3(..., multicrop="mean" | "median")                 multi-crop

    python -m b3wave.eval --model_path ... --multicrop mean ...         main_eval.py + --multicrop
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from main_eval import merge_checkpoints
from model.sls_model import ModelSLS, ssl_path
from utils.data_utils import SpoofAudioDataset, build_loudness, pad_audio, read_protocol, set_random_seed
from utils.local_eval import score_dataset

CUT = 64600
HOP = 32300


def crop_starts(n_samples, cut=CUT, hop=HOP, max_crops=8):
    if n_samples <= cut:
        return [0]
    starts = list(range(0, n_samples - cut + 1, hop))
    if starts[-1] + cut < n_samples and len(starts) < max_crops:      # cover the tail
        starts.append(n_samples - cut)
    if len(starts) > max_crops:                                       # evenly spaced subset
        pick = np.linspace(0, len(starts) - 1, max_crops).round().astype(int)
        starts = [starts[i] for i in pick]
    return starts


class MultiCropDataset(Dataset):
    """Wraps a SpoofAudioDataset (eval mode): item = (feats (n_crops, ...), utt_id)."""

    def __init__(self, base: SpoofAudioDataset, max_crops=8):
        if base.train:
            raise ValueError("MultiCropDataset is for evaluation datasets")
        self.base = base
        self.max_crops = max_crops

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        utt_id = self.base.file_list[index]
        audio, _ = self.base._load(self.base.base_dir / utt_id)
        if len(audio) < CUT:
            crops = [pad_audio(audio, CUT, random_start=False)]
        else:
            crops = [audio[s:s + CUT] for s in crop_starts(len(audio), max_crops=self.max_crops)]
        feats = torch.stack([self.base._featurize(c) for c in crops])   # exactly CUT samples: no pad/crop
        return feats, utt_id


def collate_multicrop(batch):
    feats = torch.cat([b[0] for b in batch], dim=0)
    counts = [int(b[0].shape[0]) for b in batch]
    ids = [b[1] for b in batch]
    return feats, counts, ids


def score_dataset_b3(model, dataset, device, batch_size=32, num_workers=4, multicrop="first", max_crops=8):
    """(utt_ids, p_spoof, d) in dataset order; `first` is bit-identical to utils.local_eval.score_dataset."""
    if multicrop == "first":
        return score_dataset(model, dataset, device, batch_size, num_workers)
    if multicrop not in ("mean", "median"):
        raise ValueError(f"multicrop must be first, mean or median, got {multicrop}")
    mc = MultiCropDataset(dataset, max_crops=max_crops)
    # a batch holds up to max_crops crops per clip: keep the crop count per forward pass near
    # `batch_size` so memory stays at the single-crop level (32 clips x 8 crops took 20 GB)
    clips_per_batch = max(1, batch_size // max_crops)
    loader = DataLoader(mc, batch_size=clips_per_batch, num_workers=num_workers, shuffle=False,
                        pin_memory=torch.cuda.is_available(), collate_fn=collate_multicrop)
    was_training = model.training
    model.eval()
    ids, p_all, d_all, n_crops = [], [], [], []
    try:
        with torch.no_grad():
            for feats, counts, utt_ids in loader:
                logits = model(feats.to(device)).float()
                per_clip = torch.split(logits, counts, dim=0)
                agg = torch.stack([c.mean(dim=0) if multicrop == "mean" else c.median(dim=0).values
                                   for c in per_clip])
                p_all.append(torch.softmax(agg, dim=1)[:, 0].cpu().numpy())
                d_all.append((agg[:, 0] - agg[:, 1]).cpu().numpy())
                ids.extend(utt_ids)
                n_crops.extend(counts)
    finally:
        model.train(was_training)
    score_dataset_b3.last_crop_counts = n_crops
    return ids, np.concatenate(p_all), np.concatenate(d_all)


def write_scores_b3(model, dataset, device, output_score_path, batch_size, num_workers, multicrop, max_crops):
    ids, p, _ = score_dataset_b3(model, dataset, device, batch_size, num_workers, multicrop, max_crops)
    score_dir = os.path.dirname(output_score_path)
    if score_dir:
        os.makedirs(score_dir, exist_ok=True)
    with open(output_score_path, "w", encoding="utf-8") as f:
        for utt_id, score in zip(ids, p):
            f.write(f"{utt_id} {score:.10f}\n")


def main():
    parser = argparse.ArgumentParser(description="main_eval.py with multi-crop scoring (B3 wave)")
    model_src = parser.add_mutually_exclusive_group(required=True)
    model_src.add_argument("--model_path", type=str)
    model_src.add_argument("--model_merging", action="store_true")
    parser.add_argument("--merge_list", type=str, default=None)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--protocol_path", type=str, required=True)
    parser.add_argument("--score_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_layers", type=int, default=48)
    parser.add_argument("--ssl_name", type=str, default="facebook/wav2vec2-xls-r-2b")
    parser.add_argument("--devices", type=int, nargs="+", default=None)
    parser.add_argument("--layer_split", type=int, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--loudness_norm", action="store_true", default=False)
    parser.add_argument("--loudness_norm_lufs", type=float, default=-23.0)
    parser.add_argument("--multicrop", type=str, default="first", choices=["first", "mean", "median"])
    parser.add_argument("--multicrop_max", type=int, default=8)
    parser.add_argument("--cudnn-deterministic-toggle", action="store_false", default=True)
    parser.add_argument("--cudnn-benchmark-toggle", action="store_true", default=False)
    args = parser.parse_args()
    args.ssl_name = ssl_path(args.ssl_name)
    set_random_seed(args.seed, args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    file_list, _ = read_protocol(args.protocol_path, require_label=None)
    dataset = SpoofAudioDataset(file_list=file_list, base_dir=args.eval_data_path, labels=None,
                                ssl_name=args.ssl_name, loudness=build_loudness(args))
    model = ModelSLS(args, device)
    device = model.input_device
    if args.model_merging:
        if not args.merge_list:
            raise ValueError("--model_merging needs --merge_list")
        with open(args.merge_list, encoding="utf-8") as f:
            paths = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
        state = merge_checkpoints(paths)
        list_dir = os.path.dirname(args.score_path)
        if list_dir:
            os.makedirs(list_dir, exist_ok=True)
        with open(os.path.join(list_dir, "merge_list.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(paths) + "\n")
    else:
        state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    write_scores_b3(model, dataset, device, args.score_path, args.batch_size, args.num_workers,
                    args.multicrop, args.multicrop_max)
    print(f"Scores saved to: {args.score_path} (multicrop={args.multicrop})")


if __name__ == "__main__":
    main()
