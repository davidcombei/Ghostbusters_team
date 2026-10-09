import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.sls_model import ModelSLS, ssl_path
from utils.data_utils import SpoofAudioDataset, build_loudness, read_protocol, set_random_seed

# Checkpoints averaged when --model_merging is set (all must share the same architecture/args).
MERGE_CHECKPOINTS = [
    # "exp/run1/epoch_10.pth",
    # "exp/run1/epoch_11.pth",
    # "exp/run1/epoch_12.pth",
]


def merge_checkpoints(paths):
    """Plain element-wise mean of the state_dicts in `paths`."""
    if not paths:
        raise ValueError("--model_merging set but MERGE_CHECKPOINTS in main_eval.py is empty")
    merged, dtypes = {}, {}
    for i, path in enumerate(paths):
        print(f"Merging [{i + 1}/{len(paths)}]: {path}")
        state = torch.load(path, map_location="cpu")
        if i == 0:
            for k, v in state.items():
                dtypes[k] = v.dtype
                merged[k] = v.to(torch.float64).clone()
        else:
            if state.keys() != merged.keys():
                raise ValueError(f"{path} has different keys than {paths[0]}")
            for k, v in state.items():
                merged[k] += v.to(torch.float64)
        del state
    return {k: (v / len(paths)).to(dtypes[k]) for k, v in merged.items()}


def write_scores(data_loader, model, device, output_score_path):
    model.eval()
    score_dir = os.path.dirname(output_score_path)
    if score_dir:
        os.makedirs(score_dir, exist_ok=True)

    with open(output_score_path, "w", encoding="utf-8") as f, torch.no_grad():
        for batch_x, utt_ids in tqdm(data_loader, desc="Evaluating", unit="batch"):
            batch_x = batch_x.to(device)
            probs = torch.softmax(model(batch_x), dim=1)
            fake_scores = probs[:, 0].detach().cpu().numpy()
            for utt_id, score in zip(utt_ids, fake_scores):
                f.write(f"{utt_id} {score:.10f}\n")



def main():
    parser = argparse.ArgumentParser(description="Evaluate XLSR-AASIST and write fake scores")
    model_src = parser.add_mutually_exclusive_group(required=True)
    model_src.add_argument("--model_path", type=str)
    model_src.add_argument("--model_merging", action="store_true",
                           help="average the weights of the checkpoints listed in MERGE_CHECKPOINTS")
    parser.add_argument("--merge_list", type=str, default=None,
                        help="with --model_merging: text file with one checkpoint path per line, "
                             "used instead of MERGE_CHECKPOINTS; the list used is copied next to "
                             "--score_path as merge_list.txt")
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--protocol_path", type=str, required=True)
    parser.add_argument("--score_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--arch", type=str, default="sls", choices=["aasist", "sls"],
                        help="aasist = XLS-R 300M + AASIST (model/model.py); "
                             "sls = XLS-R 2B + SLS (model/sls_model.py)")
    parser.add_argument("--n_layers", type=int, default=48,
                        help="[sls] SSL transformer layers, must match the checkpoint "
                             "(clamped to the backbone's depth)")
    parser.add_argument("--ssl_name", type=str, default="facebook/wav2vec2-xls-r-2b",
                        help="[sls] hub id or local dir of the SSL front-end, must match the checkpoint: "
                             "facebook/wav2vec2-xls-r-2b, facebook/w2v-bert-2.0, Qwen/Qwen3-ASR-0.6B or "
                             "Qwen/Qwen3-ASR-1.7B (audio encoder only; needs transformers>=5.14)")
    parser.add_argument("--devices", type=int, nargs="+", default=None,
                        help="[sls] GPU ids to split the encoder over, e.g. --devices 4 5 6 7. "
                             "Inference of the 48-layer model also fits on one 32 GB card; "
                             "overrides --device when given.")
    parser.add_argument("--layer_split", type=int, nargs="+", default=None,
                        help="[sls] layers per GPU when --devices is used")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--loudness_norm", action="store_true", default=False,
                        help="normalise every clip to --loudness_norm_lufs at read time; pass it when the "
                             "checkpoint was trained with it (config.yaml: loudness_norm)")
    parser.add_argument("--loudness_norm_lufs", type=float, default=-23.0,
                        help="target integrated loudness in LUFS (must match training)")
    parser.add_argument("--cudnn-deterministic-toggle", action="store_false", default=True)
    parser.add_argument("--cudnn-benchmark-toggle", action="store_true", default=False)
    args = parser.parse_args()
    args.ssl_name = ssl_path(args.ssl_name)     # the dataset and the model must agree on the repo id

    set_random_seed(args.seed, args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    file_list, _ = read_protocol(args.protocol_path, require_label=None)     # labels, if any, are ignored
    dataset = SpoofAudioDataset(file_list=file_list, base_dir=args.eval_data_path, labels=None,
                                ssl_name=args.ssl_name if args.arch == "sls" else None,
                                loudness=build_loudness(args))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    if args.arch == "sls":
        model = ModelSLS(args, device)          # places itself (one GPU or --devices split)
        device = model.input_device
    else:
        from model.model import Model           # only needed for --arch aasist
        model = Model(args, device).to(device)
    if args.model_merging:
        merge_paths = MERGE_CHECKPOINTS
        if args.merge_list:
            with open(args.merge_list, encoding="utf-8") as f:
                merge_paths = [line.strip() for line in f
                               if line.strip() and not line.lstrip().startswith("#")]
        state_dict = merge_checkpoints(merge_paths)
        list_dir = os.path.dirname(args.score_path)
        if list_dir:
            os.makedirs(list_dir, exist_ok=True)
        with open(os.path.join(list_dir, "merge_list.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(merge_paths) + "\n")
    else:
        state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    write_scores(loader, model, device, args.score_path)
    print(f"Scores saved to: {args.score_path}")


if __name__ == "__main__":
    main()
