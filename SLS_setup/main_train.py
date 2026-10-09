import argparse
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

#from model.model import Model
from model.sls_model import ModelSLS, ssl_path
from utils.data_utils import build_dataset_from_protocol, set_random_seed

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def build_loader(dataset, batch_size, num_workers, shuffle):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    return DataLoader(dataset, **kwargs)


def train_epoch(data_loader, model, optimizer, device, criterion, out_device=None,
                step_scheduler=None, grad_clip=0.0, ema=None):
    """
    step_scheduler: stepped after every batch (warmup_cosine; epoch-level schedulers stay in main).
    grad_clip > 0: clip the global grad norm. ema: AveragedModel updated after every step.
    All three default to off, which is the original loop.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    num_total = 0
    out_device = out_device or device
    clip_params = [p for p in model.parameters() if p.requires_grad] if grad_clip > 0 else None

    for batch_x, batch_y, _ in tqdm(data_loader, desc="Training", unit="batch", ascii=True):
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).long().to(out_device)
        batch_size = batch_x.size(0)

        optimizer.zero_grad()
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        batch_loss.backward()
        if clip_params is not None:
            torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
        optimizer.step()
        if step_scheduler is not None:
            step_scheduler.step()
        if ema is not None:
            ema.update_parameters(model)

        running_loss += batch_loss.item() * batch_size
        correct += (torch.argmax(batch_out, dim=1) == batch_y).sum().item()
        num_total += batch_size

    return running_loss / num_total, 100.0 * correct / num_total


def evaluate_dev(data_loader, model, device, criterion, out_device=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    num_total = 0
    out_device = out_device or device

    with torch.no_grad():
        for batch_x, batch_y, _ in tqdm(data_loader, desc="Validating", unit="batch", ascii=True):
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).long().to(out_device)
            batch_size = batch_x.size(0)

            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)

            running_loss += batch_loss.item() * batch_size
            correct += (torch.argmax(batch_out, dim=1) == batch_y).sum().item()
            num_total += batch_size

    return running_loss / num_total, 100.0 * correct / num_total



def write_run_config(exp_root, args, augmenter, loudness=None):
    """exp/<run>/config.yaml (+ git.diff when the tree is dirty). Never fails the run."""
    try:
        import transformers
        import yaml
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def git(*cmd):
            return subprocess.run(["git", "-C", repo, *cmd], capture_output=True, text=True).stdout

        diff = git("diff", "HEAD")
        if diff:
            with open(os.path.join(exp_root, "git.diff"), "w", encoding="utf-8") as f:
                f.write(diff)
        config = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "git_sha": git("rev-parse", "HEAD").strip(),
            "git_dirty": bool(diff),
            "host": socket.gethostname(),
            "versions": {"torch": str(torch.__version__), "transformers": str(transformers.__version__)},
            "rtc_augmentation": augmenter.describe() if augmenter is not None else "off",
            "loudness_norm": loudness.describe() if loudness is not None else "off",
            "rawboost": f"algo {args.algo}" if args.use_rawboost else "off",
            "args": json.loads(json.dumps(vars(args), default=str)),     # plain types only
        }
        text = yaml.safe_dump(config, sort_keys=False)          # serialise first: no empty file on error
        with open(os.path.join(exp_root, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as exc:                            # bookkeeping must never stop training
        print(f"[config] could not write config.yaml: {exc}")


def link_best(ckpt_dir, name, epoch_path):
    """ckpt/<name> -> epoch file, as a relative symlink (movable dir, no extra disk)."""
    link = os.path.join(ckpt_dir, name)
    if os.path.lexists(link):
        os.unlink(link)
    os.symlink(os.path.basename(epoch_path), link)


EPOCH_CKPT_RE = re.compile(r"^epoch_(\d+)_dev_loss_.*\.pth$")
LAST_STATE = "last_state.pt"


def log_line(log_path, message):
    print(message)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().replace(microsecond=0)}] {message}\n")


def read_metrics(exp_root):
    """{epoch: row} from metrics.jsonl; a later row for the same epoch wins."""
    rows = {}
    path = os.path.join(exp_root, "metrics.jsonl")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[row["epoch"]] = row
    return rows


def best_links(ckpt_dir):
    """{link name: target file} for the best_by_* symlinks in ckpt_dir."""
    return {name: os.readlink(os.path.join(ckpt_dir, name)) for name in os.listdir(ckpt_dir)
            if os.path.islink(os.path.join(ckpt_dir, name))}


def prune_checkpoints(exp_root, ckpt_dir, keep_topk, keep_epochs, log_path):
    """
    Delete epoch checkpoints that can no longer be selected: not in the top-`keep_topk` by local
    WF1, not in `keep_epochs`, not a best_by_* target. An epoch's WF1 never changes, so an epoch
    outside the current top-k can never re-enter it. Stateless (re-read from metrics.jsonl), so it
    survives --resume. Epochs without a metrics row are never touched.
    """
    rows = read_metrics(exp_root)
    ranked = sorted(rows, key=lambda e: (-rows[e]["wf1"]["wf1"], e))     # ties: earlier epoch wins
    protected = set(ranked[:keep_topk]) | set(keep_epochs)
    targets = set(best_links(ckpt_dir).values())
    for name in sorted(os.listdir(ckpt_dir)):
        match = EPOCH_CKPT_RE.match(name)
        path = os.path.join(ckpt_dir, name)
        if not match or os.path.islink(path) or name in targets:
            continue
        epoch = int(match.group(1))
        if epoch in rows and epoch not in protected:
            os.remove(path)
            log_line(log_path, f"Pruned {name} (WF1={100 * rows[epoch]['wf1']['wf1']:.2f}, "
                               f"outside top-{keep_topk})")


def build_optimizer(model, trainable, args):
    """
    Default (--optim adam, --head_lr_mult 1): the original single-group Adam over `trainable`.
    Otherwise two LR groups -- SSL backbone at --lr, head (SLS, BN, FCs) at --lr * --head_lr_mult --
    and with AdamW weight decay only on matrices (no decay on biases, norms, 1-d params).
    """
    if args.optim == "adam" and args.head_lr_mult == 1.0:
        return torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

    backbone_ids = {id(p) for p in model.ssl_model.parameters()}
    groups = []
    for name, is_backbone, lr in (("backbone", True, args.lr),
                                  ("head", False, args.lr * args.head_lr_mult)):
        params = [p for p in trainable if (id(p) in backbone_ids) == is_backbone]
        if args.optim == "adamw":
            decay = [p for p in params if p.ndim > 1]
            no_decay = [p for p in params if p.ndim <= 1]
            groups += [{"params": decay, "lr": lr, "weight_decay": args.weight_decay, "name": name},
                       {"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": name}]
        else:
            groups.append({"params": params, "lr": lr, "weight_decay": args.weight_decay, "name": name})
    groups = [g for g in groups if g["params"]]
    opt_cls = torch.optim.AdamW if args.optim == "adamw" else torch.optim.Adam
    return opt_cls(groups, lr=args.lr, weight_decay=args.weight_decay)


def warmup_cosine_lambda(total_steps, warmup_steps, floor):
    """LR multiplier per step: linear warmup to 1, then cosine down to `floor` (= lr_min / lr)."""
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


def group_lrs(optimizer):
    """{group name: lr}, first group per name (backbone / head); unnamed groups are 'lr'."""
    lrs = {}
    for g in optimizer.param_groups:
        lrs.setdefault(g.get("name", "lr"), g["lr"])
    return lrs


def save_last_state(exp_root, ckpt_dir, model, optimizer, scheduler, epoch, counters, ema=None):
    """exp/<run>/last_state.pt: everything --resume needs. Written atomically (tmp + rename)."""
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "counters": counters,
        "links": best_links(ckpt_dir),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    path = os.path.join(exp_root, LAST_STATE)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def restore_run_dir(exp_root, ckpt_dir, state, log_path):
    """
    Undo whatever a killed epoch left behind after the last completed one: metrics rows and epoch
    checkpoints of later epochs, and best_by_* links re-pointed by them.
    """
    last = state["epoch"]
    rows = read_metrics(exp_root)
    metrics_path = os.path.join(exp_root, "metrics.jsonl")
    if any(e > last for e in rows):
        with open(metrics_path + ".tmp", "w", encoding="utf-8") as f:
            for e in sorted(rows):
                if e <= last:
                    f.write(json.dumps(rows[e]) + "\n")
        os.replace(metrics_path + ".tmp", metrics_path)
        log_line(log_path, f"Dropped metrics rows of unfinished epochs > {last}")
    for name in sorted(os.listdir(ckpt_dir)):
        match = EPOCH_CKPT_RE.match(name)
        if match and int(match.group(1)) > last and not os.path.islink(os.path.join(ckpt_dir, name)):
            os.remove(os.path.join(ckpt_dir, name))
            log_line(log_path, f"Removed {name} (unfinished epoch)")
    for name in best_links(ckpt_dir):
        os.unlink(os.path.join(ckpt_dir, name))
    for name, target in state["links"].items():
        os.symlink(target, os.path.join(ckpt_dir, name))


def main():
    parser = argparse.ArgumentParser(description="Train XLSR-AASIST spoof detector")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--dev_data_path", type=str, required=True)
    parser.add_argument("--train_protocol", type=str, required=True)
    parser.add_argument("--dev_protocol", type=str, required=True)
    parser.add_argument("--track", type=str, default="xlsr_aasist")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--devices", type=int, nargs="+", default=None,
                        help="GPU ids to split the XLS-R encoder over, e.g. --devices 4 5. "
                             "Needed to fit all 48 layers; overrides --device. "
                             "Do NOT combine with CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--n_layers", type=int, default=48,
                        help="number of SSL transformer layers to keep, clamped to the backbone's depth "
                             "(48 for XLS-R 2B, 24 for W2V-BERT 2.0, 18 for Qwen3-ASR-0.6B)")
    parser.add_argument("--ssl_name", type=str, default="facebook/wav2vec2-xls-r-2b",
                        help="hub id or local dir of the SSL front-end, e.g. facebook/wav2vec2-xls-r-2b, "
                             "facebook/w2v-bert-2.0, Qwen/Qwen3-ASR-0.6B or Qwen/Qwen3-ASR-1.7B "
                             "(audio encoder only; needs transformers>=5.14)")
    parser.add_argument("--layer_split", type=int, nargs="+", default=None,
                        help="layers per GPU, e.g. --layer_split 25 23. Default is an even "
                             "split; shift layers off the last card if it runs tighter (it "
                             "also holds the stacked hidden states and the SLS head).")
    parser.add_argument("--freeze_ssl", action="store_true", default=False,
                        help="freeze the SSL front-end so only the SLS + classifier head train. "
                             "Default is end-to-end training of the whole model.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="resume from a checkpoint of this exact model")
    parser.add_argument("--out_path", type=str, default="./exp")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ce_weights", type=float, nargs=2, default=[0.1, 0.9],
                        help="CrossEntropyLoss class weights in label order "
                             "(class 0 = spoof, class 1 = bonafide); inverse frequency "
                             "is about 0.19 0.81")
    parser.add_argument("--lr_scheduler", type=str, default="none",
                        choices=["none", "cosine", "plateau", "warmup_cosine"],
                        help="optional LR schedule; 'none' keeps the constant LR (default). "
                             "cosine = CosineAnnealingLR over --num_epochs; "
                             "plateau = ReduceLROnPlateau on dev loss; "
                             "warmup_cosine = per-batch linear warmup (--warmup_frac) then cosine "
                             "to --lr_min over all --num_epochs")
    parser.add_argument("--warmup_frac", type=float, default=0.0,
                        help="[warmup_cosine] share of all training steps spent warming up")
    # --- stability recipe (all off by default = the original Adam / constant-LR loop) ---
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "adamw"],
                        help="adamw: decoupled weight decay (--weight_decay), only on matrices")
    parser.add_argument("--head_lr_mult", type=float, default=1.0,
                        help="LR of the head (SLS, BN, fc1, fc2) = --lr * this; backbone stays at --lr")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="max global grad norm; 0 = no clipping")
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="EMA of the weights, updated every step (e.g. 0.9995); the EMA weights "
                             "are the ones evaluated and saved as epoch checkpoints. 0 = off")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="CrossEntropyLoss label smoothing; 0 = off")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="[plateau] LR multiplier on each reduction")
    parser.add_argument("--lr_patience", type=int, default=8,
                        help="[plateau] stale dev-loss epochs before the LR is reduced; "
                             "keep well below --earlystop_epoch or it never fires")
    parser.add_argument("--lr_min", type=float, default=0.0,
                        help="LR floor: eta_min for cosine, min_lr for plateau")
    parser.add_argument("--earlystop_epoch", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cudnn-deterministic-toggle", action="store_false", default=True)
    parser.add_argument("--cudnn-benchmark-toggle", action="store_true", default=False)

    parser.add_argument("--algo", type=int, default=5)
    parser.add_argument("--nBands", type=int, default=5)
    parser.add_argument("--minF", type=int, default=20)
    parser.add_argument("--maxF", type=int, default=8000)
    parser.add_argument("--minBW", type=int, default=100)
    parser.add_argument("--maxBW", type=int, default=1000)
    parser.add_argument("--minCoeff", type=int, default=10)
    parser.add_argument("--maxCoeff", type=int, default=100)
    parser.add_argument("--minG", type=int, default=0)
    parser.add_argument("--maxG", type=int, default=0)
    parser.add_argument("--minBiasLinNonLin", type=int, default=5)
    parser.add_argument("--maxBiasLinNonLin", type=int, default=20)
    parser.add_argument("--N_f", type=int, default=5)
    parser.add_argument("--P", type=int, default=10)
    parser.add_argument("--g_sd", type=int, default=2)
    parser.add_argument("--SNRmin", type=int, default=10)
    parser.add_argument("--SNRmax", type=int, default=40)
    # --- RTC augmentation (utils/rtc_augment.py; pools from
    #     scripts/prepare_rtc_aug_data.py) ---
    parser.add_argument("--use_rtc_aug", action="store_true", default=False,
                        help="augment the training clips with the noisy eval conditions: "
                             "office/coffee (RNNoise), echo (CLAD), rain/footsteps/keyboard "
                             "(ESC-50) and reverb (measured RIRs). One stage per clip at most.")
    parser.add_argument("--aug_noise_dirs", nargs="*", default=["data/augm/noise"],
                        help="noise pools; each subdirectory is one scenario, tagged by its "
                             "directory name (office, coffee, rain, footsteps, keyboard)")
    parser.add_argument("--aug_music_dirs", nargs="*", default=[],
                        help="optional background-music pool (scenario tag 'music')")
    parser.add_argument("--aug_rir_dirs", nargs="*", default=["data/augm/measured_rirs"],
                        help="RIR pool for the reverb scenario (.f32 at 48 kHz, or wav)")
    parser.add_argument("--aug_p_apply", type=float, default=0.8,
                        help="probability that a training clip gets its one augmentation")
    parser.add_argument("--aug_max_stages", type=int, default=1,
                        help="kept for compatibility; clamped to 1 (one augmentation per clip)")
    parser.add_argument("--aug_use_deepfilternet", action="store_true",
                        help="add a DeepFilterNet suppression stage (needs deepfilternet)")
    parser.add_argument("--no_codec_aug", action="store_true", default=False,
                        help="with --use_rtc_aug, skip the RTC codec (QQ/Zoom/WeChat/DingTalk/"
                             "Lark/VooV/Telegram) that otherwise follows the stage in series")
    parser.add_argument("--aug_codec_p", type=float, default=1.0,
                        help="probability that a training clip then goes through one RTC codec")
    parser.add_argument("--musan", action="store_true", default=False,
                        help="add a MUSAN stage drawing from --musan_dir, scanned recursively. "
                             "Everything found there is mixed in as additive noise with no "
                             "filtering, so keep only the non-speech part in that tree. "
                             "Works with or without --use_rtc_aug and shares the same "
                             "one-augmentation-per-clip draw.")
    parser.add_argument("--musan_dir", type=str, default="data/augm/musan",
                        help="root of the (non-speech) MUSAN tree used by --musan")
    parser.add_argument("--use_rawboost", action="store_true", default=False)
    # --- level augmentation (utils/level_augment.py): random gain + AGC / compressor / limiter
    #     after the RTC augmenter, before RawBoost. Removes the level shortcut found in E0.5. ---
    parser.add_argument("--use_level_aug", action="store_true", default=False)
    parser.add_argument("--level_p_apply", type=float, default=0.8)
    parser.add_argument("--level_gain_mode", type=str, default="absolute", choices=["absolute", "relative"],
                        help="absolute: set the active level to a random target (--level_target_db), "
                             "removing the level cue; relative: add a random gain (--level_gain_db)")
    parser.add_argument("--level_target_db", type=float, nargs=2, default=[-38.0, -12.0])
    parser.add_argument("--level_gain_db", type=float, nargs=2, default=[-10.0, 10.0])
    parser.add_argument("--level_agc_p", type=float, default=0.30)
    parser.add_argument("--level_comp_p", type=float, default=0.15)
    parser.add_argument("--level_limit_p", type=float, default=0.15)
    parser.add_argument("--level_clip_p", type=float, default=0.3,
                        help="share of clips still overshooting after the dynamics stage that are "
                             "hard-clipped (the rest go through the limiter)")
    # --- loudness normalisation (utils/loudness.py): deterministic BS.1770 integrated-loudness target,
    #     applied in SpoofAudioDataset._load before every augmentation, on train AND dev. Pass the
    #     same flags to main_eval.py / scripts/local_eval.py so inference sees the same levels. ---
    parser.add_argument("--loudness_norm", action="store_true", default=False,
                        help="normalise every clip to --loudness_norm_lufs at read time (train, dev, eval)")
    parser.add_argument("--loudness_norm_lufs", type=float, default=-23.0,
                        help="target integrated loudness in LUFS (EBU R128 speech: -23; streaming: -14)")
    # --- local proxy evaluation (utils/local_eval.py; sets from scripts/build_dev_noisy_sim.py).
    #     Read-only: scored in eval mode after each epoch, never used for the loss or the LR. ---
    parser.add_argument("--local_eval_sets", type=str, default=None,
                        help="configs/local_eval_sets.yaml; enables per-epoch local WF1 and the "
                             "best_by_{wf1,noisy_eer,dev_loss}.pth symlinks (default: off)")
    parser.add_argument("--local_eval_names", nargs="+",
                        default=["dev_online_clean", "sim_matched_mini", "sim_heldout_mini"])
    parser.add_argument("--local_eval_clean_n", type=int, default=2000,
                        help="fixed seeded subsample of the role=clean set")
    parser.add_argument("--local_eval_every", type=int, default=1)
    parser.add_argument("--earlystop_metric", type=str, default="dev_loss", choices=["dev_loss", "wf1"],
                        help="wf1 needs --local_eval_sets")
    parser.add_argument("--keep_topk_by_wf1", type=int, default=0,
                        help="delete epoch checkpoints outside the top-K by local WF1 (and not in "
                             "--keep_epochs or a best_by_* target); 0 keeps all. Needs --local_eval_sets")
    parser.add_argument("--keep_epochs", type=int, nargs="*", default=[],
                        help="epoch checkpoints never pruned by --keep_topk_by_wf1")
    parser.add_argument("--resume", type=str, default=None,
                        help="exp/<run> dir to continue from its last_state.pt; pass the same "
                             "arguments as the original run (config.yaml: command)")

    args = parser.parse_args()
    args.ssl_name = ssl_path(args.ssl_name)     # the dataset and the model must agree on the repo id

    set_random_seed(args.seed, args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for path in [args.train_data_path, args.dev_data_path, args.train_protocol, args.dev_protocol]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    if args.keep_topk_by_wf1 and not args.local_eval_sets:
        raise ValueError("--keep_topk_by_wf1 needs --local_eval_sets")
    if args.resume:
        exp_root = os.path.normpath(args.resume)
        if not os.path.isfile(os.path.join(exp_root, LAST_STATE)):
            raise FileNotFoundError(f"--resume: no {LAST_STATE} in {exp_root}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        exp_name = f"{args.track}_epoch{args.num_epochs}_bs{args.batch_size}_{timestamp}"
        exp_root = os.path.join(args.out_path, exp_name)
    ckpt_dir = os.path.join(exp_root, "ckpt")
    log_dir = os.path.join(exp_root, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    train_set, train_files, _ = build_dataset_from_protocol(
        args.train_protocol, args.train_data_path, mode="train", args=args, algo=args.algo
    )
    dev_set, dev_files, _ = build_dataset_from_protocol(
        args.dev_protocol, args.dev_data_path, mode="dev", args=args
    )
    train_loader = build_loader(train_set, args.batch_size, args.num_workers, shuffle=True)
    dev_loader = build_loader(dev_set, args.batch_size, args.num_workers, shuffle=False)


    model = ModelSLS(args, device)
    in_device = model.input_device
    out_device = model.output_device
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        print(f"Model loaded: {args.model_path}")

    if model.ssl_model.devices is not None:
        print(f"Encoder split: "
              f"{dict(zip([str(d) for d in model.ssl_model.devices], model.ssl_model.layer_counts))}")
    print(f"Device: in={in_device} out={out_device}")
    print(f"SSL checkpoint: {model.ssl_model.name}")
    print(f"Layers: {model.ssl_model.n_layers}")
    if train_set.augmenter is not None:
        print(f"RTC augmentation: {train_set.augmenter.describe()}")
    else:
        print("RTC augmentation: off")
    print(f"Loudness norm: {train_set.loudness.describe() if train_set.loudness is not None else 'off'}")
    print(f"RawBoost: {'algo ' + str(args.algo) if args.use_rawboost else 'off'}")
    print(f"Train trials: {len(train_files)}")
    print(f"Dev trials: {len(dev_files)}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable)} (freeze_ssl={args.freeze_ssl})")
    if not args.resume:                         # keep the original run's config.yaml
        write_run_config(exp_root, args, train_set.augmenter, train_set.loudness)

    local_sets = None
    if args.local_eval_sets:
        from utils.local_eval import load_sets
        local_sets = load_sets(args.local_eval_sets, args.local_eval_names, ssl_name=args.ssl_name,
                               clean_n=args.local_eval_clean_n, seed=args.seed, loudness=train_set.loudness)
        print("Local eval: " + ", ".join(f"{n}({len(s['file_list'])})" for n, s in local_sets.items()))
    elif args.earlystop_metric == "wf1":
        raise ValueError("--earlystop_metric wf1 needs --local_eval_sets")
    best_wf1, best_noisy_eer, wf1_no_improve = -1.0, float("inf"), 0

    optimizer = build_optimizer(model, trainable, args)
    ce_weight = torch.FloatTensor(args.ce_weights).to(out_device)
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(weight=ce_weight, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=ce_weight)
    print(f"CE weights: spoof={args.ce_weights[0]} bonafide={args.ce_weights[1]}"
          + (f" label_smoothing={args.label_smoothing}" if args.label_smoothing > 0 else ""))

    scheduler = None
    step_scheduler = None                       # stepped per batch inside train_epoch
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.lr_min)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.lr_min)
    elif args.lr_scheduler == "warmup_cosine":
        total_steps = args.num_epochs * len(train_loader)
        warmup_steps = int(round(args.warmup_frac * total_steps))
        scheduler = step_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, warmup_cosine_lambda(total_steps, warmup_steps, args.lr_min / args.lr))
        print(f"Warmup-cosine: {warmup_steps} warmup / {total_steps} steps, floor {args.lr_min:.1e}")
    print(f"LR: {args.lr} schedule={args.lr_scheduler}")
    if args.optim != "adam" or args.head_lr_mult != 1.0:
        print(f"Optimizer: {args.optim} weight_decay={args.weight_decay} "
              f"head_lr={args.lr * args.head_lr_mult:.1e} (x{args.head_lr_mult:g})")
    if args.grad_clip > 0:
        print(f"Grad clip: {args.grad_clip}")

    ema = None
    if args.ema_decay > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=True)
        print(f"EMA: decay={args.ema_decay} (EMA weights are evaluated and saved per epoch)")
    eval_model = ema.module if ema is not None else model

    writer = SummaryWriter(log_dir=log_dir) if SummaryWriter else None

    best_dev_loss = float("inf")
    #best_model_path = os.path.join(ckpt_dir, "best_model.pth")
    best_epoch = None
    no_improve_count = 0
    start_epoch = 1

    if args.resume:
        state = torch.load(os.path.join(exp_root, LAST_STATE), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
        if ema is not None:
            if state.get("ema") is not None:
                ema.load_state_dict(state["ema"])
            else:                               # run started without EMA: restart it from here
                print("[resume] no EMA state in last_state.pt; EMA restarts from the current weights")
        c = state["counters"]
        best_dev_loss, best_epoch, no_improve_count = c["best_dev_loss"], c["best_epoch"], c["no_improve_count"]
        best_wf1, best_noisy_eer, wf1_no_improve = c["best_wf1"], c["best_noisy_eer"], c["wf1_no_improve"]
        restore_run_dir(exp_root, ckpt_dir, state, log_path)
        rng = state["rng"]
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception as exc:                # different GPU count: continuity, not bitwise
                print(f"[resume] CUDA RNG not restored: {exc}")
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        start_epoch = state["epoch"] + 1
        del state
        log_line(log_path, f"Resumed {exp_root} after epoch {start_epoch - 1}: {' '.join(sys.argv)}")

    for epoch in range(start_epoch, args.num_epochs + 1):
        train_loss, train_acc = train_epoch(train_loader, model, optimizer, in_device,
                                            criterion, out_device, step_scheduler=step_scheduler,
                                            grad_clip=args.grad_clip, ema=ema)
        dev_loss, dev_acc = evaluate_dev(dev_loader, eval_model, in_device, criterion, out_device)

        # read before stepping the scheduler: this is the LR the epoch trained with
        # (warmup_cosine steps per batch, so this is the LR of its last step)
        current_lr = optimizer.param_groups[0]["lr"]
        message = (
            f"Epoch {epoch}/{args.num_epochs} "
            f"TrainLoss={train_loss:.6f} TrainAcc={train_acc:.2f}% "
            f"DevLoss={dev_loss:.6f} DevAcc={dev_acc:.2f}% "
            f"LR={current_lr:.3e}"
        )
        if args.head_lr_mult != 1.0:
            message += f" HeadLR={group_lrs(optimizer).get('head', current_lr):.3e}"
        print(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().replace(microsecond=0)}] {message}\n")

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Loss/dev", dev_loss, epoch)
            writer.add_scalar("Acc/dev", dev_acc, epoch)
            writer.add_scalar("LR", current_lr, epoch)

        

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            no_improve_count = 0
            # best_model.pth is a relative symlink to the epoch file, so the whole
            # ckpt dir stays movable and the best weights cost no extra disk.
            #if os.path.lexists(best_model_path):
                #os.unlink(best_model_path)
           # os.symlink(os.path.basename(epoch_path), best_model_path)
            #print(f"Saved best model: {best_model_path} -> {os.path.basename(epoch_path)} "
                  #f"(epoch {epoch}, dev_loss={dev_loss:.6f})")
            epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch}_dev_loss_{dev_loss:.6f}.pth")
            torch.save(eval_model.state_dict(), epoch_path)
        else:
            no_improve_count += 1
            epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch}_dev_loss_{dev_loss:.6f}.pth")
            torch.save(eval_model.state_dict(), epoch_path)

        if local_sets is not None and epoch % args.local_eval_every == 0:
            from utils.local_eval import quick_eval
            result = quick_eval(eval_model, local_sets, in_device, args.batch_size, min(args.num_workers, 4))
            w = result["wf1"]
            if best_epoch == epoch:
                link_best(ckpt_dir, "best_by_dev_loss.pth", epoch_path)
            if w["wf1"] > best_wf1:
                best_wf1, wf1_no_improve = w["wf1"], 0
                link_best(ckpt_dir, "best_by_wf1.pth", epoch_path)
            else:
                wf1_no_improve += 1
            if w["eer_noisy"] < best_noisy_eer:
                best_noisy_eer = w["eer_noisy"]
                link_best(ckpt_dir, "best_by_noisy_eer.pth", epoch_path)
            local_msg = (f"Epoch {epoch} LocalWF1={100 * w['wf1']:.2f} F1clean={100 * w['f1_clean']:.2f} "
                         f"F1noisy={100 * w['f1_noisy']:.2f} WF1oracle={100 * w['wf1_oracle']:.2f} "
                         f"EERnoisy={100 * w['eer_noisy']:.2f}")
            print(local_msg)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().replace(microsecond=0)}] {local_msg}\n")
            with open(os.path.join(exp_root, "metrics.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
                                    "ckpt": os.path.basename(epoch_path), **result}) + "\n")
            if args.keep_topk_by_wf1:
                prune_checkpoints(exp_root, ckpt_dir, args.keep_topk_by_wf1, args.keep_epochs, log_path)
            if writer:
                writer.add_scalar("Local/wf1", w["wf1"], epoch)
                writer.add_scalar("Local/f1_noisy", w["f1_noisy"], epoch)
                writer.add_scalar("Local/eer_noisy", w["eer_noisy"], epoch)
                for name, m in result["sets"].items():
                    writer.add_scalar(f"Local/{name}/f1", m["f1"], epoch)
                    writer.add_scalar(f"Local/{name}/eer", m["eer"], epoch)


        if scheduler is not None and step_scheduler is None:
            if args.lr_scheduler == "plateau":
                scheduler.step(dev_loss)
            else:
                scheduler.step()

        save_last_state(exp_root, ckpt_dir, model, optimizer, scheduler, epoch, {
            "best_dev_loss": best_dev_loss, "best_epoch": best_epoch, "no_improve_count": no_improve_count,
            "best_wf1": best_wf1, "best_noisy_eer": best_noisy_eer, "wf1_no_improve": wf1_no_improve,
        }, ema=ema)

        stop_count = wf1_no_improve if args.earlystop_metric == "wf1" else no_improve_count
        if stop_count >= args.earlystop_epoch:
            print(f"Early stopping at epoch {epoch}. Best dev_loss={best_dev_loss:.6f}")
            break

    if writer:
        writer.close()
    print(f"Experiment saved to: {exp_root}")
    #print(f"Best model: {best_model_path} (epoch {best_epoch}, dev_loss={best_dev_loss:.6f})")


if __name__ == "__main__":
    main()
