import argparse
import os
from datetime import datetime

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

#from model.model import Model
from model.sls_model import ModelSLS, ssl_path
from utils.data_utils import build_dataset_from_protocol, set_random_seed
from utils.rtc_augment_expert import EXPERT_CODECS, EXPERT_STAGES, ExpertAugConfig, ExpertAugmenter

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


def train_epoch(data_loader, model, optimizer, device, criterion, out_device=None):
    model.train()
    running_loss = 0.0
    correct = 0
    num_total = 0
    out_device = out_device or device

    for batch_x, batch_y, _ in tqdm(data_loader, desc="Training", unit="batch", ascii=True):
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).long().to(out_device)
        batch_size = batch_x.size(0)

        optimizer.zero_grad()
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        batch_loss.backward()
        optimizer.step()

        running_loss += batch_loss.item() * batch_size
        correct += (torch.argmax(batch_out, dim=1) == batch_y).sum().item()
        num_total += batch_size

    return running_loss / num_total, 100.0 * correct / num_total


def evaluate_dev(data_loader, model, device, criterion, out_device=None):
    """Returns (loss, macro-F1 in %) at the argmax decision, i.e. a 0.5 threshold."""
    model.eval()
    running_loss = 0.0
    num_total = 0
    labels, preds = [], []
    out_device = out_device or device

    with torch.no_grad():
        for batch_x, batch_y, _ in tqdm(data_loader, desc="Validating", unit="batch", ascii=True):
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).long().to(out_device)
            batch_size = batch_x.size(0)

            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)

            running_loss += batch_loss.item() * batch_size
            num_total += batch_size
            labels.append(batch_y.cpu())
            preds.append(torch.argmax(batch_out, dim=1).cpu())

    macro_f1 = f1_score(torch.cat(labels).numpy(), torch.cat(preds).numpy(),
                        average="macro", labels=[0, 1], zero_division=0)
    return running_loss / num_total, 100.0 * macro_f1



def main():
    parser = argparse.ArgumentParser(description="Train an XLSR-SLS spoof detector expert on one RTC noise condition")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--dev_data_path", type=str, nargs="+", required=True,
                        help="one base dir per --dev_protocol, same order")
    parser.add_argument("--train_protocol", type=str, required=True)
    parser.add_argument("--dev_protocol", type=str, nargs="+", required=True,
                        help="one or more dev sets; the FIRST is the clean set and the rest are "
                             "noisy, e.g. dev_label_excl_halfB.txt (clean) + the sim matched_v1 / "
                             "heldout_v1 protocol_mini.txt (noisy, OOD). Each is scored on its own; "
                             "selection and early stop use the weighted macro-F1 (see --wf1_lambda)")
    parser.add_argument("--wf1_lambda", type=float, default=0.3,
                        help="dev WF1 = lambda * MacroF1(clean) + (1 - lambda) * MacroF1(noisy), "
                             "MacroF1(noisy) being the mean over the noisy dev sets. With a single "
                             "dev set, WF1 is that set's macro-F1")
    parser.add_argument("--track", type=str, default="QWEN_expert")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--devices", type=int, nargs="+", default=None,
                        help="GPU ids to split the XLS-R encoder over, e.g. --devices 4 5. "
                             "Needed to fit all 48 layers; overrides --device. "
                             "Do NOT combine with CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--n_layers", type=int, default=48,
                        help="number of SSL transformer layers to keep, clamped to the backbone's depth "
                             "(48 for XLS-R 2B, 24 for W2V-BERT 2.0, 18 for Qwen3-ASR-0.6B)")
    parser.add_argument("--ssl_name", type=str, default="Qwen/Qwen3-ASR-1.7B",
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
                        choices=["none", "cosine", "plateau"],
                        help="optional LR schedule; 'none' keeps the constant LR (default). "
                             "cosine = CosineAnnealingLR over --num_epochs; "
                             "plateau = ReduceLROnPlateau on dev WF1")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="[plateau] LR multiplier on each reduction")
    parser.add_argument("--lr_patience", type=int, default=8,
                        help="[plateau] stale dev-WF1 epochs before the LR is reduced; "
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
    # --- expert augmentation (utils/rtc_augment_expert.py) ---
    parser.add_argument("--expert_noise", type=str, required=True, choices=list(EXPERT_STAGES),
                        help="the one condition this expert is trained on. A clip gets this "
                             "condition with probability --aug_p_noise and, independently, one RTC "
                             "codec with probability --aug_codec_p (after the condition if both).")
    parser.add_argument("--aug_noise_dirs", nargs="*", default=["data/augm/noise"],
                        help="noise roots; the pool used is the <expert_noise> subdirectory "
                             "(office, coffee, rain, footsteps, keyboard)")
    parser.add_argument("--aug_rir_dirs", nargs="*", default=["data/augm/measured_rirs"],
                        help="RIR pool for --expert_noise reverb (.f32 at 48 kHz, or wav)")
    parser.add_argument("--musan_dir", type=str, default="data/augm/musan",
                        help="MUSAN root for --expert_noise musan_noise / musan_music "
                             "(its noise/ or music/ subtree is scanned recursively)")
    parser.add_argument("--aug_p_noise", "--aug_p_apply", dest="aug_p_noise", type=float, default=0.5,
                        help="probability that a training clip gets the expert's condition")
    parser.add_argument("--aug_codec_p", type=float, default=0.5,
                        help="probability that a training clip goes through an RTC codec, drawn "
                             "independently of the condition (clean clips can get one too)")
    parser.add_argument("--aug_codec_apps", nargs="+", default=list(EXPERT_CODECS),
                        choices=list(EXPERT_CODECS),
                        help="codecs drawn from, uniformly: the RTC apps' Opus profiles and the "
                             "telephony codecs (default: all of them)")
    parser.add_argument("--use_rawboost", action="store_true", default=False)
    args = parser.parse_args()
    args.ssl_name = ssl_path(args.ssl_name)     # the dataset and the model must agree on the repo id
    # build_dataset_from_protocol would build the general RTC augmenter from these;
    # keep it off, the expert augmenter is attached below instead
    args.use_rtc_aug = False
    args.musan = False

    set_random_seed(args.seed, args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if len(args.dev_data_path) != len(args.dev_protocol):
        raise ValueError("--dev_data_path and --dev_protocol need the same number of entries")
    for path in [args.train_data_path, args.train_protocol, *args.dev_data_path, *args.dev_protocol]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    exp_name = f"{args.track}_expert_{args.expert_noise}_epoch{args.num_epochs}_bs{args.batch_size}_{timestamp}"
    exp_root = os.path.join(args.out_path, exp_name)
    ckpt_dir = os.path.join(exp_root, "ckpt")
    log_dir = os.path.join(exp_root, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    train_set, train_files, _ = build_dataset_from_protocol(
        args.train_protocol, args.train_data_path, mode="train", args=args, algo=args.algo
    )
    # dev set name = protocol file stem, prefixed by its parent dir when the stem is generic
    dev_loaders = {}
    for data_path, protocol in zip(args.dev_data_path, args.dev_protocol):
        stem = os.path.splitext(os.path.basename(protocol))[0]
        name = stem if stem.startswith("dev") else f"{os.path.basename(os.path.dirname(protocol))}_{stem}"
        dev_set, dev_files, _ = build_dataset_from_protocol(protocol, data_path, mode="dev", args=args)
        dev_loaders[name] = build_loader(dev_set, args.batch_size, args.num_workers, shuffle=False)
        print(f"Dev set {name}: {len(dev_files)} trials")
    train_set.augmenter = ExpertAugmenter(ExpertAugConfig(
        stage=args.expert_noise,
        noise_dirs=args.aug_noise_dirs,
        rir_dirs=args.aug_rir_dirs,
        musan_dir=args.musan_dir,
        p_noise=args.aug_p_noise,
        p_codec=args.aug_codec_p,
        seed=args.seed,
        codec_apps=args.aug_codec_apps,
    ))
    train_loader = build_loader(train_set, args.batch_size, args.num_workers, shuffle=True)


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
    print(f"Expert augmentation: {train_set.augmenter.describe()}")
    print(f"RawBoost: {'algo ' + str(args.algo) if args.use_rawboost else 'off'}")
    print(f"Train trials: {len(train_files)}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable)} (freeze_ssl={args.freeze_ssl})")

    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(args.ce_weights).to(out_device))
    print(f"CE weights: spoof={args.ce_weights[0]} bonafide={args.ce_weights[1]}")

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.lr_min)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.lr_min)
    print(f"LR: {args.lr} schedule={args.lr_scheduler}")

    writer = SummaryWriter(log_dir=log_dir) if SummaryWriter else None

    best_dev_wf1 = float("-inf")
    #best_model_path = os.path.join(ckpt_dir, "best_model.pth")
    best_epoch = None
    no_improve_count = 0

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_epoch(train_loader, model, optimizer, in_device,
                                            criterion, out_device)
        dev_results = {name: evaluate_dev(loader, model, in_device, criterion, out_device)
                       for name, loader in dev_loaders.items()}
        dev_loss = sum(loss for loss, _ in dev_results.values()) / len(dev_results)
        f1s = [f1 for _, f1 in dev_results.values()]
        f1_clean = f1s[0]
        f1_noisy = sum(f1s[1:]) / len(f1s[1:]) if len(f1s) > 1 else f1_clean
        dev_wf1 = args.wf1_lambda * f1_clean + (1 - args.wf1_lambda) * f1_noisy

        # read before stepping the scheduler: this is the LR the epoch trained with
        current_lr = optimizer.param_groups[0]["lr"]
        message = (
            f"Epoch {epoch}/{args.num_epochs} "
            f"TrainLoss={train_loss:.6f} TrainAcc={train_acc:.2f}% "
            f"DevLoss={dev_loss:.6f} DevWF1={dev_wf1:.2f}% "
            f"(clean={f1_clean:.2f}% noisy={f1_noisy:.2f}%) "
            f"LR={current_lr:.3e}"
        )
        if len(dev_results) > 1:
            message += " | " + " ".join(f"{name}: loss={loss:.4f} macroF1={f1:.2f}%"
                                        for name, (loss, f1) in dev_results.items())
        print(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().replace(microsecond=0)}] {message}\n")

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Loss/dev", dev_loss, epoch)
            writer.add_scalar("WF1/dev", dev_wf1, epoch)
            writer.add_scalar("MacroF1/dev_clean", f1_clean, epoch)
            writer.add_scalar("MacroF1/dev_noisy", f1_noisy, epoch)
            writer.add_scalar("LR", current_lr, epoch)
            for name, (loss, f1) in dev_results.items():
                writer.add_scalar(f"Loss/dev_{name}", loss, epoch)
                writer.add_scalar(f"MacroF1/dev_{name}", f1, epoch)

        

        epoch_path = os.path.join(
            ckpt_dir, f"epoch_{epoch}_dev_wf1_{dev_wf1:.4f}_dev_loss_{dev_loss:.6f}.pth")
        if dev_wf1 > best_dev_wf1:
            best_dev_wf1 = dev_wf1
            best_epoch = epoch
            no_improve_count = 0
            # best_model.pth is a relative symlink to the epoch file, so the whole
            # ckpt dir stays movable and the best weights cost no extra disk.
            #if os.path.lexists(best_model_path):
                #os.unlink(best_model_path)
           # os.symlink(os.path.basename(epoch_path), best_model_path)
            #print(f"Saved best model: {best_model_path} -> {os.path.basename(epoch_path)} "
                  #f"(epoch {epoch}, dev_loss={dev_loss:.6f})")
        else:
            no_improve_count += 1
        torch.save(model.state_dict(), epoch_path)


        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(dev_wf1)
            else:
                scheduler.step()

        if no_improve_count >= args.earlystop_epoch:
            print(f"Early stopping at epoch {epoch}. Best dev WF1={best_dev_wf1:.2f}% "
                  f"(epoch {best_epoch})")
            break

    if writer:
        writer.close()
    print(f"Experiment saved to: {exp_root}")
    #print(f"Best model: {best_model_path} (epoch {best_epoch}, dev WF1={best_dev_wf1:.2f}%)")


if __name__ == "__main__":
    main()
