"""
Average the checkpoints of a merge list (e.g. scripts/select_topk.py output) and save the result as
one plain state_dict, usable anywhere a single checkpoint goes (--model_path).

    python scripts/merge_ckpts.py --merge_list SLS_setup/exp/<run>/top5_by_wf1/merge_list.txt
    python scripts/merge_ckpts.py --merge_list <list> --out exp/<run>/merged_top5.pth [--force]

Uses the same merge (main_eval.merge_checkpoints, uniform mean) and path resolution
(local_eval.read_merge_list: cwd, then SLS_setup/) as local_eval.py --merge_list, so the saved
weights equal the in-memory merge behind an existing merge_top<k> report. Writes <out> (default:
merged.pth next to the list) and <out stem>.sources.txt (resolved source paths, itself a merge list).
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SLS_DIR = REPO_ROOT / "SLS_setup"
sys.path.insert(0, str(SLS_DIR))

import torch  # noqa: E402

from local_eval import read_merge_list  # noqa: E402
from main_eval import merge_checkpoints  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Average the checkpoints of a merge list into one .pth")
    parser.add_argument("--merge_list", type=str, required=True, help="text file, one checkpoint path per line")
    parser.add_argument("--out", type=str, default=None,
                        help="merged checkpoint path (default: <merge_list dir>/merged.pth)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path(args.merge_list).parent / "merged.pth"
    if out.exists() and not args.force:
        sys.exit(f"{out} exists (use --force to overwrite)")
    try:
        paths = read_merge_list(args.merge_list)
    except FileNotFoundError as e:
        sys.exit(f"checkpoint in {args.merge_list} not found: {e}")
    if not paths:
        sys.exit(f"no checkpoints in {args.merge_list}")

    state = merge_checkpoints(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, out)
    sources = out.with_name(out.stem + ".sources.txt")
    sources.write_text("".join(os.path.abspath(p) + "\n" for p in paths), encoding="utf-8")

    n_params = sum(v.numel() for v in state.values() if v.is_floating_point())
    print(f"\nMerged {len(paths)} checkpoints: {len(state)} tensors, {n_params / 1e6:.1f}M float params, "
          f"{out.stat().st_size / 2**30:.2f} GiB")
    print(f"Merged checkpoint: {out}")
    print(f"Sources: {sources}")


if __name__ == "__main__":
    main()
