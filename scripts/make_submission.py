"""
Validate a progress/eval scores file and pack it for Codabench.

    python scripts/make_submission.py SLS_setup/exp/<run>/model_merging_top5/scores.txt
    python scripts/make_submission.py <scores.txt> --protocol /mnt/paml-research/RTCFake/data/progress.txt

scores.txt is what SLS_setup/main_eval.py writes: one "<utt_id> <P(spoof)>" line per utterance.
Checks, against the protocol: same number of lines, the same utterance ids (no missing, extra or
duplicate ids), and every score a finite number in [0, 1]. Then writes submission.zip next to
scores.txt (or --zip) with scores.txt at the archive root, and re-reads the archive to confirm.

Only the format is checked. Score distributions on progress are not reported on purpose: the
rules forbid analysing progress outputs to drive decisions (docs/experiments_roadmap.md §2).
"""

import argparse
import math
import sys
import zipfile
from pathlib import Path

DEFAULT_PROTOCOL = "/mnt/paml-research/RTCFake/data/progress.txt"


def read_ids(protocol):
    with open(protocol, encoding="utf-8") as f:
        return [line.split()[0] for line in f if line.strip()]


def check(scores_path, protocol):
    errors = []
    expected = read_ids(protocol)
    seen = {}
    with open(scores_path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 2:
                errors.append(f"line {n}: expected '<utt_id> <score>', got {line.strip()[:80]!r}")
                continue
            utt, raw = parts
            try:
                score = float(raw)
            except ValueError:
                errors.append(f"line {n}: score {raw!r} is not a number")
                continue
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                errors.append(f"line {n}: score {raw} outside [0, 1]")
            if utt in seen:
                errors.append(f"line {n}: duplicate id {utt} (first on line {seen[utt]})")
            seen[utt] = n
    expected_set = set(expected)
    missing = [u for u in expected if u not in seen]
    extra = [u for u in seen if u not in expected_set]
    if missing:
        errors.append(f"{len(missing)} protocol ids have no score, e.g. {missing[:3]}")
    if extra:
        errors.append(f"{len(extra)} scored ids are not in the protocol, e.g. {extra[:3]}")
    return len(expected), len(seen), errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--zip", default=None, help="default: submission.zip next to the scores file")
    args = parser.parse_args()

    scores = Path(args.scores)
    if not scores.is_file():
        sys.exit(f"no scores file: {scores}")
    n_expected, n_scored, errors = check(scores, args.protocol)
    print(f"protocol: {args.protocol} ({n_expected} ids)")
    print(f"scores:   {scores} ({n_scored} ids)")
    if errors:
        print("\nNOT VALID, no archive written:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        sys.exit(1)

    out = Path(args.zip) if args.zip else scores.parent / "submission.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(scores, arcname="scores.txt")
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        if names != ["scores.txt"] or zf.read("scores.txt") != scores.read_bytes():
            sys.exit(f"archive check failed: {out} holds {names}")
    print(f"\nvalid: all {n_expected} ids scored once, scores in [0, 1]")
    print(f"wrote {out} (scores.txt at the root) -- upload it to Codabench")


if __name__ == "__main__":
    main()
