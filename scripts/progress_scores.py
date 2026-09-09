import os
import zipfile

import numpy as np
from sklearn.linear_model import LogisticRegression

FEATS_DIR = "data/features/xls-r-2b"
META_DIR = "data/data/meta"
OUT_DIR = "submission"
LAYER = 9
C = 1e6
MAX_ITER = 10000


def load_labeled(metadata, feats):
    with open(os.path.join(META_DIR, metadata)) as fin:
        lines = [line.strip().split() for line in fin if line.strip()]
    Y = np.array([0 if line[1] == "bonafide" else 1 for line in lines])
    X = np.load(os.path.join(FEATS_DIR, feats))
    assert len(X) == len(Y), f"{feats}: {len(X)} feats vs {len(Y)} metadata lines"

    return X, Y


def load_unlabeled(metadata, feats):
    with open(os.path.join(META_DIR, metadata)) as fin:
        filenames = [line.strip().split()[0] for line in fin if line.strip()]
    X = np.load(os.path.join(FEATS_DIR, feats))
    assert len(X) == len(filenames), f"{feats}: {len(X)} feats vs {len(filenames)} ids"

    return X, filenames


print(f" Using layer {LAYER}...")

X_train, Y_train = load_labeled(
    "train_label.txt", f"xls-r-2b_Layer{LAYER}_RTCSDD_train.npy"
)
#X_dev, Y_dev = load_labeled("dev_label.txt", f"xls-r-2b_Layer{LAYER}_RTCSDD_dev.npy")
#X = np.concatenate([X_train, X_dev])
#Y = np.concatenate([Y_train, Y_dev])
print(f" train+dev: {len(Y_train)} ({(Y_train == 0).sum()} bonafide / {(Y_train  == 1).sum()} spoof)")

model = LogisticRegression(random_state=42, C=C, max_iter=MAX_ITER, verbose=False)
model.fit(X_train, Y_train)
# dump(model, f"logreg_RTCSDD_traindev_Layer{LAYER}.joblib")

X_prog, filenames = load_unlabeled(
    "progress.txt", f"xls-r-2b_Layer{LAYER}_RTCSDD_progress.npy"
)
Y_hat = model.predict_proba(X_prog)[:, 1]
print(
    f" progress: {len(Y_hat)} scores, "
    f"min {Y_hat.min():.4f} / mean {Y_hat.mean():.4f} / max {Y_hat.max():.4f}"
)

os.makedirs(OUT_DIR, exist_ok=True)
scores_path = os.path.join(OUT_DIR, "scores.txt")
with open(scores_path, "w") as fout:
    for filename, score in zip(filenames, Y_hat):
        fout.write(f"{filename} {score:.6f}\n")

## scores.txt has to sit at the root of the archive
zip_path = os.path.join(OUT_DIR, "submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(scores_path, arcname="scores.txt")

print(f" wrote {scores_path} and {zip_path}")
