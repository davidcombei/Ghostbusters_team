import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

FEATS_DIR = "data/features/xls-r-2b"
META_DIR = "data/data/meta"
LAMBDA = 0.3
C = 1e6
MAX_ITER = 10000


def load_dataset(metadata, feats):
    with open(os.path.join(META_DIR, metadata)) as fin:
        lines = [line.strip().split() for line in fin if line.strip()]
    Y = np.array([0 if line[1] == "bonafide" else 1 for line in lines])
    conds = np.array(
        ["noisy" if line[0].split("/")[0] == "noisy" else "clean" for line in lines]
    )
    X = np.load(os.path.join(FEATS_DIR, feats))

    return X, Y, conds


def weighted_macro_f1(Y, Y_hat, conds):
    is_noisy = conds == "noisy"
    clean = f1_score(Y[~is_noisy], Y_hat[~is_noisy], average="macro")
    if not is_noisy.any():
        return clean, clean, float("nan")
    noisy = f1_score(Y[is_noisy], Y_hat[is_noisy], average="macro")
    return LAMBDA * clean + (1 - LAMBDA) * noisy, clean, noisy


results = []

for layer in range(49):
    print(f" Using layer {layer}...")

    X_train, Y_train, _ = load_dataset(
        "train_label.txt", f"xls-r-2b_Layer{layer}_RTCSDD_train.npy"
    )
    X_dev, Y_dev, conds_dev = load_dataset(
        "dev_label.txt", f"xls-r-2b_Layer{layer}_RTCSDD_dev.npy"
    )

    model = LogisticRegression(random_state=42, C=C, max_iter=MAX_ITER, verbose=False)
    model.fit(X_train, Y_train)
    # dump(model, f"logreg_RTCSDD_Layer{layer}.joblib")

    Y_hat = model.predict(X_dev)
    weighted, clean, noisy = weighted_macro_f1(Y_dev, Y_hat, conds_dev)
    results.append((layer, weighted, clean, noisy))
    print(
        f"[layer {layer:<2}] weighted macro-F1: {weighted * 100:.1f} "
        f"(clean: {clean * 100:.1f}, noisy: {noisy * 100:.1f})"
    )

print("\n=== all layers ===")
for layer, weighted, clean, noisy in results:
    print(
        f"[layer {layer:<2}] weighted macro-F1: {weighted * 100:.1f} "
        f"(clean: {clean * 100:.1f}, noisy: {noisy * 100:.1f})"
    )
best = max(results, key=lambda r: r[1])
print(f"\nbest layer: {best[0]} with weighted macro-F1 {best[1] * 100:.1f}")
