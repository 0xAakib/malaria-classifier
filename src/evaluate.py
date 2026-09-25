"""Evaluate every trained checkpoint on the full, frozen test split.

The test split is the same for every run and is never subsampled. Scores are
balanced accuracy (the mean of per-class recall) and MCC.
"""
import csv
import itertools

import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

from models import build_model
from data import test_loader


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, labels = [], []
    for images, y in loader:
        preds.extend(model(images).argmax(dim=1).tolist())
        labels.extend(y.tolist())
    return labels, preds


def main():
    cfg = yaml.safe_load(open("configs/train.yaml"))
    print("evaluate.py: test split is the full frozen test split; no subsampling")
    rows = []
    for method, dataset, fraction, seed in itertools.product(
        cfg["methods"], cfg["datasets"], cfg["fractions"], cfg["seeds"]
    ):
        model = build_model(method, num_classes=2)
        model.load_state_dict(torch.load(f"checkpoints/{method}_{dataset}_{fraction}_{seed}.pt"))
        loader = test_loader(dataset)
        labels, preds = predict(model, loader)
        ba = balanced_accuracy_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)
        print(
            f"[eval] method={method} dataset={dataset} fraction={fraction} seed={seed} "
            f"n_test={len(labels)} balanced_acc={ba:.3f} mcc={mcc:.3f}"
        )
        rows.append([method, dataset, fraction, seed, round(ba, 3), round(mcc, 3)])

    with open("results/eval.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "dataset", "fraction", "seed", "balanced_acc", "mcc"])
        writer.writerows(rows)
    print(f"[eval] wrote results/eval.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
