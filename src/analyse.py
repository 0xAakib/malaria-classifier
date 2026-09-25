"""Label sufficiency point per (method, dataset), from results/eval.csv.

The sufficiency point is the smallest label fraction whose seed-mean balanced
accuracy reaches at least 95% of the same method's seed-mean at 100% labels.
"""
import pandas as pd

THRESHOLD = 0.95


def main():
    results = pd.read_csv("results/eval.csv")
    mean = results.groupby(["method", "dataset", "fraction"])["balanced_acc"].mean().reset_index()
    rows = []
    for (method, dataset), group in mean.groupby(["method", "dataset"], sort=False):
        full = group.loc[group["fraction"] == 1.0, "balanced_acc"].item()
        reached = group.loc[group["balanced_acc"] >= THRESHOLD * full, "fraction"]
        rows.append({"method": method, "dataset": dataset, "sufficiency_fraction": reached.min()})
    pd.DataFrame(rows).to_csv("results/sufficiency.csv", index=False)


if __name__ == "__main__":
    main()
