"""Stratified label-fraction subsets of each TRAIN split, per seed.

The test split is never touched here. Each subset keeps the train split's
class ratio.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

FRACTIONS = [1.0, 0.25, 0.1, 0.02]
SEEDS = [0, 1]


def subset(train: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction == 1.0:
        return train
    kept, _ = train_test_split(train, train_size=fraction, stratify=train["label"], random_state=seed)
    return kept


def main():
    rows = []
    for dataset in ("nih", "bbbc041"):
        train = pd.read_csv(f"data/splits/{dataset}_train.csv")
        for fraction in FRACTIONS:
            for seed in SEEDS:
                part = subset(train, fraction, seed)
                part.to_csv(f"data/subsets/{dataset}_{fraction}_{seed}.csv", index=False)
                for label, count in part["label"].value_counts().items():
                    rows.append({"dataset": dataset, "fraction": fraction, "seed": seed, "label": label, "count": count})
    pd.DataFrame(rows).to_csv("data/subset_counts.csv", index=False)


if __name__ == "__main__":
    main()
