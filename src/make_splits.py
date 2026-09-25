"""Frozen 80/20 train/test splits for both datasets.

The split is grouped by the source image (slide), so no slide contributes cells
to both train and test. BBBC041's six cell classes are collapsed to binary.
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 42

# BBBC041 cell classes -> binary label
LABEL_MAP = {
    "gametocyte": "infected",
    "ring": "infected",
    "trophozoite": "infected",
    "schizont": "infected",
    "red blood cell": "uninfected",
    "leukocyte": "uninfected",
}


def load_cells(dataset: str) -> pd.DataFrame:
    """One row per cell crop: cell_id, source_image, label."""
    cells = pd.read_parquet(f"data/raw/{dataset}_cells.parquet")
    if dataset == "bbbc041":
        cells["label"] = cells["category"].map(LABEL_MAP)
        cells = cells.dropna(subset=["label"])
    return cells[["cell_id", "source_image", "label"]]


def split(cells: pd.DataFrame):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(cells, groups=cells["source_image"]))
    return cells.iloc[train_idx], cells.iloc[test_idx]


def main():
    rows = []
    for dataset in ("nih", "bbbc041"):
        train, test = split(load_cells(dataset))
        assert set(train["source_image"]).isdisjoint(test["source_image"])
        train.to_csv(f"data/splits/{dataset}_train.csv", index=False)
        test.to_csv(f"data/splits/{dataset}_test.csv", index=False)
        for name, part in (("train", train), ("test", test)):
            for label, count in part["label"].value_counts().items():
                rows.append({"dataset": dataset, "split": name, "label": label, "count": count})
    pd.DataFrame(rows).to_csv("data/split_counts.csv", index=False)


if __name__ == "__main__":
    main()
