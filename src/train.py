"""Fine-tune every method at every label fraction to a fixed optimizer-step budget.

Training stops after exactly `total_steps` gradient updates, whatever the
subset size: small subsets are simply cycled more often. Only the train subset
is read; the test split is not loaded here at all.
"""
import itertools
import random

import numpy as np
import torch
import yaml

from models import build_model
from data import subset_loader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(method: str, dataset: str, fraction: float, seed: int, cfg: dict) -> None:
    seed_everything(seed)
    model = build_model(method, num_classes=2)
    loader = subset_loader(dataset, fraction, seed, batch_size=cfg["batch_size"], augment=cfg["augmentation"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["total_steps"])
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    batches = itertools.cycle(loader)
    for step in range(1, cfg["total_steps"] + 1):
        images, labels = next(batches)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), f"checkpoints/{method}_{dataset}_{fraction}_{seed}.pt")
    print(
        f"[train] method={method} dataset={dataset} fraction={fraction} seed={seed} "
        f"train_examples={len(loader.dataset)} steps={step}/{cfg['total_steps']} done"
    )


def main():
    cfg = yaml.safe_load(open("configs/train.yaml"))
    for method, dataset, fraction, seed in itertools.product(
        cfg["methods"], cfg["datasets"], cfg["fractions"], cfg["seeds"]
    ):
        train_one(method, dataset, fraction, seed, cfg)


if __name__ == "__main__":
    main()
