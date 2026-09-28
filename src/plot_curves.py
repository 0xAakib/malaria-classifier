"""Plot the label-efficiency curves: balanced accuracy against label fraction,
one line per method, one panel per dataset. Also write the headline numbers."""
import json

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/eval.csv")
curve = df.groupby(["dataset", "method", "fraction"], as_index=False)["balanced_acc"].mean()
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, dataset in zip(axes, ["nih", "bbbc041"]):
    for method, g in curve[curve.dataset == dataset].groupby("method"):
        g = g.sort_values("fraction")
        ax.plot(g.fraction, g.balanced_acc, marker="o", label=method)   # one line per method
    ax.set_xscale("log")
    ax.set_xlabel("label fraction")
    ax.set_title(dataset)
axes[0].set_ylabel("balanced accuracy")
axes[0].legend()
fig.savefig("figures/curves.png", dpi=150)

full = curve[curve.fraction == 1.0].set_index(["dataset", "method"]).balanced_acc
json.dump({d: {"dinov2": round(float(full[(d, "dinov2")]), 3)} for d in ["nih", "bbbc041"]}, open("results/metrics.json", "w"))
