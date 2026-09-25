# Label budget for found malaria backbones

How much labelled data do four backbones need on the NIH malaria set and on BBBC041?
Methods: DINOv2 (Sadou checkpoint), PlasmoVision ViT, a small CNN from scratch, ImageNet ResNet-18.
Label fractions 100% / 25% / 10% / 2%, two seeds, a fixed optimizer-step budget.

Run order: `src/make_splits.py`, `src/make_subsets.py`, `src/train.py`, `src/evaluate.py`, `src/analyse.py`.
