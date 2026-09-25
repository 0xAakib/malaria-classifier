<!-- deliverable: m700-crossovers — answer under each heading; do not edit headings -->

## Q1. For each dataset, list the method ranking at 100% and at 2% labels.

From `results/eval.csv`, averaging the two seeds:

- NIH, 100%: dinov2 (0.971) > plasmovision (0.962) > resnet18 (0.955) > cnn (0.948)
- NIH, 2%: dinov2 (0.931) > plasmovision (0.902) > resnet18 (0.889) > cnn (0.842)
- BBBC041, 100%: dinov2 (0.912) > resnet18 (0.901) > plasmovision (0.889) > cnn (0.861)
- BBBC041, 2%: resnet18 (0.792) > dinov2 (0.771) > plasmovision (0.731) > cnn (0.662)

## Q2. Do any two methods' curves cross? Name them, the fraction and the dataset, or say none do.

On BBBC041, ResNet-18 and DINOv2 cross between 10% and 2% labels. DINOv2 is ahead at 100%, 25% and 10%, and ResNet-18 is ahead at 2%. This holds in both seeds: at 2%, seed 0 is 0.789 vs 0.768 and seed 1 is 0.795 vs 0.774 (rows `resnet18,bbbc041,0.02,*` and `dinov2,bbbc041,0.02,*` in `results/eval.csv`). No curves cross on NIH.

## Q3 (choose one). Which claim does this result support?
- [x] A. Which of these four artifacts is most label-efficient on these two datasets
- [ ] B. That ViT architectures need fewer labels than CNNs
