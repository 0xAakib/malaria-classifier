<!-- deliverable: m100-read-and-digest — answer under each heading; do not edit headings -->

## Q1. Why does the study match optimizer steps instead of epochs across label fractions? What would a 2% run look like if epochs were matched?

An epoch over the 2% subset is about fifty times shorter than an epoch over the full train split, so the same number of epochs would give the 2% run roughly fifty times fewer gradient updates. If we matched epochs, a drop at 2% could just mean the model trained less, and we could not tell that apart from the effect of having fewer labels. Matching steps means every run gets the same number of updates and only the number of distinct examples changes.

## Q2. BBBC041 is over 95% uninfected. Why is plain accuracy the wrong metric, and what is used instead?

A model that answers "uninfected" for every cell would already score about 95% accuracy on BBBC041 while never finding a single infected cell. We use balanced accuracy, the average of the recall on each class, and MCC, which both drop to chance for a model that ignores the minority class.

## Q3. What would go wrong if the train/test split were made per cell instead of per source image?

Cells cut from the same slide would end up on both sides of the split. The model would then be tested on slides it has already seen during training, with the same staining and lighting, so the test scores would be inflated compared to a genuinely new slide.

## Q4 (choose one). The four methods differ in architecture and pretraining at once. What can the results NOT claim?
- [ ] A. Which of these four specific models is most label-efficient
- [x] B. Whether ViTs are more label-efficient than CNNs in general
- [ ] C. Nothing; every claim is valid

## Q5. Give one result that would count as a publishable null result for this study.

If all four label-efficiency curves were statistically indistinguishable once the step budget is matched, that would show a malaria-adjacent checkpoint does not need fewer labels than training from scratch, which is still a useful finding for a team choosing a backbone.
