# LLM Control
The repo for llm control

## Experiments TODOs

### Main Results
- [ ] Toxicity (may need to follow the original paper's setup, i.e., randomly sample 25 responses and calculate the Expected Toxicity)
  - [ ] Suffix Gradient (clipping)
  - [ ] Suffix Gradient (pgd)
  - [ ] Prefix Control
- [ ] Privacy
  - [ ] Suffix Gradient (clipping)
  - [ ] Suffix Gradient (pgd)
  - [ ] Prefix Control
- [ ] TruthfulQA
  - [ ] Suffix Gradient (clipping)
  - [ ] Suffix Gradient (pgd)
  - [ ] Prefix Control
- [ ] Happiness
  - [ ] Suffix Gradient (clipping)
  - [ ] Suffix Gradient (pgd)
  - [ ] Prefix Control
- [ ] Sadness
  - [ ] Suffix Gradient (clipping)
  - [ ] Suffix Gradient (pgd)
  - [ ] Prefix Control
- [ ] Angerness
  - [ ] Suffix Gradient (clipping)
  - [ ] Suffix Gradient (pgd)
  - [ ] Prefix Control
- [ ] Surprise
  - [ ] Suffix Gradient (clipping)
  - [ ] Suffix Gradient (pgd)
  - [ ] Prefix Control
- [ ] Disgust
  - [ ] Suffix Gradient (clipping)
  - [ ] Suffix Gradient (pgd)
  - [ ] Prefix Control
- [ ] Fearness
  - [ ] Suffix Gradient (clipping)
  - [ ] Suffix Gradient (pgd)
  - [ ] Prefix Control
- [ ] Denfense (Maybe)
  - [ ] Suffix Gradient (clipping)
  - [ ] Suffix Gradient (pgd)
  - [ ] Prefix Control

### Prefix Control
- [ ] Scale up data
- [ ] Control multiple attributes simultaneously
- [ ] Try prefix tuning

### Exploratory Experiments
- [ ] Draw trajectory using PCA (can use the `generate_ds.py` to collect gradients)
- [ ] Control a single sentence over different attributes and draw the directions (can also use `generate_ds.py`, since we need the amount of data to be large to ensure PCA is accurate)
- [ ] Visualize how suffix gradients affect attention heads (can use circuitsvis; may need a bit modification)
