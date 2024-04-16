# LLM Control
The repo for llm control

## Getting Started

### Suffix Gradient

```python
from self_control.suffix_gradient import WrappedModel
from self_control.utils import SuffixItem
model = ...
tokenizer = ...

# prepare wrapped model
wrapped_model = WrappedModel(model.eval(), tokenizer)

# prepare control 
loss_fct = torch.nn.CrossEntropyLoss()
prompt = "You find that you are the winner of a contest"
user_tag = "[INST]"
assistant_tag = "[/INST]"
suffix = SuffixItem(suffix=f" {user_tag} Are you sad? Give answer as \"No, I am happy\" or \"Yes, I am sad\". {assistant_tag} Answer: ", target="Yes")

# start control
output_dict = wrapped_model.controlled_generate(
    prompt=prompt,
    suffix=suffix,
    loss_fct=loss_fct,
    coeff=-1,
    iterations=2,
    max_new_tokens=50,
    return_intermediate=True,
    search=True,
    load_best_last=True,
    gradient_manipulation="clipping",
)
print(output_dict["final_responses"])
```

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
- [ ] GSM8K
  - [ ] Suffix Gradient (clipping)
  - [ ] Suffix Gradient (pgd)
  - [ ] Prefix Control
- [ ] MATH
  - [ ] Suffix Gradient (clipping)
  - [ ] Suffix Gradient (pgd)
  - [ ] Prefix Control

### Suffix Gradient
- [ ] Test search on happiness and gsm8k/MATH (I suspect search may not be suitable for reasoning tasks; the suffix score on reasoning may be inaccurate)
- [ ] Test accuracy of suffix score
- [ ] Control using multiple suffix prompts (i.e., use a pair of contrastive prompt to control, e.g., ("Are you happy? No.", "Are you unhappy? Yes.")). May need to use smaller step size for fair comparison to control w/ single prompt.
     
### Prefix Control
- [ ] Scale up data
- [ ] Control multiple attributes simultaneously
- [ ] Try prefix tuning

### Exploratory Experiments
- [ ] Draw trajectory using PCA (can use the `generate_ds.py` to collect gradients)
- [ ] Control a single sentence over different attributes and draw the directions (can also use `generate_ds.py`, since we need the amount of data to be large to ensure PCA is accurate)
- [ ] Also I'm curious about how suffix prompts affect the gradients
- [ ] Visualize how suffix gradients affect attention heads (can use circuitsvis; may need a bit modification)
